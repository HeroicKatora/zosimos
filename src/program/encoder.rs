//! Produce a stream of `Low` instructions that can be executed on a particular device.
use core::{num::NonZeroU64, ops::Range};
use std::borrow::Cow;
use std::collections::HashMap;

use crate::buffer::{ByteLayout, CanvasLayout, Descriptor};
use crate::command::Register;
use crate::pool::Pool;
use crate::program::{
    BindGroupDescriptor, BindGroupLayoutDescriptor, BindingResource, Buffer, BufferDescriptor,
    BufferDescriptorInit, BufferInitContent, BufferLayout, BufferUsage, ByteBufferAssignment,
    Capabilities, ColorAttachmentDescriptor, DeviceBuffer, DeviceTexture, Event, FragmentState,
    ImageBufferAssignment, ImageBufferPlan, ImageDescriptor, ImagePoolPlan, Initializer,
    Instruction, Knob, KnobDescriptor, LaunchError, Low, PipelineLayoutDescriptor,
    PipelineLayoutKey, PrimitiveState, RegisterAssignment, RenderPassDescriptor,
    RenderPipelineDescriptor, RenderPipelineKey, SamplerDescriptor, ShaderDescriptor,
    ShaderDescriptorKey, Texture, TextureDescriptor, TextureViewDescriptor, VertexState,
};
use crate::util::ExtendOne;
use crate::{run, shaders};

use image_canvas::layout::RowLayoutDescription;
use wgpu::StoreOp;

use super::KnobUser;

/// The encoder tracks the supposed state of `run::Descriptors` without actually executing them.
#[derive(Default)]
pub(crate) struct Encoder<Instructions: ExtendOne<Low> = Vec<Low>> {
    pub(crate) instructions: Instructions,
    /// The allocate binary data for runtime execution.
    pub(crate) binary_data: Vec<u8>,

    // Fields from `run::Descriptors` that simulate the length.
    instruction_pointer: usize,
    bind_groups: usize,
    bind_group_layouts: usize,
    buffers: usize,
    command_buffers: usize,
    pipeline_layouts: usize,
    render_pipelines: usize,
    sampler: usize,
    shaders: usize,
    textures: usize,
    texture_views: usize,
    events: usize,

    // Additional validation properties.
    is_in_command_encoder: bool,
    is_in_render_pass: bool,

    // Additional fields to map our runtime state.
    /// How we map registers to device buffers.
    buffer_plan: ImageBufferPlan,
    /// If we should build a trace of input pool images to textures.
    /// This isn't necessary for a running program but it can be relevant when one wants to restore
    /// the pool to the state exactly before, even when the program is stopped. In those cases we
    /// might still hold images in textures. This instructs us to trace where.
    trace_pool_plan: bool,
    /// How we mapped registers to images in the pool.
    pool_plan: ImagePoolPlan,
    /// Declare where we put our input registers.
    input_map: HashMap<Register, Texture>,
    /// Declare where we intend to write our outputs.
    output_map: HashMap<Register, Texture>,
    /// Declare where we intend to render our outputs.
    render_map: HashMap<Register, Texture>,
    /// The Bind Group layer Descriptor used in fragment shader, set=1.
    /// This is keyed by the number of descriptors for that layout.
    paint_group_layout: HashMap<usize, usize>,
    /// The Bind Group Descriptor used in vertex buffer, set=0.
    quad_group_layout: Option<usize>,
    /// The Bind Group Descriptor for set=2, used for parameters of fragment shader.
    fragment_data_group_layout: Option<usize>,
    /// The Pipeline Descriptor used in generic paint shaders.
    /// Will use the quad group and fragment data group layouts.
    paint_pipeline_layout: Option<usize>,
    /// The Bind Group Descriptor layout for the staging fragment shader.
    /// Alternative for the fragment data group.
    /// Since there is a limit on the number of active storage textures, we have one layout for
    /// each potential usage.
    stage_group_layout: HashMap<u32, usize>,
    known_samplers: HashMap<SamplerDescriptor, usize>,
    fragment_shaders: HashMap<shaders::FragmentShaderKey, usize>,
    vertex_shaders: HashMap<shaders::VertexShader, usize>,
    simple_quad_buffer: Option<DeviceBuffer>,
    /// The render pipeline state for staging a texture.
    staged_to_pipelines: HashMap<Texture, SimpleRenderPipeline>,
    /// The render pipeline state for undoing staging a texture.
    staged_from_pipelines: HashMap<Texture, SimpleRenderPipeline>,
    /// The texture operands collected for the next render preparation.
    operands: Vec<Texture>,
    /// Command slots that we deferred submission.
    delayed_commands: Vec<Instruction>,

    // Fields regarding the status of registers.
    /// Describes how registers where mapped to buffers.
    register_map: HashMap<Register, RegisterMap>,
    /// Describes how textures have been mapped to the GPU.
    texture_map: HashMap<Texture, TextureMap>,
    /// Describes how buffers have been mapped to the GPU.
    buffer_map: HashMap<Buffer, BufferMap>,
    /// Describes which intermediate textures have been mapped to the GPU.
    staging_map: HashMap<Texture, StagingTexture>,
    // Arena style allocators.
    // Output State with additional info relating to the input program.
    pub(crate) info: ExecutableInfo,
}

/// FIXME: remember masks for ops that can be skipped/replaced if texture is cached? Most ops
/// build objects (descriptors, etc.) that are used in construction or initialization of a
/// buffer. If we don't need to build the buffer, because we have its state in a previous cache,
/// then we don't need the inputs either.
/// We must always emulate the 'effect' (in terms of object indices) at runtime. Maybe we can
/// just use a skip-list instead of Vec for those objects though. Or switch run to a pull-based
/// model? But that would make it hard to provide bounds for `step`. Ahhh! Choices!
#[derive(Default)]
pub(crate) struct ExecutableInfo {
    /// Annotates which low op allocates a cacheable texture.
    pub(crate) texture_by_op: HashMap<usize, TextureDescriptor>,
    /// Annotates which low op allocates a cacheable buffer.
    pub(crate) buffer_by_op: HashMap<usize, BufferDescriptor>,
    /// Annotates which low op constructs a cacheable shader module.
    pub(crate) shader_by_op: HashMap<usize, ShaderDescriptorKey>,
    /// Annotates which low op allocates a pipeline that may be reused.
    pub(crate) pipeline_by_op: HashMap<usize, RenderPipelineKey>,
    /// Annotates which shader is built by what key.
    pub(crate) shader_by_idx: HashMap<usize, ShaderDescriptorKey>,
    /// Annotate which op to skip by which event.
    pub(crate) skip_by_op: HashMap<Instruction, Event>,
    /// Annotate regions of memory which can be modified.
    pub(crate) knobs: HashMap<Knob, KnobDescriptor>,
}

#[must_use = "Knob must be handled to apply its effects."]
pub enum KnobUsage {
    /// The knob does not need to be used further.
    Noop,
    CopyFrom {
        buffer: DeviceBuffer,
        range: Range<u64>,
    },
}

/// The GPU buffers associated with a register.
/// Supplements the buffer_plan by giving direct mappings to each device resource index in an
/// encoder process.
#[derive(Clone, Debug)]
pub(crate) enum RegisterMap {
    Buffer {
        reg_buffer: Buffer,
        buffer: DeviceBuffer,
        buffer_layout: BufferLayout,
    },
    Image {
        reg_texture: Texture,
        texture: DeviceTexture,
        reg_buffer: Buffer,
        buffer: DeviceBuffer,
        /// A device buffer with (COPY_DST | MAP_READ) for reading back the texture.
        map_read: Option<DeviceBuffer>,
        /// A device buffer with (COPY_SRC | MAP_WRITE) for initialization the texture.
        map_write: Option<DeviceBuffer>,
        /// A device texture for (de-)normalizing the texture contents.
        staging: Option<DeviceTexture>,
        /// The layout of the buffer.
        /// This might differ from the layout of the corresponding pool image because it must adhere to
        /// the layout requirements of the device. For example, the alignment of each row must be
        /// divisible by 256 etc.
        buffer_layout: CanvasLayout,
        /// The byte (row-wise) descriptor for the buffer layout.
        /// Same caveat applies, this must be padded to alignments. Note that all currently supported
        /// wgpu formats have a row-wise layout. When and if this changes, turn this field into an enum
        /// instead?
        byte_layout: ByteLayout,
        /// The format of the non-staging texture.
        texture_format: TextureDescriptor,
        /// The format of the staging texture.
        staging_format: Option<TextureDescriptor>,
    },
}

/// The gpu texture associated with the image.
#[derive(Clone, Debug)]
struct TextureMap {
    device: DeviceTexture,
    format: TextureDescriptor,
}

/// A 'staging' textures for rendering the internal texture to the externally chosen texel
/// format including, for example, quantizing and clamping to a different numeric format.
/// Note that the device texture needs to be a format that the device can use for color
/// operations (read and store) but it might not support the format natively. In such cases we
/// need to transform between the intended format and the native format. We could do this with
/// a blit operation while copying from a buffer but this also depends on support from the
/// device and wgpu does not allow arbitrary conversion (in 0.8 none are allowed).
/// Hence we must perform such conversion ourselves with a specialized shader. This could also
/// be a compute shader but then we must perform a buffer copy of everything so this can not be
/// part of a graphic pipeline. If a staging texture exists then copies from the buffer and to
/// the buffer always pass through it and we perform a sync from/to the staging texture before
/// and after all paint operations involving that buffer.
#[derive(Clone, Debug)]
struct StagingTexture {
    device: DeviceTexture,
    format: TextureDescriptor,
    stage_kind: shaders::stage::StageKind,
    parameter: shaders::stage::XyzParameter,
}

/// The gpu buffer associated with an image buffer.
#[derive(Clone, Debug)]
struct BufferMap {
    device: DeviceBuffer,
    layout: BufferLayout,
}

#[derive(Clone, Copy)]
pub(crate) struct SimpleRenderPipeline {
    pipeline: usize,
    buffer: DeviceBuffer,
    group: Option<usize>,
    vertex_bind: Option<usize>,
    vertices: u32,
    fragment_bind: Option<usize>,
}

struct SimpleRenderPipelineDescriptor<'data> {
    pipeline_target: PipelineTarget,
    /// Bind data for (set 0, binding 0).
    vertex_bind_data: BufferBind<'data>,
    /// Texture for (set 1, binding 0)
    fragment_texture: TextureBind,
    /// Texture for (set 2, binding 0)
    fragment_bind_data: BufferBind<'data>,
    /// How the fragment bind buffer is being filled.
    fragment_knob: KnobUsage,
    /// The vertex shader to use.
    vertex: ShaderBind,
    /// The fragment shader to use.
    fragment: ShaderBind,
}

enum PipelineTarget {
    Texture(Texture),
    PreComputedGroup { target_format: wgpu::TextureFormat },
}

pub(crate) enum TextureBind {
    /// Use the currently pushed texture operands.
    /// The arguments are taken from the back of the operand vector.
    Textures(usize),
    PreComputedGroup {
        /// The index of the bind group we're binding to set `1`, the fragment set.
        group: usize,
        /// The layout corresponding to the bind group.
        layout: usize,
    },
}

#[derive(Debug)]
enum BufferBind<'data> {
    /// Upload the data, then bind the buffer.
    Set {
        data: &'data [u8],
    },
    /// The data has already been planned beforehand.
    Planned {
        data: core::ops::Range<usize>,
    },
    None,
    // /// The data is already there, simply bind the buffer.
    // Load(DeviceBuffer),
}

enum ShaderBind {
    ShaderMain(usize),
    Shader {
        id: usize,
        entry_point: &'static str,
    },
}

impl<I: ExtendOne<Low>> Encoder<I> {
    /// Tell the encoder which commands are natively supported.
    /// Some features require GPU support. At this point we decide if our request has succeeded and
    /// we might poly-fill it with a compute shader or something similar.
    pub(crate) fn enable_capabilities(&mut self, caps: &Capabilities) {
        // FIXME: currently nothing uses features or limits.
        // Which is wrong, we can use features to skip some staging. We might also have some
        // slightly different shader features such as using push constants in some cases?
        let _ = caps;
    }

    pub(crate) fn set_buffer_plan(&mut self, plan: &ImageBufferPlan) {
        self.buffer_plan = plan.clone();
    }

    pub(crate) fn set_pool_plan(&mut self, plan: &ImagePoolPlan) {
        self.trace_pool_plan = true;
        self.pool_plan = plan.clone();
    }

    /// Validate and then add the command to the encoder.
    ///
    /// This ensures we can keep track of the expected state change, and validate the correct order
    /// of commands. More specific sequencing commands will expect correct order or assume it
    /// internally.
    pub(crate) fn push(&mut self, mut low: Low) -> Result<Instruction, LaunchError> {
        match &mut low {
            // Not yet handled.
            Low::AssertBuffer { .. } | Low::AssertTexture { .. } => {}
            Low::BindGroupLayout(_) => self.bind_group_layouts += 1,
            Low::BindGroup(_) => self.bind_groups += 1,
            Low::Buffer(ref desc) => {
                self.info
                    .buffer_by_op
                    .insert(self.instruction_pointer, desc.clone());
                self.buffers += 1;
            }
            Low::BufferInit(_) => self.buffers += 1,
            Low::PipelineLayout(_) => self.pipeline_layouts += 1,
            Low::Sampler(_) => self.sampler += 1,
            Low::Shader(ref desc) => {
                if let Some(key) = &desc.key {
                    self.info
                        .shader_by_op
                        .insert(self.instruction_pointer, key.clone());
                    self.info.shader_by_idx.insert(self.shaders, key.clone());
                }

                self.shaders += 1
            }
            Low::Texture(ref desc) => {
                self.info
                    .texture_by_op
                    .insert(self.instruction_pointer, desc.clone());
                self.textures += 1
            }
            Low::TextureView(_) | Low::RenderView(_) => self.texture_views += 1,
            Low::RenderPipeline(_) => self.render_pipelines += 1,
            Low::BeginCommands => {
                if self.is_in_command_encoder {
                    return Err(LaunchError::InternalCommandError(line!()));
                }

                self.is_in_command_encoder = true;
            }
            Low::BeginRenderPass(_) => {
                if self.is_in_render_pass {
                    return Err(LaunchError::InternalCommandError(line!()));
                }

                if !self.is_in_command_encoder {
                    return Err(LaunchError::InternalCommandError(line!()));
                }

                self.is_in_render_pass = true;
            }
            Low::EndCommands => {
                if !self.is_in_command_encoder {
                    return Err(LaunchError::InternalCommandError(line!()));
                }

                self.is_in_command_encoder = false;
                self.command_buffers += 1;
            }
            Low::EndRenderPass => {
                if !self.is_in_render_pass {
                    return Err(LaunchError::InternalCommandError(line!()));
                }

                self.is_in_render_pass = false;
            }
            Low::SetPipeline(_) => {}
            Low::SetBindGroup { group, .. } => {
                if *group >= self.bind_groups {
                    return Err(LaunchError::InternalCommandError(line!()));
                }
            }
            Low::SetVertexBuffer { buffer, .. } => {
                if *buffer >= self.buffers {
                    return Err(LaunchError::InternalCommandError(line!()));
                }
            }
            // TODO: could validate indices.
            Low::DrawOnce { .. } | Low::DrawIndexedZero { .. } | Low::SetPushConstants { .. } => {}
            Low::RunTopCommand => {
                if !self.delayed_commands.is_empty() {
                    low = Low::RunBotToTop(self.delayed_commands.len() + 1);
                    self.delayed_commands.clear();
                } else {
                    if self.command_buffers == 0 {
                        return Err(LaunchError::InternalCommandError(line!()));
                    }

                    self.command_buffers -= 1;
                }
            }
            Low::RunBotToTop(num) => {
                // If we delayed anything, run it now to preserve order of commands.
                if !self.delayed_commands.is_empty() {
                    *num += self.delayed_commands.len();
                    self.delayed_commands.clear();
                }

                if let Some(num) = self.command_buffers.checked_sub(*num) {
                    self.command_buffers = num;
                } else {
                    return Err(LaunchError::InternalCommandError(line!()));
                }
            }
            // TODO: could validate indices.
            Low::WriteImageToTexture { .. }
            | Low::CopyBufferToTexture { .. }
            | Low::CopyTextureToBuffer { .. }
            | Low::CopyBufferToBuffer { .. }
            | Low::ZeroBuffer { .. } => {
                if !self.is_in_command_encoder {
                    return Err(LaunchError::InternalCommandError(line!()));
                }
            }
            Low::WriteImageToBuffer { .. } | Low::ReadBuffer { .. } => {}
            // No validation for stack frame shuffling.
            // TODO: should we simulate stack height?
            Low::StackFrame(_) | Low::StackPop => {}
            Low::Call { .. } => {
                if self.is_in_command_encoder {
                    return Err(LaunchError::InternalCommandError(line!()));
                }

                self.plan_gpu_effects_visible()?;
            }
            Low::Return => {
                if self.is_in_command_encoder {
                    return Err(LaunchError::InternalCommandError(line!()));
                }
            }
        }

        let instruction = Instruction(self.instruction_pointer);
        self.instruction_pointer += 1;
        self.instructions.extend_one(low);
        Ok(instruction)
    }

    fn plan_run_top_command(&mut self) {
        self.delayed_commands
            .push(Instruction(self.instruction_pointer));
    }

    fn plan_gpu_effects_visible(&mut self) -> Result<(), LaunchError> {
        if !self.delayed_commands.is_empty() {
            // This synchronizes itself.
            self.push(Low::RunBotToTop(0))?;
        }

        Ok(())
    }

    fn make_texture_descriptor(
        &mut self,
        texture: Texture,
    ) -> Result<ImageDescriptor, LaunchError> {
        let descriptor = &self.buffer_plan.texture[texture.0];
        ImageDescriptor::new(descriptor)
    }

    pub(crate) fn push_operand(&mut self, texture: Texture) -> Result<(), LaunchError> {
        self.operands.push(texture);
        Ok(())
    }

    // We must trick the borrow checker here..
    pub(crate) fn allocate_register(&mut self, idx: Register) -> Result<&RegisterMap, LaunchError> {
        self.ensure_allocate_register(idx)?;
        // Trick, reborrow the thing..
        Ok(&self.register_map[&idx])
    }

    /// Construct and allocate mappings for the register.
    /// We can't return the finished mapping due to borrow checker issues (would need to borrow
    /// from self but access it 'later' in the function; will be fixed with Polonius). Thus one
    /// should prefer to call `allocate_register` instead.
    fn ensure_allocate_register(&mut self, idx: Register) -> Result<(), LaunchError> {
        let ty = self.buffer_plan.get_register_resources(idx)?;

        match ty {
            RegisterAssignment::Image(assign) => self.ensure_allocate_image(idx, assign),
            RegisterAssignment::Buffer(assign) => self.ensure_allocate_buffer(idx, assign),
        }
    }

    fn ensure_allocate_image(
        &mut self,
        idx: Register,
        assign: ImageBufferAssignment,
    ) -> Result<(), LaunchError> {
        let ImageBufferAssignment {
            buffer: reg_buffer,
            texture: reg_texture,
        } = assign;

        if self.register_map.get(&idx).is_some() {
            return Ok(());
        }

        let staged = self.make_texture_descriptor(reg_texture)?;
        let texture_format = staged.to_texture();
        let staging_format = staged.to_staging_texture();
        let descriptor = &self.buffer_plan.texture[reg_texture.0];

        let byte_layout = descriptor
            .to_aligned()
            .ok_or_else(|| LaunchError::InternalCommandError(line!()))?;

        let buffer_layout = CanvasLayout::with_row_layout(&RowLayoutDescription {
            texel: descriptor.texel.clone(),
            width: texture_format.size.0.get(),
            height: texture_format.size.1.get(),
            row_stride: byte_layout.row_stride,
        })
        .expect("valid layout");

        let (buffer, map_write, map_read) = {
            let buffer = self.buffers;
            self.push(Low::Buffer(BufferDescriptor {
                size: buffer_layout.u64_len(),
                usage: BufferUsage::DataBuffer,
            }))?;
            self.push(Low::Buffer(BufferDescriptor {
                size: buffer_layout.u64_len(),
                usage: BufferUsage::DataIn,
            }))?;
            self.push(Low::Buffer(BufferDescriptor {
                size: buffer_layout.u64_len(),
                usage: BufferUsage::DataOut,
            }))?;

            (
                DeviceBuffer(buffer),
                DeviceBuffer(buffer + 1),
                DeviceBuffer(buffer + 2),
            )
        };

        let texture = self.ensure_device_texture(reg_texture)?;
        let staging = self
            .staging_map
            .get(&reg_texture)
            .map(|staging| staging.device);

        let desc = Descriptor::from(&buffer_layout);

        let map_entry = RegisterMap::Image {
            reg_buffer,
            buffer,
            reg_texture,
            texture,
            map_read: Some(map_read),
            map_write: Some(map_write),
            staging,
            buffer_layout,
            byte_layout,
            texture_format,
            staging_format,
        };

        self.register_map
            .entry(idx)
            .and_modify(|in_map| in_map.clone_from(&map_entry))
            .or_insert_with(|| map_entry.clone());

        self.buffer_map.insert(
            reg_buffer,
            BufferMap {
                device: buffer,
                layout: BufferLayout::Texture(desc.layout),
            },
        );

        Ok(())
    }

    pub(crate) fn ensure_device_texture(
        &mut self,
        reg_texture: Texture,
    ) -> Result<DeviceTexture, LaunchError> {
        let staged = self.make_texture_descriptor(reg_texture)?;
        let texture_format = staged.to_texture();
        let staging_format = staged.to_staging_texture();

        if let Some(texture_map) = self.texture_map.get(&reg_texture) {
            return Ok(texture_map.device);
        }

        let texture = {
            let texture = self.textures;
            self.push(Low::Texture(texture_format.clone()))?;
            DeviceTexture(texture)
        };

        self.texture_map.insert(
            reg_texture,
            TextureMap {
                device: texture,
                format: texture_format.clone(),
            },
        );

        if let Some(staging) = staging_format {
            let st_parameter = staged
                .staging
                .as_ref()
                .expect("Have a format for staging texture when we have staging texture");

            let device = {
                let texture = self.textures;
                self.push(Low::Texture(staging.clone()))?;
                DeviceTexture(texture)
            };

            self.staging_map.insert(
                reg_texture,
                StagingTexture {
                    device,
                    format: staging,
                    stage_kind: st_parameter.stage_kind,
                    parameter: st_parameter.parameter,
                },
            );
        }

        Ok(texture)
    }

    fn ensure_allocate_buffer(
        &mut self,
        idx: Register,
        assign: ByteBufferAssignment,
    ) -> Result<(), LaunchError> {
        if self.register_map.contains_key(&idx) {
            return Ok(());
        }

        let ByteBufferAssignment { buffer } = assign;
        let device = self.ensure_device_buffer(buffer)?;
        let layout = &self.buffer_plan.buffer[buffer.0];

        self.register_map.insert(
            idx,
            RegisterMap::Buffer {
                buffer: device,
                reg_buffer: buffer,
                buffer_layout: layout.clone(),
            },
        );

        Ok(())
    }

    fn ensure_device_buffer(&mut self, buffer: Buffer) -> Result<DeviceBuffer, LaunchError> {
        if let Some(map) = self.buffer_map.get(&buffer) {
            return Ok(map.device);
        }

        let plan = self
            .buffer_plan
            .buffer
            .get(buffer.0)
            .map(Clone::clone)
            .ok_or_else(|| LaunchError::InternalCommandError(line!()))?;

        let device = DeviceBuffer(self.buffers);
        self.push(Low::Buffer(BufferDescriptor {
            size: plan.u64_len(),
            usage: BufferUsage::DataBuffer,
        }))?;

        self.buffer_map.insert(
            buffer,
            BufferMap {
                device,
                layout: plan,
            },
        );

        Ok(device)
    }

    fn ingest_image_data(&mut self, idx: Register) -> Result<Texture, LaunchError> {
        let texture = match self.buffer_plan.get_register_resources(idx)? {
            RegisterAssignment::Image(image) => image.texture,
            _ => return Err(LaunchError::InternalCommandError(line!())),
        };

        // FIXME: We are conflating `texture` and the index in the execution's vector of IO
        // buffers. That is, we have an entry there even when the register/texture in question has
        // no input or output behavior.
        //
        // That's a shame because, for example, we could leave images in the pool when they do not
        // get used in the pipeline.
        if self.trace_pool_plan {
            let source_key = self.pool_plan.get(idx)?;
            self.pool_plan.buffer.entry(source_key).or_insert(texture);
        }

        Ok(texture)
    }

    /// Copy from the input to the internal memory visible buffer.
    pub(crate) fn copy_input_to_buffer(&mut self, idx: Register) -> Result<(), LaunchError> {
        let regmap = self.allocate_register(idx)?.clone();
        let source_image = self.ingest_image_data(idx)?;
        let write_event = self.make_event();
        self.input_map.insert(idx, source_image);

        let RegisterMap::Image {
            reg_texture,
            buffer,
            buffer_layout,
            byte_layout,
            map_write,
            ..
        } = regmap
        else {
            unreachable!("non-image inputs not yet implemented");
        };

        let descriptor = &self.buffer_plan.texture[reg_texture.0];
        let size = descriptor.size();

        // See below, required for direct buffer-to-buffer copy.
        let sizeu64 = buffer_layout.u64_len();

        // FIXME: if it is a simple copy we can use regmap.buffer directly.
        let target_buffer = map_write.unwrap_or(buffer);

        // FIXME: should we validate size here as well for better errors?
        // Potentially all errors are internal so it might not matter.
        let inst = self.push(Low::WriteImageToBuffer {
            copy_dst_buffer: buffer,
            source_image,
            size,
            offset: (0, 0),
            target_buffer,
            target_layout: byte_layout,
            write_event,
        })?;

        self.info.skip_by_op.insert(inst, write_event);

        // FIXME: we're using wgpu internal's scheduling for writing the data to the gpu buffer but
        // this is a separate allocation. We'd instead like to use `regmap.map_write` and do our
        // own buffer mapping but this requires async scheduling. Soo.. do that later.
        // FIXME: might happen at next call within another command encoder..
        // FIXME: if we're reading from a texture, then we do not need this copy at all.
        // However, this fact is only known at runtime. So, should we defer running the
        // CopyBufferToBuffer instead to the runtime?
        if let Some(map_write) = map_write {
            self.push(Low::BeginCommands)?;
            let inst = self.push(Low::CopyBufferToBuffer {
                source_buffer: map_write,
                source: 0,
                target_buffer: buffer,
                target: 0,
                size: sizeu64,
            })?;
            self.info.skip_by_op.insert(inst, write_event);
            self.push(Low::EndCommands)?;
            // TODO: maybe also don't run it immediately?
            self.plan_run_top_command();
        }

        Ok(())
    }

    /// Copy from memory visible buffer to the texture.
    pub(crate) fn copy_buffer_to_staging(&mut self, idx: Register) -> Result<(), LaunchError> {
        let RegisterMap::Image {
            staging,
            staging_format,
            texture,
            buffer,
            byte_layout,
            ..
        } = self.allocate_register(idx)?.clone()
        else {
            unreachable!("Buffers are not copied to staging");
        };

        let (size, target_texture);
        if let Some(staging) = staging {
            target_texture = staging;
            let (width, height) = staging_format.as_ref().unwrap().size;
            size = (width.get(), height.get());
        } else {
            // .… or directly to the target buffer if we have no staging.
            target_texture = texture;
            size = self.buffer_plan.get_info(idx)?.descriptor.size();
        };

        self.push(Low::BeginCommands)?;
        self.push(Low::CopyBufferToTexture {
            source_buffer: buffer,
            source_layout: byte_layout,
            offset: (0, 0),
            size,
            target_texture,
        })?;

        self.push(Low::EndCommands)?;
        // TODO: maybe also don't run it immediately?
        self.plan_run_top_command();

        Ok(())
    }

    /// Copy quantized data to the internal buffer.
    /// Note that this may be a no-op for buffers that need no staging buffer, i.e. where
    /// quantization happens as part of the pipeline.
    pub(crate) fn copy_staging_to_texture(&mut self, idx: Texture) -> Result<(), LaunchError> {
        if let Some(staging) = self.staging_map.get(&idx) {
            // Try to use the cached version of this pipeline.
            let pipeline = if let Some(&pipeline) = self.staged_to_pipelines.get(&idx) {
                pipeline
            } else {
                let fn_ = Initializer::ToLinearOpto {
                    parameter: staging.parameter,
                    stage_kind: staging.stage_kind,
                };

                self.prepare_render(&fn_, idx)?
            };

            let dst_view = self.texture_view(idx)?;
            let attachment = ColorAttachmentDescriptor {
                texture_view: dst_view,
                ops: wgpu::Operations {
                    // TODO: we could let choose a replacement color..
                    load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                    store: StoreOp::Store,
                },
            };

            self.push(Low::BeginCommands)?;
            self.push(Low::BeginRenderPass(RenderPassDescriptor {
                color_attachments: vec![attachment],
                depth_stencil: None,
            }))?;
            self.render(pipeline)?;
            self.push(Low::EndRenderPass)?;
            self.push(Low::EndCommands)?;
            self.plan_run_top_command();
        }

        Ok(())
    }

    /// Quantize the texture to the staging buffer.
    /// May be a no-op, see reverse operation.
    pub(crate) fn copy_texture_to_staging(&mut self, idx: Texture) -> Result<(), LaunchError> {
        if let Some(staging) = self.staging_map.get(&idx) {
            let texture = staging.device;

            // Try to use the cached version of this pipeline.
            let pipeline = if let Some(&pipeline) = self.staged_from_pipelines.get(&idx) {
                pipeline
            } else {
                let fn_ = Initializer::FromLinearOpto {
                    parameter: staging.parameter,
                    stage_kind: staging.stage_kind,
                };

                self.prepare_render(&fn_, idx)?
            };

            let dst_view = {
                let descriptor = TextureViewDescriptor { texture };

                let id = self.texture_views;
                self.push(Low::TextureView(descriptor))?;
                id
            };

            let attachment = ColorAttachmentDescriptor {
                texture_view: dst_view,
                ops: wgpu::Operations {
                    // TODO: we could let choose a replacement color..
                    load: wgpu::LoadOp::Clear(wgpu::Color::RED),
                    store: StoreOp::Store,
                },
            };

            self.push(Low::BeginCommands)?;
            self.push(Low::BeginRenderPass(RenderPassDescriptor {
                color_attachments: vec![attachment],
                depth_stencil: None,
            }))?;
            self.render(pipeline)?;
            self.push(Low::EndRenderPass)?;
            self.push(Low::EndCommands)?;
            self.plan_run_top_command();
        }

        Ok(())
    }

    /// Copy from texture to the memory buffer.
    pub(crate) fn copy_staging_to_buffer(&mut self, idx: Register) -> Result<(), LaunchError> {
        let RegisterMap::Image {
            staging,
            staging_format,
            texture,
            buffer,
            byte_layout,
            ..
        } = self.allocate_register(idx)?.clone()
        else {
            unreachable!("Buffers are not copied to staging, requested {:?}", idx);
        };

        let (size, source_texture);
        if let Some(staging) = staging {
            source_texture = staging;
            let (width, height) = staging_format.as_ref().unwrap().size;
            size = (width.get(), height.get());
        } else {
            source_texture = texture;
            size = self.buffer_plan.get_info(idx)?.descriptor.size();
        };

        self.push(Low::BeginCommands)?;
        self.push(Low::CopyTextureToBuffer {
            source_texture,
            offset: (0, 0),
            size,
            target_buffer: buffer,
            target_layout: byte_layout,
        })?;
        self.push(Low::EndCommands)?;
        self.plan_run_top_command();

        Ok(())
    }

    /// Copy the memory buffer to the output.
    pub(crate) fn copy_buffer_to_output(
        &mut self,
        idx: Register,
        dst: Register,
    ) -> Result<(), LaunchError> {
        let regmap = self.allocate_register(idx)?.clone();
        let target_image = self.ingest_image_data(dst)?;
        self.output_map.insert(dst, target_image);

        let RegisterMap::Image {
            map_read,
            buffer,
            buffer_layout,
            byte_layout,
            ..
        } = regmap
        else {
            unreachable!("non-image outputs not yet implemented");
        };

        let size = self.buffer_plan.get_info(idx)?.descriptor.size();
        let sizeu64 = buffer_layout.u64_len();

        let source_buffer = map_read.unwrap_or(buffer);

        // FIXME: might happen at next call within another command encoder..
        if let Some(map_read) = map_read {
            self.push(Low::BeginCommands)?;
            self.push(Low::CopyBufferToBuffer {
                source_buffer: buffer,
                source: 0,
                target_buffer: map_read,
                target: 0,
                size: sizeu64,
            })?;
            self.push(Low::EndCommands)?;
            self.plan_run_top_command();
        }

        self.plan_gpu_effects_visible()?;
        // FIXME: if we're reading out to a texture, then we do not need this copy at all.
        // However, this fact is only known at runtime. So, should we defer running the
        // CopyBufferToBuffer instead to the runtime?
        self.push(Low::ReadBuffer {
            copy_src_buffer: buffer,
            source_buffer,
            source_layout: byte_layout,
            size,
            offset: (0, 0),
            target_image,
        })?;

        Ok(())
    }

    pub(crate) fn render_staging_to_output(
        &mut self,
        idx: Register,
        dst: Register,
    ) -> Result<(), LaunchError> {
        let RegisterMap::Image { reg_texture, .. } = self.allocate_register(idx)?.clone() else {
            // Can not render to non-image targets.
            return Err(LaunchError::InternalCommandError(line!()));
        };

        let target_image = self.ingest_image_data(dst)?;
        self.render_map.insert(dst, target_image);

        // FIXME: have the caller provide this directly?
        let dst_format = {
            let reg_texture = self.buffer_plan.get_register_texture(dst)?;
            let descriptor = &self.buffer_plan.texture[reg_texture.0];
            let descriptor = ImageDescriptor::new(descriptor)?;
            descriptor.format
        };

        self.copy_staging_to_texture(reg_texture)?;
        self.operands.push(reg_texture);
        let pipeline = {
            let shader =
                shaders::FragmentShaderInvocation::PaintOnTop(shaders::PaintOnTopKind::Copy);

            let vertex = self.vertex_shader(
                Some(shaders::VertexShader::Noop),
                shader_include_to_spirv(shaders::VERT_NOOP),
            )?;

            let shader = shader.shader();
            let key = shader.key();
            let spirv = shader.spirv_source();

            let fragment = self.fragment_shader(key, shader_include_to_spirv_static(spirv))?;
            let fragment_bind_data = shader
                .binary_data(&mut self.binary_data)
                .map(|data| self.ingest_buffer_init(data))
                .map(|data| BufferBind::Planned { data })
                .unwrap_or(BufferBind::None);

            let arguments = shader.num_args();
            self.prepare_simple_pipeline(SimpleRenderPipelineDescriptor {
                pipeline_target: PipelineTarget::PreComputedGroup {
                    target_format: dst_format,
                },
                vertex_bind_data: BufferBind::Set {
                    data: bytemuck::cast_slice(&Self::FULL_VERTEX_BUFFER[..]),
                },
                fragment_texture: TextureBind::Textures(arguments as usize),
                fragment_bind_data,
                fragment_knob: KnobUsage::Noop,
                vertex: ShaderBind::ShaderMain(vertex),
                fragment: ShaderBind::ShaderMain(fragment),
            })?
        };

        let texture_view = self.render_view(dst)?;
        let attachment = ColorAttachmentDescriptor {
            texture_view,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                store: StoreOp::Store,
            },
        };

        self.push(Low::BeginCommands)?;
        self.push(Low::BeginRenderPass(RenderPassDescriptor {
            color_attachments: vec![attachment],
            depth_stencil: None,
        }))?;
        self.render(pipeline)?;
        self.push(Low::EndRenderPass)?;
        self.push(Low::EndCommands)?;
        self.plan_run_top_command();

        Ok(())
    }

    /// Instruct to make target view of a texture.
    pub(crate) fn texture_view(&mut self, dst: Texture) -> Result<usize, LaunchError> {
        let texture = self
            .texture_map
            .get(&dst)
            // The texture was never allocated. Has it been initialized?
            .ok_or_else(|| LaunchError::InternalCommandError(line!()))?
            .device;

        let descriptor = TextureViewDescriptor { texture };

        let id = self.texture_views;
        self.push(Low::TextureView(descriptor))?;

        Ok(id)
    }

    /// Instruct to make a target view of a renderable output.
    pub(crate) fn render_view(&mut self, dst: Register) -> Result<usize, LaunchError> {
        let id = self.texture_views;
        self.push(Low::RenderView(dst))?;
        Ok(id)
    }

    fn make_quad_bind_group(&mut self) -> usize {
        let bind_group_layouts = &mut self.bind_group_layouts;
        let instructions = &mut self.instructions;
        let instruction_pointer = &mut self.instruction_pointer;
        *self.quad_group_layout.get_or_insert_with(|| {
            let descriptor = BindGroupLayoutDescriptor {
                entries: vec![wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(64),
                        ty: wgpu::BufferBindingType::Uniform,
                    },
                    count: None,
                }],
            };

            *instruction_pointer += 1;
            instructions.extend_one(Low::BindGroupLayout(descriptor));
            let descriptor_id = *bind_group_layouts;
            *bind_group_layouts += 1;
            descriptor_id
        })
    }

    fn make_generic_fragment_bind_group(&mut self) -> usize {
        let bind_group_layouts = &mut self.bind_group_layouts;
        let instructions = &mut self.instructions;
        let instruction_pointer = &mut self.instruction_pointer;
        *self.fragment_data_group_layout.get_or_insert_with(|| {
            let descriptor = BindGroupLayoutDescriptor {
                entries: vec![wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        has_dynamic_offset: false,
                        min_binding_size: None,
                        ty: wgpu::BufferBindingType::Uniform,
                    },
                    count: None,
                }],
            };

            *instruction_pointer += 1;
            instructions.extend_one(Low::BindGroupLayout(descriptor));
            let descriptor_id = *bind_group_layouts;
            *bind_group_layouts += 1;
            descriptor_id
        })
    }

    fn make_paint_group_layout(&mut self, count: usize) -> usize {
        let bind_group_layouts = &mut self.bind_group_layouts;
        let instructions = &mut self.instructions;
        let instruction_pointer = &mut self.instruction_pointer;
        *self.paint_group_layout.entry(count).or_insert_with(|| {
            let mut entries = vec![wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            }];

            for i in 0..count {
                entries.push(wgpu::BindGroupLayoutEntry {
                    binding: 1 + i as u32,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                });
            }

            let descriptor = BindGroupLayoutDescriptor { entries };
            *instruction_pointer += 1;
            instructions.extend_one(Low::BindGroupLayout(descriptor));

            let descriptor_id = *bind_group_layouts;
            *bind_group_layouts += 1;
            descriptor_id
        })
    }

    fn make_stage_group(&mut self, binding: u32) -> usize {
        use shaders::stage::StageKind;
        let bind_group_layouts = &mut self.bind_group_layouts;
        let instructions = &mut self.instructions;
        let instruction_pointer = &mut self.instruction_pointer;

        // For encoding we have two extra bindings, sampler and in_texture.
        let encode: bool = binding > StageKind::ALL.len() as u32;

        *self.stage_group_layout.entry(binding).or_insert_with(|| {
            let mut entries = vec![];
            for (num, _) in StageKind::ALL.iter().enumerate() {
                let i = num as u32;
                if i != binding {
                    continue;
                }

                entries.push(wgpu::BindGroupLayoutEntry {
                    binding: i,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                });
            }

            if encode {
                entries.push(wgpu::BindGroupLayoutEntry {
                    binding: 32,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                });

                entries.push(wgpu::BindGroupLayoutEntry {
                    binding: 33,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                });
            } else {
                entries.push(wgpu::BindGroupLayoutEntry {
                    binding: 34,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                });
            }

            let descriptor = BindGroupLayoutDescriptor { entries };
            *instruction_pointer += 1;
            instructions.extend_one(Low::BindGroupLayout(descriptor));
            let descriptor_id = *bind_group_layouts;
            *bind_group_layouts += 1;
            descriptor_id
        })
    }

    fn make_paint_layout(&mut self, desc: &SimpleRenderPipelineDescriptor) -> usize {
        let quad_bind_group = self.make_quad_bind_group();

        let mut bind_group_layouts = vec![quad_bind_group];

        match desc.fragment_texture {
            TextureBind::Textures(0) => {}
            TextureBind::Textures(count) => {
                bind_group_layouts.push(self.make_paint_group_layout(count))
            }
            TextureBind::PreComputedGroup { layout, .. } => {
                bind_group_layouts.push(layout);
            }
        };

        if let BufferBind::None = desc.fragment_bind_data {
            let layouts = &mut self.pipeline_layouts;
            let instructions = &mut self.instructions;
            let instruction_pointer = &mut self.instruction_pointer;

            *self.paint_pipeline_layout.get_or_insert_with(|| {
                let descriptor = PipelineLayoutDescriptor {
                    bind_group_layouts,
                    push_constant_ranges: &[],
                };

                *instruction_pointer += 1;
                instructions.extend_one(Low::PipelineLayout(descriptor));
                let descriptor_id = *layouts;
                *layouts += 1;
                descriptor_id
            })
        } else {
            bind_group_layouts.push(self.make_generic_fragment_bind_group());

            let layouts = &mut self.pipeline_layouts;
            let instructions = &mut self.instructions;
            let instruction_pointer = &mut self.instruction_pointer;

            let descriptor = PipelineLayoutDescriptor {
                bind_group_layouts,
                push_constant_ranges: &[],
            };

            *instruction_pointer += 1;
            instructions.extend_one(Low::PipelineLayout(descriptor));
            let descriptor_id = *layouts;
            *layouts += 1;
            descriptor_id
        }
    }

    fn make_event(&mut self) -> Event {
        let id = self.events;
        self.events += 1;
        Event(id)
    }

    fn shader(&mut self, desc: ShaderDescriptor) -> Result<usize, LaunchError> {
        let idx = self.shaders;
        self.push(Low::Shader(desc))?;
        Ok(idx)
    }

    fn fragment_shader(
        &mut self,
        kind: Option<shaders::FragmentShaderKey>,
        source: Cow<'static, [u32]>,
    ) -> Result<usize, LaunchError> {
        if let Some(&shader) = kind.as_ref().and_then(|k| self.fragment_shaders.get(&k)) {
            return Ok(shader);
        }

        self.shader(ShaderDescriptor {
            name: "",
            source_spirv: source,
            key: kind.map(Into::into),
        })
    }

    fn vertex_shader(
        &mut self,
        kind: Option<shaders::VertexShader>,
        source: Cow<'static, [u32]>,
    ) -> Result<usize, LaunchError> {
        if let Some(&shader) = kind.and_then(|k| self.vertex_shaders.get(&k)) {
            return Ok(shader);
        }

        self.shader(ShaderDescriptor {
            name: "",
            source_spirv: source,
            key: kind.map(Into::into),
        })
    }

    #[rustfmt::skip]
    fn simple_quad_buffer(&mut self) -> DeviceBuffer {
        let buffers = &mut self.buffers;
        let instructions = &mut self.instructions;
        let instruction_pointer = &mut self.instruction_pointer;
        let mut data = &mut self.binary_data;

        *self.simple_quad_buffer.get_or_insert_with(|| {
            // Sole quad!
            let content: &'static [f32; 8] = &[
                1.0, 1.0,
                1.0, 0.0,
                0.0, 1.0,
                0.0, 0.0,
            ];

            let content = Self::append_range(&mut data, content);

            let descriptor = BufferDescriptorInit {
                usage: BufferUsage::InVertices,
                content,
            };

            *instruction_pointer += 1;
            instructions.extend_one(Low::BufferInit(descriptor));

            let buffer = *buffers;
            *buffers += 1;
            DeviceBuffer(buffer)
        })
    }

    fn simple_render_pipeline(
        &mut self,
        desc: &SimpleRenderPipelineDescriptor,
    ) -> Result<usize, LaunchError> {
        // Careful of `RenderPipelineKey` if changed, i.e. no longer deterministic from other `desc` fields.
        let layout = self.make_paint_layout(desc);
        let format = match desc.pipeline_target {
            PipelineTarget::Texture(texture) => {
                let format = self.texture_map[&texture].format.format;
                format
            }
            PipelineTarget::PreComputedGroup { target_format } => target_format,
        };

        let (vertex, vertex_entry_point) = match desc.vertex {
            ShaderBind::ShaderMain(shader) => (shader, "main"),
            ShaderBind::Shader { id, entry_point } => (id, entry_point),
        };
        let (fragment, fragment_entry_point) = match desc.fragment {
            ShaderBind::ShaderMain(shader) => (shader, "main"),
            ShaderBind::Shader { id, entry_point } => (id, entry_point),
        };

        let desc_vertex = self.info.shader_by_idx.get(&vertex);
        let desc_fragment = self.info.shader_by_idx.get(&fragment);
        match (desc_vertex, desc_fragment) {
            (Some(v), Some(f)) => {
                let key = RenderPipelineKey {
                    pipeline_flavor: PipelineLayoutKey::Simple,
                    vertex_module: v.clone(),
                    vertex_entry: vertex_entry_point,
                    fragment_module: f.clone(),
                    fragment_entry: fragment_entry_point,
                    primitive: PrimitiveState::TriangleStrip,
                };

                self.info
                    .pipeline_by_op
                    .insert(self.instruction_pointer, key);
            }
            _ => {}
        }

        let pipeline = self.render_pipelines;
        self.push(Low::RenderPipeline(RenderPipelineDescriptor {
            vertex: VertexState {
                entry_point: vertex_entry_point,
                vertex_module: vertex,
            },
            fragment: FragmentState {
                entry_point: fragment_entry_point,
                fragment_module: fragment,
                // Careful of `RenderPipelineKey` if changed.
                targets: vec![wgpu::ColorTargetState {
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                    format,
                }],
            },
            // Careful of `RenderPipelineKey` if changed.
            primitive: PrimitiveState::TriangleStrip,
            // Careful of `RenderPipelineKey` if changed.
            layout,
        }))?;

        Ok(pipeline)
    }

    fn make_sampler(&mut self, descriptor: SamplerDescriptor) -> usize {
        let instructions = &mut self.instructions;
        let instruction_pointer = &mut self.instruction_pointer;
        let sampler = &mut self.sampler;
        *self
            .known_samplers
            .entry(descriptor)
            .or_insert_with_key(|desc| {
                let sampler_id = *sampler;
                *instruction_pointer += 1;
                instructions.extend_one(Low::Sampler(SamplerDescriptor {
                    address_mode: desc.address_mode,
                    border_color: desc.border_color,
                    resize_filter: desc.resize_filter,
                }));
                *sampler += 1;
                sampler_id
            })
    }

    fn make_bind_group_sampled_texture(&mut self, count: usize) -> Result<usize, LaunchError> {
        let start_of_operands = match self.operands.len().checked_sub(count) {
            None => return Err(LaunchError::InternalCommandError(line!())),
            Some(i) => i,
        };

        let sampler = self.make_sampler(SamplerDescriptor {
            address_mode: wgpu::AddressMode::default(),
            border_color: None,
            resize_filter: wgpu::FilterMode::Nearest,
        });

        let mut entries = vec![BindingResource::Sampler(sampler)];
        let textures: Vec<_> = self.operands.drain(start_of_operands..).collect();
        for texture in textures {
            let view = self.texture_view(texture)?;
            entries.push(BindingResource::TextureView(view));
        }

        let group = self.bind_groups;
        let descriptor = BindGroupDescriptor {
            layout_idx: self.make_paint_group_layout(count),
            entries,
            sparse: vec![],
        };

        self.push(Low::BindGroup(descriptor))?;
        Ok(group)
    }

    fn make_opto_fragment_group(
        &mut self,
        binding: u32,
        // The staging texture, whose staging texture is bound to IO-storage image.
        texture: Texture,
        // The non-staging texture which we bind to the sampler.
        view: Option<Texture>,
    ) -> Result<usize, LaunchError> {
        // FIXME: not always an attachment.
        let texture = self
            .staging_map
            .get(&texture)
            .ok_or_else(|| LaunchError::InternalCommandError(line!()))?
            .device;

        // FIXME: could be cached.
        let image_id = self.texture_views;
        self.push(Low::TextureView(TextureViewDescriptor { texture }))?;

        let mut sparse;

        // For encoding we have two extra bindings, sampler and in_texture.
        if let Some(view) = view {
            if binding < 16 {
                sparse = vec![(binding, BindingResource::TextureView(image_id))];
            } else {
                sparse = vec![];
            }

            // FIXME: unnecessary duplication.
            let sampler = self.make_sampler(SamplerDescriptor {
                address_mode: wgpu::AddressMode::default(),
                border_color: None,
                resize_filter: wgpu::FilterMode::Nearest,
            });

            let view = self
                .texture_map
                .get(&view)
                .ok_or_else(|| LaunchError::InternalCommandError(line!()))?
                .device;

            // FIXME: could be cached.
            let view_id = self.texture_views;
            self.push(Low::TextureView(TextureViewDescriptor { texture: view }))?;

            sparse.push((32, BindingResource::TextureView(view_id)));
            sparse.push((33, BindingResource::Sampler(sampler)));
        } else {
            if binding < 16 {
                sparse = vec![(binding, BindingResource::TextureView(image_id))];
            } else {
                sparse = vec![];
            }

            let sampler = self.make_sampler(SamplerDescriptor {
                address_mode: wgpu::AddressMode::default(),
                border_color: None,
                resize_filter: wgpu::FilterMode::Nearest,
            });

            sparse.push((34, BindingResource::Sampler(sampler)));
        }

        let group = self.bind_groups;
        let descriptor = BindGroupDescriptor {
            layout_idx: self.make_stage_group(binding),
            entries: vec![],
            sparse,
        };

        self.push(Low::BindGroup(descriptor))?;
        Ok(group)
    }

    fn make_bound_buffer(
        &mut self,
        bind: BufferBind<'_>,
        knob: KnobUsage,
        layout_idx: usize,
    ) -> Result<Option<usize>, LaunchError> {
        let buffer = match bind {
            BufferBind::None => return Ok(None),
            BufferBind::Set { data } => {
                let buffer = self.buffers;
                let content = self.ingest_data(data);
                self.push(Low::BufferInit(BufferDescriptorInit {
                    content,
                    usage: BufferUsage::Uniform,
                }))?;
                DeviceBuffer(buffer)
            }
            BufferBind::Planned { data } => {
                let buffer = self.buffers;
                self.push(Low::BufferInit(BufferDescriptorInit {
                    content: data,
                    usage: BufferUsage::Uniform,
                }))?;
                DeviceBuffer(buffer)
            }
        };

        match knob {
            KnobUsage::Noop => {}
            KnobUsage::CopyFrom {
                buffer: source,
                range,
            } => {
                debug_assert!(
                    range.end >= range.start,
                    "knob is an internal, consistent property"
                );

                self.push(Low::BeginCommands)?;
                self.push(Low::CopyBufferToBuffer {
                    source_buffer: source,
                    source: range.start,
                    target_buffer: buffer,
                    target: 0,
                    size: range.end - range.start,
                })?;
                self.push(Low::EndCommands)?;
                self.push(Low::RunTopCommand)?;
            }
        }

        let group = self.bind_groups;
        let descriptor = BindGroupDescriptor {
            layout_idx,
            entries: vec![BindingResource::Buffer {
                buffer_idx: buffer.0,
                offset: 0,
                size: None,
            }],
            sparse: vec![],
        };

        self.push(Low::BindGroup(descriptor))?;
        Ok(Some(group))
    }

    /// Render the pipeline, after all customization and buffers were bound..
    fn prepare_simple_pipeline(
        &mut self,
        descriptor: SimpleRenderPipelineDescriptor,
    ) -> Result<SimpleRenderPipeline, LaunchError> {
        let pipeline = self.simple_render_pipeline(&descriptor)?;
        let buffer = self.simple_quad_buffer();

        let group = match &descriptor.fragment_texture {
            TextureBind::Textures(0) => None,
            &TextureBind::Textures(count) => {
                let group = self.make_bind_group_sampled_texture(count)?;
                Some(group)
            }
            &TextureBind::PreComputedGroup { group, .. } => Some(group),
        };

        let vertex_layout = self.make_quad_bind_group();
        let vertex_bind = self.make_bound_buffer(
            descriptor.vertex_bind_data,
            /* we do not permit knob for the vertex data */ KnobUsage::Noop,
            vertex_layout,
        )?;

        // FIXME: this builds the layout even when it is not required.
        let vertex_layout = self.make_generic_fragment_bind_group();
        let fragment_bind = self.make_bound_buffer(
            descriptor.fragment_bind_data,
            descriptor.fragment_knob,
            vertex_layout,
        )?;

        Ok(SimpleRenderPipeline {
            pipeline,
            buffer,
            group,
            vertex_bind,
            vertices: 4,
            fragment_bind,
        })
    }

    #[rustfmt::skip]
    pub const FULL_VERTEX_BUFFER: [[f32; 2]; 8] = [
        // [min_u, min_v], [0.0, 0.0],
        [0.0, 0.0],
        // [max_u, 0.0], [0.0, 0.0],
        [1.0, 0.0],
        // [max_u, max_v], [0.0, 0.0],
        [1.0, 1.0],
        // [min_u, max_v], [0.0, 0.0],
        [0.0, 1.0],

        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ];

    pub(crate) fn render(&mut self, pipeline: SimpleRenderPipeline) -> Result<(), LaunchError> {
        let SimpleRenderPipeline {
            pipeline,
            group,
            buffer,
            vertex_bind,
            vertices,
            fragment_bind,
        } = pipeline;

        self.push(Low::SetPipeline(pipeline))?;

        let mut group_idx = 0;
        if let Some(quad) = vertex_bind {
            self.push(Low::SetBindGroup {
                group: quad,
                index: group_idx,
                offsets: Cow::Borrowed(&[]),
            })?;
            group_idx += 1;
        }

        if let Some(group) = group {
            self.push(Low::SetBindGroup {
                group,
                index: group_idx,
                offsets: Cow::Borrowed(&[]),
            })?;
            group_idx += 1;
        }

        self.push(Low::SetVertexBuffer {
            buffer: buffer.0,
            slot: 0,
        })?;

        if let Some(bind) = fragment_bind {
            self.push(Low::SetBindGroup {
                group: bind,
                index: group_idx,
                offsets: Cow::Borrowed(&[]),
            })?;
        }

        self.push(Low::DrawOnce { vertices })?;

        Ok(())
    }

    #[rustfmt::skip]
    pub(crate) fn prepare_render(
        &mut self,
        // The function we are using.
        function: &Initializer,
        // The texture we are rendering to.
        target: Texture,
    ) -> Result<SimpleRenderPipeline, LaunchError> {
        match function {
            Initializer::PaintToSelection { texture, selection, target: target_coords, viewport, shader } => {
                let (tex_width, tex_height) = self.texture_map[texture].format.size;

                // FIXME: choose this shader depending on whether target_coords are knob'd or not.
                // Since the screen space transform depends on sizes, the user in commands.rs can
                // not and importantly shouldn't need to care.
                let vertex = self.vertex_shader(
                    Some(shaders::VertexShader::Noop),
                    shader_include_to_spirv(shaders::VERT_NOOP))?;

                let super::ParameterizedFragment { invocation, knob: _ } = shader;

                let shader = invocation.shader();
                let key = shader.key();
                let spirv = shader.spirv_source();

                let fragment = self.fragment_shader(key, shader_include_to_spirv_static(spirv))?;

                let buffer: [[f32; 2]; 8];
                let min_u = (selection.x as f32) / (tex_width.get() as f32);
                let max_u = (selection.max_x as f32) / (tex_width.get() as f32);
                let min_v = (selection.y as f32) / (tex_height.get() as f32);
                let max_v = (selection.max_y as f32) / (tex_height.get() as f32);

                let coords = target_coords.to_screenspace_coords(viewport);

                // std430
                buffer = [
                    [min_u, min_v],
                    [max_u, min_v],
                    [max_u, max_v],
                    [min_u, max_v],

                    coords[0],
                    coords[1],
                    coords[2],
                    coords[3],
                ];

                self.prepare_simple_pipeline(SimpleRenderPipelineDescriptor{
                    pipeline_target: PipelineTarget::Texture(target),
                    vertex_bind_data: BufferBind::Set {
                        data: bytemuck::cast_slice(&buffer[..]),
                    },
                    fragment_texture: TextureBind::Textures(1),
                    fragment_bind_data: BufferBind::None,
                    // FIXME: see knob'able data.
                    fragment_knob: KnobUsage::Noop,
                    vertex: ShaderBind::ShaderMain(vertex),
                    fragment: ShaderBind::ShaderMain(fragment),
                })
            },
            Initializer::PaintFullScreen { shader } => {
                let vertex = self.vertex_shader(
                    Some(shaders::VertexShader::Noop),
                    shader_include_to_spirv(shaders::VERT_NOOP))?;

                let super::ParameterizedFragment { invocation, knob } = shader;

                let shader = invocation.shader();
                let key = shader.key();
                let spirv = shader.spirv_source();

                let fragment = self.fragment_shader(key, shader_include_to_spirv_static(spirv))?;
                let fragment_bind_data = shader.binary_data(&mut self.binary_data)
                    .map(|data| self.ingest_buffer_init(data))
                    .map(|data| BufferBind::Planned { data })
                    .unwrap_or(BufferBind::None);

                let fragment_knob = self.plan_knob_buffer(knob, &fragment_bind_data)?;
                let arguments = shader.num_args();

                self.prepare_simple_pipeline(SimpleRenderPipelineDescriptor{
                    pipeline_target: PipelineTarget::Texture(target),
                    vertex_bind_data: BufferBind::Set {
                        data: bytemuck::cast_slice(&Self::FULL_VERTEX_BUFFER[..]),
                    },
                    fragment_texture: TextureBind::Textures(arguments as usize),
                    fragment_bind_data,
                    fragment_knob,
                    vertex: ShaderBind::ShaderMain(vertex),
                    fragment: ShaderBind::ShaderMain(fragment),
                })
            },
            Initializer::ToLinearOpto { parameter, stage_kind } => {
                let vertex = self.vertex_shader(
                    Some(shaders::VertexShader::Noop),
                    shader_include_to_spirv(shaders::VERT_NOOP))?;

                let fragment = self.fragment_shader(
                    Some(shaders::FragmentShaderKey::Convert(shaders::Direction::Decode, *stage_kind)),
                    shader_include_to_spirv(stage_kind.decode_src()))?;

                let buffer = parameter.serialize_std140();
                // FIXME: see below, shaderc requires renamed entry points to "main".
                let _entry_point = stage_kind.encode_entry_point();

                let layout = self.make_stage_group(stage_kind.decode_binding());

                let group = self.make_opto_fragment_group(
                    stage_kind.decode_binding(),
                    target,
                    None,
                )?;

                self.prepare_simple_pipeline(SimpleRenderPipelineDescriptor{
                    pipeline_target: PipelineTarget::PreComputedGroup {
                        target_format: parameter.linear_format(),
                    },
                    vertex_bind_data: BufferBind::Set {
                        data: bytemuck::cast_slice(&Self::FULL_VERTEX_BUFFER[..]),
                    },
                    fragment_texture: TextureBind::PreComputedGroup {
                        group,
                        layout,
                    },
                    fragment_bind_data: BufferBind::Set {
                        data: bytemuck::cast_slice(&buffer[..]),
                    },
                    fragment_knob: KnobUsage::Noop,
                    vertex: ShaderBind::ShaderMain(vertex),
                    fragment: ShaderBind::Shader {
                        // FIXME: for some weird reason this MUST be `main` instead of the true
                        // entry point. This is probably a 'bug' in the pass through `shaderc` in
                        // preprocessing.
                        entry_point: "main",
                        id: fragment,
                    },
                })
            }
            Initializer::FromLinearOpto { parameter, stage_kind } => {
                let vertex = self.vertex_shader(
                    Some(shaders::VertexShader::Noop),
                    shader_include_to_spirv(shaders::VERT_NOOP))?;

                let fragment = self.fragment_shader(
                    Some(shaders::FragmentShaderKey::Convert(shaders::Direction::Encode, *stage_kind)),
                    shader_include_to_spirv(stage_kind.encode_src()))?;

                let buffer = parameter.serialize_std140();
                // FIXME: see below, shaderc requires renamed entry points to "main".
                let _entry_point = stage_kind.decode_entry_point();

                let layout = self.make_stage_group(stage_kind.encode_binding());

                let group = self.make_opto_fragment_group(
                    stage_kind.encode_binding(),
                    target,
                    Some(target),
                )?;

                self.prepare_simple_pipeline(SimpleRenderPipelineDescriptor{
                    pipeline_target: PipelineTarget::PreComputedGroup {
                        target_format: if let Some(stage) = parameter.stage_kind() {
                            stage.texture_format()
                        } else {
                            parameter.linear_format()
                        }
                    },
                    vertex_bind_data: BufferBind::Set {
                        data: bytemuck::cast_slice(&Self::FULL_VERTEX_BUFFER[..]),
                    },
                    fragment_texture: TextureBind::PreComputedGroup {
                        group,
                        layout,
                    },
                    fragment_bind_data: BufferBind::Set {
                        data: bytemuck::cast_slice(&buffer[..]),
                    },
                    fragment_knob: KnobUsage::Noop,
                    vertex: ShaderBind::ShaderMain(vertex),
                    fragment: ShaderBind::Shader {
                        // FIXME: for some weird reason this MUST be `main` instead of the true
                        // entry point. This is probably a 'bug' in the pass through `shaderc` in
                        // preprocessing.
                        entry_point: "main",
                        id: fragment,
                    },
                })
            }
        }
    }

    pub(crate) fn prepare_buffer_write(
        &mut self,
        // The function we are using.
        function: &super::BufferWrite,
        // The texture we are rendering to.
        dst: Buffer,
    ) -> Result<(), LaunchError> {
        self.ensure_device_buffer(dst)?;

        match function {
            super::BufferWrite::Zero => {
                let buf_layout = self
                    .buffer_plan
                    .buffer
                    .get(dst.0)
                    .ok_or_else(|| LaunchError::InternalCommandError(line!()))?;

                let map = self
                    .buffer_map
                    .get(&dst)
                    .ok_or_else(|| LaunchError::InternalCommandError(line!()))?;

                let size = buf_layout.u64_len();
                let target_buffer = map.device;

                self.push(Low::BeginCommands)?;
                self.push(Low::ZeroBuffer {
                    start: 0,
                    size,
                    target_buffer,
                })?;
                self.push(Low::EndCommands)?;
                self.push(Low::RunTopCommand)?;

                Ok(())
            }
            super::BufferWrite::Put {
                placement,
                data,
                knob,
            } => {
                let buf_layout = self
                    .buffer_plan
                    .buffer
                    .get(dst.0)
                    .cloned()
                    .ok_or_else(|| LaunchError::InternalCommandError(line!()))?;

                let map = self
                    .buffer_map
                    .get(&dst)
                    .map(Clone::clone)
                    .ok_or_else(|| LaunchError::InternalCommandError(line!()))?;

                let size = buf_layout.u64_len();
                let data_range = self.ingest_data(data);

                let knob = match knob {
                    None => KnobUser::None,
                    Some(knob_id) => KnobUser::Runtime(*knob_id),
                };

                match self.plan_knob_data_range(&knob, data_range.clone())? {
                    KnobUsage::Noop => {}
                    KnobUsage::CopyFrom { .. } => unreachable!("not with any knob user"),
                }

                let stage_buffer = DeviceBuffer(self.buffers);
                self.push(Low::BufferInit(BufferDescriptorInit {
                    content: data_range,
                    usage: BufferUsage::DataBuffer,
                }))?;

                self.push(Low::BeginCommands)?;
                self.push(Low::CopyBufferToBuffer {
                    source_buffer: stage_buffer,
                    source: 0,
                    target_buffer: map.device,
                    target: placement.start as u64,
                    size,
                })?;
                self.push(Low::EndCommands)?;
                self.push(Low::RunTopCommand)?;

                Ok(())
            }
        }
    }

    fn plan_knob_buffer(
        &mut self,
        knob: &KnobUser,
        fragment_bind_data: &BufferBind,
    ) -> Result<KnobUsage, LaunchError> {
        // Nothing to plan.
        if let KnobUser::None = knob {
            return Ok(KnobUsage::Noop);
        };

        // Only `planned` bindings can be knob controlled!
        let BufferBind::Planned { data } = fragment_bind_data else {
            return Err(LaunchError::InternalCommandError(line!()));
        };

        self.plan_knob_data_range(knob, data.clone())
    }

    fn plan_knob_data_range(
        &mut self,
        knob: &KnobUser,
        data: Range<usize>,
    ) -> Result<KnobUsage, LaunchError> {
        match knob {
            KnobUser::None => return Ok(KnobUsage::Noop),
            &KnobUser::Runtime(knob) => {
                self.info.knobs.insert(
                    knob,
                    KnobDescriptor {
                        range: data.clone(),
                    },
                );

                Ok(KnobUsage::Noop)
            }
            KnobUser::Buffer { buffer, range } => {
                let device = self.ensure_device_buffer(*buffer)?;

                Ok(KnobUsage::CopyFrom {
                    buffer: device,
                    range: range.clone(),
                })
            }
        }
    }

    /// Ingest the data into the encoder's active buffer data.
    fn ingest_data(&mut self, data: &[impl bytemuck::Pod]) -> Range<usize> {
        Self::append_range(&mut self.binary_data, data)
    }

    fn ingest_buffer_init(&mut self, shader: BufferInitContent) -> Range<usize> {
        match shader {
            BufferInitContent::Owned(data) => Self::append_range(&mut self.binary_data, &data),
            BufferInitContent::Defer { start, end } => start..end,
        }
    }

    fn append_range(into: &mut Vec<u8>, data: &[impl bytemuck::Pod]) -> Range<usize> {
        let start = into.len();
        into.extend_from_slice(bytemuck::cast_slice(data));
        let end = into.len();
        start..end
    }

    pub(crate) fn io_map(&self) -> run::IoMap {
        let mut io_map = run::IoMap::default();

        for (&register, texture) in &self.input_map {
            io_map.inputs.insert(register, texture.0);
        }

        for (&register, texture) in &self.output_map {
            io_map.outputs.insert(register, texture.0);
        }

        for (&register, texture) in &self.render_map {
            io_map.renders.insert(register, texture.0);
        }

        io_map
    }

    /// Using the derived pool plan, extract all utilized buffers from the pool.
    pub(crate) fn extract_buffers(
        &self,
        buffers: &mut Vec<run::Image>,
        pool: &mut Pool,
    ) -> Result<(), LaunchError> {
        for (&pool_key, &texture) in &self.pool_plan.buffer {
            let mut entry = pool
                .entry(pool_key)
                .ok_or_else(|| LaunchError::InternalCommandError(line!()))?;
            let buffer = &mut buffers[texture.0].data;

            // Decide how to retrieve this image from the pool.
            if buffer.as_bytes().is_none() {
                // Just take the buffer if we are allowed to...
                if !entry.trade(buffer) {
                    // Would need to copy from the GPU.
                    return Err(LaunchError::InternalCommandError(line!()));
                }

                // FIXME: for now we enforce host-reachability.
                // This isn't necessary if we know that we should only take GPU images belonging to
                // a particular device that we start on. This only works because this method is
                // only executed by `launch`, which knows the device as it chooses the device
                // itself.
                if buffer.as_bytes().is_none() {
                    return Err(LaunchError::InternalCommandError(line!()));
                }
            }
        }

        Ok(())
    }

    pub(crate) fn finalize(&mut self) -> Result<(), LaunchError> {
        self.plan_gpu_effects_visible()?;

        if !self.operands.is_empty() {
            return Err(LaunchError::InternalCommandError(line!()));
        }

        Ok(())
    }
}

impl RegisterMap {}

fn shader_include_to_spirv_static(src: Cow<'static, [u8]>) -> Cow<'static, [u32]> {
    if let Cow::Borrowed(src) = src {
        if let Ok(cast) = bytemuck::try_cast_slice(src) {
            return Cow::Borrowed(cast);
        }
    }

    shader_include_to_spirv(&src)
}

fn shader_include_to_spirv(src: &[u8]) -> Cow<'static, [u32]> {
    assert!(src.len() % 4 == 0);
    let mut target = vec![0u32; src.len() / 4];
    bytemuck::cast_slice_mut(&mut target).copy_from_slice(src);
    Cow::Owned(target)
}
