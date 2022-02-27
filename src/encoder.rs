//! Produce a stream of `Low` instructions that can be executed on a particular device.
use core::num::{NonZeroU32, NonZeroU64};
use std::borrow::Cow;
use std::collections::HashMap;

use crate::buffer::{
    Block, BufferLayout, Color, SampleBits, SampleParts, Samples, Texel, Transfer,
};
use crate::command::Register;
use crate::pool::Pool;
use crate::program::{
    BindGroupDescriptor, BindGroupLayoutDescriptor, BindingResource, Buffer, BufferDescriptor,
    BufferDescriptorInit, BufferInitContent, BufferUsage, Capabilities, ColorAttachmentDescriptor,
    DeviceBuffer, DeviceTexture, FragmentState, Function, ImageBufferAssignment, ImageBufferPlan,
    ImageDescriptor, ImagePoolPlan, LaunchError, Low, PipelineLayoutDescriptor, PrimitiveState,
    RenderPassDescriptor, RenderPipelineDescriptor, SamplerDescriptor, ShaderDescriptor,
    StagingDescriptor, Texture, TextureDescriptor, TextureUsage, TextureViewDescriptor,
    VertexState,
};
use crate::util::ExtendOne;
use crate::{run, shaders};

/// The encoder tracks the supposed state of `run::Descriptors` without actually executing them.
#[derive(Default)]
pub(crate) struct Encoder<Instructions: ExtendOne<Low> = Vec<Low>> {
    pub(crate) instructions: Instructions,
    /// The allocate binary data for runtime execution.
    pub(crate) binary_data: Vec<u8>,

    // Fields from `run::Descriptors` that simulate the length.
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

    // Fields related to memoization of some inputs.
    /// Declare where we put our input registers.
    input_map: HashMap<Register, Texture>,
    /// Declare where we intend to write our outputs.
    output_map: HashMap<Register, Texture>,
    /// The Bind Group layer Descriptor used in fragment shader, set=1.
    /// This is keyed by the number of descriptors for that layout.
    paint_group_layout: HashMap<usize, BindGroupLayoutIdx>,
    /// The Bind Group Descriptor used in vertex buffer, set=0.
    quad_group_layout: Option<BindGroupLayoutIdx>,
    /// The Bind Group Descriptor for set=2, used for parameters of fragment shader.
    fragment_data_group_layout: Option<BindGroupLayoutIdx>,
    /// The Pipeline Descriptor used in generic paint shaders.
    /// Will use the quad group and fragment data group layouts.
    paint_pipeline_layout: Option<PipelineLayoutIdx>,
    /// The Bind Group Descriptor layout for the staging fragment shader.
    /// Alternative for the fragment data group.
    /// Since there is a limit on the number of active storage textures, we have one layout for
    /// each potential usage.
    stage_group_layout: HashMap<u32, BindGroupLayoutIdx>,
    /// Memoized Sampler definitions, mapping to their index.
    known_samplers: HashMap<SamplerDescriptor, SamplerIdx>,
    /// Memoized fragment shaders, mapping to their index.
    fragment_shaders: HashMap<shaders::FragmentShaderKey, ShaderIdx>,
    /// Memoized vertex shaders, mapping to their index.
    vertex_shaders: HashMap<VertexShader, ShaderIdx>,
    /// Memoized buffer containing the simple quad.
    simple_quad_buffer: Option<DeviceBuffer>,
    /// The render pipeline state for staging a texture.
    staged_to_pipelines: HashMap<Texture, SimpleRenderPipeline>,
    /// The render pipeline state for undoing staging a texture.
    staged_from_pipelines: HashMap<Texture, SimpleRenderPipeline>,

    // Fields related to tracking command resource usage.
    /// The texture operands collected for the next render preparation.
    operands: Vec<Texture>,
    /// The currently knowns consumes of commands that are being built (not yet executed).
    partial_command_consumes: Vec<Consume>,
    /// The currently knowns clobbers of commands that are being built (not yet executed).
    partial_command_clobbers: Vec<Clobber>,
    /// Delayed copy commands for images.
    register_logical_state: RegisterLogicalState,

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
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
struct BindGroupIdx(usize);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
struct BindGroupLayoutIdx(usize);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
struct BufferIdx(usize);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
struct CommandBufferIdx(usize);

/// The ID of a command execution in this basic block.
/// Ordered by the order in which they were scheduled.
#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
struct ExecuteId(usize);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
struct PipelineLayoutIdx(usize);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
struct RenderPipelineIdx(usize);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
struct SamplerIdx(usize);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
struct ShaderIdx(usize);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
struct LogicalSideEffectIdx(usize);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
struct TextureIdx(usize);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
struct TextureViewIdx(usize);

/// The GPU buffers associated with a register.
/// Supplements the buffer_plan by giving direct mappings to each device resource index in an
/// encoder process.
#[derive(Clone, Debug)]
pub(crate) struct RegisterMap {
    pub(crate) logical_texture: Texture,
    pub(crate) texture: DeviceTexture,
    pub(crate) buffer: DeviceBuffer,
    /// A device buffer with (COPY_DST | MAP_READ) for reading back the texture.
    pub(crate) map_read: Option<DeviceBuffer>,
    /// A device buffer with (COPY_SRC | MAP_WRITE) for initialization the texture.
    pub(crate) map_write: Option<DeviceBuffer>,
    /// A device texture for (de-)normalizing the texture contents.
    pub(crate) staging: Option<DeviceTexture>,
    /// The layout of the buffer.
    /// This might differ from the layout of the corresponding pool image because it must adhere to
    /// the layout requirements of the device. For example, the alignment of each row must be
    /// divisible by 256 etc.
    pub(crate) buffer_layout: BufferLayout,
    /// The format of the non-staging texture.
    pub(crate) texture_format: TextureDescriptor,
    /// The format of the staging texture.
    pub(crate) staging_format: Option<TextureDescriptor>,
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
    /// A texture which we use as an attachment for encoding.
    ///
    /// The current implementation for encoding attaches the staging buffer as a storage image and
    /// the texture as a sampled source texture. However, this means we can not use either of them
    /// as the target attachment. So we have this additional texture that has the right size and
    /// format to be used as a target attachment for encoding while we _never_ actually write to it
    /// (op:write is disabled for that draw call).
    /// FIXME: find a way to avoid this texture allocation.
    temporary_attachment_buffer_for_encoding_remove_if_possible: DeviceTexture,
}

/// The gpu buffer associated with an image buffer.
#[derive(Clone, Debug)]
struct BufferMap {
    device: DeviceBuffer,
    layout: BufferLayout,
}

/// Logical image state and which commands need to be performed to catch up.
///
/// The encoder keeps track of the logical state of a register/texture, which is the state visible
/// to the outside that should _appear_ to have been scheduled. In reality, we aim to perform as
/// little copies as possible, in particular considering the physical allocation map of registers.
///
/// If the program instructs us to copy a texture from register A to B then it may happen that, due
/// to lifetime analysis, these are actually allocated to the same gpu texture. So, this copy may
/// be elided outright.
///
/// Or, the program may instruct us to unstage a texture (copy the linear-representation image to
/// the memory visible buffer), only to stage it due the next operation. This can be elided, or
/// more precisely we may delay the copy until the unstaged memory representation is being used. In
/// other words, we track commands that should have been inserted and ensure they happen:
///
/// - before commands that read from their output
/// - before commands that clobber their inputs
///
/// As a side effect we may group multiple commands into the same command run.
///
/// This contains a DAG of (pending) command execution effects. This is traversed to find dependent
/// but not-yet-scheduled commands. The effect is provided by `Clobber` (incoming edges) while the
/// needs are provided by `Consume` (outgoing edges). Clobbers hold additional information as they
/// will also inform the `RegisterLogicalState` of the kind of the logical resource behind their
/// device resource. We then track which of the device representations is considered 'fresh' for a
/// specific logical resource.
#[derive(Default)]
struct RegisterLogicalState {
    /// Specific Pipelines with register effects. May execute once or multiple times.
    side_effects: Vec<DelayedSideEffects>,
    /// Clobbers that were registered by pipelines.
    clobbers: Vec<Clobber>,
    /// Consumes that were registered by pipelines.
    consumes: Vec<Consume>,
    /// Informs us which consumes would bind to which execution.
    clobber_map: HashMap<Consume, ExecuteId>,
    /// Executed pipelines in order that they were tracked.
    scheduled: Vec<LogicalSideEffectIdx>,
    /// The past-the-end of *executed* pipelines.
    past_executed_id: ExecuteId,
    /// Buffer for a list of commands we need to perform now to avoid output/clobber conditions.
    /// Filled by the methods `resolve` (or any recursive effect it may have).
    eager_commands: Vec<Low>,
}

#[derive(Clone, Debug)]
struct DelayedSideEffects {
    /// Additional outputs clobbered by this pipeline.
    clobber: core::ops::Range<usize>,
    /// Additional outputs consumed by this pipeline.
    consume: core::ops::Range<usize>,
}

/// Some logically tracked resource about to be overwritten.
///
/// This tracks the logical state as well, which is updated to reflect the fact that its logical
/// state is now present in the buffer/texture.
#[derive(Clone, Debug)]
enum Clobber {
    Buffer {
        /// The register which we clobber.
        texture: Texture,
        /// The buffer into which we perform the write.
        device: DeviceBuffer,
    },
    Texture {
        /// The register which we clobber.
        texture: Texture,
        /// The texture into which we perform the write.
        device: DeviceTexture,
    },
}

/// Some logically tracked resource about to be read.
/// A resource for which clobber/consume is tracked, and for which those effects may be delayed.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
enum Consume {
    Buffer(DeviceBuffer),
    Texture(DeviceTexture),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum VertexShader {
    Noop,
}

#[derive(Clone, Copy)]
pub(crate) struct SimpleRenderPipeline {
    pipeline: usize,
    buffer: DeviceBuffer,
    group: Option<BindGroupIdx>,
    vertex_bind: Option<BindGroupIdx>,
    vertices: u32,
    fragment_bind: Option<BindGroupIdx>,
    side_effects: LogicalSideEffectIdx,
}

struct SimpleRenderPipelineDescriptor<'data> {
    pipeline_target: PipelineTarget,
    /// Bind data for (set 0, binding 0).
    vertex_bind_data: BufferBind<'data>,
    /// Texture for (set 1, binding 0)
    fragment_texture: TextureBind<'data>,
    /// Texture for (set 2, binding 0)
    fragment_bind_data: BufferBind<'data>,
    /// The vertex shader to use.
    vertex: ShaderBind,
    /// The fragment shader to use.
    fragment: ShaderBind,
}

enum PipelineTarget {
    Texture(Texture),
    PreComputedGroup { target_format: wgpu::TextureFormat },
}

enum TextureBind<'data> {
    /// Use the currently pushed texture operands.
    /// The arguments are taken from the back of the operand vector.
    Textures(usize),
    PreComputedGroup {
        /// The index of the bind group we're binding to set `1`, the fragment set.
        group: BindGroupIdx,
        /// The layout corresponding to the bind group.
        layout: usize,
        /// Additional clobbers.
        clobber: &'data [Clobber],
        /// Additional consumes.
        consumed: &'data [Consume],
    },
}

enum BufferBind<'data> {
    /// Upload the data, then bind the buffer.
    Set {
        data: &'data [u8],
    },
    /// The data has already been planned beforehand.
    Planned {
        data: BufferInitContent,
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

    pub(crate) fn begin_stack_frame(&mut self, frame: run::Frame) -> Result<(), LaunchError> {
        self.push(Low::StackFrame(frame))
    }

    pub(crate) fn end_stack_frame(&mut self) -> Result<(), LaunchError> {
        self.push(Low::StackPop)
    }

    pub(crate) fn begin_render_pass(
        &mut self,
        pass: RenderPassDescriptor,
    ) -> Result<(), LaunchError> {
        let clobbers = pass
            .textures
            .iter()
            .map(|texture| {
                let device = self
                    .texture_map
                    .get(texture)
                    .ok_or_else(|| LaunchError::InternalCommandError(line!()))?
                    .device;
                Ok(Clobber::Texture {
                    texture: *texture,
                    device,
                })
            })
            .collect::<Result<Vec<Clobber>, _>>()?;
        self.begin_commands(&clobbers)?;
        self.push(Low::BeginRenderPass(pass))?;
        Ok(())
    }

    pub(crate) fn end_and_run_render_pass(&mut self) -> Result<(), LaunchError> {
        self.push(Low::EndRenderPass)?;
        // FIXME: list of consumed inputs based on the render commands that had been provided since
        // starting.
        self.end_and_run_commands(&[])
    }

    pub(crate) fn copy_buffer_to_buffer(
        &mut self,
        src: Register,
        dst: Register,
    ) -> Result<(), LaunchError> {
        let &RegisterMap {
            logical_texture: texture,
            buffer: source_buffer,
            ref buffer_layout,
            ..
        } = self.allocate_register(src)?;

        let size = buffer_layout.u64_len();
        let target_buffer = self.allocate_register(dst)?.buffer;

        let clobbers = vec![Clobber::Buffer {
            texture,
            device: target_buffer,
        }];

        self.begin_commands(&clobbers)?;
        self.push(Low::CopyBufferToBuffer {
            source_buffer,
            size,
            target_buffer,
        })?;

        let consumed = vec![Consume::Buffer(source_buffer)];
        self.end_and_run_commands(&consumed)?;

        Ok(())
    }

    /// Schedule actual command execution.
    ///
    /// Gently reminds us we should provide the clobbers of this command (including pipelines,
    /// copies, etc) by requiring an argument listing them.
    fn begin_commands(&mut self, clobber: &[Clobber]) -> Result<(), LaunchError> {
        self.partial_command_clobbers.extend_from_slice(clobber);
        // Then do the _actual_ command start.
        self.push(Low::BeginCommands)
    }

    fn end_and_run_commands(&mut self, consumed: &[Consume]) -> Result<(), LaunchError> {
        self.partial_command_consumes.extend_from_slice(consumed);
        self.push(Low::EndCommands)?;

        // All inputs and outputs are known now, register this.
        let commands_idx = self.register_logical_state.register_pipeline(
            &self.partial_command_consumes,
            &self.partial_command_clobbers,
        );

        self.push(Low::RunTopCommand)
    }

    /// Validate and then add the command to the encoder.
    ///
    /// This ensures we can keep track of the expected state change, and validate the correct order
    /// of commands. More specific sequencing commands will expect correct order or assume it
    /// internally.
    fn push(&mut self, low: Low) -> Result<(), LaunchError> {
        match low {
            Low::BindGroupLayout(_) => self.bind_group_layouts += 1,
            Low::BindGroup(_) => self.bind_groups += 1,
            Low::Buffer(_) | Low::BufferInit(_) => self.buffers += 1,
            Low::PipelineLayout(_) => self.pipeline_layouts += 1,
            Low::Sampler(_) => self.sampler += 1,
            Low::Shader(_) => self.shaders += 1,
            Low::Texture(_) => self.textures += 1,
            Low::TextureView(_) => self.texture_views += 1,
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
                if group >= self.bind_groups {
                    return Err(LaunchError::InternalCommandError(line!()));
                }
            }
            Low::SetVertexBuffer { buffer, .. } => {
                if buffer >= self.buffers {
                    return Err(LaunchError::InternalCommandError(line!()));
                }
            }
            // TODO: could validate indices.
            Low::DrawOnce { .. } | Low::DrawIndexedZero { .. } | Low::SetPushConstants { .. } => {}
            Low::RunTopCommand => {
                if self.command_buffers == 0 {
                    return Err(LaunchError::InternalCommandError(line!()));
                }

                self.command_buffers -= 1;
            }
            Low::RunBotToTop(num) | Low::RunTopToBot(num) | Low::PopCommands(num) => {
                if num >= self.command_buffers {
                    return Err(LaunchError::InternalCommandError(line!()));
                }

                self.command_buffers -= num;
            }
            // TODO: could validate indices.
            Low::WriteImageToTexture { .. }
            | Low::CopyBufferToTexture { .. }
            | Low::CopyTextureToBuffer { .. }
            | Low::CopyBufferToBuffer { .. } => {
                if !self.is_in_command_encoder {
                    return Err(LaunchError::InternalCommandError(line!()));
                }
            }
            Low::WriteImageToBuffer { .. } | Low::ReadBuffer { .. } => {}
            // No validation for stack frame shuffling.
            // TODO: should we simulate stack height?
            Low::StackFrame(_) | Low::StackPop => {}
        }

        self.instructions.extend_one(low);
        Ok(())
    }

    fn make_texture_descriptor(
        &mut self,
        texture: Texture,
    ) -> Result<ImageDescriptor, LaunchError> {
        let descriptor = &self.buffer_plan.texture[texture.0];

        fn validate_size(layout: &BufferLayout) -> Option<(NonZeroU32, NonZeroU32)> {
            Some((
                NonZeroU32::new(layout.width)?,
                NonZeroU32::new(layout.height)?,
            ))
        }

        let size = validate_size(&descriptor.layout)
            .ok_or_else(|| LaunchError::InternalCommandError(line!()))?;
        let mut staging = None;

        let format = match descriptor.texel {
            Texel {
                block: Block::Pixel,
                samples:
                    Samples {
                        bits: SampleBits::Int8x4,
                        parts: SampleParts::Rgba,
                    },
                color:
                    Color::Rgb {
                        transfer: Transfer::Srgb,
                        ..
                    },
            } => wgpu::TextureFormat::Rgba8UnormSrgb,
            Texel {
                block: Block::Pixel,
                samples:
                    Samples {
                        bits: SampleBits::Int8x4,
                        parts: SampleParts::Rgba,
                    },
                color:
                    Color::Rgb {
                        transfer: Transfer::Linear,
                        ..
                    },
            } => wgpu::TextureFormat::Rgba8Unorm,
            Texel {
                block: Block::Pixel,
                samples,
                color: Color::Rgb { transfer, .. },
            }
            | Texel {
                block: Block::Pixel,
                samples,
                color: Color::Scalars { transfer, .. },
            } => {
                let parameter = shaders::stage::XyzParameter {
                    transfer: transfer.into(),
                    parts: samples.parts,
                    bits: samples.bits,
                };

                let result = parameter.linear_format();
                let stage_kind = parameter
                    .stage_kind()
                    // Unsupported format.
                    .ok_or_else(|| LaunchError::InternalCommandError(line!()))?;

                staging = Some(StagingDescriptor {
                    stage_kind,
                    parameter,
                });

                result
            }
            Texel {
                block: Block::Pixel,
                samples:
                    Samples {
                        bits,
                        parts: parts @ (SampleParts::LChA | SampleParts::LabA),
                    },
                color: Color::Oklab,
            } => {
                let parameter = shaders::stage::XyzParameter {
                    transfer: match parts {
                        SampleParts::LChA => shaders::stage::Transfer::Oklab,
                        SampleParts::LabA => shaders::stage::Transfer::Rgb(Transfer::Linear),
                        _ => return Err(LaunchError::InternalCommandError(line!())),
                    },
                    parts: SampleParts::LChA,
                    bits,
                };

                // FIXME: duplicate code.
                let result = parameter.linear_format();
                let stage_kind = parameter
                    .stage_kind()
                    // Unsupported format.
                    .ok_or_else(|| LaunchError::InternalCommandError(line!()))?;

                staging = Some(StagingDescriptor {
                    stage_kind,
                    parameter,
                });

                result
            }
            _ => return Err(LaunchError::InternalCommandError(line!())),
        };

        Ok(ImageDescriptor {
            format,
            staging,
            size,
        })
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
        let ImageBufferAssignment {
            buffer: reg_buffer,
            texture: reg_texture,
        } = self.buffer_plan.get(idx)?;

        if self.register_map.get(&idx).is_some() {
            return Ok(());
        }

        let staged = self.make_texture_descriptor(reg_texture)?;
        let texture_format = staged.to_texture();
        let staging_format = staged.to_staging_texture();
        let descriptor = &self.buffer_plan.texture[reg_texture.0];

        let bytes_per_row = (descriptor.layout.bytes_per_texel as u32)
            .checked_mul(texture_format.size.0.get())
            .ok_or_else(|| LaunchError::InternalCommandError(line!()))?;
        let bytes_per_row = (bytes_per_row / 256 + u32::from(bytes_per_row % 256 != 0))
            .checked_mul(256)
            .ok_or_else(|| LaunchError::InternalCommandError(line!()))?;

        let buffer_layout = BufferLayout {
            bytes_per_texel: descriptor.layout.bytes_per_texel,
            width: texture_format.size.0.get(),
            height: texture_format.size.1.get(),
            bytes_per_row,
        };

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

            // eprintln!("Buffer {:?} {:?}", buffer, buffer_layout.u64_len());
            // eprintln!("Buffer {:?} {:?}", buffer + 1, buffer_layout.u64_len());
            // eprintln!("Buffer {:?} {:?}", buffer + 2, buffer_layout.u64_len());

            (
                DeviceBuffer(buffer),
                DeviceBuffer(buffer + 1),
                DeviceBuffer(buffer + 2),
            )
        };

        let texture = self.ensure_allocate_texture(reg_texture)?;
        let staging = self
            .staging_map
            .get(&reg_texture)
            .map(|staging| staging.device);

        let map_entry = RegisterMap {
            logical_texture: reg_texture,
            buffer,
            texture,
            map_read: Some(map_read),
            map_write: Some(map_write),
            staging,
            buffer_layout,
            texture_format,
            staging_format,
        };

        // TODO do a match instead?
        let in_map = self
            .register_map
            .entry(idx)
            .or_insert_with(|| map_entry.clone());
        *in_map = map_entry.clone();

        self.buffer_map.insert(
            reg_buffer,
            BufferMap {
                device: buffer,
                layout: in_map.buffer_layout.clone(),
            },
        );

        Ok(())
    }

    pub(crate) fn ensure_allocate_texture(
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
            // eprintln!("Texture {:?} {:?}", texture, &texture_format);
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
                // eprintln!("Storage Texture {:?} {:?}", texture, &staging);
                self.push(Low::Texture(staging.clone()))?;
                DeviceTexture(texture)
            };

            let fallback = {
                let texture = self.textures;
                // eprintln!("Fallback Texture {:?} {:?}", texture, &staging);
                self.push(Low::Texture(TextureDescriptor {
                    usage: TextureUsage::Transient,
                    size: staging.size,
                    ..texture_format
                }))?;
                DeviceTexture(texture)
            };

            // eprintln!("{} {:?}", reg_texture.0, staging);
            self.staging_map.insert(
                reg_texture,
                StagingTexture {
                    device,
                    format: staging,
                    stage_kind: st_parameter.stage_kind,
                    parameter: st_parameter.parameter,
                    temporary_attachment_buffer_for_encoding_remove_if_possible: fallback,
                },
            );
        }

        Ok(texture)
    }

    fn ingest_image_data(&mut self, idx: Register) -> Result<Texture, LaunchError> {
        let texture = self.buffer_plan.get(idx)?.texture;

        // FIXME: We are conflating `texture` and the index in the execution's vector of IO
        // buffers. That is, we have an entry there even when the register/texture in question has
        // no input or output behavior.
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
        self.input_map.insert(idx, source_image);

        let descriptor = &self.buffer_plan.texture[regmap.texture.0];
        let size = descriptor.size();

        // See below, required for direct buffer-to-buffer copy.
        let sizeu64 = regmap.buffer_layout.u64_len();

        // FIXME: if it is a simple copy we can use regmap.buffer directly.
        let target_buffer = regmap.map_write.unwrap_or(regmap.buffer);

        self.push(Low::WriteImageToBuffer {
            source_image,
            size,
            offset: (0, 0),
            target_buffer,
            target_layout: regmap.buffer_layout,
        })?;

        // FIXME: we're using wgpu internal's scheduling for writing the data to the gpu buffer but
        // this is a separate allocation. We'd instead like to use `regmap.map_write` and do our
        // own buffer mapping but this requires async scheduling. Soo.. do that later.
        // FIXME: might happen at next call within another command encoder..
        if let Some(map_write) = regmap.map_write {
            self.push(Low::BeginCommands)?;
            self.push(Low::CopyBufferToBuffer {
                source_buffer: map_write,
                size: sizeu64,
                target_buffer: regmap.buffer,
            })?;
            self.push(Low::EndCommands)?;
            // TODO: maybe also don't run it immediately?
            self.push(Low::RunTopCommand)?;
        }

        Ok(())
    }

    /// Copy from memory visible buffer to the texture.
    pub(crate) fn copy_buffer_to_staging(&mut self, idx: Register) -> Result<(), LaunchError> {
        let regmap = self.allocate_register(idx)?.clone();

        let (size, target_texture);
        if let Some(staging) = regmap.staging {
            target_texture = staging;
            let (width, height) = regmap.staging_format.as_ref().unwrap().size;
            size = (width.get(), height.get());
        } else {
            // .… or directly to the target buffer if we have no staging.
            target_texture = regmap.texture;
            size = self.buffer_plan.get_info(idx)?.descriptor.size();
        };

        self.push(Low::BeginCommands)?;
        self.push(Low::CopyBufferToTexture {
            source_buffer: regmap.buffer,
            source_layout: regmap.buffer_layout,
            offset: (0, 0),
            size,
            target_texture,
        })?;

        self.push(Low::EndCommands)?;
        // TODO: maybe also don't run it immediately?
        self.push(Low::RunTopCommand)?;

        Ok(())
    }

    /// Copy quantized data to the internal buffer.
    /// Note that this may be a no-op for buffers that need no staging buffer, i.e. where
    /// quantization happens as part of the pipeline.
    pub(crate) fn copy_staging_to_texture(&mut self, idx: Texture) -> Result<(), LaunchError> {
        if let Some(staging) = self.staging_map.get(&idx) {
            // eprintln!("{} {:?}", idx.0, staging);
            // Try to use the cached version of this pipeline.
            let pipeline = if let Some(&pipeline) = self.staged_to_pipelines.get(&idx) {
                pipeline
            } else {
                let fn_ = Function::ToLinearOpto {
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
                    store: true,
                },
            };

            self.push(Low::BeginCommands)?;
            self.push(Low::BeginRenderPass(RenderPassDescriptor {
                color_attachments: vec![attachment],
                depth_stencil: None,
                textures: vec![idx],
            }))?;
            self.render(pipeline)?;
            self.push(Low::EndRenderPass)?;
            self.push(Low::EndCommands)?;

            self.push(Low::RunTopCommand)?;
        }

        Ok(())
    }

    /// Quantize the texture to the staging buffer.
    /// May be a no-op, see reverse operation.
    pub(crate) fn copy_texture_to_staging(&mut self, idx: Texture) -> Result<(), LaunchError> {
        if let Some(staging) = self.staging_map.get(&idx) {
            let texture = staging.temporary_attachment_buffer_for_encoding_remove_if_possible;

            // eprintln!("{} {:?}", idx.0, staging);
            // Try to use the cached version of this pipeline.
            let pipeline = if let Some(&pipeline) = self.staged_from_pipelines.get(&idx) {
                pipeline
            } else {
                let fn_ = Function::FromLinearOpto {
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
                    store: false,
                },
            };

            self.push(Low::BeginCommands)?;
            self.push(Low::BeginRenderPass(RenderPassDescriptor {
                color_attachments: vec![attachment],
                depth_stencil: None,
                textures: vec![idx],
            }))?;
            self.render(pipeline)?;
            self.push(Low::EndRenderPass)?;
            self.push(Low::EndCommands)?;

            self.push(Low::RunTopCommand)?;
        }

        Ok(())
    }

    /// Copy from texture to the memory buffer.
    pub(crate) fn copy_staging_to_buffer(&mut self, idx: Register) -> Result<(), LaunchError> {
        let regmap = self.allocate_register(idx)?.clone();

        let (size, source_texture);
        if let Some(staging) = regmap.staging {
            source_texture = staging;
            let (width, height) = regmap.staging_format.as_ref().unwrap().size;
            size = (width.get(), height.get());
        } else {
            source_texture = regmap.texture;
            size = self.buffer_plan.get_info(idx)?.descriptor.size();
        };

        self.push(Low::BeginCommands)?;
        self.push(Low::CopyTextureToBuffer {
            source_texture,
            offset: (0, 0),
            size,
            target_buffer: regmap.buffer,
            target_layout: regmap.buffer_layout,
        })?;
        self.push(Low::EndCommands)?;
        // TODO: maybe also don't run it immediately?
        self.push(Low::RunTopCommand)?;

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

        let size = self.buffer_plan.get_info(idx)?.descriptor.size();
        let sizeu64 = regmap.buffer_layout.u64_len();

        let source_buffer = regmap.map_read.unwrap_or(regmap.buffer);

        // FIXME: might happen at next call within another command encoder..
        if let Some(map_read) = regmap.map_read {
            self.push(Low::BeginCommands)?;
            self.push(Low::CopyBufferToBuffer {
                source_buffer: regmap.buffer,
                size: sizeu64,
                target_buffer: map_read,
            })?;
            self.push(Low::EndCommands)?;
            self.push(Low::RunTopCommand)?;
        }

        self.push(Low::ReadBuffer {
            source_buffer,
            source_layout: regmap.buffer_layout,
            size,
            offset: (0, 0),
            target_image,
        })
    }

    /// FIXME: we might want to make this a detail of encoder.
    /// Since this would make it easier to change the details of `Low::TextureView`—adding
    /// additional parameter such format conversion, aspect, mip mapping, w/e.
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

    fn make_quad_bind_group(&mut self) -> BindGroupLayoutIdx {
        let bind_group_layouts = &mut self.bind_group_layouts;
        let instructions = &mut self.instructions;
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

            instructions.extend_one(Low::BindGroupLayout(descriptor));
            let descriptor_id = *bind_group_layouts;
            *bind_group_layouts += 1;
            BindGroupLayoutIdx(descriptor_id)
        })
    }

    fn make_generic_fragment_bind_group(&mut self) -> BindGroupLayoutIdx {
        let bind_group_layouts = &mut self.bind_group_layouts;
        let instructions = &mut self.instructions;
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

            instructions.extend_one(Low::BindGroupLayout(descriptor));
            let descriptor_id = *bind_group_layouts;
            *bind_group_layouts += 1;
            BindGroupLayoutIdx(descriptor_id)
        })
    }

    fn make_paint_group_layout(&mut self, count: usize) -> BindGroupLayoutIdx {
        let bind_group_layouts = &mut self.bind_group_layouts;
        let instructions = &mut self.instructions;
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
            instructions.extend_one(Low::BindGroupLayout(descriptor));

            let descriptor_id = *bind_group_layouts;
            *bind_group_layouts += 1;
            BindGroupLayoutIdx(descriptor_id)
        })
    }

    fn make_stage_group(&mut self, binding: u32) -> BindGroupLayoutIdx {
        use shaders::stage::StageKind;
        let bind_group_layouts = &mut self.bind_group_layouts;
        let instructions = &mut self.instructions;

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

            for (num, kind) in StageKind::ALL.iter().enumerate() {
                let i = num as u32 + 16;
                if i != binding {
                    continue;
                }

                entries.push(wgpu::BindGroupLayoutEntry {
                    binding: i,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: kind.texture_format(),
                        view_dimension: wgpu::TextureViewDimension::D2,
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
            instructions.extend_one(Low::BindGroupLayout(descriptor));
            let descriptor_id = *bind_group_layouts;
            *bind_group_layouts += 1;
            BindGroupLayoutIdx(descriptor_id)
        })
    }

    fn make_paint_layout(&mut self, desc: &SimpleRenderPipelineDescriptor) -> PipelineLayoutIdx {
        let quad_bind_group = self.make_quad_bind_group();

        let mut bind_group_layouts = vec![quad_bind_group.0];

        match desc.fragment_texture {
            TextureBind::Textures(0) => {}
            TextureBind::Textures(count) => {
                bind_group_layouts.push(self.make_paint_group_layout(count).0)
            }
            TextureBind::PreComputedGroup { layout, .. } => {
                bind_group_layouts.push(layout);
            }
        };

        if let BufferBind::None = desc.fragment_bind_data {
            let layouts = &mut self.pipeline_layouts;
            let instructions = &mut self.instructions;

            *self.paint_pipeline_layout.get_or_insert_with(|| {
                let descriptor = PipelineLayoutDescriptor {
                    bind_group_layouts,
                    push_constant_ranges: &[],
                };

                instructions.extend_one(Low::PipelineLayout(descriptor));
                let descriptor_id = *layouts;
                *layouts += 1;
                PipelineLayoutIdx(descriptor_id)
            })
        } else {
            bind_group_layouts.push(self.make_generic_fragment_bind_group().0);

            let layouts = &mut self.pipeline_layouts;
            let instructions = &mut self.instructions;

            let descriptor = PipelineLayoutDescriptor {
                bind_group_layouts,
                push_constant_ranges: &[],
            };

            instructions.extend_one(Low::PipelineLayout(descriptor));
            let descriptor_id = *layouts;
            *layouts += 1;
            PipelineLayoutIdx(descriptor_id)
        }
    }

    fn shader(&mut self, desc: ShaderDescriptor) -> Result<ShaderIdx, LaunchError> {
        self.instructions.extend_one(Low::Shader(desc));
        let idx = self.shaders;
        self.shaders += 1;
        Ok(ShaderIdx(idx))
    }

    fn fragment_shader(
        &mut self,
        kind: Option<shaders::FragmentShaderKey>,
        source: Cow<'static, [u32]>,
    ) -> Result<ShaderIdx, LaunchError> {
        if let Some(&shader) = kind.and_then(|k| self.fragment_shaders.get(&k)) {
            return Ok(shader);
        }

        self.shader(ShaderDescriptor {
            name: "",
            source_spirv: source,
        })
    }

    fn vertex_shader(
        &mut self,
        kind: Option<VertexShader>,
        source: Cow<'static, [u32]>,
    ) -> Result<ShaderIdx, LaunchError> {
        if let Some(&shader) = kind.and_then(|k| self.vertex_shaders.get(&k)) {
            return Ok(shader);
        }

        self.shader(ShaderDescriptor {
            name: "",
            source_spirv: source,
        })
    }

    #[rustfmt::skip]
    fn simple_quad_buffer(&mut self) -> DeviceBuffer {
        let buffers = &mut self.buffers;
        let instructions = &mut self.instructions;
        *self.simple_quad_buffer.get_or_insert_with(|| {
            // Sole quad!
            let content: &'static [f32; 8] = &[
                1.0, 1.0,
                1.0, 0.0,
                0.0, 1.0,
                0.0, 0.0,
            ];

            let descriptor = BufferDescriptorInit {
                usage: BufferUsage::InVertices,
                content: bytemuck::cast_slice(content).to_vec().into(),
            };

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
        let PipelineLayoutIdx(layout) = self.make_paint_layout(desc);
        let format = match desc.pipeline_target {
            PipelineTarget::Texture(texture) => {
                let format = self.texture_map[&texture].format.format;
                // eprintln!("Target texture {:?} with format {:?}", texture, format);
                format
            }
            PipelineTarget::PreComputedGroup { target_format } => {
                // eprintln!("Target attachment with format {:?}", target_format);
                target_format
            }
        };

        let (vertex, vertex_entry_point) = match desc.vertex {
            ShaderBind::ShaderMain(shader) => (shader, "main"),
            ShaderBind::Shader { id, entry_point } => (id, entry_point),
        };
        let (fragment, fragment_entry_point) = match desc.fragment {
            ShaderBind::ShaderMain(shader) => (shader, "main"),
            ShaderBind::Shader { id, entry_point } => (id, entry_point),
        };

        self.instructions
            .extend_one(Low::RenderPipeline(RenderPipelineDescriptor {
                vertex: VertexState {
                    entry_point: vertex_entry_point,
                    vertex_module: vertex,
                },
                fragment: FragmentState {
                    entry_point: fragment_entry_point,
                    fragment_module: fragment,
                    targets: vec![wgpu::ColorTargetState {
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                        format,
                    }],
                },
                primitive: PrimitiveState::TriangleStrip,
                layout,
            }));

        let pipeline = self.render_pipelines;
        self.render_pipelines += 1;
        Ok(pipeline)
    }

    fn make_sampler(&mut self, descriptor: SamplerDescriptor) -> SamplerIdx {
        let instructions = &mut self.instructions;
        let sampler = &mut self.sampler;
        *self
            .known_samplers
            .entry(descriptor)
            .or_insert_with_key(|desc| {
                let sampler_id = *sampler;
                instructions.extend_one(Low::Sampler(SamplerDescriptor {
                    address_mode: desc.address_mode,
                    border_color: desc.border_color,
                    resize_filter: desc.resize_filter,
                }));
                *sampler += 1;
                SamplerIdx(sampler_id)
            })
    }

    /// FIXME: really, we need to allocate here?
    /// What if we made `consumed` a dyn-Iterator.
    fn make_bind_group_consumes(&mut self, count: usize) -> Result<Vec<Consume>, LaunchError> {
        let start_of_operands = match self.operands.len().checked_sub(count) {
            None => return Err(LaunchError::InternalCommandError(line!())),
            Some(i) => i,
        };

        self.operands[start_of_operands..]
            .iter()
            .map(|texture| {
                let texture = self
                    .texture_map
                    .get(&texture)
                    // The texture was never allocated. Has it been initialized?
                    .ok_or_else(|| LaunchError::InternalCommandError(line!()))?
                    .device;

                Ok(Consume::Texture(texture))
            })
            .collect()
    }

    fn make_bind_group_sampled_texture(
        &mut self,
        count: usize,
    ) -> Result<BindGroupIdx, LaunchError> {
        let start_of_operands = match self.operands.len().checked_sub(count) {
            None => return Err(LaunchError::InternalCommandError(line!())),
            Some(i) => i,
        };

        let SamplerIdx(sampler) = self.make_sampler(SamplerDescriptor {
            address_mode: wgpu::AddressMode::default(),
            border_color: Some(wgpu::SamplerBorderColor::TransparentBlack),
            resize_filter: wgpu::FilterMode::Nearest,
        });

        let mut entries = vec![BindingResource::Sampler(sampler)];
        let textures: Vec<_> = self.operands.drain(start_of_operands..).collect();
        for texture in textures {
            let view = self.texture_view(texture)?;
            entries.push(BindingResource::TextureView(view));
        }

        let group = self.bind_groups;
        let BindGroupLayoutIdx(layout_idx) = self.make_paint_group_layout(count);
        let descriptor = BindGroupDescriptor {
            layout_idx,
            entries,
            sparse: vec![],
        };

        self.push(Low::BindGroup(descriptor))?;
        Ok(BindGroupIdx(group))
    }

    fn make_opto_fragment_group(
        &mut self,
        binding: u32,
        // The staging texture, whose staging texture is bound to IO-storage image.
        texture: DeviceTexture,
        // The non-staging texture which we bind to the sampler.
        view: Option<Texture>,
    ) -> Result<BindGroupIdx, LaunchError> {
        // FIXME: could be cached.
        let image_id = self.texture_views;
        self.push(Low::TextureView(TextureViewDescriptor { texture }))?;

        let mut sparse;

        // For encoding we have two extra bindings, sampler and in_texture.
        if let Some(view) = view {
            sparse = vec![(binding, BindingResource::TextureView(image_id))];

            // FIXME: unnecessary duplication.
            let SamplerIdx(sampler) = self.make_sampler(SamplerDescriptor {
                address_mode: wgpu::AddressMode::default(),
                border_color: Some(wgpu::SamplerBorderColor::TransparentBlack),
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
            sparse = vec![(binding, BindingResource::TextureView(image_id))];

            let SamplerIdx(sampler) = self.make_sampler(SamplerDescriptor {
                address_mode: wgpu::AddressMode::default(),
                border_color: Some(wgpu::SamplerBorderColor::TransparentBlack),
                resize_filter: wgpu::FilterMode::Nearest,
            });

            sparse.push((34, BindingResource::Sampler(sampler)));
        }

        let group = self.bind_groups;
        let BindGroupLayoutIdx(layout_idx) = self.make_stage_group(binding);
        let descriptor = BindGroupDescriptor {
            layout_idx,
            entries: vec![],
            sparse,
        };

        self.push(Low::BindGroup(descriptor))?;
        Ok(BindGroupIdx(group))
    }

    fn make_bound_buffer(
        &mut self,
        bind: BufferBind<'_>,
        layout_idx: usize,
    ) -> Result<Option<BindGroupIdx>, LaunchError> {
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
        Ok(Some(BindGroupIdx(group)))
    }

    /// Render the pipeline, after all customization and buffers were bound..
    fn prepare_simple_pipeline(
        &mut self,
        descriptor: SimpleRenderPipelineDescriptor,
    ) -> Result<SimpleRenderPipeline, LaunchError> {
        let pipeline = self.simple_render_pipeline(&descriptor)?;
        let buffer = self.simple_quad_buffer();

        let side_effects;
        let group = match &descriptor.fragment_texture {
            TextureBind::Textures(0) => {
                side_effects = self.register_logical_state.register_pipeline(&[], &[]);
                None
            }
            &TextureBind::Textures(count) => {
                let consume = self.make_bind_group_consumes(count)?;
                side_effects = self.register_logical_state.register_pipeline(&consume, &[]);

                let group = self.make_bind_group_sampled_texture(count)?;
                // eprintln!("Using Texture {:?} as group {:?}", texture, group);
                Some(group)
            }
            &TextureBind::PreComputedGroup {
                group,
                consumed: group_consume,
                clobber: group_clobber,
                ..
            } => {
                side_effects = self
                    .register_logical_state
                    .register_pipeline(group_consume, group_clobber);

                // eprintln!("Using Target Group {:?}", group);
                Some(group)
            }
        };

        let BindGroupLayoutIdx(vertex_layout) = self.make_quad_bind_group();
        let vertex_bind = self.make_bound_buffer(descriptor.vertex_bind_data, vertex_layout)?;

        // FIXME: this builds the layout even when it is not required.
        let BindGroupLayoutIdx(vertex_layout) = self.make_generic_fragment_bind_group();
        let fragment_bind = self.make_bound_buffer(descriptor.fragment_bind_data, vertex_layout)?;

        Ok(SimpleRenderPipeline {
            pipeline,
            buffer,
            group,
            vertex_bind,
            vertices: 4,
            fragment_bind,
            side_effects,
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
            side_effects,
        } = pipeline;

        self.push(Low::SetPipeline(pipeline))?;

        let mut group_idx = 0;
        if let Some(BindGroupIdx(quad)) = vertex_bind {
            self.push(Low::SetBindGroup {
                group: quad,
                index: group_idx,
                offsets: Cow::Borrowed(&[]),
            })?;
            group_idx += 1;
        }

        if let Some(BindGroupIdx(group)) = group {
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

        if let Some(BindGroupIdx(bind)) = fragment_bind {
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
        function: &Function,
        // The texture we are rendering to.
        target: Texture,
    ) -> Result<SimpleRenderPipeline, LaunchError> {
        match function {
            Function::PaintToSelection { texture, selection, target: target_coords, viewport, shader } => {
                let (tex_width, tex_height) = self.texture_map[texture].format.size;

                let ShaderIdx(vertex) = self.vertex_shader(
                    Some(VertexShader::Noop),
                    shader_include_to_spirv(shaders::VERT_NOOP))?;

                let shader = shader.shader();
                let key = shader.key();
                let spirv = shader.spirv_source();

                let ShaderIdx(fragment) = self.fragment_shader(key, shader_include_to_spirv_static(spirv))?;

                let buffer: [[f32; 2]; 8];
                // FIXME: there seems to be two floats padding after each vec2.
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
                    vertex: ShaderBind::ShaderMain(vertex),
                    fragment: ShaderBind::ShaderMain(fragment),
                })
            },
            Function::PaintFullScreen { shader } => {
                let ShaderIdx(vertex) = self.vertex_shader(
                    Some(VertexShader::Noop),
                    shader_include_to_spirv(shaders::VERT_NOOP))?;

                let shader = shader.shader();
                let key = shader.key();
                let spirv = shader.spirv_source();

                let ShaderIdx(fragment) = self.fragment_shader(key, shader_include_to_spirv_static(spirv))?;
                let fragment_bind_data = shader.binary_data(&mut self.binary_data)
                    .map(|data| BufferBind::Planned { data })
                    .unwrap_or(BufferBind::None);

                let arguments = shader.num_args();

                self.prepare_simple_pipeline(SimpleRenderPipelineDescriptor{
                    pipeline_target: PipelineTarget::Texture(target),
                    vertex_bind_data: BufferBind::Set {
                        data: bytemuck::cast_slice(&Self::FULL_VERTEX_BUFFER[..]),
                    },
                    fragment_texture: TextureBind::Textures(arguments as usize),
                    fragment_bind_data,
                    vertex: ShaderBind::ShaderMain(vertex),
                    fragment: ShaderBind::ShaderMain(fragment),
                })
            },
            Function::ToLinearOpto { parameter, stage_kind } => {
                let ShaderIdx(vertex) = self.vertex_shader(
                    Some(VertexShader::Noop),
                    shader_include_to_spirv(shaders::VERT_NOOP))?;

                let ShaderIdx(fragment) = self.fragment_shader(
                    Some(shaders::FragmentShaderKey::Convert),
                    shader_include_to_spirv(stage_kind.decode_src()))?;

                let buffer = parameter.serialize_std140();
                // FIXME: see below, shaderc requires renamed entry points to "main".
                let _entry_point = stage_kind.encode_entry_point();

                let BindGroupLayoutIdx(layout) = self.make_stage_group(stage_kind.decode_binding());

                let texture = self
                    .staging_map
                    .get(&target)
                    .ok_or_else(|| LaunchError::InternalCommandError(line!()))?
                    .device;

                let group = self.make_opto_fragment_group(
                    stage_kind.decode_binding(),
                    texture,
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
                        consumed: &[
                            Consume::Texture(texture),
                        ],
                        // Does not have side-effects on any buffer or texture.
                        clobber: &[],
                    },
                    fragment_bind_data: BufferBind::Set {
                        data: bytemuck::cast_slice(&buffer[..]),
                    },
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
            Function::FromLinearOpto { parameter, stage_kind } => {
                let ShaderIdx(vertex) = self.vertex_shader(
                    Some(VertexShader::Noop),
                    shader_include_to_spirv(shaders::VERT_NOOP))?;

                let ShaderIdx(fragment) = self.fragment_shader(
                    Some(shaders::FragmentShaderKey::Convert),
                    shader_include_to_spirv(stage_kind.encode_src()))?;

                let buffer = parameter.serialize_std140();
                // FIXME: see below, shaderc requires renamed entry points to "main".
                let _entry_point = stage_kind.decode_entry_point();

                let BindGroupLayoutIdx(layout) = self.make_stage_group(stage_kind.encode_binding());

                let target_texture = self
                    .staging_map
                    .get(&target)
                    .ok_or_else(|| LaunchError::InternalCommandError(line!()))?
                    .device;

                let source_texture = self
                    .texture_map
                    .get(&target)
                    .ok_or_else(|| LaunchError::InternalCommandError(line!()))?
                    .device;

                let group = self.make_opto_fragment_group(
                    stage_kind.encode_binding(),
                    target_texture,
                    Some(target),
                )?;

                // eprintln!("{:?} {:?}", parameter, buffer);

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
                        consumed: &[Consume::Texture(source_texture)],
                        clobber: &[Clobber::Texture {
                            texture: target,
                            device: target_texture,
                        }],
                    },
                    fragment_bind_data: BufferBind::Set {
                        data: bytemuck::cast_slice(&buffer[..]),
                    },
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

    /// Ingest the data into the encoder's active buffer data.
    fn ingest_data(&mut self, data: &[impl bytemuck::Pod]) -> BufferInitContent {
        BufferInitContent::new(&mut self.binary_data, data)
    }

    pub(crate) fn io_map(&self) -> run::IoMap {
        let mut io_map = run::IoMap::default();

        for (&register, texture) in &self.input_map {
            io_map.inputs.insert(register, texture.0);
        }

        for (&register, texture) in &self.output_map {
            io_map.outputs.insert(register, texture.0);
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
        if !self.operands.is_empty() {
            // eprintln!("{:?}", self.operands.as_slice());
            return Err(LaunchError::InternalCommandError(line!()));
        }

        Ok(())
    }
}

impl RegisterLogicalState {
    /// Register a set of side effects on resources.
    fn register_pipeline(
        &mut self,
        consume: &[Consume],
        clobber: &[Clobber],
    ) -> LogicalSideEffectIdx {
        let consume_start = self.consumes.len();
        self.consumes.extend_from_slice(consume);
        let clobber_start = self.clobbers.len();
        self.clobbers.extend_from_slice(clobber);

        let pipeline = self.side_effects.len();
        self.side_effects.push(DelayedSideEffects {
            consume: consume_start..self.consumes.len(),
            clobber: clobber_start..self.clobbers.len(),
        });

        LogicalSideEffectIdx(pipeline)
    }

    /// Schedule execution of a specific set of side effects.
    ///
    /// Consumes the consumes, puts all clobbered resources into a new fresh state etc.
    /// The pipeline isn't executed immediately, we're made aware of the stack though.
    fn schedule_pipeline(&mut self, LogicalSideEffectIdx(effect): LogicalSideEffectIdx)
        -> Result<ExecuteId, LaunchError>
    {
        let DelayedSideEffects { consume, clobber } = self.side_effects.get(effect)
            .ok_or_else(|| LaunchError::InternalCommandError(line!()))?;

        // Bind to all consumes currently open.
        for consume in &self.consumes[consume.clone()] {
            if let Some(exec) = self.clobber_map.get(consume) {
                todo!()
            } else {
                // Consume without preceding effect that wrote it.
                // 'Uninitialized' use of the resource.
                return Err(LaunchError::InternalCommandError(line!()));
            }
        }

        // Create symmetric other end of clobber edges.
        // Note: 'atomic' with regards to the effect on failure.
        let mut consume_side = vec![];
        for clobber in &self.clobbers[clobber.clone()] {
            consume_side.push(match clobber {
                &Clobber::Buffer { texture, device } => {
                    todo!("{:?}", texture);
                    Consume::Buffer(device)
                }
                &Clobber::Texture { texture, device } => {
                    todo!("{:?}", texture);
                    Consume::Texture(device)
                }
            });
        }

        let id = ExecuteId(self.scheduled.len());
        self.scheduled.push(LogicalSideEffectIdx(effect));

        // Mark all clobbers to be available for consumption.
        consume_side
            .into_iter()
            .for_each(|consume| {
                self.clobber_map.insert(consume, id);
            });

        Ok(id)
    }

    /// Check if a copy operation would be a logical no-op.
    ///
    /// This takes as input the target of the copy, the source is equivalently given by any fresh
    /// representation of the logical resource. Will return an `Err` if there is no such fresh
    /// representation.
    ///
    /// Images are held in multiple representations: buffers on the host, buffers on the gpu,
    /// textures on the gpu, a linear color texture on the gpu.
    ///
    /// To consume one specific version, we must perform a copy from a fresh representation (one
    /// that was written to last). After such a copy we have more than one fresh representation.
    /// The encoder may happen to receive instructions to perform a copy from one fresh
    /// representation to another. These instructions can be elided completely.
    ///
    /// This is in general a hard problem if we also consider copies from _different_ logical
    /// resources as there is no uniquely minimal solution. As a consequence we don't track this
    /// yet and just perform a greedy algorithm: The first initialization wins, other copies get
    /// elided.
    ///
    /// Example for unclear minimization. Either 3. or 4. can be elided together with their
    /// original copy. Which is better may depend on their original workload, latency introduced by
    /// blocking other executions, …
    ///
    /// ```
    /// 1. Buffer_A --> Buffer_B
    /// 2. Texture_A --> Texture_B
    /// 3. Buffer_B --> Texture_B
    /// 4. Texture_B --> Buffer_B
    /// ```
    fn is_copy_fresh(&self, clobber: &[Clobber]) -> Result<bool, LaunchError> {
        todo!()
    }

    /// Schedule execution of commands necessary to prepare the consume items provided.
    fn resolve(&mut self, consumes: &[Consume]) -> Result<&mut Vec<Low>, LaunchError> {
        // All dependencies are a DAG based on consume/clobber pairs as (hyper-) edges.
        // Resolve all nodes that are reachable from the given inputs.
        // For simplicity, we will schedule all pending commands even if not necessary for this
        // resolution. This will keep them in order and reduces our problem to simply finding the
        // maximum ID of all executions that feed these consumes.
        //
        // TODO: prepare a dependency graph of executions for a later optimization pass. The rough
        // idea would be:
        // - Bunch commands together, with some limit on the amount of work in one such group.
        // - Reorder executions such that these bunches are as big as possible under the
        //   constraints of the dependency graph order.
        let mut max_id = None;
        for consume in consumes {
            // No clobber writing this consume? That means it is uninitialized.
            let &ExecuteId(id) = self
                .clobber_map
                .get(consume)
                .ok_or_else(|| LaunchError::InternalCommandError(line!()))?;

            max_id = max_id.max(Some(id));
        }

        self.eager_commands.clear();

        // There may not be anything to do.
        if let Some(max_id) = max_id {
            let ExecuteId(past_id) = self.past_executed_id;
            // Schedule all unscheduled commands.
            let range = past_id.min(max_id)..max_id;
            self.eager_commands.push(Low::RunBotToTop(range.len()));
            self.past_executed_id = ExecuteId(past_id.max(max_id));
        }

        Ok(&mut self.eager_commands)
    }
}

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
