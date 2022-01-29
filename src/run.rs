use core::{iter::once, num::NonZeroU32, pin::Pin};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::buffer::{BufferLayout, Descriptor, Texel};
use crate::command::Register;
use crate::pool::{ImageData, Pool, PoolImage, PoolKey};
use crate::program::{self, Capabilities, DeviceBuffer, DeviceTexture, Low};

use wgpu::{Device, Queue};

/// The code of a linear pipeline, lowered to a particular set of device capabilities.
///
/// The struct is created by calling [`lower_to`] on [`Program`].
///
/// [`lower_to`]: crate::program::Program::lower_to()
/// [`Program`]: crate::program::Program
///
/// This contains the program and a partial part of the inputs and the device to execute on. There
/// are two main ways to instantiate this struct: By lowering a `Program` or by softly retiring an
/// existing, running `Execution`. In the later case we might be able to salvage parts of the
/// execution such that it may be re-used in a later run on the same device.
pub struct Executable {
    /// The list of instructions to perform.
    pub(crate) instructions: Arc<[Low]>,
    /// The host-side data which is used to initialize buffers.
    pub(crate) binary_data: Vec<u8>,
    /// All device related state which we have preserved.
    pub(crate) descriptors: Descriptors,
    /// Input/Output buffers used for execution.
    pub(crate) buffers: Vec<Image>,
    /// The map from registers to the index in image data.
    pub(crate) io_map: Arc<IoMap>,
    /// The capabilities required from devices to execute this.
    pub(crate) capabilities: Capabilities,
}

/// Configures devices and input/output buffers for an executable.
///
/// This is created via the [`Executable::from_pool`].
///
/// This is merely a configuration structure. It does not modify the pools passed in until the
/// executable is actually launched. An environment collects references (keys) to the inner buffers
/// until it is time.
///
/// Note that the type system does not stop you from submitting it to a different program but it
/// may be rejected if the inputs and device capabilities differ.
pub struct Environment<'pool> {
    pool: &'pool mut Pool,
    gpu: Gpu,
    buffers: Vec<Image>,
    io_map: Arc<IoMap>,
}

/// A running `Executable`.
pub struct Execution {
    pub(crate) machine: Machine,
    pub(crate) gpu: Gpu,
    pub(crate) descriptors: Descriptors,
    pub(crate) command_encoder: Option<wgpu::CommandEncoder>,
    pub(crate) buffers: Vec<Image>,
    pub(crate) binary_data: Vec<u8>,
    pub(crate) io_map: Arc<IoMap>,
    pub(crate) debug_stack: Vec<Frame>,
}

#[derive(Clone, Debug)]
pub(crate) struct Frame {
    // Used through Debug
    #[allow(dead_code)]
    pub(crate) name: String,
}

pub(crate) struct InitialState {
    pub(crate) instructions: Arc<[Low]>,
    pub(crate) device: Device,
    pub(crate) queue: Queue,
    pub(crate) buffers: Vec<Image>,
    pub(crate) binary_data: Vec<u8>,
    pub(crate) io_map: IoMap,
}

/// An image owned by the execution state but compatible with extracting it.
pub(crate) struct Image {
    /// The binary data and its layout.
    pub(crate) data: ImageData,
    /// The texel interpretation of the data.
    /// This is used mainly for IO where we reconstruct a free-standing image with all its
    /// associated data when moving from an `Image`.
    pub(crate) texel: Texel,
    /// The pool key corresponding to this image.
    pub(crate) key: Option<PoolKey>,
}

#[derive(Default)]
pub struct IoMap {
    /// Map input registers to their index in `buffers`.
    pub(crate) inputs: HashMap<Register, usize>,
    /// Map output registers to their index in `buffers`.
    pub(crate) outputs: HashMap<Register, usize>,
}

#[derive(Default)]
pub(crate) struct Descriptors {
    bind_groups: Vec<wgpu::BindGroup>,
    bind_group_layouts: Vec<wgpu::BindGroupLayout>,
    buffers: Vec<wgpu::Buffer>,
    command_buffers: Vec<wgpu::CommandBuffer>,
    shaders: Vec<wgpu::ShaderModule>,
    pipeline_layouts: Vec<wgpu::PipelineLayout>,
    render_pipelines: Vec<wgpu::RenderPipeline>,
    sampler: Vec<wgpu::Sampler>,
    textures: Vec<wgpu::Texture>,
    texture_views: Vec<wgpu::TextureView>,
}

pub(crate) struct Gpu {
    pub(crate) device: Device,
    pub(crate) queue: Queue,
}

type DynStep = dyn core::future::Future<Output = Result<Cleanup, StepError>>;

struct DevicePolled<'exe> {
    future: Pin<Box<DynStep>>,
    execution: &'exe mut Execution,
}

enum Cleanup {
    Buffers {
        buffers: Vec<wgpu::Buffer>,
        image_data: Vec<Image>,
    },
}

pub struct SyncPoint<'exe> {
    future: Option<DevicePolled<'exe>>,
    marker: core::marker::PhantomData<&'exe mut Execution>,
}

/// Represents a stopped execution instance, without information abouts its outputs.
pub struct Retire<'pool> {
    /// The retiring execution instance.
    execution: Execution,
    pool: &'pool mut Pool,
}

pub(crate) struct Machine {
    instructions: Arc<[Low]>,
    instruction_pointer: usize,
}

#[derive(Debug)]
pub struct StartError {
    kind: LaunchErrorKind,
}

#[derive(Debug)]
pub enum LaunchErrorKind {
    FromLine(u32),
}

#[derive(Debug)]
pub struct StepError {
    inner: StepErrorKind,
    instruction_pointer: usize,
}

#[derive(Debug)]
enum StepErrorKind {
    InvalidInstruction(u32),
    BadInstruction(BadInstruction),
    ProgramEnd,
    RenderPassDidNotEnd,
}

#[derive(Debug)]
pub struct BadInstruction {
    inner: String,
}

#[derive(Debug)]
pub struct RetireError {
    inner: RetireErrorKind,
}

#[derive(Debug)]
pub enum RetireErrorKind {
    NoSuchOutput,
    BadInstruction,
}

impl Image {
    /// Create an image without binary data, promising to set it up later.
    pub(crate) fn with_late_bound(descriptor: &Descriptor) -> Self {
        Image {
            data: ImageData::LateBound(descriptor.layout.clone()),
            texel: descriptor.texel.clone(),
            // Not related to any pooled image.
            key: None,
        }
    }

    pub(crate) fn clone_like(&self) -> Self {
        Image {
            data: ImageData::LateBound(self.data.layout().clone()),
            texel: self.texel.clone(),
            key: self.key,
        }
    }
}

impl Executable {
    pub fn from_pool<'p>(&self, pool: &'p mut Pool) -> Option<Environment<'p>> {
        let (_, gpu) = pool.select_device(&self.capabilities)?;
        Some(Environment {
            pool,
            gpu,
            buffers: self.buffers.iter().map(Image::clone_like).collect(),
            io_map: self.io_map.clone(),
        })
    }

    pub fn launch(&self, mut env: Environment) -> Result<Execution, StartError> {
        self.check_satisfiable(&mut env)?;

        Ok(Execution {
            machine: Machine::new(Arc::clone(&self.instructions)),
            gpu: env.gpu,
            descriptors: Descriptors::default(),
            command_encoder: None,
            buffers: env.buffers,
            binary_data: self.binary_data.clone(),
            io_map: self.io_map.clone(),
            debug_stack: vec![],
        })
    }

    /// Run the executable but take all by value.
    pub fn launch_once(self, mut env: Environment) -> Result<Execution, StartError> {
        self.check_satisfiable(&mut env)?;
        Ok(Execution {
            machine: Machine::new(Arc::clone(&self.instructions)),
            gpu: env.gpu,
            descriptors: self.descriptors,
            command_encoder: None,
            buffers: env.buffers,
            binary_data: self.binary_data,
            io_map: self.io_map.clone(),
            debug_stack: vec![],
        })
    }

    /// Run validation testing if everything is read to launch.
    ///
    /// Note a mad lad could have passed a completely different environment so we, once again,
    /// validate that the buffer descriptors are okay.
    fn check_satisfiable(&self, env: &mut Environment) -> Result<(), StartError> {
        let mut used_keys = HashSet::new();
        for &input in self.io_map.inputs.values() {
            let buffer = env
                .buffers
                .get(input)
                .ok_or_else(|| StartError::InternalCommandError(line!()))?;
            // It's okay to index our own state with our own index.
            let reference = &self.buffers[input];

            if reference.data.layout() != buffer.data.layout() {
                return Err(StartError::InternalCommandError(line!()));
            }

            if reference.texel != buffer.texel {
                // FIXME: not quite such an 'unknown' error.
                return Err(StartError::InternalCommandError(line!()));
            }

            // Oh, this image is always already bound? Cool.
            if !matches!(buffer.data, ImageData::LateBound(_)) {
                continue;
            }

            let key = match buffer.key {
                None => return Err(StartError::InternalCommandError(line!())),
                Some(key) => key,
            };

            // FIXME: we could catch this much earlier.
            if !used_keys.insert(key) {
                return Err(StartError::InternalCommandError(line!()));
            }

            if env.pool.entry(key).is_none() {
                return Err(StartError::InternalCommandError(line!()));
            }
        }

        // Okay, checks done, let's patch up the state.
        for &input in self.io_map.inputs.values() {
            let buffer = &mut env.buffers[input];

            // Unwrap okay, check this earlier.
            let key = buffer.key.unwrap();
            let mut pool_img = env.pool.entry(key).unwrap();

            pool_img.trade(&mut buffer.data);
        }

        for &output in self.io_map.outputs.values() {
            env.buffers[output].data.host_allocate();
        }

        Ok(())
    }
}

impl Environment<'_> {
    pub fn bind(&mut self, reg: Register, key: PoolKey) -> Result<(), StartError> {
        let &idx = self
            .io_map
            .inputs
            .get(&reg)
            .ok_or_else(|| StartError::InternalCommandError(line!()))?;

        let pool_img = self
            .pool
            .entry(key)
            .ok_or_else(|| StartError::InternalCommandError(line!()))?;

        let image = &mut self.buffers[idx];
        let descriptor = pool_img.descriptor();

        if descriptor.layout != *image.data.layout() {
            return Err(StartError::InternalCommandError(line!()));
        }

        if descriptor.texel != image.texel {
            // FIXME: not quite such an 'unknown' error.
            return Err(StartError::InternalCommandError(line!()));
        }

        image.key = Some(pool_img.key());

        Ok(())
    }
}

impl Execution {
    pub(crate) fn new(init: InitialState) -> Self {
        init.device.start_capture();
        Execution {
            machine: Machine::new(init.instructions),
            gpu: Gpu {
                device: init.device,
                queue: init.queue,
            },
            descriptors: Descriptors::default(),
            buffers: init.buffers,
            command_encoder: None,
            binary_data: init.binary_data,
            io_map: Arc::new(init.io_map),
            debug_stack: vec![],
        }
    }

    /// Check if the machine is still running.
    pub fn is_running(&self) -> bool {
        self.machine.instruction_pointer < self.machine.instructions.len()
    }

    pub fn step(&mut self) -> Result<SyncPoint<'_>, StepError> {
        let instruction_pointer = self.machine.instruction_pointer;

        match self.step_inner() {
            Ok(sync) => Ok(sync),
            Err(mut error) => {
                // Add tracing information..
                error.instruction_pointer = instruction_pointer;
                Err(error)
            }
        }
    }

    fn step_inner(&mut self) -> Result<SyncPoint<'_>, StepError> {
        struct DumpFrame<'stack> {
            stack: Option<&'stack mut Vec<Frame>>,
        }

        impl Drop for DumpFrame<'_> {
            fn drop(&mut self) {
                if !std::thread::panicking() {
                    return;
                }

                if let Some(frames) = self.stack.as_ref() {
                    eprintln!("Dump of logical stack:");
                    for (idx, frame) in frames.iter().rev().enumerate() {
                        eprintln!("{:4}: {}", idx, frame.name);
                    }
                }
            }
        }

        let mut _dump_on_panic = DumpFrame {
            stack: Some(&mut self.debug_stack),
        };

        match self.machine.next_instruction()? {
            Low::BindGroupLayout(desc) => {
                let mut entry_buffer = vec![];
                let group = self
                    .descriptors
                    .bind_group_layout(desc, &mut entry_buffer)?;
                // eprintln!("Made {}: {:?}", self.descriptors.bind_group_layouts.len(), group);
                let group = self.gpu.device.create_bind_group_layout(&group);
                self.descriptors.bind_group_layouts.push(group);
                Ok(SyncPoint::NO_SYNC)
            }
            Low::BindGroup(desc) => {
                let mut entry_buffer = vec![];
                let group = self.descriptors.bind_group(desc, &mut entry_buffer)?;
                // eprintln!("{}: {:?}", desc.layout_idx, group);
                let group = self.gpu.device.create_bind_group(&group);
                self.descriptors.bind_groups.push(group);
                Ok(SyncPoint::NO_SYNC)
            }
            Low::Buffer(desc) => {
                let desc = wgpu::BufferDescriptor {
                    label: None,
                    size: desc.size,
                    usage: desc.usage.to_wgpu(),
                    mapped_at_creation: false,
                };

                let buffer = self.gpu.device.create_buffer(&desc);
                self.descriptors.buffers.push(buffer);
                Ok(SyncPoint::NO_SYNC)
            }
            Low::BufferInit(desc) => {
                use wgpu::util::DeviceExt;
                let desc = wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: desc.content.as_slice(&self.binary_data),
                    usage: desc.usage.to_wgpu(),
                };

                let buffer = self.gpu.device.create_buffer_init(&desc);
                self.descriptors.buffers.push(buffer);
                Ok(SyncPoint::NO_SYNC)
            }
            Low::Shader(desc) => {
                let shader;
                if std::env::var("STEALTH_PAINT_PASSTHROUGH").is_err() {
                    let desc = wgpu::ShaderModuleDescriptor {
                        label: Some(desc.name),
                        source: wgpu::ShaderSource::SpirV(desc.source_spirv.as_ref().into()),
                    };

                    shader = self.gpu.device.create_shader_module(&desc);
                } else {
                    let desc = wgpu::ShaderModuleDescriptorSpirV {
                        label: Some(desc.name),
                        source: desc.source_spirv.as_ref().into(),
                    };

                    // SAFETY: who knows. FIXME: once naga's validation is good enough.
                    shader = unsafe { self.gpu.device.create_shader_module_spirv(&desc) };
                };

                self.descriptors.shaders.push(shader);
                Ok(SyncPoint::NO_SYNC)
            }
            Low::PipelineLayout(desc) => {
                let mut entry_buffer = vec![];
                let layout = self.descriptors.pipeline_layout(desc, &mut entry_buffer)?;
                let layout = self.gpu.device.create_pipeline_layout(&layout);
                self.descriptors.pipeline_layouts.push(layout);
                Ok(SyncPoint::NO_SYNC)
            }
            Low::Sampler(desc) => {
                let desc = wgpu::SamplerDescriptor {
                    label: None,
                    address_mode_u: desc.address_mode,
                    address_mode_v: desc.address_mode,
                    address_mode_w: desc.address_mode,
                    mag_filter: desc.resize_filter,
                    min_filter: desc.resize_filter,
                    mipmap_filter: desc.resize_filter,
                    lod_min_clamp: 0.0,
                    lod_max_clamp: 0.0,
                    compare: None,
                    anisotropy_clamp: None,
                    border_color: desc.border_color,
                };
                let sampler = self.gpu.device.create_sampler(&desc);
                self.descriptors.sampler.push(sampler);
                Ok(SyncPoint::NO_SYNC)
            }
            Low::TextureView(desc) => {
                let texture = self
                    .descriptors
                    .textures
                    .get(desc.texture.0)
                    .ok_or_else(|| StepError::InvalidInstruction(line!()))?;
                let desc = wgpu::TextureViewDescriptor {
                    label: None,
                    format: None,
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: 0,
                    mip_level_count: None,
                    base_array_layer: 0,
                    array_layer_count: None,
                };
                let view = texture.create_view(&desc);
                self.descriptors.texture_views.push(view);
                Ok(SyncPoint::NO_SYNC)
            }
            Low::Texture(desc) => {
                use wgpu::TextureUsages as U;
                let desc = wgpu::TextureDescriptor {
                    label: None,
                    size: wgpu::Extent3d {
                        width: desc.size.0.get(),
                        height: desc.size.1.get(),
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: desc.format,
                    usage: match desc.usage {
                        program::TextureUsage::DataIn => U::COPY_DST | U::TEXTURE_BINDING,
                        program::TextureUsage::DataOut => U::COPY_SRC | U::RENDER_ATTACHMENT,
                        program::TextureUsage::Attachment => {
                            U::COPY_SRC | U::COPY_DST | U::TEXTURE_BINDING | U::RENDER_ATTACHMENT
                        }
                        program::TextureUsage::Staging => {
                            U::COPY_SRC | U::COPY_DST | U::STORAGE_BINDING | U::TEXTURE_BINDING
                        }
                        program::TextureUsage::Transient => {
                            U::TEXTURE_BINDING | U::RENDER_ATTACHMENT
                        }
                    },
                };
                let texture = self.gpu.device.create_texture(&desc);
                self.descriptors.textures.push(texture);
                Ok(SyncPoint::NO_SYNC)
            }
            Low::RenderPipeline(desc) => {
                let mut vertex_buffers = vec![];
                let mut fragments = vec![];

                let pipeline =
                    self.descriptors
                        .pipeline(desc, &mut vertex_buffers, &mut fragments)?;
                let pipeline = self.gpu.device.create_render_pipeline(&pipeline);
                self.descriptors.render_pipelines.push(pipeline);
                Ok(SyncPoint::NO_SYNC)
            }
            Low::BeginCommands => {
                if self.command_encoder.is_some() {
                    return Err(StepError::InvalidInstruction(line!()));
                }

                let descriptor = wgpu::CommandEncoderDescriptor { label: None };

                self.command_encoder = Some(self.gpu.device.create_command_encoder(&descriptor));
                Ok(SyncPoint::NO_SYNC)
            }
            Low::BeginRenderPass(descriptor) => {
                let mut attachment_buf = vec![];
                let descriptor = self
                    .descriptors
                    .render_pass(descriptor, &mut attachment_buf)?;
                let encoder = match &mut self.command_encoder {
                    Some(encoder) => encoder,
                    None => return Err(StepError::InvalidInstruction(line!())),
                };

                let pass = encoder.begin_render_pass(&descriptor);
                drop(attachment_buf);
                self.machine.render_pass(&self.descriptors, pass)?;

                Ok(SyncPoint::NO_SYNC)
            }
            Low::EndCommands => match self.command_encoder.take() {
                None => Err(StepError::InvalidInstruction(line!())),
                Some(encoder) => {
                    self.descriptors.command_buffers.push(encoder.finish());
                    Ok(SyncPoint::NO_SYNC)
                }
            },
            &Low::RunTopCommand => {
                let command = self
                    .descriptors
                    .command_buffers
                    .pop()
                    .ok_or_else(|| StepError::InvalidInstruction(line!()))?;
                self.gpu.queue.submit(once(command));
                Ok(SyncPoint::NO_SYNC)
            }
            &Low::RunTopToBot(many) => {
                if many > self.descriptors.command_buffers.len() {
                    return Err(StepError::InvalidInstruction(line!()));
                }

                let commands = self.descriptors.command_buffers.drain(many..);
                self.gpu.queue.submit(commands.rev());
                Ok(SyncPoint::NO_SYNC)
            }
            &Low::RunBotToTop(many) => {
                if many > self.descriptors.command_buffers.len() {
                    return Err(StepError::InvalidInstruction(line!()));
                }

                let commands = self.descriptors.command_buffers.drain(many..);
                self.gpu.queue.submit(commands);
                Ok(SyncPoint::NO_SYNC)
            }
            &Low::WriteImageToBuffer {
                source_image,
                offset,
                size,
                target_buffer,
                ref target_layout,
            } => {
                if offset != (0, 0) {
                    return Err(StepError::InvalidInstruction(line!()));
                }

                let source = match self.buffers.get(source_image.0) {
                    None => return Err(StepError::InvalidInstruction(line!())),
                    Some(source) => &source.data,
                };

                if source.as_bytes().is_none() {
                    return Err(StepError::InvalidInstruction(line!()));
                }

                let layout = source.layout().clone();

                if (target_layout.width, target_layout.height) != (layout.width, layout.height) {
                    return Err(StepError::InvalidInstruction(line!()));
                }

                /* FIXME: we could use this, which would integrate it in the next command encoder,
                 * but this requires the target buffer to be COPY_DST which the read buffer is NOT.
                 * In other words we would have to point to the image buffer directly. This is,
                 * however, not quite desirable as it breaks the layering.
                 * Instead, we could have the `program` module decide to to a direct write or the
                 * encoder could fuse it in its `extend_one` call with limited lookback. Fun
                 * optimization ideas.
                if layout.bytes_per_row == target_layout.bytes_per_row {
                    // Simply case, just write the whole buffer.
                    self.gpu.queue.write_buffer(buffer.buffer, 0, data);
                    return Ok(SyncPoint::NO_SYNC)
                }
                */

                if size != (target_layout.width, target_layout.height) {
                    // Not yet supported (or needed).
                    return Err(StepError::InvalidInstruction(line!()));
                }

                let (width, height) = size;
                let bytes_per_row = target_layout.bytes_per_row;
                let bytes_per_texel = target_layout.bytes_per_texel;
                let bytes_to_copy = (u32::from(bytes_per_texel) * width) as usize;

                // Complex case, we need to instrument our own copy.
                let buffers = core::mem::take(&mut self.descriptors.buffers);
                let mut image_data = core::mem::take(&mut self.buffers);

                let box_me = async move {
                    {
                        let image = &mut image_data[source_image.0].data;
                        let buffer = &buffers[target_buffer.0];

                        let slice = buffer.slice(..);
                        slice.map_async(wgpu::MapMode::Write).await.map_err(
                            |wgpu::BufferAsyncError| StepError::InvalidInstruction(line!()),
                        )?;

                        let mut data = slice.get_mapped_range_mut();

                        // TODO: defensive programming, don't assume cast works.
                        let target_pitch = bytes_per_row as usize;
                        let source_pitch = image.layout().bytes_per_row as usize;

                        // We've checked that this image can be seen as host bytes.
                        let source: &[u8] = image.as_bytes().unwrap();
                        let target: &mut [u8] = &mut data[..];

                        for x in 0..height {
                            let source_row = &source[(x as usize * source_pitch)..][..source_pitch];
                            let target_row =
                                &mut target[(x as usize * target_pitch)..][..target_pitch];

                            target_row[..bytes_to_copy]
                                .copy_from_slice(&source_row[..bytes_to_copy]);
                        }
                    }

                    buffers[target_buffer.0].unmap();

                    Ok(Cleanup::Buffers {
                        buffers,
                        image_data,
                    })
                };

                drop(_dump_on_panic);

                Ok(SyncPoint {
                    future: Some(DevicePolled {
                        future: Box::pin(box_me),
                        execution: self,
                    }),
                    marker: core::marker::PhantomData,
                })
            }
            &Low::CopyBufferToTexture {
                source_buffer,
                ref source_layout,
                offset,
                size,
                target_texture,
            } => {
                if offset != (0, 0) {
                    return Err(StepError::InvalidInstruction(line!()));
                }

                let encoder = match &mut self.command_encoder {
                    Some(encoder) => encoder,
                    None => return Err(StepError::InvalidInstruction(line!())),
                };

                let buffer = self.descriptors.buffer(source_buffer, source_layout)?;
                let texture = self.descriptors.texture(target_texture)?;

                let extent = wgpu::Extent3d {
                    width: size.0,
                    height: size.1,
                    depth_or_array_layers: 1,
                };

                // eprintln!("Source: {:?}", source_buffer);
                // eprintln!("Target: {:?}", target_texture);

                // eprintln!("{:?}", buffer);
                // eprintln!("{:?}", texture);
                // eprintln!("{:?}", extent);

                encoder.copy_buffer_to_texture(buffer, texture, extent);

                Ok(SyncPoint::NO_SYNC)
            }
            &Low::CopyTextureToBuffer {
                source_texture,
                offset,
                size,
                target_buffer,
                ref target_layout,
            } => {
                if offset != (0, 0) {
                    return Err(StepError::InvalidInstruction(line!()));
                }

                let encoder = match &mut self.command_encoder {
                    Some(encoder) => encoder,
                    None => return Err(StepError::InvalidInstruction(line!())),
                };

                let texture = self.descriptors.texture(source_texture)?;
                let buffer = self.descriptors.buffer(target_buffer, target_layout)?;

                let extent = wgpu::Extent3d {
                    width: size.0,
                    height: size.1,
                    depth_or_array_layers: 1,
                };

                encoder.copy_texture_to_buffer(texture, buffer, extent);

                Ok(SyncPoint::NO_SYNC)
            }
            &Low::CopyBufferToBuffer {
                source_buffer,
                size,
                target_buffer,
            } => {
                let encoder = match &mut self.command_encoder {
                    Some(encoder) => encoder,
                    None => return Err(StepError::InvalidInstruction(line!())),
                };

                let source = match self.descriptors.buffers.get(source_buffer.0) {
                    Some(source) => source,
                    None => return Err(StepError::InvalidInstruction(line!())),
                };

                let target = match self.descriptors.buffers.get(target_buffer.0) {
                    Some(target) => target,
                    None => return Err(StepError::InvalidInstruction(line!())),
                };

                encoder.copy_buffer_to_buffer(source, 0, target, 0, size);

                Ok(SyncPoint::NO_SYNC)
            }
            &Low::ReadBuffer {
                source_buffer,
                ref source_layout,
                offset,
                size,
                target_image,
            } => {
                let buffers = core::mem::take(&mut self.descriptors.buffers);
                let mut image_data = core::mem::take(&mut self.buffers);

                if offset != (0, 0) {
                    return Err(StepError::InvalidInstruction(line!()));
                }

                if size != (source_layout.width, source_layout.height) {
                    return Err(StepError::InvalidInstruction(line!()));
                }

                let (width, height) = size;
                let bytes_per_row = source_layout.bytes_per_row;
                let bytes_per_texel = source_layout.bytes_per_texel;
                let bytes_to_copy = (u32::from(bytes_per_texel) * width) as usize;

                let box_me = async move {
                    {
                        let buffer = &buffers[source_buffer.0];
                        let image = &mut image_data[target_image.0].data;

                        let slice = buffer.slice(..);
                        slice.map_async(wgpu::MapMode::Read).await.map_err(
                            |wgpu::BufferAsyncError| StepError::InvalidInstruction(line!()),
                        )?;

                        let data = slice.get_mapped_range();

                        // TODO: defensive programming, don't assume cast works.
                        let source_pitch = bytes_per_row as usize;
                        let target_pitch = image.layout().bytes_per_row as usize;

                        let source: &[u8] = &data[..];
                        let target: &mut [u8] = image.as_bytes_mut().unwrap();

                        for x in 0..height {
                            let source_row = &source[(x as usize * source_pitch)..][..source_pitch];
                            let target_row =
                                &mut target[(x as usize * target_pitch)..][..target_pitch];

                            target_row[..bytes_to_copy]
                                .copy_from_slice(&source_row[..bytes_to_copy]);
                        }
                    }

                    Ok(Cleanup::Buffers {
                        buffers,
                        image_data,
                    })
                };

                drop(_dump_on_panic);

                Ok(SyncPoint {
                    future: Some(DevicePolled {
                        future: Box::pin(box_me),
                        execution: self,
                    }),
                    marker: core::marker::PhantomData,
                })
            }
            Low::StackFrame(frame) => {
                if let Some(ref mut frames) = _dump_on_panic.stack {
                    frames.push(frame.clone());
                }

                Ok(SyncPoint::NO_SYNC)
            }
            Low::StackPop => {
                if let Some(ref mut frames) = _dump_on_panic.stack {
                    let _ = frames.pop();
                }

                Ok(SyncPoint::NO_SYNC)
            }
            inner => {
                return Err(StepError::BadInstruction(BadInstruction {
                    inner: format!("{:?}", inner),
                }))
            }
        }
    }

    /// Stop the execution.
    ///
    /// Discards all resources that are still held like buffers, the device, etc.
    pub fn retire(self) -> Result<(), RetireError> {
        let mut pool = Pool::new();

        let retire = self.retire_gracefully(&mut pool);
        retire.finish_by_discarding();

        drop(pool);
        Ok(())
    }

    /// Stop the execution, depositing all resources into the provided pool.
    #[must_use = "You won't get the ids of outputs."]
    pub fn retire_gracefully(self, pool: &mut Pool) -> Retire<'_> {
        self.gpu.device.stop_capture();
        Retire {
            execution: self,
            pool,
        }
    }
}

impl Descriptors {
    fn bind_group<'set>(
        &'set self,
        desc: &program::BindGroupDescriptor,
        buf: &'set mut Vec<wgpu::BindGroupEntry<'set>>,
    ) -> Result<wgpu::BindGroupDescriptor<'set>, StepError> {
        buf.clear();

        for (idx, entry) in desc.entries.iter().enumerate() {
            let resource = self.binding_resource(entry)?;
            buf.push(wgpu::BindGroupEntry {
                binding: idx as u32,
                resource,
            });
        }

        for &(idx, ref entry) in desc.sparse.iter() {
            let resource = self.binding_resource(entry)?;
            buf.push(wgpu::BindGroupEntry {
                binding: idx as u32,
                resource,
            });
        }

        Ok(wgpu::BindGroupDescriptor {
            label: None,
            layout: self
                .bind_group_layouts
                .get(desc.layout_idx)
                .ok_or_else(|| StepError::InvalidInstruction(line!()))?,
            entries: buf,
        })
    }

    fn binding_resource(
        &self,
        desc: &program::BindingResource,
    ) -> Result<wgpu::BindingResource<'_>, StepError> {
        use program::BindingResource::{Buffer, Sampler, TextureView};
        // eprintln!("{:?}", desc);
        match *desc {
            Buffer {
                buffer_idx,
                offset,
                size,
            } => {
                let buffer = self
                    .buffers
                    .get(buffer_idx)
                    .ok_or_else(|| StepError::InvalidInstruction(line!()))?;
                Ok(wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer,
                    offset,
                    size,
                }))
            }
            Sampler(idx) => self
                .sampler
                .get(idx)
                .ok_or_else(|| StepError::InvalidInstruction(line!()))
                .map(wgpu::BindingResource::Sampler),
            TextureView(idx) => self
                .texture_views
                .get(idx)
                .ok_or_else(|| StepError::InvalidInstruction(line!()))
                .map(wgpu::BindingResource::TextureView),
        }
    }

    fn bind_group_layout<'set>(
        &'set self,
        desc: &program::BindGroupLayoutDescriptor,
        buf: &'set mut Vec<wgpu::BindGroupLayoutEntry>,
    ) -> Result<wgpu::BindGroupLayoutDescriptor<'_>, StepError> {
        buf.clear();
        buf.extend_from_slice(&desc.entries);
        Ok(wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: buf,
        })
    }

    fn render_pass<'set, 'buf>(
        &'set self,
        desc: &program::RenderPassDescriptor,
        buf: &'buf mut Vec<wgpu::RenderPassColorAttachment<'set>>,
    ) -> Result<wgpu::RenderPassDescriptor<'set, 'buf>, StepError> {
        buf.clear();

        for attachment in &desc.color_attachments {
            buf.push(self.color_attachment(attachment)?);
        }

        Ok(wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: buf,
            depth_stencil_attachment: None,
        })
    }

    fn color_attachment(
        &self,
        desc: &program::ColorAttachmentDescriptor,
    ) -> Result<wgpu::RenderPassColorAttachment<'_>, StepError> {
        // eprintln!("Attaching {:?}", desc);
        Ok(wgpu::RenderPassColorAttachment {
            view: self
                .texture_views
                .get(desc.texture_view)
                .ok_or_else(|| StepError::InvalidInstruction(line!()))?,
            resolve_target: None,
            ops: desc.ops,
        })
    }

    fn pipeline<'set>(
        &'set self,
        desc: &program::RenderPipelineDescriptor,
        vertex_buffers: &'set mut Vec<wgpu::VertexBufferLayout<'set>>,
        fragments: &'set mut Vec<wgpu::ColorTargetState>,
    ) -> Result<wgpu::RenderPipelineDescriptor<'set>, StepError> {
        Ok(wgpu::RenderPipelineDescriptor {
            label: None,
            layout: self.pipeline_layouts.get(desc.layout),
            vertex: self.vertex_state(&desc.vertex, vertex_buffers)?,
            primitive: match desc.primitive {
                program::PrimitiveState::TriangleStrip => wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Cw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(self.fragment_state(&desc.fragment, fragments)?),
            // TODO: could be an efficient way to paint multiple times, with _different_ sets of
            // parameters. As opposed to rebinding buffers between paints.
            //
            // Add to the list of vertex, tessellation control, tessellation
            // evaluation, geometry, and fragment shader built-ins:
            // highp int gl_ViewIndex;
            multiview: None,
        })
    }

    fn pipeline_layout<'set>(
        &'set self,
        desc: &program::PipelineLayoutDescriptor,
        buf: &'set mut Vec<&'set wgpu::BindGroupLayout>,
    ) -> Result<wgpu::PipelineLayoutDescriptor<'_>, StepError> {
        buf.clear();

        for &layout in &desc.bind_group_layouts {
            let group = self
                .bind_group_layouts
                .get(layout)
                .ok_or_else(|| StepError::InvalidInstruction(line!()))?;
            buf.push(group);
        }

        Ok(wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: buf,
            push_constant_ranges: desc.push_constant_ranges,
        })
    }

    fn vertex_state<'set>(
        &'set self,
        desc: &program::VertexState,
        buf: &'set mut Vec<wgpu::VertexBufferLayout<'set>>,
    ) -> Result<wgpu::VertexState<'set>, StepError> {
        buf.clear();
        buf.push(wgpu::VertexBufferLayout {
            array_stride: 8,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 0,
                shader_location: 0,
            }],
        });
        Ok(wgpu::VertexState {
            module: self
                .shaders
                .get(desc.vertex_module)
                .ok_or_else(|| StepError::InvalidInstruction(line!()))?,
            entry_point: desc.entry_point,
            buffers: buf,
        })
    }

    fn fragment_state<'set>(
        &'set self,
        desc: &program::FragmentState,
        buf: &'set mut Vec<wgpu::ColorTargetState>,
    ) -> Result<wgpu::FragmentState<'_>, StepError> {
        buf.clear();
        buf.extend_from_slice(&desc.targets);
        Ok(wgpu::FragmentState {
            module: self
                .shaders
                .get(desc.fragment_module)
                .ok_or_else(|| StepError::InvalidInstruction(line!()))?,
            entry_point: desc.entry_point,
            targets: buf,
        })
    }

    fn buffer(
        &self,
        buffer: DeviceBuffer,
        layout: &BufferLayout,
    ) -> Result<wgpu::ImageCopyBuffer<'_>, StepError> {
        let buffer = match self.buffers.get(buffer.0) {
            None => return Err(StepError::InvalidInstruction(line!())),
            Some(buffer) => buffer,
        };
        Ok(wgpu::ImageCopyBufferBase {
            buffer,
            layout: wgpu::ImageDataLayout {
                bytes_per_row: NonZeroU32::new(layout.bytes_per_row),
                offset: 0,
                rows_per_image: NonZeroU32::new(layout.height),
            },
        })
    }

    fn texture(&self, texture: DeviceTexture) -> Result<wgpu::ImageCopyTexture<'_>, StepError> {
        let texture = match self.textures.get(texture.0) {
            None => return Err(StepError::InvalidInstruction(line!())),
            Some(texture) => texture,
        };
        Ok(wgpu::ImageCopyTextureBase {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        })
    }
}

impl Machine {
    pub(crate) fn new(instructions: Arc<[Low]>) -> Self {
        Machine {
            instructions,
            instruction_pointer: 0,
        }
    }

    fn next_instruction(&mut self) -> Result<&Low, StepError> {
        let instruction = self
            .instructions
            .get(self.instruction_pointer)
            .ok_or(StepError::ProgramEnd)?;
        self.instruction_pointer += 1;
        Ok(instruction)
    }

    fn render_pass<'pass>(
        &mut self,
        descriptors: &'pass Descriptors,
        mut pass: wgpu::RenderPass<'pass>,
    ) -> Result<(), StepError> {
        loop {
            let instruction = match self.next_instruction() {
                Err(StepError {
                    inner: StepErrorKind::ProgramEnd,
                    ..
                }) => return Err(StepError::RenderPassDidNotEnd),
                other => other?,
            };

            match instruction {
                &Low::SetPipeline(idx) => {
                    let pipeline = descriptors
                        .render_pipelines
                        .get(idx)
                        .ok_or_else(|| StepError::InvalidInstruction(line!()))?;
                    pass.set_pipeline(pipeline);
                }
                &Low::SetBindGroup {
                    group,
                    index,
                    ref offsets,
                } => {
                    let group = descriptors
                        .bind_groups
                        .get(group)
                        .ok_or_else(|| StepError::InvalidInstruction(line!()))?;
                    pass.set_bind_group(index, group, offsets);
                }
                &Low::SetVertexBuffer { slot, buffer } => {
                    let buffer = descriptors
                        .buffers
                        .get(buffer)
                        .ok_or_else(|| StepError::InvalidInstruction(line!()))?;
                    pass.set_vertex_buffer(slot, buffer.slice(..));
                }
                &Low::DrawOnce { vertices } => {
                    pass.draw(0..vertices, 0..1);
                }
                &Low::DrawIndexedZero { vertices } => {
                    pass.draw_indexed(0..vertices, 0, 0..1);
                }
                &Low::SetPushConstants {
                    stages,
                    offset,
                    ref data,
                } => {
                    pass.set_push_constants(stages, offset, data);
                }
                Low::EndRenderPass => return Ok(()),
                inner => {
                    return Err(StepError::BadInstruction(BadInstruction {
                        inner: format!("Unexpectedly within render pass: {:?}", inner),
                    }))
                }
            }
        }
    }
}

impl StartError {
    #[allow(non_snake_case)]
    // FIXME: find a better error representation but it's okay for now.
    // #[deprecated = "This should be cleaned up"]
    pub(crate) fn InternalCommandError(line: u32) -> Self {
        StartError {
            kind: LaunchErrorKind::FromLine(line),
        }
    }
}

#[allow(non_snake_case)]
#[allow(non_upper_case_globals)]
impl StepError {
    fn InvalidInstruction(line: u32) -> Self {
        StepError {
            inner: StepErrorKind::InvalidInstruction(line),
            ..Self::DEFAULT
        }
    }

    fn BadInstruction(bad: BadInstruction) -> Self {
        StepError {
            inner: StepErrorKind::BadInstruction(bad),
            ..Self::DEFAULT
        }
    }

    pub(crate) const ProgramEnd: Self = StepError {
        inner: StepErrorKind::ProgramEnd,
        ..Self::DEFAULT
    };

    pub(crate) const RenderPassDidNotEnd: Self = StepError {
        inner: StepErrorKind::RenderPassDidNotEnd,
        ..Self::DEFAULT
    };

    pub(crate) const DEFAULT: Self = StepError {
        inner: StepErrorKind::ProgramEnd,
        instruction_pointer: 0,
    };
}

impl Retire<'_> {
    /// Move the output image corresponding to `reg` into the pool.
    ///
    /// Return the image as viewed inside the pool. This is not arbitrary. See
    /// [`output_key`](Self::output_key) for more details (WIP).
    pub fn output(&mut self, reg: Register) -> Result<PoolImage<'_>, RetireError> {
        let index = self
            .execution
            .io_map
            .outputs
            .get(&reg)
            .copied()
            .ok_or(RetireError {
                inner: RetireErrorKind::NoSuchOutput,
            })?;

        // FIXME: don't take some random image, use the right one..
        // Also: should we leave the actual image? This would allow restarting the pipeline.
        let image = &mut self.execution.buffers[index];

        let descriptor = Descriptor {
            layout: image.data.layout().clone(),
            texel: image.texel.clone(),
        };

        let mut pool_image = self.pool.declare(descriptor);
        pool_image.swap(&mut image.data);

        Ok(pool_image.into())
    }

    /// Determine the pool key that will be preferred when calling `output`.
    pub fn output_key(&self, reg: Register) -> Result<Option<PoolKey>, RetireError> {
        let index = self
            .execution
            .io_map
            .outputs
            .get(&reg)
            .copied()
            .ok_or(RetireError {
                inner: RetireErrorKind::NoSuchOutput,
            })?;

        Ok(self.execution.buffers[index].key)
    }

    /// Drop any temporary buffers that had been allocated during execution.
    ///
    /// This leaves only IO resources alive, potentially reducing the amount of allocated memory
    /// space. In particular if you plan on calling [`finish`](Self::finish) where they would
    /// otherwise stay allocated indefinitely (until the underlying pool itself dropped, that is).
    pub fn prune(&mut self) {
        // FIXME: not implemented.
        // But also we don't put any image into the pool yet.
    }

    /// Collect all remaining items into the pool.
    ///
    /// Note that _every_ remaining buffer that can be reused will be put into the underlying pool.
    /// Be careful not to cause leaky behavior, that is clean up any such temporary buffers when
    /// they can no longer be used.
    ///
    /// You might wish to call [`prune`](Self::prune) to prevent such buffers from staying
    /// allocated in the pool in the first place.
    pub fn finish(self) {
        self.pool.reinsert_device(self.execution.gpu);
    }

    /// Explicitly discard all remaining items.
    ///
    /// The effect of this is the same as merely dropping it however it is much more explicit. In
    /// many cases you may want to call [`finish`](Self::finish) instead, ensuring that remaining
    /// resources that _can_ be reused are kept alive and can be re-acquired in future runs.
    pub fn finish_by_discarding(self) {
        // Yes, we want to drop everything.
        drop(self);
    }
}

impl SyncPoint<'_> {
    /// Block on synchronization, finishing device work.
    ///
    /// This will also poll the device if required (on native targets).
    pub fn block_on(&mut self) -> Result<(), StepError> {
        use crate::program::block_on;
        match self.future.take() {
            None => Ok(()),
            Some(polled) => {
                let execution = polled.execution;
                match block_on(polled.future, Some(&execution.gpu.device))? {
                    Cleanup::Buffers {
                        buffers,
                        image_data,
                    } => {
                        execution.descriptors.buffers = buffers;
                        execution.buffers = image_data;
                        Ok(())
                    }
                }
            }
        }
    }

    const NO_SYNC: Self = SyncPoint {
        future: None,
        marker: core::marker::PhantomData,
    };
}

impl Drop for SyncPoint<'_> {
    fn drop(&mut self) {
        if self.future.is_some() {
            let _ = self.block_on();
        }
    }
}
