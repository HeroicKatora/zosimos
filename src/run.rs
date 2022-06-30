use core::{future::Future, iter::once, marker::Unpin, num::NonZeroU32, pin::Pin};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::buffer::{ByteLayout, Descriptor};
use crate::command::Register;
use crate::pool::{BufferKey, ImageData, GpuKey, Pool, PoolImage, PoolKey, TextureKey};
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
    /// The auxiliary information on the program.
    pub(crate) info: Arc<ProgramInfo>,
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

pub(crate) struct ProgramInfo {
    pub(crate) texture_by_op: HashMap<usize, program::TextureDescriptor>,
    /// Annotates which low op allocates a cacheable buffer.
    pub(crate) buffer_by_op: HashMap<usize, program::BufferDescriptor>,
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
    /// The gpu, potentially from the pool.
    gpu: Gpu,
    /// The old gpu key from within the pool.
    gpu_key: Option<GpuKey>,
    /// Pre-allocated buffers.
    buffers: Vec<Image>,
    /// Map of program input/outputs as signature information (inverse of retiring).
    io_map: Arc<IoMap>,
    /// Static info about the program, i.e. resource it will require or benefit from cache/prefetching.
    info: Arc<ProgramInfo>,
    /// Cache state of this environment.
    cache: Cache,
}

/// A running `Executable`.
pub struct Execution {
    /// The gpu processor handles.
    pub(crate) gpu: core::cell::RefCell<Gpu>,
    /// All the host state of execution.
    pub(crate) host: Host,
    /// All cached data (from one or more pools).
    pub(crate) cache: Cache,
}

pub(crate) struct Host {
    pub(crate) machine: Machine,
    pub(crate) descriptors: Descriptors,
    pub(crate) command_encoder: Option<wgpu::CommandEncoder>,
    pub(crate) buffers: Vec<Image>,
    pub(crate) binary_data: Vec<u8>,
    pub(crate) io_map: Arc<IoMap>,
    pub(crate) debug_stack: Vec<Frame>,
    pub(crate) usage: ResourcesUsed,
}

#[derive(Default)]
pub(crate) struct Cache {
    preallocated_textures: HashMap<usize, wgpu::Texture>,
    preallocated_buffers: HashMap<usize, wgpu::Buffer>,
}

#[derive(Debug, Default)]
pub struct ResourcesUsed {
    buffer_mem: u64,
    buffer_reused: u64,
    texture_mem: u64,
    texture_reused: u64,
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
    /// The full descriptor of the data.
    /// This is used mainly for IO where we reconstruct a free-standing image with all its
    /// associated data when moving from an `Image`.
    pub(crate) descriptor: Descriptor,
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
    /// Post-information: descriptors with which buffers were created.
    /// These buffers can be reused if retired to a pool.
    buffer_descriptors: HashMap<usize, program::BufferDescriptor>,
    /// Post-information: descriptors with which textures were created.
    /// These textures may be reused.
    texture_descriptors: HashMap<usize, program::TextureDescriptor>,
}

pub(crate) struct Gpu {
    pub(crate) device: Device,
    pub(crate) queue: Queue,
}

/// Total memory recovered from previous buffers.
#[derive(Debug, Default)]
pub struct RecoveredBufferStats {
    mem: u64,
}

/// Total memory retired into retained buffers.
#[derive(Debug, Default)]
pub struct RetiredBufferStats {
    mem: u64,
    texture_keys: Vec<TextureKey>,
    buffer_keys: Vec<BufferKey>,
}

trait WithGpu {
    fn with_gpu<T>(&self, once: impl FnOnce(&mut Gpu) -> T) -> T;
}

impl WithGpu for core::cell::RefCell<Gpu> {
    fn with_gpu<T>(&self, once: impl FnOnce(&mut Gpu) -> T) -> T {
        let mut borrow = self.borrow_mut();
        once(&mut *borrow)
    }
}

impl WithGpu for &'_ core::cell::RefCell<Gpu> {
    fn with_gpu<T>(&self, once: impl FnOnce(&mut Gpu) -> T) -> T {
        let mut borrow = self.borrow_mut();
        once(&mut *borrow)
    }
}

type DynStep<'exe> = dyn core::future::Future<Output = Result<(), StepError>> + 'exe;

struct DevicePolled<'exe, T: WithGpu> {
    future: Pin<Box<DynStep<'exe>>>,
    gpu: T,
}

pub struct SyncPoint<'exe> {
    future: Option<DevicePolled<'exe, &'exe core::cell::RefCell<Gpu>>>,
    marker: core::marker::PhantomData<&'exe mut Execution>,
}

/// Represents a stopped execution instance, without information abouts its outputs.
pub struct Retire<'pool> {
    /// The retiring execution instance.
    execution: Execution,
    pool: &'pool mut Pool,
    uncorrected_gpu_textures: Vec<TextureKey>,
    uncorrected_gpu_buffers: Vec<BufferKey>,
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
    NoSuchInput,
    NoSuchOutput,
    BadInstruction,
}

impl Image {
    /// Create an image without binary data, promising to set it up later.
    pub(crate) fn with_late_bound(descriptor: &Descriptor) -> Self {
        Image {
            data: ImageData::LateBound(descriptor.to_canvas()),
            descriptor: descriptor.clone(),
            // Not related to any pooled image.
            key: None,
        }
    }

    pub(crate) fn clone_like(&self) -> Self {
        Image {
            data: ImageData::LateBound(self.data.layout().clone()),
            descriptor: self.descriptor.clone(),
            key: self.key,
        }
    }
}

impl Executable {
    pub fn from_pool<'p>(&self, pool: &'p mut Pool) -> Option<Environment<'p>> {
        let (gpu_key, gpu) = pool.select_device(&self.capabilities)?;
        Some(Environment {
            pool,
            gpu,
            gpu_key: Some(gpu_key),
            info: self.info.clone(),
            buffers: self.buffers.iter().map(Image::clone_like).collect(),
            io_map: self.io_map.clone(),
            cache: Cache::default(),
        })
    }

    pub fn launch(&self, mut env: Environment) -> Result<Execution, StartError> {
        self.check_satisfiable(&mut env)?;
        Ok(Execution {
            gpu: env.gpu.into(),
            host: Host {
                machine: Machine::new(Arc::clone(&self.instructions)),
                descriptors: Descriptors::default(),
                command_encoder: None,
                buffers: env.buffers,
                binary_data: self.binary_data.clone(),
                io_map: self.io_map.clone(),
                debug_stack: vec![],
                usage: ResourcesUsed::default(),
            },
            cache: env.cache,
        })
    }

    /// Run the executable but take all by value.
    pub fn launch_once(self, mut env: Environment) -> Result<Execution, StartError> {
        self.check_satisfiable(&mut env)?;
        Ok(Execution {
            gpu: env.gpu.into(),
            host: Host {
                machine: Machine::new(Arc::clone(&self.instructions)),
                descriptors: self.descriptors,
                command_encoder: None,
                buffers: env.buffers,
                binary_data: self.binary_data,
                io_map: self.io_map.clone(),
                debug_stack: vec![],
                usage: ResourcesUsed::default(),
            },
            cache: env.cache,
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

            if reference.descriptor != buffer.descriptor {
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

        // FIXME: Check env.cache against our program info?

        // Okay, checks done, let's patch up the state.
        for &input in self.io_map.inputs.values() {
            let buffer = &mut env.buffers[input];

            // Unwrap okay, check this earlier.
            let key = buffer.key.unwrap();
            let mut pool_img = env.pool.entry(key).unwrap();

            pool_img.trade(&mut buffer.data);
        }

        for &output in self.io_map.outputs.values() {
            if let Some(key) = env.buffers[output].key {
                let mut pool_img = env.pool.entry(key).unwrap();
                let buffer = &mut env.buffers[output];
                pool_img.swap(&mut buffer.data);
            } else {
                env.buffers[output].data.host_allocate();
            }
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

        // FIXME: we're ignoring color semantics here. Okay?
        if descriptor.layout != image.descriptor.layout {
            return Err(StartError::InternalCommandError(line!()));
        }

        if descriptor.texel != image.descriptor.texel {
            return Err(StartError::InternalCommandError(line!()));
        }

        match pool_img.data() {
            ImageData::Host(_) | ImageData::GpuTexture { .. } => {}
            _ => {
                eprintln!("Bad binding: {:?}", reg);
                return Err(StartError::InternalCommandError(line!()));
            }
        }

        image.key = Some(pool_img.key());

        Ok(())
    }

    pub fn bind_output(&mut self, reg: Register, key: PoolKey) -> Result<(), StartError> {
        let &idx = self
            .io_map
            .outputs
            .get(&reg)
            .ok_or_else(|| {
                StartError::InternalCommandError(line!())
            })?;

        let pool_img = self
            .pool
            .entry(key)
            .ok_or_else(|| StartError::InternalCommandError(line!()))?;

        let image = &mut self.buffers[idx];
        let descriptor = pool_img.descriptor();

        // FIXME: we're ignoring color semantics here. Okay?
        if descriptor.layout != image.descriptor.layout {
            return Err(StartError::InternalCommandError(line!()));
        }

        if descriptor.texel != image.descriptor.texel {
            return Err(StartError::InternalCommandError(line!()));
        }

        match pool_img.data() {
            ImageData::Host(_) | ImageData::GpuTexture { .. } => {}
            _ => {
                eprintln!("Bad binding: {:?}", reg);
                return Err(StartError::InternalCommandError(line!()));
            }
        }

        image.key = Some(pool_img.key());

        Ok(())
    }

    /// Retrieve matching temporary buffers from the pool.
    ///
    /// This reuses of allocations of buffers, textures, etc. from previous iterations of this
    /// program run where possible.
    pub fn recover_buffers(&mut self) -> RecoveredBufferStats {
        let mut stats = RecoveredBufferStats::default();

        let mut pool_cache = self.pool.as_cache(match self.gpu_key {
            Some(key) => key,
            None => return stats,
        });

        for (&inst, desc) in &self.info.texture_by_op {
            if let Some(texture) = pool_cache.extract_texture(desc) {
                stats.mem += desc.u64_len();
                self.cache.preallocated_textures.insert(inst, texture);
            }
        }

        for (&inst, desc) in &self.info.buffer_by_op {
            if let Some(buffer) = pool_cache.extract_buffer(desc) {
                stats.mem += desc.u64_len();
                self.cache.preallocated_buffers.insert(inst, buffer);
            }
        }

        stats
    }
}

impl Execution {
    pub(crate) fn new(init: InitialState) -> Self {
        init.device.start_capture();
        Execution {
            gpu: Gpu {
                device: init.device,
                queue: init.queue,
            }
            .into(),
            host: Host {
                machine: Machine::new(init.instructions),
                descriptors: Descriptors::default(),
                buffers: init.buffers,
                command_encoder: None,
                binary_data: init.binary_data,
                io_map: Arc::new(init.io_map),
                debug_stack: vec![],
                usage: ResourcesUsed::default(),
            },
            cache: Cache::default(),
        }
    }

    /// Check if the machine is still running.
    pub fn is_running(&self) -> bool {
        self.host.machine.instruction_pointer < self.host.machine.instructions.len()
    }

    /// FIXME: a way to pass a `&wgpu::SurfaceTexture` as output?
    /// Otherwise, have to make an extra copy call in the pool.
    pub fn step(&mut self) -> Result<SyncPoint<'_>, StepError> {
        let instruction_pointer = self.host.machine.instruction_pointer;

        let Execution { ref gpu, host, cache } = self;
        let async_step = async move {
            match host.step_inner(cache, gpu).await {
                Err(mut error) => {
                    // Add tracing information..
                    error.instruction_pointer = instruction_pointer;
                    Err(error)
                }
                other => other,
            }
        };

        // TODO: test the waters with one no-waker poll?
        Ok(SyncPoint {
            future: Some(DevicePolled {
                future: Box::pin(async_step),
                gpu: &self.gpu,
            }),
            marker: core::marker::PhantomData,
        })
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
        self.gpu.with_gpu(|gpu| gpu.device.stop_capture());
        Retire {
            execution: self,
            pool,
            uncorrected_gpu_textures: vec![],
            uncorrected_gpu_buffers: vec![],
        }
    }

    /// Debug how many resources were used, any how.
    pub fn resources_used(&self) -> &ResourcesUsed {
        &self.host.usage
    }
}

impl Host {
    async fn step_inner(&mut self, cache: &mut Cache, gpu: impl WithGpu) -> Result<(), StepError> {
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

        let inst = self.machine.instruction_pointer;
        match self.machine.next_instruction()? {
            Low::BindGroupLayout(desc) => {
                let mut entry_buffer = vec![];
                let group = self
                    .descriptors
                    .bind_group_layout(desc, &mut entry_buffer)?;
                // eprintln!("Made {}: {:?}", self.descriptors.bind_group_layouts.len(), group);
                let group = gpu.with_gpu(|gpu| gpu.device.create_bind_group_layout(&group));
                self.descriptors.bind_group_layouts.push(group);
                Ok(())
            }
            Low::BindGroup(desc) => {
                let mut entry_buffer = vec![];
                let group = self.descriptors.bind_group(desc, &mut entry_buffer)?;
                // eprintln!("{}: {:?}", desc.layout_idx, group);
                let group = gpu.with_gpu(|gpu| gpu.device.create_bind_group(&group));
                self.descriptors.bind_groups.push(group);
                Ok(())
            }
            Low::Buffer(desc) => {
                let wgpu_desc = wgpu::BufferDescriptor {
                    label: None,
                    size: desc.size,
                    usage: desc.usage.to_wgpu(),
                    mapped_at_creation: false,
                };

                let buffer = if let Some(buffer) = cache.preallocated_buffers.remove(&inst) {
                    self.usage.buffer_reused += desc.u64_len();
                    buffer
                } else {
                    self.usage.buffer_mem += desc.u64_len();
                    gpu.with_gpu(|gpu| gpu.device.create_buffer(&wgpu_desc))
                };

                let buffer_idx = self.descriptors.buffers.len();
                self.descriptors.buffer_descriptors.insert(buffer_idx, desc.clone());
                self.descriptors.buffers.push(buffer);
                Ok(())
            }
            Low::BufferInit(desc) => {
                use wgpu::util::DeviceExt;
                let wgpu_desc = wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: desc.content.as_slice(&self.binary_data),
                    usage: desc.usage.to_wgpu(),
                };

                self.usage.buffer_mem += desc.u64_len();
                let buffer = gpu.with_gpu(|gpu| gpu.device.create_buffer_init(&wgpu_desc));
                self.descriptors.buffers.push(buffer);
                Ok(())
            }
            Low::Shader(desc) => {
                let shader;
                if std::env::var("STEALTH_PAINT_PASSTHROUGH").is_err() {
                    let desc = wgpu::ShaderModuleDescriptor {
                        label: Some(desc.name),
                        source: wgpu::ShaderSource::SpirV(desc.source_spirv.as_ref().into()),
                    };

                    shader = gpu.with_gpu(|gpu| gpu.device.create_shader_module(&desc));
                } else {
                    let desc = wgpu::ShaderModuleDescriptor {
                        label: Some(desc.name),
                        source: wgpu::ShaderSource::SpirV(desc.source_spirv.as_ref().into()),
                    };

                    shader = gpu.with_gpu(|gpu| gpu.device.create_shader_module(&desc));
                };

                self.descriptors.shaders.push(shader);
                Ok(())
            }
            Low::PipelineLayout(desc) => {
                let mut entry_buffer = vec![];
                let layout = self.descriptors.pipeline_layout(desc, &mut entry_buffer)?;
                let layout = gpu.with_gpu(|gpu| gpu.device.create_pipeline_layout(&layout));
                self.descriptors.pipeline_layouts.push(layout);
                Ok(())
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
                let sampler = gpu.with_gpu(|gpu| gpu.device.create_sampler(&desc));
                self.descriptors.sampler.push(sampler);
                Ok(())
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
                Ok(())
            }
            Low::Texture(desc) => {
                use wgpu::TextureUsages as U;
                let wgpu_desc = wgpu::TextureDescriptor {
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

                let texture = if let Some(texture) = cache.preallocated_textures.remove(&inst) {
                    self.usage.texture_reused += desc.u64_len();
                    texture
                } else {
                    self.usage.texture_mem += desc.u64_len();
                    gpu.with_gpu(|gpu| gpu.device.create_texture(&wgpu_desc))
                };

                let texture_idx = self.descriptors.textures.len();
                self.descriptors.texture_descriptors.insert(texture_idx, desc.clone());
                self.descriptors.textures.push(texture);
                Ok(())
            }
            Low::RenderPipeline(desc) => {
                let mut vertex_buffers = vec![];
                let mut fragments = vec![];

                let pipeline =
                    self.descriptors
                        .pipeline(desc, &mut vertex_buffers, &mut fragments)?;
                let pipeline = gpu.with_gpu(|gpu| gpu.device.create_render_pipeline(&pipeline));
                self.descriptors.render_pipelines.push(pipeline);
                Ok(())
            }
            Low::BeginCommands => {
                if self.command_encoder.is_some() {
                    return Err(StepError::InvalidInstruction(line!()));
                }

                let descriptor = wgpu::CommandEncoderDescriptor { label: None };

                let encoder = gpu.with_gpu(|gpu| gpu.device.create_command_encoder(&descriptor));
                self.command_encoder = Some(encoder);
                Ok(())
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

                Ok(())
            }
            Low::EndCommands => match self.command_encoder.take() {
                None => Err(StepError::InvalidInstruction(line!())),
                Some(encoder) => {
                    self.descriptors.command_buffers.push(encoder.finish());
                    Ok(())
                }
            },
            &Low::RunTopCommand => {
                let command = self
                    .descriptors
                    .command_buffers
                    .pop()
                    .ok_or_else(|| StepError::InvalidInstruction(line!()))?;
                gpu.with_gpu(|gpu| gpu.queue.submit(once(command)));
                Ok(())
            }
            &Low::RunTopToBot(many) => {
                if many > self.descriptors.command_buffers.len() {
                    return Err(StepError::InvalidInstruction(line!()));
                }

                let commands = self.descriptors.command_buffers.drain(many..);
                gpu.with_gpu(|gpu| gpu.queue.submit(commands.rev()));
                Ok(())
            }
            &Low::RunBotToTop(many) => {
                if many > self.descriptors.command_buffers.len() {
                    return Err(StepError::InvalidInstruction(line!()));
                }

                let commands = self.descriptors.command_buffers.drain(many..);
                gpu.with_gpu(|gpu| gpu.queue.submit(commands));
                Ok(())
            }
            &Low::WriteImageToBuffer {
                source_image,
                offset,
                size,
                target_buffer,
                ref target_layout,
                copy_dst_buffer,
            } => {
                if offset != (0, 0) {
                    return Err(StepError::InvalidInstruction(line!()));
                }

                let source = match self.buffers.get(source_image.0) {
                    None => return Err(StepError::InvalidInstruction(line!())),
                    Some(source) => &source.data,
                };

                let layout = source.layout().clone();
                let target_size = (target_layout.width, target_layout.height);

                if target_size != (layout.width(), layout.height()) {
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
                    gpu.queue.write_buffer(buffer.buffer, 0, data);
                    return Ok(SyncPoint::NO_SYNC)
                }
                */

                if size != target_size {
                    // Not yet supported (or needed).
                    return Err(StepError::InvalidInstruction(line!()));
                }

                let (width, height) = target_size;
                let bytes_per_row = target_layout.row_stride;
                let bytes_per_texel = target_layout.texel_stride;
                let bytes_to_copy = (u32::from(bytes_per_texel) * width) as usize;

                let image = &mut self.buffers[source_image.0].data;
                let buffer = &self.descriptors.buffers[target_buffer.0];

                if let ImageData::GpuTexture {
                    texture,
                    // FIXME: validate layout? What for?
                    layout: _,
                    gpu: _,
                } = image
                {
                    let descriptor = wgpu::CommandEncoderDescriptor { label: None };
                    let mut encoder = gpu.with_gpu(|gpu| gpu.device.create_command_encoder(&descriptor));

                    encoder.copy_texture_to_buffer(
                        texture.as_image_copy(),
                        wgpu::ImageCopyBufferBase {
                            buffer: &self.descriptors.buffers[copy_dst_buffer.0],
                            layout: wgpu::ImageDataLayout {
                                bytes_per_row: NonZeroU32::new(bytes_per_row as u32),
                                offset: 0,
                                rows_per_image: NonZeroU32::new(size.1),
                            },
                        },
                        wgpu::Extent3d {
                            width: size.0,
                            height: size.1,
                            depth_or_array_layers: 1,
                        },
                    );

                    let command = encoder.finish();
                    gpu.with_gpu(|gpu| gpu.queue.submit(once(command)));

                    return Ok(());
                }

                if image.as_bytes().is_none() {
                    return Err(StepError::InvalidInstruction(line!()));
                }

                let slice = buffer.slice(..);
                slice
                    .map_async(wgpu::MapMode::Write)
                    .await
                    .map_err(|wgpu::BufferAsyncError| StepError::InvalidInstruction(line!()))?;

                // eprintln!("WriteImageToBuffer");
                // eprintln!(" Source: {:?}", source_image.0);
                // eprintln!(" Target: {:?}", target_buffer.0);
                let mut data = slice.get_mapped_range_mut();

                // We've checked that this image can be seen as host bytes.
                let source: &[u8] = image.as_bytes().unwrap();
                let target: &mut [u8] = &mut data[..];

                // TODO: defensive programming, don't assume cast works. Proof?
                let target_pitch = bytes_per_row as usize;
                // TODO(perf): should this use our internal descriptor?
                let source_pitch = image.layout().as_row_layout().row_stride as usize;

                for x in 0..height {
                    let source_line = x as usize * source_pitch;
                    debug_assert!(source.get(source_line..).is_some());
                    debug_assert!(source[source_line..].len() >= source_pitch);
                    let source_row = &source[source_line..][..source_pitch];
                    let target_line = x as usize * target_pitch;
                    debug_assert!(target.get(target_line..).is_some());
                    debug_assert!(target[target_line..].len() >= target_pitch);
                    let target_row = &mut target[target_line..][..target_pitch];
                    target_row[..bytes_to_copy].copy_from_slice(&source_row[..bytes_to_copy]);
                }

                drop(data);
                buffer.unmap();

                Ok(())
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

                // eprintln!("CopyBufferToTexture");
                // eprintln!(" Source: {:?}", source_buffer);
                // eprintln!(" Target: {:?}", target_texture);

                // eprintln!("{:?}", buffer);
                // eprintln!("{:?}", texture);
                // eprintln!(" {:?}", extent);

                encoder.copy_buffer_to_texture(buffer, texture, extent);

                Ok(())
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

                Ok(())
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

                // eprintln!("CopyBufferToBuffer");
                // eprintln!(" Source: {:?}", source_buffer.0);
                // eprintln!(" Target: {:?}", target_buffer.0);
                // eprintln!(" Size: {:?}", size);

                encoder.copy_buffer_to_buffer(source, 0, target, 0, size);

                Ok(())
            }
            &Low::ReadBuffer {
                source_buffer,
                ref source_layout,
                offset,
                size,
                target_image,
                copy_src_buffer,
            } => {
                let source_size = (source_layout.width, source_layout.height);

                if offset != (0, 0) {
                    return Err(StepError::InvalidInstruction(line!()));
                }

                if size != source_size {
                    return Err(StepError::InvalidInstruction(line!()));
                }

                let bytes_per_row = source_layout.row_stride;
                let bytes_per_texel = source_layout.texel_stride;
                let (width, height) = size;
                let bytes_to_copy = (u32::from(bytes_per_texel) * width) as usize;

                let buffer = &self.descriptors.buffers[source_buffer.0];
                let image = &mut self.buffers[target_image.0].data;

                if let ImageData::GpuTexture {
                    texture,
                    // FIXME: validate layout? What for?
                    layout: _,
                    gpu: _,
                } = image
                {
                    let descriptor = wgpu::CommandEncoderDescriptor { label: None };
                    let mut encoder = gpu.with_gpu(|gpu| gpu.device.create_command_encoder(&descriptor));

                    encoder.copy_buffer_to_texture(
                        wgpu::ImageCopyBufferBase {
                            buffer: &self.descriptors.buffers[copy_src_buffer.0],
                            layout: wgpu::ImageDataLayout {
                                bytes_per_row: NonZeroU32::new(bytes_per_row as u32),
                                offset: 0,
                                rows_per_image: NonZeroU32::new(size.1),
                            },
                        },
                        texture.as_image_copy(),
                        wgpu::Extent3d {
                            width: size.0,
                            height: size.1,
                            depth_or_array_layers: 1,
                        },
                    );

                    let command = encoder.finish();
                    gpu.with_gpu(|gpu| gpu.queue.submit(once(command)));

                    return Ok(());
                }

                if image.as_bytes().is_none() {
                    return Err(StepError::InvalidInstruction(line!()));
                }

                let slice = buffer.slice(..);
                slice
                    .map_async(wgpu::MapMode::Read)
                    .await
                    .map_err(|wgpu::BufferAsyncError| StepError::InvalidInstruction(line!()))?;

                let data = slice.get_mapped_range();

                // TODO: defensive programming, don't assume cast works.
                let source_pitch = bytes_per_row as usize;
                let target_pitch = image.layout().as_row_layout().row_stride as usize;

                let source: &[u8] = &data[..];
                let target: &mut [u8] = image.as_bytes_mut().unwrap();

                for x in 0..height {
                    let source_row = &source[(x as usize * source_pitch)..][..source_pitch];
                    let target_row = &mut target[(x as usize * target_pitch)..][..target_pitch];

                    target_row[..bytes_to_copy].copy_from_slice(&source_row[..bytes_to_copy]);
                }

                drop(data);
                buffer.unmap();

                Ok(())
            }
            Low::StackFrame(frame) => {
                if let Some(ref mut frames) = _dump_on_panic.stack {
                    frames.push(frame.clone());
                }

                Ok(())
            }
            Low::StackPop => {
                if let Some(ref mut frames) = _dump_on_panic.stack {
                    let _ = frames.pop();
                }

                Ok(())
            }
            inner => {
                return Err(StepError::BadInstruction(BadInstruction {
                    inner: format!("{:?}", inner),
                }))
            }
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
        layout: &ByteLayout,
    ) -> Result<wgpu::ImageCopyBuffer<'_>, StepError> {
        let buffer = match self.buffers.get(buffer.0) {
            None => return Err(StepError::InvalidInstruction(line!())),
            Some(buffer) => buffer,
        };
        Ok(wgpu::ImageCopyBufferBase {
            buffer,
            layout: wgpu::ImageDataLayout {
                bytes_per_row: NonZeroU32::new(layout.row_stride as u32),
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
    /// Move the input image corresponding to `reg` back into the pool.
    ///
    /// Return the image as viewed inside the pool. This is not arbitrary. See
    /// [`output_key`](Self::output_key) for more details (WIP).
    pub fn input(&mut self, reg: Register) -> Result<PoolImage<'_>, RetireError> {
        let index = self
            .execution
            .host
            .io_map
            .inputs
            .get(&reg)
            .copied()
            .ok_or(RetireError {
                inner: RetireErrorKind::NoSuchInput,
            })?;

        let image = &mut self.execution.host.buffers[index];
        let descriptor = image.data.layout().clone();

        let mut pool_image;
        let pool = &mut self.pool;
        match image.key.filter(|key| pool.entry(*key).is_some()) {
            Some(key) => pool_image = self.pool.entry(key).unwrap(),
            None => {
                let descriptor = Descriptor::from(&descriptor);
                pool_image = self.pool.declare(descriptor);
            }
        };

        pool_image.swap(&mut image.data);

        Ok(pool_image.into())
    }

    /// Move the output image corresponding to `reg` into the pool.
    ///
    /// Return the image as viewed inside the pool. This is not arbitrary. See
    /// [`output_key`](Self::output_key) for more details (WIP).
    pub fn output(&mut self, reg: Register) -> Result<PoolImage<'_>, RetireError> {
        let index = self
            .execution
            .host
            .io_map
            .outputs
            .get(&reg)
            .copied()
            .ok_or(RetireError {
                inner: RetireErrorKind::NoSuchOutput,
            })?;

        let image = &mut self.execution.host.buffers[index];
        let descriptor = image.data.layout().clone();

        let mut pool_image;
        let pool = &mut self.pool;
        match image.key.filter(|key| pool.entry(*key).is_some()) {
            Some(key) => pool_image = self.pool.entry(key).unwrap(),
            None => {
                let descriptor = Descriptor::from(&descriptor);
                pool_image = self.pool.declare(descriptor);
            }
        };

        pool_image.swap(&mut image.data);

        Ok(pool_image.into())
    }

    /// Determine the pool key that will be preferred when calling `output`.
    pub fn output_key(&self, reg: Register) -> Result<Option<PoolKey>, RetireError> {
        let index = self
            .execution
            .host
            .io_map
            .outputs
            .get(&reg)
            .copied()
            .ok_or(RetireError {
                inner: RetireErrorKind::NoSuchOutput,
            })?;

        Ok(self.execution.host.buffers[index].key)
    }

    /// Retain temporary buffers that had been allocated during execution.
    pub fn retire_buffers(&mut self) -> RetiredBufferStats {
        let mut stats = RetiredBufferStats::default();

        let descriptors = &mut self.execution.host.descriptors;

        let tidx = 0..descriptors.textures.len();
        let textures = descriptors.textures.drain(..).zip(tidx);
        for (texture, idx) in textures {
            let descriptor = match descriptors.texture_descriptors.get(&idx) {
                None => continue,
                Some(descriptor) => descriptor,
            };

            let key = self.pool.insert_cacheable_texture(descriptor, texture);
            stats.mem += descriptor.u64_len();
            stats.texture_keys.push(key);
            self.uncorrected_gpu_textures.push(key);
        }

        let tidx = 0..descriptors.buffers.len();
        let textures = descriptors.buffers.drain(..).zip(tidx);

        for (texture, idx) in textures {
            let descriptor = match descriptors.buffer_descriptors.get(&idx) {
                None => continue,
                Some(descriptor) => descriptor,
            };

            let key = self.pool.insert_cacheable_buffer(descriptor, texture);
            stats.mem += descriptor.u64_len();
            stats.buffer_keys.push(key);
            self.uncorrected_gpu_buffers.push(key);
        }

        stats
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
    pub fn finish(mut self) {
        let _ = self.retire_buffers();

        let gpu = self.execution.gpu.into_inner();
        let gpu_key = self.pool.reinsert_device(gpu);

        // Fixup the gpu reference for all inserted gpu buffers.
        for pool_key in self.uncorrected_gpu_textures.into_iter() {
            self.pool.reassign_texture_gpu_unguarded(pool_key, gpu_key);
        }

        for pool_key in self.uncorrected_gpu_buffers.into_iter() {
            self.pool.reassign_buffer_gpu_unguarded(pool_key, gpu_key);
        }
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
        match self.future.take() {
            None => Ok(()),
            Some(polled) => {
                let DevicePolled { future, gpu } = polled;
                block_on(future, Some(gpu))?;
                Ok(())
            }
        }
    }
}

impl Drop for SyncPoint<'_> {
    fn drop(&mut self) {
        if self.future.is_some() {
            let _ = self.block_on();
        }
    }
}

/// Block on an async future that may depend on a device being polled.
pub(crate) fn block_on<F, T>(future: F, device: Option<&core::cell::RefCell<Gpu>>) -> T
where
    F: Future<Output = T> + Unpin,
    T: 'static,
{
    fn spin_block<R>(mut f: impl Future<Output = R> + Unpin) -> R {
        use core::hint::spin_loop;
        use core::task::{Context, Poll};

        let mut f = Pin::new(&mut f);

        // create the context
        let waker = waker_fn::waker_fn(|| {});
        let mut ctx = Context::from_waker(&waker);

        // poll future in a loop
        loop {
            match f.as_mut().poll(&mut ctx) {
                Poll::Ready(o) => return o,
                Poll::Pending => spin_loop(),
            }
        }
    }

    if let Some(device) = device.as_ref() {
        // We have to manually poll the device.  That is, we ensure that it keeps being polled
        // and each time will also poll the device. This isn't super efficient but a dirty way
        // to actually finish this future.
        struct DevicePolled<'dev, F, D: WithGpu> {
            future: F,
            device: &'dev D,
        }

        impl<F, D: WithGpu> Future for DevicePolled<'_, F, D>
        where
            F: Future + Unpin,
        {
            type Output = F::Output;
            fn poll(
                self: core::pin::Pin<&mut Self>,
                ctx: &mut core::task::Context,
            ) -> core::task::Poll<F::Output> {
                self.as_ref()
                    .device
                    .with_gpu(|gpu| gpu.device.poll(wgpu::Maintain::Poll));
                // Ugh, noooo...
                ctx.waker().wake_by_ref();
                Pin::new(&mut self.get_mut().future).poll(ctx)
            }
        }

        spin_block(DevicePolled { future, device })
    } else {
        spin_block(future)
    }
}
