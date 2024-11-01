mod timing;

use core::{
    future::Future,
    iter::once,
    marker::{PhantomData, Unpin},
    ops::Range,
    pin::Pin,
};

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use crate::buffer::{BufferLayout, ByteLayout, Descriptor};
use crate::command::{Register, RegisterKnob};
use crate::pool::{
    BufferKey, Gpu, GpuKey, ImageData, PipelineKey, Pool, PoolImage, PoolKey, ShaderKey, TextureKey,
};
use crate::program::{self, Capabilities, DeviceBuffer, DeviceTexture, Knob, Low};
use crate::util::Ping;

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
    pub(crate) image_io_buffers: Vec<Image>,
    /// The map from registers to the index in image data.
    /// FIXME: this makes us a little inflexible for multiple entry points. I'd like singular entry
    /// point to not become too engrained in thinking and design, avoiding what went wrong with
    /// SPIR-V. On the other hand, elf does not have any such ideas either. Anyways the whole
    /// program info could also be rebuildable from other parts and in particular `ProgramInfo` is
    /// designed for one entry point, too. Its buffer recovery can not work with a function called
    /// multiple times.
    pub(crate) io_map: Arc<IoMap>,
    /// The main function to start execution at. This is the function for which ll the other
    /// descriptors and the io_map will apply.
    pub(crate) entry_point: program::Function,
    /// The capabilities required from devices to execute this.
    pub(crate) capabilities: Capabilities,
}

pub(crate) struct ProgramInfo {
    pub(crate) texture_by_op: HashMap<usize, program::TextureDescriptor>,
    pub(crate) buffer_by_op: HashMap<usize, program::BufferDescriptor>,
    pub(crate) shader_by_op: HashMap<usize, program::ShaderDescriptorKey>,
    pub(crate) pipeline_by_op: HashMap<usize, program::RenderPipelineKey>,
    /// When event `val` is already achieved, op `key` becomes irrelevant.
    /// TODO: some instruction results supply multiple events.
    pub(crate) skip_by_op: HashMap<program::Instruction, program::Event>,
    pub(crate) functions: HashMap<program::Function, program::FunctionFrame>,
    pub(crate) knobs: HashMap<RegisterKnob, program::Knob>,
    pub(crate) knob_descriptors: HashMap<program::Knob, program::KnobDescriptor>,
    pub(crate) knob_starts: BTreeMap<usize, program::Knob>,
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
    /// The pool to act on when referencing resources.
    pool: &'pool mut Pool,
    /// The gpu, potentially from the pool.
    gpu: Gpu,
    /// The old gpu key from within the pool.
    gpu_key: Option<GpuKey>,
    /// Pre-allocated buffers.
    buffers: Vec<Image>,
    /// Map of program input/outputs as signature information (inverse of retiring).
    io_map: Arc<IoMap>,
    knob_data: Vec<u8>,
    knobs: HashMap<Knob, Range<usize>>,
    /// Static info about the program, i.e. resource it will require or benefit from cache/prefetching.
    info: Arc<ProgramInfo>,
    /// Cache state of this environment.
    cache: Cache,
}

/// A running [`Executable`] with some particular function stack and resources.
pub struct Execution {
    /// The gpu processor handles.
    pub(crate) gpu: Gpu,
    /// The key of the gpu if it was extracted from a pool.
    pub(crate) gpu_key: Option<GpuKey>,
    /// All the host state of execution.
    pub(crate) host: Host,
    /// All cached data (from one or more pools).
    pub(crate) cache: Cache,
}

pub(crate) struct Host {
    pub(crate) machine: Machine,
    pub(crate) binary_data: Vec<u8>,
    pub(crate) io_map: Arc<IoMap>,
    pub(crate) info: Arc<ProgramInfo>,

    knob_data: Arc<[u8]>,
    knobs: HashMap<Knob, Range<usize>>,

    /// Variable information during execution.
    pub(crate) command_encoder: Option<wgpu::CommandEncoder>,
    pub(crate) descriptors: Descriptors,
    pub(crate) call_stack: Vec<Descriptors>,
    pub(crate) debug_stack: Vec<Frame>,
    pub(crate) usage: ResourcesUsed,

    /// Submits inserted by the execution.
    /// FIXME: really, we should not have these. The encoder should somehow plan for the
    /// eventuality of IO with GPU buffers. In particular we can delay the effect of texturing
    /// until such scheduling happens as it is GPU synchronous instead of host-synchronous, when we
    /// need not map a buffer.
    pub(crate) delayed_submits: usize,

    /// Debug information.
    pub(crate) debug: Debug,
}

#[derive(Default, Debug)]
#[allow(unused)]
pub(crate) struct Debug {
    bind_groups: HashMap<usize, DebugInfo>,
    bind_group_layouts: HashMap<usize, DebugInfo>,
    buffers: HashMap<DeviceBuffer, DebugBufferInfo>,
    buffer_history: HashMap<DeviceBuffer, Vec<TextureInitState>>,
    command_buffers: HashMap<usize, DebugInfo>,
    shaders: HashMap<usize, DebugInfo>,
    pipeline_layouts: HashMap<usize, DebugInfo>,
    render_pipelines: HashMap<usize, DebugInfo>,
    sampler: HashMap<usize, DebugInfo>,
    textures: HashMap<DeviceTexture, DebugTextureInfo>,
    texture_history: HashMap<DeviceTexture, Vec<TextureInitState>>,
    texture_views: HashMap<usize, DebugViewInfo>,
}

#[derive(Default, Debug)]
#[allow(unused)]
pub(crate) struct DebugInfo {
    /// Information attached at runtime to this object, i.e. a special kind of name.
    assertion_data: Option<String>,
}

#[derive(Default, Debug)]
#[allow(unused)]
pub(crate) struct DebugBufferInfo {
    info: DebugInfo,
    init: TextureInitState,
}

#[derive(Default, Debug)]
#[allow(unused)]
pub(crate) struct DebugTextureInfo {
    info: DebugInfo,
    init: TextureInitState,
}

#[derive(Debug)]
#[allow(unused)]
pub(crate) struct DebugViewInfo {
    info: DebugInfo,
    init: Option<DeviceTexture>,
}

#[derive(Clone, Copy, Default, Debug)]
enum TextureInitState {
    #[default]
    Uninit,
    WriteTo,
    /// Utilized as a shader source.
    UseRead,
    /// Utilized as a shader target.
    UseWrite,
    ReadFrom,
}

#[derive(Default)]
pub(crate) struct Cache {
    preallocated_textures: HashMap<usize, wgpu::Texture>,
    preallocated_buffers: HashMap<usize, wgpu::Buffer>,
    precompiled_shader: HashMap<usize, wgpu::ShaderModule>,
    precompiled_pipelines: HashMap<usize, wgpu::RenderPipeline>,
}

#[derive(Debug, Default)]
pub struct ResourcesUsed {
    buffer_mem: u64,
    buffer_reused: u64,
    texture_mem: u64,
    texture_reused: u64,
    shaders_compiled: u64,
    shaders_reused: u64,
    pipelines_compiled: u64,
    pipelines_reused: u64,
}

pub struct StepLimits {
    instructions: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct Frame {
    // Used through Debug
    #[allow(dead_code)]
    pub(crate) name: String,
}

pub(crate) struct InitialState {
    pub(crate) entry_point: program::Function,
    pub(crate) instructions: Arc<[Low]>,
    pub(crate) info: Arc<ProgramInfo>,
    pub(crate) device: Device,
    pub(crate) queue: Queue,
    pub(crate) image_io_buffers: Vec<Image>,
    pub(crate) binary_data: Vec<u8>,
    pub(crate) io_map: Arc<IoMap>,
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

#[derive(Default, Debug)]
pub struct IoMap {
    /// Map input registers to their index in `buffers`.
    pub(crate) inputs: HashMap<Register, usize>,
    /// Map output registers to their index in `buffers`.
    pub(crate) outputs: HashMap<Register, usize>,
    /// Map output registers to their index in `buffers`.
    pub(crate) renders: HashMap<Register, usize>,
}

#[derive(Default)]
pub(crate) struct Descriptors {
    pub(crate) image_io_buffers: Vec<Image>,
    bind_groups: Vec<wgpu::BindGroup>,
    bind_group_layouts: Vec<wgpu::BindGroupLayout>,
    buffers: Vec<Arc<wgpu::Buffer>>,
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
    /// Post-information: descriptors with which textures were created.
    /// These textures may be reused.
    shader_descriptors: HashMap<usize, program::ShaderDescriptorKey>,
    /// Post-information: descriptors of shaders and parameters part of a pipeline.
    /// These may be reused as compiling modules is expensive on the driver.
    pipeline_descriptors: HashMap<usize, program::RenderPipelineKey>,
    /// Instructions, whose intended effect was made unobservable by the environment.
    ///
    /// Consider a `ReadBuffer`. When the target is a `texture` then it is not necessary to perform
    /// the buffer-to-buffer copy into the host-mappable read buffer. This instruction is,
    /// nevertheless, part of the instruction stream. The reverse is also true but required, not
    /// merely an optimization. A WriteImageToBuffer from a GPU texture will initialize the target
    /// buffer and we must consequently skip the buffer-to-buffer from the mappable write buffer.
    precomputed: HashMap<program::Event, Precomputed>,
}

/// Information about the source of computation for skipped instructions.
#[derive(Default)]
struct Precomputed;

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
    shader_keys: Vec<ShaderKey>,
    pipeline_keys: Vec<PipelineKey>,
}

trait WithGpu {
    fn with_gpu<T>(&self, once: impl FnOnce(&Gpu) -> T) -> T;
}

impl WithGpu for Gpu {
    fn with_gpu<T>(&self, once: impl FnOnce(&Gpu) -> T) -> T {
        once(self)
    }
}

impl WithGpu for &'_ Gpu {
    fn with_gpu<T>(&self, once: impl FnOnce(&Gpu) -> T) -> T {
        once(&**self)
    }
}

type DynStep<'exe> = dyn core::future::Future<Output = Result<(), StepError>> + 'exe;

struct DevicePolled<'exe, T: WithGpu> {
    future: Pin<Box<DynStep<'exe>>>,
    gpu: T,
}

pub struct SyncPoint<'exe> {
    future: Option<DevicePolled<'exe, Gpu>>,
    submit_check: Arc<AtomicU64>,
    marker: PhantomData<&'exe mut Execution>,
    time: timing::TimeAccountant,
    debug_mark: Option<String>,
}

/// Represents a stopped execution instance, without information abouts its outputs.
pub struct Retire<'pool> {
    /// The retiring execution instance.
    execution: Execution,
    pool: &'pool mut Pool,
    uncorrected_gpu_textures: Vec<TextureKey>,
    uncorrected_gpu_buffers: Vec<BufferKey>,
    uncorrected_shaders: Vec<ShaderKey>,
    uncorrected_pipelines: Vec<PipelineKey>,
}

pub(crate) struct Machine {
    instructions: Arc<[Low]>,
    instruction_pointer: Vec<Range<usize>>,
}

#[derive(Debug)]
pub struct StartError {
    kind: LaunchErrorKind,
}

impl core::fmt::Display for StartError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.kind)
    }
}

#[derive(Debug)]
pub enum LaunchErrorKind {
    FromLine(u32),
    MismatchedDescriptor {
        register: Register,
        expected: Descriptor,
        supplied: Descriptor,
    },
    MissingKey {
        register: Register,
        descriptor: Descriptor,
    },
}

#[derive(Debug)]
pub struct StepError {
    inner: StepErrorKind,
    instruction_pointer: usize,
}

#[derive(Debug)]
// Okay, as long as debug works.
#[allow(unused)]
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

#[derive(Default)]
struct Submissions {
    /// Did we submit to the device, i.e. if we want to sync can we `on_submitted_work_done` or
    /// not? If multi-device then this should become a set or map from gpu keys.
    submit: bool,
}

impl core::fmt::Display for BadInstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Bad Instruction: {:?}", self.inner)
    }
}

#[derive(Debug)]
pub struct RetireError {
    inner: RetireErrorKind,
}

impl core::fmt::Display for RetireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Execution inconsistent during retiring: {:?}",
            self.inner
        )
    }
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
            buffers: self
                .image_io_buffers
                .iter()
                .map(Image::clone_like)
                .collect(),
            knob_data: vec![],
            knobs: HashMap::default(),
            io_map: self.io_map.clone(),
            cache: Cache::default(),
        })
    }

    /// Produce a `dot` describing the pipeline.
    pub fn dot(&self) -> String {
        use core::fmt::Write;
        let mut cons = String::new();
        let mut queue_graph = String::new();

        let mut _bind_group_layouts = 0;
        let mut bind_groups = 0;
        let mut buffers = 0;
        let mut buffer_states = HashMap::new();
        let mut pipeline_layouts = 0;
        let mut samplers = 0;
        let mut shaders = 0;
        let mut textures = 0;
        let mut texture_states = HashMap::new();
        let mut texture_views = 0;
        let mut texture_view_map = HashMap::<usize, DeviceTexture>::new();
        let mut render_pipelines = 0;

        let mut commands = 0;
        let mut command_stack = vec![0usize; 0];
        let mut queue = 0;
        let mut render_passes = 0;

        for instr in &self.instructions[..] {
            match instr {
                Low::BindGroupLayout(_layout) => {
                    /*
                    let _ = write!(
                        &mut cons,
                        " bind_group_layout_{0} [label=\"Bind Group Layout {0}\"];",
                        _bind_group_layouts
                    );
                    */
                    _bind_group_layouts += 1;
                }
                Low::BindGroup(group) => {
                    let idx = bind_groups;
                    let _ = write!(
                        &mut cons,
                        " bind_group_{0} [label=\"Bind Group {0}\"];",
                        idx
                    );

                    /*
                    let _ = write!(
                        &mut cons,
                        " bind_group_layout_{} -> bind_group_{};",
                        group.layout_idx, idx
                    );
                    */
                    for entry in &group.entries {
                        match entry {
                            program::BindingResource::Buffer { buffer_idx, .. } => {
                                let _ = write!(
                                    &mut cons,
                                    " buffer_{} -> bind_group_{};",
                                    buffer_idx, idx
                                );
                                let st = buffer_states[buffer_idx];
                                let _ = write!(
                                    &mut cons,
                                    " buffer_{}_{} -> bind_group_{};",
                                    buffer_idx, st, idx
                                );
                            }
                            program::BindingResource::Sampler(sampler) => {
                                let _ = write!(
                                    &mut cons,
                                    " sampler_{} -> bind_group_{};",
                                    sampler, idx
                                );
                            }
                            program::BindingResource::TextureView(view) => {
                                let _ = write!(
                                    &mut cons,
                                    " texture_view_{} -> bind_group_{};",
                                    view, idx
                                );

                                if let Some(tex) = texture_view_map.get(&view) {
                                    let st = texture_states[&tex.0];
                                    let _ = write!(
                                        &mut cons,
                                        " texture_{}_{} -> bind_group_{};",
                                        tex.0, st, idx
                                    );
                                }
                            }
                        }
                    }

                    for entry in &group.sparse {
                        match entry.1 {
                            program::BindingResource::Buffer { buffer_idx, .. } => {
                                let _ = write!(
                                    &mut cons,
                                    " buffer_{} -> bind_group_{};",
                                    buffer_idx, idx
                                );
                                let st = buffer_states[&buffer_idx];
                                let _ = write!(
                                    &mut cons,
                                    " buffer_{}_{} -> bind_group_{};",
                                    buffer_idx, st, idx
                                );
                            }
                            program::BindingResource::Sampler(sampler) => {
                                let _ = write!(
                                    &mut cons,
                                    " sampler_{} -> bind_group_{};",
                                    sampler, idx
                                );
                            }
                            program::BindingResource::TextureView(view) => {
                                let _ = write!(
                                    &mut cons,
                                    " texture_view_{} -> bind_group_{};",
                                    view, idx
                                );

                                if let Some(tex) = texture_view_map.get(&view) {
                                    let st = texture_states[&tex.0];
                                    let _ = write!(
                                        &mut cons,
                                        " texture_{}_{} -> bind_group_{};",
                                        tex.0, st, idx
                                    );
                                }
                            }
                        }
                    }

                    bind_groups += 1;
                }
                Low::Buffer(_) => {
                    buffer_states.insert(buffers, 0);
                    buffers += 1;
                }
                Low::BufferInit(_) => {
                    let _ = write!(
                        &mut cons,
                        " buffer_{0} [label=\"Buffer {0} (init)\"];",
                        buffers
                    );
                    buffer_states.insert(buffers, 0);
                    buffers += 1;
                }
                Low::PipelineLayout(_layout) => {
                    let _ = write!(
                        &mut cons,
                        " pipeline_layout_{0} [label=\"Pipeline Layout {0}\"];",
                        pipeline_layouts
                    );
                    /*
                    for bg in &layout.bind_group_layouts {
                        let _ = write!(
                            &mut cons,
                            " bind_group_layout_{} -> pipeline_layout_{};",
                            bg, pipeline_layouts
                        );
                    }
                    */
                    pipeline_layouts += 1;
                }
                Low::Sampler(_) => {
                    let _ = write!(&mut cons, " sampler_{0} [label=\"Sampler {0}\"];", samplers);
                    samplers += 1;
                }
                Low::Shader(shader) => {
                    let _ = write!(
                        &mut cons,
                        " shader_{0} [label=\"Shader {0}: {1:?}\"];",
                        shaders, shader.key
                    );
                    shaders += 1;
                }
                Low::Texture(_) => {
                    texture_states.insert(textures, 0);
                    textures += 1;
                }
                Low::TextureView(view) => {
                    let _ = write!(
                        &mut cons,
                        " texture_view_{0} [label=\"Texture View {0}\"];",
                        texture_views
                    );
                    let _ = write!(
                        &mut cons,
                        " texture_{} -> texture_view_{};",
                        view.texture.0, texture_views
                    );
                    texture_view_map.insert(texture_views, view.texture);
                    texture_views += 1;
                }
                Low::RenderView(register) => {
                    let _ = write!(
                        &mut cons,
                        " texture_view_{0} [label=\"Render View {0}\"];",
                        texture_views
                    );
                    let _ = write!(
                        &mut cons,
                        " input_{} -> texture_view_{};",
                        register.0, texture_views
                    );
                    texture_views += 1;
                }
                Low::RenderPipeline(pipeline) => {
                    let idx = render_pipelines;
                    let _ = write!(
                        &mut cons,
                        " render_pipeline_{0} [label=\"Render Pipeline {0}\"];",
                        idx
                    );
                    let _ = write!(
                        &mut cons,
                        " pipeline_layout_{} -> render_pipeline_{};",
                        pipeline.layout, idx
                    );
                    let _ = write!(
                        &mut cons,
                        " shader_{} -> render_pipeline_{} [label=\"Fragment Shader\"];",
                        pipeline.fragment.fragment_module, idx
                    );
                    let _ = write!(
                        &mut cons,
                        " shader_{} -> render_pipeline_{} [label=\"Vertex Shader\"];",
                        pipeline.vertex.vertex_module, idx
                    );
                    render_pipelines += 1;
                }

                Low::BeginCommands => {
                    let _ = write!(
                        &mut cons,
                        " command_buffer_{0} [label=\"Command Buffer {0}\"];",
                        commands
                    );
                }
                Low::BeginRenderPass(pass) => {
                    let idx = render_passes;
                    let _ = write!(
                        &mut cons,
                        " render_pass_{0} [label=\"Render Pass {0}\"];",
                        idx
                    );
                    let _ = write!(
                        &mut cons,
                        " render_pass_{} -> command_buffer_{};",
                        idx, commands
                    );
                    for attach in &pass.color_attachments {
                        let _ = write!(
                            &mut cons,
                            " render_pass_{} -> texture_view_{};",
                            idx, attach.texture_view
                        );
                        if let Some(tex) = texture_view_map.get(&attach.texture_view) {
                            *texture_states.get_mut(&tex.0).unwrap() += 1;
                            let st = texture_states[&tex.0];
                            let _ = write!(
                                &mut cons,
                                " render_pass_{} -> texture_{}_{};",
                                idx, tex.0, st,
                            );
                        }
                    }
                }
                Low::EndCommands => {
                    command_stack.push(commands);
                    commands += 1;
                }
                Low::EndRenderPass => {
                    render_passes += 1;
                }
                Low::SetPipeline(pipeline) => {
                    let _ = write!(
                        &mut cons,
                        " render_pipeline_{} -> render_pass_{};",
                        pipeline, render_passes
                    );
                }
                Low::SetBindGroup { group, .. } => {
                    let _ = write!(
                        &mut cons,
                        " bind_group_{} -> render_pass_{};",
                        group, render_passes
                    );
                }
                Low::SetVertexBuffer { buffer, .. } => {
                    let _ = write!(
                        &mut cons,
                        " buffer_{}_{} -> render_pass_{};",
                        buffer, buffer_states[buffer], render_passes
                    );
                }
                Low::DrawOnce { vertices: _ } => {}
                Low::DrawIndexedZero { vertices: _ } => {}
                Low::SetPushConstants {
                    stages: _,
                    offset: _,
                    data: _,
                } => {}

                Low::RunTopCommand => {
                    let idx = queue;
                    let command = command_stack.pop().unwrap();
                    let _ = write!(&mut cons, " queue_{};", idx);
                    let _ = write!(&mut cons, " command_buffer_{} -> queue_{};", command, idx);
                    let _ = write!(&mut queue_graph, " queue_{} -> queue_{};", idx, idx + 1);
                    queue += 1;
                }
                Low::RunBotToTop(count) => {
                    let start = command_stack.len() - count;
                    for command in command_stack.drain(start..) {
                        let idx = queue;
                        let _ = write!(&mut cons, " queue_{};", idx);
                        let _ = write!(&mut cons, " command_buffer_{} -> queue_{};", command, idx);
                        let _ = write!(&mut queue_graph, " queue_{} -> queue_{};", idx, idx + 1);
                        queue += 1;
                    }
                }

                Low::WriteImageToBuffer {
                    source_image,
                    offset: _,
                    size: _,
                    target_buffer,
                    target_layout: _,
                    copy_dst_buffer,
                    write_event: _,
                } => {
                    let idx = queue;
                    let _ = write!(&mut cons, " queue_{};", idx);
                    let _ = write!(&mut cons, " register_{} -> queue_{};", source_image.0, idx);
                    *buffer_states.get_mut(&target_buffer.0).unwrap() += 1;
                    let st = buffer_states[&target_buffer.0];
                    let _ = write!(
                        &mut cons,
                        " queue_{} -> buffer_{}_{};",
                        idx, target_buffer.0, st
                    );
                    *buffer_states.get_mut(&copy_dst_buffer.0).unwrap() += 1;
                    let st = buffer_states[&copy_dst_buffer.0];
                    let _ = write!(
                        &mut cons,
                        " queue_{} -> buffer_{}_{};",
                        idx, copy_dst_buffer.0, st
                    );
                    let _ = write!(&mut queue_graph, " queue_{} -> queue_{};", idx, idx + 1);
                    queue += 1;
                }
                Low::CopyBufferToTexture {
                    source_buffer,
                    source_layout: _,
                    offset: _,
                    size: _,
                    target_texture,
                } => {
                    let idx = queue;
                    let _ = write!(&mut cons, " queue_{};", idx);
                    let st = buffer_states[&source_buffer.0];
                    let _ = write!(
                        &mut cons,
                        " buffer_{}_{} -> queue_{};",
                        source_buffer.0, st, idx
                    );
                    *texture_states.get_mut(&target_texture.0).unwrap() += 1;
                    let st = texture_states[&target_texture.0];
                    let _ = write!(
                        &mut cons,
                        " queue_{} -> texture_{}_{};",
                        idx, target_texture.0, st
                    );
                    let _ = write!(&mut queue_graph, " queue_{} -> queue_{};", idx, idx + 1);
                    queue += 1;
                }
                Low::CopyTextureToBuffer {
                    source_texture,
                    offset: _,
                    size: _,
                    target_buffer,
                    target_layout: _,
                } => {
                    let idx = queue;
                    let _ = write!(&mut cons, " queue_{};", idx);
                    let st = texture_states[&source_texture.0];
                    let _ = write!(
                        &mut cons,
                        " texture_{}_{} -> queue_{};",
                        source_texture.0, st, idx
                    );
                    *buffer_states.get_mut(&target_buffer.0).unwrap() += 1;
                    let st = buffer_states[&target_buffer.0];
                    let _ = write!(
                        &mut cons,
                        " queue_{} -> buffer_{}_{};",
                        idx, target_buffer.0, st
                    );
                    let _ = write!(&mut queue_graph, " queue_{} -> queue_{};", idx, idx + 1);
                    queue += 1;
                }
                Low::CopyBufferToBuffer {
                    source_buffer,
                    size: _,
                    target_buffer,
                } => {
                    let idx = queue;
                    let _ = write!(&mut cons, " queue_{};", idx);
                    let st = buffer_states[&source_buffer.0];
                    let _ = write!(
                        &mut cons,
                        " buffer_{}_{} -> queue_{};",
                        source_buffer.0, st, idx
                    );
                    *buffer_states.get_mut(&target_buffer.0).unwrap() += 1;
                    let st = buffer_states[&target_buffer.0];
                    let _ = write!(
                        &mut cons,
                        " queue_{} -> buffer_{}_{};",
                        idx, target_buffer.0, st
                    );
                    let _ = write!(&mut queue_graph, " queue_{} -> queue_{};", idx, idx + 1);
                    queue += 1;
                }
                Low::ReadBuffer {
                    source_buffer,
                    source_layout: _,
                    offset: _,
                    size: _,
                    target_image,
                    copy_src_buffer,
                } => {
                    let idx = queue;
                    let _ = write!(&mut cons, " queue_{};", idx);
                    let st = buffer_states[&source_buffer.0];
                    let _ = write!(
                        &mut cons,
                        " buffer_{}_{} -> queue_{};",
                        source_buffer.0, st, idx
                    );
                    let st = buffer_states[&copy_src_buffer.0];
                    let _ = write!(
                        &mut cons,
                        " buffer_{}_{} -> queue_{};",
                        copy_src_buffer.0, st, idx
                    );
                    let _ = write!(&mut cons, " queue_{} -> register_{};", idx, target_image.0);
                    let _ = write!(&mut queue_graph, " queue_{} -> queue_{};", idx, idx + 1);
                    queue += 1;
                }

                Low::StackFrame(_) => {}
                Low::StackPop => {}

                Low::Call { .. } => {
                    todo!()
                }

                _ => {}
            }
        }

        let texture_graphs = texture_states
            .into_iter()
            .map(|(id, count)| {
                let mut inner = String::new();
                let _ = write!(&mut cons, " texture_{0} [label=\"Texture {0}\"];", id);
                let _ = write!(&mut inner, " texture_{} -> texture_{}_0;", id, id);
                for i in 0..=count {
                    let _ = write!(&mut inner, " texture_{}_{};", id, i);
                }
                for i in (0..=count).skip(1) {
                    let _ = write!(
                        &mut inner,
                        " texture_{}_{} -> texture_{}_{};",
                        id,
                        i - 1,
                        id,
                        i
                    );
                }

                format!("subgraph clusterTexture{} {{ {} }}", id, inner)
            })
            .collect::<String>();

        let buffer_graphs = buffer_states
            .into_iter()
            .map(|(id, count)| {
                let mut inner = String::new();
                let _ = write!(&mut cons, " buffer_{0} [label=\"Buffer {0}\"];", id);
                let _ = write!(&mut inner, " buffer_{} -> buffer_{}_0;", id, id);
                for i in 0..=count {
                    let _ = write!(&mut inner, " buffer_{}_{};", id, i);
                }
                for i in (0..=count).skip(1) {
                    let _ = write!(
                        &mut inner,
                        " buffer_{}_{} -> buffer_{}_{};",
                        id,
                        i - 1,
                        id,
                        i
                    );
                }

                format!("subgraph clusterBuffer{} {{ {} }}", id, inner)
            })
            .collect::<String>();

        let queue_graph = format!("subgraph clusterA {{ {} }}", queue_graph);

        format!(
            "digraph exe {{ {} {} {} {} }} ",
            cons, queue_graph, texture_graphs, buffer_graphs
        )
    }

    pub fn query_knob(&self, knob: RegisterKnob) -> Option<Knob> {
        self.info.knobs.get(&knob).copied()
    }

    pub fn launch(&self, mut env: Environment) -> Result<Execution, StartError> {
        log::info!("Instructions {:#?}", self.instructions);
        self.check_satisfiable(&mut env)?;
        env.gpu.device().start_capture();

        Ok(Execution {
            gpu: env.gpu.into(),
            gpu_key: env.gpu_key,
            host: Host {
                machine: Machine::new(self),
                descriptors: Descriptors {
                    image_io_buffers: env.buffers,
                    ..Descriptors::default()
                },
                command_encoder: None,
                binary_data: self.binary_data.clone(),
                io_map: self.io_map.clone(),
                knob_data: env.knob_data.into(),
                knobs: env.knobs.into(),
                info: self.info.clone(),
                call_stack: vec![],
                debug_stack: vec![],
                usage: ResourcesUsed::default(),
                delayed_submits: 0,
                debug: Debug::default(),
            },
            cache: env.cache,
        })
    }

    /// Run the executable but take all by value.
    pub fn launch_once(self, mut env: Environment) -> Result<Execution, StartError> {
        self.check_satisfiable(&mut env)?;
        env.gpu.device().start_capture();

        Ok(Execution {
            gpu: env.gpu.into(),
            gpu_key: env.gpu_key,
            host: Host {
                machine: Machine::new(&self),
                descriptors: Descriptors {
                    image_io_buffers: env.buffers,
                    ..self.descriptors
                },
                command_encoder: None,
                binary_data: self.binary_data,
                io_map: self.io_map.clone(),
                knob_data: env.knob_data.into(),
                knobs: env.knobs.into(),
                info: self.info.clone(),
                call_stack: vec![],
                debug_stack: vec![],
                usage: ResourcesUsed::default(),
                delayed_submits: 0,
                debug: Debug::default(),
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
            let reference = &self.image_io_buffers[input];

            if reference.data.layout() != buffer.data.layout() {
                return Err(StartError::InternalCommandError(line!()));
            }

            if reference.descriptor != buffer.descriptor {
                return Err(StartError {
                    kind: LaunchErrorKind::MismatchedDescriptor {
                        register: Register(input),
                        expected: reference.descriptor.clone(),
                        supplied: buffer.descriptor.clone(),
                    },
                });
            }

            // Oh, this image is always already bound? Cool.
            if !matches!(buffer.data, ImageData::LateBound(_)) {
                continue;
            }

            let Some(key) = buffer.key else {
                return Err(StartError {
                    kind: LaunchErrorKind::MissingKey {
                        register: Register(input),
                        descriptor: buffer.descriptor.clone(),
                    },
                });
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
                pool_img.swap_image(&mut buffer.data);
            } else {
                env.buffers[output].data.host_allocate();
            }
        }

        for &render in self.io_map.renders.values() {
            if let Some(key) = env.buffers[render].key {
                let mut pool_img = env.pool.entry(key).unwrap();
                let buffer = &mut env.buffers[render];
                pool_img.swap_image(&mut buffer.data);
            } else {
                todo!();
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
            ImageData::Host(_) | ImageData::GpuTexture { .. } | ImageData::GpuBuffer { .. } => {}
            _ => {
                return Err(StartError::InternalCommandError(line!()));
            }
        }

        image.key = Some(pool_img.key());

        Ok(())
    }

    pub fn bind_output(&mut self, reg: Register, key: PoolKey) -> Result<(), StartError> {
        // FIXME: duplication with `bind`.
        let &idx = self
            .io_map
            .outputs
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
            ImageData::Host(_) | ImageData::GpuTexture { .. } | ImageData::GpuBuffer { .. } => {}
            _ => {
                return Err(StartError::InternalCommandError(line!()));
            }
        }

        image.key = Some(pool_img.key());

        Ok(())
    }

    pub fn bind_render(&mut self, reg: Register, key: PoolKey) -> Result<(), StartError> {
        // FIXME: duplication with `bind_output`.
        let &idx = self
            .io_map
            .renders
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
            ImageData::GpuTexture { .. } => {}
            _ => {
                return Err(StartError::InternalCommandError(line!()));
            }
        }

        image.key = Some(pool_img.key());

        Ok(())
    }

    /// Define the knob data for this run, by register.
    #[track_caller]
    pub fn knob_by_register(&mut self, knob: &RegisterKnob, data: &[u8]) -> Result<(), StartError> {
        let knob = self
            .info
            .knobs
            .get(knob)
            .expect("Knob does not exist in this program");
        self.knob(*knob, data)
    }

    pub fn knob(&mut self, knob: Knob, data: &[u8]) -> Result<(), StartError> {
        let desc = &self.info.knob_descriptors[&knob];

        if data.len() != desc.range.len() {
            // FIXME: better error!
            return Err(StartError::InternalCommandError(line!()));
        }

        let start = self.knob_data.len();
        self.knob_data.extend_from_slice(data);
        let end = self.knob_data.len();
        self.knobs.insert(knob, start..end);

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

        for (&inst, desc) in &self.info.shader_by_op {
            if let Some(buffer) = pool_cache.extract_shader(desc) {
                self.cache.precompiled_shader.insert(inst, buffer);
            }
        }

        for (&inst, desc) in &self.info.pipeline_by_op {
            if let Some(pipeline) = pool_cache.extract_pipeline(desc) {
                self.cache.precompiled_pipelines.insert(inst, pipeline);
            }
        }

        stats
    }
}

impl Execution {
    pub(crate) fn new(init: InitialState) -> Self {
        init.device.start_capture();

        let range = init.info.functions[&init.entry_point].range.clone();
        Execution {
            gpu: Gpu::new(init.device, init.queue).into(),
            gpu_key: None,
            host: Host {
                machine: Machine::with_instructions(init.instructions, range),
                descriptors: Descriptors {
                    image_io_buffers: init.image_io_buffers,
                    ..Descriptors::default()
                },
                command_encoder: None,
                binary_data: init.binary_data,
                io_map: init.io_map,
                knob_data: Arc::default(),
                knobs: Default::default(),
                info: init.info,
                call_stack: vec![],
                debug_stack: vec![],
                usage: ResourcesUsed::default(),
                delayed_submits: 0,
                debug: Debug::default(),
            },
            cache: Cache::default(),
        }
    }

    /// Check if the machine is still running.
    pub fn is_running(&self) -> bool {
        self.host.machine.is_running()
    }

    /// Do a single step of the program.
    ///
    /// Realize that this can be expensive due to the extra synchronization.
    pub fn step(&mut self) -> Result<SyncPoint<'_>, StepError> {
        self.step_to(StepLimits { instructions: 1 })
    }

    /// Do a number of limited steps.
    pub fn step_to(&mut self, limits: StepLimits) -> Result<SyncPoint<'_>, StepError> {
        let instruction_pointer = self
            .host
            .machine
            .instruction_pointer
            .last()
            .map_or(usize::MAX, |range| range.start);

        let time = timing::TimeAccountant::from_now();

        let debug_mark = self
            .host
            .machine
            .instructions
            .get(instruction_pointer)
            .map(|instr| format!("{:?}", instr));

        let Execution {
            ref gpu,
            gpu_key: _,
            host,
            cache,
        } = self;

        let submit_flag: Arc<AtomicU64> = Arc::default();
        let submit_check = submit_flag.clone();

        let async_step = async move {
            let mut limits = limits;

            loop {
                match host.step_inner(cache, gpu, &mut limits).await {
                    Err(StepError {
                        inner: StepErrorKind::ProgramEnd,
                        ..
                    }) => break,
                    Err(mut error) => {
                        // Add tracing information..
                        error.instruction_pointer = instruction_pointer;
                        return Err(error);
                    }
                    Ok(submission) => {
                        if submission.submit {
                            submit_flag.fetch_add(1, Ordering::Release);
                        }
                    }
                }

                if limits.is_exhausted() {
                    break;
                }

                if !host.machine.is_running() {
                    break;
                }
            }

            Ok(())
        };

        Ok(SyncPoint {
            future: Some(DevicePolled {
                future: Box::pin(async_step),
                gpu: self.gpu.clone(),
            }),
            submit_check,
            marker: PhantomData,
            time,
            debug_mark,
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
        self.gpu.with_gpu(|gpu| gpu.device().stop_capture());

        Retire {
            execution: self,
            pool,
            uncorrected_gpu_textures: vec![],
            uncorrected_gpu_buffers: vec![],
            uncorrected_shaders: vec![],
            uncorrected_pipelines: vec![],
        }
    }

    /// Debug how many resources were used, any how.
    pub fn resources_used(&self) -> &ResourcesUsed {
        &self.host.usage
    }
}

impl Host {
    async fn step_inner(
        &mut self,
        cache: &mut Cache,
        gpu: &Gpu,
        limits: &mut StepLimits,
    ) -> Result<Submissions, StepError> {
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

        let (inst, low) = self.machine.next_instruction()?;
        if let Some(event) = self.info.skip_by_op.get(&inst) {
            if self.descriptors.precomputed.contains_key(event) {
                // FIXME: Incorrect in general. We need to simulate the creation of whatever
                // resource it was meant to create / grab this resource from the environment.
                // Otherwise, the index-tracking of other resources gets messed up as the encoder
                // relies on this being essentially a stack machine.
                return Ok(Submissions::default());
            }
        }

        limits.instructions = limits.instructions.saturating_sub(1);

        match low {
            Low::BindGroupLayout(desc) => {
                let mut entry_buffer = vec![];
                let group = self
                    .descriptors
                    .bind_group_layout(desc, &mut entry_buffer)?;
                // eprintln!("Made {}: {:?}", self.descriptors.bind_group_layouts.len(), group);
                let group = gpu.with_gpu(|gpu| gpu.device().create_bind_group_layout(&group));
                self.descriptors.bind_group_layouts.push(group);
                Ok(Submissions::default())
            }
            Low::BindGroup(desc) => {
                let mut entry_buffer = vec![];
                let group =
                    self.descriptors
                        .bind_group(desc, &mut entry_buffer, &mut self.debug)?;
                // eprintln!("{}: {:?}", desc.layout_idx, group);
                let group = gpu.with_gpu(|gpu| gpu.device().create_bind_group(&group));
                self.descriptors.bind_groups.push(group);
                Ok(Submissions::default())
            }
            Low::Buffer(desc) => {
                let wgpu_desc = wgpu::BufferDescriptor {
                    label: None,
                    size: desc.size,
                    usage: desc.usage.to_wgpu(),
                    mapped_at_creation: false,
                };

                let buffer = if let Some(buffer) = cache.preallocated_buffers.remove(&inst.0) {
                    self.usage.buffer_reused += desc.u64_len();
                    buffer
                } else {
                    self.usage.buffer_mem += desc.u64_len();
                    gpu.with_gpu(|gpu| gpu.device().create_buffer(&wgpu_desc))
                };

                let buffer_idx = self.descriptors.buffers.len();
                self.debug
                    .buffer_use(DeviceBuffer(buffer_idx), TextureInitState::Uninit);
                self.descriptors
                    .buffer_descriptors
                    .insert(buffer_idx, desc.clone());
                self.descriptors.buffers.push(Arc::new(buffer));
                Ok(Submissions::default())
            }
            Low::BufferInit(desc) => {
                use wgpu::util::DeviceExt;

                let knob_range = if let Some(knob) = self.info.knob_starts.get(&desc.content.start)
                {
                    let kdesc = &self.info.knob_descriptors[knob];
                    assert_eq!(&kdesc.range, &desc.content, "Unhandled encoding error");
                    self.knobs.get(knob)
                } else {
                    None
                };

                let contents = if let Some(knob_range) = knob_range {
                    &self.knob_data[knob_range.clone()]
                } else {
                    self.binary_data
                        .get(desc.content.clone())
                        .ok_or_else(|| StepError::InvalidInstruction(line!()))?
                };

                let wgpu_desc = wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents,
                    usage: desc.usage.to_wgpu(),
                };

                let buffer_idx = self.descriptors.buffers.len();
                self.debug
                    .buffer_use(DeviceBuffer(buffer_idx), TextureInitState::WriteTo);

                self.usage.buffer_mem += desc.u64_len();
                let buffer = gpu.with_gpu(|gpu| gpu.device().create_buffer_init(&wgpu_desc));
                self.descriptors.buffers.push(Arc::new(buffer));
                Ok(Submissions::default())
            }
            Low::Shader(desc) => {
                let shader;
                if std::env::var("ZOSIMOS_PASSTHROUGH").is_err() {
                    let wgpu_desc = wgpu::ShaderModuleDescriptor {
                        label: Some(desc.name),
                        source: wgpu::ShaderSource::SpirV(desc.source_spirv.as_ref().into()),
                    };

                    shader = if let Some(shader) = cache.precompiled_shader.remove(&inst.0) {
                        self.usage.shaders_reused += 1;
                        shader
                    } else {
                        self.usage.shaders_compiled += 1;
                        gpu.with_gpu(|gpu| gpu.device().create_shader_module(wgpu_desc))
                    };
                } else {
                    let wgpu_desc = wgpu::ShaderModuleDescriptor {
                        label: Some(desc.name),
                        source: wgpu::ShaderSource::SpirV(desc.source_spirv.as_ref().into()),
                    };

                    shader = if let Some(shader) = cache.precompiled_shader.remove(&inst.0) {
                        self.usage.shaders_reused += 1;
                        shader
                    } else {
                        self.usage.shaders_compiled += 1;
                        gpu.with_gpu(|gpu| gpu.device().create_shader_module(wgpu_desc))
                    };
                };

                if let Some(key) = &desc.key {
                    let shader_idx = self.descriptors.shaders.len();
                    self.descriptors
                        .shader_descriptors
                        .insert(shader_idx, key.clone());
                }

                self.descriptors.shaders.push(shader);
                Ok(Submissions::default())
            }
            Low::PipelineLayout(desc) => {
                let mut entry_buffer = vec![];
                let layout = self.descriptors.pipeline_layout(desc, &mut entry_buffer)?;
                let layout = gpu.with_gpu(|gpu| gpu.device().create_pipeline_layout(&layout));
                self.descriptors.pipeline_layouts.push(layout);
                Ok(Submissions::default())
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
                    anisotropy_clamp: 1,
                    // FIXME(webGL2): on webGL2 a non-None *panics*.
                    // This is due to wgpu_hal-gles using sampler_parameter_f32_slice which panics
                    // in its implementation in glow's webGL1/2.
                    border_color: desc.border_color,
                };
                let sampler = gpu.with_gpu(|gpu| gpu.device().create_sampler(&desc));
                self.descriptors.sampler.push(sampler);
                Ok(Submissions::default())
            }
            Low::TextureView(desc) => {
                let texture = self
                    .descriptors
                    .textures
                    .get(desc.texture.0)
                    .ok_or_else(|| StepError::InvalidInstruction(line!()))?;

                self.debug
                    .texture_view(self.descriptors.texture_views.len(), desc.texture);

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
                Ok(Submissions::default())
            }
            Low::RenderView(source_image) => {
                let texture = match &mut self.descriptors.image_io_buffers[source_image.0].data {
                    ImageData::GpuTexture {
                        texture,
                        // FIXME: validate layout? What for?
                        layout: _,
                        gpu: _,
                    } => texture,
                    _ => return Err(StepError::InvalidInstruction(line!())),
                };

                self.debug.image_view(self.descriptors.texture_views.len());

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
                Ok(Submissions::default())
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
                            U::COPY_SRC | U::COPY_DST | U::TEXTURE_BINDING | U::RENDER_ATTACHMENT
                        }
                        program::TextureUsage::Transient => {
                            U::TEXTURE_BINDING | U::RENDER_ATTACHMENT
                        }
                    },
                    view_formats: &[desc.format],
                };

                let texture = if let Some(texture) = cache.preallocated_textures.remove(&inst.0) {
                    self.usage.texture_reused += desc.u64_len();
                    texture
                } else {
                    self.usage.texture_mem += desc.u64_len();
                    gpu.with_gpu(|gpu| gpu.device().create_texture(&wgpu_desc))
                };

                let texture_idx = self.descriptors.textures.len();

                self.descriptors
                    .texture_descriptors
                    .insert(texture_idx, desc.clone());

                self.debug.texture_use(
                    DeviceTexture(self.descriptors.textures.len()),
                    TextureInitState::default(),
                );

                self.descriptors.textures.push(texture);
                Ok(Submissions::default())
            }
            Low::RenderPipeline(desc) => {
                let pipeline = if let Some(pipeline) = cache.precompiled_pipelines.remove(&inst.0) {
                    self.usage.pipelines_reused += 1;
                    pipeline
                } else {
                    self.usage.pipelines_compiled += 1;

                    let mut vertex_buffers = vec![];
                    let mut fragments = vec![];

                    let pipeline =
                        self.descriptors
                            .pipeline(desc, &mut vertex_buffers, &mut fragments)?;
                    gpu.with_gpu(|gpu| gpu.device().create_render_pipeline(&pipeline))
                };

                let pipeline_idx = self.descriptors.render_pipelines.len();
                // Pipelines are not cacheable (at least we don't assume so) from their descriptor
                // alone which may refer to some arbitrary module instances and state that are not
                // easily summarized. So, only cache this if the encoder generated a key for it.
                if let Some(desc) = self.info.pipeline_by_op.get(&inst.0) {
                    self.descriptors
                        .pipeline_descriptors
                        .insert(pipeline_idx, desc.clone());
                }

                self.descriptors.render_pipelines.push(pipeline);
                Ok(Submissions::default())
            }
            Low::BeginCommands => {
                if self.command_encoder.is_some() {
                    return Err(StepError::InvalidInstruction(line!()));
                }

                let descriptor = wgpu::CommandEncoderDescriptor { label: None };

                let encoder = gpu.with_gpu(|gpu| gpu.device().create_command_encoder(&descriptor));
                self.command_encoder = Some(encoder);
                Ok(Submissions::default())
            }
            Low::BeginRenderPass(descriptor) => {
                let mut attachment_buf = vec![];
                let descriptor = self.descriptors.prepare_attachments_for_render_pass(
                    descriptor,
                    &mut attachment_buf,
                    &mut self.debug,
                )?;
                let encoder = match &mut self.command_encoder {
                    Some(encoder) => encoder,
                    None => return Err(StepError::InvalidInstruction(line!())),
                };

                let pass = encoder.begin_render_pass(&descriptor);
                drop(attachment_buf);
                self.machine.render_pass(&self.descriptors, pass)?;

                Ok(Submissions::default())
            }
            Low::EndCommands => match self.command_encoder.take() {
                None => Err(StepError::InvalidInstruction(line!())),
                Some(encoder) => {
                    self.descriptors.command_buffers.push(encoder.finish());
                    Ok(Submissions::default())
                }
            },
            &Low::RunTopCommand if self.delayed_submits == 0 => {
                let command = self
                    .descriptors
                    .command_buffers
                    .pop()
                    .ok_or_else(|| StepError::InvalidInstruction(line!()))?;
                gpu.with_gpu(|gpu| gpu.queue().submit(once(command)));
                Ok(Submissions { submit: true })
            }
            &Low::RunTopCommand => {
                let many = 1 + self.delayed_submits;
                if let Some(top) = self.descriptors.command_buffers.len().checked_sub(many) {
                    let commands = self.descriptors.command_buffers.drain(top..);
                    gpu.with_gpu(|gpu| gpu.queue().submit(commands));
                    self.delayed_submits = 0;
                } else {
                    return Err(StepError::InvalidInstruction(line!()));
                }

                Ok(Submissions { submit: true })
            }
            &Low::RunBotToTop(many) => {
                let many = many + self.delayed_submits;
                if let Some(top) = self.descriptors.command_buffers.len().checked_sub(many) {
                    let commands = self.descriptors.command_buffers.drain(top..);
                    gpu.with_gpu(|gpu| gpu.queue().submit(commands));
                    self.delayed_submits = 0;
                } else {
                    return Err(StepError::InvalidInstruction(line!()));
                }

                Ok(Submissions { submit: true })
            }
            &Low::WriteImageToBuffer {
                source_image,
                offset,
                size,
                target_buffer,
                ref target_layout,
                copy_dst_buffer,
                write_event,
            } => {
                if offset != (0, 0) {
                    return Err(StepError::InvalidInstruction(line!()));
                }

                let source = match self.descriptors.image_io_buffers.get(source_image.0) {
                    None => return Err(StepError::InvalidInstruction(line!())),
                    Some(source) => &source.data,
                };

                self.debug
                    .buffer_use(target_buffer, TextureInitState::WriteTo);

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

                let (width, _height) = target_size;
                let bytes_per_row = target_layout.row_stride;
                let bytes_per_texel = target_layout.texel_stride;
                let _bytes_to_copy = (u32::from(bytes_per_texel) * width) as usize;

                let image = &mut self.descriptors.image_io_buffers[source_image.0].data;

                if let ImageData::GpuTexture {
                    texture,
                    // FIXME: validate layout? What for?
                    layout: _,
                    gpu: _,
                } = image
                {
                    let descriptor = wgpu::CommandEncoderDescriptor { label: None };
                    let mut encoder =
                        gpu.with_gpu(|gpu| gpu.device().create_command_encoder(&descriptor));

                    encoder.copy_texture_to_buffer(
                        texture.as_image_copy(),
                        wgpu::ImageCopyBufferBase {
                            buffer: &self.descriptors.buffers[copy_dst_buffer.0],
                            layout: wgpu::ImageDataLayout {
                                bytes_per_row: Some(bytes_per_row as u32),
                                offset: 0,
                                rows_per_image: Some(size.1),
                            },
                        },
                        wgpu::Extent3d {
                            width: size.0,
                            height: size.1,
                            depth_or_array_layers: 1,
                        },
                    );

                    let command = encoder.finish();
                    self.descriptors.command_buffers.push(command);
                    self.delayed_submits += 1;

                    self.descriptors
                        .precomputed
                        .insert(write_event, Precomputed);

                    return Ok(Submissions::default());
                } else if let ImageData::GpuBuffer {
                    buffer: src_buffer,
                    // FIXME: validate layout? What for?
                    layout: _,
                    gpu: _,
                } = image
                {
                    let descriptor = wgpu::CommandEncoderDescriptor { label: None };
                    let mut encoder =
                        gpu.with_gpu(|gpu| gpu.device().create_command_encoder(&descriptor));

                    encoder.copy_buffer_to_buffer(
                        &*src_buffer,
                        0,
                        &self.descriptors.buffers[copy_dst_buffer.0],
                        0,
                        layout.u64_len(),
                    );

                    let command = encoder.finish();
                    self.descriptors.command_buffers.push(command);
                    self.delayed_submits += 1;

                    self.descriptors
                        .precomputed
                        .insert(write_event, Precomputed);

                    return Ok(Submissions::default());
                }

                if image.as_bytes().is_none() {
                    return Err(StepError::InvalidInstruction(line!()));
                }

                // The buffer we copy to is specifically for uploading this argument.
                let buffer = &self.descriptors.buffers[target_buffer.0];
                let slice = buffer.slice(..);
                let (ping, waker) = Ping::new(Box::new(|| wgpu::BufferAsyncError));
                slice.map_async(wgpu::MapMode::Write, |res| waker.complete(res.is_ok()));

                ping.await
                    .map_err(|wgpu::BufferAsyncError| StepError::InvalidInstruction(line!()))?;

                // eprintln!("WriteImageToBuffer");
                // eprintln!(" Source: {:?}", source_image.0);
                // eprintln!(" Target: {:?}", target_buffer.0);
                let mut data = slice.get_mapped_range_mut();

                // We've checked that this image can be seen as host bytes.
                let source: &[u8] = image.as_bytes().unwrap();
                let target: &mut [u8] = &mut data[..];
                copy_host_to_buffer(source, target, image.layout(), *target_layout);

                drop(data);
                buffer.unmap();

                Ok(Submissions::default())
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

                self.debug
                    .texture_use(target_texture, TextureInitState::WriteTo);
                self.debug
                    .buffer_use(source_buffer, TextureInitState::UseRead);

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

                Ok(Submissions::default())
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

                self.debug
                    .texture_use(source_texture, TextureInitState::UseRead);
                self.debug
                    .buffer_use(target_buffer, TextureInitState::WriteTo);

                let texture = self.descriptors.texture(source_texture)?;
                let buffer = self.descriptors.buffer(target_buffer, target_layout)?;

                let extent = wgpu::Extent3d {
                    width: size.0,
                    height: size.1,
                    depth_or_array_layers: 1,
                };

                encoder.copy_texture_to_buffer(texture, buffer, extent);

                Ok(Submissions::default())
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

                self.debug
                    .buffer_use(source_buffer, TextureInitState::UseRead);
                self.debug
                    .buffer_use(target_buffer, TextureInitState::WriteTo);

                // eprintln!("CopyBufferToBuffer");
                // eprintln!(" Source: {:?}", source_buffer.0);
                // eprintln!(" Target: {:?}", target_buffer.0);
                // eprintln!(" Size: {:?}", size);

                encoder.copy_buffer_to_buffer(source, 0, target, 0, size);

                Ok(Submissions::default())
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

                self.debug
                    .buffer_use(source_buffer, TextureInitState::ReadFrom);

                let bytes_per_row = source_layout.row_stride;
                let bytes_per_texel = source_layout.texel_stride;
                let (width, height) = size;
                let bytes_to_copy = (u32::from(bytes_per_texel) * width) as usize;

                let buffer = &self.descriptors.buffers[source_buffer.0];
                let image = &mut self.descriptors.image_io_buffers[target_image.0].data;

                if let ImageData::GpuTexture {
                    texture,
                    // FIXME: validate layout? What for?
                    layout: _,
                    gpu: _,
                } = image
                {
                    let descriptor = wgpu::CommandEncoderDescriptor { label: None };
                    let mut encoder =
                        gpu.with_gpu(|gpu| gpu.device().create_command_encoder(&descriptor));

                    encoder.copy_buffer_to_texture(
                        wgpu::ImageCopyBufferBase {
                            buffer: &self.descriptors.buffers[copy_src_buffer.0],
                            layout: wgpu::ImageDataLayout {
                                bytes_per_row: Some(bytes_per_row as u32),
                                offset: 0,
                                rows_per_image: Some(size.1),
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
                    gpu.with_gpu(|gpu| gpu.queue().submit(once(command)));

                    return Ok(Submissions { submit: true });
                } else if let ImageData::GpuBuffer {
                    buffer,
                    layout,
                    gpu: _,
                } = image
                {
                    let descriptor = wgpu::CommandEncoderDescriptor { label: None };
                    let mut encoder =
                        gpu.with_gpu(|gpu| gpu.device().create_command_encoder(&descriptor));

                    encoder.copy_buffer_to_buffer(
                        &self.descriptors.buffers[copy_src_buffer.0],
                        0,
                        buffer,
                        0,
                        layout.u64_len(),
                    );

                    let command = encoder.finish();
                    gpu.with_gpu(|gpu| gpu.queue().submit(once(command)));

                    return Ok(Submissions { submit: true });
                }

                if image.as_bytes().is_none() {
                    return Err(StepError::InvalidInstruction(line!()));
                }

                let slice = buffer.slice(..);
                let (ping, waker) = Ping::new(Box::new(|| wgpu::BufferAsyncError));
                slice.map_async(wgpu::MapMode::Read, |res| waker.complete(res.is_ok()));

                ping.await
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

                Ok(Submissions::default())
            }
            Low::StackFrame(frame) => {
                if let Some(ref mut frames) = _dump_on_panic.stack {
                    frames.push(frame.clone());
                }

                Ok(Submissions::default())
            }
            Low::StackPop => {
                if let Some(ref mut frames) = _dump_on_panic.stack {
                    let _ = frames.pop();
                }

                Ok(Submissions::default())
            }
            Low::Call {
                function,
                io_buffers,
            } => {
                let function = *function;
                let fn_info = &self.info.functions[&function];

                let mut new_io: Vec<_> = fn_info.io.iter().map(Image::with_late_bound).collect();

                for argument in io_buffers {
                    let buffer = self.descriptors.buffers[argument.buffer.0].clone();
                    let layout = argument.descriptor.to_canvas();

                    let io_texture = if let Some(idx) = fn_info.io_map.inputs.get(&argument.in_io) {
                        *idx
                    } else if let Some(idx) = fn_info.io_map.outputs.get(&argument.in_io) {
                        *idx
                    } else {
                        return Err(StepError::InvalidInstruction(line!()));
                    };

                    let matches_descriptor = new_io
                        .get(io_texture)
                        .map_or(false, |expected| expected.descriptor == argument.descriptor);

                    if !matches_descriptor {
                        return Err(StepError::InvalidInstruction(line!()));
                    }

                    new_io[io_texture] = Image {
                        data: ImageData::GpuBuffer {
                            gpu: slotmap::DefaultKey::default(),
                            layout,
                            buffer,
                        },
                        descriptor: argument.descriptor.clone(),
                        key: None,
                    };
                }

                let stack_descriptors = core::mem::take(&mut self.descriptors);
                self.descriptors.image_io_buffers = new_io;
                self.call_stack.push(stack_descriptors);

                // alright also we need to activate this new stack frame. And then it resumes
                // control at the next instruction after this (which will clean up the ABI pass of
                // the buffer).
                let instruction_range = fn_info.range.clone();
                self.machine.instruction_pointer.push(instruction_range);

                Ok(Submissions::default())
            }
            Low::Return => {
                // A bit questionable. We do not return from the entry point function.. That'll at
                // least preserve correctness if it is called recursively. (Although we do not have
                // any form of jump at the moment of writing so that is a bug).
                if self.machine.instruction_pointer.len() > 1 {
                    self.machine.instruction_pointer.pop();

                    // Drop existing descriptors, replace with previous call frame.
                    self.descriptors = self
                        .call_stack
                        .pop()
                        .ok_or_else(|| StepError::InvalidInstruction(line!()))?;
                }

                Ok(Submissions::default())
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
        debug: &mut Debug,
    ) -> Result<wgpu::BindGroupDescriptor<'set>, StepError> {
        buf.clear();

        for (idx, entry) in desc.entries.iter().enumerate() {
            let resource = self.binding_resource(entry, debug)?;
            buf.push(wgpu::BindGroupEntry {
                binding: idx as u32,
                resource,
            });
        }

        for &(idx, ref entry) in desc.sparse.iter() {
            let resource = self.binding_resource(entry, debug)?;
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
        debug: &mut Debug,
    ) -> Result<wgpu::BindingResource<'_>, StepError> {
        use program::BindingResource::{Buffer, Sampler, TextureView};
        // eprintln!("{:?}", desc);
        match *desc {
            Buffer {
                buffer_idx,
                offset,
                size,
            } => {
                debug.buffer_use(DeviceBuffer(buffer_idx), TextureInitState::UseRead);
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
            TextureView(idx) => {
                debug.view_use(idx, TextureInitState::UseRead);

                self.texture_views
                    .get(idx)
                    .ok_or_else(|| StepError::InvalidInstruction(line!()))
                    .map(wgpu::BindingResource::TextureView)
            }
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

    fn prepare_attachments_for_render_pass<'set, 'buf>(
        &'set self,
        desc: &program::RenderPassDescriptor,
        buf: &'buf mut Vec<Option<wgpu::RenderPassColorAttachment<'set>>>,
        debug: &mut Debug,
    ) -> Result<wgpu::RenderPassDescriptor<'buf>, StepError> {
        buf.clear();

        for attachment in &desc.color_attachments {
            debug.view_use(attachment.texture_view, TextureInitState::UseWrite);
            buf.push(Some(self.color_attachment(attachment)?));
        }

        Ok(wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: buf,
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
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
        fragments: &'set mut Vec<Option<wgpu::ColorTargetState>>,
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
            cache: None,
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
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        })
    }

    fn fragment_state<'set>(
        &'set self,
        desc: &program::FragmentState,
        buf: &'set mut Vec<Option<wgpu::ColorTargetState>>,
    ) -> Result<wgpu::FragmentState<'_>, StepError> {
        buf.clear();
        buf.extend(desc.targets.iter().cloned().map(Some));
        // eprintln!("{:?}", buf);
        Ok(wgpu::FragmentState {
            module: self
                .shaders
                .get(desc.fragment_module)
                .ok_or_else(|| StepError::InvalidInstruction(line!()))?,
            entry_point: desc.entry_point,
            targets: buf,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
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
                bytes_per_row: Some(layout.row_stride as u32),
                offset: 0,
                rows_per_image: Some(layout.height),
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
    pub(crate) fn new(exec: &Executable) -> Self {
        // eprintln!("{:#?}", exec.instructions);
        let range = exec.info.functions[&exec.entry_point].range.clone();
        Machine::with_instructions(exec.instructions.clone(), range)
    }

    pub(crate) fn with_instructions(
        instructions: Arc<[Low]>,
        entry: core::ops::Range<usize>,
    ) -> Self {
        Machine {
            instruction_pointer: vec![entry],
            instructions,
        }
    }

    fn is_running(&self) -> bool {
        !self.instruction_pointer.is_empty()
    }

    fn next_instruction(&mut self) -> Result<(program::Instruction, &Low), StepError> {
        let instruction = loop {
            let ip = self
                .instruction_pointer
                .last_mut()
                .ok_or(StepError::ProgramEnd)?;

            // This is equivalent to flowing off the end of a function. Normally this must not
            // happen without explicit return instructions, except at the end of a program. That
            // is, if this occurs further instructions will not receive their expected values such
            // as instead having broken or (logically) uninitialized buffer.
            //
            // The state being consumed immediately afterwards is okay though, which happens at the
            // end of the program.
            if let Some(instruction) = ip.next() {
                break instruction;
            }

            let _ = self.instruction_pointer.pop();
        };

        let low = self
            .instructions
            .get(instruction)
            .ok_or_else(|| StepError::InvalidInstruction(line!()))?;

        Ok((program::Instruction(instruction), low))
    }

    fn render_pass<'pass>(
        &mut self,
        descriptors: &'pass Descriptors,
        mut pass: wgpu::RenderPass<'pass>,
    ) -> Result<(), StepError> {
        loop {
            let (_, instruction) = match self.next_instruction() {
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
        self.retire_image(index)
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
        self.retire_image(index)
    }

    /// Move the render target corresponding to `reg` into the pool.
    ///
    /// Return the image as viewed inside the pool.
    pub fn render(&mut self, reg: Register) -> Result<PoolImage<'_>, RetireError> {
        let index = self
            .execution
            .host
            .io_map
            .renders
            .get(&reg)
            .copied()
            .ok_or(RetireError {
                inner: RetireErrorKind::NoSuchOutput,
            })?;
        self.retire_image(index)
    }

    pub(crate) fn retire_image(&mut self, index: usize) -> Result<PoolImage<'_>, RetireError> {
        let image = &mut self.execution.host.descriptors.image_io_buffers[index];
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

        pool_image.swap_image(&mut image.data);

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

        Ok(self.execution.host.descriptors.image_io_buffers[index].key)
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

            // If the texture is still shared, we can not cache its contents as it may be
            // overridden. It should not still be shared, but what can you do.
            let Some(texture) = Arc::into_inner(texture) else {
                continue;
            };

            let key = self.pool.insert_cacheable_buffer(descriptor, texture);
            stats.mem += descriptor.u64_len();
            stats.buffer_keys.push(key);
            self.uncorrected_gpu_buffers.push(key);
        }

        let tidx = 0..descriptors.shaders.len();
        let shaders = descriptors.shaders.drain(..).zip(tidx);
        for (shader, idx) in shaders {
            let descriptor = match descriptors.shader_descriptors.get(&idx) {
                None => continue,
                Some(descriptor) => descriptor,
            };

            let key = self.pool.insert_cacheable_shader(descriptor, shader);
            stats.shader_keys.push(key);
            self.uncorrected_shaders.push(key);
        }

        let tidx = 0..descriptors.render_pipelines.len();
        let pipelines = descriptors.render_pipelines.drain(..).zip(tidx);
        for (pipeline, idx) in pipelines {
            let descriptor = match descriptors.pipeline_descriptors.get(&idx) {
                None => continue,
                Some(descriptor) => descriptor,
            };

            let key = self.pool.insert_cacheable_pipeline(descriptor, pipeline);
            stats.pipeline_keys.push(key);
            self.uncorrected_pipelines.push(key);
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

        let gpu = self.execution.gpu;
        if let Some(gpu_key) = self.execution.gpu_key {
            self.pool.reinsert_device(gpu_key, gpu);

            // Fixup the gpu reference for all inserted gpu buffers.
            for pool_key in self.uncorrected_gpu_textures.into_iter() {
                self.pool.reassign_texture_gpu_unguarded(pool_key, gpu_key);
            }

            for pool_key in self.uncorrected_gpu_buffers.into_iter() {
                self.pool.reassign_buffer_gpu_unguarded(pool_key, gpu_key);
            }

            for shader_key in self.uncorrected_shaders.into_iter() {
                self.pool.reassign_shader_gpu_unguarded(shader_key, gpu_key);
            }

            for pipeline_key in self.uncorrected_pipelines.into_iter() {
                self.pool
                    .reassign_pipeline_gpu_unguarded(pipeline_key, gpu_key);
            }
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

impl StepLimits {
    pub fn new() -> Self {
        StepLimits { instructions: 1 }
    }

    pub fn with_steps(mut self, instructions: usize) -> Self {
        self.instructions = instructions;
        self
    }

    fn is_exhausted(&self) -> bool {
        self.instructions == 0
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
                block_on(future, Some(&gpu))?;
                self.time.checkpoint();
                Ok(())
            }
        }
    }

    /// Step towards synchronization with the end of instructions, asynchronously.
    ///
    /// The provided closure must schedule polling of the GPU via some unspecified internal means,
    /// *if* it is called. On a wasm32 web target for instance this is not necessary and will be
    /// done automatically by itself in the background.
    ///
    /// The `queue_poll` returns a guard value. When the value is dropped, the polling loop can and
    /// should be stopped. While it's possible to spawn a thread or task with routine polling, an
    /// integration maintaining multiple concurrent executions may want to optimize to check the
    /// device's ID instead.
    pub async fn finish<Guard>(
        &mut self,
        // A method that will ensure the GPU queue to be polled while its guard is live.
        queue_poll: impl FnOnce(Gpu) -> Guard,
    ) -> Result<(), StepError> {
        let Some(polled) = &mut self.future else {
            return Ok(());
        };

        // In contrast to the synchronous `block_on` code, here we want to avoid integrating the
        // GPU device itself deeply into the loop. If possible, the caller is responsible. This is
        // due to wasm32-web where no polling needs to be done at all. Even with native code the
        // necessary polling should not block the main asynchronous thread itself.
        //
        // Instead, rely on `on_submitted_work_done` callbacks for decisions. We still want to have
        // concurrency between each work as progress can be made by submitting more work, except
        // for specific synchronization points such as mapping a buffer for read-back.
        struct ResubmitCheck<'q> {
            /// Number of submits we can wait for.
            submit_check: Arc<AtomicU64>,
            /// Number of submits that have been enqueued.
            submitted: u64,
            /// Number of submits that are complete.
            submit_done: Arc<AtomicU64>,
            /// The queue to submit on_submitted_work_done checks against.
            queue: &'q Queue,
        }

        impl core::future::Future for ResubmitCheck<'_> {
            type Output = ();

            fn poll(self: Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<()> {
                // Maintain: submit_check >= submitted >= submit_done locally for ResubmitCheck
                let this = Pin::into_inner(self);
                let check = this.submit_check.load(Ordering::SeqCst);
                let done = this.submit_done.load(Ordering::Acquire);

                // Are we to register a new waker for outstanding submissions?
                if this.submitted == check {
                    // We must maintain the property: either
                    // - the `done` loaded has updated to the final value and we do not suspend
                    // - the `done` loaded has not yet updated to the final value, `wake` will get
                    //   called after this suspend.
                    return if this.submitted == done {
                        std::task::Poll::Ready(())
                    } else {
                        // Here we must guarantee wake was _after_ the call to poll (not its
                        // completion and return). The happens-before relationship is established
                        // by the SeqCst chain on `submit_done`. We got here because the
                        // `fetch_max` catching up to `submitted` did not happen-before the above
                        // load. That implies it happens-after since SeqCst requires at least one
                        // of these two orderings to hold. The `wake()`  in that callback is
                        // happens-after by local ordering.
                        std::task::Poll::Pending
                    };
                }

                struct CompleteOnDrop {
                    check: u64,
                    submit_done: Arc<AtomicU64>,
                    waker: Option<std::task::Waker>,
                }

                impl Drop for CompleteOnDrop {
                    fn drop(&mut self) {
                        // See above comment on this ordering of operations, and SeqCst.
                        self.submit_done.fetch_max(self.check, Ordering::SeqCst);
                        self.waker.take().unwrap().wake();
                    }
                }

                let callback_or_dropped = CompleteOnDrop {
                    check,
                    submit_done: this.submit_done.clone(),
                    waker: Some(cx.waker().clone()),
                };

                this.queue.on_submitted_work_done(move || {
                    drop(callback_or_dropped);
                });

                this.submitted = check;
                // The on_submitted_work_done callback will happen-after the entry to this function
                // by itself. We just require that it actually happens in finite time and that also
                // happens should the device itself get dropped (assuming this behaves usual).
                std::task::Poll::Pending
            }
        }

        let check = self.submit_check.clone();
        let DevicePolled { future, gpu } = polled;

        // TODO: this could be lazy!
        let _guard = queue_poll(gpu.clone());

        let submit_gpu = gpu.clone();
        // Technically optional, but polling this future will ensure that the effects of the
        // SyncPoint have been stabilized before stepping the next time. In particular, this
        // protects the guard from dropping before the need of polling is done.
        let submits_done_future = ResubmitCheck {
            submit_check: check,
            submitted: 0,
            submit_done: Arc::default(),
            queue: submit_gpu.queue(),
        };

        // FIXME: we may want to poll these as a single join to get timely status updates. But then
        // again, this is just fine. We only need that we do no drop the `_guard` before all submit
        // calls have been polled for.
        let result = future.await;
        // Avoid polling the future (on Drop) after this point, it's done.
        self.future = None;
        submits_done_future.await;

        result
    }

    /// Report the time spent in this sync point.
    ///
    /// If no timing interfaces are configured (e.g. on wasm32-none) then reports no duration.
    /// Otherwise, based on total system time. This is convenience, should probably instead report
    /// it as time spent keyed by each device actually utilized.
    pub fn time_spent(&self) -> std::time::Duration {
        self.time.spent()
    }

    /// Report, as debugging, the marker for the program instruction being synced on.
    pub fn debug_mark(&self) -> Option<&str> {
        self.debug_mark.as_deref()
    }
}

impl Debug {
    fn buffer_use(&mut self, tex: DeviceBuffer, state: TextureInitState) {
        self.buffers.entry(tex).or_default().init = state;
        self.buffer_history.entry(tex).or_default().push(state);
    }

    fn texture_use(&mut self, tex: DeviceTexture, state: TextureInitState) {
        self.textures.entry(tex).or_default().init = state;
        self.texture_history.entry(tex).or_default().push(state);
    }

    fn view_use(&mut self, view: usize, state: TextureInitState) {
        if let Some(texture) = self.texture_views[&view].init {
            self.texture_use(texture, state);
        }
    }

    fn texture_view(&mut self, view: usize, init: DeviceTexture) {
        self.texture_views.insert(
            view,
            DebugViewInfo {
                info: DebugInfo::default(),
                init: Some(init),
            },
        );
    }

    fn image_view(&mut self, view: usize) {
        self.texture_views.insert(
            view,
            DebugViewInfo {
                info: DebugInfo::default(),
                init: None,
            },
        );
    }
}

/// Block on an async future that may depend on a device being polled.
pub(crate) fn block_on<F, T>(future: F, device: Option<&Gpu>) -> T
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
                    .with_gpu(|gpu| gpu.device().poll(wgpu::Maintain::Poll));
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

pub(crate) fn copy_host_to_buffer(
    source: &[u8],
    target: &mut [u8],
    source_layout: &BufferLayout,
    target_layout: ByteLayout,
) {
    let width = target_layout.width;
    let height = target_layout.height;

    // TODO: defensive programming, don't assume cast works. Proof?
    let target_pitch = target_layout.row_stride as usize;
    let bytes_per_texel = target_layout.texel_stride;
    let bytes_to_copy = (u32::from(bytes_per_texel) * width) as usize;
    // TODO(perf): should this use our internal descriptor?
    let source_pitch = source_layout.as_row_layout().row_stride as usize;

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
}

impl Drop for SyncPoint<'_> {
    fn drop(&mut self) {
        // FIXME: not a good strategy to poll to completion. However, dropping the future without
        // polling it to completion seems in some instance to have a dead-lock effect. Should
        // investigate alternate strategies for correctly destroying this future or unconditionally
        // panic to enforce an explicit choice by the caller (when async polling is implemented).
        if self.future.is_some() {
            let _ = self.block_on();
        }
    }
}
