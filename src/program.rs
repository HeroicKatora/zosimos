use core::ops::Range;
use std::borrow::Cow;
use std::collections::HashMap;
use crate::command::{High, Rectangle, Register};
use crate::buffer::{BufferLayout, Descriptor};
use crate::pool::{Pool, PoolKey};
use crate::run::Execution;

/// Planned out and intrinsically validated command buffer.
///
/// This does not necessarily plan out a commands of low leve execution instruction set flavor.
/// This is selected based on the available device and its capabilities, which is performed during
/// launch.
pub struct Program {
    pub(crate) ops: Vec<High>,
    pub(crate) textures: Textures,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) enum Function {
    /// VS: id
    ///   in: vec3 position
    ///   in: vec2 vertUv
    ///   out: vec2 uv
    /// FS:
    ///   in: vec2 uv
    ///   pc: vec4 (parameter)
    ///   bind: sampler2D[2]
    ///   out: vec4 (color)
    PaintOnTop {
        // Source selection.
        lower_region: [Rectangle; 2],
        // Target viewport.
        upper_region: Rectangle,
        // The shader to execute with that pipeline.
        fragment_shader: &'static [u8],
    },
}

#[derive(Default)]
pub struct Textures {
    vec: Vec<Descriptor>,
    by_layout: HashMap<BufferLayout, usize>,
}

/// Identifies one resources in the render pipeline, by an index.
#[derive(Clone, Copy)]
pub(crate) struct Texture(usize);

#[derive(Debug)]
pub struct LaunchError {
}

/// Low level instruction.
///
/// Can be scheduled/ran directly on a machine state. Our state machine is a simplified GL-like API
/// that fully manages lists of all created texture samples, shader modules, command buffers,
/// attachments, descriptors and passes.
///
/// Currently, resources are never deleted until the end of the program. All commands reference a
/// particular selected device/queue that is implicit global context.
pub(crate) enum Low {
    // Descriptor modification commands.
    /// Create (and store) a bind group layout.
    BindGroupLayout(BindGroupLayoutDescriptor),
    /// Create (and store) a bind group, referencing one of the layouts.
    BindGroup(BindGroupDescriptor),
    /// Create (and store) a new buffer.
    Buffer(BufferDescriptor),
    /// Describe (and store) a new pipeline layout.
    PipelineLayout(PipelineLayoutDescriptor),
    /// Create (and store) a new sampler.
    Sampler(SamplerDescriptor),
    /// Upload (and store) a new shader.
    Shader(ShaderDescriptor),
    /// Create (and store) a new texture .
    Texture(TextureDescriptor),
    /// Create (and store) a view on a texture .
    /// Due to internal restrictions this isn't really helpful.
    TextureView(TextureViewDescriptor),
    /// Create (and store) a render pipeline with specified parameters.
    RenderPipeline(RenderPipelineDescriptor),

    // Render state commands.
    /// Start a new command recording.  It reaches until `EndCommands` but can be interleaved with
    /// arbitrary other commands.
    BeginCommands,
    /// Starts a new render pass within the current command buffer, which can only contain render
    /// instructions. Has effect until `EndRenderPass`.
    BeginRenderPass(RenderPassDescriptor),
    /// Ends the command, push a new `CommandBuffer` to our list.
    EndCommands,
    /// End the render pass.
    EndRenderPass,

    // Command context.

    // Render pass commands.
    SetPipeline(usize),
    SetBindGroup {
        group: usize,
        index: u32,
        offsets: Cow<'static, [u32]>,
    },
    SetVertexBuffer {
        slot: u32,
        buffer: usize,
    },
    DrawOnce {
        vertices: u32,
    },
    DrawIndexedZero {
        vertices: u32,
    },
    SetPushConstants {
        stages: wgpu::ShaderStage,
        offset: u32,
        data: Cow<'static, [u8]>,
    },

    // Render execution commands.
    /// Run one command buffer previously created.
    RunTopCommand,
    /// Run multiple commands at once.
    RunTopToBot(usize),
    /// Run multiple commands at once.
    RunBotToTop(usize),
    /// Read a buffer into host image data.
    /// Will map the buffer then do row-wise writes.
    WriteImageToBuffer {
        source_image: PoolKey,
        offset: (u32, u32),
        size: (u32, u32),
        target_buffer: usize,
        target_layout: BufferLayout,
    },
    WriteImageToTexture {
        source_image: PoolKey,
        offset: (u32, u32),
        size: (u32, u32),
        target_texture: usize,
    },
    /// Read a buffer into host image data.
    /// Will map the buffer then do row-wise reads.
    ReadBuffer {
        source_buffer: usize,
        source_layout: BufferLayout,
        offset: (u32, u32),
        size: (u32, u32),
        target_image: usize,
    },
}

/// Create a bind group.
pub(crate) struct BindGroupDescriptor {
    /// Select the nth layout.
    pub layout_idx: usize,
    /// All entries at their natural position.
    pub entries: Vec<BindingResource>,
}

pub(crate) enum BindingResource {
    Buffer {
        buffer_idx: usize,
        offset: wgpu::BufferAddress,
        size: Option<wgpu::BufferSize>,
    },
    Sampler(usize),
    TextureView(usize),
}

/// Describe a bind group.
pub(crate) struct BindGroupLayoutDescriptor {
    pub entries: Vec<wgpu::BindGroupLayoutEntry>,
}

/// Create a render pass.
pub(crate) struct RenderPassDescriptor {
    pub color_attachments: Vec<ColorAttachmentDescriptor>,
    pub depth_stencil: Option<DepthStencilDescriptor>,
}

pub(crate) struct ColorAttachmentDescriptor {
    pub texture_view: usize,
    pub ops: wgpu::Operations<wgpu::Color>,
}

pub(crate) struct DepthStencilDescriptor {
    pub texture_view: usize,
    pub depth_ops: Option<wgpu::Operations<f32>>,
    pub stencil_ops: Option<wgpu::Operations<u32>>,
}

/// The vertex+fragment shaders, primitive mode, layout and stencils.
/// Ignore multi sampling.
pub(crate) struct RenderPipelineDescriptor {
    pub layout: usize,
    pub vertex: VertexState,
    pub primitive: PrimitiveState,
    pub fragment: FragmentState,
}

pub(crate) struct VertexState {
    pub vertex_module: usize,
    pub entry_point: &'static str,
    pub targets: Vec<wgpu::ColorTargetState>,
}

pub(crate) enum PrimitiveState {
    SoleQuad,
}

pub(crate) struct FragmentState {
    pub fragment_module: usize,
    pub entry_point: &'static str,
    pub targets: Vec<wgpu::ColorTargetState>,
}

pub(crate) struct PipelineLayoutDescriptor {
    pub bind_group_layouts: Vec<usize>,
    pub push_constant_ranges: &'static [wgpu::PushConstantRange],
}

/// For constructing a new buffer, of anonymous memory.
pub(crate) struct BufferDescriptor {
    pub size: wgpu::BufferAddress,
    pub usage: BufferUsage,
}

pub(crate) struct ShaderDescriptor {
    pub name: &'static str,
    pub source_spirv: Cow<'static, [u32]>,
    pub flags: wgpu::ShaderFlags,
}

pub(crate) enum BufferUsage {
    /// Map Write + Vertex
    InVertices,
    /// Map Write + Storage + Copy Src
    DataIn,
    /// Map Read + Storage + Copy Dst
    DataOut,
    /// Map Read/Write + Storage + Copy Src/Dst
    DataInOut,
    /// Map Write + Uniform + Copy Src
    Uniform,
}

/// For constructing a new texture.
/// Ignores mip level, sample count, and some usages.
pub(crate) struct TextureDescriptor {
    pub size: (u32, u32),
    pub format: wgpu::TextureFormat,
    pub usage: TextureUsage,
}

pub(crate) enum TextureUsage {
    /// Copy Dst + Sampled
    DataIn,
    /// Copy Src + Render Attachment
    DataOut,
    /// A storage texture
    /// Copy Src/Dst + Sampled + Render Attachment
    Storage,
}

pub(crate) struct TextureViewDescriptor {
    pub texture: usize,
}

// FIXME: useless at the moment of writing, for our purposes.
// For reinterpreting parts of a texture.
// Ignores format (due to library restrictions), cube, aspect, mip level.
// pub(crate) struct TextureViewDescriptor;

/// For constructing a texture samples.
/// Ignores lod attributes
pub(crate) struct SamplerDescriptor {
    /// In all directions.
    pub address_mode: wgpu::AddressMode,
    pub resize_filter: wgpu::FilterMode,
    // TODO: evaluate if necessary or beneficial
    // compare: Option<wgpu::CompareFunction>,
    pub border_color: Option<wgpu::SamplerBorderColor>,
}

/// Cost planning data.
///
/// This helps quantify, approximate, or at least guess relative costs of operations with the goal
/// of supporting the planning of an execution plan. The internal unit of measurement is a copy of
/// one page of host memory to another page, based on the idea of directly expressing the costs for
/// a trivial pipeline with this.
pub struct CostModel {
    /// Do a 4×4 matrix multiplication on top of the copy.
    cpu_overhead_mul4x4: f32,
    /// Transfer a page to the default GPU.
    gpu_default_tx: f32,
    /// Transfer a page from the default GPU.
    gpu_default_rx: f32,
    /// Latency of scheduling something on the GPU.
    gpu_latency: f32,
}

/// The commands could not be made into a program.
#[derive(Debug)]
pub enum CompileError {
    #[deprecated = "We should strive to remove these"]
    NotYetImplemented,
}

/// Something won't work with this program and pool combination, no matter the amount of
/// configuration.
#[derive(Debug)]
pub struct MismatchError {
}

/// Prepare program execution with a specific pool.
///
/// Some additional assembly and configuration might be required and possible. For example choose
/// specific devices for running, add push attributes,
pub struct Launcher<'program> {
    program: &'program Program,
    pool: &'program mut Pool,
}

impl Textures {
    pub(crate) fn allocate_for(&mut self, desc: &Descriptor, _: Range<usize>)
        -> Texture
    {
        // FIXME: we could de-duplicate textures using liveness information.
        let idx = self.vec.len();
        self.vec.push(desc.clone());
        self.by_layout.insert(desc.layout.clone(), idx);
        Texture(idx)
    }
}

impl Program {
    /// Choose an applicable adapter from one of the presented ones.
    pub fn choose_adapter(&self, mut from: impl Iterator<Item=wgpu::Adapter>)
        -> Result<wgpu::Adapter, MismatchError>
    {
        while let Some(adapter) = from.next() {
            // FIXME: check limits.
            // FIXME: collect required texture formats from `self.textures`
            let basic_format = adapter.get_texture_format_features(wgpu::TextureFormat::Rgba8Uint);
            if !basic_format.allowed_usages.contains(wgpu::TextureUsage::all()) {
                continue;
            }

            from.for_each(drop);
            return Ok(adapter)
        }

        Err(MismatchError {})
    }

    /// Return a descriptor for a device that's capable of executing the program.
    pub fn device_descriptor(&self) -> wgpu::DeviceDescriptor<'static> {
        wgpu::DeviceDescriptor {
            label: None,
            features: wgpu::Features::empty(),
            limits: wgpu::Limits::default(),
        }
    }

    /// Run this program with a pool.
    ///
    /// Required input and output image descriptors must match those declared, or be convertible
    /// to them when a normalization operation was declared.
    pub fn launch<'pool>(&'pool self, pool: &'pool mut Pool)
        -> Launcher<'pool>
    {
        Launcher { program: self, pool }
    }
}

impl Launcher<'_> {
    /// Bind an image in the pool to an input register.
    ///
    /// Returns an error if the register does not specify an input, or when there is no image under
    /// the key in the pool, or when the image in the pool does not match the declared format.
    pub fn bind(self, Register(reg): Register, img: PoolKey)
        -> Result<Self, LaunchError>
    {
        let (target, descriptor) = match self.program.ops.get(reg) {
            Some(High::Input(target, descriptor)) => (target, descriptor),
            _ => return Err(LaunchError { })
        };

        Ok(self)
    }

    /// Really launch, potentially failing if configuration or inputs were missing etc.
    pub fn launch(self, adapter: &wgpu::Adapter) -> Result<Execution, LaunchError> {
        let request = adapter.request_device(&self.program.device_descriptor(), None);
        let (device, queue) = match block_on(request) {
            Ok(tuple) => tuple,
            Err(_) => return Err(LaunchError {}),
        };
        todo!()
    }
}

fn block_on<F, T>(future: F) -> T
where
    F: core::future::Future<Output = T> + 'static
{
    #[cfg(target_arch = "wasm32")] {
        use std::rc::Rc;
        use core::cell::RefCell;

        async fn the_thing<F, T>(future: F, buffer: Rc<RefCell<Option<T>>>) {
            let result = future.await;
            *buffer.borrow_mut() = result;
        }

        let result = Rc::new(RefCell::new(None));
        let mover = Rc::clone(&result);

        wasm_bindgen_futures::spawn_local(the_thing(future, mover));

        result.try_unwrap().unwrap()
    }

    #[cfg(not(target_arch = "wasm32"))] {
        async_io::block_on(future)
    }
}
