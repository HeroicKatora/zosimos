use core::{future::Future, num::{NonZeroU32, NonZeroU64}, ops::Range};

use std::borrow::Cow;
use std::collections::HashMap;

use crate::command::{High, Rectangle, Register, Target};
use crate::buffer::{BufferLayout, Color, Descriptor};
use crate::pool::{ImageData, Pool, PoolKey};
use crate::{run, shaders};
use crate::util::ExtendOne;

/// Planned out and intrinsically validated command buffer.
///
/// This does not necessarily plan out a commands of low level execution instruction set flavor.
/// This is selected based on the available device and its capabilities, which is performed during
/// launch.
pub struct Program {
    pub(crate) ops: Vec<High>,
    /// Assigns resources to each image based on liveness.
    /// This translates the SSA form into a mutable mapping where each image can be represented by
    /// a texture and a buffer. The difference is that the texture is assigned based on the _exact_
    /// descriptor while the buffer only requires the same byte layout and is treated as untyped
    /// memory.
    /// Note that, still, these are virtual registers. The encoder need not make use of them and it
    /// might allocate multiple physical textures if this is required to execute a conversion
    /// shader etc. It is however guaranteed that using the buffers of a _live_ register can not
    /// affect any other images.
    /// The encoder can make use of this mapping as intermediate resources for transfer between
    /// different images or from host to graphic device etc.
    pub(crate) textures: ImageBufferPlan,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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
        /// Source selection (relative to texture coordinates).
        selection: Rectangle,
        /// Target location in target texture.
        target: Rectangle,
        /// Rectangle that the draw call targets in the target texture.
        /// This gives the coordinate basis.
        viewport: Rectangle,
        paint_on_top: PaintOnTopKind,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum PaintOnTopKind {
    Copy,
}

#[derive(Clone, Debug, Default)]
pub struct ImageBufferPlan {
    pub(crate) texture: Vec<Descriptor>,
    pub(crate) buffer: Vec<BufferLayout>,
    pub(crate) by_register: Vec<ImageBufferAssignment>,
    pub(crate) by_layout: HashMap<BufferLayout, Texture>,
}

/// Contains the data on how images relate to the launcher's pool.
#[derive(Default, Clone, Debug)]
pub struct ImagePoolPlan {
    /// Maps registers to the pool image we took it from.
    pub(crate) plan: HashMap<Register, PoolKey>,
    /// Maps pool images to the texture in the buffer list.
    pub(crate) buffer: HashMap<PoolKey, Texture>,
}

#[derive(Clone, Copy, Debug)]
pub struct ImageBufferAssignment {
    pub(crate) texture: Texture,
    pub(crate) buffer: Buffer,
}

/// Identifies one layout based buffer in the render pipeline, by an index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Buffer(pub(crate) usize);

/// Identifies one descriptor based resource in the render pipeline, by an index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Texture(pub(crate) usize);

/// The encoder tracks the supposed state of `run::Descriptors` without actually executing them.
#[derive(Default)]
struct Encoder<Instructions: ExtendOne<Low> = Vec<Low>> {
    instructions: Instructions,
    
    // Replicated fields from `run::Descriptors` but only length.
    bind_groups: usize,
    bind_group_layouts: usize,
    buffers: usize,
    command_buffers: usize,
    modules: usize,
    pipeline_layouts: usize,
    render_pipelines: usize,
    sampler: usize,
    shaders: usize,
    textures: usize,
    texture_views: usize,

    // Additional validation properties.
    is_in_command_encoder: bool,
    is_in_render_pass: bool,
    commands: usize,

    // Additional fields to map our runtime state.
    /// How we map registers to device buffers.
    buffer_plan: ImageBufferPlan,
    /// Howe we mapped registers to images in the pool.
    pool_plan: ImagePoolPlan,
    paint_group_layout: Option<usize>,
    quad_group_layout: Option<usize>,
    paint_pipeline_layout: Option<usize>,
    default_sampler: Option<usize>,
    fragment_shaders: HashMap<FragmentShader, usize>,
    vertex_shaders: HashMap<VertexShader, usize>,
    simple_quad_buffer: Option<DeviceBuffer>,

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

/// The GPU buffers associated with a register.
/// Supplements the buffer_plan by giving direct mappings to each device resource index in an
/// encoder process.
#[derive(Clone, Debug)]
struct RegisterMap {
    texture: DeviceTexture,
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
    buffer_layout: BufferLayout,
    /// The format of the non-staging texture.
    texture_format: TextureDescriptor,
    /// The format of the staging texture.
    staging_format: Option<TextureDescriptor>,
}

/// A gpu buffer associated with an image buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceBuffer(pub(crate) usize);

/// A gpu texture associated with an image buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceTexture(pub(crate) usize);

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
}

/// The gpu buffer associated with an image buffer.
#[derive(Clone, Debug)]
struct BufferMap {
    device: DeviceBuffer,
    layout: BufferLayout,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum VertexShader {
    Noop,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum FragmentShader {
    PaintOnTop(PaintOnTopKind),
}

#[derive(Debug)]
pub struct LaunchError {
    line: u32,
}

/// Low level instruction.
///
/// Can be scheduled/ran directly on a machine state. Our state machine is a simplified GL-like API
/// that fully manages lists of all created texture samples, shader modules, command buffers,
/// attachments, descriptors and passes.
///
/// Currently, resources are never deleted until the end of the program. All commands reference a
/// particular selected device/queue that is implicit global context.
#[derive(Debug)]
pub(crate) enum Low {
    // Descriptor modification commands.
    /// Create (and store) a bind group layout.
    BindGroupLayout(BindGroupLayoutDescriptor),
    /// Create (and store) a bind group, referencing one of the layouts.
    BindGroup(BindGroupDescriptor),
    /// Create (and store) a new buffer.
    Buffer(BufferDescriptor),
    /// Create (and store) a new buffer with initial contents.
    BufferInit(BufferDescriptorInit),
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
        source_image: Texture,
        offset: (u32, u32),
        size: (u32, u32),
        target_buffer: DeviceBuffer,
        target_layout: BufferLayout,
    },
    WriteImageToTexture {
        source_image: Texture,
        offset: (u32, u32),
        size: (u32, u32),
        target_texture: DeviceTexture,
    },
    /// Copy a buffer to a texture with the same (!) layout.
    CopyBufferToTexture {
        source_buffer: DeviceBuffer,
        source_layout: BufferLayout,
        offset: (u32, u32),
        size: (u32, u32),
        target_texture: DeviceTexture,
    },
    /// Copy a texture to a buffer with fitting layout.
    CopyTextureToBuffer {
        source_texture: DeviceTexture,
        offset: (u32, u32),
        size: (u32, u32),
        target_buffer: DeviceBuffer,
        target_layout: BufferLayout,
    },
    CopyBufferToBuffer {
        source_buffer: DeviceBuffer,
        size: u64,
        target_buffer: DeviceBuffer,
    },
    /// Read a buffer into host image data.
    /// Will map the buffer then do row-wise reads.
    ReadBuffer {
        source_buffer: DeviceBuffer,
        source_layout: BufferLayout,
        offset: (u32, u32),
        size: (u32, u32),
        target_image: Texture,
    },
}

/// Create a bind group.
#[derive(Debug)]
pub(crate) struct BindGroupDescriptor {
    /// Select the nth layout.
    pub layout_idx: usize,
    /// All entries at their natural position.
    pub entries: Vec<BindingResource>,
}

#[derive(Debug)]
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
#[derive(Debug)]
pub(crate) struct BindGroupLayoutDescriptor {
    pub entries: Vec<wgpu::BindGroupLayoutEntry>,
}

/// Create a render pass.
#[derive(Debug)]
pub(crate) struct RenderPassDescriptor {
    pub color_attachments: Vec<ColorAttachmentDescriptor>,
    pub depth_stencil: Option<DepthStencilDescriptor>,
}

#[derive(Debug)]
pub(crate) struct ColorAttachmentDescriptor {
    pub texture_view: usize,
    pub ops: wgpu::Operations<wgpu::Color>,
}

#[derive(Debug)]
pub(crate) struct DepthStencilDescriptor {
    pub texture_view: usize,
    pub depth_ops: Option<wgpu::Operations<f32>>,
    pub stencil_ops: Option<wgpu::Operations<u32>>,
}

/// The vertex+fragment shaders, primitive mode, layout and stencils.
/// Ignore multi sampling.
#[derive(Debug)]
pub(crate) struct RenderPipelineDescriptor {
    pub layout: usize,
    pub vertex: VertexState,
    pub primitive: PrimitiveState,
    pub fragment: FragmentState,
}

#[derive(Debug)]
pub(crate) struct VertexState {
    pub vertex_module: usize,
    pub entry_point: &'static str,
}

#[derive(Debug)]
pub(crate) enum PrimitiveState {
    TriangleStrip,
}

#[derive(Debug)]
pub(crate) struct FragmentState {
    pub fragment_module: usize,
    pub entry_point: &'static str,
    pub targets: Vec<wgpu::ColorTargetState>,
}

#[derive(Debug)]
pub(crate) struct PipelineLayoutDescriptor {
    pub bind_group_layouts: Vec<usize>,
    pub push_constant_ranges: &'static [wgpu::PushConstantRange],
}

/// For constructing a new buffer, of anonymous memory.
#[derive(Debug)]
pub(crate) struct BufferDescriptor {
    pub size: wgpu::BufferAddress,
    pub usage: BufferUsage,
}

/// For constructing a new buffer, of anonymous memory.
#[derive(Debug)]
pub(crate) struct BufferDescriptorInit {
    pub content: Cow<'static, [u8]>,
    pub usage: BufferUsage,
}

#[derive(Debug)]
pub(crate) struct ShaderDescriptor {
    pub name: &'static str,
    pub source_spirv: Cow<'static, [u32]>,
    pub flags: wgpu::ShaderFlags,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum BufferUsage {
    /// Map Write + Vertex
    InVertices,
    /// Map Write + Storage + Copy Src
    DataIn,
    /// Map Read + Storage + Copy Dst
    DataOut,
    /// Storage + Copy Src/Dst
    DataBuffer,
    /// Map Write + Uniform + Copy Src
    Uniform,
}

/// For constructing a new texture.
/// Ignores mip level, sample count, and some usages.
#[derive(Clone, Debug)]
pub(crate) struct TextureDescriptor {
    /// The size, not that zero-sized textures have to be emulated by us.
    pub size: (NonZeroU32, NonZeroU32),
    pub format: wgpu::TextureFormat,
    pub usage: TextureUsage,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum TextureUsage {
    /// Copy Dst + Sampled
    DataIn,
    /// Copy Src + Render Attachment
    DataOut,
    /// A storage texture
    /// Copy Src/Dst + Sampled + Render Attachment
    Storage,
}

#[derive(Debug)]
pub(crate) struct TextureViewDescriptor {
    pub texture: DeviceTexture,
}

// FIXME: useless at the moment of writing, for our purposes.
// For reinterpreting parts of a texture.
// Ignores format (due to library restrictions), cube, aspect, mip level.
// pub(crate) struct TextureViewDescriptor;

/// For constructing a texture samples.
/// Ignores lod attributes
#[derive(Debug)]
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
    /// The host image data for each texture (if any).
    /// Otherwise this a placeholder image.
    binds: Vec<ImageData>,
    /// Assigns images from the internal pool to registers.
    /// They may be transferred from an input pool, and conversely we assign outputs. We can use
    /// the plan to put back all images into the pool when retiring the execution.
    pool_plan: ImagePoolPlan,
}

struct SimpleRenderPipelineDescriptor<'data> {
    vertex_bind_data: QuadBufferBind<'data>,
    fragment_texture: Texture,
    vertex: usize,
    fragment: usize,
}

enum QuadBufferBind<'data> {
    /// Upload the data, then bind the buffer.
    Set {
        data: &'data [u8],
    },
    /// The data is already there, simply bind the buffer.
    Load(DeviceBuffer),
}

struct SimpleRenderPipeline {
    pipeline: usize,
    buffer: DeviceBuffer,
    group: usize,
    quad: usize,
    vertices: u32,
}

impl ImageBufferPlan {
    pub(crate) fn allocate_for(&mut self, desc: &Descriptor, _: Range<usize>)
        -> ImageBufferAssignment
    {
        // FIXME: we could de-duplicate textures using liveness information.
        let texture = Texture(self.texture.len());
        self.texture.push(desc.clone());
        let buffer = Buffer(self.buffer.len());
        self.buffer.push(desc.layout.clone());
        self.by_layout.insert(desc.layout.clone(), texture);
        let assigned = ImageBufferAssignment {
            buffer,
            texture,
        };
        self.by_register.push(assigned);
        assigned
    }

    pub(crate) fn get(&self, idx: Register)
        -> Result<ImageBufferAssignment, LaunchError>
    {
        self.by_register.get(idx.0)
            .ok_or_else(|| LaunchError::InternalCommandError(line!()))
            .map(ImageBufferAssignment::clone)
    }
}

impl ImagePoolPlan {
    pub(crate) fn choose_output(&self, pool: &mut Pool, desc: &Descriptor) -> PoolKey {
        let mut entry = pool.declare(desc.clone());
        entry.host_allocate();
        entry.key()
    }

    pub(crate) fn get(&self, idx: Register)
        -> Result<PoolKey, LaunchError>
    {
        self.plan.get(&idx)
            .ok_or_else(|| LaunchError::InternalCommandError(line!()))
            .map(PoolKey::clone)
    }

    pub(crate) fn get_texture(&self, idx: Register)
        -> Option<Texture>
    {
        let key = self.plan.get(&idx)?;
        self.buffer.get(key).cloned()
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
        // Create empty bind assignments as a start, with respective layouts.
        let binds = self.textures.texture
            .iter()
            .map(|desciptor| ImageData::LateBound(desciptor.layout.clone()))
            .collect();

        Launcher {
            program: self,
            pool,
            binds,
            pool_plan: ImagePoolPlan::default(),
        }
    }
}

impl Launcher<'_> {
    /// Bind an image in the pool to an input register.
    ///
    /// Returns an error if the register does not specify an input, or when there is no image under
    /// the key in the pool, or when the image in the pool does not match the declared format.
    pub fn bind(mut self, Register(reg): Register, img: PoolKey)
        -> Result<Self, LaunchError>
    {
        let mut entry = match self.pool.entry(img) {
            Some(entry) => entry,
            None => return Err(LaunchError::InternalCommandError(line!())),
        };

        let Texture(texture) = match self.program.textures.by_register.get(reg) {
            Some(assigned) => assigned.texture,
            None => return Err(LaunchError::InternalCommandError(line!())),
        };

        self.pool_plan.plan.insert(Register(reg), img);
        self.pool_plan.buffer.insert(img, Texture(texture));

        Ok(self)
    }

    /// Determine images to use for outputs.
    ///
    /// You do not need to call this prior to launching as it will be performed automatically.
    /// However, you might get more detailed error information and in a future version might
    /// pre-determine the keys that will be used.
    pub fn bind_remaining_outputs(mut self)
        -> Result<Self, LaunchError>
    {
        for high in &self.program.ops {
            if let &High::Output { src: register, dst } = high {
                let assigned = &self.program.textures.by_register[register.0];
                let descriptor = &self.program.textures.texture[assigned.texture.0];
                let key = self.pool_plan.choose_output(&mut *self.pool, descriptor);
                self.pool_plan.plan.insert(dst, key);
            }
        }

        Ok(self)
    }

    /// Really launch, potentially failing if configuration or inputs were missing etc.
    pub fn launch(mut self, adapter: &wgpu::Adapter) -> Result<run::Execution, LaunchError> {
        let request = adapter.request_device(&self.program.device_descriptor(), None);

        // For all inputs check that they have now been supplied.
        for high in &self.program.ops {
            if let &High::Input(register, _) = high {
                if self.pool_plan.get_texture(register).is_none() {
                    return Err(LaunchError::InternalCommandError(line!()))
                }
            }
        }

        // Bind remaining outputs.
        self = self.bind_remaining_outputs()?;

        let (device, queue) = match block_on(request, None) {
            Ok(tuple) => tuple,
            Err(_) => return Err(LaunchError::InternalCommandError(line!())),
        };

        let mut encoder = Encoder::default();
        encoder.set_buffer_plan(&self.program.textures);
        encoder.set_pool_plan(&self.pool_plan);
        encoder.enable_capabilities(&device);

        for high in &self.program.ops {
            match high {
                &High::Done(_) => {
                    // TODO: should deallocate textures that aren't live anymore.
                }
                &High::Input(dst, _) => {
                    // Identify how we ingest this image.
                    // If it is a texture format that we support then we will allocate and upload
                    // it directly. If it is not then we will allocate a generic version capable of
                    // holding a lossless convert variant of it and add instructions to convert
                    // into that buffer.
                    encoder.copy_input_to_buffer(dst)?;
                    encoder.copy_buffer_to_staging(dst)?;
                }
                &High::Output { src, dst } => {
                    // Identify if we need to transform the texture from the internal format to the
                    // one actually chosen for this texture.
                    encoder.copy_staging_to_buffer(src)?;
                    encoder.copy_buffer_to_output(src, dst)?;
                }
                High::Construct { dst, op } => {
                    todo!()
                }
                High::Paint { texture, dst, fn_ } => {
                    let dst_texture = match dst {
                        Target::Discard(texture) | Target::Load(texture) => *texture,
                    };

                    encoder.ensure_allocate_texture(dst_texture)?;
                    encoder.copy_staging_to_texture(*texture)?;

                    let dst_view = encoder.texture_view(dst_texture)?;
                    // eprintln!("tex{:?} +> tex{:?}", texture, dst_texture);

                    let ops = match dst {
                        Target::Discard(_) => {
                            wgpu::Operations {
                                // TODO: we could let choose a replacement color..
                                load: wgpu::LoadOp::Clear(wgpu::Color::RED),
                                store: true,
                            }
                        },
                        Target::Load(_) => {
                            wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: true,
                            }
                        },
                    };

                    let attachment = ColorAttachmentDescriptor {
                        texture_view: dst_view,
                        ops,
                    };

                    let render = encoder.prepare_render(*texture, fn_)?;

                    // TODO: we need to remember the attachment format here.
                    // This is need to to automatically construct the shader pipeline.
                    encoder.push(Low::BeginCommands)?;
                    encoder.push(Low::BeginRenderPass(RenderPassDescriptor {
                        color_attachments: vec![attachment],
                        depth_stencil: None,
                    }))?;
                    encoder.render(render)?;
                    encoder.push(Low::EndRenderPass)?;
                    encoder.push(Low::EndCommands)?;

                    // Actually run it immediately.
                    // TODO: this might not be the most efficient.
                    encoder.push(Low::RunTopCommand)?;

                    // Post paint, make sure we quantize everything.
                    encoder.copy_texture_to_staging(*texture)?;
                }
            }
        }

        let mut buffers = self.binds;
        encoder.extract_buffers(&mut buffers, &mut self.pool)?;

        let init = run::InitialState {
            instructions: encoder.instructions,
            device,
            queue,
            buffers,
        };

        Ok(run::Execution::new(init))
    }
}

impl<I: ExtendOne<Low>> Encoder<I> {
    /// Tell the encoder which commands are natively supported.
    /// Some features require GPU support. At this point we decide if our request has succeeded and
    /// we might poly-fill it with a compute shader or something similar.
    fn enable_capabilities(&mut self, device: &wgpu::Device) {
        // currently no feature selection..
        let _ = device.features();
        let _ = device.limits();
    }

    fn set_buffer_plan(&mut self, plan: &ImageBufferPlan) {
        self.buffer_plan = plan.clone();
    }

    fn set_pool_plan(&mut self, plan: &ImagePoolPlan) {
        self.pool_plan = plan.clone();
    }

    fn extract_buffers(&self, buffers: &mut Vec<ImageData>, pool: &mut Pool) -> Result<(), LaunchError> {
        for (&pool_key, &texture) in &self.pool_plan.buffer {
            let mut entry = pool.entry(pool_key)
                .ok_or_else(|| LaunchError::InternalCommandError(line!()))?;
            let buffer = &mut buffers[texture.0];

            if buffer.as_bytes().is_none() {
                entry.swap(buffer);

                if buffer.as_bytes().is_none() {
                    return Err(LaunchError::InternalCommandError(line!()));
                }
            }
        }

        Ok(())
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
            },
            Low::BeginRenderPass(_) => {
                if self.is_in_render_pass {
                    return Err(LaunchError::InternalCommandError(line!()));
                }

                if !self.is_in_command_encoder {
                    return Err(LaunchError::InternalCommandError(line!()));
                }

                self.is_in_render_pass = true;
            },
            Low::EndCommands => {
                if !self.is_in_command_encoder {
                    return Err(LaunchError::InternalCommandError(line!()));
                }

                self.is_in_command_encoder = false;
                self.commands += 1;
            },
            Low::EndRenderPass => {
                if !self.is_in_render_pass {
                    return Err(LaunchError::InternalCommandError(line!()));
                }

                self.is_in_render_pass = false;
            }
            Low::SetPipeline(_) => {},
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
            Low::DrawOnce { .. }
            | Low::DrawIndexedZero { .. }
            | Low::SetPushConstants { .. } => {},
            Low::RunTopCommand => {
                if self.commands == 0{
                    return Err(LaunchError::InternalCommandError(line!()));
                }

                self.commands -= 1;
            }
            Low::RunBotToTop(num) | Low::RunTopToBot(num) => {
                if num >= self.commands {
                    return Err(LaunchError::InternalCommandError(line!()));
                }

                self.commands -= num;
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
            Low::WriteImageToBuffer { .. }
            | Low::ReadBuffer { .. } => {},
        }

        self.instructions.extend_one(low);
        Ok(())
    }

    fn make_texture_descriptor(&mut self, texture: Texture)
        -> Result<TextureDescriptor, LaunchError>
    {
        let descriptor = &self.buffer_plan.texture[texture.0];

        fn validate_size(layout: &BufferLayout) -> Option<(NonZeroU32, NonZeroU32)> {
            Some((
                NonZeroU32::new(layout.width)?,
                NonZeroU32::new(layout.height)?,
            ))
        }

        let size = validate_size(&descriptor.layout)
            .ok_or_else(|| LaunchError::InternalCommandError(line!()))?;

        let format = match descriptor.texel.color {
            Color::SRGB => wgpu::TextureFormat::Rgba8UnormSrgb,
            Color::Xyz { .. } => return Err(LaunchError::InternalCommandError(line!())),
        };

        // TODO: be more precise?
        let usage = TextureUsage::Storage;

        Ok(TextureDescriptor {
            format,
            size,
            usage,
        })
    }

    fn allocate_register(&mut self, idx: Register) -> Result<&RegisterMap, LaunchError> {
        self.ensure_allocate_register(idx)?;
        // Trick, reborrow the thing..
        Ok(&self.register_map[&idx])
    }

    // We must trick the borrow checker here..
    fn ensure_allocate_register(&mut self, idx: Register) -> Result<(), LaunchError> {
        let ImageBufferAssignment {
            buffer: reg_buffer,
            texture: reg_texture,
        } = self.buffer_plan.get(idx)?;

        if let Some(_) = self.register_map.get(&idx) {
            return Ok(());
        }

        let texture_format = self.make_texture_descriptor(reg_texture)?;
        let descriptor = &self.buffer_plan.texture[reg_texture.0];

        let bytes_per_row = (descriptor.layout.bytes_per_texel as u32)
            .checked_mul(texture_format.size.0.get())
            .ok_or_else(|| LaunchError::InternalCommandError(line!()))?;
        let bytes_per_row = (bytes_per_row/256 + u32::from(bytes_per_row%256 != 0))
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

            (
                DeviceBuffer(buffer),
                DeviceBuffer(buffer+1),
                DeviceBuffer(buffer+2),
            )
        };

        let texture = self.ensure_allocate_texture(reg_texture)?;

        let map_entry = RegisterMap {
            buffer,
            texture,
            map_read: Some(map_read),
            map_write: Some(map_write),
            staging: None,
            buffer_layout,
            texture_format,
            staging_format: None,
        };

        // TODO do a match instead?
        let in_map = self.register_map
            .entry(idx)
            .or_insert(map_entry.clone());
        *in_map = map_entry.clone();

        self.buffer_map.insert(reg_buffer, BufferMap {
            device: buffer,
            layout: in_map.buffer_layout.clone(),
        });

        if let Some(staging) = in_map.staging {
            self.staging_map.insert(reg_texture, StagingTexture {
                device: staging,
                format: in_map.staging_format
                    .clone()
                    .expect("Have a format for staging texture when we have staging texture"),
            });
        }

        Ok(())
    }

    fn ensure_allocate_texture(&mut self, reg_texture: Texture) -> Result<DeviceTexture, LaunchError> {
        let texture_format = self.make_texture_descriptor(reg_texture)?;

        if let Some(texture_map) = self.texture_map.get(&reg_texture) {
            return Ok(texture_map.device);
        }

        let texture = {
            let texture = self.textures;
            self.push(Low::Texture(texture_format.clone()))?;
            DeviceTexture(texture)
        };

        self.texture_map.insert(reg_texture, TextureMap {
            device: texture,
            format: texture_format.clone(),
        });

        Ok(texture)
    }

    fn ingest_image_data(&mut self, idx: Register) -> Result<Texture, LaunchError> {
        let source_key = self.pool_plan.get(idx)?;
        let texture = self.buffer_plan.get(idx)?.texture;
        self.pool_plan.buffer
            .entry(source_key)
            .or_insert(texture);
        Ok(texture)
    }

    /// Copy from the input to the internal memory visible buffer.
    fn copy_input_to_buffer(&mut self, idx: Register) -> Result<(), LaunchError> {
        let regmap = self.allocate_register(idx)?.clone();
        let source_image = self.ingest_image_data(idx)?;

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
            target_buffer: target_buffer,
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
    fn copy_buffer_to_staging(&mut self, idx: Register) -> Result<(), LaunchError> {
        let regmap = self.allocate_register(idx)?.clone();
        let size = self.buffer_plan.texture[regmap.texture.0].size();

        self.push(Low::BeginCommands)?;
        self.push(Low::CopyBufferToTexture {
            source_buffer: regmap.buffer,
            source_layout: regmap.buffer_layout,
            offset: (0, 0),
            size,
            target_texture: regmap.texture,
        })?;
        // eprintln!("buf{:?} -> tex{:?} ({:?})", regmap.buffer, regmap.texture, size);

        self.push(Low::EndCommands)?;
        // TODO: maybe also don't run it immediately?
        self.push(Low::RunTopCommand)?;

        Ok(())
    }

    /// Copy quantized data to the internal buffer.
    /// Note that this may be a no-op for buffers that need no staging buffer, i.e. where
    /// quantization happens as part of the pipeline.
    fn copy_staging_to_texture(&mut self, idx: Texture) -> Result<(), LaunchError> {
        if let Some(_) = self.staging_map.get(&idx) {
            todo!()
        } else {
            Ok(())
        }
    }

    /// Quantize the texture to the staging buffer.
    /// May be a no-op, see reverse operation.
    fn copy_texture_to_staging(&mut self, idx: Texture) -> Result<(), LaunchError> {
        if let Some(_) = self.staging_map.get(&idx) {
            todo!()
        } else {
            Ok(())
        }
    }

    /// Copy from texture to the memory buffer.
    fn copy_staging_to_buffer(&mut self, idx: Register) -> Result<(), LaunchError> {
        let regmap = self.allocate_register(idx)?.clone();
        let descriptor = &self.buffer_plan.texture[regmap.texture.0];
        let size = descriptor.size();
        // eprintln!("tex{:?} -> buf{:?} ({:?})", regmap.texture, regmap.buffer, size);

        self.push(Low::BeginCommands)?;
        self.push(Low::CopyTextureToBuffer {
            source_texture: regmap.texture,
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
    fn copy_buffer_to_output(&mut self, idx: Register, dst: Register)
        -> Result<(), LaunchError>
    {
        let regmap = self.allocate_register(idx)?.clone();
        let target_image = self.ingest_image_data(dst)?;
        let descriptor = &self.buffer_plan.texture[regmap.texture.0];

        let size = descriptor.size();
        let sizeu64 = regmap.buffer_layout.u64_len();
        
        let source_buffer = regmap.map_read
            .unwrap_or(regmap.buffer);

        // FIXME: might happen at next call within another command encoder..
        if let Some(map_read) = regmap.map_read {
            self.push(Low::BeginCommands)?;
            self.push(Low::CopyBufferToBuffer {
                source_buffer: regmap.buffer,
                size: sizeu64,
                target_buffer: map_read,
            })?;
            // eprintln!("buf{:?} -> buf{:?} ({})", regmap.buffer, map_read, sizeu64);
            self.push(Low::EndCommands)?;
            // TODO: maybe also don't run it immediately?
            self.push(Low::RunTopCommand)?;
        }
        // eprintln!("buf{:?} -> img{:?} ({:?})", source_buffer, target_image, size);

        self.push(Low::ReadBuffer {
            source_buffer,
            source_layout: regmap.buffer_layout,
            size,
            offset: (0, 0),
            target_image,
        })
    }

    fn texture_view(&mut self, dst: Texture) -> Result<usize, LaunchError> {
        let texture = self.texture_map.get(&dst)
            .ok_or_else(|| LaunchError::InternalCommandError(line!()))?
            .device;
        // eprintln!("tex{:?} == tex{:?}", dst, texture);

        let descriptor = TextureViewDescriptor { texture };

        self.instructions.extend_one(Low::TextureView(descriptor));
        let id = self.texture_views;
        self.texture_views += 1;

        Ok(id)
    }

    fn make_quad_bind_group(&mut self) -> usize {
        let bind_group_layouts = &mut self.bind_group_layouts;
        let instructions = &mut self.instructions;
        *self.quad_group_layout.get_or_insert_with(|| {
            let descriptor = BindGroupLayoutDescriptor {
                entries: vec![
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(2 * 8 * 4),
                            ty: wgpu::BufferBindingType::Uniform,
                        },
                        count: None,
                    },
                ],
            };

            instructions.extend_one(Low::BindGroupLayout(descriptor));
            let descriptor_id = *bind_group_layouts;
            *bind_group_layouts += 1;
            descriptor_id
        })
    }

    fn make_paint_group(&mut self) -> usize {
        let bind_group_layouts = &mut self.bind_group_layouts;
        let instructions = &mut self.instructions;
        *self.paint_group_layout.get_or_insert_with(|| {
            let descriptor = BindGroupLayoutDescriptor {
                entries: vec![
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Sampler {
                            filtering: true,
                            comparison: false,
                        },
                        count: None,
                    },
                ],
            };

            instructions.extend_one(Low::BindGroupLayout(descriptor));
            let descriptor_id = *bind_group_layouts;
            *bind_group_layouts += 1;
            descriptor_id
        })
    }

    fn make_paint_layout(&mut self) -> usize {
        let quad_bind_group = self.make_quad_bind_group();
        let paint_bind_group = self.make_paint_group();
        let layouts = &mut self.pipeline_layouts;
        let instructions = &mut self.instructions;

        *self.paint_pipeline_layout.get_or_insert_with(|| {
            let descriptor = PipelineLayoutDescriptor {
                bind_group_layouts: vec![
                    quad_bind_group,
                    paint_bind_group,
                ],
                push_constant_ranges: &[]
            };

            instructions.extend_one(Low::PipelineLayout(descriptor));
            let descriptor_id = *layouts;
            *layouts += 1;
            descriptor_id
        })
    }

    fn shader(&mut self, desc: ShaderDescriptor) -> Result<usize, LaunchError> {
        self.instructions.extend_one(Low::Shader(desc));
        let idx = self.shaders;
        self.shaders += 1;
        Ok(idx)
    }

    fn fragment_shader(&mut self, kind: Option<FragmentShader>, source: Cow<'static, [u32]>)
        -> Result<usize, LaunchError>
    {
        if let Some(&shader) = kind.and_then(|k| self.fragment_shaders.get(&k)) {
            return Ok(shader);
        }

        self.shader(ShaderDescriptor {
            name: "",
            flags: wgpu::ShaderFlags::empty(),
            source_spirv: source,
        })
    }

    fn vertex_shader(&mut self, kind: Option<VertexShader>, source: Cow<'static, [u32]>)
        -> Result<usize, LaunchError>
    {
        if let Some(&shader) = kind.and_then(|k| self.vertex_shaders.get(&k)) {
            return Ok(shader);
        }

        self.shader(ShaderDescriptor {
            name: "",
            flags: wgpu::ShaderFlags::empty(),
            source_spirv: source,
        })
    }

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
                content: bytemuck::cast_slice(content).into(),
            };

            instructions.extend_one(Low::BufferInit(descriptor));

            let buffer = *buffers;
            *buffers += 1;
            DeviceBuffer(buffer)
        })
    }
    
    fn simple_render_pipeline(
        &mut self,
        vertex: usize,
        fragment: usize,
        format: wgpu::TextureFormat,
    ) -> Result<usize, LaunchError> {
        let layout = self.make_paint_layout();

        self.instructions.extend_one(Low::RenderPipeline(RenderPipelineDescriptor {
            vertex: VertexState {
                entry_point: "main",
                vertex_module: vertex,
            },
            fragment: FragmentState {
                entry_point: "main",
                fragment_module: fragment,
                targets: vec![wgpu::ColorTargetState {
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrite::ALL,
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

    fn make_bind_group_sampled_texture(&mut self, texture: Texture)
        -> Result<usize, LaunchError>
    {
        let view = self.texture_view(texture)?;

        let sampler = {
            let instructions = &mut self.instructions;
            let sampler = &mut self.sampler;
            *self.default_sampler.get_or_insert_with(|| {
                let sampler_id = *sampler;
                instructions.extend_one(Low::Sampler(SamplerDescriptor {
                    address_mode: wgpu::AddressMode::default(),
                    border_color: Some(wgpu::SamplerBorderColor::TransparentBlack),
                    resize_filter: wgpu::FilterMode::Linear,
                }));
                *sampler += 1;
                sampler_id
            })
        };

        let group = self.bind_groups;
        let descriptor = BindGroupDescriptor {
            layout_idx: self.make_paint_group(),
            entries: vec![
                BindingResource::TextureView(view),
                BindingResource::Sampler(sampler),
            ],
        };

        self.push(Low::BindGroup(descriptor))?;
        Ok(group)
    }

    fn make_bind_group_quad_parameter(&mut self, bind: QuadBufferBind<'_>)
        -> Result<usize, LaunchError>
    {
        let buffer = match bind {
            QuadBufferBind::Load(buffer) => buffer,
            QuadBufferBind::Set { data } => {
                let buffer = self.buffers;
                self.push(Low::BufferInit(BufferDescriptorInit {
                    // FIXME: avoid the allocation here?
                    content: Cow::Owned(data.to_vec()),
                    usage: BufferUsage::Uniform,
                }))?;
                DeviceBuffer(buffer)
            },
        };

        let group = self.bind_groups;
        let descriptor = BindGroupDescriptor {
            layout_idx: self.make_quad_bind_group(),
            entries: vec![
                BindingResource::Buffer {
                    buffer_idx: buffer.0,
                    offset: 0,
                    size: None,
                }
            ],
        };

        self.push(Low::BindGroup(descriptor))?;
        Ok(group)
    }

    /// Render the pipeline, after all customization and buffers were bound..
    fn prepare_simple_pipeline(&mut self, descriptor: SimpleRenderPipelineDescriptor)
        -> Result<SimpleRenderPipeline, LaunchError>
    {
        let SimpleRenderPipelineDescriptor {
            vertex_bind_data,
            fragment_texture: texture,
            vertex,
            fragment,
        } = descriptor;

        let format = self.texture_map[&texture].format.format;

        let buffer = self.simple_quad_buffer();
        let pipeline = self.simple_render_pipeline(vertex, fragment, format)?;

        let group = self.make_bind_group_sampled_texture(texture)?;
        let quad = self.make_bind_group_quad_parameter(vertex_bind_data)?;

        Ok(SimpleRenderPipeline {
            pipeline,
            buffer,
            group,
            quad,
            vertices: 4,
        })
    }

    fn render(&mut self, pipeline: SimpleRenderPipeline) -> Result<(), LaunchError> {
        let SimpleRenderPipeline {
            pipeline,
            group,
            quad,
            buffer,
            vertices,
        } = pipeline;

        self.push(Low::SetPipeline(pipeline))?;

        self.push(Low::SetBindGroup {
            group: quad,
            index: 0,
            offsets: Cow::Borrowed(&[]),
        })?;

        self.push(Low::SetBindGroup {
            group,
            index: 1,
            offsets: Cow::Borrowed(&[]),
        })?;

        self.push(Low::SetVertexBuffer {
            buffer: buffer.0,
            slot: 0,
        })?;

        self.push(Low::DrawOnce { vertices })?;

        Ok(())
    }

    fn prepare_render(&mut self, texture: Texture, function: &Function)
        -> Result<SimpleRenderPipeline, LaunchError>
    {
        match function {
            Function::PaintOnTop { selection, target, viewport, paint_on_top } => {
                let (tex_width, tex_height) = self.texture_map[&texture].format.size;

                let vertex = self.vertex_shader(
                    Some(VertexShader::Noop),
                    shader_include_to_spirv(shaders::VERT_NOOP))?;

                let fragment = paint_on_top.fragment_shader();
                let fragment = self.fragment_shader(
                    Some(FragmentShader::PaintOnTop(paint_on_top.clone())),
                    shader_include_to_spirv(fragment))?;

                // FIXME: there seems to be two floats padding after each vec2.
                let min_u = (selection.x as f32) / (tex_width.get() as f32);
                let max_u = (selection.max_x as f32) / (tex_width.get() as f32);
                let min_v = (selection.y as f32) / (tex_height.get() as f32);
                let max_v = (selection.max_y as f32) / (tex_height.get() as f32);

                let target = viewport.meet_in_local_coordinates(*target);
                let min_a = (target.x as f32) / (viewport.width() as f32);
                let max_a = (target.max_x as f32) / (viewport.width() as f32);
                let min_b = (target.y as f32) / (viewport.height() as f32);
                let max_b = (target.max_y as f32) / (viewport.height() as f32);

                // I'm pretty sure this wrong.
                let buffer: [[f32; 2]; 16] = [
                    [min_u, min_v], [0.0, 0.0],
                    [max_u, min_v], [0.0, 0.0],
                    [max_u, max_v], [0.0, 0.0],
                    [min_u, max_v], [0.0, 0.0],

                    [min_a, min_b], [0.0, 0.0],
                    [max_a, min_b], [0.0, 0.0],
                    [max_a, max_b], [0.0, 0.0],
                    [min_a, max_b], [0.0, 0.0],
                ];

                self.prepare_simple_pipeline(SimpleRenderPipelineDescriptor{
                    vertex_bind_data: QuadBufferBind::Set {
                        data: bytemuck::cast_slice(&buffer[..]),
                    },
                    fragment_texture: texture,
                    vertex,
                    fragment,
                })
            },
        }
    }
}

fn shader_include_to_spirv(src: &[u8]) -> Cow<'static, [u32]> {
    assert!(src.len() % 4 == 0);
    let mut target = vec![0u32; src.len() / 4];
    bytemuck::cast_slice_mut(&mut target).copy_from_slice(src);
    Cow::Owned(target)
}

impl PaintOnTopKind {
    fn fragment_shader(&self) -> &[u8] {
        match self {
            PaintOnTopKind::Copy => shaders::FRAG_COPY,
        }
    }
}

impl BufferUsage {
    pub fn to_wgpu(self) -> wgpu::BufferUsage {
        use wgpu::BufferUsage as U;
        match self {
            BufferUsage::InVertices => U::COPY_DST | U::VERTEX,
            BufferUsage::DataIn => U::MAP_WRITE | U::COPY_SRC,
            BufferUsage::DataOut => U::MAP_READ | U::COPY_DST,
            BufferUsage::DataBuffer => {
                U::STORAGE | U::COPY_SRC | U::COPY_DST
            }
            BufferUsage::Uniform => U::COPY_DST | U::UNIFORM,
        }
    }
}

impl LaunchError {
    #[deprecated = "Should be removed and implemented"]
    pub(crate) const UNIMPLEMENTED_CHECK: Self = LaunchError { line: 0 };
    #[allow(non_snake_case)]
    #[deprecated = "This should be cleaned up"]
    pub(crate) fn InternalCommandError(line: u32) -> Self {
        LaunchError { line }
    }
}

/// Block on an async future that may depend on a device being polled.
pub(crate) fn block_on<F, T>(future: F, device: Option<&wgpu::Device>) -> T
where
    F: Future<Output = T> + 'static
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
        if let Some(device) = device {
            // We have to manually poll the device.  That is, we ensure that it keeps being polled
            // and each time will also poll the device. This isn't super efficient but a dirty way
            // to actually finish this future.
            struct DevicePolled<'dev, F> {
                future: F,
                device: &'dev wgpu::Device,
            }

            impl<F: Future> Future for DevicePolled<'_, F> {
                type Output = F::Output;
                fn poll(self: core::pin::Pin<&mut Self>, ctx: &mut core::task::Context)
                    -> core::task::Poll<F::Output>
                {
                    self.as_ref().device.poll(wgpu::Maintain::Poll);
                    // Ugh, noooo...
                    ctx.waker().wake_by_ref();
                    let future = unsafe { self.map_unchecked_mut(|this| &mut this.future) };
                    future.poll(ctx)
                }
            }

            async_io::block_on(DevicePolled {
                future,
                device,
            })
        } else {
            async_io::block_on(future)
        }
    }
}
