use core::{
    future::Future,
    num::{NonZeroU32, NonZeroU64},
    ops::Range,
};

use std::borrow::Cow;
use std::collections::HashMap;

use crate::buffer::{
    Block, BufferLayout, Color, Descriptor, RowMatrix, SampleBits, SampleParts, Samples, Texel,
    Transfer,
};
use crate::command::{High, Rectangle, Register, Target};
use crate::pool::{ImageData, Pool, PoolKey};
use crate::util::ExtendOne;
use crate::{run, shaders};

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

/// Describes a function call in more common terms.
///
/// The bind sets follow the logic that functions will use the same setup for the vertex shader to
/// pain a source rectangle at a target rectangle location. Then the required images are bound. And
/// finally we have the various dynamic/uniform parameters of the differing functions.
///
/// A single command might be translated to multiple functions.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Function {
    /// Execute a shader on an target rectangle.
    ///
    /// The UV coordinates and position is determined by vertex shader parameters computed from a
    /// selection, a target location, and a viewport.
    /// VS: id
    ///   in: vec3 position
    ///   in: vec2 vertUv
    ///   bind(0,1): rectangles
    ///   out: vec2 uv
    /// FS:
    ///   in: vec2 uv
    ///   bind(1,0): texture
    ///   bind(1,1): sampler2D
    ///   bind(2,0): shader specific data.
    ///   out: vec4 (color)
    PaintToSelection {
        /// The texture which is used as source.
        /// We require this to compute the specific quad coordinates.
        texture: Texture,
        /// Source selection (relative to texture coordinates).
        selection: Rectangle,
        /// Target location in target texture.
        target: QuadTarget,
        /// Rectangle that the draw call targets in the target texture.
        /// The target coordinates are relative to this and the fragment shader given by
        /// paint_on_top is only executed within that rectangle.
        viewport: Rectangle,
        shader: shaders::FragmentShader,
    },
    /// Execute a shader on full textures.
    /// VS: id
    ///   in: vec3 position
    ///   in: vec2 vertUv
    ///   bind(0,1): rectangles
    ///   out: vec2 uv
    /// FS:
    ///   in: vec2 uv
    ///   bind(1,0): texture
    ///   bind(1,1): sampler2D
    ///   bind(2,0): shader specific data.
    ///   out: vec4 (color)
    PaintFullScreen {
        shader: shaders::FragmentShader,
    },
    /// VS: id
    /// FS:
    ///   bind(1, …) readonly inputs uimage2D
    ///   bind(3, 0) struct {
    ///     vec4: transfer, sample parts, sample bits
    ///   }
    ToLinearOpto {
        parameter: shaders::stage::XyzParameter,
        stage_kind: shaders::stage::StageKind,
    },
    /// VS: id
    /// FS:
    ///   bind(2, …) writeonly inputs uimage2D
    ///   bind(3, 0) struct {
    ///     vec4: transfer, sample parts, sample bits
    ///   }
    FromLinearOpto {
        parameter: shaders::stage::XyzParameter,
        stage_kind: shaders::stage::StageKind,
    },
}

/// Describes a method of calculating the screen space coordinates of the painted quad.
#[derive(Clone, Debug, PartialEq)]
pub enum QuadTarget {
    Rect(Rectangle),
    Absolute([[f32; 2]; 4]),
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

/// Get the descriptors of a particular buffer plan.
#[derive(Clone, Copy, Debug)]
pub struct ImageBufferDescriptors<'a> {
    descriptor: &'a Descriptor,
    layout: &'a BufferLayout,
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
    /// Howe we mapped registers to images in the pool.
    pool_plan: ImagePoolPlan,
    /// The Bind Group layer Descriptor used in fragment shader, set=1.
    paint_group_layout: Option<usize>,
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
    vertex_shaders: HashMap<VertexShader, usize>,
    simple_quad_buffer: Option<DeviceBuffer>,
    /// The render pipeline state for staging a texture.
    staged_to_pipelines: HashMap<Texture, SimpleRenderPipeline>,
    /// The render pipeline state for undoing staging a texture.
    staged_from_pipelines: HashMap<Texture, SimpleRenderPipeline>,
    /// The allocate binary data for runtime execution.
    binary_data: Vec<u8>,
    /// The texture operands collected for the next render preparation.
    operands: Vec<Texture>,

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum VertexShader {
    Noop,
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
// FIXME: ideally we only have instructions we use.
// And we should use all of these for optimizations.
#[allow(unused)]
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
    /// Sparse entries that are not at their natural position.
    pub sparse: Vec<(u32, BindingResource)>,
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
    pub content: BufferInitContent,
    pub usage: BufferUsage,
}

#[derive(Debug)]
pub(crate) enum BufferInitContent {
    Owned(Vec<u8>),
    /// The buffer init data is from the program 'data segment'.
    Defer {
        start: usize,
        end: usize,
    },
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

#[derive(Clone, Debug)]
pub(crate) struct ImageDescriptor {
    pub size: (NonZeroU32, NonZeroU32),
    pub format: wgpu::TextureFormat,
    pub staging: Option<StagingDescriptor>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct StagingDescriptor {
    pub(crate) parameter: shaders::stage::XyzParameter,
    pub(crate) stage_kind: shaders::stage::StageKind,
}

impl ImageDescriptor {
    fn to_texture(&self) -> TextureDescriptor {
        TextureDescriptor {
            size: self.size,
            format: self.format,
            usage: TextureUsage::Attachment,
        }
    }

    fn to_staging_texture(&self) -> Option<TextureDescriptor> {
        self.staging.map(|staging| TextureDescriptor {
            size: self.size,
            format: staging.stage_kind.texture_format(),
            usage: TextureUsage::Staging,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum TextureUsage {
    /// Copy Dst + Sampled
    DataIn,
    /// Copy Src + Render Attachment
    DataOut,
    /// A storage texture
    /// Copy Src/Dst + Sampled + Render Attachment
    Attachment,
    /// A staging texture
    /// Copy Src/Dst + Storage.
    Staging,
    /// A texture which we never reach from.
    /// Sampled + Render Attachment
    Transient,
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
#[derive(Debug, PartialEq, Eq, Hash)]
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
    // FIXME: turn this warning on to find things to implement.
    // #[deprecated = "We should strive to remove these"]
    NotYetImplemented,
}

/// Something won't work with this program and pool combination, no matter the amount of
/// configuration.
#[derive(Debug)]
pub struct MismatchError {}

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
    pipeline_target: PipelineTarget,
    /// Bind data for (set 0, binding 0).
    vertex_bind_data: BufferBind<'data>,
    /// Texture for (set 1, binding 0)
    fragment_texture: TextureBind,
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

enum TextureBind {
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

#[derive(Clone, Copy)]
struct SimpleRenderPipeline {
    pipeline: usize,
    buffer: DeviceBuffer,
    group: Option<usize>,
    vertex_bind: Option<usize>,
    vertices: u32,
    fragment_bind: Option<usize>,
}

impl ImageBufferPlan {
    pub(crate) fn allocate_for(
        &mut self,
        desc: &Descriptor,
        _: Range<usize>,
    ) -> ImageBufferAssignment {
        // FIXME: we could de-duplicate textures using liveness information.
        let texture = Texture(self.texture.len());
        self.texture.push(desc.clone());
        let buffer = Buffer(self.buffer.len());
        self.buffer.push(desc.layout.clone());
        self.by_layout.insert(desc.layout.clone(), texture);
        let assigned = ImageBufferAssignment { buffer, texture };
        self.by_register.push(assigned);
        assigned
    }

    pub(crate) fn get(&self, idx: Register) -> Result<ImageBufferAssignment, LaunchError> {
        self.by_register
            .get(idx.0)
            .ok_or_else(|| LaunchError::InternalCommandError(line!()))
            .map(ImageBufferAssignment::clone)
    }

    pub(crate) fn get_info(
        &self,
        idx: Register,
    ) -> Result<ImageBufferDescriptors<'_>, LaunchError> {
        let assigned = self.get(idx)?;
        Ok(self.describe(&assigned))
    }

    pub(crate) fn describe(&self, assigned: &ImageBufferAssignment) -> ImageBufferDescriptors<'_> {
        ImageBufferDescriptors {
            descriptor: &self.texture[assigned.texture.0],
            layout: &self.buffer[assigned.buffer.0],
        }
    }
}

impl ImagePoolPlan {
    pub(crate) fn choose_output(&self, pool: &mut Pool, desc: &Descriptor) -> PoolKey {
        let mut entry = pool.declare(desc.clone());
        entry.host_allocate();
        entry.key()
    }

    pub(crate) fn get(&self, idx: Register) -> Result<PoolKey, LaunchError> {
        self.plan
            .get(&idx)
            .ok_or_else(|| LaunchError::InternalCommandError(line!()))
            .map(PoolKey::clone)
    }

    pub(crate) fn get_texture(&self, idx: Register) -> Option<Texture> {
        let key = self.plan.get(&idx)?;
        self.buffer.get(key).cloned()
    }
}

impl Program {
    /// Choose an applicable adapter from one of the presented ones.
    pub fn choose_adapter(
        &self,
        mut from: impl Iterator<Item = wgpu::Adapter>,
    ) -> Result<wgpu::Adapter, MismatchError> {
        #[allow(non_snake_case)]
        let ALL_TEXTURE_USAGE: wgpu::TextureUsage = wgpu::TextureUsage::COPY_DST
            | wgpu::TextureUsage::COPY_SRC
            | wgpu::TextureUsage::SAMPLED
            | wgpu::TextureUsage::RENDER_ATTACHMENT;

        while let Some(adapter) = from.next() {
            // FIXME: check limits.
            // FIXME: collect required texture formats from `self.textures`
            let basic_format =
                adapter.get_texture_format_features(wgpu::TextureFormat::Rgba8UnormSrgb);
            if !basic_format.allowed_usages.contains(ALL_TEXTURE_USAGE) {
                continue;
            }

            from.for_each(drop);
            return Ok(adapter);
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
    pub fn launch<'pool>(&'pool self, pool: &'pool mut Pool) -> Launcher<'pool> {
        // Create empty bind assignments as a start, with respective layouts.
        let binds = self
            .textures
            .texture
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
    pub fn bind(mut self, Register(reg): Register, img: PoolKey) -> Result<Self, LaunchError> {
        if self.pool.entry(img).is_none() {
            return Err(LaunchError::InternalCommandError(line!()));
        }

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
    pub fn bind_remaining_outputs(mut self) -> Result<Self, LaunchError> {
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
                    return Err(LaunchError::InternalCommandError(line!()));
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
                    // eprintln!("Output {:?} to {:?}", src, dst);
                    // Identify if we need to transform the texture from the internal format to the
                    // one actually chosen for this texture.
                    encoder.copy_staging_to_buffer(src)?;
                    encoder.copy_buffer_to_output(src, dst)?;
                }
                &High::PushOperand(texture) => {
                    encoder.copy_staging_to_texture(texture)?;
                    encoder.push_operand(texture)?;
                }
                High::Construct { dst, fn_ } => {
                    let dst_texture = match dst {
                        Target::Discard(texture) | Target::Load(texture) => *texture,
                    };

                    encoder.ensure_allocate_texture(dst_texture)?;
                    let dst_view = encoder.texture_view(dst_texture)?;

                    let ops = match dst {
                        Target::Discard(_) => {
                            wgpu::Operations {
                                // TODO: we could let choose a replacement color..
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLUE),
                                store: true,
                            }
                        }
                        Target::Load(_) => wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: true,
                        },
                    };

                    let attachment = ColorAttachmentDescriptor {
                        texture_view: dst_view,
                        ops,
                    };

                    let render = encoder.prepare_render(fn_, dst_texture)?;

                    // TODO: we need to remember the attachment format here.
                    // This is need to to automatically construct the shader pipeline.
                    encoder.push(Low::BeginCommands)?;
                    encoder.push(Low::BeginRenderPass(RenderPassDescriptor {
                        // FIXME: allocation?
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
                    encoder.copy_texture_to_staging(dst_texture)?;
                }
                High::Copy { src, dst } => {
                    let &RegisterMap { buffer: source_buffer, ref buffer_layout, .. } =
                        encoder.allocate_register(*src)?;
                    let size = buffer_layout.u64_len();
                    let target_buffer = encoder.allocate_register(*dst)?.buffer;

                    encoder.copy_staging_to_buffer(*src)?;

                    encoder.push(Low::BeginCommands)?;
                    encoder.push(Low::CopyBufferToBuffer {
                        source_buffer,
                        size,
                        target_buffer,
                    })?;
                    encoder.push(Low::EndCommands)?;
                    encoder.push(Low::RunTopCommand)?;

                    encoder.copy_buffer_to_staging(*dst)?;
                }
            }
        }

        let mut buffers = self.binds;
        encoder.extract_buffers(&mut buffers, &mut self.pool)?;

        // Unbalanced operands shouldn't happen.
        // This is part of validation layer but cheap and we always do it.
        if !encoder.operands.is_empty() {
            eprintln!("{:?}", encoder.operands.as_slice());
            return Err(LaunchError::InternalCommandError(line!()));
        }

        let init = run::InitialState {
            instructions: encoder.instructions,
            device,
            queue,
            buffers,
            binary_data: encoder.binary_data,
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

    fn extract_buffers(
        &self,
        buffers: &mut Vec<ImageData>,
        pool: &mut Pool,
    ) -> Result<(), LaunchError> {
        for (&pool_key, &texture) in &self.pool_plan.buffer {
            let mut entry = pool
                .entry(pool_key)
                .ok_or_else(|| LaunchError::InternalCommandError(line!()))?;
            let buffer = &mut buffers[texture.0];

            // Decide how to retrieve this image from the pool.
            if buffer.as_bytes().is_none() {
                // Just take the buffer if we are allowed to...
                if entry.meta().no_read {
                    entry.swap(buffer);
                } else if let Some(copy) = entry.host_copy() {
                    *buffer = ImageData::Host(copy);
                } else {
                    // Would need to copy from the GPU.
                    return Err(LaunchError::InternalCommandError(line!()));
                }

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
            Low::RunBotToTop(num) | Low::RunTopToBot(num) => {
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
                    Color::Xyz {
                        // Match only that which is necessary to get the right numbers in the shader.
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
                    Color::Xyz {
                        // Match only that which is necessary to get the right numbers in the shader.
                        transfer: Transfer::Linear,
                        ..
                    },
            } => wgpu::TextureFormat::Rgba8Unorm,
            Texel {
                block: Block::Pixel,
                samples,
                color: Color::Xyz { transfer, .. },
            } => {
                let parameter = shaders::stage::XyzParameter {
                    transfer,
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
            _ => return Err(LaunchError::InternalCommandError(line!())),
        };

        Ok(ImageDescriptor {
            format,
            staging,
            size,
        })
    }

    fn push_operand(&mut self, texture: Texture) -> Result<(), LaunchError> {
        self.operands.push(texture);
        Ok(())
    }

    // We must trick the borrow checker here..
    fn allocate_register(&mut self, idx: Register) -> Result<&RegisterMap, LaunchError> {
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

        if let Some(_) = self.register_map.get(&idx) {
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
        let staging = if let Some(staging) = self.staging_map.get(&reg_texture) {
            Some(staging.device)
        } else {
            None
        };

        let map_entry = RegisterMap {
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
        let in_map = self.register_map.entry(idx).or_insert(map_entry.clone());
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

    fn ensure_allocate_texture(
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
                    format: staging.clone(),
                    stage_kind: st_parameter.stage_kind,
                    parameter: st_parameter.parameter.clone(),
                    temporary_attachment_buffer_for_encoding_remove_if_possible: fallback,
                },
            );
        }

        Ok(texture)
    }

    fn ingest_image_data(&mut self, idx: Register) -> Result<Texture, LaunchError> {
        let source_key = self.pool_plan.get(idx)?;
        let texture = self.buffer_plan.get(idx)?.texture;
        self.pool_plan.buffer.entry(source_key).or_insert(texture);
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

        let target_texture = if let Some(staging) = regmap.staging {
            staging
        } else {
            // .… or directly to the target buffer if we have no staging.
            regmap.texture
        };

        // eprintln!("!!! Copying {:?}: to {:?}", idx, target_texture);

        self.push(Low::BeginCommands)?;
        self.push(Low::CopyBufferToTexture {
            source_buffer: regmap.buffer,
            source_layout: regmap.buffer_layout,
            offset: (0, 0),
            size,
            target_texture,
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
        if let Some(staging) = self.staging_map.get(&idx) {
            // eprintln!("{} {:?}", idx.0, staging);
            // Try to use the cached version of this pipeline.
            let pipeline = if let Some(pipeline) = self.staged_to_pipelines.get(&idx) {
                pipeline.clone()
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
    fn copy_texture_to_staging(&mut self, idx: Texture) -> Result<(), LaunchError> {
        if let Some(staging) = self.staging_map.get(&idx) {
            let texture = staging.temporary_attachment_buffer_for_encoding_remove_if_possible;

            // eprintln!("{} {:?}", idx.0, staging);
            // Try to use the cached version of this pipeline.
            let pipeline = if let Some(pipeline) = self.staged_from_pipelines.get(&idx) {
                pipeline.clone()
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
                self.texture_views += 1;
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
            }))?;
            self.render(pipeline)?;
            self.push(Low::EndRenderPass)?;
            self.push(Low::EndCommands)?;

            self.push(Low::RunTopCommand)?;
        }

        Ok(())
    }

    /// Copy from texture to the memory buffer.
    fn copy_staging_to_buffer(&mut self, idx: Register) -> Result<(), LaunchError> {
        let regmap = self.allocate_register(idx)?.clone();
        let size = self.buffer_plan.get_info(idx)?.descriptor.size();

        let source_texture = if let Some(staging) = regmap.staging {
            staging
        } else {
            regmap.texture
        };

        // eprintln!("!!! Copying {:?}: from {:?}", idx, source_texture);

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
    fn copy_buffer_to_output(&mut self, idx: Register, dst: Register) -> Result<(), LaunchError> {
        let regmap = self.allocate_register(idx)?.clone();
        let target_image = self.ingest_image_data(dst)?;

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
        let texture = self
            .texture_map
            .get(&dst)
            .ok_or_else(|| LaunchError::InternalCommandError(line!()))?
            .device;

        let descriptor = TextureViewDescriptor { texture };

        self.instructions.extend_one(Low::TextureView(descriptor));
        let id = self.texture_views;
        self.texture_views += 1;

        // eprintln!("Texture {:?} (Device {:?}) in View {:?}", dst, texture, id);

        Ok(id)
    }

    fn make_quad_bind_group(&mut self) -> usize {
        let bind_group_layouts = &mut self.bind_group_layouts;
        let instructions = &mut self.instructions;
        *self.quad_group_layout.get_or_insert_with(|| {
            let descriptor = BindGroupLayoutDescriptor {
                entries: vec![wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(2 * 8 * 4),
                        ty: wgpu::BufferBindingType::Uniform,
                    },
                    count: None,
                }],
            };

            instructions.extend_one(Low::BindGroupLayout(descriptor));
            let descriptor_id = *bind_group_layouts;
            *bind_group_layouts += 1;
            descriptor_id
        })
    }

    fn make_generic_fragment_bind_group(&mut self) -> usize {
        let bind_group_layouts = &mut self.bind_group_layouts;
        let instructions = &mut self.instructions;
        *self.fragment_data_group_layout.get_or_insert_with(|| {
            let descriptor = BindGroupLayoutDescriptor {
                entries: vec![wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::FRAGMENT,
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
            descriptor_id
        })
    }

    fn make_paint_group_layout(&mut self, count: usize) -> usize {
        let bind_group_layouts = &mut self.bind_group_layouts;
        let instructions = &mut self.instructions;
        *self.paint_group_layout.get_or_insert_with(|| {
            let mut entries = vec![
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Sampler {
                        filtering: true,
                        comparison: false,
                    },
                    count: None,
                },
            ];

            for i in 0..count {
                entries.push(wgpu::BindGroupLayoutEntry {
                    binding: 1 + i as u32,
                    visibility: wgpu::ShaderStage::FRAGMENT,
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
            descriptor_id
        })
    }

    fn make_stage_group(&mut self, binding: u32) -> usize {
        use shaders::stage::StageKind;
        let bind_group_layouts = &mut self.bind_group_layouts;
        let instructions = &mut self.instructions;

        // For encoding we have two extra bindings, sampler and in_texture.
        let encode: bool = binding > StageKind::ALL.len() as u32;

        *self.stage_group_layout.entry(binding).or_insert_with(|| {
            let mut entries = vec![];
            for (num, kind) in StageKind::ALL.iter().enumerate() {
                let i = num as u32;
                if i != binding {
                    continue;
                }

                entries.push(wgpu::BindGroupLayoutEntry {
                    binding: i,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadOnly,
                        format: kind.texture_format(),
                        view_dimension: wgpu::TextureViewDimension::D2,
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
                    visibility: wgpu::ShaderStage::FRAGMENT,
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
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                });

                entries.push(wgpu::BindGroupLayoutEntry {
                    binding: 33,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Sampler {
                        filtering: true,
                        comparison: false,
                    },
                    count: None,
                });
            }

            let descriptor = BindGroupLayoutDescriptor { entries };
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
            TextureBind::Textures(0) => {},
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

            *self.paint_pipeline_layout.get_or_insert_with(|| {
                let descriptor = PipelineLayoutDescriptor {
                    bind_group_layouts,
                    push_constant_ranges: &[],
                };

                instructions.extend_one(Low::PipelineLayout(descriptor));
                let descriptor_id = *layouts;
                *layouts += 1;
                descriptor_id
            })
        } else {
            bind_group_layouts.push(self.make_generic_fragment_bind_group());

            let layouts = &mut self.pipeline_layouts;
            let instructions = &mut self.instructions;

            let descriptor = PipelineLayoutDescriptor {
                bind_group_layouts,
                push_constant_ranges: &[],
            };

            instructions.extend_one(Low::PipelineLayout(descriptor));
            let descriptor_id = *layouts;
            *layouts += 1;
            descriptor_id
        }
    }

    fn shader(&mut self, desc: ShaderDescriptor) -> Result<usize, LaunchError> {
        self.instructions.extend_one(Low::Shader(desc));
        let idx = self.shaders;
        self.shaders += 1;
        Ok(idx)
    }

    fn fragment_shader(
        &mut self,
        kind: Option<shaders::FragmentShaderKey>,
        source: Cow<'static, [u32]>,
    ) -> Result<usize, LaunchError> {
        if let Some(&shader) = kind.and_then(|k| self.fragment_shaders.get(&k)) {
            return Ok(shader);
        }

        self.shader(ShaderDescriptor {
            name: "",
            flags: wgpu::ShaderFlags::empty(),
            source_spirv: source,
        })
    }

    fn vertex_shader(
        &mut self,
        kind: Option<VertexShader>,
        source: Cow<'static, [u32]>,
    ) -> Result<usize, LaunchError> {
        if let Some(&shader) = kind.and_then(|k| self.vertex_shaders.get(&k)) {
            return Ok(shader);
        }

        self.shader(ShaderDescriptor {
            name: "",
            flags: wgpu::ShaderFlags::empty(),
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
        let layout = self.make_paint_layout(desc);
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

    fn make_sampler(&mut self, descriptor: SamplerDescriptor) -> usize {
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
        let descriptor = BindGroupDescriptor {
            layout_idx: self.make_paint_group_layout(count),
            entries: entries,
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
        let texture = self
            .staging_map
            .get(&texture)
            .ok_or_else(|| LaunchError::InternalCommandError(line!()))?
            .device;

        // FIXME: could be cached.
        let image_id = self.texture_views;
        self.push(Low::TextureView(TextureViewDescriptor { texture }))?;

        let mut sparse = vec![(binding, BindingResource::TextureView(image_id))];

        // For encoding we have two extra bindings, sampler and in_texture.
        if let Some(view) = view {
            // FIXME: unnecessary duplication.
            let sampler = self.make_sampler(SamplerDescriptor {
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
        layout_idx: usize,
    ) -> Result<Option<usize>, LaunchError> {
        let buffer = match bind {
            BufferBind::None => return Ok(None),
            BufferBind::Set { data } => {
                let buffer = self.buffers;
                let content = self.ingest_data(data);
                self.push(Low::BufferInit(BufferDescriptorInit {
                    content: content,
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
        Ok(Some(group))
    }

    /// Render the pipeline, after all customization and buffers were bound..
    fn prepare_simple_pipeline(
        &mut self,
        descriptor: SimpleRenderPipelineDescriptor,
    ) -> Result<SimpleRenderPipeline, LaunchError> {
        let SimpleRenderPipelineDescriptor {
            pipeline_target: _,
            vertex_bind_data,
            fragment_texture: texture,
            fragment_bind_data,
            vertex: _,
            fragment: _,
        } = &descriptor;

        let pipeline = self.simple_render_pipeline(&descriptor)?;
        let buffer = self.simple_quad_buffer();

        let group = match texture {
            TextureBind::Textures(0) => None,
            &TextureBind::Textures(count) => {
                let group = self.make_bind_group_sampled_texture(count)?;
                // eprintln!("Using Texture {:?} as group {:?}", texture, group);
                Some(group)
            }
            &TextureBind::PreComputedGroup { group, .. } => {
                // eprintln!("Using Target Group {:?}", group);
                Some(group)
            }
        };

        let vertex_layout = self.make_quad_bind_group();
        let vertex_bind = self.make_bound_buffer(descriptor.vertex_bind_data, vertex_layout)?;

        // FIXME: this builds the layout even when it is not required.
        let vertex_layout = self.make_generic_fragment_bind_group();
        let fragment_bind = self.make_bound_buffer(descriptor.fragment_bind_data, vertex_layout)?;

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
    pub const FULL_VERTEX_BUFFER: [[f32; 2]; 16] = [
        // [min_u, min_v], [0.0, 0.0],
        [0.0, 0.0], [0.0, 0.0],
        // [max_u, 0.0], [0.0, 0.0],
        [1.0, 0.0], [0.0, 0.0],
        // [max_u, max_v], [0.0, 0.0],
        [1.0, 1.0], [0.0, 0.0],
        // [min_u, max_v], [0.0, 0.0],
        [0.0, 1.0], [0.0, 0.0],

        [0.0, 0.0], [0.0, 0.0],
        [1.0, 0.0], [0.0, 0.0],
        [1.0, 1.0], [0.0, 0.0],
        [0.0, 1.0], [0.0, 0.0],
    ];

    fn render(&mut self, pipeline: SimpleRenderPipeline) -> Result<(), LaunchError> {
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
    fn prepare_render(
        &mut self,
        // The function we are using.
        function: &Function,
        // The texture we are rendering to.
        target: Texture,
    ) -> Result<SimpleRenderPipeline, LaunchError> {
        match function {
            Function::PaintToSelection { texture, selection, target: target_coords, viewport, shader } => {
                let (tex_width, tex_height) = self.texture_map[&texture].format.size;

                let vertex = self.vertex_shader(
                    Some(VertexShader::Noop),
                    shader_include_to_spirv(shaders::VERT_NOOP))?;

                let shader = shader.shader();
                let key = shader.key();
                let spirv = shader.spirv_source();

                let fragment = self.fragment_shader(key, shader_include_to_spirv_static(spirv))?;

                let buffer: [[f32; 2]; 16];
                // FIXME: there seems to be two floats padding after each vec2.
                let min_u = (selection.x as f32) / (tex_width.get() as f32);
                let max_u = (selection.max_x as f32) / (tex_width.get() as f32);
                let min_v = (selection.y as f32) / (tex_height.get() as f32);
                let max_v = (selection.max_y as f32) / (tex_height.get() as f32);

                let coords = target_coords.to_screenspace_coords(viewport);

                // std140, always pad to 16 bytes.
                buffer = [
                    [min_u, min_v], [0.0, 0.0],
                    [max_u, min_v], [0.0, 0.0],
                    [max_u, max_v], [0.0, 0.0],
                    [min_u, max_v], [0.0, 0.0],

                    coords[0], [0.0, 0.0],
                    coords[1], [0.0, 0.0],
                    coords[2], [0.0, 0.0],
                    coords[3], [0.0, 0.0],
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
                let vertex = self.vertex_shader(
                    Some(VertexShader::Noop),
                    shader_include_to_spirv(shaders::VERT_NOOP))?;

                let shader = shader.shader();
                let key = shader.key();
                let spirv = shader.spirv_source();

                let fragment = self.fragment_shader(key, shader_include_to_spirv_static(spirv))?;
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
                let vertex = self.vertex_shader(
                    Some(VertexShader::Noop),
                    shader_include_to_spirv(shaders::VERT_NOOP))?;

                let fragment = self.fragment_shader(
                    Some(shaders::FragmentShaderKey::Convert),
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
                let vertex = self.vertex_shader(
                    Some(VertexShader::Noop),
                    shader_include_to_spirv(shaders::VERT_NOOP))?;

                let fragment = self.fragment_shader(
                    Some(shaders::FragmentShaderKey::Convert),
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

    fn ingest_data(&mut self, data: &[impl bytemuck::Pod]) -> BufferInitContent {
        BufferInitContent::new(&mut self.binary_data, data)
    }
}

impl BufferInitContent {
    /// Construct a reference to data by allocating it freshly within the buffer.
    pub fn new(buf: &mut Vec<u8>, data: &[impl bytemuck::Pod]) -> Self {
        let start = buf.len();
        buf.extend_from_slice(bytemuck::cast_slice(data));
        let end = buf.len();
        BufferInitContent::Defer {
            start,
            end,
        }
    }

    /// Get a reference to the binary data, given the allocator/buffer.
    pub fn as_slice<'lt>(&'lt self, buffer: &'lt Vec<u8>) -> &'lt [u8] {
        match self {
            BufferInitContent::Owned(ref data) => &data,
            &BufferInitContent::Defer { start, end } => &buffer[start..end],
        }
    }
}

impl From<Vec<u8>> for BufferInitContent {
    fn from(vec: Vec<u8>) -> Self {
        BufferInitContent::Owned(vec)
    }
}

impl QuadTarget {
    pub(crate) fn affine(&self, transform: &RowMatrix) -> Self {
        let [a, b, c, d] = self.to_screenspace_coords(&Rectangle::with_width_height(1, 1));
        QuadTarget::Absolute([
            transform.multiply_point(a),
            transform.multiply_point(b),
            transform.multiply_point(c),
            transform.multiply_point(d),
        ])
    }

    pub(crate) fn to_screenspace_coords(&self, viewport: &Rectangle) -> [[f32; 2]; 4] {
        match self {
            QuadTarget::Rect(target) => {
                let min_a = (target.x as f32) / (viewport.width() as f32);
                let max_a = (target.max_x as f32) / (viewport.width() as f32);
                let min_b = (target.y as f32) / (viewport.height() as f32);
                let max_b = (target.max_y as f32) / (viewport.height() as f32);

                [
                    [min_a, min_b],
                    [max_a, min_b],
                    [max_a, max_b],
                    [min_a, max_b],
                ]
            }
            QuadTarget::Absolute(coord) => {
                let xy = |[cx, cy]: [f32; 2]| {
                    [
                        (cx - viewport.x as f32) / viewport.width() as f32,
                        (cy - viewport.y as f32) / viewport.height() as f32,
                    ]
                };

                [xy(coord[0]), xy(coord[1]), xy(coord[2]), xy(coord[3])]
            }
        }
    }
}

impl From<Rectangle> for QuadTarget {
    fn from(target: Rectangle) -> Self {
        QuadTarget::Rect(target)
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

impl BufferUsage {
    pub fn to_wgpu(self) -> wgpu::BufferUsage {
        use wgpu::BufferUsage as U;
        match self {
            BufferUsage::InVertices => U::COPY_DST | U::VERTEX,
            BufferUsage::DataIn => U::MAP_WRITE | U::COPY_SRC,
            BufferUsage::DataOut => U::MAP_READ | U::COPY_DST,
            BufferUsage::DataBuffer => U::STORAGE | U::COPY_SRC | U::COPY_DST,
            BufferUsage::Uniform => U::COPY_DST | U::UNIFORM,
        }
    }
}

impl LaunchError {
    #[allow(non_snake_case)]
    // FIXME: find a better error representation but it's okay for now.
    // #[deprecated = "This should be cleaned up"]
    pub(crate) fn InternalCommandError(line: u32) -> Self {
        LaunchError { line }
    }
}

/// Block on an async future that may depend on a device being polled.
pub(crate) fn block_on<F, T>(future: F, device: Option<&wgpu::Device>) -> T
where
    F: Future<Output = T> + 'static,
    T: 'static,
{
    #[cfg(target_arch = "wasm32")]
    {
        use core::cell::RefCell;
        use std::rc::Rc;

        async fn the_thing<T: 'static, F: Future<Output = T> + 'static>(
            future: F,
            buffer: Rc<RefCell<Option<T>>>,
        ) {
            let result = future.await;
            *buffer.borrow_mut() = Some(result);
        }

        let result = Rc::new(RefCell::new(None));
        let mover = Rc::clone(&result);

        wasm_bindgen_futures::spawn_local(the_thing(future, mover));

        match Rc::try_unwrap(result) {
            Ok(cell) => match cell.into_inner() {
                Some(result) => result,
                None => unreachable!("In this case we shouldn't have returned here"),
            },
            _ => unreachable!("There should be no reference to mover left"),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
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
                fn poll(
                    self: core::pin::Pin<&mut Self>,
                    ctx: &mut core::task::Context,
                ) -> core::task::Poll<F::Output> {
                    self.as_ref().device.poll(wgpu::Maintain::Poll);
                    // Ugh, noooo...
                    ctx.waker().wake_by_ref();
                    let future = unsafe { self.map_unchecked_mut(|this| &mut this.future) };
                    future.poll(ctx)
                }
            }

            async_io::block_on(DevicePolled { future, device })
        } else {
            async_io::block_on(future)
        }
    }
}
