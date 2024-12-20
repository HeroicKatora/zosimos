mod encoder;

use core::{num::NonZeroU32, ops::Range};

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use crate::buffer::{
    Block, ByteLayout, Color, Descriptor, SampleBits, SampleParts, Texel, Transfer,
};
use crate::color_matrix::RowMatrix;
use crate::command::{Rectangle, Register, RegisterKnob};
use crate::pool::{Pool, PoolKey};
use crate::{run, shaders};

use encoder::{Encoder, RegisterMap};

/// Planned out and intrinsically validated command buffer.
///
/// This does not necessarily plan out a commands of low level execution instruction set flavor.
/// This is selected based on the available device and its capabilities, which is performed during
/// launch.
pub struct Program {
    pub(crate) ops: Vec<High>,
    /// The different functions.
    pub(crate) functions: Vec<FunctionLinked>,
    /// The entry point into the program, the function by which to layout the required input and
    /// the output buffer plans.
    pub(crate) entry_index: usize,
    /// Annotates which function allocates a cacheable texture.
    pub(crate) texture_by_op: HashMap<usize, TextureDescriptor>,
    /// Annotates which function allocates a cacheable buffer.
    pub(crate) buffer_by_op: HashMap<usize, BufferDescriptor>,
    /// The maps of registers to persistent global knobs indices.
    pub(crate) knobs: HashMap<RegisterKnob, Knob>,
}

pub(crate) struct FunctionLinked {
    /// The sequence in `ops` that belongs to this function.
    pub(crate) ops: core::ops::Range<usize>,
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
    pub(crate) image_buffers: ImageBufferPlan,
    /// The register IDs that a caller will target when issuing a `Call` against the function.
    ///
    /// Utilized to validate that such calls are valid, as well as to match the arguments with
    /// their register. The register is then the stable form of reference until runtime where it is
    /// mapped to a concrete IO slot of the function, based on information from the encoder. We're
    /// assigning multiple purposes to the intermediate layers but the register is pretty clean.
    /// Just slightly slow.
    ///
    /// NOTE: this is in signature order. Not in register sort order, hence the vector instead of a
    /// set representation.
    pub(crate) signature_registers: Vec<Register>,
}

/// A high-level, device independent, translation of ops.
///
/// This is a translated version of `Op`, for consumption in the interpreter internals instead of
/// the programmer. The main difference to Op is that this is no longer masking as an SSA-form and
/// simplified to the architecture. Operations are mostly in terms of textures they require
/// already, not high-level values, and depend on actual capabilities. These functions are also
/// monomorphized over the texture size / types of their operands whereas the high level command
/// structure is planned to have generics.
///
/// In particular it will be amenable to the initial liveness analysis. This will also return the
/// _available_ strategies for one operation. For example, some texels can not be represented on
/// the GPU directly, depending on available formats, and need to be either processed on the CPU
/// (with SIMD hopefully) or they must be converted first, potentially in a compute shader.
#[derive(Clone, Debug)]
pub(crate) enum High {
    /// Assign a texture id to an input with given descriptor.
    /// This instructs the program to insert instructions that load the image from the input in the
    /// pool into the associated texture buffer.
    Input(Register),
    /// Designate the ith textures as output n, according to the position in sequence of outputs.
    Output {
        /// The source register/texture/buffers.
        src: Register,
        /// The target texture.
        dst: Register,
    },
    Render {
        /// The source register/texture/buffers.
        src: Register,
        /// The target texture.
        dst: Register,
    },
    /// Add an additional texture operand to the next operation.
    PushOperand(Texture),
    /// Call a function on the currently prepared operands.
    DrawInto {
        dst: Target,
        fn_: Initializer,
    },
    /// Create all the state for a texture, without doing anything in it.
    Uninit {
        dst: Target,
    },
    WriteInto {
        dst: Buffer,
        fn_: BufferWrite,
    },
    /// Last phase marking a register as done.
    /// This is emitted after the Command defining the register has been translated.
    Done(Register),
    /// Copy binary data from a buffer to another.
    Copy {
        src: Register,
        dst: Register,
    },
    /// Push one high-level function marker.
    StackPush(Frame),
    /// Pop a high-level function marker.
    StackPop,
    Call {
        function: Function,
        /// Initial IO state of the callee.
        image_io_buffers: Arc<[CallBinding]>,
    },
}

/// The target image texture of a paint operation (pipeline).
#[derive(Clone, Copy, Debug)]
pub(crate) enum Target {
    /// The data in the texture is to be discarded.
    Discard(Texture),
    /// The data in the texture must be loaded.
    Load(Texture),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ParameterizedFragment {
    pub(crate) invocation: shaders::FragmentShaderInvocation,
    pub(crate) knob: KnobUser,
}

/// A data portion that is dynamically changed, in some way.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum KnobUser {
    /// The data is static.
    None,

    /// The data is optionally overridden as a runtime parameter.
    Runtime(Knob),

    /// The data is copied from an existing previously filled buffer.
    Buffer { buffer: Buffer, range: Range<u64> },
}

/// Describes a function call in more common terms.
///
/// The bind sets follow the logic that functions will use the same setup for the vertex shader to
/// pain a source rectangle at a target rectangle location. Then the required images are bound. And
/// finally we have the various dynamic/uniform parameters of the differing functions.
///
/// A single command might be translated to multiple functions.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Initializer {
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
        shader: ParameterizedFragment,
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
    PaintFullScreen { shader: ParameterizedFragment },
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

/// An operation that modifies the contents of a buffer.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum BufferWrite {
    Zero,

    Put {
        placement: Range<usize>,
        data: Arc<[u8]>,
        knob: Option<Knob>,
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
    pub(crate) by_register: HashMap<Register, RegisterAssignment>,
    pub(crate) by_layout: HashMap<ByteLayout, Texture>,
}

/// The *definitional* size of a buffer. On the device each can be described as a pure linear u64
/// size but here we also retain the derivation.
#[derive(Clone, Debug)]
pub(crate) enum BufferLayout {
    Linear(u64),
    Texture(ByteLayout),
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
pub enum RegisterAssignment {
    Image(ImageBufferAssignment),
    Buffer(ByteBufferAssignment),
}

#[derive(Clone, Copy, Debug)]
pub struct ImageBufferAssignment {
    pub(crate) texture: Texture,
    pub(crate) buffer: Buffer,
}

#[derive(Clone, Copy, Debug)]
pub struct ByteBufferAssignment {
    pub(crate) buffer: Buffer,
}

/// Get the descriptors of a particular buffer plan.
#[derive(Clone, Copy, Debug)]
// We allow the byte layout to be unused, it's proof of work that there is one. This is in a sense
// restricting the code to do sensible constructions. It'd be thinkable that there should be a form
// of validation which later compares that layout to a real, achieved one. Especially with further
// implementation of generic monomorphization this may become important again to catch some
// hard-to-observe / hard-to-root-cause-analyze bugs.
#[allow(dead_code)]
pub struct ImageBufferDescriptors<'a> {
    pub(crate) descriptor: &'a Descriptor,
    pub(crate) layout: &'a ByteLayout,
}

#[derive(Clone, Debug)]
pub(crate) struct Frame {
    pub(crate) name: String,
}

#[derive(Clone, Debug)]
#[non_exhaustive]
pub(crate) struct KnobDescriptor {
    /// The range of the initial data in the binary.
    pub range: Range<usize>,
}

/// A gpu buffer associated with an image buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceBuffer(pub(crate) usize);

/// A gpu texture associated with an image buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceTexture(pub(crate) usize);

/// Identifies one layout based buffer in the render pipeline, by an index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Buffer(pub(crate) usize);

/// Identifies one sequence of instructions that operate on the same stack.
///
/// Arguments and results are passed by the initial state of the stack, including a portion of IO
/// buffers that can be textures and pre-allocated buffers from the GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Function(pub(crate) usize);

/// Identifies one descriptor based resource in the render pipeline, by an index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Texture(pub(crate) usize);

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct Instruction(pub(crate) usize);

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct Event(pub(crate) usize);

/// Identifies binary data which can be updated from the host as execution parameter.
///
/// This will map to a region inside the assembled program's blob data. When a buffer is changed on
/// the host side, the execution will automatically ensure the used buffer state is updated to the
/// bytes indicated. Unchanged bytes will be left or initialized to the value in the binary.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct Knob(pub(crate) usize);

/// A map of features which we may use during encoding.
#[derive(Clone, Debug)]
pub struct Capabilities {
    features: wgpu::Features,
    limits: wgpu::Limits,
}

#[derive(Debug)]
#[allow(unused)]
pub struct LaunchError {
    kind: LaunchErrorKind,
}

#[derive(Debug)]
pub enum LaunchErrorKind {
    FromLine(u32),
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
    /// Create (and store) a view on a texture.
    /// Due to internal restrictions this isn't really helpful.
    TextureView(TextureViewDescriptor),
    /// Create (and store) a view on a render output.
    RenderView(Register),
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
        stages: wgpu::ShaderStages,
        offset: u32,
        data: Cow<'static, [u8]>,
    },

    // Render execution commands.
    /// Run one command buffer previously created.
    RunTopCommand,
    /// Run multiple commands at once.
    RunBotToTop(usize),
    /// Read a buffer into host image data.
    /// Will map the buffer then do row-wise writes.
    WriteImageToBuffer {
        source_image: Texture,
        offset: (u32, u32),
        size: (u32, u32),
        target_buffer: DeviceBuffer,
        target_layout: ByteLayout,
        copy_dst_buffer: DeviceBuffer,
        // FIXME: maybe this should be issued as a separate instruction with only the host-relevant
        // parameters, then mark both the `WriteImageToBuffer` and `CopyBufferToBuffer` with the
        // event.
        write_event: Event,
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
        source_layout: ByteLayout,
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
        target_layout: ByteLayout,
    },

    // FIXME: requires a *dynamic* alignment condition. We should thus have an emulated version of
    // this copy for buffers that can be bound? Unsure exactly what to make it of it.
    CopyBufferToBuffer {
        source_buffer: DeviceBuffer,
        source: u64,
        target_buffer: DeviceBuffer,
        target: u64,
        size: u64,
    },

    ZeroBuffer {
        start: u64,
        size: u64,
        target_buffer: DeviceBuffer,
    },

    /// Read a buffer into host image data.
    /// Will map the buffer then do row-wise reads.
    ReadBuffer {
        /// Equivalent buffer we're allowed to map.
        source_buffer: DeviceBuffer,
        /// Layout of the buffers with this data.
        source_layout: ByteLayout,
        offset: (u32, u32),
        size: (u32, u32),
        target_image: Texture,
        /// The buffer that we're allowed to use a COPY_SRC.
        copy_src_buffer: DeviceBuffer,
    },

    StackFrame(run::Frame),
    StackPop,
    AssertBuffer {
        buffer: DeviceBuffer,
        info: String,
    },
    AssertTexture {
        buffer: DeviceBuffer,
        info: String,
    },

    Call {
        function: Function,
        io_buffers: Vec<CallImageArgument>,
    },
    Return,
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

#[derive(Debug)]
pub(crate) enum CallBinding {
    /// Texture that is initialized on entry, and just copied.
    InTexture {
        texture: Texture,
        register: Register,
    },
    /// Texture that gets initialized by this call.
    OutTexture {
        texture: Texture,
        register: Register,
    },
}

/// FIXME: name... is it appropriate to use the same component `Call` for High and Low since this
/// confuses any readers trying to match the structs that make up the attributes.
#[derive(Debug)]
pub(crate) struct CallImageArgument {
    pub buffer: DeviceBuffer,
    pub descriptor: Descriptor,
    /// FIXME: using `Texture` with multiple uses hurts here. See the complaint of this assignment
    /// in `program.rs`. But here it is even slightly worse as we're requiring there to *be* a
    /// `Texture` for all arguments even those which do not realistically are textures.
    ///
    /// Also we had a bug where we tried passing the `Texture` but as it was allocated at the
    /// caller which is the wrong index. (This does not necessarily crash). In the current system
    /// the `Texture` assignment requires a full lowering of the callee, only after which the
    /// texture assignment is finalized. However this requires either the call-graph to be acyclic
    /// or that we fixup the assignment by an indirection. We choose the indirection storing only
    /// the target register and using the later io_map.
    pub in_io: Register,
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

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) struct RenderPipelineKey {
    pub pipeline_flavor: PipelineLayoutKey,
    pub vertex_module: ShaderDescriptorKey,
    pub vertex_entry: &'static str,
    pub fragment_module: ShaderDescriptorKey,
    pub fragment_entry: &'static str,
    pub primitive: PrimitiveState,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) enum PipelineLayoutKey {
    /// The pipeline layout is uniquely determined for its modules / primitive.
    Simple,
}

#[derive(Debug)]
pub(crate) struct VertexState {
    pub vertex_module: usize,
    pub entry_point: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
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
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) struct BufferDescriptor {
    pub size: wgpu::BufferAddress,
    pub usage: BufferUsage,
}

/// For constructing a new buffer, of anonymous memory.
#[derive(Debug)]
pub(crate) struct BufferDescriptorInit {
    pub content: Range<usize>,
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
pub(crate) struct FunctionFrame {
    pub(crate) range: core::ops::Range<usize>,
    /// The IO descriptors that must be defined in the call, as `image_io_buffers`.
    pub(crate) io: Arc<[Descriptor]>,
    pub(crate) io_map: Arc<run::IoMap>,
}

#[derive(Debug)]
pub(crate) struct BufferInitContentBuilder<'trgt> {
    buf: &'trgt mut Vec<u8>,
    start: usize,
}

// Debug manually implemented for performance and readability.
pub(crate) struct ShaderDescriptor {
    pub name: &'static str,
    pub source_spirv: Cow<'static, [u32]>,
    pub key: Option<ShaderDescriptorKey>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) enum ShaderDescriptorKey {
    Fragment(shaders::FragmentShaderKey),
    Vertex(shaders::VertexShader),
}

impl From<shaders::FragmentShaderKey> for ShaderDescriptorKey {
    fn from(key: shaders::FragmentShaderKey) -> Self {
        ShaderDescriptorKey::Fragment(key)
    }
}

impl From<shaders::VertexShader> for ShaderDescriptorKey {
    fn from(key: shaders::VertexShader) -> Self {
        ShaderDescriptorKey::Vertex(key)
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) enum BufferUsage {
    /// Map Write + Vertex
    InVertices,
    /// Map Write + Copy Src
    DataIn,
    /// Map Read + Copy Dst
    DataOut,
    /// Storage + Copy Src/Dst
    DataBuffer,
    /// Uniform + Copy Dst
    Uniform,
}

/// For constructing a new texture.
/// Ignores mip level, sample count, and some usages.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) struct TextureDescriptor {
    /// The size, not that zero-sized textures have to be emulated by us.
    pub size: (NonZeroU32, NonZeroU32),
    pub format: wgpu::TextureFormat,
    pub usage: TextureUsage,
}

/// Describe an image for the purpose of determining resource we want to associate with it.
#[derive(Clone, Debug)]
pub(crate) struct ImageDescriptor {
    pub size: (NonZeroU32, NonZeroU32),
    pub format: wgpu::TextureFormat,
    pub staging: Option<StagingDescriptor>,
}

/// The information on _how_ to stage (convert a texel-encoding to linear color) a texture.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StagingDescriptor {
    pub(crate) parameter: shaders::stage::XyzParameter,
    pub(crate) stage_kind: shaders::stage::StageKind,
}

impl ImageDescriptor {
    pub(crate) fn new(descriptor: &Descriptor) -> Result<Self, LaunchError> {
        fn validate_size(layout: &ByteLayout) -> Option<(NonZeroU32, NonZeroU32)> {
            Some((
                NonZeroU32::new(layout.width)?,
                NonZeroU32::new(layout.height)?,
            ))
        }

        let size = validate_size(&descriptor.layout)
            .ok_or_else(|| LaunchError::InternalCommandError(line!()))?;
        let mut staging = None;

        let format = match (&descriptor.texel, &descriptor.color) {
            (
                Texel {
                    block: Block::Pixel,
                    bits: SampleBits::UInt8x4,
                    parts: SampleParts::RgbA,
                },
                Color::Rgb {
                    transfer: Transfer::Srgb,
                    ..
                },
            ) => wgpu::TextureFormat::Rgba8UnormSrgb,
            (
                Texel {
                    block: Block::Pixel,
                    bits: SampleBits::UInt8x4,
                    parts: SampleParts::BgrA,
                },
                Color::Rgb {
                    transfer: Transfer::Srgb,
                    ..
                },
            ) => wgpu::TextureFormat::Bgra8UnormSrgb,
            (
                Texel {
                    block: Block::Pixel,
                    bits: SampleBits::UInt8x4,
                    parts: SampleParts::RgbA,
                },
                Color::Rgb {
                    transfer: Transfer::Linear,
                    ..
                },
            ) => wgpu::TextureFormat::Rgba8Unorm,
            (
                Texel {
                    block: Block::Pixel,
                    bits: SampleBits::UInt8x4,
                    parts: SampleParts::BgrA,
                },
                Color::Rgb {
                    transfer: Transfer::Linear,
                    ..
                },
            ) => wgpu::TextureFormat::Bgra8Unorm,
            (
                Texel {
                    block: Block::Pixel,
                    bits,
                    parts,
                },
                Color::Rgb { transfer, .. },
            )
            | (
                Texel {
                    block: Block::Pixel,
                    bits,
                    parts,
                },
                Color::Scalars { transfer, .. },
            ) => {
                let parameter = shaders::stage::XyzParameter {
                    transfer: shaders::stage::Transfer::Rgb(*transfer),
                    bits: *bits,
                    parts: *parts,
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
            (
                Texel {
                    block: Block::Pixel,
                    bits,
                    parts: parts @ (SampleParts::LchA | SampleParts::LabA),
                },
                Color::Oklab,
            ) => {
                let parameter = shaders::stage::XyzParameter {
                    transfer: match *parts {
                        SampleParts::LchA => shaders::stage::Transfer::LabLch,
                        SampleParts::LabA => shaders::stage::Transfer::Rgb(Transfer::Linear),
                        _ => return Err(LaunchError::InternalCommandError(line!())),
                    },
                    parts: SampleParts::LchA,
                    bits: *bits,
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
            // FIXME: very, very duplicate code.
            (
                Texel {
                    block: Block::Pixel,
                    bits,
                    parts: parts @ (SampleParts::LchA | SampleParts::LabA),
                },
                Color::SrLab2 { .. },
            ) => {
                let parameter = shaders::stage::XyzParameter {
                    transfer: match *parts {
                        SampleParts::LchA => shaders::stage::Transfer::LabLch,
                        SampleParts::LabA => shaders::stage::Transfer::Rgb(Transfer::Linear),
                        _ => return Err(LaunchError::InternalCommandError(line!())),
                    },
                    parts: SampleParts::LchA,
                    bits: *bits,
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

    pub(crate) fn to_texture(&self) -> TextureDescriptor {
        TextureDescriptor {
            size: self.size,
            format: self.format,
            usage: TextureUsage::Attachment,
        }
    }

    pub(crate) fn to_staging_texture(&self) -> Option<TextureDescriptor> {
        self.staging.map(|staging| TextureDescriptor {
            size: staging.stage_kind.stage_size(self.size),
            format: staging.stage_kind.texture_format(),
            usage: TextureUsage::Staging,
        })
    }
}

/// The usage of a texture, of those we differentiate.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
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
    main: &'program FunctionLinked,
    /// The host image data for each texture (if any).
    /// Otherwise this a placeholder image.
    binds: Vec<run::Image>,
    /// Assigns images from the internal pool to registers.
    /// They may be transferred from an input pool, and conversely we assign outputs. We can use
    /// the plan to put back all images into the pool when retiring the execution.
    pool_plan: ImagePoolPlan,
}

impl ImageBufferPlan {
    pub(crate) fn alloc_texture_for(
        &mut self,
        desc: &Descriptor,
        _: Range<usize>,
        register: Register,
    ) -> ImageBufferAssignment {
        // FIXME: we could de-duplicate textures using liveness information.
        let texture = Texture(self.texture.len());
        self.texture.push(desc.clone());
        let buffer = Buffer(self.buffer.len());
        self.buffer.push(BufferLayout::Texture(desc.layout.clone()));
        self.by_layout.insert(desc.layout.clone(), texture);
        let assigned = ImageBufferAssignment { buffer, texture };
        self.by_register
            .insert(register, RegisterAssignment::Image(assigned));
        assigned
    }

    pub(crate) fn alloc_buffer_for(
        &mut self,
        len: u64,
        _: Range<usize>,
        register: Register,
    ) -> ByteBufferAssignment {
        let buffer = Buffer(self.buffer.len());
        self.buffer.push(BufferLayout::Linear(len));
        let assigned = ByteBufferAssignment { buffer };
        self.by_register
            .insert(register, RegisterAssignment::Buffer(assigned));
        assigned
    }

    pub(crate) fn get_register_resources(
        &self,
        idx: Register,
    ) -> Result<RegisterAssignment, LaunchError> {
        self.by_register
            .get(&idx)
            .ok_or_else(|| LaunchError::InternalCommandError(line!()))
            .map(RegisterAssignment::clone)
    }

    pub(crate) fn get_register_texture(&self, idx: Register) -> Result<Texture, LaunchError> {
        match self.get_register_resources(idx)? {
            RegisterAssignment::Image(image) => Ok(image.texture),
            _ => Err(LaunchError::InternalCommandError(line!())),
        }
    }

    pub(crate) fn get_info(
        &self,
        idx: Register,
    ) -> Result<ImageBufferDescriptors<'_>, LaunchError> {
        let RegisterAssignment::Image(assigned) = self.get_register_resources(idx)? else {
            return Err(LaunchError::InternalCommandError(line!()));
        };

        Ok(self.describe(&assigned))
    }

    pub(crate) fn describe(&self, assigned: &ImageBufferAssignment) -> ImageBufferDescriptors<'_> {
        ImageBufferDescriptors {
            descriptor: &self.texture[assigned.texture.0],
            layout: match &self.buffer[assigned.buffer.0] {
                BufferLayout::Texture(desc) => desc,
                _ => panic!("Image appear assigned to non-image buffer. Mixed up buffer plans?"),
            },
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

impl BufferLayout {
    fn u64_len(&self) -> u64 {
        match self {
            &BufferLayout::Linear(len) => len,
            BufferLayout::Texture(tex) => u64::from(tex.height) * tex.row_stride,
        }
    }
}

impl Program {
    pub fn describe_register(&self, reg: Register) -> Option<&'_ Descriptor> {
        let main = &self.functions[self.entry_index];
        let texture = main.image_buffers.get_info(reg).ok()?;
        Some(texture.descriptor)
    }

    /// Request an adapter, hoping to get a proper one.
    pub fn request_adapter(instance: &wgpu::Instance) -> Result<wgpu::Adapter, MismatchError> {
        let request = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        });

        let choice = run::block_on(Box::pin(request), None);
        Self::minimum_adapter(choice.into_iter())
    }

    pub fn request_compatible_adapter(
        instance: &wgpu::Instance,
        options: &wgpu::RequestAdapterOptions,
    ) -> Result<wgpu::Adapter, MismatchError> {
        let request = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            ..*options
        });

        let choice = run::block_on(Box::pin(request), None);
        Self::minimum_adapter(choice.into_iter())
    }

    /// Choose an applicable adapter from one of the presented ones.
    pub fn choose_adapter(
        &self,
        from: impl Iterator<Item = wgpu::Adapter>,
    ) -> Result<wgpu::Adapter, MismatchError> {
        // FIXME: no. We could derive 'trait bounds' on the system that are necessary for executing
        // the operations. If we can make sure these are purely additive.
        Program::minimum_adapter(from)
    }

    /// Select an adapter that fulfills the minimum requirements for running programs.
    ///
    /// The library may be able to utilize any additional features on top but, following the design
    /// of `wgpu`, these need to be explicitly enabled before lowering. [WIP]: there are no actual
    /// uses of any additional features. So currently we require (a subset of WebGPU):
    ///
    /// * Generic support for `rgba8UnormSrgb`.
    /// * Load/Store textures for `luma32uint`, `rgba16uint`.
    /// * Load/Store for `rgba32uint` might allow additional texel support.
    /// * `precise` (bit-reproducible) shaders are WIP in wgpu anyways.
    ///
    /// What could be available as options in the (near/far) future:
    /// * No `PushConstants` but some shaders might benefit.
    /// * [WIP] We don't do limit checks yet. But we really should because it's handled by panic.
    /// * Timestamp Queries and Pipeline Statistics would be necessary for profiling (though only
    ///     accurate on native). This would also be optional.
    /// * `AddressModeClampToBorder` would be interesting because we'd need to emulate that right
    ///     now. However, not sure how useful.
    ///
    /// However, given the current scheme any utilization of functions with arity >= 4 would
    /// require additional opt-in as this hits the limit for number of sampled textures (that is,
    /// the minimum required limit). Luckily, we do not have any such functions yet.
    pub fn minimum_adapter(
        mut from: impl Iterator<Item = wgpu::Adapter>,
    ) -> Result<wgpu::Adapter, MismatchError> {
        #[allow(non_snake_case)]
        let ALL_TEXTURE_USAGE: wgpu::TextureUsages = wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::RENDER_ATTACHMENT;

        #[allow(non_snake_case)]
        let STAGE_TEXTURE_USAGE: wgpu::TextureUsages = wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::RENDER_ATTACHMENT;

        log::info!("Searching for fitting adapter");
        let mut adapters_search = 0;
        while let Some(adapter) = from.next() {
            log::info!("Considering {:?}", adapter.get_info());
            adapters_search += 1;
            // FIXME: check limits.
            // FIXME: collect required texture formats from `self.textures`
            let basic_format =
                adapter.get_texture_format_features(wgpu::TextureFormat::Rgba8UnormSrgb);
            if !basic_format.allowed_usages.contains(ALL_TEXTURE_USAGE) {
                log::info!("Missing basic format {:?}", basic_format);
                continue;
            }

            let storage_format = adapter.get_texture_format_features(wgpu::TextureFormat::R32Uint);
            if !storage_format.allowed_usages.contains(STAGE_TEXTURE_USAGE) {
                log::info!("Missing basic format {:?}", storage_format);
                continue;
            }

            from.for_each(drop);
            return Ok(adapter);
        }

        if adapters_search == 0 {
            log::warn!("No adapters considered!");
        }

        Err(MismatchError {})
    }

    /// Return a descriptor for a device that's capable of executing the program.
    pub fn device_descriptor(&self) -> wgpu::DeviceDescriptor<'static> {
        // FIXME: no. We could derive 'trait bounds' on the system that are necessary for executing
        // the operations. If we can make sure these are purely additive.
        Self::minimal_device_descriptor()
    }

    pub fn minimal_device_descriptor() -> wgpu::DeviceDescriptor<'static> {
        wgpu::DeviceDescriptor {
            label: None,
            required_features: if std::env::var("ZOSIMOS_PASSTHROUGH").is_err() {
                wgpu::Features::empty()
            } else {
                wgpu::Features::SPIRV_SHADER_PASSTHROUGH
            },
            // Well, technically... We need the texture format.
            // But should be able to workaround most other restrictions.
            // FIXME: make the use of this configurable.
            required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
            memory_hints: wgpu::MemoryHints::Performance,
        }
    }

    /// Run this program with a pool.
    ///
    /// Required input and output image descriptors must match those declared, or be convertible
    /// to them when a normalization operation was declared.
    pub fn launch<'pool>(&'pool self, pool: &'pool mut Pool) -> Launcher<'pool> {
        let main = &self.functions[self.entry_index];

        // Create empty bind assignments as a start, with respective layouts.
        let binds = main
            .image_buffers
            .texture
            .iter()
            .map(run::Image::with_late_bound)
            .collect();

        Launcher {
            program: self,
            pool,
            main,
            binds,
            pool_plan: ImagePoolPlan::default(),
        }
    }

    pub fn lower_to(&self, capabilities: Capabilities) -> Result<run::Executable, LaunchError> {
        let main = &self.functions[self.entry_index];

        let mut instructions = vec![];
        let mut functions = HashMap::new();
        let mut binary_data = vec![];
        let mut skip_by_op = HashMap::new();
        let mut knobs = HashMap::new();

        let mut encoder = self.lower_to_impl(&capabilities, main, None)?;
        encoder.finalize()?;
        let io_map: Arc<run::IoMap> = encoder.io_map().into();

        instructions.extend(encoder.instructions);
        binary_data.extend(encoder.binary_data);
        skip_by_op.extend(encoder.info.skip_by_op);

        knobs.extend(
            encoder
                .info
                .knobs
                .into_iter()
                .map(|(knob, info)| (knob, info)),
        );

        functions.insert(
            Function(self.entry_index),
            FunctionFrame {
                range: 0..instructions.len(),
                io: Arc::from(main.image_buffers.texture.to_vec()),
                io_map: io_map.clone(),
            },
        );

        for (idx, ops) in self.functions.iter().enumerate() {
            if idx == self.entry_index {
                continue;
            }

            let mut encoder = self.lower_to_impl(&capabilities, ops, None)?;
            encoder.finalize()?;
            let io_map = encoder.io_map().into();

            let reloc_base = binary_data.len();
            binary_data.extend(encoder.binary_data);

            for op in &mut encoder.instructions {
                Self::relocate_binary(reloc_base, op)?;
            }

            let start = instructions.len();
            instructions.extend(encoder.instructions);
            let end = instructions.len();

            skip_by_op.extend(
                encoder
                    .info
                    .skip_by_op
                    .into_iter()
                    .map(|(inst, event)| (Instruction(inst.0 + start), event)),
            );

            knobs.extend(
                encoder
                    .info
                    .knobs
                    .into_iter()
                    .map(|(knob, info)| (knob, info.relocate(reloc_base))),
            );

            functions.insert(
                Function(idx),
                FunctionFrame {
                    range: start..end,
                    io: Arc::from(ops.image_buffers.texture.to_vec()),
                    io_map,
                },
            );
        }

        // Convert all textures to buffers.
        // FIXME: _All_ textures? No, some amount of textures might not be IO.
        // Currently this is true but no in general.
        let image_io_buffers = main
            .image_buffers
            .texture
            .iter()
            .map(run::Image::with_late_bound)
            .collect();

        let knob_starts: BTreeMap<_, _> = {
            knobs
                .iter()
                .map(|(&key, info)| (info.range.start, key))
                .collect()
        };

        assert_eq!(knob_starts.len(), knobs.len());

        Ok(run::Executable {
            entry_point: Function(self.entry_index),
            instructions: instructions.into(),
            info: Arc::new(run::ProgramInfo {
                buffer_by_op: encoder.info.buffer_by_op,
                texture_by_op: encoder.info.texture_by_op,
                shader_by_op: encoder.info.shader_by_op,
                pipeline_by_op: encoder.info.pipeline_by_op,
                skip_by_op,
                functions,
                knob_descriptors: knobs,
                knobs: self.knobs.clone(),
                knob_starts,
            }),
            binary_data,
            descriptors: run::Descriptors::default(),
            image_io_buffers,
            capabilities,
            io_map: io_map.into(),
        })
    }

    fn lower_to_impl(
        &self,
        capabilities: &Capabilities,
        function: &FunctionLinked,
        pool_plan: Option<&ImagePoolPlan>,
    ) -> Result<Encoder, LaunchError> {
        let mut encoder = Encoder::default();
        encoder.enable_capabilities(capabilities);

        encoder.set_buffer_plan(&function.image_buffers);
        if let Some(pool_plan) = pool_plan {
            encoder.set_pool_plan(pool_plan);
        }

        for high in &self.ops[function.ops.clone()] {
            let with_stack_frame = match high {
                High::StackPush(_) | High::StackPop => false,
                other => {
                    encoder.push(Low::StackFrame(run::Frame {
                        name: format!("Operation: {:#?}", other),
                    }))?;
                    true
                }
            };

            match high {
                &High::Done(_) => {
                    // TODO: should deallocate/put up for reuse textures that aren't live anymore.
                    // But we must have a validation strategy for the liveness map, in particular
                    // for any code changes and additions to the `High` intermediate instruction
                    // format to avoid regressions.
                }
                &High::Input(dst) => {
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
                &High::Render { src, dst } => {
                    encoder.render_staging_to_output(src, dst)?;
                }
                &High::PushOperand(texture) => {
                    encoder.copy_staging_to_texture(texture)?;
                    encoder.push_operand(texture)?;
                }
                &High::Uninit { dst } => {
                    encoder.ensure_device_texture(match dst {
                        Target::Discard(texture) | Target::Load(texture) => texture,
                    })?;

                    // Nothing more to do.
                }
                High::DrawInto { dst, fn_ } => {
                    let dst_texture = match dst {
                        Target::Discard(texture) | Target::Load(texture) => *texture,
                    };

                    encoder.ensure_device_texture(dst_texture)?;
                    let dst_view = encoder.texture_view(dst_texture)?;

                    let ops = match dst {
                        Target::Discard(_) => {
                            wgpu::Operations {
                                // TODO: we could let choose a replacement color..
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLUE),
                                store: wgpu::StoreOp::Store,
                            }
                        }
                        Target::Load(_) => wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
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
                    let size;
                    let source_buffer = match encoder.allocate_register(*src)? {
                        &RegisterMap::Image {
                            buffer,
                            ref buffer_layout,
                            ..
                        } => {
                            size = buffer_layout.u64_len();
                            encoder.copy_staging_to_buffer(*src)?;
                            buffer
                        }
                        &RegisterMap::Buffer {
                            buffer,
                            ref buffer_layout,
                            ..
                        } => {
                            size = buffer_layout.u64_len();
                            buffer
                        }
                    };

                    let target_buffer = match encoder.allocate_register(*dst)? {
                        RegisterMap::Image { buffer, .. } | RegisterMap::Buffer { buffer, .. } => {
                            *buffer
                        }
                    };

                    encoder.push(Low::BeginCommands)?;
                    encoder.push(Low::CopyBufferToBuffer {
                        source_buffer,
                        source: 0,
                        target_buffer,
                        target: 0,
                        size,
                    })?;
                    encoder.push(Low::EndCommands)?;
                    encoder.push(Low::RunTopCommand)?;

                    if let RegisterMap::Image { .. } = encoder.allocate_register(*dst)? {
                        encoder.copy_buffer_to_staging(*dst)?;
                    }
                }
                High::WriteInto { dst, fn_ } => {
                    encoder.prepare_buffer_write(fn_, *dst)?;
                }
                High::StackPush(frame) => {
                    encoder.push(Low::StackFrame(run::Frame {
                        name: frame.name.clone(),
                    }))?;
                }
                High::StackPop => {
                    encoder.push(Low::StackPop)?;
                }
                High::Call {
                    function: fn_idx,
                    image_io_buffers,
                } => {
                    // We pass images as their encoded buffers. This is most generic.
                    let mut io_buffers = vec![];
                    let mut post_textures = vec![];

                    let signature = &self.functions[fn_idx.0].signature_registers;

                    for (&in_io, param) in signature.iter().zip(&image_io_buffers[..]) {
                        match param {
                            &CallBinding::InTexture { texture, register } => {
                                let regmap = encoder.allocate_register(register)?.clone();
                                encoder.ensure_device_texture(texture)?;
                                encoder.copy_staging_to_buffer(register)?;
                                let descriptor = &function.image_buffers.texture[texture.0];
                                let (RegisterMap::Image { buffer, .. }
                                | RegisterMap::Buffer { buffer, .. }) = regmap;

                                io_buffers.push(CallImageArgument {
                                    buffer,
                                    descriptor: descriptor.clone(),
                                    in_io,
                                });
                            }
                            &CallBinding::OutTexture { texture, register } => {
                                let regmap = encoder.allocate_register(register)?.clone();
                                encoder.ensure_device_texture(texture)?;
                                let descriptor = &function.image_buffers.texture[texture.0];
                                post_textures.push(register);
                                let (RegisterMap::Image { buffer, .. }
                                | RegisterMap::Buffer { buffer, .. }) = regmap;

                                io_buffers.push(CallImageArgument {
                                    buffer,
                                    descriptor: descriptor.clone(),
                                    in_io,
                                });
                            }
                        }
                    }

                    encoder.push(Low::Call {
                        function: *fn_idx,
                        io_buffers,
                    })?;

                    // Retrieve arguments that were rendered to.
                    for register in post_textures {
                        encoder.copy_buffer_to_staging(register)?;
                    }
                }
            }

            if with_stack_frame {
                encoder.push(Low::StackPop)?;
            }
        }

        encoder.push(Low::Return)?;

        Ok(encoder)
    }

    fn relocate_binary(base: usize, op: &mut Low) -> Result<(), LaunchError> {
        match op {
            Low::BufferInit(range) => {
                match &mut range.content {
                    range => {
                        range.start = range
                            .start
                            .checked_add(base)
                            .ok_or_else(|| LaunchError::InternalCommandError(line!()))?;
                        range.end = range
                            .end
                            .checked_add(base)
                            .ok_or_else(|| LaunchError::InternalCommandError(line!()))?;
                    }
                }

                Ok(())
            }

            Low::BindGroup(_)
            | Low::BindGroupLayout(_)
            | Low::Buffer(_)
            | Low::PipelineLayout(_)
            | Low::Sampler(_)
            | Low::Shader(_)
            | Low::Texture(_)
            | Low::TextureView(_)
            | Low::RenderView(_)
            | Low::RenderPipeline(_)
            | Low::BeginCommands
            | Low::BeginRenderPass(_)
            | Low::EndCommands
            | Low::EndRenderPass
            | Low::SetPipeline(_)
            | Low::SetBindGroup { .. }
            | Low::SetVertexBuffer { .. }
            | Low::DrawOnce { .. }
            | Low::DrawIndexedZero { .. }
            | Low::SetPushConstants { .. }
            | Low::RunTopCommand
            | Low::RunBotToTop(_)
            | Low::WriteImageToBuffer { .. }
            | Low::WriteImageToTexture { .. }
            | Low::CopyBufferToTexture { .. }
            | Low::CopyTextureToBuffer { .. }
            | Low::CopyBufferToBuffer { .. }
            | Low::ZeroBuffer { .. }
            | Low::ReadBuffer { .. }
            | Low::StackFrame(_)
            | Low::StackPop
            | Low::AssertBuffer { .. }
            | Low::AssertTexture { .. }
            | Low::Call { .. }
            | Low::Return => Ok(()),
        }
    }
}

impl Launcher<'_> {
    /// Bind an image in the pool to an input register.
    ///
    /// Returns an error if the register does not specify an input, or when there is no image under
    /// the key in the pool, or when the image in the pool does not match the declared format.
    pub fn bind(mut self, reg: Register, img: PoolKey) -> Result<Self, LaunchError> {
        if self.pool.entry(img).is_none() {
            return Err(LaunchError::InternalCommandError(line!()));
        }

        let RegisterAssignment::Image(ImageBufferAssignment {
            texture: Texture(texture),
            ..
        }) = self.main.image_buffers.get_register_resources(reg)?
        else {
            return Err(LaunchError::InternalCommandError(line!()));
        };

        self.pool_plan.plan.insert(reg, img);
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
            if let High::Output { src: register, dst } = *high {
                match self.main.image_buffers.get_register_resources(register)? {
                    RegisterAssignment::Image(assigned) => {
                        let descriptor = &self.main.image_buffers.texture[assigned.texture.0];
                        let key = self.pool_plan.choose_output(&mut *self.pool, descriptor);
                        self.pool_plan.plan.insert(dst, key);
                    }
                    RegisterAssignment::Buffer(_) => todo!("No such output yet"),
                }
            }
        }

        Ok(self)
    }

    /// Really launch, potentially failing if configuration or inputs were missing etc.
    pub fn launch(mut self, adapter: &wgpu::Adapter) -> Result<run::Execution, LaunchError> {
        let request = adapter.request_device(&self.program.device_descriptor(), None);

        // For all inputs check that they have now been supplied.
        for high in &self.program.ops {
            if let High::Input(register) = *high {
                if self.pool_plan.get_texture(register).is_none() {
                    return Err(LaunchError::InternalCommandError(line!()));
                }
            }
        }

        // Bind remaining outputs.
        self = self.bind_remaining_outputs()?;

        let request = Box::pin(request);
        let (device, queue) = match run::block_on(request, None) {
            Ok(tuple) => tuple,
            Err(_) => return Err(LaunchError::InternalCommandError(line!())),
        };

        let capabilities = Capabilities::from(&device);

        let mut encoder =
            self.program
                .lower_to_impl(&capabilities, self.main, Some(&self.pool_plan))?;

        let mut image_io_buffers = self.binds;
        encoder.extract_buffers(&mut image_io_buffers, &mut self.pool)?;
        let io_descriptors: Vec<_> = image_io_buffers
            .iter()
            .map(|img| img.descriptor.clone())
            .collect();

        // Unbalanced operands shouldn't happen.
        // This is part of validation layer but cheap and we always do it.
        encoder.finalize()?;
        let io_map: Arc<run::IoMap> = encoder.io_map().into();
        let instructions: Arc<[_]> = encoder.instructions.into();
        let all_range = 0..instructions.len();

        let knob_starts: BTreeMap<_, _> = {
            encoder
                .info
                .knobs
                .iter()
                .map(|(&key, info)| (info.range.start, key))
                .collect()
        };

        let init = run::InitialState {
            // TODO: shared with lower_to. Find a better way to reap the `encoder` for its
            // resources and descriptors.
            instructions,
            entry_point: Function(0),
            info: Arc::new(run::ProgramInfo {
                buffer_by_op: encoder.info.buffer_by_op,
                texture_by_op: encoder.info.texture_by_op,
                shader_by_op: encoder.info.shader_by_op,
                pipeline_by_op: encoder.info.pipeline_by_op,
                skip_by_op: encoder.info.skip_by_op,
                functions: vec![(
                    Function(0),
                    FunctionFrame {
                        range: all_range,
                        io: Arc::from(io_descriptors),
                        io_map: io_map.clone(),
                    },
                )]
                .into_iter()
                .collect(),
                knob_descriptors: encoder.info.knobs,
                knobs: self.program.knobs.clone(),
                knob_starts,
            }),
            device,
            queue,
            image_io_buffers,
            binary_data: encoder.binary_data,
            io_map,
        };

        Ok(run::Execution::new(init))
    }
}

impl<'trgt> BufferInitContentBuilder<'trgt> {
    pub fn extend_from_pods(&mut self, data: &[impl bytemuck::Pod]) {
        self.buf.extend_from_slice(bytemuck::cast_slice(data));
    }

    /// Align to the given power-of-two.
    pub fn align_by_exponent(&mut self, by: u8) {
        let align = 1usize << by;
        let len = self.buf.len();
        if len % align != 0 {
            let add = align - len % align;
            self.buf.resize(len + add, 0);
        }
    }

    pub fn build(self) -> BufferInitContent {
        BufferInitContent::Defer {
            start: self.start,
            end: self.buf.len(),
        }
    }
}

impl BufferInitContent {
    /// Construct a reference to data by allocating it freshly within the buffer.
    pub fn new(buf: &mut Vec<u8>, data: &[impl bytemuck::Pod]) -> Self {
        let start = buf.len();
        buf.extend_from_slice(bytemuck::cast_slice(data));
        let end = buf.len();
        BufferInitContent::Defer { start, end }
    }

    /// Start allocating data into a buffer
    pub fn builder(buf: &mut Vec<u8>) -> BufferInitContentBuilder<'_> {
        let start = buf.len();
        BufferInitContentBuilder { buf, start }
    }

    /// Get a reference to the binary data, given the allocator/buffer.
    pub fn as_slice<'lt>(&'lt self, buffer: &'lt Vec<u8>) -> &'lt [u8] {
        match self {
            BufferInitContent::Owned(ref data) => data,
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

impl From<&'_ wgpu::Device> for Capabilities {
    fn from(device: &'_ wgpu::Device) -> Self {
        Capabilities {
            features: device.features(),
            limits: device.limits(),
        }
    }
}

impl BufferUsage {
    pub fn to_wgpu(self) -> wgpu::BufferUsages {
        use wgpu::BufferUsages as U;
        match self {
            BufferUsage::InVertices => U::COPY_DST | U::VERTEX,
            BufferUsage::DataIn => U::MAP_WRITE | U::COPY_SRC,
            BufferUsage::DataOut => U::MAP_READ | U::COPY_DST,
            BufferUsage::DataBuffer => U::STORAGE | U::COPY_SRC | U::COPY_DST,
            BufferUsage::Uniform => U::COPY_DST | U::UNIFORM,
        }
    }
}

impl BufferDescriptor {
    pub fn u64_len(&self) -> u64 {
        self.size
    }
}

impl BufferDescriptorInit {
    pub fn u64_len(&self) -> u64 {
        self.content.end.wrapping_sub(self.content.start) as u64
    }
}

impl TextureDescriptor {
    pub fn u64_len(&self) -> u64 {
        let (w, h) = self.size;
        // FIXME: not really accurate.
        4 * u64::from(w.get()) * u64::from(h.get())
    }
}

impl KnobDescriptor {
    pub fn relocate(self, by: usize) -> Self {
        let end = self.range.end.checked_add(by).unwrap();
        let start = self.range.start.checked_add(by).unwrap();

        KnobDescriptor {
            range: start..end,
            ..self
        }
    }
}

impl LaunchError {
    #[allow(non_snake_case)]
    // FIXME: find a better error representation but it's okay for now.
    // #[deprecated = "This should be cleaned up"]
    pub(crate) fn InternalCommandError(line: u32) -> Self {
        LaunchError {
            kind: LaunchErrorKind::FromLine(line),
        }
    }
}

impl core::fmt::Debug for ShaderDescriptor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShaderDescriptor")
            .field("name", &self.name)
            .field("key", &self.key)
            .field("source_spirv", &"opaque")
            .finish()
    }
}
