use core::{future::Future, num::NonZeroU32, ops::Range};

use std::borrow::Cow;
use std::collections::HashMap;

use crate::buffer::{BufferLayout, Descriptor, RowMatrix};
use crate::command::{High, Rectangle, Register, Target};
use crate::encoder::{Encoder, RegisterMap};
use crate::pool::{Pool, PoolKey};
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
    PaintFullScreen { shader: shaders::FragmentShader },
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
    pub(crate) descriptor: &'a Descriptor,
    pub(crate) layout: &'a BufferLayout,
}

/// A gpu buffer associated with an image buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceBuffer(pub(crate) usize);

/// A gpu texture associated with an image buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceTexture(pub(crate) usize);

/// Identifies one layout based buffer in the render pipeline, by an index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Buffer(pub(crate) usize);

/// Identifies one descriptor based resource in the render pipeline, by an index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Texture(pub(crate) usize);

/// A map of features which we may use during encoding.
#[derive(Clone, Debug)]
pub struct Capabilities {
    features: wgpu::Features,
    limits: wgpu::Limits,
}

#[derive(Debug)]
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
        stages: wgpu::ShaderStages,
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
    binds: Vec<run::Image>,
    /// Assigns images from the internal pool to registers.
    /// They may be transferred from an input pool, and conversely we assign outputs. We can use
    /// the plan to put back all images into the pool when retiring the execution.
    pool_plan: ImagePoolPlan,
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
        from: impl Iterator<Item = wgpu::Adapter>,
    ) -> Result<wgpu::Adapter, MismatchError> {
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

        while let Some(adapter) = from.next() {
            eprintln!("{:?}", adapter);
            // FIXME: check limits.
            // FIXME: collect required texture formats from `self.textures`
            let basic_format =
                adapter.get_texture_format_features(wgpu::TextureFormat::Rgba8UnormSrgb);
            if !basic_format.allowed_usages.contains(ALL_TEXTURE_USAGE) {
                eprintln!("No rgba8 support {:?}", basic_format.allowed_usages);
                continue;
            }

            let storage_format =
                adapter.get_texture_format_features(wgpu::TextureFormat::R32Uint);
            if !storage_format.allowed_usages.contains(STAGE_TEXTURE_USAGE) {
                eprintln!("No r32uint storage support {:?}", basic_format.allowed_usages);
                continue;
            }

            from.for_each(|ad| eprintln!("{:?}", ad));
            return Ok(adapter);
        }

        Err(MismatchError {})
    }

    /// Return a descriptor for a device that's capable of executing the program.
    pub fn device_descriptor(&self) -> wgpu::DeviceDescriptor<'static> {
        Self::minimal_device_descriptor()
    }

    pub fn minimal_device_descriptor() -> wgpu::DeviceDescriptor<'static> {
        wgpu::DeviceDescriptor {
            label: None,
            features: wgpu::Features::SPIRV_SHADER_PASSTHROUGH
                | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
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
            .map(run::Image::with_late_bound)
            .collect();

        Launcher {
            program: self,
            pool,
            binds,
            pool_plan: ImagePoolPlan::default(),
        }
    }

    pub fn lower_to(&self, capabilities: Capabilities) -> Result<run::Executable, LaunchError> {
        let mut encoder = self.lower_to_impl(&capabilities, None)?;
        encoder.finalize()?;
        let io_map = encoder.io_map();

        // Convert all textures to buffers.
        // FIXME: _All_ textures? No, some amount of textures might not be IO.
        // Currently this is true but no in general.
        let buffers = self
            .textures
            .texture
            .iter()
            .map(run::Image::with_late_bound)
            .collect();

        Ok(run::Executable {
            instructions: encoder.instructions.into(),
            binary_data: encoder.binary_data,
            descriptors: run::Descriptors::default(),
            buffers,
            capabilities,
            io_map,
        })
    }

    fn lower_to_impl(
        &self,
        capabilities: &Capabilities,
        pool_plan: Option<&ImagePoolPlan>,
    ) -> Result<Encoder, LaunchError> {
        let mut encoder = Encoder::default();
        encoder.enable_capabilities(&capabilities);

        encoder.set_buffer_plan(&self.textures);
        if let Some(pool_plan) = pool_plan {
            encoder.set_pool_plan(pool_plan);
        }

        for high in &self.ops {
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
                    let &RegisterMap {
                        buffer: source_buffer,
                        ref buffer_layout,
                        ..
                    } = encoder.allocate_register(*src)?;
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

        Ok(encoder)
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

        let capabilities = Capabilities::from(&device);

        let mut encoder = self
            .program
            .lower_to_impl(&capabilities, Some(&self.pool_plan))?;
        let mut buffers = self.binds;
        encoder.extract_buffers(&mut buffers, &mut self.pool)?;

        // Unbalanced operands shouldn't happen.
        // This is part of validation layer but cheap and we always do it.
        encoder.finalize()?;

        let init = run::InitialState {
            instructions: encoder.instructions.into(),
            device,
            queue,
            buffers,
            binary_data: encoder.binary_data,
        };

        Ok(run::Execution::new(init))
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
