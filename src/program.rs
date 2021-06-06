use core::ops::Range;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use crate::command::{High, Rectangle, Register, Target};
use crate::buffer::{BufferLayout, Descriptor};
use crate::pool::{ImageData, Pool, PoolKey};
use crate::{run, shaders};
use crate::util::ExtendOne;

/// Planned out and intrinsically validated command buffer.
///
/// This does not necessarily plan out a commands of low leve execution instruction set flavor.
/// This is selected based on the available device and its capabilities, which is performed during
/// launch.
pub struct Program {
    pub(crate) ops: Vec<High>,
    pub(crate) textures: Textures,
    pub(crate) by_register: HashMap<Register, Texture>,
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
        paint_on_top: PaintOnTopKind,
    },
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) enum PaintOnTopKind {
    Copy,
}

#[derive(Default)]
pub struct Textures {
    vec: Vec<Descriptor>,
    by_layout: HashMap<BufferLayout, usize>,
}

/// Identifies one resources in the render pipeline, by an index.
#[derive(Clone, Copy)]
pub(crate) struct Texture(usize);

/// The encoder tracks the supposed state of `run::Descriptors` without actually executing them.
#[derive(Default)]
struct Encoder<Instructions: ExtendOne<Low> = Vec<Low>> {
    instructions: Instructions,
    
    // Replicated fields from `run::Descriptors`
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
    paint_group_layout: Option<usize>,
    paint_pipeline_layout: Option<usize>,
}

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
    binds: Vec<ImageData>,
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
        let binds = self.textures.vec
            .iter()
            .map(|desciptor| ImageData::LateBound(desciptor.layout.clone()))
            .collect();
        Launcher {
            program: self,
            pool,
            binds,
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
            None => return Err(LaunchError { }),
        };

        let (_, _) = match self.program.ops.get(reg) {
            Some(High::Input(target, descriptor)) => (target, descriptor),
            _ => return Err(LaunchError { })
        };

        let Texture(texture) = match self.program.by_register.get(&Register(reg)) {
            Some(texture) => *texture,
            None => return Err(LaunchError { }),
        };

        entry.swap(&mut self.binds[texture]);

        Ok(self)
    }

    /// Really launch, potentially failing if configuration or inputs were missing etc.
    pub fn launch(self, adapter: &wgpu::Adapter) -> Result<run::Execution, LaunchError> {
        let request = adapter.request_device(&self.program.device_descriptor(), None);

        for high in &self.program.ops {
            match high {
                &High::Input(Texture(texture), _) => {
                    let entry = &self.binds[texture];
                    if matches!(entry, ImageData::LateBound(_)) {
                        return Err(LaunchError { })
                    }
                }
                _ => {},
            }
        }

        let (device, queue) = match block_on(request) {
            Ok(tuple) => tuple,
            Err(_) => return Err(LaunchError {}),
        };

        let mut encoder = Encoder::default();
        encoder.enable_capabilities(&device);

        for high in &self.program.ops {
            match high {
                High::Allocate(texture) => {
                    todo!()
                },
                High::Discard(texture) => {
                    todo!()
                },
                High::Input(dst, what) => {
                    let idx = encoder.input_id();
                    // Identify how we ingest this image.
                    // If it is a texture format that we support then we will allocate and upload
                    // it directly. If it is not then we will allocate a generic version capable of
                    // holding a lossless convert variant of it and add instructions to convert
                    // into that buffer.
                    todo!()
                },
                High::Output(texture) => {
                    let idx = encoder.output_id();
                    // Identify if we need to transform the texture from the internal format to the
                    // one actually chosen for this texture.
                },
                High::Construct { dst, op } => {
                    todo!()
                },
                High::Paint { texture, dst, fn_ } => {
                    let layout = encoder.make_paint_layout();

                    let dst_view = encoder.texture_view(TextureViewDescriptor {
                        texture: match dst {
                            Target::Discard(texture) | Target::Load(texture) => texture.0,
                        }
                    });

                    let ops = match dst {
                        Target::Discard(_) => {
                            wgpu::Operations {
                                // TODO: we could let choose a replacement color..
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
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

                    encoder.push(Low::BeginCommands)?;
                    encoder.push(Low::BeginRenderPass(RenderPassDescriptor {
                        color_attachments: vec![attachment],
                        depth_stencil: None,
                    }))?;
                    encoder.render(fn_);
                    encoder.push(Low::EndRenderPass)?;
                    encoder.push(Low::EndCommands)?;

                    // Actually run it immediately.
                    // TODO: this might not be the most efficient.
                    encoder.push(Low::RunTopCommand)?;
                },
            }
        }

        let init = run::InitialState {
            instructions: encoder.instructions,
            device,
            queue,
            buffers: core::mem::take(self.pool),
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

    /// Validate and then add the command to the encoder.
    ///
    /// This ensures we can keep track of the expected state change, and validate the correct order
    /// of commands. More specific sequencing commands will expect correct order or assume it
    /// internally.
    fn push(&mut self, low: Low) -> Result<(), LaunchError> {
        match low {
            Low::BindGroupLayout(_) => self.bind_group_layouts += 1,
            Low::BindGroup(_) => self.bind_groups += 1,
            Low::Buffer(_) => self.buffers += 1,
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
            Low::SetPipeline(_) => todo!(),
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
            Low::WriteImageToBuffer { .. }
            | Low::WriteImageToTexture { .. }
            | Low::ReadBuffer { .. } => {},
        }

        self.instructions.extend_one(low);
        Ok(())
    }

    fn input_id(&mut self) -> usize {
        todo!()
    }

    fn output_id(&mut self) -> usize {
        todo!()
    }

    fn texture_view(&mut self, descriptor: TextureViewDescriptor) -> usize {
        self.instructions.extend_one(Low::TextureView(descriptor));
        let id = self.texture_views;
        self.texture_views += 1;
        id
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
                        ty: wgpu::BindingType::Sampler {
                            filtering: true,
                            comparison: true,
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
        let bind_group = self.make_paint_group();
        let layouts = &mut self.pipeline_layouts;
        let instructions = &mut self.instructions;
        *self.paint_pipeline_layout.get_or_insert_with(|| {
            let descriptor = PipelineLayoutDescriptor {
                bind_group_layouts: vec![bind_group],
                push_constant_ranges: &[
                    wgpu::PushConstantRange {
                        stages: wgpu::ShaderStage::FRAGMENT,
                        range: 0..16,
                    },
                ],
            };

            instructions.extend_one(Low::PipelineLayout(descriptor));
            let descriptor_id = *layouts;
            *layouts += 1;
            descriptor_id
        })
    }

    fn render(&mut self, function: &Function) {
        match function {
            Function::PaintOnTop { lower_region, upper_region, paint_on_top } => {
                let vertex = shaders::VERT_NOOP;
                let fragment = paint_on_top.fragment_shader();

                todo!("Actual shader painting")
            },
        }
    }
}

impl PaintOnTopKind {
    fn fragment_shader(&self) -> &[u8] {
        match self {
            PaintOnTopKind::Copy => shaders::FRAG_COPY,
        }
    }
}

impl LaunchError {
    #[deprecated = "Should be removed and implemented"]
    pub(crate) const UNIMPLEMENTED_CHECK: Self = LaunchError {};
    #[allow(non_snake_case)]
    #[deprecated = "This should be cleaned up"]
    pub(crate) fn InternalCommandError(line: u32) -> Self {
        // FIXME: this should not be here..
        eprintln!("In line {}", line);
        LaunchError {}
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
