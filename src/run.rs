use core::{iter::once, num::NonZeroU32};

use crate::buffer::BufferLayout;
use crate::command::{Rectangle, Register};
use crate::pool::{ImageData, Pool, PoolImage, PoolKey};
use crate::program::{self, DeviceBuffer, DeviceTexture, Low};

use wgpu::{Device, Queue};

pub struct Execution {
    pub(crate) machine: Machine,
    pub(crate) gpu: Gpu,
    pub(crate) descriptors: Descriptors,
    pub(crate) command_encoder: Option<wgpu::CommandEncoder>,
    pub(crate) buffers: Vec<ImageData>,
}

pub(crate) struct InitialState {
    pub(crate) instructions: Vec<Low>,
    pub(crate) device: Device,
    pub(crate) queue: Queue,
    pub(crate) buffers: Vec<ImageData>,
}

#[derive(Default)]
pub(crate) struct Descriptors {
    bind_groups: Vec<wgpu::BindGroup>,
    bind_group_layouts: Vec<wgpu::BindGroupLayout>,
    buffers: Vec<wgpu::Buffer>,
    command_buffers: Vec<wgpu::CommandBuffer>,
    modules: Vec<wgpu::ShaderModule>,
    pipeline_layouts: Vec<wgpu::PipelineLayout>,
    render_pipelines: Vec<wgpu::RenderPipeline>,
    sampler: Vec<wgpu::Sampler>,
    textures: Vec<wgpu::Texture>,
    texture_views: Vec<wgpu::TextureView>,
}

pub(crate) struct Gpu {
    pub(crate) device: Device,
    pub(crate) queue: Queue,
    pub(crate) modules: Vec<wgpu::ShaderModule>,
}

/// One fragment shader execution with pipeline:
/// FS:
///   in: vec2 uv
///   region: vec4 (parameter)
///   bind: sampler2D
///   out: vec4 (color)
pub(crate) struct PaintRectFragment {
    /// The 'selected' region relative to which uv is to be interpreted.
    region: Rectangle,
    /// The index of the sampler which we should bind to `bind`.
    region_sampler_id: usize,
    /// The target region we want to paint.
    target: Rectangle,
    /// The shader to compile for this.
    fragment_shader: &'static [u8],
}

type DynStep = dyn core::future::Future<Output=Result<Cleanup, StepError>>;

enum Cleanup {
    Buffers {
        buffers: Vec<wgpu::Buffer>,
        image_data: Vec<ImageData>,
    }
}

pub struct SyncPoint<'a> {
    future: Option<(core::pin::Pin<Box<DynStep>>, &'a mut Execution)>,
    marker: core::marker::PhantomData<&'a mut Execution>,
}

/// Represents a stopped execution instance, without information abouts its outputs.
pub struct Retire<'pool> {
    /// The retiring execution instance.
    execution: Execution,
    pool: &'pool mut Pool,
}

pub(crate) struct Machine {
    instructions: Vec<Low>,
    instruction_pointer: usize,
}

#[derive(Debug)]
pub enum StepError {
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
pub enum RetireError {
}

impl Execution {
    pub(crate) fn new(init: InitialState) -> Self {
        Execution {
            machine: Machine::new(dbg!(init.instructions)),
            gpu: Gpu {
                device: init.device,
                queue: init.queue,
                modules: vec![],
            },
            descriptors: Descriptors::default(),
            buffers: dbg!(init.buffers),
            command_encoder: None,
        }
    }

    /// Check if the machine is still running.
    pub fn is_running(&self) -> bool {
        self.machine.instruction_pointer < self.machine.instructions.len()
    }

    pub fn step(&mut self) -> Result<SyncPoint<'_>, StepError> {
        match dbg!(self.machine.next_instruction()?) {
            Low::BindGroupLayout(desc) => {
                let mut entry_buffer = vec![];
                let group = self.descriptors.bind_group_layout(desc, &mut entry_buffer)?;
                let group = self.gpu.device.create_bind_group_layout(&group);
                self.descriptors.bind_group_layouts.push(group);
                Ok(SyncPoint::NO_SYNC)
            }
            Low::BindGroup(desc) => {
                let mut entry_buffer = vec![];
                let group = self.descriptors.bind_group(desc, &mut entry_buffer)?;
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
                    contents: &desc.content,
                    usage: desc.usage.to_wgpu(),
                };

                let buffer = self.gpu.device.create_buffer_init(&desc);
                self.descriptors.buffers.push(buffer);
                Ok(SyncPoint::NO_SYNC)
            }
            Low::Shader(desc) => {
                let desc = wgpu::ShaderModuleDescriptor {
                    label: Some(desc.name),
                    source: wgpu::ShaderSource::SpirV(desc.source_spirv.as_ref().into()),
                    flags: desc.flags,
                };

                let module = self.gpu.device.create_shader_module(&desc);
                self.descriptors.modules.push(module);
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
                dbg!(desc);
                let texture = self.descriptors.textures
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
                use wgpu::TextureUsage as U;
                let desc = wgpu::TextureDescriptor {
                    label: None,
                    size: wgpu::Extent3d {
                        width: desc.size.0.get(),
                        height: desc.size.1.get(),
                        depth_or_array_layers: 1
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: desc.format,
                    usage: match desc.usage {
                        program::TextureUsage::DataIn => U::COPY_DST | U::SAMPLED,
                        program::TextureUsage::DataOut => U::COPY_SRC | U::RENDER_ATTACHMENT,
                        program::TextureUsage::Storage => {
                            U::COPY_SRC | U::COPY_DST | U::SAMPLED | U::RENDER_ATTACHMENT
                        },
                    }
                };
                let texture = self.gpu.device.create_texture(&desc);
                self.descriptors.textures.push(texture);
                Ok(SyncPoint::NO_SYNC)
            }
            Low::RenderPipeline(desc) => {
                let mut vertex_buffers = vec![];
                let mut fragments = vec![];

                let pipeline = self.descriptors.pipeline(desc, &mut vertex_buffers, &mut fragments)?;
                let pipeline = self.gpu.device.create_render_pipeline(&pipeline);
                self.descriptors.render_pipelines.push(pipeline);
                Ok(SyncPoint::NO_SYNC)
            }
            Low::BeginCommands => {
                if self.command_encoder.is_some() {
                    return Err(StepError::InvalidInstruction(line!()));
                }

                let descriptor = wgpu::CommandEncoderDescriptor {
                    label: None,
                };

                self.command_encoder = Some(self.gpu.device.create_command_encoder(&descriptor));
                Ok(SyncPoint::NO_SYNC)
            },
            Low::BeginRenderPass(descriptor) => {
                let mut attachment_buf = vec![];
                let descriptor = self.descriptors.render_pass(descriptor, &mut attachment_buf)?;
                let encoder = match &mut self.command_encoder {
                    Some(encoder) => encoder,
                    None => return Err(StepError::InvalidInstruction(line!())),
                };

                let pass = encoder.begin_render_pass(&descriptor);
                drop(attachment_buf);
                self.machine.render_pass(&self.descriptors, pass)?;

                Ok(SyncPoint::NO_SYNC)
            },
            Low::EndCommands => {
                match self.command_encoder.take() {
                    None => Err(StepError::InvalidInstruction(line!())),
                    Some(encoder) => {
                        self.descriptors.command_buffers.push(encoder.finish());
                        Ok(SyncPoint::NO_SYNC)
                    }
                }
            }
            &Low::RunTopCommand => {
                let command = self.descriptors.command_buffers
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
            &Low::WriteImageToBuffer { source_image, offset, size: _, target_buffer, ref target_layout } => {
                if offset != (0, 0) {
                    return Err(StepError::InvalidInstruction(line!()));
                }

                let buffer = self.descriptors.buffer(target_buffer, target_layout)?;
                let data = self.buffers.get(source_image.0)
                    .ok_or(StepError::InvalidInstruction(line!()))?
                    .as_bytes()
                    .ok_or(StepError::InvalidInstruction(line!()))?;
                self.gpu.queue
                    .write_buffer(buffer.buffer, 0, data);

                Ok(SyncPoint::NO_SYNC)
            }
            &Low::CopyBufferToTexture { source_buffer, ref source_layout, offset, size, target_texture } => {
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
                    depth_or_array_layers: 0,
                };

                encoder.copy_buffer_to_texture(
                    buffer,
                    texture,
                    extent,
                );

                Ok(SyncPoint::NO_SYNC)
            }
            &Low::CopyTextureToBuffer { source_texture, offset, size, target_buffer, ref target_layout } => {
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
                    depth_or_array_layers: 0,
                };

                encoder.copy_texture_to_buffer(
                    texture,
                    buffer,
                    extent,
                );

                Ok(SyncPoint::NO_SYNC)
            }
            &Low::CopyBufferToBuffer { source_buffer, size, target_buffer } => {
                let encoder = match &mut self.command_encoder {
                    Some(encoder) => encoder,
                    None => return Err(StepError::InvalidInstruction(line!())),
                };

                let source = self.descriptors.buffers
                    .get(source_buffer.0)
                    .ok_or(StepError::InvalidInstruction(line!()))?;
                let target = self.descriptors.buffers
                    .get(target_buffer.0)
                    .ok_or(StepError::InvalidInstruction(line!()))?;

                encoder.copy_buffer_to_buffer(
                    source, 0,
                    target, 0,
                    size,
                );

                Ok(SyncPoint::NO_SYNC)
            }
            &Low::ReadBuffer { source_buffer, ref source_layout, offset, size, target_image } => {
                let mut buffers = core::mem::take(&mut self.descriptors.buffers);
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

                let box_me = async move {
                    {
                        let buffer = &buffers[source_buffer.0];
                        let image = &mut image_data[target_image.0];

                        let slice = buffer.slice(..);
                        slice.map_async(wgpu::MapMode::Read)
                            .await
                            .map_err(|wgpu::BufferAsyncError| StepError::InvalidInstruction(line!()))?;

                        eprintln!("Got buffer..");
                        let data = slice.get_mapped_range();

                        // TODO: defensive programming, don't assume cast works.
                        let source_pitch = bytes_per_row as usize;
                        let target_pitch = image.layout().bytes_per_row as usize;
                        let bytes_to_copy = (u32::from(bytes_per_texel) * width) as usize;

                        let source: &[u8] = &data[..];
                        let target: &mut [u8] = image.as_bytes_mut().unwrap();

                        for x in 0..height {
                            eprintln!("{}", x);
                            let source_row = &source[(x as usize * source_pitch)..][..source_pitch];
                            let target_row = &mut target[(x as usize * target_pitch)..][..target_pitch];

                            target_row[..bytes_to_copy].copy_from_slice(&source_row[..bytes_to_copy]);
                        }
                    }

                    Ok(Cleanup::Buffers {
                        buffers,
                        image_data,
                    })
                };

                Ok(SyncPoint {
                    future: Some((Box::pin(box_me), self)),
                    marker: core::marker::PhantomData,
                })
            }
            inner => return Err(StepError::BadInstruction(BadInstruction {
                inner: format!("{:?}", inner),
            })),
        }
    }

    /// Stop the execution.
    pub fn retire(self) -> Result<(), RetireError> {
        todo!()
    }

    /// Stop the execution, depositing all resources into the provided pool.
    #[must_use = "You won't get the ids of outputs."]
    pub fn retire_gracefully<'pool>(self, pool: &'pool mut Pool) -> Retire<'pool> {
        Retire { execution: self, pool }
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

        Ok(wgpu::BindGroupDescriptor {
            label: None,
            layout: self.bind_group_layouts
                .get(desc.layout_idx)
                .ok_or_else(|| StepError::InvalidInstruction(line!()))?,
            entries: buf,
        })
    }

    fn binding_resource(&self, desc: &program::BindingResource)
        -> Result<wgpu::BindingResource<'_>, StepError>
    {
        use program::BindingResource::{Buffer, Sampler, TextureView};
        match desc {
            &Buffer { buffer_idx, offset, size } => {
                let buffer = self.buffers
                    .get(buffer_idx)
                    .ok_or_else(|| StepError::InvalidInstruction(line!()))?;
                Ok(wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer,
                    offset,
                    size,
                }))
            }
            &Sampler(idx) => {
                self.sampler
                    .get(idx)
                    .ok_or_else(|| StepError::InvalidInstruction(line!()))
                    .map(wgpu::BindingResource::Sampler)
            }
            &TextureView(idx) => {
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

    fn color_attachment(&self, desc: &program::ColorAttachmentDescriptor)
        -> Result<wgpu::RenderPassColorAttachment<'_>, StepError>
    {
        Ok(wgpu::RenderPassColorAttachment{
            view: self.texture_views
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
            layout: self.pipeline_layouts
                .get(desc.layout),
            vertex: self.vertex_state(&desc.vertex, vertex_buffers)?,
            primitive: match desc.primitive {
                program::PrimitiveState::SoleQuad => wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    clamp_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                }
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(self.fragment_state(&desc.fragment, fragments)?),
        })
    }

    fn pipeline_layout<'set>(
        &'set self,
        desc: &program::PipelineLayoutDescriptor,
        buf: &'set mut Vec<&'set wgpu::BindGroupLayout>,
    ) -> Result<wgpu::PipelineLayoutDescriptor<'_>, StepError> {
        buf.clear();

        for &layout in &desc.bind_group_layouts {
            let group = self.bind_group_layouts
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
        Ok(wgpu::VertexState {
            module: self.modules
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
            module: self.modules
                .get(desc.fragment_module)
                .ok_or_else(|| StepError::InvalidInstruction(line!()))?,
            entry_point: desc.entry_point,
            targets: buf,
        })
    }

    fn buffer(&self, buffer: DeviceBuffer, layout: &BufferLayout)
        -> Result<wgpu::ImageCopyBuffer<'_>, StepError>
    {
        Ok(wgpu::ImageCopyBufferBase {
            buffer: self.buffers.get(buffer.0).ok_or(StepError::InvalidInstruction(line!()))?,
            layout: wgpu::ImageDataLayout {
                bytes_per_row: NonZeroU32::new(layout.bytes_per_row),
                offset: 0,
                rows_per_image: NonZeroU32::new(layout.height),
            },
        })
    }

    fn texture(&self, texture: DeviceTexture)
        -> Result<wgpu::ImageCopyTexture<'_>, StepError>
    {
        Ok(wgpu::ImageCopyTextureBase {
            texture: self.textures.get(texture.0).ok_or(StepError::InvalidInstruction(line!()))?,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
        })
    }
}

impl Machine {
    pub(crate) fn new(instructions: Vec<Low>) -> Self {
        Machine {
            instructions,
            instruction_pointer: 0,
        }
    }

    fn next_instruction(&mut self) -> Result<&Low, StepError> {
        let instruction = self.instructions
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
            match dbg!(self.next_instruction()?) {
                &Low::SetPipeline(idx) => {
                    let pipeline = descriptors.render_pipelines
                        .get(idx)
                        .ok_or_else(|| StepError::InvalidInstruction(line!()))?;
                    pass.set_pipeline(pipeline);
                }
                &Low::SetBindGroup{ group, index, ref offsets } => {
                    let group = descriptors.bind_groups
                        .get(group)
                        .ok_or_else(|| StepError::InvalidInstruction(line!()))?;
                    pass.set_bind_group(index, group, offsets);
                }
                &Low::SetVertexBuffer { slot, buffer } => {
                    let buffer = descriptors.buffers
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
                &Low::SetPushConstants { stages, offset, ref data } => {
                    pass.set_push_constants(stages, offset, data);
                }
                Low::EndRenderPass => return Ok(()),
                inner => return Err(StepError::BadInstruction(BadInstruction {
                    inner: format!("Unexpectedly within render pass: {:?}", inner),
                })),
            }
        }
    }
}

impl Retire<'_> {
    pub fn output(&mut self, _: Register) -> Result<PoolImage<'_>, RetireError> {
        let data = self.execution.buffers.pop().unwrap();

        let descriptor = crate::buffer::Descriptor {
            layout: data.layout().clone(),
            texel: crate::buffer::Texel::with_srgb_image(&image::DynamicImage::ImageRgb8(Default::default())),
        };

        let mut image = self.pool.declare(descriptor);
        image.replace(data);

        Ok(image.into())
    }

    pub fn output_key(&self, _: Register) -> Result<PoolKey, RetireError> {
        todo!()
    }
}

impl SyncPoint<'_> {
    pub fn block_on(&mut self) -> Result<(), StepError> {
        use crate::program::block_on;
        match self.future.take() {
            None => Ok(()),
            Some((fut, execution)) => match block_on(fut)? {
                Cleanup::Buffers { buffers, image_data } => {
                    execution.descriptors.buffers = buffers;
                    execution.buffers = image_data;
                    Ok(())
                }
            },
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
