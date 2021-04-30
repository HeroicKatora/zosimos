use core::iter::once;

use crate::pool::{Pool, PoolImage, PoolKey};
use crate::command::Register;
use crate::program::{self, Low};

use wgpu::{Device, Queue};

pub struct Execution {
    pub(crate) machine: Machine,
    pub(crate) gpu: Gpu,
    pub(crate) descriptors: Descriptors,
    pub(crate) command_encoder: Option<wgpu::CommandEncoder>,
    pub(crate) buffers: Pool,
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

pub struct SyncPoint<'a> {
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
    InvalidInstruction,
    ProgramEnd,
    RenderPassDidNotEnd,
}

#[derive(Debug)]
pub enum RetireError {
}

impl Execution {
    /// Check if the machine is still running.
    pub fn is_running(&self) -> bool {
        self.machine.instruction_pointer < self.machine.instructions.len()
    }

    pub fn step(&mut self) -> Result<SyncPoint<'_>, StepError> {
        match self.machine.next_instruction()? {
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
                use wgpu::BufferUsage as U;
                let desc = wgpu::BufferDescriptor {
                    label: None,
                    size: desc.size,
                    usage: match desc.usage {
                        program::BufferUsage::InVertices => U::MAP_WRITE | U::VERTEX,
                        program::BufferUsage::DataIn => U::MAP_WRITE | U::STORAGE | U::COPY_SRC,
                        program::BufferUsage::DataOut => U::MAP_READ | U::STORAGE | U::COPY_DST,
                        program::BufferUsage::DataInOut => {
                            U::MAP_READ | U::MAP_WRITE | U::STORAGE | U::COPY_SRC | U::COPY_DST
                        }
                        program::BufferUsage::Uniform => U::MAP_WRITE | U::STORAGE | U::COPY_SRC,
                    },
                    mapped_at_creation: false,
                };
                let buffer = self.gpu.device.create_buffer(&desc);
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
                let texture = self.descriptors.textures
                    .get(desc.texture)
                    .ok_or_else(|| StepError::InvalidInstruction)?;
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
                        width: desc.size.0,
                        height: desc.size.1,
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
                    return Err(StepError::InvalidInstruction);
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
                    None => return Err(StepError::InvalidInstruction),
                };

                let pass = encoder.begin_render_pass(&descriptor);
                drop(attachment_buf);
                self.machine.render_pass(&self.descriptors, pass)?;

                Ok(SyncPoint::NO_SYNC)
            },
            Low::EndCommands => {
                match self.command_encoder.take() {
                    None => Err(StepError::InvalidInstruction),
                    Some(encoder) => {
                        self.descriptors.command_buffers.push(encoder.finish());
                        Ok(SyncPoint::NO_SYNC)
                    }
                }
            }
            &Low::RunTopCommand => {
                let command = self.descriptors.command_buffers
                    .pop()
                    .ok_or_else(|| StepError::InvalidInstruction)?;
                self.gpu.queue.submit(once(command));
                Ok(SyncPoint::NO_SYNC)
            }
            &Low::RunTopToBot(many) => {
                if many > self.descriptors.command_buffers.len() {
                    return Err(StepError::InvalidInstruction);
                }

                let commands = self.descriptors.command_buffers.drain(many..);
                self.gpu.queue.submit(commands.rev());
                Ok(SyncPoint::NO_SYNC)
            }
            &Low::RunBotToTop(many) => {
                if many > self.descriptors.command_buffers.len() {
                    return Err(StepError::InvalidInstruction);
                }

                let commands = self.descriptors.command_buffers.drain(many..);
                self.gpu.queue.submit(commands);
                Ok(SyncPoint::NO_SYNC)
            }
            _ => Err(StepError::InvalidInstruction),
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
                .ok_or_else(|| StepError::InvalidInstruction)?,
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
                    .ok_or_else(|| StepError::InvalidInstruction)?;
                Ok(wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer,
                    offset,
                    size,
                }))
            }
            &Sampler(idx) => {
                self.sampler
                    .get(idx)
                    .ok_or_else(|| StepError::InvalidInstruction)
                    .map(wgpu::BindingResource::Sampler)
            }
            &TextureView(idx) => {
                self.texture_views
                    .get(idx)
                    .ok_or_else(|| StepError::InvalidInstruction)
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
                .ok_or_else(|| StepError::InvalidInstruction)?,
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
                .ok_or_else(|| StepError::InvalidInstruction)?;
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
                .ok_or_else(|| StepError::InvalidInstruction)?,
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
                .ok_or_else(|| StepError::InvalidInstruction)?,
            entry_point: desc.entry_point,
            targets: buf,
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
            match self.next_instruction()? {
                &Low::SetPipeline(idx) => {
                    let pipeline = descriptors.render_pipelines
                        .get(idx)
                        .ok_or_else(|| StepError::InvalidInstruction)?;
                    pass.set_pipeline(pipeline);
                }
                &Low::SetBindGroup{ group, index, ref offsets } => {
                    let group = descriptors.bind_groups
                        .get(group)
                        .ok_or_else(|| StepError::InvalidInstruction)?;
                    pass.set_bind_group(index, group, offsets);
                }
                &Low::SetVertexBuffer { slot, buffer } => {
                    let buffer = descriptors.buffers
                        .get(buffer)
                        .ok_or_else(|| StepError::InvalidInstruction)?;
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
                _ => return Err(StepError::InvalidInstruction),
            }
        }
    }
}

impl Retire<'_> {
    pub fn output(&self, _: Register) -> Result<PoolImage<'_>, RetireError> {
        todo!()
    }

    pub fn output_key(&self, _: Register) -> Result<PoolKey, RetireError> {
        todo!()
    }
}

impl SyncPoint<'_> {
    const NO_SYNC: Self = SyncPoint { marker: core::marker::PhantomData };
}
