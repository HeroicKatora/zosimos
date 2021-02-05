use crate::pool::Pool;
use crate::program::{self, Low};

use wgpu::{Device, Queue};

pub enum LaunchError {
}

pub struct Execution {
    machine: Machine,
    gpu: Gpu,
    descriptors: Descriptors,
    command_encoder: Option<wgpu::CommandEncoder>,
    buffers: Pool,
}

struct Descriptors {
    bind_groups: Vec<wgpu::BindGroup>,
    bind_group_layouts: Vec<wgpu::BindGroupLayout>,
    buffers: Vec<wgpu::Buffer>,
    command_buffers: Vec<wgpu::CommandBuffer>,
    render_pipelines: Vec<wgpu::RenderPipeline>,
    sampler: Vec<wgpu::Sampler>,
    textures: Vec<wgpu::Texture>,
    texture_views: Vec<wgpu::TextureView>,
}

struct Gpu {
    device: Device,
    queue: Queue,
}

pub struct SyncPoint<'a> {
    marker: core::marker::PhantomData<&'a mut Execution>,
    _private: u8,
}

struct Machine {
    instructions: Vec<Low>,
    instruction_pointer: usize,
}

pub enum StepError {
    InvalidInstruction,
    ProgramEnd,
    RenderPassDidNotEnd,
}

pub enum RetireError {
}

impl Execution {
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
                    level_count: None,
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
                        depth: 1
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
                let pipeline = self.descriptors.pipeline(desc)?;
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
                self.machine.render_pass(pass)?;

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
            _ => Err(StepError::InvalidInstruction),
        }
    }

    /// Stop the execution.
    pub fn retire(self) -> Result<(), RetireError> {
        todo!()
    }

    /// Stop the execution, depositing all resources into the provided pool.
    pub fn retire_gracefully(self, pool: &mut Pool) -> Result<(), RetireError> {
        todo!()
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
                Ok(wgpu::BindingResource::Buffer {
                    buffer,
                    offset,
                    size,
                })
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
        buf: &'buf mut Vec<wgpu::RenderPassColorAttachmentDescriptor<'set>>,
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
        -> Result<wgpu::RenderPassColorAttachmentDescriptor<'_>, StepError>
    {
        Ok(wgpu::RenderPassColorAttachmentDescriptor {
            attachment: self.texture_views
                .get(desc.texture_view)
                .ok_or_else(|| StepError::InvalidInstruction)?,
            resolve_target: None,
            ops: desc.ops,
        })
    }

    fn pipeline(&self, _: &program::RenderPipelineDescriptor)
        -> Result<wgpu::RenderPipelineDescriptor<'_>, StepError>
    {
        todo!()
    }
}

impl Machine {
    fn next_instruction(&mut self) -> Result<&Low, StepError> {
        let instruction = self.instructions
            .get(self.instruction_pointer)
            .ok_or(StepError::ProgramEnd)?;
        self.instruction_pointer += 1;
        Ok(instruction)
    }

    fn render_pass(&mut self, pass: wgpu::RenderPass<'_>)
        -> Result<(), StepError>
    {
        loop {
            match self.next_instruction()? {
                Low::EndRenderPass => return Ok(()),
                _ => return Err(StepError::InvalidInstruction),
            }
        }
    }
}

impl SyncPoint<'_> {
    const NO_SYNC: Self = SyncPoint { marker: core::marker::PhantomData, _private: 0 };
}
