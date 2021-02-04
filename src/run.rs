use crate::pool::Pool;
use crate::program::{Low, RenderPassDescriptor, RenderPipelineDescriptor};

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
    command_buffers: Vec<wgpu::CommandBuffer>,
    render_pipelines: Vec<wgpu::RenderPipeline>,
    bind_groups: Vec<wgpu::BindGroup>,
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
                let descriptor = self.descriptors.render_pass(descriptor)?;
                let encoder = match &mut self.command_encoder {
                    Some(encoder) => encoder,
                    None => return Err(StepError::InvalidInstruction),
                };

                let pass = encoder.begin_render_pass(&descriptor);
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
    fn render_pass(&self, _: &RenderPassDescriptor)
        -> Result<wgpu::RenderPassDescriptor<'_, '_>, StepError>
    {
        todo!()
    }

    fn pipeline(&self, _: &RenderPipelineDescriptor)
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
