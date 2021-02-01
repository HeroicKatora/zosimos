use crate::pool::Pool;
use crate::run::{Execution, LaunchError};

/// Planned out and intrinsically validated command buffer.
pub struct Program {
    _todo: u8,
}

/// Low level instruction.
/// Can be scheduled/ran directly on a machine state.
enum Low {
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
pub enum CompileError {
}

/// Something won't work with this program and pool combination, no matter the amount of
/// configuration.
pub enum MismatchError {
}

/// Prepare program execution with a specific pool.
///
/// Some additional assembly and configuration might be required and possible. For example choose
/// specific devices for running, add push attributes,
pub struct Launcher<'program> {
    program: &'program Program,
    pool: &'program mut Pool,
}

impl Program {
    /// Run this program with a pool.
    ///
    /// Required input and output image descriptors must match those declared, or be convertible
    /// to them when a normalization operation was declared.
    pub fn launch<'pool>(&'pool self, pool: &'pool mut Pool)
        -> Result<Launcher<'pool>, MismatchError>
    {
        Ok(Launcher {
            program: self,
            pool,
        })
    }
}

impl Launcher<'_> {
    /// Really launch, potentially failing if configuration or inputs were missing etc.
    pub fn launch(self) -> Result<Execution, LaunchError> {
        todo!()
    }
}
