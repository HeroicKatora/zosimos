use crate::pool::Pool;
use crate::run::{Execution, LaunchError};

/// Planned out and intrinsically validated command buffer.
pub struct Program {
    _todo: u8,
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
