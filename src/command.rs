use crate::program::{CompileError, Program};

/// A reference to one particular value.
pub struct Register(usize);

/// One linear sequence of instructions.
pub struct CommandBuffer {
    _todo: u8,
}

impl CommandBuffer {
    pub fn compile(&self) -> Result<Program, CompileError> {
        todo!()
    }
}
