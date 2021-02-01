use crate::program::{CompileError, Program};
use crate::buffer::Descriptor;

/// A reference to one particular value.
pub struct Register(usize);

/// One linear sequence of instructions.
pub struct CommandBuffer {
    _todo: u8,
}

pub struct Rectangle {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

pub struct Affine {
    transformation: [f32; 9],
}

impl CommandBuffer {
    /// Declare an input.
    ///
    /// Inputs MUST later be bound from the pool during launch.
    pub fn input(&mut self, _: Descriptor) -> Register {
        todo!()
    }

    /// Select a rectangular part of an image.
    pub fn crop(&mut self, src: Register, rect: Rectangle) -> Register {
        todo!()
    }

    /// Embed this image as part of a larger one.
    pub fn enlarge(&mut self, src: Register, _: Rectangle, new: Register) -> Register {
        todo!()
    }

    /// A solid color image, from a descriptor and a single texel.
    pub fn solid(&mut self, _: Descriptor, _: &[u8]) -> Register {
        todo!()
    }

    /// An affine transformation of the image.
    pub fn affine(&mut self, src: Register, _: Affine) -> Register {
        todo!()
    }

    /// Declare an output.
    ///
    /// Outputs MUST later be bound from the pool during launch.
    pub fn output(&mut self, src: Register, _: Descriptor) -> Register {
        todo!()
    }

    pub fn compile(&self) -> Result<Program, CompileError> {
        todo!()
    }
}
