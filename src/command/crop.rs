use super::{CommandBuffer, CommandError, Rectangle, Register, Static, UnaryOp};
use crate::types::{Layouter, StaticArena, StaticKind};

#[derive(Clone, Debug)]
pub(crate) struct Op {
    pub rectangle: Rectangle,
    pub slot: Option<Static>,
}

/// A register referring to a crop operation that can be modified.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct CropRegister(Register);

/// A temporary borrow of a command buffer to configure a crop operation.
pub struct Crop<'enc> {
    encoder: &'enc mut CommandBuffer,
    /// Our own register.
    reg: Register,
}

pub trait Encoder {
    fn crop(&mut self, src: Register, rect: Rectangle) -> Result<Register, CommandError>;
    fn crop_with(&mut self, src: Register, rect: Rectangle) -> Result<Crop<'_>, CommandError>;
}

impl Encoder for super::CommandBuffer {
    fn crop(&mut self, src: Register, rect: Rectangle) -> Result<Register, CommandError> {
        let desc = self.describe_reg(src)?.clone();
        Ok(self.push(super::Op::Unary {
            src,
            op: UnaryOp::Crop(Op {
                rectangle: rect,
                slot: None,
            }),
            desc,
        }))
    }

    fn crop_with(&mut self, src: Register, rect: Rectangle) -> Result<Crop<'_>, CommandError> {
        let reg = Encoder::crop(self, src, rect)?;
        Ok(Crop { encoder: self, reg })
    }
}

impl Crop<'_> {
    fn op(&mut self) -> &mut Op {
        match self.encoder.ops[self.reg.0] {
            super::Op::Unary { op: UnaryOp::Crop(ref mut op), .. } => op,
            _ => unreachable!("was added as a crop operation"),
        }
    }

    /// Request the ability to modify the values later.
    ///
    /// WIP: there is no way to actually override these values later.
    pub fn keep_static(&mut self) -> CropRegister {
        let r#static = self.encoder.add_static(&|arena: &mut StaticArena| {
            let point = arena.push(StaticKind::Vec2);
            let rect = arena.push(StaticKind::Array {
                base: point,
                count: 4,
                layout: Layouter::Std430,
            });
            arena.push(StaticKind::After {
                before: rect,
                after: rect,
                layout: Layouter::Std430,
            })
        });

        self.op().slot = Some(r#static);
        CropRegister(self.reg)
    }
}
