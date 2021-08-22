use std::borrow::Cow;
use crate::buffer::RowMatrix;
use crate::program::{BufferInitContent, FragmentShaderKey, PaintOnTopKind};

pub mod stage;

/// A vertex box shader, rendering a sole quad with given vertex and uv coordinate system.
pub const VERT_NOOP: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spirv/box.vert.v"));

/// A 'noop' copy from the sampled texture to the output color based on the supplied UVs.
pub const FRAG_COPY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spirv/copy.frag.v"));
#[allow(dead_code)]
pub const FRAG_MIX_RGBA: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spirv/inject.frag.v"));
/// a linear transformation on rgb color.
pub const FRAG_LINEAR: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spirv/linear.frag.v"));

/// A simplification of a fragment shader interface.
pub(crate) trait FragmentShader:
    core::fmt::Debug 
{
    /// The unique key identifying this shader module.
    fn key(&self) -> Option<FragmentShaderKey>;
    /// The SPIR-V shader source code.
    fn shader(&self) -> Cow<'static, [u8]>;
    /// Encode the shader's data into the buffer, returning the descriptor to that.
    fn binary_data(&self, _: &mut Vec<u8>) -> Option<BufferInitContent> {
        None
    }
}

impl FragmentShader for PaintOnTopKind {
    fn key(&self) -> Option<FragmentShaderKey> {
        Some(FragmentShaderKey::PaintOnTop(self.clone()))
    }

    fn shader(&self) -> Cow<'static, [u8]> {
        Cow::Borrowed(FRAG_COPY)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LinearColorTransform {
    matrix: RowMatrix,
}

impl FragmentShader for LinearColorTransform {
    fn key(&self) -> Option<FragmentShaderKey> {
        Some(FragmentShaderKey::LinearColorMatrix)
    }

    fn shader(&self) -> Cow<'static, [u8]> {
        Cow::Borrowed(FRAG_LINEAR)
    }
}
