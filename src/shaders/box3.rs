use std::borrow::Cow;

use super::{BufferInitContent, FragmentShaderData, FragmentShaderKey};
use crate::color_matrix::RowMatrix;

/// a linear transformation on rgb color.
pub const SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spirv/box3.frag.v"));

/// The palette shader, computing texture coordinates from an input color.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Shader {
    matrix: RowMatrix,
}

impl Shader {
    pub fn new(matrix: RowMatrix) -> Self {
        Shader { matrix }
    }
}

impl FragmentShaderData for Shader {
    /// The unique key identifying this shader module.
    fn key(&self) -> Option<FragmentShaderKey> {
        Some(FragmentShaderKey::Box3)
    }

    /// The SPIR-V shader source code.
    fn spirv_source(&self) -> Cow<'static, [u8]> {
        Cow::Borrowed(SHADER)
    }

    /// Encode the shader's data into the buffer, returning the descriptor to that.
    fn binary_data(&self, buffer: &mut Vec<u8>) -> Option<BufferInitContent> {
        let data = self.matrix.into_mat3x3_std140();

        Some(BufferInitContent::new(buffer, &data))
    }

    fn num_args(&self) -> u32 {
        1
    }
}
