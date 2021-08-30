use std::borrow::Cow;

use super::{BufferInitContent, FragmentShaderData, FragmentShaderKey};

/// a bilinear initialization on a (up to) 4-component color.
pub const SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spirv/bilinear.frag.v"));

/// The palette shader, computing texture coordinates from an input color.
#[derive(Clone, Debug, PartialEq)]
pub struct Shader {
    pub u_min: [f32; 4],
    pub u_max: [f32; 4],
    pub v_min: [f32; 4],
    pub v_max: [f32; 4],
}

impl FragmentShaderData for Shader {
    fn key(&self) -> Option<FragmentShaderKey> {
        Some(FragmentShaderKey::Bilinear)
    }

    fn spirv_source(&self) -> Cow<'static, [u8]> {
        Cow::Borrowed(SHADER)
    }

    fn binary_data(&self, buffer: &mut Vec<u8>) -> Option<BufferInitContent> {
        let mat4x2 = [self.u_min, self.u_max, self.v_min, self.v_max];

        Some(BufferInitContent::new(buffer, &mat4x2))
    }

    fn num_args(&self) -> u32 {
        0
    }
}
