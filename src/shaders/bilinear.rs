use std::borrow::Cow;

use super::{BufferInitContent, FragmentShaderData, FragmentShaderKey};

/// a bilinear initialization on a (up to) 4-component color.
pub const SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spirv/bilinear.frag.v"));

/// The palette shader, computing texture coordinates from an input color.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Shader {
    pub u_min: [f32; 4],
    pub u_max: [f32; 4],
    pub v_min: [f32; 4],
    pub v_max: [f32; 4],
    pub uv_min: [f32; 4],
    pub uv_max: [f32; 4],
}

impl Shader {
    pub fn mgrid(width: f32, height: f32) -> Self {
        Shader {
            u_min: [0.0; 4],
            u_max: [width, 0.0, 0.0, 0.0],
            v_min: [0.0; 4],
            v_max: [0.0, height, 0.0, 0.0],
            uv_min: [0.0; 4],
            uv_max: [0.0; 4],
        }
    }

    pub fn into_std430(&self) -> Vec<u8> {
        let mat = [
            self.u_min,
            self.u_max,
            self.v_min,
            self.v_max,
            self.uv_min,
            self.uv_max,
        ];

        bytemuck::bytes_of(&mat).to_vec()
    }
}

impl FragmentShaderData for Shader {
    fn key(&self) -> Option<FragmentShaderKey> {
        Some(FragmentShaderKey::Bilinear)
    }

    fn spirv_source(&self) -> Cow<'static, [u8]> {
        Cow::Borrowed(SHADER)
    }

    fn binary_data(&self, buffer: &mut Vec<u8>) -> Option<BufferInitContent> {
        let mat = [
            self.u_min,
            self.u_max,
            self.v_min,
            self.v_max,
            self.uv_min,
            self.uv_max,
        ];

        Some(BufferInitContent::new(buffer, &mat))
    }

    fn num_args(&self) -> u32 {
        0
    }
}
