use std::borrow::Cow;

use super::{BufferInitContent, FragmentShaderData, FragmentShaderKey};

/// a bilinear initialization on a (up to) 4-component color.
pub const SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spirv/inject.frag.v"));

/// The palette shader, computing texture coordinates from an input color.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Shader {
    pub mix: [f32; 4],
}

impl FragmentShaderData for Shader {
    fn key(&self) -> Option<FragmentShaderKey> {
        Some(FragmentShaderKey::Inject)
    }

    fn spirv_source(&self) -> Cow<'static, [u8]> {
        Cow::Borrowed(SHADER)
    }

    fn binary_data(&self, buffer: &mut Vec<u8>) -> Option<BufferInitContent> {
        Some(BufferInitContent::new(buffer, &self.mix))
    }

    fn num_args(&self) -> u32 {
        2
    }
}
