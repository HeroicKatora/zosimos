use crate::program::BufferInitContent;

#[derive(Debug, Clone, PartialEq)]
pub struct Shader([f32; 4]);

pub const SHADER_SOURCE: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/spirv/solid_rgb.frag.v"));

impl super::FragmentShaderData for Shader {
    fn key(&self) -> Option<super::FragmentShaderKey> {
        None
    }

    fn spirv_source(&self) -> std::borrow::Cow<'static, [u8]> {
        std::borrow::Cow::Borrowed(SHADER_SOURCE)
    }

    fn binary_data(&self, buffer: &mut Vec<u8>) -> Option<BufferInitContent> {
        Some(BufferInitContent::new(buffer, &self.0))
    }

    fn num_args(&self) -> u32 {
        0
    }
}

impl From<[f32; 4]> for Shader {
    fn from(value: [f32; 4]) -> Self {
        Shader(value)
    }
}
