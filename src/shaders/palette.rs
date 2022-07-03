use std::borrow::Cow;

use super::{BufferInitContent, FragmentShaderData, FragmentShaderKey};
use crate::buffer::{ChannelPosition, ColorChannel};

/// a linear transformation on rgb color.
pub const SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spirv/palette.frag.v"));

/// The palette shader, computing texture coordinates from an input color.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Shader {
    pub(crate) x_coord: [f32; 4],
    pub(crate) y_coord: [f32; 4],
    pub(crate) base_x: i32,
    pub(crate) base_y: i32,
}

impl ChannelPosition {
    /// Find the channel index in the normalized, linear representation of said color.
    ///
    /// The caller in `command` is responsible for ensuring
    ///
    /// Reminder: we are looking for the position of the color channel in the _linear_
    /// representation of the color, i.e. within the vec4 loaded from the sampled texture.
    // For further colors later.
    #[allow(unreachable_patterns)]
    pub(crate) fn new(channel: ColorChannel) -> Option<Self> {
        use ColorChannel as Cc;
        Some(match channel {
            Cc::R => ChannelPosition::First,
            Cc::G => ChannelPosition::Second,
            Cc::B => ChannelPosition::Third,
            _ => return None,
        })
    }

    pub(crate) fn into_vec4(self) -> [f32; 4] {
        let mut p = [0.0; 4];
        p[self as usize] = 1.0;
        p
    }
}

impl FragmentShaderData for Shader {
    fn key(&self) -> Option<FragmentShaderKey> {
        Some(FragmentShaderKey::Palette)
    }

    fn spirv_source(&self) -> Cow<'static, [u8]> {
        Cow::Borrowed(SHADER)
    }

    #[rustfmt::skip]
    fn binary_data(&self, buffer: &mut Vec<u8>) -> Option<BufferInitContent> {
        let mat4x2 = [
            self.x_coord[0], self.y_coord[0],
            self.x_coord[1], self.y_coord[1],
            self.x_coord[2], self.y_coord[2],
            self.x_coord[3], self.y_coord[3],
        ];

        Some(BufferInitContent::new(buffer, &mat4x2))
    }

    fn num_args(&self) -> u32 {
        2
    }
}
