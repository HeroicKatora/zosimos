use std::borrow::Cow;
use super::{BufferInitContent, FragmentShaderData, FragmentShaderKey};

/// a linear transformation on rgb color.
pub const SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spirv/distribution_normal2d.frag.v"));

#[derive(Clone, Debug, PartialEq)]
pub struct Shader {
    pub expectation: [f32; 2],
    pub covariance_inverse: Mat2,
    pub pseudo_determinant: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Mat2 {
    row_major: [f32; 4],
}

impl Shader {
    pub fn with_diagonal(var0: f32, var1: f32) -> Shader {
        let d0 = if var0 == 0.0 { 0.0 } else { 1.0 / var0 };
        let d1 = if var1 == 0.0 { 0.0 } else { 1.0 / var1 };

        let f0 = if var0 == 0.0 { 1.0 } else { 3.14159265 * var0 };
        let f1 = if var1 == 0.0 { 1.0 } else { 3.14159265 * var1 };

        Shader {
            expectation: [0.0, 0.0],
            covariance_inverse: Mat2 {
                row_major: [d0, 0.0, 0.0, d1],
            },
            pseudo_determinant: f0 * f1,
        }
    }
}

impl FragmentShaderData for Shader {
    fn key(&self) -> Option<FragmentShaderKey> {
        Some(FragmentShaderKey::DistributionNormal2d)
    }

    fn spirv_source(&self) -> Cow<'static, [u8]> {
        Cow::Borrowed(SHADER)
    }

    fn binary_data(&self, buffer: &mut Vec<u8>) -> Option<BufferInitContent> {
        let Shader {
            expectation: exp,
            covariance_inverse: Mat2 {
                row_major: inv,
            },
            pseudo_determinant: det,
        } = self;

        let rgb_data: [f32; 13] = [
            exp[0], exp[1], 0.0, 0.0,
            inv[0], inv[1], 0.0, 0.0,
            inv[2], inv[3], 0.0, 0.0,
            *det,
        ];

        Some(BufferInitContent::new(buffer, &rgb_data))
    }
}
