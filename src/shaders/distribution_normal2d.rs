use super::{BufferInitContent, FragmentShaderData, FragmentShaderKey};
use std::borrow::Cow;
use std::f32::consts::PI as PIf32;

/// a linear transformation on rgb color.
pub const SHADER: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/spirv/distribution_normal2d.frag.v"
));

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
    /// Construct a centered normal distribution based on variance in X and Y direction.
    pub fn with_diagonal(var0: f32, var1: f32) -> Shader {
        let d0 = if var0 == 0.0 { 0.0 } else { 1.0 / var0 };
        let d1 = if var1 == 0.0 { 0.0 } else { 1.0 / var1 };

        let f0 = if var0 == 0.0 { 1.0 } else { 2.0 * PIf32 * var0 };
        let f1 = if var1 == 0.0 { 1.0 } else { 2.0 * PIf32 * var1 };

        Shader {
            expectation: [0.0, 0.0],
            covariance_inverse: Mat2 {
                row_major: [d0, 0.0, 0.0, d1],
            },
            pseudo_determinant: f0 * f1,
        }
    }

    /// Construct a 1d, centered normal distribution given the direction.
    ///
    /// The length of the vector defines the variance. The positive direction of the coordinate
    /// axes is towards the bottom and right.
    ///
    /// This is more stable than a construction from the covariance matrices. It is, however, not
    /// stable when the direction vector is very short.
    /// # Panics
    /// This method will panic when the squared length of `dir` is not finite.
    pub fn with_direction(dir: [f32; 2]) -> Shader {
        let [x, y] = dir;
        // The covariance matrix is given by dir^T·dir, with one non-zero eigen value
        // dir·dir^T = length². The pseudo determinant is thusly length².
        // The pseudo inverse is given by matrix (dir^T·dir / length^4)
        // Alternate, four-step, single precision (due to Jean-Michel Muller)
        // w = x*x; u = fma(x, -x, w); v = fma(y, y, w); v - u
        let (xt, yt) = (f64::from(x), f64::from(y));
        let length_sq = (xt * xt + yt * yt) as f32;
        assert!(length_sq.is_finite());

        // x*x / (x*x + y*y)²
        // Optimized with Herbie
        //
        // I don't seriously understand this except for the hypot transform.
        fn herbie_symmetric(x: f32, y: f32) -> f32 {
            let hypot = x.hypot(y);
            let up = (1.0 / hypot) * (x / hypot);
            let low = x + y * (y / x);
            up / low
        }

        // x*y / (x*x + y*y)²
        // Optimized with Herbie
        //
        // Seems quite logical except for the precise order of divisions which might not matter at
        // all in the end.
        fn herbie_asymmetric(x: f32, y: f32) -> f32 {
            let (x, y) = (x.min(y), x.max(y));
            let hypot = x.hypot(y);
            // (x² + y²) / y but different.
            let inner = x.mul_add(x / y, y);
            ((x / hypot) / inner) / hypot
        }

        // ([x y]^T  · [x y]) / (x² + y²)
        let row_major = [
            herbie_symmetric(x, x),
            herbie_asymmetric(x, y),
            herbie_asymmetric(y, x),
            herbie_symmetric(y, y),
        ];

        Shader {
            expectation: [0.0, 0.0],
            covariance_inverse: Mat2 { row_major },
            pseudo_determinant: 2.0 * PIf32 * length_sq,
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

    #[rustfmt::skip]
    fn binary_data(&self, buffer: &mut Vec<u8>) -> Option<BufferInitContent> {
        let Shader {
            expectation: exp,
            covariance_inverse: Mat2 { row_major: inv },
            pseudo_determinant: det,
        } = self;

        let rgb_data: [f32; 8] = [
            exp[0], exp[1],
            inv[0], inv[1], inv[2], inv[3],
            *det,
            0.0
        ];

        Some(BufferInitContent::new(buffer, &rgb_data))
    }

    fn num_args(&self) -> u32 {
        0
    }
}
