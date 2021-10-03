use crate::buffer::RowMatrix;
use crate::program::{BufferInitContent, DeferredBufferInitContentBuilder};
use std::borrow::Cow;

pub mod bilinear;
pub mod box3;
pub mod distribution_normal2d;
pub mod fractal_noise;
pub mod inject;
pub mod oklab;
pub mod palette;
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
pub(crate) trait FragmentShaderData: core::fmt::Debug {
    /// The unique key identifying this shader module.
    fn key(&self) -> Option<FragmentShaderKey>;
    /// The SPIR-V shader source code.
    fn spirv_source(&self) -> Cow<'static, [u8]>;
    /// Encode the shader's data into the buffer, returning the descriptor to that.
    fn binary_data(&self, _: &mut Vec<u8>) -> Option<BufferInitContent> {
        None
    }
    fn num_args(&self) -> u32 {
        1
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum FragmentShaderKey {
    PaintOnTop(PaintOnTopKind),
    /// Linear color transformation.
    LinearColorMatrix,
    /// The conversion of texel format.
    /// FIXME: there are multiple sources of this.
    Convert,
    /// The generic distribution normal 2d.
    DistributionNormal2d,
    /// Generic fractal brownian noise.
    FractalNoise,
    /// Sample discrete colors from a palette.
    Palette,
    /// A bilinear function of colors.
    Bilinear,
    /// A shader mixing two colors, logically injecting ones channel into the other.
    Inject,
    /// A shader transforming between XYZ and Oklab color space.
    OklabTransform(bool),
    /// A convolution with a 3-by-3 box function.
    Box3,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum FragmentShader {
    PaintOnTop(PaintOnTopKind),
    LinearColorMatrix(LinearColorTransform),
    Normal2d(DistributionNormal2d),
    FractalNoise(FractalNoise),
    Palette(self::palette::Shader),
    Bilinear(self::bilinear::Shader),
    Inject(self::inject::Shader),
    Oklab(self::oklab::Shader),
    Box3(self::box3::Shader),
}

impl FragmentShader {
    pub(crate) fn shader(&self) -> &dyn FragmentShaderData {
        match self {
            FragmentShader::PaintOnTop(kind) => kind,
            FragmentShader::LinearColorMatrix(shader) => shader,
            FragmentShader::Normal2d(normal) => normal,
            FragmentShader::FractalNoise(noise) => noise,
            FragmentShader::Palette(palette) => palette,
            FragmentShader::Bilinear(bilinear) => bilinear,
            FragmentShader::Inject(inject) => inject,
            FragmentShader::Oklab(oklab) => oklab,
            FragmentShader::Box3(box3) => box3,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum PaintOnTopKind {
    Copy,
}

impl PaintOnTopKind {
    pub(crate) fn fragment_shader(&self) -> &'static [u8] {
        match self {
            PaintOnTopKind::Copy => FRAG_COPY,
        }
    }
}

impl FragmentShaderData for PaintOnTopKind {
    fn key(&self) -> Option<FragmentShaderKey> {
        Some(FragmentShaderKey::PaintOnTop(self.clone()))
    }

    fn spirv_source(&self) -> Cow<'static, [u8]> {
        Cow::Borrowed(self.fragment_shader())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LinearColorTransform {
    pub matrix: RowMatrix,
}

impl FragmentShaderData for LinearColorTransform {
    fn key(&self) -> Option<FragmentShaderKey> {
        Some(FragmentShaderKey::LinearColorMatrix)
    }

    fn spirv_source(&self) -> Cow<'static, [u8]> {
        Cow::Borrowed(FRAG_LINEAR)
    }

    fn binary_data(&self, buffer: &mut Vec<u8>) -> Option<BufferInitContent> {
        let rgb_matrix: [f32; 12] = self.matrix.into_mat3x3_std140();
        Some(BufferInitContent::new(buffer, &rgb_matrix))
    }
}

pub(crate) use self::distribution_normal2d::Shader as DistributionNormal2d;
pub(crate) use self::fractal_noise::Shader as FractalNoise;
pub(crate) use self::palette::Shader as PaletteShader;
