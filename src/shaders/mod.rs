use std::borrow::Cow;
use crate::buffer::RowMatrix;
use crate::program::BufferInitContent;

pub mod stage;
pub mod distribution_normal2d;
pub mod palette;

/// A vertex box shader, rendering a sole quad with given vertex and uv coordinate system.
pub const VERT_NOOP: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spirv/box.vert.v"));

/// A 'noop' copy from the sampled texture to the output color based on the supplied UVs.
pub const FRAG_COPY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spirv/copy.frag.v"));
#[allow(dead_code)]
pub const FRAG_MIX_RGBA: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spirv/inject.frag.v"));
/// a linear transformation on rgb color.
pub const FRAG_LINEAR: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spirv/linear.frag.v"));

/// A simplification of a fragment shader interface.
pub(crate) trait FragmentShaderData:
    core::fmt::Debug 
{
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
    /// Sample discrete colors from a palette.
    Palette,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum FragmentShader {
    PaintOnTop(PaintOnTopKind),
    LinearColorMatrix(LinearColorTransform),
    Normal2d(DistributionNormal2d),
    Palette(self::palette::Shader),
}

impl FragmentShader {
    pub(crate) fn shader(&self) -> &dyn FragmentShaderData {
        match self {
            FragmentShader::PaintOnTop(kind) => kind,
            FragmentShader::LinearColorMatrix(shader) => shader,
            FragmentShader::Normal2d(normal) => normal,
            FragmentShader::Palette(palette) => palette,
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
        let matrix = self.matrix.into_inner();

        // std140, always pad to 16 bytes.
        // matrix is an array of its columns.
        let rgb_matrix: [f32; 12] = [
            matrix[0], matrix[3], matrix[6], 0.0,
            matrix[1], matrix[4], matrix[7], 0.0,
            matrix[2], matrix[5], matrix[8], 0.0,
        ];

        Some(BufferInitContent::new(buffer, &rgb_matrix))
    }
}

pub(crate) use self::distribution_normal2d::{Shader as DistributionNormal2d};
pub(crate) use self::palette::Shader as PaletteShader;
