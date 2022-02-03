use crate::buffer::RowMatrix;
use crate::program::BufferInitContent;
use std::borrow::Cow;
use std::sync::Arc;

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

/// Holds binary representation of a shader's argument.
///
/// This, also, exposes all other publicly available methods to configure the `ShaderInvocation`
/// that will occur when executing the provided shader.
pub struct ShaderData<'lt> {
    data_buffer: &'lt mut Vec<u8>,
    /// Which region of the data buffer corresponds to the initializer for the buffer binding.
    /// Is `None` if the shader does not have a buffer binding.
    content: &'lt mut Option<BufferInitContent>,
}

/// A simple shader invocation.
///
/// This represents _one_ instance of a shader invocation. The compilation will evaluate the
/// methods to determine how the invocation is executed by the runtime pipeline.
pub trait ShaderInvocation: Send + Sync {
    /// Shared, binary shader SPIR-V source.
    ///
    /// It is more efficient if invocations sharing the same shader source return clones of the
    /// exact same allocated source.
    fn spirv_source(&self) -> Arc<[u8]>;

    /// Configure this invocation, such as providing bind buffer data as binary.
    ///
    /// See `ShaderData` for more information. All configuration is performed by calling its
    /// methods, the object is provided by surrounding runtime. You shouldn't depend on the exact
    /// timing of this call relative to other invocations as such ordering may be fragile and
    /// depend on optimization reordering performed during encoding of commands. More specific
    /// guarantees may be provided at a later version of the library.
    fn shader_data(&self, _: ShaderData<'_>);

    /// Provide a debug representation.
    fn debug(&self) -> &dyn core::fmt::Debug {
        static REPLACEMENT: &'static str = "No debug data for shader invocation";
        &REPLACEMENT
    }
}

/// A simplification of a fragment shader interface.
pub(crate) trait FragmentShaderData: core::fmt::Debug {
    /// The unique key identifying this shader pipeline setup.
    /// If two invocations return the same key then they are optimized and _not_ recompiled.
    /// Instead, we reuse setup from a previous shader module creation.
    fn key(&self) -> Option<FragmentShaderKey>;
    /// The SPIR-V shader source code.
    fn spirv_source(&self) -> Cow<'static, [u8]>;
    /// Encode the shader's data into the buffer, returning the descriptor to that.
    fn binary_data(&self, _: &mut Vec<u8>) -> Option<BufferInitContent> {
        None
    }
    /// Number of argument images consumed by the shader.
    /// This must match the number of arguments provided as `High::PushOperand`.
    fn num_args(&self) -> u32 {
        1
    }
}

struct DynamicShader {
    invocation: Box<dyn ShaderInvocation>,
    spirv: Arc<[u8]>,
}

impl core::fmt::Debug for DynamicShader {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "{:?}", self.invocation.debug())
    }
}

impl FragmentShaderData for DynamicShader {
    fn key(&self) -> Option<FragmentShaderKey> {
        Some(FragmentShaderKey::Dynamic(self.spirv.as_ptr() as usize))
    }
    fn spirv_source(&self) -> Cow<'static, [u8]> {
        Cow::Owned(self.spirv.to_vec())
    }
    fn binary_data(&self, data_buffer: &mut Vec<u8>) -> Option<BufferInitContent> {
        let mut content = None;
        self.invocation.shader_data(ShaderData {
            data_buffer,
            content: &mut content,
        });
        content
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
    /// The key is the address of some dynamic object, unique for the duration of the pipeline.
    /// One shouldn't rely on uniqueness of soundness.
    Dynamic(usize),
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
