mod dynamic;

pub use self::dynamic::{ShaderCommand, ShaderData, ShaderSource};

use crate::buffer::{BufferLayout, ByteLayout, ChannelPosition, Descriptor, TexelExt};
use crate::color_matrix::RowMatrix;
use crate::pool::PoolImage;
use crate::program::{
    CompileError, Frame, Function, ImageBufferAssignment, ImageBufferPlan, ImageDescriptor, Program, QuadTarget,
    Texture,
};
pub use crate::shaders::bilinear::Shader as Bilinear;
pub use crate::shaders::distribution_normal2d::Shader as DistributionNormal2d;
pub use crate::shaders::fractal_noise::Shader as FractalNoise;

use crate::shaders::{self, FragmentShader, PaintOnTopKind, ShaderInvocation};

use image_canvas::color::{Color, ColorChannel, Whitepoint};
use image_canvas::layout::{SampleParts, Texel};

use std::cmp::Ordering;
use std::collections::HashMap;

/// A reference to one particular value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Register(pub(crate) usize);

/// One linear sequence of instructions.
///
/// The machine model is a single basic block in SSA where registers are strongly typed with their
/// buffer descriptors.
///
/// *Why not a … stack machine*: The author believes that stack machines are a poor fit for image
/// editing in general. Their typical core assumption is that a) registers have the same size b)
/// copying them is cheap. Neither is true for images.
///
/// *Why not a … mutable model*: Because this would complicate the tracking of types. Due to the
/// lack of loops there is little reason for mutability anyways. If you wrap the program in a loop
/// to simulate branches yourself then each `launch` provides the opportunity to rebind the images
/// or bind an image to an output, where it can be mutated.
///
/// The strict typing and SSA-liveness analysis allows for a clean analysis of required temporary
/// resources, re-usage of those as an internal optimization, and in general simple analysis and
/// verification.
#[derive(Default)]
pub struct CommandBuffer {
    ops: Vec<Op>,
}

#[derive(Clone, Debug)]
enum Op {
    /// i := in()
    Input { desc: Descriptor },
    /// out(src)
    ///
    /// WIP: and is_cpu_type(desc)
    /// for the eventuality of gpu-only buffer layouts.
    Output { src: Register },
    /// target(src)
    /// where is_linear_type(src)
    ///
    /// for the eventuality of gpu-only buffer layouts.
    /// FIXME: already contain proof of is_linear_type?
    Render { src: Register },
    /// i := op()
    /// where type(i) = desc
    Construct { desc: Descriptor, op: ConstructOp },
    /// i := unary(src)
    /// where type(i) =? Op[type(src)]
    Unary {
        src: Register,
        op: UnaryOp,
        desc: Descriptor,
    },
    /// i := binary(lhs, rhs)
    /// where type(i) =? Op[type(lhs), type(rhs)]
    Binary {
        lhs: Register,
        rhs: Register,
        op: BinaryOp,
        desc: Descriptor,
    },
    Dynamic {
        call: OperandKind,
        /// The planned shader invocation.
        command: ShaderInvocation,
        desc: Descriptor,
    },
}

#[derive(Clone, Debug)]
enum OperandKind {
    Construct,
    Unary(Register),
    Binary { lhs: Register, rhs: Register },
}

#[derive(Clone, Debug)]
pub(crate) enum ConstructOp {
    Bilinear(Bilinear),
    /// A 2d normal distribution.
    DistributionNormal(shaders::DistributionNormal2d),
    /// Fractal noise
    DistributionNoise(shaders::FractalNoise),
    // TODO: can optimize this repr for the common case.
    Solid(Vec<u8>),
}

/// A high-level, device independent, translation of ops.
/// The main difference to Op is that this is no longer in SSA-form, and it may reinterpret and
/// reuse resources. In particular it will ran after the initial liveness analysis.
/// This will also return the _available_ strategies for one operation. For example, some texels
/// can not be represented on the GPU directly, depending on available formats, and need to be
/// either processed on the CPU (with SIMD hopefully) or they must be converted first, potentially
/// in a compute shader.
#[derive(Clone, Debug)]
pub(crate) enum High {
    /// Assign a texture id to an input with given descriptor.
    /// This instructs the program to insert instructions that load the image from the input in the
    /// pool into the associated texture buffer.
    Input(Register, Descriptor),
    /// Designate the ith textures as output n, according to the position in sequence of outputs.
    Output {
        /// The source register/texture/buffers.
        src: Register,
        /// The target texture.
        dst: Register,
    },
    Render {
        /// The source register/texture/buffers.
        src: Register,
        /// The target texture.
        dst: Register,
    },
    /// Add an additional texture operand to the next operation.
    PushOperand(Texture),
    /// Call a function on the currently prepared operands.
    Construct { dst: Target, fn_: Function },
    /// Last phase marking a register as done.
    /// This is emitted after the Command defining the register has been translated.
    Done(Register),
    /// Copy binary data from a buffer to another.
    Copy { src: Register, dst: Register },
    /// Push one high-level function marker.
    StackPush(Frame),
    /// Pop a high-level function marker.
    StackPop,
}

/// The target image texture of a paint operation (pipeline).
#[derive(Clone, Copy, Debug)]
pub(crate) enum Target {
    /// The data in the texture is to be discarded.
    Discard(Texture),
    /// The data in the texture must be loaded.
    Load(Texture),
}

#[derive(Clone, Debug)]
pub(crate) enum UnaryOp {
    /// Op = id
    Crop(Rectangle),
    /// Op(color)[T] = T[.color=color]
    /// And color needs to be 'color compatible' with the prior T (see module).
    ColorConvert(ColorConversion),
    /// Op(T) = T[.color=select(channel, color)]
    Extract { channel: ChannelPosition },
    /// Op(T) = T[.whitepoint=target]
    /// This is a partial method for CIE XYZ-ish color spaces. Note that ICC requires adaptation to
    /// D50 for example as the reference color space.
    ChromaticAdaptation(ChromaticAdaptation),
    /// Op(T) = T[.texel=texel]
    /// And the byte width of new texel must be consistent with the current byte width.
    Transmute,
    /// Op(T) = T
    Derivative(Derivative),
}

#[derive(Clone, Debug)]
pub(crate) enum BinaryOp {
    /// Op = id
    Affine(Affine),
    /// Op[T, U] = T
    /// where T = U
    Inscribe { placement: Rectangle },
    /// Replace a channel T with U itself.
    /// Op[T, U] = T
    /// where select(channel, T.color) = U.color
    Inject {
        channel: ChannelPosition,
        from_channels: Texel,
    },
    /// Sample from a palette based on the color value of another image.
    /// Op[T, U] = T
    Palette(shaders::PaletteShader),
}

/// A rectangle in `u32` space.
/// It's describe by minimum and maximum coordinates, inclusive and exclusive respectively. Any
/// rectangle where the order is not correct is interpreted as empty. This has the advantage of
/// simplifying certain operations that would otherwise need to check for correctness.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Rectangle {
    pub x: u32,
    pub y: u32,
    pub max_x: u32,
    pub max_y: u32,
}

#[derive(Clone, Copy)]
#[non_exhaustive]
pub enum Blend {
    Alpha,
}

/// Describes an affine transformation of an image.
///
/// Affine transformations are a combination of scaling, translation, rotation. They describe a
/// transformation of the 2D space of the original image.
#[derive(Clone, Copy, Debug)]
pub struct Affine {
    /// The affine transformation, as a row-major homogeneous matrix.
    ///
    /// Note that the top-left pixel starts at (0, 0), the bottom right pixel ends at (1, 1).
    pub transformation: [f32; 9],
    /// How pixels are resolved from the underlying texture.
    pub sampling: AffineSample,
}

/// The way to perform sampling of an texture that was transformed with an affine transformation.
///
/// You have to be careful that there is NO built-in functionality to avoid attacks that downscale
/// an image so far that a very particular subset of pixels (or linear interpolation) is shown that
/// results in an image visually very different from the original. Such an attack works because
/// scaling down leads to many pixels being ignored.
#[derive(Clone, Copy, Debug)]
pub enum AffineSample {
    /// Choose the nearest pixel.
    ///
    /// This method works with all color models.
    Nearest,
    /// Interpolate bi-linearly between nearest pixels.
    ///
    /// We rely on the executing GPU sampler2D for determining the color, in particular it will happen
    /// in _linear_ RGB and this method can only be used on RGB-ish images.
    BiLinear,
}

/// The parameters of color conversion which we will use in the draw call.
#[derive(Clone, Debug)]
pub(crate) enum ColorConversion {
    Xyz {
        /// The matrix converting source to XYZ.
        to_xyz_matrix: RowMatrix,
        /// The matrix converting from XYZ to target.
        from_xyz_matrix: RowMatrix,
    },
    XyzToOklab {
        /// The matrix converting source to XYZ.
        to_xyz_matrix: RowMatrix,
    },
    OklabToXyz {
        /// The matrix converting from XYZ to target.
        from_xyz_matrix: RowMatrix,
    },
    XyzToSrLab2 {
        /// The matrix converting source to XYZ.
        to_xyz_matrix: RowMatrix,
        /// The SrLAb2 target whitepoint.
        whitepoint: Whitepoint,
    },
    SrLab2ToXyz {
        /// The matrix converting from XYZ to target.
        from_xyz_matrix: RowMatrix,
        /// The SrLAb2 source whitepoint.
        whitepoint: Whitepoint,
    },
}

/// Reference of matrices and more: http://brucelindbloom.com/index.html?Eqn_ChromAdapt.html
///
/// A similar technique can simulate cone deficiencies:
/// * deuteranomaly (green cone cells defective),
/// * protanomaly (red cone cells defective),
/// * and tritanomaly (blue cone cells defective).
/// More information here: http://colorspace.r-forge.r-project.org/articles/color_vision_deficiency.html
///
/// Matrix for transforming cone response into the opponent color space which is assumed to be a
/// mostly sufficient input to recreate a particular color impression. In other words, simulate a
/// dichromacy response to a color by matching it for a 'standard' human. We can also estimate if a
/// particular color can be made visible for someone afflicted with a cone deficiency etc.
/// (see: Gustavo M. Machado, et.al A Physiologically-based Model for Simulation of Color Vision Deficiency)
///
/// ```text
/// 0.600    0.400    0.000
/// 0.240    0.105   −0.700
/// 1.200   −1.600    0.400
/// ```
#[derive(Clone, Debug)]
pub(crate) struct ChromaticAdaptation {
    /// The matrix converting source to XYZ.
    to_xyz_matrix: RowMatrix,
    /// The target whitepoint of the adaptation.
    source: Whitepoint,
    /// The method to use.
    method: ChromaticAdaptationMethod,
    /// The matrix converting from XYZ to target.
    from_xyz_matrix: RowMatrix,
    /// The target whitepoint of the adaptation.
    target: Whitepoint,
}

#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum ChromaticAdaptationMethod {
    /// Naive adaptation based on component-wise linear transform in XYZ.
    Xyz,
    /// A component-wise transform in LMS (cone response) coordinates.
    ///
    /// The matrix for whitepoint E (equal intensity) is:
    ///
    /// ```latex
    /// \begin{bmatrix}
    /// 0.38971 & 0.68898 & -0.07868\\
    /// -0.22981 & 1.18340 & 0.04641\\
    /// 0.00000 & 0.00000 & 1.00000
    /// \end{bmatrix}
    /// ```
    ///
    /// The D65 normalized XYZ -> LMS matrix is:
    ///
    /// ```text
    /// 0.4002400  0.7076000 -0.0808100
    /// -0.2263000  1.1653200  0.0457000
    /// 0.0000000  0.0000000  0.9182200
    /// ```
    VonKries,
    /// Bradford's modified (sharpened) LMS definition with linear VonKries adaptation.
    /// Used in ICC.
    ///
    /// ```latex
    /// \begin{bmatrix}
    /// 0.8951 & 0.2664 & -0.1614 \\
    /// -0.7502 & 1.7135 & 0.0367 \\
    /// 0.0389 & -0.0685 & 1.0296
    /// \end{bmatrix}
    /// ```
    BradfordVonKries,
    /// Bradford's originally intended adaptation.
    BradfordNonLinear,
}

/// A palette lookup operation.
///
/// FIXME description and implementation
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Palette {
    /// Which color channel will provide the texture coordinate along width axis.
    pub width: Option<ColorChannel>,
    /// Which color channel will provide the texture coordinate along height axis.
    pub height: Option<ColorChannel>,
    /// The base coordinate for sampling along width.
    pub width_base: i32,
    /// The base coordinate for sampling along height.
    pub height_base: i32,
    // FIXME: wrapping?
}

/// Calculate a first derivative.
#[derive(Clone, Debug, Hash)]
pub struct Derivative {
    pub method: DerivativeMethod,
    pub direction: Direction,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Direction {
    /// Along the height of the image.
    Height,
    /// Along the width of the image.
    Width,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DerivativeMethod {
    /// A 2-sized filter with diagonal basis direction.
    ///     [[1 0] [0 -1]]
    ///     [[0 1] [-1 0]]
    Roberts,
    /// The result of derivative with average smoothing.
    ///     1/6 [1 1 1]^T [1 0 -1]
    Prewitt,
    /// The result of derivative with weighted smoothing.
    ///     1/8 [1 2 1]^T [1 0 -1]
    Sobel,
    /// The Scharr 3×3 operator, floating point precision with the error bound of the paper.
    ///
    /// There is a single numerically consistent choice in differentiation but the smoothing for
    /// rotational symmetry is:
    ///     [46.84 162.32 46.84]/256
    ///
    /// It is optimized to most accurately model the perfect transfer function `pi·i·w` (the first
    /// derivative operator) while providing the ability to steer the direction. That is, you can
    /// calculate the derivative into any direction by composing Dx and Dy. The weights for this
    /// optimization had been `w(k) = ∏i:(1 to D) cos^4(π/2·k_i)` where `D = 2` in the flat image
    /// case.
    ///
    /// It's a bit constricting to use it here because on the GPU integer arithmetic is _NOT_ the
    /// most efficient form of computation (which is the motivation behind using a quantized
    /// integer matrix). They nevertheless exist for compatibility reasons.
    ///
    /// Reference: Scharr's dissertation, Optimale Operatoren in der Digitalen Bildverarbeitung
    /// <https://doi.org/10.11588/heidok.00000962>
    /// <http://archiv.ub.uni-heidelberg.de/volltextserver/962/1/Diss.pdf>
    Scharr3,
    /// A 4-tab derivative operator by Scharr.
    /// * Derivative: `[77.68 139.48 …]/256`
    /// * Smoothing: `[16.44 111.56 …]/256`
    Scharr4,
    /// A 5-tab derivative by Scharr.
    /// * Derivative: `[21.27 85.46 0 …]/256`
    /// * Smoothing: `[5.91 61.77 120.64 …]/256`
    Scharr5,
    /// The 4-bit approximated Scharr operator.
    ///     1/32 [3 10 3]^T [1 0 -1]
    ///
    /// This is provided for compatibility! For accuracy you may instead prefer to use Scharr3 or
    /// Schar5Tab.
    Scharr3To4Bit,
    /// A non-smoothed 5-tab derivative.
    ///     `[-0.262 1.525 0 -1.525 0.262]`
    Scharr5Tab,
    /// The 8-bit approximated Scharr operator.
    ///
    /// This is provided for compatibility! For accuracy you may instead prefer to use Scharr3.
    Scharr3To8Bit,
}

/// Methods for removing noise from an image.
///
/// WIP: these are not yet implemented.
///
/// This intuitive understanding applies to single valued, gray scale images. The operator will
/// also work for any colored images as long as the color space defines a luminance, lightness,
/// or value channel. We will then choose a pixel by median of that channel.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SmoothingMethod {
    /// Also called: average, arithmetic mean.
    Laplace,
    /// Weighted average using a gauss kernel.
    Gaussian,
    /// Choose the median value from surrounding pixels.
    ///
    /// The choice is made through the Luma channel.
    Median,
    /// Bilateral filter, weighting pixels by values.
    ///
    /// The weighting is made through the Luma channel.
    Bilteral,
    /// Chooses a value from the surrounding region with minimal variance.
    ///
    /// This implements Kuwahara's initial approach where the representation is chosen to be the
    /// mean value and the regions are defined as exactly 4 regions overlapping on the axes and the
    /// pixel itself.
    ///
    /// Using hexadecimal bit masks to represent the regions, an example with 3×3 regions:
    ///
    /// ```text
    /// 11322
    /// 11322
    /// 55faa
    /// 44c88
    /// 44c88
    /// ```
    Kuwahara,
}

#[derive(Debug)]
pub struct CommandError {
    inner: CommandErrorKind,
}

#[derive(Debug)]
enum CommandErrorKind {
    BadDescriptor(Descriptor),
    ConflictingTypes(Descriptor, Descriptor),
    GenericTypeError,
    Other,
    Unimplemented,
}

/// Intrinsically defined methods of manipulating images.
///
/// FIXME: missing functions
/// - a way to represent colors as a function of wavelength, then evaluate with the standard
/// observers.
/// - implement simulation of color blindness (reuses matrix multiplication, related to observer)
/// - generate color ramps
/// - color interpolation along a cubic spline (how do we represent the parameters?)
/// - hue shift, transforms on the a*b* circle, such as mobius transform (z-a)/(1-z·adj(a)) i.e or
///   other holomorphic functions. Ways to construct mobius transfrom from three key points. That
///   is particular relevant for color correction.
///
/// For developers aiming to add extensions to the system, see the other impl-block.
///
/// The order of arguments is generally
/// 1. Input argument
///   a. Configuration of first argument.
///   b. Configuration of second argument …
///   c. … (no operator with more argument atm)
/// 2. Arguments to the command itself
impl CommandBuffer {
    /// Declare an input.
    ///
    /// Inputs MUST later be bound from the pool during launch.
    pub fn input(&mut self, desc: Descriptor) -> Result<Register, CommandError> {
        if !desc.is_consistent() {
            return Err(CommandError {
                inner: CommandErrorKind::BadDescriptor(desc),
            });
        }

        Ok(self.push(Op::Input { desc }))
    }

    /// Declare an image as input.
    ///
    /// Returns its register if the image has a valid descriptor, otherwise returns an error.
    pub fn input_from(&mut self, img: PoolImage) -> Register {
        let descriptor = img.descriptor();
        self.input(descriptor)
            .expect("Pool image descriptor should be valid")
    }

    /// Select a rectangular part of an image.
    pub fn crop(&mut self, src: Register, rect: Rectangle) -> Result<Register, CommandError> {
        let desc = self.describe_reg(src)?.clone();
        Ok(self.push(Op::Unary {
            src,
            op: UnaryOp::Crop(rect),
            desc,
        }))
    }

    /// Create an image with different color encoding.
    ///
    /// This goes through linear RGB, not ICC, and requires the two models to have same whitepoint.
    pub fn color_convert(
        &mut self,
        src: Register,
        color: Color,
        texel: Texel,
    ) -> Result<Register, CommandError> {
        let desc_src = self.describe_reg(src)?;
        let conversion;

        // Pretend that all colors with the same whitepoint will be mapped from encoded to
        // linear RGB when loading, and re-encoded in target format when storing them. This is
        // almost correct, but not all GPUs will support all texel kinds. In particular
        // some channel orders or bit-field channels are likely to be unsupported. In these
        // cases, we will later add some temporary conversion.
        //
        // FIXME: this is growing a bit ugly with non-rgb color spaces. We should find a more
        // general way to handle these, and in particular also handle non-rgb-to-non-rgb because
        // that's already happening in the command encoder anyways..

        match (&desc_src.color, &color) {
            (
                Color::Rgb {
                    primary: primary_src,
                    whitepoint: wp_src,
                    ..
                },
                Color::Rgb {
                    primary: primary_dst,
                    whitepoint: wp_dst,
                    ..
                },
            ) if wp_src == wp_dst => {
                conversion = ColorConversion::Xyz {
                    from_xyz_matrix: RowMatrix(primary_src.to_xyz_row_matrix(*wp_src)),
                    to_xyz_matrix: RowMatrix(primary_dst.to_xyz_row_matrix(*wp_dst)),
                };
            }
            (
                Color::Rgb {
                    primary,
                    whitepoint: Whitepoint::D65,
                    ..
                },
                Color::Oklab,
            ) => {
                conversion = ColorConversion::XyzToOklab {
                    to_xyz_matrix: RowMatrix(primary.to_xyz_row_matrix(Whitepoint::D65)),
                };
            }
            (
                Color::Oklab,
                Color::Rgb {
                    primary,
                    whitepoint: Whitepoint::D65,
                    ..
                },
            ) => {
                conversion = ColorConversion::OklabToXyz {
                    from_xyz_matrix: RowMatrix(primary.to_xyz_row_matrix(Whitepoint::D65)),
                };
            }
            (
                Color::Rgb {
                    primary,
                    whitepoint: rgb_wp,
                    ..
                },
                Color::SrLab2 { whitepoint },
            ) => {
                conversion = ColorConversion::XyzToSrLab2 {
                    to_xyz_matrix: RowMatrix(primary.to_xyz_row_matrix(*rgb_wp)),
                    whitepoint: *whitepoint,
                };
            }
            (
                Color::SrLab2 { whitepoint },
                Color::Rgb {
                    primary,
                    whitepoint: rgb_wp,
                    ..
                },
            ) => {
                conversion = ColorConversion::SrLab2ToXyz {
                    from_xyz_matrix: RowMatrix(primary.to_xyz_row_matrix(*rgb_wp)),
                    whitepoint: *whitepoint,
                };
            }
            _ => {
                return Err(CommandError {
                    inner: CommandErrorKind::BadDescriptor(desc_src.clone()),
                })
            }
        }

        // FIXME: validate memory condition.
        let layout = ByteLayout {
            width: desc_src.layout.width,
            height: desc_src.layout.height,
            texel_stride: texel.bits.bytes(),
            row_stride: desc_src.layout.width as u64 * texel.bits.bytes() as u64,
        };

        let op = Op::Unary {
            src,
            op: UnaryOp::ColorConvert(conversion),
            desc: Descriptor {
                color,
                layout,
                texel,
            },
        };

        Ok(self.push(op))
    }

    /// Perform a whitepoint adaptation.
    ///
    /// The `function` describes the method and target whitepoint of the chromatic adaptation.
    #[allow(unreachable_patterns)]
    pub fn chromatic_adaptation(
        &mut self,
        src: Register,
        method: ChromaticAdaptationMethod,
        target: Whitepoint,
    ) -> Result<Register, CommandError> {
        let desc_src = self.describe_reg(src)?;
        let texel_color;
        let source_wp;
        let (to_xyz_matrix, from_xyz_matrix);

        match desc_src.color {
            Color::Rgb {
                whitepoint,
                primary,
                transfer,
                luminance,
            } => {
                texel_color = Color::Rgb {
                    whitepoint: target,
                    primary,
                    transfer,
                    luminance,
                };

                to_xyz_matrix = RowMatrix(primary.to_xyz_row_matrix(whitepoint));
                from_xyz_matrix = RowMatrix(primary.from_xyz_row_matrix(target));
                source_wp = whitepoint;
            }
            // Forward compatibility.
            _ => {
                return Err(CommandError {
                    inner: CommandErrorKind::BadDescriptor(desc_src.clone()),
                })
            }
        };

        let desc = Descriptor {
            color: texel_color,
            ..desc_src.clone()
        };

        let op = Op::Unary {
            src,
            op: UnaryOp::ChromaticAdaptation(ChromaticAdaptation {
                to_xyz_matrix,
                source: source_wp,
                target,
                from_xyz_matrix,
                method,
            }),
            desc,
        };

        Ok(self.push(op))
    }

    /// Embed this image as part of a larger one.
    pub fn inscribe(
        &mut self,
        below: Register,
        rect: Rectangle,
        above: Register,
    ) -> Result<Register, CommandError> {
        let desc_below = self.describe_reg(below)?;
        let desc_above = self.describe_reg(above)?;

        if desc_above.texel != desc_below.texel {
            return Err(CommandError {
                inner: CommandErrorKind::ConflictingTypes(desc_below.clone(), desc_above.clone()),
            });
        }

        if Rectangle::with_layout(&desc_above.layout) != rect {
            return Err(CommandError::OTHER);
        }

        if !Rectangle::with_layout(&desc_below.layout).contains(rect) {
            return Err(CommandError::OTHER);
        }

        let op = Op::Binary {
            lhs: below,
            rhs: above,
            op: BinaryOp::Inscribe {
                placement: rect.normalize(),
            },
            desc: desc_below.clone(),
        };

        Ok(self.push(op))
    }

    /// Extract some channels from an image data into a new view.
    pub fn extract(
        &mut self,
        src: Register,
        channel: ColorChannel,
    ) -> Result<Register, CommandError> {
        let desc = self.describe_reg(src)?;
        let texel = desc
            .texel
            .channel_texel(channel)
            .ok_or(CommandError::OTHER)?;

        let layout = ByteLayout {
            texel_stride: texel.bits.bytes(),
            width: desc.layout.width,
            height: desc.layout.height,
            row_stride: (texel.bits.bytes() as u64) * u64::from(desc.layout.width),
        };

        let color = desc.color.clone();

        // Check that we can actually extract that channel.
        // This could be unimplemented if the position of a particular channel is not yet a stable
        // detail. Also, we might introduce 'virtual' channels such as `Luminance` on an RGB image
        // where such channels are computed by linear combination instead of a binary incidence
        // vector. Then there might be colors where this does not exist.
        let channel = ChannelPosition::new(channel).ok_or(CommandError::OTHER)?;

        let op = Op::Unary {
            src,
            op: UnaryOp::Extract { channel },
            desc: Descriptor {
                color,
                layout,
                texel,
            },
        };

        Ok(self.push(op))
    }

    /// Reinterpret the bytes of an image as another type.
    ///
    /// This command requires that the texel type of the register and the descriptor have the same
    /// size. It will return an error if this is not the case. Additionally, the provided texel
    /// must be internally consistent.
    ///
    /// One important use of this method is to add or removed the color interpretation of an image.
    /// This can be necessary when it has been algorithmically created or when one wants to
    /// intentionally ignore such meaning.
    pub fn transmute(&mut self, src: Register, into: Descriptor) -> Result<Register, CommandError> {
        let desc = self.describe_reg(src)?;

        let supposed_type = into;

        if desc.layout != supposed_type.layout {
            return Err(CommandError {
                inner: CommandErrorKind::BadDescriptor(supposed_type),
            });
        }

        if desc.texel.bits.bytes() != supposed_type.texel.bits.bytes() {
            return Err(CommandError {
                inner: CommandErrorKind::ConflictingTypes(desc.clone(), supposed_type),
            });
        }

        if !supposed_type.is_consistent() {
            return Err(CommandError {
                inner: CommandErrorKind::BadDescriptor(supposed_type),
            });
        }

        let op = Op::Unary {
            src,
            op: UnaryOp::Transmute,
            desc: supposed_type,
        };

        Ok(self.push(op))
    }

    /// Overwrite some channels with overlaid data.
    ///
    /// This performs an implicit conversion of the overlaid data to the color channels which is
    /// performed as if by transmutation. However, contrary to the transmutation we will _only_
    /// allow the sample parts to be changed arbitrarily.
    pub fn inject(
        &mut self,
        below: Register,
        channel: ColorChannel,
        above: Register,
    ) -> Result<Register, CommandError> {
        let desc_below = self.describe_reg(below)?;
        let expected_texel = desc_below
            .texel
            .channel_texel(channel)
            .ok_or(CommandError::OTHER)?;
        let mut desc_above = self.describe_reg(above)?.clone();

        if desc_above.texel.parts.num_components() != expected_texel.parts.num_components() {
            let wanted = Descriptor {
                texel: expected_texel,
                ..desc_below.clone()
            };

            return Err(CommandError {
                inner: CommandErrorKind::ConflictingTypes(wanted, desc_above),
            });
        }

        // Override the sample part interpretation.
        let from_channels = desc_above.texel.clone();
        desc_above.texel.parts = expected_texel.parts;

        // FIXME: should we do parsing instead of validation?
        // Some type like ChannelPosition but for multiple.
        if from_channels.channel_weight_vec4().is_none() {
            return Err(CommandError::OTHER);
        }

        if expected_texel != desc_above.texel {
            let wanted = Descriptor {
                texel: expected_texel,
                ..desc_below.clone()
            };

            return Err(CommandError {
                inner: CommandErrorKind::ConflictingTypes(wanted, desc_above),
            });
        }

        // Find where to insert, see `extract` for this step.
        let channel = ChannelPosition::new(channel).ok_or(CommandError::OTHER)?;

        let op = Op::Binary {
            lhs: below,
            rhs: above,
            op: BinaryOp::Inject {
                channel,
                from_channels,
            },
            desc: desc_below.clone(),
        };

        Ok(self.push(op))
    }

    /// Grab colors from a palette based on an underlying image of indices.
    pub fn palette(
        &mut self,
        palette: Register,
        config: Palette,
        indices: Register,
    ) -> Result<Register, CommandError> {
        let desc = self.describe_reg(palette)?;
        let idx_desc = self.describe_reg(indices)?;

        // FIXME: check that channels are actually in indices' color type.
        let x_coord = if let Some(coord) = config.width {
            let pos = ChannelPosition::new(coord).ok_or(CommandError::TYPE_ERR)?;
            pos.into_vec4()
        } else {
            [0.0; 4]
        };

        let y_coord = if let Some(coord) = config.height {
            let pos = ChannelPosition::new(coord).ok_or(CommandError::TYPE_ERR)?;
            pos.into_vec4()
        } else {
            [0.0; 4]
        };

        // Compute the target layout (and that we can represent it).
        let target_layout = Descriptor::with_texel(
            desc.texel.clone(),
            idx_desc.layout.width,
            idx_desc.layout.height,
        )
        .ok_or(CommandError::OTHER)?;

        let op = Op::Binary {
            lhs: palette,
            rhs: indices,
            op: BinaryOp::Palette(shaders::PaletteShader {
                x_coord,
                y_coord,
                base_x: config.width_base,
                base_y: config.height_base,
            }),
            desc: Descriptor {
                layout: target_layout.layout,
                ..desc.clone()
            },
        };

        Ok(self.push(op))
    }

    /// Calculate the derivative of an image.
    ///
    /// Currently, will only calculate the derivative for color channels. The alpha channel will be
    /// copied from the source pixel. To also calculate a derivative over the alpha channel you
    /// should extract it as a value channel, calculate the derivative there and the inject the
    /// result back to the image.
    pub fn derivative(
        &mut self,
        image: Register,
        config: Derivative,
    ) -> Result<Register, CommandError> {
        let desc = self.describe_reg(image)?.clone();

        let op = Op::Unary {
            src: image,
            op: UnaryOp::Derivative(config),
            desc,
        };

        Ok(self.push(op))
    }

    /// Overlay this image as part of a larger one, performing blending.
    pub fn blend(
        &mut self,
        _below: Register,
        _rect: Rectangle,
        _above: Register,
        _blend: Blend,
    ) -> Result<Register, CommandError> {
        // TODO: What blending should we support
        Err(CommandError::UNIMPLEMENTED)
    }

    /// A solid color image, from a descriptor and a single texel.
    pub fn solid(&mut self, describe: Descriptor, data: &[u8]) -> Result<Register, CommandError> {
        if !describe.is_consistent() {
            return Err(CommandError {
                inner: CommandErrorKind::BadDescriptor(describe),
            });
        }

        if data.len() != usize::from(describe.layout.texel_stride) {
            return Err(CommandError {
                inner: CommandErrorKind::BadDescriptor(describe),
            });
        }

        Ok(self.push(Op::Construct {
            desc: describe,
            op: ConstructOp::Solid(data.to_owned()),
        }))
    }

    /// A 2d image with a normal distribution.
    ///
    /// The parameters are controlled through the `distribution` parameter while the `texel`
    /// parameter controls the eventual binary encoding of the image. It must be compatible with a
    /// single gray channel (but you can have electrical transfer functions, choose arbitrary bit
    /// widths etc.).
    pub fn distribution_normal2d(
        &mut self,
        describe: Descriptor,
        distribution: shaders::DistributionNormal2d,
    ) -> Result<Register, CommandError> {
        if !describe.is_consistent() {
            return Err(CommandError {
                inner: CommandErrorKind::BadDescriptor(describe),
            });
        }

        if describe.texel.parts != SampleParts::Luma && describe.texel.parts != SampleParts::LumaA {
            return Err(CommandError {
                inner: CommandErrorKind::BadDescriptor(describe),
            });
        }

        Ok(self.push(Op::Construct {
            desc: describe,
            op: ConstructOp::DistributionNormal(distribution),
        }))
    }

    /// A 2d image with fractal brownian noise.
    ///
    /// The parameters are controlled through the `distribution` parameter. Output contains
    /// in each of the 4 color channels uncorrelated, 1 dimensional fractal perlin noise.
    pub fn distribution_fractal_noise(
        &mut self,
        describe: Descriptor,
        distribution: shaders::FractalNoise,
    ) -> Result<Register, CommandError> {
        if !describe.is_consistent() {
            return Err(CommandError {
                inner: CommandErrorKind::BadDescriptor(describe),
            });
        }

        Ok(self.push(Op::Construct {
            desc: describe,
            op: ConstructOp::DistributionNoise(distribution),
        }))
    }

    /// Evaluate a bilinear function over a 2d image.
    ///
    /// For each color channel, the parameter contains intervals of values that define how its
    /// value is determined along the width and height axis.
    ///
    /// This can be used similar to `numpy`'s `mgrid`.
    pub fn bilinear(
        &mut self,
        describe: Descriptor,
        distribution: Bilinear,
    ) -> Result<Register, CommandError> {
        if !describe.is_consistent() {
            return Err(CommandError {
                inner: CommandErrorKind::BadDescriptor(describe),
            });
        }

        Ok(self.push(Op::Construct {
            desc: describe,
            op: ConstructOp::Bilinear(distribution),
        }))
    }

    /// Overlay an affine transformation of the image.
    pub fn affine(
        &mut self,
        below: Register,
        affine: Affine,
        above: Register,
    ) -> Result<Register, CommandError> {
        // TODO: should we check affine here?
        let lhs = self.describe_reg(below)?.clone();
        let rhs = self.describe_reg(above)?.clone();

        if lhs.texel != rhs.texel {
            return Err(CommandError::TYPE_ERR);
        }

        match RowMatrix::new(affine.transformation)
            .det()
            .abs()
            .partial_cmp(&f32::EPSILON)
        {
            Some(Ordering::Greater | Ordering::Equal) => {}
            _ => return Err(CommandError::OTHER),
        }

        match affine.sampling {
            AffineSample::Nearest => (),
            AffineSample::BiLinear => {
                // "Check for a color which we can sample bi-linearly"
                return Err(CommandError::UNIMPLEMENTED);
            }
        }

        Ok(self.push(Op::Binary {
            lhs: below,
            rhs: above,
            op: BinaryOp::Affine(affine),
            desc: lhs,
        }))
    }

    pub fn resize(&mut self, below: Register, upper: (u32, u32)) -> Result<Register, CommandError> {
        let (width, height) = upper;
        let grid_layout = Descriptor::with_texel(Texel::new_u8(SampleParts::RgbA), width, height)
            .ok_or(CommandError::OTHER)?;

        let grid = self.bilinear(
            grid_layout,
            Bilinear {
                u_min: [0.0, 0.0, 0.0, 1.0],
                v_min: [0.0, 0.0, 0.0, 1.0],
                uv_min: [0.0, 0.0, 0.0, 1.0],
                u_max: [1.0, 0.0, 0.0, 1.0],
                v_max: [0.0, 1.0, 0.0, 1.0],
                uv_max: [0.0, 0.0, 0.0, 1.0],
            },
        )?;

        self.palette(
            below,
            Palette {
                width: Some(ColorChannel::R),
                height: Some(ColorChannel::G),
                width_base: 0,
                height_base: 0,
            },
            grid,
        )
    }

    /// Declare an output.
    ///
    /// Outputs MUST later be bound from the pool during launch.
    pub fn output(&mut self, src: Register) -> Result<(Register, Descriptor), CommandError> {
        let outformat = self.describe_reg(src)?.clone();
        // Ignore this, it doesn't really produce a register.
        let register = self.push(Op::Output { src });
        Ok((register, outformat))
    }

    /// Declare a render target.
    ///
    /// Render targets MUST later be bound from the pool during launch, similar to outputs.
    /// However, they are not assumed to be readable afterwards and will never be a copy target.
    ///
    /// The target register must be renderable, i.e. a color with a native texture representation.
    pub fn render(&mut self, src: Register) -> Result<(Register, Descriptor), CommandError> {
        let outformat = self.describe_reg(src)?.clone();

        // FIXME: this is too conservative! We need to ensure that our internal assumption about
        // the texture descriptor is compatible with available wgpu formats (and yields the same
        // result).
        if ImageDescriptor::new(&outformat).is_err() {
            return Err(CommandError::TYPE_ERR);
        }

        // Ignore this, it doesn't really produce a register.
        let register = self.push(Op::Render { src });
        Ok((register, outformat))
    }

    pub fn compile(&self) -> Result<Program, CompileError> {
        let steps = self.ops.len();

        let mut last_use = vec![0; steps];
        let mut first_use = vec![steps; steps];

        let mut high_ops = vec![];

        // Liveness analysis.
        for (back_idx, op) in self.ops.iter().rev().enumerate() {
            let idx = self.ops.len() - 1 - back_idx;
            match op {
                Op::Input { .. }
                | Op::Construct { .. }
                | Op::Dynamic {
                    call: OperandKind::Construct,
                    ..
                } => {}
                &Op::Output { src: Register(src) } => {
                    last_use[src] = last_use[src].max(idx);
                    first_use[src] = first_use[src].min(idx);
                }
                &Op::Render { src: Register(src) } => {
                    last_use[src] = last_use[src].max(idx);
                    first_use[src] = first_use[src].min(idx);
                }
                &Op::Unary {
                    src: Register(src), ..
                }
                | &Op::Dynamic {
                    call: OperandKind::Unary(Register(src)),
                    ..
                } => {
                    last_use[src] = last_use[src].max(idx);
                    first_use[src] = first_use[src].min(idx);
                }
                &Op::Binary {
                    lhs: Register(lhs),
                    rhs: Register(rhs),
                    ..
                }
                | &Op::Dynamic {
                    call:
                        OperandKind::Binary {
                            lhs: Register(lhs),
                            rhs: Register(rhs),
                        },
                    ..
                } => {
                    last_use[rhs] = last_use[rhs].max(idx);
                    first_use[rhs] = first_use[rhs].min(idx);
                    last_use[lhs] = last_use[lhs].max(idx);
                    first_use[lhs] = first_use[lhs].min(idx);
                }
            }
        }

        let mut image_buffers = ImageBufferPlan::default();
        let mut reg_to_texture: HashMap<Register, Texture> = HashMap::default();

        for (idx, op) in self.ops.iter().enumerate() {
            let liveness = first_use[idx]..last_use[idx];
            let descriptor = self
                .describe_reg(if let Op::Output { src } = op {
                    *src
                } else if let Op::Render { src } = op {
                    *src
                } else {
                    Register(idx)
                })
                .expect("A non-output register");

            let ImageBufferAssignment { buffer: _, texture } =
                image_buffers.allocate_for(descriptor, liveness);

            high_ops.push(High::StackPush(Frame {
                name: format!("Command: {:#?}", op),
            }));

            match op {
                Op::Input { desc } => {
                    high_ops.push(High::Input(Register(idx), desc.clone()));
                    reg_to_texture.insert(Register(idx), texture);
                }
                &Op::Output { src } => {
                    high_ops.push(High::Output {
                        src,
                        dst: Register(idx),
                    });
                }
                &Op::Render { src } => {
                    high_ops.push(High::Render {
                        src,
                        dst: Register(idx),
                    });
                }
                Op::Construct { desc: _, op } => {
                    match op {
                        &ConstructOp::DistributionNormal(ref distribution) => {
                            high_ops.push(High::Construct {
                                dst: Target::Discard(texture),
                                fn_: Function::PaintFullScreen {
                                    shader: FragmentShader::Normal2d(distribution.clone()),
                                },
                            })
                        }
                        ConstructOp::DistributionNoise(ref noise_params) => {
                            high_ops.push(High::Construct {
                                dst: Target::Discard(texture),
                                fn_: Function::PaintFullScreen {
                                    shader: FragmentShader::FractalNoise(noise_params.clone()),
                                },
                            })
                        }
                        ConstructOp::Bilinear(bilinear) => high_ops.push(High::Construct {
                            dst: Target::Discard(texture),
                            fn_: Function::PaintFullScreen {
                                shader: FragmentShader::Bilinear(bilinear.clone()),
                            },
                        }),
                        _ => return Err(CompileError::NotYetImplemented),
                    }

                    reg_to_texture.insert(Register(idx), texture);
                }
                Op::Unary { desc: _, src, op } => {
                    match op {
                        &UnaryOp::Crop(region) => {
                            let target =
                                Rectangle::with_width_height(region.width(), region.height());
                            high_ops.push(High::PushOperand(reg_to_texture[src]));
                            high_ops.push(High::Construct {
                                dst: Target::Discard(texture),
                                fn_: Function::PaintToSelection {
                                    texture: reg_to_texture[src],
                                    selection: region,
                                    target: target.into(),
                                    viewport: target,
                                    shader: FragmentShader::PaintOnTop(PaintOnTopKind::Copy),
                                },
                            });
                        }
                        UnaryOp::ChromaticAdaptation(adaptation) => {
                            // Determine matrix for converting to xyz, then adapt, then back.
                            let adapt = RowMatrix::new(adaptation.to_matrix()?);
                            let output = adapt.multiply_right(adaptation.to_xyz_matrix.into());
                            let matrix = adaptation.from_xyz_matrix.multiply_right(output);

                            // If you want to debug this (for comparison to reference):
                            // eprintln!("{:?}", adaptation.to_xyz_matrix);
                            // eprintln!("{:?}", adaptation.from_xyz_matrix);
                            // eprintln!("{:?}", adapt);
                            // eprintln!("{:?}", matrix);

                            high_ops.push(High::PushOperand(reg_to_texture[src]));
                            high_ops.push(High::Construct {
                                dst: Target::Discard(texture),
                                fn_: Function::PaintFullScreen {
                                    shader: FragmentShader::LinearColorMatrix(
                                        shaders::LinearColorTransform {
                                            matrix: matrix.into(),
                                        },
                                    ),
                                },
                            });
                        }
                        UnaryOp::ColorConvert(color) => {
                            // The inherent OptoToLinear transformation gets us to a linear light
                            // representation. We want to convert this into a compatible (that is,
                            // using the same observer definition) other linear light
                            // representation that we then transfer back to an electrical form.
                            // Note that these two steps happen, conveniently, automatically.
                            // Usually it is ensured that only two images with the same linear
                            // light representation are used in a single paint call but this
                            // violates it on purpose.

                            high_ops.push(High::PushOperand(reg_to_texture[src]));
                            // FIXME: using a copy here but this means we do this in unnecessarily
                            // many steps. We first decode to linear color, then draw, then code
                            // back to the non-linear electrical space.
                            // We could do this directly from one matrix to another or try using an
                            // ephemeral intermediate attachment?
                            high_ops.push(High::Construct {
                                dst: Target::Discard(texture),
                                fn_: Function::PaintFullScreen {
                                    shader: color.to_shader(),
                                },
                            });
                        }
                        UnaryOp::Extract { channel: _ } => {
                            high_ops.push(High::PushOperand(reg_to_texture[src]));

                            high_ops.push(High::Construct {
                                dst: Target::Discard(texture),
                                fn_: Function::PaintFullScreen {
                                    // This will grab the right channel, that is all of them.
                                    // The actual conversion is done in de-staging of the result.
                                    // TODO: evaluate if this is the right way to do it. We could
                                    // also perform a LinearColorMatrix shader here with close to
                                    // the same amount of shader code but a precise result.
                                    shader: FragmentShader::PaintOnTop(PaintOnTopKind::Copy),
                                },
                            })
                        }
                        UnaryOp::Derivative(derivative) => {
                            let shader = derivative.method.to_shader(derivative.direction)?;

                            high_ops.push(High::PushOperand(reg_to_texture[src]));
                            high_ops.push(High::Construct {
                                dst: Target::Discard(texture),
                                fn_: Function::PaintFullScreen { shader },
                            })
                        }
                        UnaryOp::Transmute => high_ops.push(High::Copy {
                            src: *src,
                            dst: Register(idx),
                        }),
                    }

                    reg_to_texture.insert(Register(idx), texture);
                }
                Op::Binary {
                    desc: _,
                    lhs,
                    rhs,
                    op,
                } => {
                    let lower_region = Rectangle::from(self.describe_reg(*lhs).unwrap());
                    let upper_region = Rectangle::from(self.describe_reg(*rhs).unwrap());

                    match op {
                        BinaryOp::Affine(affine) => {
                            let affine_matrix = RowMatrix::new(affine.transformation);

                            high_ops.push(High::PushOperand(reg_to_texture[lhs]));
                            high_ops.push(High::Construct {
                                dst: Target::Discard(texture),
                                fn_: Function::PaintToSelection {
                                    texture: reg_to_texture[lhs],
                                    selection: lower_region,
                                    target: lower_region.into(),
                                    viewport: lower_region,
                                    shader: FragmentShader::PaintOnTop(PaintOnTopKind::Copy),
                                },
                            });

                            high_ops.push(High::PushOperand(reg_to_texture[rhs]));
                            high_ops.push(High::Construct {
                                dst: Target::Load(texture),
                                fn_: Function::PaintToSelection {
                                    texture: reg_to_texture[rhs],
                                    selection: upper_region,
                                    target: QuadTarget::from(upper_region).affine(&affine_matrix),
                                    viewport: lower_region,
                                    shader: FragmentShader::PaintOnTop(
                                        affine.sampling.as_paint_on_top()?,
                                    ),
                                },
                            })
                        }
                        BinaryOp::Inject {
                            channel,
                            from_channels,
                        } => {
                            high_ops.push(High::PushOperand(reg_to_texture[lhs]));
                            high_ops.push(High::PushOperand(reg_to_texture[rhs]));

                            high_ops.push(High::Construct {
                                dst: Target::Discard(texture),
                                fn_: Function::PaintFullScreen {
                                    shader: FragmentShader::Inject(shaders::inject::Shader {
                                        mix: channel.into_vec4(),
                                        color: from_channels.channel_weight_vec4().unwrap(),
                                    }),
                                },
                            })
                        }
                        BinaryOp::Inscribe { placement } => {
                            high_ops.push(High::PushOperand(reg_to_texture[lhs]));
                            high_ops.push(High::Construct {
                                dst: Target::Discard(texture),
                                fn_: Function::PaintToSelection {
                                    texture: reg_to_texture[lhs],
                                    selection: lower_region,
                                    target: lower_region.into(),
                                    viewport: lower_region,
                                    shader: FragmentShader::PaintOnTop(PaintOnTopKind::Copy),
                                },
                            });

                            high_ops.push(High::PushOperand(reg_to_texture[rhs]));
                            high_ops.push(High::Construct {
                                dst: Target::Load(texture),
                                fn_: Function::PaintToSelection {
                                    texture: reg_to_texture[rhs],
                                    selection: upper_region,
                                    target: (*placement).into(),
                                    viewport: lower_region,
                                    shader: FragmentShader::PaintOnTop(PaintOnTopKind::Copy),
                                },
                            });
                        }
                        BinaryOp::Palette(shader) => {
                            high_ops.push(High::PushOperand(reg_to_texture[lhs]));
                            high_ops.push(High::PushOperand(reg_to_texture[rhs]));

                            high_ops.push(High::Construct {
                                dst: Target::Load(texture),
                                fn_: Function::PaintFullScreen {
                                    shader: FragmentShader::Palette(shader.clone()),
                                },
                            });
                        }
                    }

                    reg_to_texture.insert(Register(idx), texture);
                }
                Op::Dynamic { call, command, .. } => {
                    let (op_unary, op_binary, arguments);

                    match call {
                        OperandKind::Construct => {
                            arguments = &[][..];
                            reg_to_texture.insert(Register(idx), texture);
                        }
                        OperandKind::Unary(reg) => {
                            op_unary = [reg_to_texture[reg]];
                            arguments = &op_unary[..];
                            reg_to_texture.insert(Register(idx), texture);
                        }
                        OperandKind::Binary { lhs, rhs } => {
                            op_binary = [reg_to_texture[lhs], reg_to_texture[rhs]];
                            arguments = &op_binary[..];
                            reg_to_texture.insert(Register(idx), texture);
                        }
                    }

                    for &operand in arguments {
                        high_ops.push(High::PushOperand(operand));
                    }

                    high_ops.push(High::Construct {
                        dst: Target::Discard(texture),
                        fn_: Function::PaintFullScreen {
                            shader: FragmentShader::Dynamic(command.clone()),
                        },
                    })
                }
                // In case we add a new case.
                #[allow(unreachable_patterns)]
                _ => {
                    return Err(CompileError::NotYetImplemented);
                }
            }

            high_ops.push(High::Done(Register(idx)));
            high_ops.push(High::StackPop);
        }

        Ok(Program {
            ops: high_ops,
            image_buffers,
            buffer_by_op: HashMap::default(),
            texture_by_op: HashMap::default(),
        })
    }

    /// Get the descriptor for a register.
    fn describe_reg(&self, Register(reg): Register) -> Result<&Descriptor, CommandError> {
        match self.ops.get(reg) {
            None | Some(Op::Output { .. }) | Some(Op::Render{ .. }) => Err(CommandError::BAD_REGISTER),
            Some(Op::Input { desc })
            | Some(Op::Construct { desc, .. })
            | Some(Op::Unary { desc, .. })
            | Some(Op::Binary { desc, .. })
            | Some(Op::Dynamic { desc, .. }) => Ok(desc),
        }
    }
    fn push(&mut self, op: Op) -> Register {
        let reg = Register(self.ops.len());
        self.ops.push(op);
        reg
    }
}

/// Impls on `CommandBuffer` that allow defining custom SPIR-V extensions.
///
/// Generally, the steps on the dynamic shader are:
///
/// 1. Check the kind, get SPIR-v code.
/// 2. Determine the dynamic typing of the result.
/// 3. Have the shader create binary representation of its data.
/// 3. Create a new entry on the command buffer.
/// 4. Not yet performed: (Validate the SPIR-V module inputs against the data definition)
impl CommandBuffer {
    /// Record a _constructor_.
    pub fn construct_dynamic(&mut self, dynamic: &dyn ShaderCommand) -> Register {
        let mut data = vec![];
        let mut content = None;

        let source = dynamic.source();
        let desc = dynamic.data(ShaderData {
            data_buffer: &mut data,
            content: &mut content,
        });

        self.push(Op::Dynamic {
            call: OperandKind::Construct,
            // FIXME: maybe this conversion should be delayed.
            // In particular, converting source to SPIR-V may take some form of 'compiler' argument
            // that's only available during `compile` phase.
            command: ShaderInvocation {
                spirv: match source {
                    ShaderSource::SpirV(spirv) => spirv,
                },
                shader_data: match content {
                    None => None,
                    Some(c) => Some(c.as_slice(&data).into()),
                },
                num_args: 0,
            },
            desc,
        })
    }

    /// Record a _unary operator_.
    pub fn unary_dynamic(&mut self, lhs: Register, dynamic: &dyn ShaderCommand) -> Register {
        todo!()
    }

    /// Record a _binary operator_.
    pub fn binary_dynamic(
        &mut self,
        lhs: Register,
        rhs: Register,
        dynamic: &dyn ShaderCommand,
    ) -> Register {
        todo!()
    }
}

impl ColorConversion {
    pub(crate) fn to_shader(&self) -> FragmentShader {
        match self {
            ColorConversion::Xyz {
                to_xyz_matrix,
                from_xyz_matrix,
            } => {
                let from = from_xyz_matrix.inv();
                let matrix = to_xyz_matrix.multiply_right(from.into()).into();

                FragmentShader::LinearColorMatrix(shaders::LinearColorTransform { matrix })
            }
            ColorConversion::XyzToOklab { to_xyz_matrix } => {
                FragmentShader::Oklab(shaders::oklab::Shader::with_encode(*to_xyz_matrix))
            }
            ColorConversion::OklabToXyz { from_xyz_matrix } => {
                let from_xyz_matrix = from_xyz_matrix.inv();
                FragmentShader::Oklab(shaders::oklab::Shader::with_decode(from_xyz_matrix))
            }
            ColorConversion::XyzToSrLab2 {
                to_xyz_matrix,
                whitepoint,
            } => FragmentShader::SrLab2(shaders::srlab2::Shader::with_encode(
                *to_xyz_matrix,
                *whitepoint,
            )),
            ColorConversion::SrLab2ToXyz {
                from_xyz_matrix,
                whitepoint,
            } => {
                let from_xyz_matrix = from_xyz_matrix.inv();
                FragmentShader::SrLab2(shaders::srlab2::Shader::with_decode(
                    from_xyz_matrix,
                    *whitepoint,
                ))
            }
        }
    }
}

impl ChromaticAdaptation {
    pub(crate) fn to_matrix(&self) -> Result<[f32; 9], CompileError> {
        use palette::{
            chromatic_adaptation::{Method, TransformMatrix},
            white_point as wp,
        };

        // FIXME: when you adjust the value-to-type translation, also adjust it within `method`.
        macro_rules! translate_matrix {
            ($source:expr, $target:expr, $($lhs:ident => $lhsty:ty)|*) => {
                $(
                    translate_matrix!(
                        @$source, $target, $lhs => $lhsty :
                        A => wp::A | B => wp::B | C => wp::C
                        | D50 => wp::D50 | D55 => wp::D55 | D65 => wp::D65
                        | D75 => wp::D75 | E => wp::E | F2 => wp::F2
                        | F7 => wp::F7 | F11 => wp::F11
                    );
                )*
            };
            (@$source:expr, $target:expr, $lhs:ident => $lhsty:ty : $($rhs:ident => $ty:ty)|*) => {
                $(
                    if let (Whitepoint::$lhs, Whitepoint::$rhs) = ($source, $target) {
                        return Ok(<Method as TransformMatrix<$lhsty, $ty, f32>>::generate_transform_matrix
                                  as fn(&Method) -> [f32;9]);
                    }
                )*
            };
        }

        // FIXME: when you adjust the value-to-type translation, also adjust it within
        // `translate_matrix!`
        let method = (|| {
            translate_matrix! {
                self.source, self.target,
                A => wp::A | B => wp::B | C => wp::C
                | D50 => wp::D50 | D55 => wp::D55 | D65 => wp::D65
                | D75 => wp::D75 | E => wp::E | F2 => wp::F2
                | F7 => wp::F7 | F11 => wp::F11
            };

            Err(CompileError::NotYetImplemented)
        })()?;

        let matrices = method(match self.method {
            // Bradford's original method that does slight blue non-linearity is not yet supported.
            // Please implement the paper if you feel compelled to.
            ChromaticAdaptationMethod::BradfordNonLinear => {
                return Err(CompileError::NotYetImplemented)
            }
            ChromaticAdaptationMethod::BradfordVonKries => &Method::Bradford,
            ChromaticAdaptationMethod::VonKries => &Method::VonKries,
            ChromaticAdaptationMethod::Xyz => &Method::XyzScaling,
        });

        Ok(matrices)
    }
}

#[rustfmt::skip]
impl DerivativeMethod {
    fn to_shader(&self, direction: Direction) -> Result<FragmentShader, CompileError> {
        use DerivativeMethod::*;
        use shaders::box3;
        match self {
            Prewitt => {
                let matrix = RowMatrix::with_outer_product(
                    [1./3., 1./3., 1./3.],
                    [0.5, 0.0, -0.5],
                );

                let shader = box3::Shader::new(direction.adjust_vertical_box(matrix));
                Ok(shaders::FragmentShader::Box3(shader))
            }
            Sobel => {
                let matrix = RowMatrix::with_outer_product(
                    [1./4., 1./2., 1./4.],
                    [0.5, 0.0, -0.5],
                );

                let shader = box3::Shader::new(direction.adjust_vertical_box(matrix));
                Ok(shaders::FragmentShader::Box3(shader))
            }
            Scharr3 => {
                let matrix = RowMatrix::with_outer_product(
                    [46.84/256., 162.32/256., 46.84/256.],
                    [0.5, 0.0, -0.5],
                );

                let shader = box3::Shader::new(direction.adjust_vertical_box(matrix));
                Ok(shaders::FragmentShader::Box3(shader))
            }
            Scharr3To4Bit => {
                let matrix = RowMatrix::with_outer_product(
                    [3./16., 10./16., 3./16.],
                    [0.5, 0.0, -0.5],
                );

                let shader = box3::Shader::new(direction.adjust_vertical_box(matrix));
                Ok(shaders::FragmentShader::Box3(shader))
            }
            Scharr3To8Bit => {
                let matrix = RowMatrix::with_outer_product(
                    [47./256., 162./256., 47./256.],
                    [0.5, 0.0, -0.5],
                );

                let shader = box3::Shader::new(direction.adjust_vertical_box(matrix));
                Ok(shaders::FragmentShader::Box3(shader))
            }
            // FIXME: implement these.
            // When you do add them to tests/blend.rs
            | Roberts
            | Scharr4
            | Scharr5
            | Scharr5Tab => Err(CompileError::NotYetImplemented)
        }
    }
}

impl Direction {
    fn adjust_vertical_box(self, mat: RowMatrix) -> RowMatrix {
        match self {
            Direction::Width => mat,
            Direction::Height => mat.transpose(),
        }
    }
}

#[rustfmt::skip]
impl Affine {
    /// Create affine parameters with identity transformation.
    pub fn new(sampling: AffineSample) -> Self {
        Affine {
            transformation: [
                1.0, 0., 0.,
                0., 1.0, 0.,
                0., 0., 1.0,
            ],
            sampling,
        }
    }

    /// After the transformation, also scale everything.
    ///
    /// This corresponds to a left-side multiplication of the transformation matrix.
    pub fn scale(self, x: f32, y: f32) -> Self {
        let post = RowMatrix::diag(x, y, 1.0)
            .multiply_right(RowMatrix::new(self.transformation).into());
        let transformation = RowMatrix::from(post).into_inner();

        Affine {
            transformation,
            ..self
        }
    }

    /// After the transformation, rotate everything clockwise.
    ///
    /// This corresponds to a left-side multiplication of the transformation matrix.
    pub fn rotate(self, rad: f32) -> Self {
        let post = RowMatrix::new([
            rad.cos(), rad.sin(), 0.,
            -rad.sin(), rad.cos(), 0.,
            0., 0., 1.,
        ]);

        let post = post.multiply_right(RowMatrix::new(self.transformation).into());
        let transformation = RowMatrix::from(post).into_inner();

        Affine {
            transformation,
            ..self
        }
    }

    /// After the transformation, shift by an x and y offset.
    ///
    /// This corresponds to a left-side multiplication of the transformation matrix.
    pub fn shift(self, x: f32, y: f32) -> Self {
        let post = RowMatrix::new([
            1., 0., x,
            0., 1., y,
            0., 0., 1.,
        ]);

        let post = post.multiply_right(RowMatrix::new(self.transformation).into());
        let transformation = RowMatrix::from(post).into_inner();

        Affine {
            transformation,
            ..self
        }
    }
}

impl AffineSample {
    fn as_paint_on_top(self) -> Result<PaintOnTopKind, CompileError> {
        match self {
            AffineSample::Nearest => Ok(PaintOnTopKind::Copy),
            _ => Err(CompileError::NotYetImplemented),
        }
    }
}

impl Rectangle {
    /// A rectangle at the origin with given width (x) and height (y).
    pub fn with_width_height(width: u32, height: u32) -> Self {
        Rectangle {
            x: 0,
            y: 0,
            max_x: width,
            max_y: height,
        }
    }

    /// A rectangle describing a complete buffer.
    pub fn with_layout(buffer: &ByteLayout) -> Self {
        Self::with_width_height(buffer.width, buffer.height)
    }

    /// The apparent width.
    pub fn width(self) -> u32 {
        self.max_x.saturating_sub(self.x)
    }

    /// The apparent height.
    pub fn height(self) -> u32 {
        self.max_y.saturating_sub(self.y)
    }

    /// Return true if this rectangle fully contains `other`.
    pub fn contains(self, other: Self) -> bool {
        self.x <= other.x && self.y <= other.y && {
            // Offsets are surely non-wrapping.
            let offset_x = other.x - self.x;
            let offset_y = other.y - self.y;
            let rel_width = self.width().checked_sub(offset_x);
            let rel_height = self.height().checked_sub(offset_y);
            rel_width >= Some(other.width()) && rel_height >= Some(other.height())
        }
    }

    /// Bring the rectangle into normalized form where minimum and maximum form a true interval.
    #[must_use]
    pub fn normalize(self) -> Rectangle {
        Rectangle {
            x: self.x,
            y: self.y,
            max_x: self.x + self.width(),
            max_y: self.y + self.width(),
        }
    }

    /// A rectangle that the overlap of the two.
    #[must_use]
    pub fn meet(self, other: Self) -> Rectangle {
        Rectangle {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
            max_x: self.max_x.min(other.max_x),
            max_y: self.max_y.min(other.max_y),
        }
    }

    /// The meet, relative to the coordinates of this rectangle.
    #[must_use]
    pub fn meet_in_local_coordinates(self, other: Self) -> Rectangle {
        // Normalize to ensure that max_{x,y} is not less than {x,y}
        let meet = self.normalize().meet(other);
        Rectangle {
            x: meet.x - self.x,
            y: meet.y - self.y,
            max_x: meet.max_x - self.x,
            max_y: meet.max_y - self.y,
        }
    }

    /// A rectangle that contains both.
    #[must_use]
    pub fn join(self, other: Self) -> Rectangle {
        Rectangle {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
            max_x: self.max_x.max(other.max_x),
            max_y: self.max_y.max(other.max_y),
        }
    }

    /// Remove border from all sides.
    /// When the image is smaller than `border` in some dimension then the result is empty and
    /// contained in the original image but otherwise unspecified.
    #[must_use]
    pub fn inset(self, border: u32) -> Self {
        Rectangle {
            x: self.x.saturating_add(border),
            y: self.y.saturating_add(border),
            max_x: self.max_x.saturating_sub(border),
            max_y: self.max_y.saturating_sub(border),
        }
    }
}

impl From<&'_ ByteLayout> for Rectangle {
    fn from(buffer: &ByteLayout) -> Rectangle {
        Rectangle::with_width_height(buffer.width, buffer.height)
    }
}

impl From<&'_ BufferLayout> for Rectangle {
    fn from(buffer: &BufferLayout) -> Rectangle {
        Rectangle::with_width_height(buffer.width(), buffer.height())
    }
}

impl From<&'_ Descriptor> for Rectangle {
    fn from(buffer: &Descriptor) -> Rectangle {
        Rectangle::from(&buffer.layout)
    }
}

impl CommandError {
    /// Indicates a very generic type error.
    const TYPE_ERR: Self = CommandError {
        inner: CommandErrorKind::GenericTypeError,
    };

    /// Indicates a very generic other error.
    /// E.g. the usage of a command requires an extension? Not quite sure yet.
    const OTHER: Self = CommandError {
        inner: CommandErrorKind::Other,
    };

    /// Specifies that a register reference was invalid.
    const BAD_REGISTER: Self = Self::OTHER;

    /// This has not yet been implemented, sorry.
    ///
    /// Errors of this kind will be removed over the course of bringing the crate to a first stable
    /// release, this this will be removed. The method, and importantly its signature, are already
    /// added for the purpose of exposition and documenting the intention.
    const UNIMPLEMENTED: Self = CommandError {
        inner: CommandErrorKind::Unimplemented,
    };

    pub fn is_type_err(&self) -> bool {
        matches!(
            self.inner,
            CommandErrorKind::GenericTypeError
                | CommandErrorKind::ConflictingTypes(_, _)
                | CommandErrorKind::BadDescriptor(_)
        )
    }
}

#[test]
fn rectangles() {
    let small = Rectangle::with_width_height(2, 2);
    let large = Rectangle::with_width_height(4, 4);

    assert_eq!(large, large.join(small));
    assert_eq!(small, large.meet(small));
    assert!(large.contains(small));
    assert!(!small.contains(large));
}

#[test]
fn simple_program() {
    use crate::pool::Pool;
    use image::GenericImageView;

    const BACKGROUND: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/input/background.png");
    const FOREGROUND: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/input/foreground.png");

    let mut pool = Pool::new();
    let mut commands = CommandBuffer::default();

    let background = image::open(BACKGROUND).expect("Background image opened");
    let foreground = image::open(FOREGROUND).expect("Background image opened");
    let expected = ByteLayout::from(&background);

    let placement = Rectangle {
        x: 0,
        y: 0,
        max_x: foreground.width(),
        max_y: foreground.height(),
    };

    let background = pool.insert_srgb(&background);
    let background = commands.input_from(background.into());

    let foreground = pool.insert_srgb(&foreground);
    let foreground = commands.input_from(foreground.into());

    let result = commands
        .inscribe(background, placement, foreground)
        .expect("Valid to inscribe");
    let (_, outformat) = commands.output(result).expect("Valid for output");

    let _ = commands.compile().expect("Could build command buffer");
    assert_eq!(outformat.layout, expected);
}
