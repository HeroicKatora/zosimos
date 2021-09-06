//! Defines layout and buffer of our images.
use canvas::{layout::Layout, Canvas};
use core::convert::TryFrom;

/// The byte layout of a buffer.
///
/// An inner invariant is that the layout fits in memory, and in particular into a `usize`, while
/// at the same time also fitting inside a `u64` of bytes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferLayout {
    /// The number of texels along our width.
    pub(crate) width: u32,
    /// The number of texels along our height.
    pub(crate) height: u32,
    /// The number of bytes per texel.
    /// We need to be able to infallibly convert to both `usize` and `u32`. Thus we have chosen
    /// `u8` for now because no actual texel is that large. However, we could use some other type
    /// to represent the intersection of our two target types (i.e. the `index-ext` crate has
    /// `mem::Umem32` with those exact semantics).
    pub(crate) bytes_per_texel: u8,
    /// The number of bytes per row.
    /// This is a u32 for compatibility with `wgpu`.
    pub(crate) bytes_per_row: u32,
}

/// Describe a row-major rectangular matrix layout.
///
/// This is only concerned with byte-buffer compatibility and not type or color space semantics of
/// texels. It assumes a row-major layout without space between texels of a row as that is the most
/// efficient and common such layout.
///
/// For usage as an actual image buffer, to convert it to a `BufferLayout` by calling
/// [`BufferLayout::with_row_layout`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RowLayoutDescription {
    pub width: u32,
    pub height: u32,
    pub texel_stride: u64,
    pub row_stride: u64,
}

#[derive(Clone)]
pub struct ImageBuffer {
    inner: Canvas<BufferLayout>,
}

/// Describes an image semantically.
#[derive(Clone, Debug, PartialEq)]
pub struct Descriptor {
    /// The byte and physical layout of the buffer.
    pub layout: BufferLayout,
    /// Describe how each single texel is interpreted.
    pub texel: Texel,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Texel {
    /// Which part of the image a single texel refers to.
    pub block: Block,
    /// How numbers and channels are encoded into the texel.
    pub samples: Samples,
    /// How the numbers relate to physical quantities, important for conversion.
    pub color: Color,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
#[repr(u8)]
pub enum Block {
    /// Each texel is a single pixel.
    Pixel = 0,
    /// Each texel refers to two pixels across width.
    Sub1x2 = 1,
    /// Each texel refers to four pixels across width.
    Sub1x4 = 2,
    /// Each texel refers to a two-by-two block.
    Sub2x2 = 3,
    /// Each texel refers to a two-by-four block.
    Sub2x4 = 4,
    /// Each texel refers to a four-by-four block.
    Sub4x4 = 5,
}

/// The bit encoding of values within the texel bytes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Samples {
    /// Which values are encoded, which controls the applicable color spaces.
    pub parts: SampleParts,
    /// How the values are encoded as bits in the bytes.
    pub bits: SampleBits,
}

/// Describes which values are present in a texel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
#[repr(u16)]
pub enum SampleParts {
    A = 0,
    R,
    G,
    B,
    Luma,
    LumaA,
    Rgb,
    Bgr,
    Rgba,
    Rgbx,
    Bgra,
    Bgrx,
    Argb,
    Xrgb,
    Abgr,
    Xbgr,
    Yuv,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
#[repr(u8)]
pub enum SampleBits {
    /// A single 8-bit integer.
    Int8,
    /// Three packed integer.
    Int332,
    /// Three packed integer.
    Int233,
    /// A single 16-bit integer.
    Int16,
    /// Four packed integer.
    Int4x4,
    /// Four packed integer, one component ignored.
    Inti444,
    /// Four packed integer, one component ignored.
    Int444i,
    /// Three packed integer.
    Int565,
    /// Two 8-bit integers.
    Int8x2,
    /// Three 8-bit integer.
    Int8x3,
    /// Four 8-bit integer.
    Int8x4,
    /// Two 16-bit integers.
    Int16x2,
    /// Three 16-bit integer.
    Int16x3,
    /// Four 16-bit integer.
    Int16x4,
    /// Four packed integer.
    Int1010102,
    /// Four packed integer.
    Int2101010,
    /// Three packed integer, one component ignored.
    Int101010i,
    /// Three packed integer, one component ignored.
    Inti101010,
    /// Four half-floats.
    Float16x4,
    /// Four floats.
    Float32x4,
}

/// Describes a single channel from an image.
/// Note that it must match the descriptor when used in `extract` and `inject`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorChannel {
    R,
    G,
    B,
}

/// Denotes the 'position' of a channel in the sample parts.
///
/// This is private for now because the constructor might be a bit confusing. In actuality, we are
/// interested in the position of a channel in the _linear_ color representation. For example, all
/// RGB-ish colors (including the variant `Bgra`) are mapped to a `vec4` in the order `rgba` in the
/// shader execution. Thus, the 'position' of the `R` channel is _always_ `First` in these cases.
///
/// This can only make sense with internal knowledge about how we remap color representations into
/// the texture during the Staging phase of loading a color image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum ChannelPosition {
    First = 0,
    Second = 1,
    Third = 2,
    Fourth = 3,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Color {
    /// A common model based on the CIE 1931 XYZ observer.
    Xyz {
        primary: Primaries,
        transfer: Transfer,
        whitepoint: Whitepoint,
        luminance: Luminance,
    },
}

/// Transfer functions from encoded chromatic samples to physical quantity.
///
/// Ignoring viewing environmental effects, this describes a pair of functions that are each others
/// inverse: An electro-optical transfer (EOTF) and opto-electronic transfer function (OETF) that
/// describes how scene lighting is encoded as an electric signal. These are applied to each
/// stimulus value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
#[non_exhaustive]
pub enum Transfer {
    /// Non-linear electrical data of Bt.709
    Bt709,
    Bt470M,
    /// Non-linear electrical data of Bt.601
    Bt601,
    /// Non-linear electrical data of Smpte-240
    Smpte240,
    /// Linear color in display luminance.
    Linear,
    /// Non-linear electrical data of Srgb
    Srgb,
    /// Non-linear electrical data of Bt2020 that was 10-bit quantized
    Bt2020_10bit,
    /// Non-linear electrical data of Bt2020 that was 12-bit quantized
    Bt2020_12bit,
    /// Non-linear electrical data of Smpte-2048
    Smpte2084,
    /// Another name for Smpte2084.
    Bt2100Pq,
    /// Non-linear electrical data of Bt2100 Hybrid-Log-Gamma.
    Bt2100Hlg,
    /// Linear color in scene luminance of Bt2100.
    /// This is perfect for an artistic composition pipeline. The rest of the type system will
    /// ensure this is not accidentally and unwittingly mixed with `Linear` but otherwise this is
    /// treated as `Linear`. You might always transmute.
    Bt2100Scene,
}

/// The reference brightness of the color specification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Luminance {
    /// 100cd/m².
    Sdr,
    /// 10_000cd/m².
    /// Known as high-dynamic range.
    Hdr,
    /// 160cd/m².
    AdobeRgb,
}

/// The relative stimuli of the three corners of a triangular gamut.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Primaries {
    Bt601_525,
    Bt601_625,
    Bt709,
    Smpte240,
    Bt2020,
    Bt2100,
}

/// The whitepoint/standard illuminant.
///
/// ```text
/// Illuminant 	X 	Y 	Z
/// A   	1.09850 	1.00000 	0.35585
/// B   	0.99072 	1.00000 	0.85223
/// C   	0.98074 	1.00000 	1.18232
/// D50 	0.96422 	1.00000 	0.82521
/// D55 	0.95682 	1.00000 	0.92149
/// D65 	0.95047 	1.00000 	1.08883
/// D75 	0.94972 	1.00000 	1.22638
/// E   	1.00000 	1.00000 	1.00000
/// F2  	0.99186 	1.00000 	0.67393
/// F7  	0.95041 	1.00000 	1.08747
/// F11 	1.00962 	1.00000 	0.64350
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Whitepoint {
    A,
    B,
    C,
    D50,
    D55,
    D65,
    D75,
    E,
    F2,
    F7,
    F11,
}

/// A column major matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ColMatrix([[f32; 3]; 3]);

/// A row major matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RowMatrix([f32; 9]);

impl Descriptor {
    pub const EMPTY: Self = Descriptor {
        layout: BufferLayout {
            width: 0,
            height: 0,
            bytes_per_texel: 4,
            bytes_per_row: 0,
        },
        texel: Texel {
            block: Block::Pixel,
            color: Color::SRGB,
            samples: Samples {
                bits: SampleBits::Int8x4,
                parts: SampleParts::Rgba,
            },
        },
    };

    /// Get the texel describing a single channel.
    /// Returns None if the channel is not contained, or if it can not be extracted on its own.
    pub fn channel_texel(&self, channel: ColorChannel) -> Option<Texel> {
        self.texel.channel_texel(channel)
    }

    /// Check if the descriptor is consistent.
    ///
    /// A consistent descriptor makes inherent sense. That is, the different fields contain values
    /// that are not contradictory. For example, the color channels parts and the color model
    /// correspond to each other, and the sample parts and sample bits field is correct, and the
    /// texel descriptor has the same number of bytes as the layout, etc.
    pub fn is_consistent(&self) -> bool {
        // FIXME: other checks.
        self.texel.samples.bits.bytes() == usize::from(self.layout.bytes_per_texel)
    }

    pub fn pixel_width(&self) -> u32 {
        self.layout.width * self.texel.block.width()
    }

    pub fn pixel_height(&self) -> u32 {
        self.layout.height * self.texel.block.height()
    }

    pub fn size(&self) -> (u32, u32) {
        (self.layout.width, self.layout.height)
    }
}

impl Texel {
    pub fn with_srgb_image(img: &image::DynamicImage) -> Self {
        use image::DynamicImage::*;
        let samples = match img {
            ImageLuma8(_) => Samples {
                bits: SampleBits::Int8,
                parts: SampleParts::Luma,
            },
            ImageLumaA8(_) => Samples {
                bits: SampleBits::Int8x2,
                parts: SampleParts::LumaA,
            },
            ImageLuma16(_) => Samples {
                bits: SampleBits::Int16,
                parts: SampleParts::Luma,
            },
            ImageLumaA16(_) => Samples {
                bits: SampleBits::Int16x2,
                parts: SampleParts::LumaA,
            },
            ImageRgb8(_) => Samples {
                bits: SampleBits::Int8x3,
                parts: SampleParts::Rgb,
            },
            ImageRgba8(_) => Samples {
                bits: SampleBits::Int8x4,
                parts: SampleParts::Rgba,
            },
            ImageBgr8(_) => Samples {
                bits: SampleBits::Int8x3,
                parts: SampleParts::Bgr,
            },
            ImageBgra8(_) => Samples {
                bits: SampleBits::Int8x4,
                parts: SampleParts::Bgra,
            },
            ImageRgb16(_) => Samples {
                bits: SampleBits::Int16x3,
                parts: SampleParts::Rgb,
            },
            ImageRgba16(_) => Samples {
                bits: SampleBits::Int16x4,
                parts: SampleParts::Rgba,
            },
        };

        Texel {
            block: Block::Pixel,
            color: Color::SRGB,
            samples,
        }
    }

    /// Get the texel describing a single channel.
    /// Returns None if the channel is not contained, or if it can not be extracted on its own.
    pub fn channel_texel(&self, channel: ColorChannel) -> Option<Texel> {
        use Block::*;
        use SampleBits::*;
        use SampleParts::*;
        let parts = match self.samples.parts {
            Rgb | Rgbx | Rgba | Bgrx | Bgra | Abgr | Argb | Xrgb | Xbgr => match channel {
                ColorChannel::R => R,
                ColorChannel::G => G,
                ColorChannel::B => B,
            },
            _ => return None,
        };
        let bits = match self.samples.bits {
            Int8 | Int8x3 | Int8x4 => Int8,
            _ => return None,
        };
        let block = match self.block {
            Pixel | Sub1x2 | Sub1x4 | Sub2x2 | Sub2x4 | Sub4x4 => self.block,
        };
        Some(Texel {
            samples: Samples { bits, parts },
            block,
            color: self.color.clone(),
        })
    }
}

impl SampleBits {
    /// Determine the number of bytes for texels containing these samples.
    pub fn bytes(self) -> usize {
        use SampleBits::*;
        match self {
            Int8 | Int332 | Int233 => 1,
            Int8x2 | Int16 | Int565 | Int4x4 | Int444i | Inti444 => 2,
            Int8x3 => 3,
            Int8x4 | Int16x2 | Int1010102 | Int2101010 | Int101010i | Inti101010 => 4,
            Int16x3 => 6,
            Int16x4 | Float16x4 => 8,
            Float32x4 => 16,
        }
    }
}

impl SampleParts {
    pub fn num_components(self) -> u8 {
        use SampleParts::*;
        match self {
            A | R | G | B | Luma => 1,
            LumaA => 2,
            Rgb | Bgr | Yuv => 3,
            Rgba | Bgra | Rgbx | Bgrx | Argb | Xrgb | Abgr | Xbgr => 4,
        }
    }

    pub fn into_vec4(self) -> Option<[f32; 4]> {
        Some(match self {
            Self::A => [0.0, 0.0, 0.0, 1.0],
            Self::R => [1.0, 0.0, 0.0, 0.0],
            Self::G => [0.0, 1.0, 0.0, 0.0],
            Self::B => [0.0, 0.0, 1.0, 0.0],
            _ => return None,
        })
    }
}

impl Color {
    pub const SRGB: Color = Color::Xyz {
        luminance: Luminance::Sdr,
        primary: Primaries::Bt709,
        transfer: Transfer::Srgb,
        whitepoint: Whitepoint::D65,
    };

    pub const BT709: Color = Color::Xyz {
        luminance: Luminance::Sdr,
        primary: Primaries::Bt709,
        transfer: Transfer::Bt709,
        whitepoint: Whitepoint::D65,
    };

    /// Check if this color space contains the sample parts.
    ///
    /// For example, an Xyz color is expressed in terms of a subset of Rgb while HSV color spaces
    /// contains the Hsv parts (duh!) and CIECAM and similar spaces have a polar representation of
    /// hue etc.
    ///
    /// Note that one can always combine a color space with an alpha component.
    pub fn is_consistent(&self, parts: SampleParts) -> bool {
        use SampleParts::*;
        match (self, parts) {
            (Color::Xyz { .. }, R | G | B | Rgb | Rgba | Rgbx) => true,
            _ => false,
        }
    }
}

#[rustfmt::skip]
impl Primaries {
    pub(crate) fn to_xyz(&self, white: Whitepoint) -> RowMatrix {
        use Primaries::*;
        // Rec.BT.601
        // https://en.wikipedia.org/wiki/Color_spaces_with_RGB_primaries#Specifications_with_RGB_primaries
        let xy: [[f32; 2]; 3] = match *self {
            Bt601_525 | Smpte240 => [[0.63, 0.34], [0.31, 0.595], [0.155, 0.07]],
            Bt601_625 => [[0.64, 0.33], [0.29, 0.6], [0.15, 0.06]],
            Bt709 => [[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]],
            Bt2020 | Bt2100 => [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]],
        };

        // A column of CIE XYZ intensities for that primary.
        let xyz = |[x, y]: [f32; 2]| {
            return [x / y, 1.0, (1.0 - x - y)/y];
        };

        let xyz_r = xyz(xy[0]);
        let xyz_g = xyz(xy[1]);
        let xyz_b = xyz(xy[2]);

        // Virtually, N = [xyz_r | xyz_g | xyz_b]
        // As the unweighted conversion matrix for:
        //  XYZ = N · RGB
        let RowMatrix(n1) = ColMatrix([xyz_r, xyz_g, xyz_b]).inv();

        // http://www.brucelindbloom.com/index.html
        let w = white.to_xyz();

        // s is the weights that give the whitepoint when converted to xyz.
        // That is we're solving:
        //  W = N · S
        let s = [
            (w[0]*n1[0] + w[1]*n1[1] + w[2]*n1[2]),
            (w[0]*n1[3] + w[1]*n1[4] + w[2]*n1[5]),
            (w[0]*n1[6] + w[1]*n1[7] + w[2]*n1[8]),
        ];

        /* If you want to debug this (for comparison to reference):
        eprintln!("{:?} for {:?}", self, white);
        eprintln!("{:?}", ColMatrix([xyz_r, xyz_g, xyz_b]));
        eprintln!("W: {:?}", w);
        eprintln!("N1: {:?}", RowMatrix(n1));
        eprintln!("S: {:?}", s);
        */

        RowMatrix([
            s[0]*xyz_r[0], s[1]*xyz_g[0], s[2]*xyz_b[0],
            s[0]*xyz_r[1], s[1]*xyz_g[1], s[2]*xyz_b[1],
            s[0]*xyz_r[2], s[1]*xyz_g[2], s[2]*xyz_b[2],
        ])
    }

}

#[rustfmt::skip]
impl ColMatrix {
    fn adj(self) -> RowMatrix {
        let m = self.0;

        let det = |c1: usize, c2: usize, r1: usize, r2: usize| {
            m[c1][r1] * m[c2][r2] - m[c2][r1] * m[c1][r2]
        };

        RowMatrix([
            det(1, 2, 1, 2), -det(1, 2, 0, 2), det(1, 2, 0, 1),
            -det(0, 2, 1, 2), det(0, 2, 0, 2), -det(0, 2, 0, 1),
            det(0, 1, 1, 2), -det(0, 1, 0, 2), det(0, 1, 0, 1),
        ])
    }

    fn det(self) -> f32 {
        let det2 = |ma, mb, na, nb| {
            ma * nb - na * mb
        };
        let [x, y, z] = self.0;
        x[0] * det2(y[1], y[2], z[1], z[2])
            - x[1] * det2(y[0], y[2], z[0], z[2])
            + x[2] * det2(y[0], y[1], z[0], z[1])
    }

    pub(crate) fn inv(self) -> RowMatrix {
        let RowMatrix(adj) = self.adj();
        let det_n = self.det();

        RowMatrix([
            adj[0] / det_n, adj[1] / det_n, adj[2] / det_n,
            adj[3] / det_n, adj[4] / det_n, adj[5] / det_n,
            adj[6] / det_n, adj[7] / det_n, adj[8] / det_n,
        ])
    }
}

#[rustfmt::skip]
impl RowMatrix {
    pub(crate) fn new(rows: [f32; 9]) -> RowMatrix {
        RowMatrix(rows)
    }

    pub(crate) fn diag(x: f32, y: f32, z: f32) -> Self {
        RowMatrix([
            x, 0., 0.,
            0., y, 0.,
            0., 0., z,
        ])
    }

    pub(crate) fn inv(self) -> RowMatrix {
        ColMatrix::from(self).inv()
    }

    pub(crate) fn det(self) -> f32 {
        ColMatrix::from(self).det()
    }

    /// Multiply with a homogeneous point.
    /// Note: might produce NaN if the matrix isn't a valid transform and may produce infinite
    /// points.
    pub(crate) fn multiply_point(self, point: [f32; 2]) -> [f32; 2] {
        let [x, y, z] = self.multiply_column([point[0], point[1], 1.0]);
        [x / z, y / z]
    }

    /// Calculate self · other
    pub(crate) fn multiply_column(self, col: [f32; 3]) -> [f32; 3] {
        let x = &self.0[0..3];
        let y = &self.0[3..6];
        let z = &self.0[6..9];

        let dot = |r: &[f32], c: [f32; 3]| {
            r[0] * c[0] + r[1] * c[1] + r[2] * c[2]
        };

        [dot(x, col), dot(y, col), dot(z, col)]
    }

    /// Calculate self · other
    pub(crate) fn multiply_right(self, ColMatrix([a, b, c]): ColMatrix) -> ColMatrix {
        ColMatrix([
            self.multiply_column(a),
            self.multiply_column(b),
            self.multiply_column(c),
        ])
    }

    pub(crate) fn into_inner(self) -> [f32; 9] {
        self.0
    }
}

#[rustfmt::skip]
impl From<ColMatrix> for RowMatrix {
    fn from(ColMatrix(m): ColMatrix) -> RowMatrix {
        RowMatrix([
            m[0][0], m[1][0], m[2][0],
            m[0][1], m[1][1], m[2][1],
            m[0][2], m[1][2], m[2][2],
        ])
    }
}

#[rustfmt::skip]
impl From<RowMatrix> for ColMatrix {
    fn from(RowMatrix(r): RowMatrix) -> ColMatrix {
        ColMatrix([
            [r[0], r[3], r[6]],
            [r[1], r[4], r[7]],
            [r[2], r[5], r[8]],
        ])
    }
}

impl Whitepoint {
    pub(crate) fn to_xyz(self) -> [f32; 3] {
        use Whitepoint::*;
        match self {
            A => [1.09850, 1.00000, 0.35585],
            B => [0.99072, 1.00000, 0.85223],
            C => [0.98074, 1.00000, 1.18232],
            D50 => [0.96422, 1.00000, 0.82521],
            D55 => [0.95682, 1.00000, 0.92149],
            D65 => [0.95047, 1.00000, 1.08883],
            D75 => [0.94972, 1.00000, 1.22638],
            E => [1.00000, 1.00000, 1.00000],
            F2 => [0.99186, 1.00000, 0.67393],
            F7 => [0.95041, 1.00000, 1.08747],
            F11 => [1.00962, 1.00000, 0.64350],
        }
    }
}

impl Block {
    pub fn width(&self) -> u32 {
        use Block::*;
        match self {
            Pixel => 1,
            Sub1x2 | Sub2x2 => 2,
            Sub1x4 | Sub2x4 | Sub4x4 => 4,
        }
    }

    pub fn height(&self) -> u32 {
        use Block::*;
        match self {
            Pixel | Sub1x2 | Sub1x4 => 1,
            Sub2x2 | Sub2x4 => 2,
            Sub4x4 => 3,
        }
    }
}

impl ImageBuffer {
    /// Allocate a new image buffer given its layout.
    pub fn with_layout(layout: &BufferLayout) -> Self {
        let inner = Canvas::new(layout.clone());
        ImageBuffer { inner }
    }

    pub fn layout(&self) -> &BufferLayout {
        self.inner.layout()
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.inner.as_bytes()
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.inner.as_bytes_mut()
    }
}

impl BufferLayout {
    pub fn with_row_layout(rows: RowLayoutDescription) -> Option<Self> {
        let bytes_per_texel = u8::try_from(rows.texel_stride).ok()?;
        let bytes_per_row = u32::try_from(rows.row_stride).ok()?;

        // Enforce that the layout makes sense and does not alias.
        let _ = u32::from(bytes_per_texel)
            .checked_mul(rows.width)
            .filter(|&bwidth| bwidth <= bytes_per_row)?;

        // Enforce our inner invariant.
        let u64_len = u64::from(rows.height).checked_mul(rows.row_stride)?;
        let _ = usize::try_from(u64_len).ok()?;

        Some(BufferLayout {
            width: rows.width,
            height: rows.height,
            bytes_per_texel,
            bytes_per_row,
        })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn u64_len(&self) -> u64 {
        // No overflow due to inner invariant.
        u64::from(self.bytes_per_row) * u64::from(self.height)
    }

    pub fn byte_len(&self) -> usize {
        // No overflow due to inner invariant.
        (self.bytes_per_row as usize) * (self.height as usize)
    }

    pub fn as_row_layout(&self) -> RowLayoutDescription {
        RowLayoutDescription {
            width: self.width,
            height: self.height,
            texel_stride: u64::from(self.bytes_per_texel),
            row_stride: u64::from(self.bytes_per_row),
        }
    }
}

impl Layout for BufferLayout {
    fn byte_len(&self) -> usize {
        BufferLayout::byte_len(self)
    }
}

impl From<&'_ image::DynamicImage> for BufferLayout {
    fn from(image: &'_ image::DynamicImage) -> BufferLayout {
        use image::GenericImageView;
        let (width, height) = image.dimensions();
        let bytes_per_texel = image.color().bytes_per_pixel();
        let bytes_per_row = width * u32::from(bytes_per_texel);

        BufferLayout {
            width,
            height,
            bytes_per_texel,
            bytes_per_row,
        }
    }
}

impl From<&'_ image::DynamicImage> for ImageBuffer {
    fn from(image: &'_ image::DynamicImage) -> ImageBuffer {
        let layout = BufferLayout::from(image);
        let inner = Canvas::with_bytes(layout, image.as_bytes());
        ImageBuffer { inner }
    }
}
