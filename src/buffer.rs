//! Defines layout and buffer of our images.
use canvas::{Canvas, layout::Layout};

/// The byte layout of a buffer.
///
/// An inner invariant is that the layout fits in memory and in particular into a `usize`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferLayout {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) bytes_per_texel: u8,
    pub(crate) bytes_per_row: u32,
}

/// Describe a row-major rectangular matrix layout.
///
/// This is only concerned with byte-buffer compatibility and not type or color space semantics of
/// texels. It assumes a row-major layout without space between texels of a row as that is the most
/// efficient and common such layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RowLayoutDescription {
    pub width: u32,
    pub height: u32,
    pub stride: u64,
}

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
pub enum Block {
    /// Each texel is a single pixel.
    Pixel,
    /// Each texel refers to two pixels across width.
    Sub1x2,
    /// Each texel refers to four pixels across width.
    Sub1x4,
    /// Each texel refers to a two-by-two block.
    Sub2x2,
    /// Each texel refers to a two-by-four block.
    Sub2x4,
    /// Each texel refers to a four-by-four block.
    Sub4x4,
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
pub enum SampleParts {
    A,
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
    /// Linear color in scene luminance.
    /// This is perfect for an artistic composition pipeline.
    LinearScene,
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
        use SampleParts::*;
        use SampleBits::*;
        let parts = match self.samples.parts {
            Rgb | Rgbx | Rgba | Bgrx | Bgra | Abgr | Argb | Xrgb | Xbgr => match channel {
                ColorChannel::R => R,
                ColorChannel::G => G,
                ColorChannel::B => B,
                _ => return None,
            },
            _ => return None,
        };
        let bits = match self.samples.bits {
            Int8 | Int8x3 | Int8x4 => Int8,
            _ => return None,
        };
        let block = match self.block {
            Pixel | Sub1x2 | Sub1x4 | Sub2x2 | Sub2x4 | Sub4x4 => self.block,
            _ => return None,
        };
        Some(Texel {
            samples: Samples {
                bits,
                parts
            },
            block,
            color: self.color.clone(),
        })
    }
}

impl Samples {
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

impl Color {
    pub const SRGB: Color = Color::Xyz {
        luminance: Luminance::Sdr,
        primary: Primaries::Bt709,
        transfer: Transfer::Srgb,
        whitepoint: Whitepoint::D65,
    };
}

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

impl RowMatrix {
    pub(crate) fn new(rows: [f32; 9]) -> RowMatrix {
        RowMatrix(rows)
    }

    pub(crate) fn inv(self) -> RowMatrix {
        ColMatrix::from(self).inv()
    }

    /// Calculate self · other
    pub(crate) fn multiply_right(self, ColMatrix([a, b, c]): ColMatrix) -> ColMatrix {
        let x = &self.0[0..3];
        let y = &self.0[3..6];
        let z = &self.0[6..9];

        let dot = |r: &[f32], c: [f32; 3]| {
            r[0] * c[0] + r[1] * c[1] + r[2] * c[2]
        };

        ColMatrix([
            [dot(x, a), dot(y, a), dot(z, a)],
            [dot(x, b), dot(y, b), dot(z, b)],
            [dot(x, c), dot(y, c), dot(z, c)],
        ])
    }

    pub(crate) fn into_inner(self) -> [f32; 9] {
        self.0
    }
}

impl From<ColMatrix> for RowMatrix {
    fn from(ColMatrix(m): ColMatrix) -> RowMatrix {
        RowMatrix([
            m[0][0], m[1][0], m[2][0],
            m[0][1], m[1][1], m[2][1],
            m[0][2], m[1][2], m[2][2],
        ])
    }
}

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
            B => [0.99072 , 1.00000 , 0.85223],
            C => [0.98074 , 1.00000 , 1.18232],
            D50 => [0.96422 , 1.00000 , 0.82521],
            D55 => [0.95682 , 1.00000 , 0.92149],
            D65 => [0.95047 , 1.00000 , 1.08883],
            D75 => [0.94972 , 1.00000 , 1.22638],
            E => [1.00000 , 1.00000 , 1.00000],
            F2 => [0.99186 , 1.00000 , 0.67393],
            F7 => [0.95041 , 1.00000 , 1.08747],
            F11 => [1.00962 , 1.00000 , 0.64350],
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
    pub fn with_row_layout(_: RowLayoutDescription) -> Option<Self> {
        todo!()
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
