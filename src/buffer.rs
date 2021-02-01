//! Defines layout and buffer of our images.
use canvas::layout::Layout;

pub struct BufferLayout {
    width: u32,
    height: u32,
}

/// Describe a row-major rectangular matrix layout.
///
/// This is only concerned with byte-buffer compatibility and not type or color space semantics of
/// texels. It assumes a row-major layout without space between texels of a row as that is the most
/// efficient and common such layout.
pub struct RowLayoutDescription {
    pub width: u32,
    pub height: u32,
    pub stride: u64,
}

pub struct ImageBuffer {
    inner: canvas::Canvas<BufferLayout>,
}

/// Describes an image semantically.
pub struct Descriptor {
    /// The byte and physical layout of the buffer.
    pub layout: BufferLayout,
    /// Describe how each single texel is interpreted.
    pub texel: Texel,
}

pub struct Texel {
    /// Which part of the image a single texel refers to.
    pub block: Block,
    /// How numbers and channels are encoded into the texel.
    pub samples: Samples,
    /// How the numbers relate to physical quantities, important for conversion.
    pub color: Color,
}

pub enum Block {
    /// Each texel is a single pixel.
    Pixel,
    Sub1x2,
    Sub1x4,
    Sub2x2,
    Sub2x4,
    Sub4x4,
}

/// The bit encoding of values within the texel bytes.
pub struct Samples {
    /// Which values are encoded, which controls the applicable color spaces.
    pub parts: SampleParts,
    /// How the values are encoded as bits in the bytes.
    pub bits: SampleBits,
}

/// Describes which values are present in a texel.
pub enum SampleParts {
    A,
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

pub enum SampleBits {
    /// A single 8-bit integer.
    Int8,
    /// Three packed integer.
    Int332,
    /// Three packed integer.
    Int233,
    /// Four packed integer.
    Int4x4,
    /// Four packed integer, one component ignored.
    Inti444,
    /// Four packed integer, one component ignored.
    Int444i,
    /// Three packed integer.
    Int565,
    /// Three 8-bit integer.
    Int8x3,
    /// Four 8-bit integer.
    Int8x4,
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
#[non_exhaustive]
pub enum Transfer {
    Bt709,
    Bt470M,
    Bt601,
    Smpte240,
    Linear,
    Srgb,
    Bt2020_10bit,
    Bt2020_12bit,
    Smpte2084,
    /// Another name for Smpte2084.
    Bt2100Pq,
    Bt2100Hlg,
}

/// The reference brightness of the color specification.
#[non_exhaustive]
pub enum Luminance {
    /// 100cd/m².
    Sdr,
    /// 10_000cd/m².
    /// Known as high-dynamic range.
    Hdr,
}

/// The relative stimuli of the three corners of a triangular gamut.
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
#[non_exhaustive]
pub enum Whitepoint {
    D65,
}

impl ImageBuffer {
    pub fn layout(&self) -> &BufferLayout {
        self.inner.layout()
    }
}

impl Layout for BufferLayout {
    fn byte_len(&self) -> usize {
        todo!()
    }
}
