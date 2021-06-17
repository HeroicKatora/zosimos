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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Whitepoint {
    D65,
}

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
        u64::from(self.width) * u64::from(self.height) * u64::from(self.bytes_per_texel)
    }

    pub fn byte_len(&self) -> usize {
        // No overflow due to inner invariant.
        (self.width as usize) * (self.height as usize) * usize::from(self.bytes_per_texel)
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
