//! Defines layout and buffer of our images.
pub use image_canvas::{
    Canvas,
    layout::{
        CanvasLayout as BufferLayout,
        Block,
        RowLayoutDescription,
        SampleBits,
        SampleParts,
        Texel,
    },
    color::{
        Color,
        Transfer,
        Whitepoint,
    },
};

#[derive(Clone)]
pub struct ImageBuffer {
    inner: Canvas,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferSize {
    pub width: u32,
    pub height: u32,
    pub texel_stride: u8,
    pub row_stride: u64,
}

/// Describes an image semantically.
#[derive(Clone, Debug, PartialEq)]
pub struct Descriptor {
    /// The byte and physical layout of the buffer.
    pub layout: BufferSize,
    /// The color interpretation of texels.
    pub color: Color,
    /// Describe how each single texel is interpreted.
    pub texel: Texel,
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

/// A column major matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ColMatrix([[f32; 3]; 3]);

/// A row major matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RowMatrix([f32; 9]);

impl Descriptor {
    pub const EMPTY: Self = Descriptor {
        layout: BufferSize {
            width: 0,
            height: 0,
            row_stride: 0,
            texel_stride: 0,
        },
        color: Color::SRGB,
        texel: Texel::new_u8(SampleParts::RgbA),
    };

    fn with_texel(texel: Texel, width: u32, height: u32) -> Option<Self> {
        let layout = BufferLayout::with_texel(&texel, width, height)?;
        let color = Color::Scalars { transfer: Transfer::Linear };
        Some(Descriptor { color, layout, texel })
    }

    /// Check if the descriptor is consistent.
    ///
    /// A consistent descriptor makes inherent sense. That is, the different fields contain values
    /// that are not contradictory. For example, the color channels parts and the color model
    /// correspond to each other, and the sample parts and sample bits field is correct, and the
    /// texel descriptor has the same number of bytes as the layout, etc.
    pub fn is_consistent(&self) -> bool {
        // FIXME: other checks.
        self.texel.bits.bytes() == <_>::from(self.layout.texel_stride)
    }

    /// Calculate the total number of pixels in width of this layout.
    pub fn pixel_width(&self) -> u32 {
        self.layout.width * self.texel.block.width()
    }

    /// Calculate the total number of pixels in height of this layout.
    pub fn pixel_height(&self) -> u32 {
        self.layout.height * self.texel.block.height()
    }

    /// Calculate the number of texels in width and height dimension.
    pub fn size(&self) -> (u32, u32) {
        (self.layout.width, self.layout.height)
    }
}

type ImageAllocator = fn(u32, u32, &[u8]) -> Option<image::DynamicImage>;

impl Descriptor {
    /// Creates a descriptor for an sRGB encoded image, with the indicated color type.
    pub fn with_srgb_image(image: &'_ image::DynamicImage) -> Descriptor {
        Descriptor {
            color: Color::SRGB,
            layout: BufferSize::from(image),
            texel: Self::texel(image),
        }
    }

    pub(crate) fn into_vec4(that: SampleParts) -> Option<[f32; 4]> {
        Some(match that {
            SampleParts::A => [0.0, 0.0, 0.0, 1.0],
            SampleParts::R => [1.0, 0.0, 0.0, 0.0],
            SampleParts::G => [0.0, 1.0, 0.0, 0.0],
            SampleParts::B => [0.0, 0.0, 1.0, 0.0],
            _ => return None,
        })
    }

    pub(crate) fn as_image_allocator(that: &Texel) -> Option<ImageAllocator> {
        use SampleBits as B;
        use SampleParts as P;

        Some(match that {
            Texel {
                block: Block::Pixel,
                parts: P::Luma,
                bits: B::Int8,
            } => |width, height, source| {
                let buffer = image::ImageBuffer::from_vec(width, height, source.to_vec())?;
                Some(image::DynamicImage::ImageLuma8(buffer))
            },
            Texel {
                block: Block::Pixel,
                parts: P::LumaA,
                bits: B::Int8x2,
            } => |width, height, source| {
                let buffer = image::ImageBuffer::from_vec(width, height, source.to_vec())?;
                Some(image::DynamicImage::ImageLumaA8(buffer))
            },
            Texel {
                block: Block::Pixel,
                parts: P::Rgb,
                bits: B::Int8x3,
            } => |width, height, source| {
                let buffer = image::ImageBuffer::from_vec(width, height, source.to_vec())?;
                Some(image::DynamicImage::ImageRgb8(buffer))
            },
            Texel {
                block: Block::Pixel,
                parts: P::RgbA,
                bits: B::Int8x4,
            } => |width, height, source| {
                let buffer = image::ImageBuffer::from_vec(width, height, source.to_vec())?;
                Some(image::DynamicImage::ImageRgba8(buffer))
            },
            // TODO: quite a lot of duplication below. Can we somehow reduce that?
            Texel {
                block: Block::Pixel,
                parts: P::Luma,
                bits: B::Int16,
            } => |width, height, source| {
                let source = &source[..(source.len() / 2) * 2];
                let buffer = image::ImageBuffer::from_vec(width, height, {
                    let mut data = vec![0u16; source.len() / 2];
                    bytemuck::cast_slice_mut(&mut data).copy_from_slice(source);
                    data
                })?;
                Some(image::DynamicImage::ImageLuma16(buffer))
            },
            Texel {
                block: Block::Pixel,
                parts: P::LumaA,
                bits: B::Int16x2,
            } => |width, height, source| {
                let source = &source[..(source.len() / 2) * 2];
                let buffer = image::ImageBuffer::from_vec(width, height, {
                    let mut data = vec![0u16; source.len() / 2];
                    bytemuck::cast_slice_mut(&mut data).copy_from_slice(source);
                    data
                })?;
                Some(image::DynamicImage::ImageLumaA16(buffer))
            },
            Texel {
                block: Block::Pixel,
                parts: P::Rgb,
                bits: B::Int16x3,
            } => |width, height, source| {
                let source = &source[..(source.len() / 2) * 2];
                let buffer = image::ImageBuffer::from_vec(width, height, {
                    let mut data = vec![0u16; source.len() / 2];
                    bytemuck::cast_slice_mut(&mut data).copy_from_slice(source);
                    data
                })?;
                Some(image::DynamicImage::ImageRgb16(buffer))
            },
            Texel {
                block: Block::Pixel,
                parts: P::RgbA,
                bits: B::Int16x4,
            } => |width, height, source| {
                let source = &source[..(source.len() / 2) * 2];
                let buffer = image::ImageBuffer::from_vec(width, height, {
                    let mut data = vec![0u16; source.len() / 2];
                    bytemuck::cast_slice_mut(&mut data).copy_from_slice(source);
                    data
                })?;
                Some(image::DynamicImage::ImageRgba16(buffer))
            },
            _ => return None,
        })
    }

    fn texel(image: &image::DynamicImage) -> Texel {
        use image::DynamicImage::*;
        match image {
            ImageLuma8(_) => Texel {
                block: Block::Pixel,
                bits: SampleBits::Int8,
                parts: SampleParts::Luma,
            },
            ImageLumaA8(_) => Texel {
                block: Block::Pixel,
                bits: SampleBits::Int8x2,
                parts: SampleParts::LumaA,
            },
            ImageLuma16(_) => Texel {
                block: Block::Pixel,
                bits: SampleBits::Int16,
                parts: SampleParts::Luma,
            },
            ImageLumaA16(_) => Texel {
                block: Block::Pixel,
                bits: SampleBits::Int16x2,
                parts: SampleParts::LumaA,
            },
            ImageRgb8(_) => Texel {
                block: Block::Pixel,
                bits: SampleBits::Int8x3,
                parts: SampleParts::Rgb,
            },
            ImageRgba8(_) => Texel {
                block: Block::Pixel,
                bits: SampleBits::Int8x4,
                parts: SampleParts::RgbA,
            },
            ImageRgb16(_) => Texel {
                block: Block::Pixel,
                bits: SampleBits::Int16x3,
                parts: SampleParts::Rgb,
            },
            ImageRgba16(_) => Texel {
                block: Block::Pixel,
                bits: SampleBits::Int16x4,
                parts: SampleParts::RgbA,
            },
            _ => unreachable!("Promise, we match the rest"),
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

impl From<&'_ image::DynamicImage> for BufferSize {
    fn from(image: &'_ image::DynamicImage) -> BufferSize {
        use image::GenericImageView;
        let (width, height) = image.dimensions();
        let color = image.color();
        let bpp = color.bytes_per_pixel();

        BufferSize {
            width,
            height,
            row_stride: u64::from(width) * u64::from(bpp),
            texel_stride: <_>::from(bpp),
        }
    }
}

impl From<&'_ image::DynamicImage> for ImageBuffer {
    fn from(image: &'_ image::DynamicImage) -> ImageBuffer {
        let descriptor = Descriptor::with_srgb_image(image);
        let mut layout = BufferLayout::with_row_layout(&RowLayoutDescription {
            width: image.width(),
            height: image.height(),
            row_stride: descriptor.layout.row_stride,
            texel: descriptor.texel
        }).expect("Valid layout");
        layout.set_color(descriptor.color);
        let inner = Canvas::new(layout);
        ImageBuffer { inner }
    }
}
