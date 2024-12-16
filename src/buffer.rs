//! Defines layout and buffer of our images.
pub use image_canvas::{
    color::{Color, ColorChannel, Transfer, Whitepoint},
    layout::{Block, CanvasLayout, RowLayoutDescription, SampleBits, SampleParts, Texel},
    Canvas,
};

#[derive(Clone)]
pub struct ImageBuffer {
    inner: Canvas,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ByteLayout {
    pub width: u32,
    pub height: u32,
    pub row_stride: u64,
    pub texel_stride: u16,
}

/// Describes an image semantically.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Descriptor {
    /// The byte and physical layout of the buffer.
    pub layout: ByteLayout,
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
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum ChannelPosition {
    First = 0,
    Second = 1,
    Third = 2,
    Fourth = 3,
}

pub(crate) trait TexelExt {
    fn channel_texel(&self, _: ColorChannel) -> Option<Texel>;
    fn channel_weight_vec4(&self) -> Option<[f32; 4]>;
}

impl TexelExt for Texel {
    fn channel_texel(&self, ch: ColorChannel) -> Option<Texel> {
        Some(Texel {
            parts: self.parts.with_channel(ch)?,
            ..*self
        })
    }

    fn channel_weight_vec4(&self) -> Option<[f32; 4]> {
        use ColorChannel::*;

        let ch = match self.parts.color_channels() {
            [Some(ch), None, None, None] => ch,
            _ => return None,
        };

        Some(match ch {
            R | Luma | L | X | Scalar0 => [1.0, 0.0, 0.0, 0.0],
            G | Cb | LABa | C | Y | Scalar1 => [0.0, 1.0, 0.0, 0.0],
            B | Cr | LABb | LABh | Z | Scalar2 => [0.0, 0.0, 1.0, 0.0],
            Alpha => [0.0, 0.0, 0.0, 1.0],
            _ => return None,
        })
    }
}

impl Descriptor {
    pub fn empty() -> Self {
        Descriptor {
            layout: ByteLayout {
                width: 0,
                height: 0,
                row_stride: 0,
                texel_stride: 0,
            },
            color: Color::SRGB,
            texel: Texel::new_u8(SampleParts::RgbA),
        }
    }

    pub fn with_texel(texel: Texel, width: u32, height: u32) -> Option<Self> {
        let layout = ByteLayout {
            width,
            height,
            row_stride: u64::from(texel.bits.bytes()) * u64::from(width),
            texel_stride: texel.bits.bytes(),
        };

        let color = Color::Scalars {
            transfer: Transfer::Linear,
        };

        let this = Descriptor {
            color,
            layout,
            texel,
        };

        let _ = this.try_to_canvas()?;
        Some(this)
    }

    /// Create the highly row-aligned layout with the same row bytes.
    ///
    /// This can overflow as it uses more bytes than the underlying flexible CPU layout. Returns
    /// `None` when this occurs.
    pub(crate) fn to_aligned(&self) -> Option<ByteLayout> {
        let bytes_per_row = (self.layout.texel_stride as u32).checked_mul(self.layout.width)?;
        let bytes_per_row = bytes_per_row.next_multiple_of(256);

        // Verify the total byte count can still be expressed as u64.
        let _ = bytes_per_row.checked_mul(self.layout.height)?;

        Some(ByteLayout {
            texel_stride: self.texel.bits.bytes(),
            width: self.layout.width,
            height: self.layout.height,
            row_stride: bytes_per_row.into(),
        })
    }

    /// Calculate the necessary length to represent this on the GPU.
    pub(crate) fn u64_gpu_len(&self) -> Option<u64> {
        let layout = self.to_aligned()?;
        layout.row_stride.checked_mul(layout.height.into())
    }

    /// Convert to the underlying `image-canvas` layout for the host.
    pub(crate) fn to_canvas(&self) -> CanvasLayout {
        self.try_to_canvas()
            .expect("To be validated on construction")
    }

    pub(crate) fn try_to_canvas(&self) -> Option<CanvasLayout> {
        let descriptor = RowLayoutDescription {
            width: self.layout.width,
            height: self.layout.height,
            row_stride: self.layout.row_stride,
            texel: self.texel.clone(),
        };

        CanvasLayout::with_row_layout(&descriptor).ok()
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

impl ByteLayout {
    /// Calculate the length in bytes, wrapping on overflow.
    ///
    /// The layout contained in a `Descriptor` for instance is pre-verified not to wrap when it is
    /// used in a command buffer. In this case the length is also the mathematical length.
    /// Otherwise you'll need to verify this for yourself.
    pub fn wrapping_len(&self) -> u64 {
        u64::from(self.height).wrapping_mul(self.row_stride)
    }
}

type ImageAllocator = fn(u32, u32, &[u8]) -> Option<image::DynamicImage>;

impl Descriptor {
    /// Creates a descriptor for an sRGB encoded image, with the indicated color type.
    pub fn with_srgb_image(image: &'_ image::DynamicImage) -> Descriptor {
        Descriptor {
            color: Color::SRGB,
            layout: ByteLayout::from(image),
            texel: Self::texel(image),
        }
    }

    pub(crate) fn as_image_allocator(that: &Texel) -> Option<ImageAllocator> {
        use SampleBits as B;
        use SampleParts as P;

        Some(match that {
            Texel {
                block: Block::Pixel,
                parts: P::Luma,
                bits: B::UInt8,
            } => |width, height, source| {
                let buffer = image::ImageBuffer::from_vec(width, height, source.to_vec())?;
                Some(image::DynamicImage::ImageLuma8(buffer))
            },
            Texel {
                block: Block::Pixel,
                parts: P::LumaA,
                bits: B::UInt8x2,
            } => |width, height, source| {
                let buffer = image::ImageBuffer::from_vec(width, height, source.to_vec())?;
                Some(image::DynamicImage::ImageLumaA8(buffer))
            },
            Texel {
                block: Block::Pixel,
                parts: P::Rgb,
                bits: B::UInt8x3,
            } => |width, height, source| {
                let buffer = image::ImageBuffer::from_vec(width, height, source.to_vec())?;
                Some(image::DynamicImage::ImageRgb8(buffer))
            },
            Texel {
                block: Block::Pixel,
                parts: P::RgbA,
                bits: B::UInt8x4,
            } => |width, height, source| {
                let buffer = image::ImageBuffer::from_vec(width, height, source.to_vec())?;
                Some(image::DynamicImage::ImageRgba8(buffer))
            },
            // TODO: quite a lot of duplication below. Can we somehow reduce that?
            Texel {
                block: Block::Pixel,
                parts: P::Luma,
                bits: B::UInt16,
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
                bits: B::UInt16x2,
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
                bits: B::UInt16x3,
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
                bits: B::UInt16x4,
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
                bits: SampleBits::UInt8,
                parts: SampleParts::Luma,
            },
            ImageLumaA8(_) => Texel {
                block: Block::Pixel,
                bits: SampleBits::UInt8x2,
                parts: SampleParts::LumaA,
            },
            ImageLuma16(_) => Texel {
                block: Block::Pixel,
                bits: SampleBits::UInt16,
                parts: SampleParts::Luma,
            },
            ImageLumaA16(_) => Texel {
                block: Block::Pixel,
                bits: SampleBits::UInt16x2,
                parts: SampleParts::LumaA,
            },
            ImageRgb8(_) => Texel {
                block: Block::Pixel,
                bits: SampleBits::UInt8x3,
                parts: SampleParts::Rgb,
            },
            ImageRgba8(_) => Texel {
                block: Block::Pixel,
                bits: SampleBits::UInt8x4,
                parts: SampleParts::RgbA,
            },
            ImageRgb16(_) => Texel {
                block: Block::Pixel,
                bits: SampleBits::UInt16x3,
                parts: SampleParts::Rgb,
            },
            ImageRgba16(_) => Texel {
                block: Block::Pixel,
                bits: SampleBits::UInt16x4,
                parts: SampleParts::RgbA,
            },
            _ => unreachable!("Promise, we match the rest"),
        }
    }
}

impl ImageBuffer {
    /// Allocate a new image buffer given its layout.
    pub fn with_layout(layout: &CanvasLayout) -> Self {
        let inner = Canvas::new(layout.clone());
        ImageBuffer { inner }
    }

    /// Allocate a new image buffer given its layout.
    pub fn with_descriptor(descriptor: &Descriptor) -> Self {
        let layout = descriptor.to_canvas();
        Self::with_layout(&layout)
    }

    pub fn layout(&self) -> &CanvasLayout {
        self.inner.layout()
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.inner.as_bytes()
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.inner.as_bytes_mut()
    }
}

impl From<&'_ image::DynamicImage> for ByteLayout {
    fn from(image: &'_ image::DynamicImage) -> ByteLayout {
        use image::GenericImageView;
        let (width, height) = image.dimensions();
        let color = image.color();
        let bpp = color.bytes_per_pixel();

        ByteLayout {
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
        let mut layout = CanvasLayout::with_row_layout(&RowLayoutDescription {
            width: image.width(),
            height: image.height(),
            row_stride: descriptor.layout.row_stride,
            texel: descriptor.texel,
        })
        .expect("Valid layout");

        layout
            .set_color(descriptor.color)
            .expect("valid srgb color");
        let mut inner = Canvas::new(layout);

        let source = image.as_bytes();
        let target = inner.as_bytes_mut();
        let len = source.len().min(target.len());
        target[..len].copy_from_slice(&source[..len]);

        ImageBuffer { inner }
    }
}

impl From<&'_ CanvasLayout> for Descriptor {
    fn from(buf: &CanvasLayout) -> Descriptor {
        // FIXME: panics on purpose.
        let _plane = buf.as_plane().unwrap();

        let layout = ByteLayout {
            width: buf.width(),
            height: buf.height(),
            row_stride: buf.as_row_layout().row_stride,
            texel_stride: buf.texel().bits.bytes(),
        };

        Descriptor {
            layout,
            // FIXME: this makes me reconsider if it should be a From-impl. Maybe just a regular
            // method that notes this default in its name?
            color: buf.color().unwrap_or(&Color::SRGB).clone(),
            texel: buf.texel().clone(),
        }
    }
}
