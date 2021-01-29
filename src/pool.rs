use slotmap::{DefaultKey, SlotMap};
use wgpu::{Buffer, Texture};

use crate::buffer::{BufferLayout, ImageBuffer};

/// Holds a number of image buffers, their descriptors and meta data.
///
/// The buffers can be owned in various different manners.
#[derive(Default)]
pub struct Pool {
    items: SlotMap<DefaultKey, Image>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct PoolKey(DefaultKey);

/// A handle on an image inside the pool.
pub struct PoolImage<'pool> {
    /// The key of the slot map referring to this entry.
    key: DefaultKey,
    /// The image inside the pool.
    image: &'pool mut Image,
}

struct Image {
    meta: ImageMeta,
    data: ImageData,
}

/// Meta data distinct from the layout questions.
pub(crate) struct ImageMeta {
    /// Do we guarantee consistent content to read?
    /// Images with this set to `false` may be arbitrarily used as a temporary buffer for other
    /// operations, overwriting the contents at will.
    no_read: bool,
    /// Should we permit writing to this image?
    /// If not then the device can allocate/cache it differently.
    no_write: bool,
}

enum ImageData {
    Host(ImageBuffer),
    /// The data lives in a generic buffer.
    Gpu(Buffer, BufferLayout),
    /// The data lives in a texture buffer on the device.
    GpuTexture(Texture, BufferLayout),
    /// The image data will be provided by the caller.
    /// Such data can only be used in operations that do not keep a reference, e.g. it is not
    /// possible to create a mere view.
    LateBound(BufferLayout),
}

impl Pool {
    /// Create an empty pool.
    pub fn new() -> Self {
        Pool::default()
    }

    /// Get a mutable handle of an image in the pool.
    pub fn entry(&mut self, PoolKey(key): PoolKey) -> Option<PoolImage<'_>> {
        Some(PoolImage {
            key,
            image: self.items.get_mut(key)?,
        })
    }

    /// Gift the pool an image allocated on the host.
    pub fn insert(&mut self, image: ImageBuffer) -> PoolImage<'_> {
        self.new_with_data(ImageData::Host(image))
    }

    /// Create the descriptor for an image buffer that is provided by the caller.
    pub fn declare(&mut self, layout: BufferLayout) -> PoolImage<'_> {
        self.new_with_data(ImageData::LateBound(layout))
    }

    fn new_with_data(&mut self, data: ImageData) -> PoolImage<'_> {
        let key = self.items.insert(Image {
            meta: ImageMeta::default(),
            data,
        });

        PoolImage {
            key,
            image: &mut self.items[key],
        }
    }
}

impl PoolImage<'_> {
    pub fn key(&self) -> PoolKey {
        PoolKey(self.key)
    }

    pub fn layout(&self) -> &BufferLayout {
        self.image.data.layout()
    }
}

impl ImageData {
    fn layout(&self) -> &BufferLayout {
        match self {
            ImageData::Host(canvas) => canvas.layout(),
            ImageData::Gpu(_, layout) => layout,
            ImageData::GpuTexture(_, layout) => layout,
            ImageData::LateBound(layout) => layout,
        }
    }
}

impl Default for ImageMeta {
    fn default() -> Self {
        ImageMeta {
            no_read: false,
            no_write: false,
        }
    }
}
