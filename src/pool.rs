use core::{fmt, mem, num};
use std::collections::HashMap;

use slotmap::{DefaultKey, SlotMap};
use wgpu::{Buffer, Texture};

use crate::buffer::{BufferLayout, Color, Descriptor, ImageBuffer};
use crate::program::{BufferDescriptor, Capabilities, TextureDescriptor};
use crate::run::{block_on, Gpu};

/// Holds a number of image buffers, their descriptors and meta data.
///
/// The buffers can be owned in various different manners.
#[derive(Default)]
pub struct Pool {
    items: SlotMap<DefaultKey, Image>,
    buffers: SlotMap<DefaultKey, (BufferDescriptor, GpuKey, wgpu::Buffer)>,
    textures: SlotMap<DefaultKey, (TextureDescriptor, GpuKey, wgpu::Texture)>,
    devices: SlotMap<DefaultKey, Gpu>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PoolKey(DefaultKey);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GpuKey(DefaultKey);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct TextureKey(DefaultKey);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct BufferKey(DefaultKey);

pub(crate) struct Image {
    pub(crate) meta: ImageMeta,
    pub(crate) data: ImageData,
    pub(crate) descriptor: Descriptor,
}

/// A view on an image inside the pool.
pub struct PoolImage<'pool> {
    key: DefaultKey,
    image: &'pool Image,
}

/// A handle on an image inside the pool.
pub struct PoolImageMut<'pool> {
    /// The key of the slot map referring to this entry.
    key: DefaultKey,
    /// The image inside the pool.
    image: &'pool mut Image,
    /// All other devices.
    devices: &'pool SlotMap<DefaultKey, Gpu>,
}

pub struct Iter<'pool> {
    inner: slotmap::basic::Iter<'pool, DefaultKey, Image>,
}

pub struct IterMut<'pool> {
    inner: slotmap::basic::IterMut<'pool, DefaultKey, Image>,
    devices: &'pool SlotMap<DefaultKey, Gpu>,
}

/// Indexes a pool for extracting unused buffers.
pub struct Cache<'pool> {
    texture_sets: HashMap<TextureDescriptor, Vec<PoolKey>>,
    buffer_sets: HashMap<BufferDescriptor, Vec<BufferKey>>,
    pool: &'pool mut Pool,
}

/// Meta data distinct from the layout questions.
pub(crate) struct ImageMeta {
    /// Do we guarantee consistent content to read?
    /// Images with this set to `false` may be arbitrarily used as a temporary buffer for other
    /// operations, overwriting the contents at will.
    pub(crate) no_read: bool,
    /// Should we permit writing to this image?
    /// If not then the device can allocate/cache it differently.
    pub(crate) no_write: bool,
}

/// TODO: figure out if we should expose this or a privacy wrapper.
pub(crate) enum ImageData {
    Host(ImageBuffer),
    /// The data lives in a generic buffer.
    /// This buffer should be associated to one of the GPU devices.
    Gpu {
        buffer: Buffer,
        layout: BufferLayout,
        gpu: DefaultKey,
    },
    /// The data lives in a texture buffer on the device.
    /// This buffer should be associated to one of the GPU devices.
    GpuTexture {
        texture: Texture,
        layout: BufferLayout,
        gpu: DefaultKey,
    },
    /// The image data will be provided by the caller.
    /// Such data can only be used in operations that do not keep a reference, e.g. it is not
    /// possible to create a mere view.
    LateBound(BufferLayout),
}

impl PoolKey {
    /// Create a new pool key that does not name any image.
    pub fn null() -> Self {
        PoolKey(DefaultKey::default())
    }
}

impl Pool {
    /// Create an empty pool.
    pub fn new() -> Self {
        Pool::default()
    }

    /// Create a device given a descriptor of requested features.
    ///
    /// This will request a device from the adaptor according to the provided descriptor and then
    /// directly insert it into the pool. Then it returns the unique key for that newly created
    /// device and queue.
    pub fn request_device(
        &mut self,
        adapter: &wgpu::Adapter,
        device: wgpu::DeviceDescriptor,
    ) -> Result<GpuKey, wgpu::RequestDeviceError> {
        let request = adapter.request_device(&device, None);
        let request = Box::pin(request);
        let (device, queue) = block_on(request, None)?;
        let gpu_key = self.devices.insert(Gpu { device, queue });
        Ok(GpuKey(gpu_key))
    }

    pub fn iter_devices(&self) -> impl Iterator<Item = &'_ wgpu::Device> {
        self.devices.iter().map(|kv| &kv.1.device)
    }

    pub(crate) fn reinsert_device(&mut self, gpu: Gpu) -> GpuKey {
        GpuKey(self.devices.insert(gpu))
    }

    pub(crate) fn select_device(&mut self, caps: &Capabilities) -> Option<(GpuKey, Gpu)> {
        let key = self.select_device_key(caps)?;
        let device = self.devices.remove(key).unwrap();
        Some((GpuKey(key), device))
    }

    fn select_device_key(&mut self, _: &Capabilities) -> Option<DefaultKey> {
        // FIXME: check device against capabilities.
        self.devices.keys().next()
    }

    /// Get a mutable handle of an image in the pool.
    pub fn entry(&mut self, PoolKey(key): PoolKey) -> Option<PoolImageMut<'_>> {
        Some(PoolImageMut {
            key,
            image: self.items.get_mut(key)?,
            devices: &self.devices,
        })
    }

    /// Gift the pool an image allocated on the host.
    ///
    /// You must describe the texels of the image buffer.
    pub fn insert(&mut self, image: ImageBuffer, descriptor: Descriptor) -> PoolImageMut<'_> {
        // FIXME: check for consistency of buffer and descriptor
        self.new_with_data(ImageData::Host(image), descriptor)
    }

    /// Insert an simple SRGB image into the pool.
    ///
    /// Note that this can not be performed without an allocation since the pool image uses its own
    /// special allocation tactic.
    pub fn insert_srgb(&mut self, image: &image::DynamicImage) -> PoolImageMut<'_> {
        let buffer = ImageBuffer::from(image);
        let descriptor = Descriptor::with_srgb_image(image);
        self.insert(buffer, descriptor)
    }

    /// Create the image based on an entry.
    ///
    /// This allocates a host-accessible buffer with the same layout and metadata as the image. If
    /// possible it will also copy the data from the source entry.  It is always possible to clone
    /// images that have host-allocated data.
    ///
    /// # Panics
    /// This method panics when the key is not valid for the pool.
    pub fn allocate_like(&mut self, key: PoolKey) -> PoolImageMut<'_> {
        let entry = self.entry(key).expect("Not a valid pool key");
        let mut buffer = ImageBuffer::with_layout(entry.layout());
        if let Some(data) = entry.as_bytes() {
            buffer.as_bytes_mut().copy_from_slice(data);
        }
        let texel = entry.image.descriptor.clone();
        self.new_with_data(ImageData::Host(buffer), texel)
    }

    /// Create the descriptor for an image buffer that is provided by the caller.
    ///
    /// # Panics
    /// This method will panic if the layout is inconsistent.
    pub fn declare(&mut self, desc: Descriptor) -> PoolImageMut<'_> {
        assert!(desc.is_consistent(), "{:?}", desc);
        let layout = desc.to_canvas();
        self.new_with_data(ImageData::LateBound(layout), desc)
    }

    pub(crate) fn insert_cacheable_texture(
        &mut self,
        desc: &TextureDescriptor,
        data: wgpu::Texture,
    ) -> TextureKey {
        let gpu = GpuKey(slotmap::KeyData::from_ffi(0).into());
        let key = self.textures.insert((desc.clone(), gpu, data));
        TextureKey(key)
    }

    pub(crate) fn insert_cacheable_buffer(
        &mut self,
        desc: &BufferDescriptor,
        data: wgpu::Buffer,
    ) -> BufferKey {
        let gpu = GpuKey(slotmap::KeyData::from_ffi(0).into());
        let key = self.buffers.insert((desc.clone(), gpu, data));
        BufferKey(key)
    }

    pub(crate) fn reassign_texture_gpu_unguarded(&mut self, key: TextureKey, gpu: GpuKey) {
        if let Some((_, old_gpu, _)) = self.textures.get_mut(key.0) {
            *old_gpu = gpu;
        }
    }

    pub(crate) fn reassign_buffer_gpu_unguarded(&mut self, key: BufferKey, gpu: GpuKey) {
        if let Some((_, old_gpu, _)) = self.buffers.get_mut(key.0) {
            *old_gpu = gpu;
        }
    }

    /// Iterate over all entries in the pool.
    pub fn iter(&self) -> Iter<'_> {
        Iter {
            inner: self.items.iter(),
        }
    }

    /// Iterate over all entries in the pool.
    pub fn iter_mut(&mut self) -> IterMut<'_> {
        IterMut {
            inner: self.items.iter_mut(),
            devices: &mut self.devices,
        }
    }

    /// Discard all cached textures, buffers, etc.
    pub fn clear_cache(&mut self) {
        self.buffers.clear();
        self.textures.clear();
    }

    pub(crate) fn as_cache(&mut self, gpu: GpuKey) -> Cache<'_> {
        let mut buffer_sets = HashMap::<_, Vec<_>>::new();
        let mut texture_sets = HashMap::<_, Vec<_>>::new();

        for (key, (descriptor, gpu_key, _)) in self.buffers.iter() {
            if gpu_key.0 != gpu.0 {
                continue;
            }

            buffer_sets
                .entry(descriptor.clone())
                .or_default()
                .push(BufferKey(key));
        }

        for (key, (descriptor, gpu_key, _)) in self.textures.iter() {
            if gpu_key.0 != gpu.0 {
                continue;
            }

            texture_sets.entry(descriptor.clone())
                .or_default()
                .push(PoolKey(key));
        }

        Cache {
            buffer_sets,
            texture_sets,
            pool: self,
        }
    }

    fn new_with_data(&mut self, data: ImageData, descriptor: Descriptor) -> PoolImageMut<'_> {
        let key = self.items.insert(Image {
            meta: ImageMeta::default(),
            data,
            descriptor,
        });

        PoolImageMut {
            key,
            image: &mut self.items[key],
            devices: &self.devices,
        }
    }
}

impl ImageData {
    pub(crate) fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            ImageData::Host(ref buffer) => Some(buffer.as_bytes()),
            _ => None,
        }
    }

    pub(crate) fn as_bytes_mut(&mut self) -> Option<&mut [u8]> {
        match self {
            ImageData::Host(ref mut buffer) => Some(buffer.as_bytes_mut()),
            _ => None,
        }
    }

    pub(crate) fn layout(&self) -> &BufferLayout {
        match self {
            ImageData::Host(canvas) => canvas.layout(),
            ImageData::Gpu { layout, .. } => layout,
            ImageData::GpuTexture { layout, .. } => layout,
            ImageData::LateBound(layout) => layout,
        }
    }

    pub(crate) fn host_allocate(&mut self) -> Self {
        let buffer = ImageBuffer::with_layout(self.layout());
        mem::replace(self, ImageData::Host(buffer))
    }

    pub(crate) fn gpu_texture(&mut self) -> Option<wgpu::Texture> {
        let late = ImageData::LateBound(self.layout().clone());
        match mem::replace(self, late) {
            ImageData::GpuTexture { texture, .. } => Some(texture),
            other => {
                *self = other;
                None
            }
        }
    }
}

impl PoolImage<'_> {
    pub fn to_image(&self) -> Option<image::DynamicImage> {
        let data = self.as_bytes()?;
        let layout = self.layout();
        let descriptor = &self.image.descriptor;

        let image = Descriptor::as_image_allocator(&descriptor.texel)?;
        let image = image(layout.width(), layout.height(), data)?;
        Some(image)
    }

    pub fn key(&self) -> PoolKey {
        PoolKey(self.key)
    }

    pub fn layout(&self) -> &BufferLayout {
        self.image.data.layout()
    }

    /// The full descriptor for this image.
    ///
    /// This is only available if a valid `Texel` descriptor has been configured.
    pub fn descriptor(&self) -> Descriptor {
        // TODO: return reference?
        self.image.descriptor.clone()
    }

    /// View the buffer as bytes.
    ///
    /// This return `Some` if the image is a host allocated buffer and `None` otherwise.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        self.image.data.as_bytes()
    }
}

impl PoolImageMut<'_> {
    /// Get the key associated with the image.
    ///
    /// You can use the key to access this same image again.
    pub fn key(&self) -> PoolKey {
        PoolKey(self.key)
    }

    /// Get the buffer layout describing the byte occupancy.
    pub fn layout(&self) -> &BufferLayout {
        self.image.data.layout()
    }

    /// The full descriptor for this image.
    ///
    /// This is only available if a valid `Texel` descriptor has been configured.
    pub fn descriptor(&self) -> Descriptor {
        self.image.descriptor.clone()
    }

    /// View the buffer as bytes.
    ///
    /// This return `Some` if the image is a host allocated buffer and `None` otherwise.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        self.image.data.as_bytes()
    }

    /// View the buffer as bytes.
    ///
    /// This return `Some` if the image is a host allocated buffer and `None` otherwise.
    pub fn as_bytes_mut(&mut self) -> Option<&mut [u8]> {
        self.image.data.as_bytes_mut()
    }

    /// Configure the color of this image, not changing any data.
    pub fn set_color(&mut self, color: Color) {
        // FIXME: re-add assert?
        // let parts = self.image.descriptor.texel.parts;
        // assert!(color.is_consistent(parts));
        self.image.descriptor.color = color;
    }

    /// Replace this image with a descriptor for an image buffer that is provided by the caller.
    ///
    /// # Panics
    /// This method will panic if the layout is inconsistent.
    pub fn declare(&mut self, descriptor: Descriptor) {
        let layout = descriptor.to_canvas();
        self.image.descriptor = descriptor;
        self.image.data = ImageData::LateBound(layout);
    }

    /// Replace this image with host allocated data, changing the layout.
    pub fn set_srgb(&mut self, image: &image::DynamicImage) {
        let buffer = ImageBuffer::from(image);
        let descriptor = Descriptor::with_srgb_image(image);
        self.image.descriptor = descriptor;
        self.image.data = ImageData::Host(buffer);
    }

    /// Insert the texture instead of the current image.
    ///
    /// A replacement with the same format is allocated instead.
    /// # Panics
    ///
    /// This may panic later if the texture is not from the same gpu device as used by the pool, or
    /// if the texture does not fit with the layout.
    pub fn replace_texture_unguarded(&mut self, texture: &mut wgpu::Texture, GpuKey(gpu): GpuKey) {
        let layout = self.layout().clone();

        let ttexture = texture;
        let tgpu = gpu;

        if let ImageData::GpuTexture {
            texture,
            layout: _,
            gpu,
        } = &mut self.image.data
        {
            mem::swap(ttexture, texture);
            *gpu = tgpu;
        }

        let mut replace;
        match self.devices.get(tgpu) {
            None => return,
            Some(gpu) => {
                replace = gpu.device.create_texture(&wgpu::TextureDescriptor {
                    label: None,
                    size: wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::R8Unorm,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                });

                mem::swap(&mut replace, ttexture);
            }
        }

        self.image.data = ImageData::GpuTexture {
            texture: replace,
            layout,
            gpu,
        };
    }

    /// Get the metadata associated with the entry.
    pub(crate) fn meta(&self) -> &ImageMeta {
        &self.image.meta
    }

    pub(crate) fn data(&self) -> &ImageData {
        &self.image.data
    }

    /// Replace the data with a host allocated buffer of the correct layout.
    /// Returns the previous image data.
    /// TODO: figure out if we should expose this..
    pub(crate) fn host_allocate(&mut self) -> ImageData {
        self.image.data.host_allocate()
    }

    /// Make a copy of this host accessible image as a host allocated image.
    pub(crate) fn host_copy(&self) -> Option<ImageBuffer> {
        let data = self.as_bytes()?;
        let mut buffer = ImageBuffer::with_layout(self.layout());
        buffer.as_bytes_mut().copy_from_slice(data);
        Some(buffer)
    }

    /// TODO: figure out if assert/panicking is ergonomic enough for making it pub.
    /// FIXME: ignores reference to GPU or others to this pool's other resources.
    pub(crate) fn swap(&mut self, image: &mut ImageData) {
        assert_eq!(self.image.data.layout(), image.layout());
        // FIXME: When we are doing this should we temporarily assign a 'dangling' key
        // (DefaultKey::null) as the gpu is only fixed later in `finish`. In particular, if
        // this is *not* the same buffer we retrieved input images from then the key may refer
        // to a different device which can confusingly error later.
        // For now, the device is not critically relevant and we assume proper usage..
        mem::swap(&mut self.image.data, image)
    }

    /// If this image is not read on the host (as determined by meta) then execute a swap.
    /// Otherwise try to perform a copy. Returns if the transaction succeeded.
    pub(crate) fn trade(&mut self, image: &mut ImageData) -> bool {
        if self.meta().no_read {
            self.swap(image);
            true
        } else if let Some(copy) = self.host_copy() {
            // TODO: this variant _mighty_ be able to re-use existing buffer in `image`.
            *image = ImageData::Host(copy);
            true
        } else {
            false
        }
    }
}

impl Cache<'_> {
    // FIXME: what about buffer_init? Avoid allocation? Only if buffer is write-once?

    pub(crate) fn extract_texture(&mut self, desc: &TextureDescriptor) -> Option<wgpu::Texture> {
        let PoolKey(key) = self.texture_sets.get_mut(desc)?.pop()?;
        let (_, _, texture) = self.pool.textures.remove(key)?;
        Some(texture)
    }

    pub(crate) fn extract_buffer(&mut self, desc: &BufferDescriptor) -> Option<wgpu::Buffer> {
        let BufferKey(key) = self.buffer_sets.get_mut(desc)?.pop()?;
        let (_, _, buffer) = self.pool.buffers.remove(key)?;
        Some(buffer)
    }
}

impl<'pool> From<PoolImageMut<'pool>> for PoolImage<'pool> {
    fn from(img: PoolImageMut<'pool>) -> Self {
        PoolImage {
            key: img.key,
            image: img.image,
        }
    }
}

impl<'pool> Iterator for Iter<'pool> {
    type Item = PoolImage<'pool>;
    fn next(&mut self) -> Option<Self::Item> {
        let (key, image) = self.inner.next()?;
        Some(PoolImage { key, image })
    }
}

impl<'pool> Iterator for IterMut<'pool> {
    type Item = PoolImageMut<'pool>;
    fn next(&mut self) -> Option<Self::Item> {
        let (key, image) = self.inner.next()?;
        let devices = self.devices;
        Some(PoolImageMut {
            key,
            image,
            devices,
        })
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

impl fmt::Debug for ImageData {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ImageData::LateBound(layout) => write!(f, "ImageData::LayoutBound({:?})", layout),
            ImageData::Host(buffer) => write!(f, "ImageData::Host({:?})", buffer.layout()),
            ImageData::GpuTexture { layout, .. } => {
                write!(f, "ImageData::GpuTexture({:?})", layout)
            }
            ImageData::Gpu { layout, .. } => write!(f, "ImageData::GpuBuffer({:?})", layout),
        }
    }
}
