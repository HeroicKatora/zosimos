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
