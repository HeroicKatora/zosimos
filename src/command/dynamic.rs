use std::sync::Arc;

use crate::buffer::Descriptor;
use crate::program::BufferInitContent;

/// A custom shader implementing a command.
pub trait ShaderCommand: Send + Sync {
    /// Shared, binary shader SPIR-V source.
    ///
    /// It is more efficient if invocations sharing the same shader source return clones of the
    /// exact same allocated source.
    fn source(&self) -> ShaderSource;

    /// Configure this invocation, such as providing bind buffer data as binary.
    ///
    /// See `ShaderData` for more information. All configuration is performed by calling its
    /// methods, the object is provided by surrounding runtime. You shouldn't depend on the exact
    /// timing of this call relative to other invocations as such ordering may be fragile and
    /// depend on optimization reordering performed during encoding of commands. More specific
    /// guarantees may be provided at a later version of the library.
    fn data(&self, _: ShaderData<'_>) -> Descriptor;

    /// Provide a debug representation.
    fn debug(&self) -> &dyn core::fmt::Debug {
        static REPLACEMENT: &'static str = "No debug data for shader invocation";
        &REPLACEMENT
    }
}

/// Provide the shader source code to be executed.
#[non_exhaustive]
pub enum ShaderSource {
    SpirV(Arc<[u8]>),
}

/// Holds binary representation of a shader's argument.
///
/// This, also, exposes all other publicly available methods to configure the `ShaderInvocation`
/// that will occur when executing the provided shader.
pub struct ShaderData<'lt> {
    pub(super) data_buffer: &'lt mut Vec<u8>,
    /// Which region of the data buffer corresponds to the initializer for the buffer binding.
    /// Is `None` if the shader does not have a buffer binding.
    pub(super) content: &'lt mut Option<BufferInitContent>,
}

impl ShaderData<'_> {
    pub fn set_data(&mut self, data: &[impl bytemuck::Pod]) {
        *self.content = Some(BufferInitContent::new(&mut self.data_buffer, data));
    }
}
