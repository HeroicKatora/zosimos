pub mod stage;

/// A vertex box shader, rendering a sole quad with given vertex and uv coordinate system.
pub const VERT_NOOP: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spirv/box.vert.v"));

/// A 'noop' copy from the sampled texture to the output color based on the supplied UVs.
pub const FRAG_COPY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spirv/copy.frag.v"));
pub const FRAG_MIX_RGBA: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spirv/inject.frag.v"));
/// A full color conversion, exact semantics documented in the `stage` module.
pub const FRAG_CONVERT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spirv/stage.frag.v"));
/// a linear transformation on rgb color.
pub const FRAG_LINEAR: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spirv/linear.frag.v"));

/// Push constants required in different shader invocations.
pub enum Invoke {
    MixRgba {
        /// Mix-factors of all color channels.
        rgba: [f32; 4],
    },
}
