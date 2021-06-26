pub mod stage;

pub const VERT_NOOP: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spirv/noop.vert.v"));

pub const FRAG_COPY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spirv/copy.frag.v"));
pub const FRAG_MIX_RGBA: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spirv/inject.frag.v"));
pub const FRAG_CONVERT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spirv/stage.frag.v"));

/// Push constants required in different shader invocations.
pub enum Invoke {
    MixRgba {
        /// Mix-factors of all color channels.
        rgba: [f32; 4],
    },
}
