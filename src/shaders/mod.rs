pub const VERT_NOOP: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spirv/noop.vert.v"));

pub const FRAG_COPY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spirv/copy.frag.v"));
