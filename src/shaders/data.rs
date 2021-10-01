use std::collections::HashMap;
use bytemuck::Pod;

use crate::program::BufferInitContent;
use crate::types::Static;

#[derive(Default, Clone)]
pub struct ShaderData {
    pub(crate) binary: Vec<u8>,
    pub(crate) location: HashMap<Static, usize>,
}

impl ShaderData {
    pub(crate) fn add(&mut self, data: &[impl Pod]) -> BufferInitContent {
        BufferInitContent::new(&mut self.binary, data)
    }
}
