/// 

/// Defines one transfer function.
pub struct Transfer(u32);

pub struct Primaries {
}

/// A padding-free, well laid-out struct to bind to the fragment stage.
pub struct Stage {
    pub transfer: Transfer,
}

impl Transfer {
}
