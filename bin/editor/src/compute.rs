mod not_wren;

use std::future::Future;

use crate::surface::Surface;
use stealth_paint::pool::{GpuKey, Pool};

/// A compute graph.
pub struct Compute {
    pool: Pool,
    key: GpuKey,
}

impl Compute {
    /// Create a compute graph supplying to a surface.
    pub fn new(surface: &mut Surface) -> Compute {
        let mut pool = Pool::new();
        let key = surface.configure_pool(&mut pool);
        Compute { pool, key }
    }

    /// Spawn a task to maintain derivations.
    pub fn run(&self) -> Box<dyn Future<Output=()> + 'static> {
        todo!()
    }
}
