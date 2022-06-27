use crate::surface::Surface;

use stealth_paint::pool::Pool;

pub struct Compute {
    pool: Pool,
}

impl Compute {
    pub fn new(surface: &Surface) -> Compute {
        let mut pool = Pool::new();
        surface.configure_pool(&mut pool);
        Compute { pool }
    }
}
