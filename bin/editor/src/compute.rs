use std::future::Future;
use std::sync::Arc;

use crate::surface::Surface;
use arc_swap::ArcSwapAny;

use stealth_paint::command::CommandBuffer;
use stealth_paint::pool::{GpuKey, Pool};

/// A compute graph.
pub struct Compute {
    pool: Pool,
    key: GpuKey,
    program: ArcSwapAny<Arc<Program>>,
}

struct Program {
    commands: Option<Arc<CommandBuffer>>,
    have: u64,
}

/// Represents a (potentially past) snapshot of the program's command buffer.
#[derive(Default)]
pub struct ComputeTailCommands {
    pub have: u64,
    pub commands: Option<Arc<CommandBuffer>>,
}

impl Compute {
    /// Create a compute graph supplying to a surface.
    pub fn new(surface: &mut Surface) -> Compute {
        let mut pool = Pool::new();
        let key = surface.configure_pool(&mut pool);

        let program = Arc::new(Program {
            commands: None,
            have: 0,
        });

        let program = ArcSwapAny::new(program);
        Compute { pool, key, program }
    }

    pub fn acquire(&self, ctr: &mut ComputeTailCommands) -> bool {
        let load = self.program.load();
        let update = load.have != ctr.have;
        ctr.commands = load.commands.clone();
        update
    }

    /// Spawn a task to maintain derivations.
    pub fn run(&self) -> Box<dyn Future<Output = ()> + 'static> {
        todo!()
    }
}
