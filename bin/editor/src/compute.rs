use std::future::Future;
use std::sync::Arc;

use crate::surface::Surface;
use arc_swap::ArcSwapAny;

use zosimos::command::{CommandBuffer, Register};
use zosimos::pool::{GpuKey, Pool, SwapChain};

/// A compute graph.
pub struct Compute {
    /// The pool we use for computing.
    pool: Pool,
    key: GpuKey,
    program: ArcSwapAny<Arc<Program>>,
    dirty: bool,
    swap: SwapChain,
}

struct Program {
    commands: Option<Arc<ComputeCommands>>,
    have: u64,
}

/// Represents a (potentially past) snapshot of the program's command buffer.
#[derive(Default)]
pub struct ComputeTailCommands {
    pub have: u64,
    pub commands: Option<Arc<ComputeCommands>>,
}

pub struct ComputeCommands {
    pub buffer: CommandBuffer,
    pub registers: Vec<Register>,
}

impl Compute {
    /// Create a compute graph supplying to a surface.
    pub fn new(surface: &mut Surface) -> Compute {
        let mut pool = Pool::new();

        let key = surface.configure_pool(&mut pool);
        let swap = surface.configure_swap_chain(2);

        let program = Arc::new(Program {
            commands: None,
            have: 0,
        });

        let program = ArcSwapAny::new(program);

        Compute {
            pool,
            key,
            program,
            dirty: false,
            swap,
        }
    }

    pub fn acquire(&mut self, ctr: &mut ComputeTailCommands) -> bool {
        let load = self.program.load();
        let update = load.have != ctr.have;

        if !update {
            return false;
        }

        ctr.commands = load.commands.clone();
        self.dirty |= update;
        update
    }

    /// Spawn a task to maintain derivations.
    pub fn run(&mut self) -> Box<dyn Future<Output = ()> + 'static> {
        let program = self.program.load().clone();
        let Some(commands) = &program.commands else {
            return Box::new(core::future::ready(()));
        };

        let Ok(program) = commands.buffer.compile() else {
            return Box::new(core::future::ready(()));
        };

        let mut launcher = program.launch(&mut self.pool);
        for &input in &commands.registers {
            let _key = todo!();
            launcher.bind(input, _key);
        }

        todo!()
    }
}
