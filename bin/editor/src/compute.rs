use std::collections::HashMap;
use std::future::Future;
use std::sync::{mpsc, Arc};

use crate::surface::Surface;
use arc_swap::ArcSwapAny;

use zosimos::command::{CommandBuffer, Register};
use zosimos::pool::{GpuKey, Pool, PoolBridge, PoolKey, SwapChain};
use zosimos::run::StepLimits;

/// A compute graph.
pub struct Compute {
    /// The pool we use for computing.
    pool: Pool,
    key: GpuKey,
    adapter: Arc<wgpu::Adapter>,
    program: ArcSwapAny<Arc<Program>>,
    bindings: HashMap<Register, PoolKey>,
    swap_in_surface_pool: SwapChain,
    exit_send: mpsc::Sender<Exit>,
    exit: mpsc::Receiver<Exit>,
    dirty: bool,
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
    pub in_registers: Vec<Register>,
    pub out_registers: Vec<Register>,
}

#[derive(Default)]
struct Exit {
    pool: Pool,
}

impl Compute {
    /// Create a compute graph supplying to a surface.
    pub fn new(surface: &mut Surface) -> Compute {
        let mut pool = Pool::new();

        let adapter = surface.adapter();
        let key = surface.configure_pool(&mut pool);
        let swap = surface.configure_swap_chain(2);

        let program = Arc::new(Program {
            commands: None,
            have: 0,
        });

        let program = ArcSwapAny::new(program);
        let (exit_send, exit) = mpsc::channel();

        Compute {
            pool,
            key,
            adapter,
            program,
            bindings: Default::default(),
            dirty: false,
            exit_send,
            exit,
            swap_in_surface_pool: swap,
        }
    }

    pub fn insert_image(&mut self, image: &image::DynamicImage) -> PoolKey {
        self.dirty = false;
        self.pool.insert_srgb(image).key()
    }

    pub fn bind(&mut self, reg: Register, key: PoolKey) {
        self.dirty = false;
        self.bindings.insert(reg, key);
    }

    pub fn bindings_clear(&mut self) {
        self.dirty = false;
        self.bindings.clear();
    }

    /// Update the program to compute from inputs.
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
    pub fn run(&mut self) -> Box<dyn Future<Output = ()> + Send + 'static> {
        let program = self.program.load().clone();
        let exit_send = self.exit_send.clone();
        let Some(commands) = &program.commands else {
            return Box::new(core::future::ready(()));
        };

        let Ok(program) = commands.buffer.compile() else {
            return Box::new(core::future::ready(()));
        };

        let mut pool = Pool::new();
        let mut bridge = PoolBridge::default();
        bridge.share_device(&self.pool, self.key, &mut pool);

        let mut bindings = HashMap::new();
        for (&reg, &key) in &self.bindings {
            let from_image = self.pool.entry(key).unwrap();
            let run_image = pool.declare(from_image.descriptor());
            let key = run_image.key();
            bridge.swap_image(from_image, run_image);
            bindings.insert(reg, key);
        }

        // Move resource into temporary running pool.
        let mut launcher = program.launch(&mut pool);
        for (&reg, &key) in &bindings {
            launcher = launcher.bind(reg, key).unwrap();
        }

        let exec = launcher.launch(&self.adapter).unwrap();

        Box::new(async move {
            let mut exec = exec;

            while exec.is_running() {
                let Ok(mut point) = exec.step_to(StepLimits::new().with_steps(32)) else {
                    break;
                };

                let Ok(()) = point.block_on() else {
                    break;
                };
            }

            let mut retire = exec.retire_gracefully(&mut pool);
            for (&reg, _) in &bindings {
                retire.input(reg).unwrap();
            }

            let _ = exit_send.send(Exit { pool });
        })
    }
}

fn spin_block<R>(mut f: impl core::future::Future<Output = R>) -> R {
    use core::hint::spin_loop;
    use core::task::{Context, Poll};

    let mut f = core::pin::pin!(f);

    // create the context
    let waker = waker_fn::waker_fn(|| {});
    let mut ctx = Context::from_waker(&waker);

    // poll future in a loop
    loop {
        match f.as_mut().poll(&mut ctx) {
            Poll::Ready(o) => return o,
            Poll::Pending => spin_loop(),
        }
    }
}

impl crate::winit::ScopedCompute for Compute {
    fn maybe_launch<'a>(&mut self, scope: &'a std::thread::Scope<'a, '_>) {
        if !self.dirty {
            return;
        }

        log::info!("Running compute due to changes");
        self.dirty = false;
        let future = self.run();
        scope.spawn(move || {
            spin_block(Box::into_pin(future));
        });
    }
}
