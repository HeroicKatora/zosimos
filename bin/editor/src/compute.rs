use std::collections::HashMap;
use std::future::Future;
use std::sync::{mpsc, Arc};

use crate::surface::Surface;
use arc_swap::ArcSwapAny;

use zosimos::buffer::Descriptor;
use zosimos::command::{CommandBuffer, Register};
use zosimos::pool::{GpuKey, Pool, PoolBridge, PoolKey, SwapChain};
use zosimos::run::StepLimits;

/// A compute graph.
pub struct Compute {
    /// The pool we use for computing.
    pool: Pool,
    /// The bridge between the surface and compute pools.
    bridge: PoolBridge,
    gpu: GpuKey,
    /// Descriptors for each in the swap chain.
    render_target: Vec<PoolKey>,
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

struct Exit {
    pool: Pool,
    bridge: PoolBridge,
    /// The key of the target buffer in the `Compute::pool`.
    render_target: PoolKey,
    /// The key of the buffer we rendered to, in the `Exit::pool`.
    render_exec: PoolKey,
}

impl Compute {
    /// Create a compute graph supplying to a surface.
    pub fn new(surface: &mut Surface) -> Compute {
        let mut pool = Pool::new();

        let adapter = surface.adapter();
        let (gpu, bridge) = surface.configure_pool(&mut pool);
        let swap = surface.configure_swap_chain(2);

        // Just allocate a buffer. We'll change the details later to align with the swap chain.
        let render_target = swap
            .empty
            .iter()
            .map(|_| {
                pool.declare(Descriptor::with_srgb_image(
                    &image::DynamicImage::new_luma_a8(0, 0),
                ))
                .key()
            })
            .collect();

        let program = Arc::new(Program {
            commands: None,
            have: 0,
        });

        let program = ArcSwapAny::new(program);
        let (exit_send, exit) = mpsc::channel();

        Compute {
            pool,
            bridge,
            gpu,
            render_target,
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

        let &[out_reg] = &commands.out_registers[..] else {
            log::warn!("Only support compute with precisely one output register");
            return Box::new(core::future::ready(()));
        };

        let Ok(program) = commands.buffer.compile() else {
            return Box::new(core::future::ready(()));
        };

        let Some(render_target) = self.render_target.pop() else {
            return Box::new(core::future::ready(()));
        };

        let Some(render_desc) = program.describe_register(out_reg) else {
            log::warn!("Failed to describe output register, this is a bug");
            return Box::new(core::future::ready(()));
        };

        let mut pool = Pool::new();
        let mut bridge = PoolBridge::default();
        bridge.share_device(&self.pool, self.gpu, &mut pool);

        let mut bindings = HashMap::new();
        for (&reg, &key) in &self.bindings {
            let from_image = self.pool.entry(key).unwrap();
            let run_image = pool.declare(from_image.descriptor());
            let key = run_image.key();
            bridge.swap_image(from_image, run_image);
            bindings.insert(reg, key);
        }

        let render_exec = {
            let mut from_image = self
                .pool
                .entry(render_target)
                .expect("Registered target descriptor");

            let upload = from_image.set_texture(self.gpu, &render_desc);

            if upload.is_err() {
                log::warn!("Failed to allocate output texture on GPU");
                return Box::new(core::future::ready(()));
            }

            let run_image = pool.declare(from_image.descriptor());
            let key = run_image.key();

            bridge.swap_image(from_image, run_image);
            bindings.insert(out_reg, key);

            key
        };

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

            let _ = exit_send.send(Exit {
                pool,
                bridge,
                render_target,
                render_exec,
            });
        })
    }

    /// Collect complete computations, update the surface.
    pub fn reap(&mut self, surface: &mut Surface) {
        let Ok(mut exit) = self.exit.try_recv() else {
            surface.set_from_swap_chain(&mut self.swap_in_surface_pool);
            return;
        };

        let Some(present_target) = self.swap_in_surface_pool.empty.pop_back() else {
            surface.set_from_swap_chain(&mut self.swap_in_surface_pool);
            return;
        };

        let render_exec = exit
            .pool
            .entry(exit.render_exec)
            .expect("Render target not in pool");
        let render_target = self
            .pool
            .entry(exit.render_target)
            .expect("Render target not in compute");

        exit.bridge.swap_image(render_exec, render_target);

        let render_target = self
            .pool
            .entry(exit.render_target)
            .expect("Render target not in compute");
        surface.swap_into(render_target, present_target, &self.bridge);
        self.swap_in_surface_pool.full.push_back(present_target);

        surface.set_from_swap_chain(&mut self.swap_in_surface_pool);
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
