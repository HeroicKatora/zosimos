#[path = "util.rs"]
mod util;

use zosimos::buffer::Descriptor;
use zosimos::command::{self, CommandBuffer, Register};
use zosimos::pool::{Gpu, Pool, PoolKey};
use zosimos::program::{Capabilities, Program};
use zosimos::run::{Executable, Retire};

const BACKGROUND: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/input/background.png");
const FOREGROUND: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/input/foreground.png");

#[tokio::test(flavor = "multi_thread")]
async fn step_async() {
    env_logger::init();

    const ANY: wgpu::Backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: ANY,
        ..Default::default()
    });

    let adapter = Program::minimum_adapter(instance.enumerate_adapters(ANY).into_iter())
        .expect("to get an adapter");

    let mut pool = Pool::new();

    let background = image::open(BACKGROUND).expect("Background image opened");
    let foreground = image::open(FOREGROUND).expect("Background image opened");

    let pool_background = {
        let entry = pool.insert_srgb(&background);
        (entry.key(), entry.descriptor())
    };

    let pool_foreground = {
        let entry = pool.insert_srgb(&foreground);
        (entry.key(), entry.descriptor())
    };

    pool.request_device(&adapter, Program::minimal_device_descriptor())
        .expect("to get a device");

    run_affine(&mut pool, pool_foreground.clone(), pool_background.clone()).await;
}

async fn run_affine(
    pool: &mut Pool,
    (fg_key, foreground): (PoolKey, Descriptor),
    (bg_key, background): (PoolKey, Descriptor),
) {
    let mut commands = CommandBuffer::default();

    /* This block is copied from the `blend.rs` test. */
    let affine = command::Affine::new(command::AffineSample::Nearest)
        // Move the foreground center to origin.
        .shift(
            -((foreground.layout.width / 2) as f32),
            -((foreground.layout.height / 2) as f32),
        )
        // Rotate 45°
        .rotate(std::f32::consts::PI / 4.)
        // Move origin to the background center.
        .shift(
            (background.layout.width / 2) as f32,
            (background.layout.height / 2) as f32,
        );

    // Describe the pipeline:
    // 0: in (background)
    // 1: in (foreground)
    // 2: affine(0, affine, 1)
    // 3: out(2)
    let background = commands.input(background).unwrap();
    let foreground = commands.input(foreground).unwrap();

    let result_affine = commands
        .affine(background, affine, foreground)
        .expect("Valid to paint with affine transformation");

    let (output_affine, _outformat) = commands.output(result_affine).expect("Valid for output");

    let result = run_once_with_output(
        commands,
        pool,
        vec![(background, bg_key), (foreground, fg_key)],
        util::retire_with_one_image(output_affine),
    )
    .await;

    let image_affine = pool.entry(result).unwrap();
    // Using the same as synchronous code in blend on purpose. This is a form of consistency check.
    util::assert_reference(image_affine.into(), "affine.crc.png");
}

pub async fn run_once_with_output<T>(
    commands: CommandBuffer,
    pool: &mut Pool,
    binds: impl IntoIterator<Item = (Register, PoolKey)>,
    output: impl FnOnce(&mut Retire) -> T,
) -> T {
    let plan = commands.compile().expect("Could build command buffer");
    let capabilities = Capabilities::from({
        let mut devices = pool.iter_devices();
        devices.next().expect("the pool to contain a device")
    });

    let executable = plan
        .lower_to(capabilities)
        .expect("No extras beyond device required");

    run_executable_with_output(&executable, pool, binds, output).await
}

pub async fn run_executable_with_output<T>(
    executable: &Executable,
    pool: &mut Pool,
    binds: impl IntoIterator<Item = (Register, PoolKey)>,
    output: impl FnOnce(&mut Retire) -> T,
) -> T {
    let mut environment = executable.from_pool(pool).expect("no device found in pool");

    for (target, key) in binds {
        environment.bind(target, key).unwrap();
    }

    let _ = environment.recover_buffers();
    let mut execution = executable.launch(environment).expect("Launching failed");
    // Prepare pool to be clear for cache.
    pool.clear_cache();

    let poll_gpu = |gpu: Gpu| {
        let handle = tokio::task::spawn(async move {
            loop {
                gpu.device().poll(wgpu::Maintain::Poll);
                // The cancellation point!
                tokio::task::yield_now().await;
            }
        })
        .abort_handle();

        struct AbortOnDrop(tokio::task::AbortHandle);

        impl Drop for AbortOnDrop {
            fn drop(&mut self) {
                if !self.0.is_finished() {
                    self.0.abort();
                }
            }
        }

        AbortOnDrop(handle)
    };

    while execution.is_running() {
        let mut syncstep = execution.step().expect("Shouldn't fail but");
        syncstep.finish(poll_gpu).await.expect("Shouldn't fail but");
    }

    let mut retire = execution.retire_gracefully(pool);
    let result = output(&mut retire);
    let _ = retire.retire_buffers();
    retire.finish();
    result
}
