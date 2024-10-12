//! This test ensures that the direct `Program::launch` interface can be used.
use zosimos::buffer::Descriptor;
use zosimos::command::{CommandBuffer, Rectangle};
use zosimos::pool::{Pool, PoolKey};

#[path = "util.rs"]
mod util;

const BACKGROUND: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/input/background.png");
const FOREGROUND: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/input/foreground.png");

#[test]
fn standard() {
    env_logger::init();

    const ANY: wgpu::Backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: ANY,
        ..Default::default()
    });

    let background = image::open(BACKGROUND).expect("Background image opened");
    let foreground = image::open(FOREGROUND).expect("Background image opened");

    let mut pool = Pool::new();
    let pool_background = {
        let entry = pool.insert_srgb(&background);
        (entry.key(), entry.descriptor())
    };

    let pool_foreground = {
        let entry = pool.insert_srgb(&foreground);
        (entry.key(), entry.descriptor())
    };

    run_blending(
        &mut pool,
        instance.enumerate_adapters(ANY).into_iter(),
        pool_foreground,
        pool_background,
    );
}

fn run_blending(
    pool: &mut Pool,
    adapters: impl Iterator<Item = wgpu::Adapter>,
    (fg_key, foreground): (PoolKey, Descriptor),
    (bg_key, background): (PoolKey, Descriptor),
) {
    let mut commands = CommandBuffer::default();

    let placement = Rectangle {
        x: 0,
        y: 0,
        max_x: foreground.layout.width,
        max_y: foreground.layout.height,
    };

    // Describe the pipeline:
    // 0: in (background)
    // 1: in (foreground)
    // 2: inscribe(0, placement, 1)
    // 3: out(2)
    let background = commands.input(background).unwrap();
    let foreground = commands.input(foreground).unwrap();

    let result = commands
        .inscribe(background, placement, foreground)
        .expect("Valid to inscribe");

    let (output, _outformat) = commands.output(result).expect("Valid for output");

    let plan = commands.compile().expect("Could build command buffer");
    let adapter = plan
        .choose_adapter(adapters)
        .expect("Did not find any adapter for executing the blend operation");

    let mut execution = plan
        .launch(pool)
        .bind(background, bg_key)
        .unwrap()
        .bind(foreground, fg_key)
        .unwrap()
        .launch(&adapter)
        .expect("Launching failed");

    while execution.is_running() {
        let _wait_point = execution.step().expect("Shouldn't fail but");
    }

    let mut retire = execution.retire_gracefully(pool);

    let image = retire.output(output).expect("A valid image output");
    util::assert_reference(image, "composed.crc.png");
}
