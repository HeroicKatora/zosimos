//! This tests re-running programs.
//! Loop de Loop.
#[path = "util.rs"]
mod util;

use zosimos::command::{self, CommandBuffer};
use zosimos::pool::Pool;
use zosimos::program::{Capabilities, Program};

use self::util::{retire_with_one_image, run_executable_with_output};

const BACKGROUND: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/input/background.png");
const FOREGROUND: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/input/foreground.png");

/// Integration test the whole pipeline against reference images.
#[test]
fn integration() {
    env_logger::init();

    const ANY: wgpu::Backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: ANY,
        ..Default::default()
    });

    let adapter = Program::minimum_adapter(instance.enumerate_adapters(ANY).into_iter())
        .expect("to get an adapter");

    let background = image::open(BACKGROUND).expect("Background image opened");
    let foreground = image::open(FOREGROUND).expect("Background image opened");

    let mut pool = Pool::new();
    let (bg_key, background) = {
        let entry = pool.insert_srgb(&background);
        (entry.key(), entry.descriptor())
    };

    let (fg_key, foreground) = {
        let entry = pool.insert_srgb(&foreground);
        (entry.key(), entry.descriptor())
    };

    pool.request_device(&adapter, Program::minimal_device_descriptor())
        .expect("to get a device");

    /* This is the familiar simple inscribe placement
     */
    let mut commands = CommandBuffer::default();

    let placement = command::Rectangle {
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
    let capabilities = Capabilities::from({
        let mut devices = pool.iter_devices();
        devices.next().expect("the pool to contain a device")
    });

    let executable = plan
        .lower_to(capabilities)
        .expect("No extras beyond device required");

    let mut result = bg_key;
    // At the time of writing (2021-Sep) we get around 240 fps with this.
    // That's rookie numbers, gotta pump those numbers up. 16% spent in memmove..
    for _ in 0..200 {
        result = run_executable_with_output(
            &executable,
            &mut pool,
            vec![(background, bg_key), (foreground, fg_key)],
            [],
            retire_with_one_image(output),
        );
    }

    let image = pool.entry(result).unwrap();
    util::assert_reference(image.into(), "composed.crc.png");
}
