//! This file checks various basic aspects of buffer operations.
#[path = "util.rs"]
mod util;

use zosimos::buffer::Descriptor;
use zosimos::command::CommandBuffer;
use zosimos::pool::Pool;
use zosimos::program::Program;

use self::util::{retire_with_one_image, run_once_with_output};

// const BACKGROUND: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/input/background.png");
// const FOREGROUND: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/input/foreground.png");

#[test]
fn buffer_interop() {
    env_logger::init();

    const ANY: wgpu::Backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: ANY,
        ..Default::default()
    });

    let adapter = Program::request_adapter(&instance).expect("to get an adapter");

    let mut pool = Pool::new();

    pool.request_device(&adapter, Program::minimal_device_descriptor())
        .expect("to get a device");

    run_from_buffer(&mut pool);
}

fn run_from_buffer(pool: &mut Pool) {
    let mut commands = CommandBuffer::default();

    let descriptor = Descriptor::with_srgb_image(&image::DynamicImage::new_luma8(2, 2));

    // So. What is going on here, why is this a description for a 2x2 image?
    //
    // It's an implementation detail and not necessarily intended to work this way. First note that
    // all buffers we want to copy must be aligned to the buffer copy size which is `4`. Then, the
    // layout of a buffer on the GPU is texel-row-by-row where the row itself is also highly
    // aligned, to 256 to be exact. Hence, the first two pixels here occupy two bytes but the first
    // row occupies 256. We must also pad the second row to a multiple of the copy size.
    let mut a = [0xff; 260];
    a[256] = 0x00;
    let buffer = commands.buffer_init(&a);

    let result = commands
        .from_buffer(buffer, descriptor)
        .expect("Buffer valid for this image descriptor");

    let (output, _outformat) = commands.output(result).expect("Valid for output");

    let result = run_once_with_output(commands, pool, vec![], retire_with_one_image(output));

    let image = pool.entry(result).unwrap();
    util::assert_reference(image.into(), "from_buffer.crc.png");
}
