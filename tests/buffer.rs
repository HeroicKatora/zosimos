//! This file checks various basic aspects of buffer operations.
#[path = "util.rs"]
mod util;

use zosimos::command::{Bilinear, CommandBuffer, RegisterKnob};
use zosimos::pool::Pool;
use zosimos::program::Program;
use zosimos::{buffer::Descriptor, program::Capabilities};

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

    run_from_buffer_knob(&mut pool);

    run_bilinear(&mut pool);
}

fn run_from_buffer(pool: &mut Pool) {
    let mut commands = CommandBuffer::default();

    let descriptor = Descriptor::with_srgb_image(&image::DynamicImage::new_luma8(8, 8));

    // So. What is going on here, why is this a description for a 8x8 image?
    //
    // It's an implementation detail and not necessarily intended to work this way. First note that
    // all buffers we want to copy must be aligned to the buffer copy size which is `4`. Then, the
    // layout of a buffer on the GPU is texel-row-by-row where the row itself is also highly
    // aligned, to 256 to be exact. Hence, the first two pixels here occupy two bytes but the first
    // row occupies 256. We must also pad the second row to a multiple of the copy size.
    let mut a = [0xff; 8 * 256];
    a[256..][..8].copy_from_slice(&[0x00; 8]);
    let buffer = commands.buffer_init(&a);

    let result = commands
        .from_buffer(buffer, descriptor)
        .expect("Buffer valid for this image descriptor");

    let (output, _outformat) = commands.output(result).expect("Valid for output");

    let result = run_once_with_output(commands, pool, vec![], retire_with_one_image(output));

    let image = pool.entry(result).unwrap();
    util::assert_reference(image.into(), "from_buffer.crc.png");
}

fn run_from_buffer_knob(pool: &mut Pool) {
    let mut commands = CommandBuffer::default();

    let descriptor = Descriptor::with_srgb_image(&image::DynamicImage::new_luma8(8, 8));

    let a = [0xff; 8 * 256];
    let buffer = commands
        .with_knob()
        .buffer_init(&a)
        .expect("Valid for knob");

    let result = commands
        .from_buffer(buffer, descriptor)
        .expect("Buffer valid for this image descriptor");

    let (output, _outformat) = commands.output(result).expect("Valid for output");

    let executable = {
        let plan = commands.compile().expect("Could build command buffer");

        let capabilities = Capabilities::from({
            let mut devices = pool.iter_devices();
            devices.next().expect("the pool to contain a device")
        });

        plan.lower_to(capabilities)
            .expect("No extras beyond device required")
    };

    let knob = executable
        .query_knob(RegisterKnob {
            link_idx: 0,
            register: buffer,
        })
        .unwrap();

    // Patch up the knob so that the result is the same as in the reference test.
    let mut a = a;
    a[256..][..8].copy_from_slice(&[0x00; 8]);

    let result = util::run_executable_with_output(
        &executable,
        pool,
        vec![],
        vec![(knob, &a[..])],
        retire_with_one_image(output),
    );

    let image = pool.entry(result).unwrap();
    util::assert_reference(image.into(), "from_buffer-with-knob.crc.png");
}

fn run_bilinear(pool: &mut Pool) {
    let mut commands = CommandBuffer::default();

    let descriptor = Descriptor::with_srgb_image(&image::DynamicImage::new_rgba8(256, 256));

    let a: [[f32; 4]; 6] = [
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.7, 1.0],
        [0.0, 0.0, 0.3, 1.0],
        [0.0, 1.0, 0.3, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    let buffer = commands.buffer_init(bytemuck::bytes_of(&a));

    let result = commands
        .with_buffer(buffer)
        .expect("Buffer valid for with_buffer")
        .bilinear(descriptor, Bilinear::default())
        .expect("Buffer valid for this image descriptor");

    let (output, _outformat) = commands.output(result).expect("Valid for output");

    let result = run_once_with_output(commands, pool, vec![], retire_with_one_image(output));

    let image = pool.entry(result).unwrap();
    util::assert_reference(image.into(), "bilinear_from_buffer.crc.png");
}
