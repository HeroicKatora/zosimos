//! This test verifies the functionality of knobs.
//!
//! Knobs control a portion of a compiled program's binary data such that they can be changed
//! without going through a recompilation, linking, lowering pipeline.
#[path = "util.rs"]
mod util;

use zosimos::buffer::Descriptor;
use zosimos::command::{self, CommandBuffer};
use zosimos::pool::{Pool, PoolKey};
use zosimos::program::{Capabilities, Program};

use self::util::{retire_with_one_image, run_executable_with_output};

const BACKGROUND: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/input/background.png");
const FOREGROUND: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/input/foreground.png");

#[test]
fn knob_on_overlay() {
    env_logger::init();

    const ANY: wgpu::Backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: ANY,
        ..Default::default()
    });

    let adapter = Program::request_adapter(&instance).expect("to get an adapter");

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

    pool.request_device(&adapter, Program::minimal_device_descriptor())
        .expect("to get a device");

    run_bilinear(
        &mut pool,
        pool_foreground.clone(),
        pool_background.clone(),
        &[
            command::Bilinear {
                u_min: [0.0, 0.0, 0.5, 0.5],
                u_max: [0.5, 0.5, 0.5, 0.5],
                v_min: [0.0, 0.0, 0.5, 0.5],
                v_max: [0.5, 0.5, 0.5, 0.5],
                uv_min: [0.0, 0.0, 0.0, 0.0],
                uv_max: [0.0, 0.0, 0.0, 0.0],
            },
            command::Bilinear {
                u_min: [0.2, 0.0, 0.5, 0.5],
                u_max: [0.5, 0.5, 0.5, 0.5],
                v_min: [0.2, 0.0, 0.5, 0.5],
                v_max: [0.5, 0.5, 0.5, 0.5],
                uv_min: [0.0, 0.0, 0.0, 0.0],
                uv_max: [0.0, 0.0, 0.0, 0.0],
            },
            command::Bilinear {
                u_min: [0.4, 0.0, 0.5, 0.5],
                u_max: [0.5, 0.5, 0.5, 0.5],
                v_min: [0.4, 0.0, 0.5, 0.5],
                v_max: [0.5, 0.5, 0.5, 0.5],
                uv_min: [0.0, 0.0, 0.0, 0.0],
                uv_max: [0.0, 0.0, 0.0, 0.0],
            },
            command::Bilinear {
                u_min: [0.5, 0.0, 0.4, 0.5],
                u_max: [0.5, 0.5, 0.5, 0.5],
                v_min: [0.5, 0.0, 0.4, 0.5],
                v_max: [0.5, 0.5, 0.5, 0.5],
                uv_min: [0.0, 0.0, 0.0, 0.0],
                uv_max: [0.0, 0.0, 0.0, 0.0],
            },
            command::Bilinear {
                u_min: [0.5, 0.0, 0.2, 0.5],
                u_max: [0.5, 0.5, 0.5, 0.5],
                v_min: [0.5, 0.0, 0.2, 0.5],
                v_max: [0.5, 0.5, 0.5, 0.5],
                uv_min: [0.0, 0.0, 0.0, 0.0],
                uv_max: [0.0, 0.0, 0.0, 0.0],
            },
        ],
    );
}

fn run_bilinear(
    pool: &mut Pool,
    (fg_key, foreground): (PoolKey, Descriptor),
    (bg_key, background): (PoolKey, Descriptor),
    parameterizations: &[command::Bilinear],
) {
    let mut commands = CommandBuffer::default();

    let bilinear = command::Bilinear {
        u_min: [0.0, 0.0, 1.0, 1.0],
        u_max: [1.0, 1.0, 1.0, 1.0],
        v_min: [0.0, 0.0, 1.0, 1.0],
        v_max: [1.0, 1.0, 1.0, 1.0],
        uv_min: [0.0, 0.0, 0.0, 0.0],
        uv_max: [0.0, 0.0, 0.0, 0.0],
    };

    // Describe the pipeline:
    // 0: in (background)
    // 1: in (foreground)
    // 2: affine(0, affine, 1)
    // 3: out(2)
    let background = commands.input(background).unwrap();
    let foreground = commands.input(foreground).unwrap();

    let like = Descriptor::with_srgb_image(&image::DynamicImage::new_rgba8(512, 512));

    let result_affine = commands
        .with_knob()
        .bilinear(like, bilinear)
        .expect("Valid to paint with affine transformation");

    let (output_affine, _outformat) = commands.output(result_affine).expect("Valid for output");

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
        .query_knob(command::RegisterKnob {
            link_idx: 0,
            register: result_affine,
        })
        .unwrap();

    for (idx, input) in parameterizations.iter().enumerate() {
        let data = input.into_std430();

        let result = run_executable_with_output(
            &executable,
            pool,
            vec![(background, bg_key), (foreground, fg_key)],
            [(knob, data.as_slice())],
            retire_with_one_image(output_affine),
        );

        let image_with_knob = pool.entry(result).unwrap();
        let reference = format!("bilinear-knob-{idx}.crc.png");
        util::assert_reference(image_with_knob.into(), &reference);
    }
}
