//! This test file check the functionality, coverage tests basically.
//!
//! Since we want to be somewhat efficient we keep the same pool, instance, adapter, device around
//! while trying to ensure that it returns to an equivalent state after each test.
//!
//! FIXME: This stops after a single test failure. It would be more useful to have our own harness
//! here and do all of the tests, as separate as possible, recovering by recreating the whole state
//! when a test has failed (because that could have corrupted shared state).
#[path = "util.rs"]
mod util;

use zosimos::buffer::{self, Descriptor, Whitepoint};
use zosimos::command::{self, CommandBuffer, Rectangle};
use zosimos::pool::{Pool, PoolKey};
use zosimos::program::Program;

use self::util::{retire_with_one_image, run_once_with_output};

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

    run_blending(&mut pool, pool_foreground.clone(), pool_background.clone());

    run_affine(&mut pool, pool_foreground.clone(), pool_background.clone());

    run_adaptation(&mut pool, pool_background.clone());

    run_conversion(&mut pool, pool_background.clone());

    run_distribution(&mut pool);

    run_distribution_normal1d(&mut pool);

    run_distribution_u8(&mut pool);

    run_fractal_noise(&mut pool);

    run_transmute(&mut pool, pool_background.clone());

    run_palette(&mut pool, pool_background.clone());

    run_swap(&mut pool, pool_background.clone());

    run_oklab(&mut pool);

    run_srlab2(&mut pool);

    run_derivative(&mut pool, pool_background.clone());

    run_solid(&mut pool);
}

fn run_blending(
    pool: &mut Pool,
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

    let result = run_once_with_output(
        commands,
        pool,
        vec![(background, bg_key), (foreground, fg_key)],
        retire_with_one_image(output),
    );

    let image = pool.entry(result).unwrap();
    util::assert_reference(image.into(), "composed.crc.png");
}

fn run_affine(
    pool: &mut Pool,
    (fg_key, foreground): (PoolKey, Descriptor),
    (bg_key, background): (PoolKey, Descriptor),
) {
    let mut commands = CommandBuffer::default();

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
        retire_with_one_image(output_affine),
    );

    // This test is used in `async.rs`, with the same test image!
    let image_affine = pool.entry(result).unwrap();
    util::assert_reference(image_affine.into(), "affine.crc.png");
}

fn run_adaptation(pool: &mut Pool, (bg_key, background): (PoolKey, Descriptor)) {
    let mut commands = CommandBuffer::default();

    // Describe the pipeline:
    // 0: in (background)
    // 1: chromatic_adaptation(0, adapt)
    // 2: out(2)
    let background = commands.input(background).unwrap();

    let adapted = commands
        .chromatic_adaptation(
            background,
            command::ChromaticAdaptationMethod::VonKries,
            Whitepoint::D50,
        )
        .unwrap();

    let (output_affine, _outformat) = commands.output(adapted).expect("Valid for output");

    let result = run_once_with_output(
        commands,
        pool,
        vec![(background, bg_key)],
        retire_with_one_image(output_affine),
    );

    let image_adapted = pool.entry(result).unwrap();
    util::assert_reference(image_adapted.into(), "adapted.crc.png");
}

fn run_conversion(pool: &mut Pool, (orig_key, orig_descriptor): (PoolKey, Descriptor)) {
    // Pretend the input is BT709 instead of SRGB.
    let (bt_key, bt_descriptor) = {
        let mut bt = pool.allocate_like(orig_key);
        bt.set_color(buffer::Color::BT709_RGB);
        (bt.key(), bt.descriptor())
    };

    let mut commands = CommandBuffer::default();
    let input = commands.input(bt_descriptor).unwrap();

    let converted = commands
        .color_convert(input, orig_descriptor.color, orig_descriptor.texel)
        .unwrap();

    let (output, _outformat) = commands.output(converted).expect("Valid for output");

    let result = run_once_with_output(
        commands,
        pool,
        vec![(input, bt_key)],
        retire_with_one_image(output),
    );

    let image_converted = pool.entry(result).unwrap();
    util::assert_reference(image_converted.into(), "convert_bt709.crc.png");
}

fn run_distribution(pool: &mut Pool) {
    let mut layout = image::DynamicImage::new_luma_a16(400, 400);
    let descriptor = Descriptor::with_srgb_image(&layout);

    let mut commands = CommandBuffer::default();
    let generated = commands
        .distribution_normal2d(
            descriptor,
            command::DistributionNormal2d::with_diagonal(0.2, 0.2),
        )
        .unwrap();

    let (output, _outformat) = commands.output(generated).expect("Valid for output");

    let result = run_once_with_output(commands, pool, vec![], retire_with_one_image(output));

    let image_generated = pool.entry(result).unwrap();

    match layout {
        image::DynamicImage::ImageLumaA16(ref mut buffer) => {
            let bytes = image_generated.as_bytes().expect("Not a byte image");
            bytemuck::cast_slice_mut(&mut *buffer).copy_from_slice(bytes);
        }
        _ => unreachable!(),
    }

    util::assert_reference_image(layout, "distribution_normal2d.crc.png");
}

fn run_distribution_normal1d(pool: &mut Pool) {
    let mut layout = image::DynamicImage::new_luma_a16(400, 400);
    let descriptor = Descriptor::with_srgb_image(&layout);

    let mut commands = CommandBuffer::default();
    let generated = commands
        .distribution_normal2d(
            descriptor,
            command::DistributionNormal2d::with_direction([0.04998, 0.0501]),
        )
        .unwrap();

    let (output, _outformat) = commands.output(generated).expect("Valid for output");

    let result = run_once_with_output(commands, pool, vec![], retire_with_one_image(output));

    let image_generated = pool.entry(result).unwrap();

    match layout {
        image::DynamicImage::ImageLumaA16(ref mut buffer) => {
            let bytes = image_generated.as_bytes().expect("Not a byte image");
            bytemuck::cast_slice_mut(&mut *buffer).copy_from_slice(bytes);
        }
        _ => unreachable!(),
    }

    util::assert_reference_image(layout, "distribution_normal1d.crc.png");
}

fn run_distribution_u8(pool: &mut Pool) {
    let mut layout = image::DynamicImage::new_luma8(400, 400);
    let descriptor = Descriptor::with_srgb_image(&layout);

    let mut commands = CommandBuffer::default();
    let generated = commands
        .distribution_normal2d(
            descriptor,
            command::DistributionNormal2d::with_direction([0.04998, 0.0501]),
        )
        .unwrap();

    let (output, _outformat) = commands.output(generated).expect("Valid for output");

    let result = run_once_with_output(commands, pool, vec![], retire_with_one_image(output));

    let image_generated = pool.entry(result).unwrap();

    match layout {
        image::DynamicImage::ImageLuma8(ref mut buffer) => {
            let bytes = image_generated.as_bytes().expect("Not a byte image");
            bytemuck::cast_slice_mut(&mut *buffer).copy_from_slice(bytes);
        }
        _ => unreachable!(),
    }

    util::assert_reference_image(layout, "distribution_u8.crc.png");
}

fn run_fractal_noise(pool: &mut Pool) {
    let mut layout = image::DynamicImage::new_rgba8(400, 400);
    let descriptor = Descriptor::with_srgb_image(&layout);

    let mut commands = CommandBuffer::default();
    let generated = commands
        .distribution_fractal_noise(descriptor, command::FractalNoise::with_octaves(4))
        .unwrap();

    let (output, _outformat) = commands.output(generated).expect("Valid for output");

    let result = run_once_with_output(commands, pool, vec![], retire_with_one_image(output));

    let image_generated = pool.entry(result).unwrap();

    match layout {
        image::DynamicImage::ImageRgba8(ref mut buffer) => {
            let bytes = image_generated.as_bytes().expect("Not a byte image");
            bytemuck::cast_slice_mut(&mut *buffer).copy_from_slice(bytes);
        }
        _ => unreachable!(),
    }

    util::assert_reference_image(layout, "distribution_fractal2d.crc.png");
}

fn run_transmute(pool: &mut Pool, (orig_key, orig_descriptor): (PoolKey, Descriptor)) {
    let mut commands = CommandBuffer::default();
    let (width, height) = orig_descriptor.size();
    let mut layout = image::DynamicImage::new_luma_a16(width, height);

    let input = commands.input(orig_descriptor).unwrap();

    let transmute = commands
        // FIXME: make nice again?
        .transmute(input, Descriptor::with_srgb_image(&layout))
        .unwrap();

    let (output, _outformat) = commands.output(transmute).expect("Valid for output");

    let result = run_once_with_output(
        commands,
        pool,
        vec![(input, orig_key)],
        retire_with_one_image(output),
    );

    let image_generated = pool.entry(result).unwrap();

    match layout {
        image::DynamicImage::ImageLumaA16(ref mut buffer) => {
            let bytes = image_generated.as_bytes().expect("Not a byte image");
            bytemuck::cast_slice_mut(&mut *buffer).copy_from_slice(bytes);
        }
        _ => unreachable!(),
    }

    let bg_image = pool.entry(orig_key).unwrap();
    assert_eq!(layout.as_bytes(), bg_image.as_bytes().unwrap());

    util::assert_reference_image(layout, "transmute.crc.png");
}

fn run_palette(pool: &mut Pool, (orig_key, orig_descriptor): (PoolKey, Descriptor)) {
    let distribution_layout = {
        let layout = image::DynamicImage::new_rgba8(400, 400);
        Descriptor::with_srgb_image(&layout)
    };

    let mut commands = CommandBuffer::default();

    let input = commands.input(orig_descriptor).unwrap();
    // Some arbitrary weird, red-green-color ramps.
    let ramp = commands
        .bilinear(
            distribution_layout,
            command::Bilinear {
                u_min: [0.0, 0.0, 0.0, 1.0],
                v_min: [0.0, 0.0, 0.0, 1.0],
                uv_min: [0.0, 0.0, 0.0, 1.0],
                u_max: [0.7, 0.0, 0.0, 1.0],
                v_max: [0.0, 0.7, 0.0, 1.0],
                uv_max: [0.3, 0.3, 0.0, 1.0],
            },
        )
        .unwrap();

    let palette = command::Palette {
        width: Some(buffer::ColorChannel::R),
        height: Some(buffer::ColorChannel::G),
        width_base: 0,
        height_base: 0,
    };

    let sampled = commands.palette(input, palette, ramp).unwrap();

    let (output, _outformat) = commands.output(sampled).expect("Valid for output");

    let result = run_once_with_output(
        commands,
        pool,
        vec![(input, orig_key)],
        retire_with_one_image(output),
    );

    let image_sampled = pool.entry(result).unwrap();
    util::assert_reference(image_sampled.into(), "palette.crc.png");
}

fn run_swap(pool: &mut Pool, (orig_key, orig_descriptor): (PoolKey, Descriptor)) {
    use buffer::ColorChannel;
    let mut commands = CommandBuffer::default();

    let input = commands.input(orig_descriptor).unwrap();
    let channel_r = commands.extract(input, ColorChannel::R).unwrap();
    let channel_g = commands.extract(input, ColorChannel::G).unwrap();

    let intermediate = commands.inject(input, ColorChannel::G, channel_r).unwrap();
    let swapped = commands
        .inject(intermediate, ColorChannel::R, channel_g)
        .unwrap();

    let (output, _outformat) = commands.output(swapped).expect("Valid for output");

    let result = run_once_with_output(
        commands,
        pool,
        vec![(input, orig_key)],
        retire_with_one_image(output),
    );

    let image_swapped = pool.entry(result).unwrap();
    util::assert_reference(image_swapped.into(), "swapped.crc.png");
}

fn run_oklab(pool: &mut Pool) {
    let mut commands = CommandBuffer::default();

    let output = image::DynamicImage::new_rgba8(400, 400);
    let color_descriptor = buffer::Descriptor::with_srgb_image(&output);

    let distribution_layout = buffer::Descriptor {
        color: buffer::Color::Scalars {
            transfer: buffer::Transfer::Linear,
        },
        ..color_descriptor.clone()
    };

    let oklab_texel = buffer::Descriptor {
        texel: buffer::Texel {
            block: distribution_layout.texel.block,
            bits: distribution_layout.texel.bits,
            parts: buffer::SampleParts::LchA,
        },
        color: buffer::Color::Oklab,
        ..distribution_layout.clone()
    };

    let sampling_grid = commands
        .bilinear(
            distribution_layout,
            command::Bilinear {
                // lightness, chromaticity, hue
                // This is constant lightness (0.8),
                // chromaticity from 0.0 to 1.0 and
                // all hues.
                // Note that many values may be clamped into sRGB.
                u_min: [0.4, 0.0, 0.0, 1.0],
                v_min: [0.4, 0.0, 0.0, 1.0],
                u_max: [0.4, 0.0, 1.0, 1.0],
                v_max: [0.4, 1.0, 0.0, 1.0],
                uv_min: [0.0, 0.0, 0.0, 0.0],
                uv_max: [0.0, 0.0, 0.0, 0.0],
            },
        )
        .unwrap();

    let lch = commands
        .transmute(sampling_grid, oklab_texel)
        .expect("Valid transmute");
    let converted = commands
        .color_convert(
            lch,
            color_descriptor.color.clone(),
            color_descriptor.texel.clone(),
        )
        .expect("Valid for conversion");

    let (output, _) = commands.output(converted).expect("Valid for output");

    let result = run_once_with_output(commands, pool, vec![], retire_with_one_image(output));

    let image_show = pool.entry(result).unwrap();
    util::assert_reference(image_show.into(), "oklab.crc.png");
}

fn run_srlab2(pool: &mut Pool) {
    let mut commands = CommandBuffer::default();

    let output = image::DynamicImage::new_rgba8(400, 400);
    let color_descriptor = buffer::Descriptor::with_srgb_image(&output);

    let distribution_layout = buffer::Descriptor {
        layout: color_descriptor.layout.clone(),
        texel: buffer::Texel {
            block: buffer::Block::Pixel,
            ..color_descriptor.texel
        },
        color: buffer::Color::Scalars {
            transfer: buffer::Transfer::Linear,
        },
    };

    let srlab2_texel = buffer::Descriptor {
        color: buffer::Color::SrLab2 {
            whitepoint: buffer::Whitepoint::D65,
        },
        texel: buffer::Texel {
            block: distribution_layout.texel.block,
            bits: distribution_layout.texel.bits,
            parts: buffer::SampleParts::LchA,
        },
        ..distribution_layout
    };

    let sampling_grid = commands
        .bilinear(
            distribution_layout,
            command::Bilinear {
                // lightness, chromaticity, hue
                // This is constant lightness (0.8),
                // chromaticity from 0.0 to 1.0 and
                // all hues.
                // Note that many values may be clamped into sRGB.
                u_min: [0.4, 0.0, 0.0, 1.0],
                v_min: [0.4, 0.0, 0.0, 1.0],
                u_max: [0.4, 0.0, 1.0, 1.0],
                v_max: [0.4, 1.0, 0.0, 1.0],
                uv_min: [0.0, 0.0, 0.0, 0.0],
                uv_max: [0.0, 0.0, 0.0, 0.0],
            },
        )
        .unwrap();

    let lch = commands
        .transmute(sampling_grid, srlab2_texel)
        .expect("Valid transmute");
    let converted = commands
        .color_convert(
            lch,
            color_descriptor.color.clone(),
            color_descriptor.texel.clone(),
        )
        .expect("Valid for conversion");

    let (output, _) = commands.output(converted).expect("Valid for output");

    let result = run_once_with_output(commands, pool, vec![], retire_with_one_image(output));

    let image_show = pool.entry(result).unwrap();
    util::assert_reference(image_show.into(), "srlab2.crc.png");
}

fn run_derivative(pool: &mut Pool, (bg_key, background): (PoolKey, Descriptor)) {
    const METHODS: &[command::DerivativeMethod] = &[
        command::DerivativeMethod::Scharr3,
        command::DerivativeMethod::Scharr3To4Bit,
        command::DerivativeMethod::Scharr3To8Bit,
        command::DerivativeMethod::Prewitt,
        command::DerivativeMethod::Sobel,
    ];

    for method in METHODS {
        let mut commands = CommandBuffer::default();

        // Describe the pipeline:
        // 0: in (background)
        // 1: derivative(0, derive)
        // 2: out(2)
        let background = commands.input(background.clone()).unwrap();

        let derived = commands
            .derivative(
                background,
                command::Derivative {
                    method: method.clone(),
                    direction: command::Direction::Width,
                },
            )
            .unwrap();

        let (output_derived, _outformat) = commands.output(derived).expect("Valid for output");

        let result = run_once_with_output(
            commands,
            pool,
            vec![(background, bg_key)],
            retire_with_one_image(output_derived),
        );

        let image_derived = pool.entry(result).unwrap();
        let reference = format!("derived_{:?}.crc.png", method);
        util::assert_reference(image_derived.into(), &reference);
    }
}

fn run_solid(pool: &mut Pool) {
    let mut layout = image::DynamicImage::new_rgba8(400, 400);
    let descriptor = Descriptor::with_srgb_image(&layout);

    let mut commands = CommandBuffer::default();
    let generated = commands
        .solid_rgba(descriptor, [0.5, 0.5, 1.0, 1.0])
        .unwrap();

    let (output, _outformat) = commands.output(generated).expect("Valid for output");

    let result = run_once_with_output(commands, pool, vec![], retire_with_one_image(output));

    let image_generated = pool.entry(result).unwrap();

    match layout {
        image::DynamicImage::ImageRgba8(ref mut buffer) => {
            let bytes = image_generated.as_bytes().expect("Not a byte image");
            bytemuck::cast_slice_mut(&mut *buffer).copy_from_slice(bytes);
        }
        _ => unreachable!(),
    }

    util::assert_reference_image(layout, "solid.crc.png");
}
