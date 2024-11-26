//! Everything testing our extension points.
#[path = "util.rs"]
mod util;

use zosimos::buffer::{self, Descriptor};
use zosimos::command::{self, CommandBuffer, ShaderCommand};
use zosimos::pool::Pool;
use zosimos::program::Program;

use self::util::{retire_with_one_image, run_once_with_output};

const BACKGROUND: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/input/background.png");

#[test]
fn mandelbrot() {
    struct Mandelbrot {
        scale: (f32, f32),
        source: &'static [u8],
        descriptor: Descriptor,
    }

    impl Mandelbrot {
        fn new(descriptor: Descriptor) -> Self {
            pub const SHADER_ENCODE: &[u8] =
                include_bytes!(concat!(env!("OUT_DIR"), "/spirv/mandelbrot.frag.v"));

            Mandelbrot {
                scale: (3.0, 3.0),
                source: SHADER_ENCODE,
                descriptor,
            }
        }
    }

    impl ShaderCommand for Mandelbrot {
        fn source(&self) -> command::ShaderSource {
            command::ShaderSource::SpirV(self.source.into())
        }

        fn data(&self, mut data: command::ShaderData<'_>) -> Descriptor {
            data.set_data(&[self.scale.0, self.scale.1, 0.6, 0.5]);

            self.descriptor.clone()
        }
    }

    let _ = env_logger::try_init();

    const ANY: wgpu::Backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: ANY,
        ..Default::default()
    });

    let adapter = Program::minimum_adapter(instance.enumerate_adapters(ANY).into_iter())
        .expect("to get an adapter");

    let mut pool = Pool::new();

    let descriptor = Program::minimal_device_descriptor();
    let descriptor = wgpu::DeviceDescriptor {
        required_limits: wgpu::Limits {
            max_texture_dimension_2d: 1 << 12,
            ..descriptor.required_limits
        },
        ..descriptor
    };

    pool.request_device(&adapter, descriptor)
        .expect("to get a device");

    // Actual program begins here.
    let target = image::DynamicImage::ImageRgba8(image::RgbaImage::new(2048, 2048));

    let mut commands = CommandBuffer::default();
    let brot = commands.construct_dynamic(&Mandelbrot::new(Descriptor {
        layout: buffer::ByteLayout::from(&target),
        color: buffer::Color::Oklab,
        texel: buffer::Texel {
            block: buffer::Block::Pixel,
            bits: buffer::SampleBits::UInt8x4,
            parts: buffer::SampleParts::LchA,
        },
    }));

    let srgb = Descriptor::with_srgb_image(&target);
    let srgb = commands
        .color_convert(brot, srgb.color, srgb.texel)
        .expect("Valid for color conversion");
    let (output, _outformat) = commands.output(srgb).expect("Valid for output");

    let result = run_once_with_output(commands, &mut pool, vec![], retire_with_one_image(output));

    let image = pool.entry(result).unwrap();
    util::assert_reference(image.into(), "mandelbrot.crc.png");
}

#[test]
fn crt() {
    struct Crt {
        source: &'static [u8],
        descriptor: Descriptor,
    }

    impl Crt {
        fn new(descriptor: Descriptor) -> Self {
            pub const SHADER_ENCODE: &[u8] =
                include_bytes!(concat!(env!("OUT_DIR"), "/spirv/crt.frag.v"));
            let (w, h) = descriptor.size();
            let (w, h) = (3 * w, 3 * h);

            // Change the descriptor to the appropriate size, same color and texel interpretation.
            let descriptor = Descriptor {
                color: descriptor.color.clone(),
                ..Descriptor::with_texel(descriptor.texel.clone(), w, h).unwrap()
            };

            Crt {
                source: SHADER_ENCODE,
                descriptor,
            }
        }
    }

    impl ShaderCommand for Crt {
        fn source(&self) -> command::ShaderSource {
            command::ShaderSource::SpirV(self.source.into())
        }

        fn data(&self, mut data: command::ShaderData<'_>) -> Descriptor {
            let (w, h) = self.descriptor.size();
            data.set_data(&[w, h]);
            self.descriptor.clone()
        }
    }

    let _ = env_logger::try_init();

    const ANY: wgpu::Backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: ANY,
        ..Default::default()
    });

    let adapter = Program::minimum_adapter(instance.enumerate_adapters(ANY).into_iter())
        .expect("to get an adapter");

    let mut pool = Pool::new();

    let (bg_key, bg_descriptor) = {
        let background = image::open(BACKGROUND).expect("Background image opened");
        let entry = pool.insert_srgb(&background);
        (entry.key(), entry.descriptor())
    };

    pool.request_device(&adapter, Program::minimal_device_descriptor())
        .expect("to get a device");

    // Actual program begins here.
    let target = image::DynamicImage::ImageRgba8(image::RgbaImage::new(2048, 2048));

    let mut commands = CommandBuffer::default();
    let reg_background = commands
        .input(bg_descriptor.clone())
        .expect("Valid for input");

    let brot = commands
        .unary_dynamic(reg_background, &Crt::new(bg_descriptor.clone()))
        .expect("Valid for call");

    let srgb = Descriptor::with_srgb_image(&target);
    let srgb = commands
        .color_convert(brot, srgb.color, srgb.texel)
        .expect("Valid for color conversion");
    let (output, _outformat) = commands.output(srgb).expect("Valid for output");

    let result = run_once_with_output(
        commands,
        &mut pool,
        vec![(reg_background, bg_key)],
        retire_with_one_image(output),
    );

    let image = pool.entry(result).unwrap();
    util::assert_reference(image.into(), "crt.crc.png");
}

#[test]
fn flat_correction() {
    struct FlatField {
        source: &'static [u8],
        descriptor: Descriptor,
        recovered_mean: f32,
    }

    impl FlatField {
        fn new(descriptor: Descriptor, flat: &Descriptor, mean: f32) -> Self {
            pub const SHADER_ENCODE: &[u8] =
                include_bytes!(concat!(env!("OUT_DIR"), "/spirv/flat_field.frag.v"));

            assert_eq!(descriptor.size(), flat.size());

            FlatField {
                source: SHADER_ENCODE,
                descriptor,
                recovered_mean: mean,
            }
        }
    }

    impl ShaderCommand for FlatField {
        fn source(&self) -> command::ShaderSource {
            command::ShaderSource::SpirV(self.source.into())
        }

        fn data(&self, mut data: command::ShaderData<'_>) -> Descriptor {
            data.set_data(&[self.recovered_mean]);
            self.descriptor.clone()
        }
    }

    let _ = env_logger::try_init();

    const ANY: wgpu::Backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: ANY,
        ..Default::default()
    });

    let adapter = Program::minimum_adapter(instance.enumerate_adapters(ANY).into_iter())
        .expect("to get an adapter");

    let mut pool = Pool::new();

    let (bg_key, bg_descriptor) = {
        let background = image::open(BACKGROUND).expect("Background image opened");
        let entry = pool.insert_srgb(&background);
        (entry.key(), entry.descriptor())
    };

    let flat_descriptor = {
        let flat = image::DynamicImage::ImageLuma8(image::GrayImage::new(512, 512));
        Descriptor::with_srgb_image(&flat)
    };

    pool.request_device(&adapter, Program::minimal_device_descriptor())
        .expect("to get a device");

    // We synthesize it here as noise with known amplitude. Realistically you want the parameters
    // to be a runtime-computed knob, i.e. supplied via some source buffer.
    let flat_field = FlatField::new(bg_descriptor.clone(), &flat_descriptor, 0.124 / 2.0);

    // Actual program begins here.
    let target = image::DynamicImage::ImageRgba8(image::RgbaImage::new(512, 512));

    let mut commands = CommandBuffer::default();
    let reg_background = commands
        .input(bg_descriptor.clone())
        .expect("Valid for input");

    let reg_flat = commands
        .distribution_fractal_noise(
            flat_descriptor.clone(),
            command::FractalNoise {
                num_octaves: 8,
                initial_amplitude: 0.1,
                amplitude_damping: 0.2,
                grid_scale: [10.0, 10.0],
            },
        )
        .expect("Value as fractal noise");

    let corrected = commands
        .binary_dynamic(reg_background, reg_flat, &flat_field)
        .expect("Valid for call");

    let srgb = Descriptor::with_srgb_image(&target);
    let srgb = commands
        .color_convert(corrected, srgb.color, srgb.texel)
        .expect("Valid for color conversion");
    let (output, _outformat) = commands.output(srgb).expect("Valid for output");

    let result = run_once_with_output(
        commands,
        &mut pool,
        vec![(reg_background, bg_key)],
        retire_with_one_image(output),
    );

    let image = pool.entry(result).unwrap();
    util::assert_reference(image.into(), "flat_field.crc.png");
}
