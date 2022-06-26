//! Everything testing our extension points.
#[path = "util.rs"]
mod util;

use stealth_paint::buffer::{self, Descriptor};
use stealth_paint::command::{self, CommandBuffer, ShaderCommand};
use stealth_paint::pool::Pool;
use stealth_paint::program::Program;

use self::util::{retire_with_one_image, run_once_with_output};

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

    env_logger::init();

    const ANY: wgpu::Backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(ANY);

    let adapter =
        Program::minimum_adapter(instance.enumerate_adapters(ANY)).expect("to get an adapter");

    let mut pool = Pool::new();

    pool.request_device(&adapter, Program::minimal_device_descriptor())
        .expect("to get a device");

    // Actual program begins here.
    let target = image::DynamicImage::ImageRgba8(image::RgbaImage::new(4000, 4000));

    let mut commands = CommandBuffer::default();
    let brot = commands.construct_dynamic(&Mandelbrot::new(Descriptor {
        layout: buffer::ByteLayout::from(&target),
        color: buffer::Color::Oklab,
        texel: buffer::Texel {
            block: buffer::Block::Pixel,
            bits: buffer::SampleBits::Int8x4,
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
    util::assert_reference(image.into(), "mandelbrot.png.crc");
}
