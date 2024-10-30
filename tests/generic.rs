//! Check that the generics interface kind of works.
#[path = "util.rs"]
mod util;

use zosimos::buffer::{ColorChannel, Descriptor, SampleParts, Texel};
use zosimos::command::{self, Bilinear, CommandBuffer, CommandError, GenericDeclaration, Palette};
use zosimos::pool::Pool;
use zosimos::program::{Capabilities, Program};

use self::util::{retire_with_one_image, run_executable_with_output};

const BACKGROUND: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/input/background.png");

#[test]
fn generic_palette() {
    env_logger::init();

    const ANY: wgpu::Backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: ANY,
        ..Default::default()
    });

    let adapter = Program::minimum_adapter(instance.enumerate_adapters(ANY).into_iter())
        .expect("to get an adapter");

    let mut pool = Pool::new();

    pool.request_device(&adapter, Program::minimal_device_descriptor())
        .expect("to get a device");

    let fixed_palette = (|| {
        let mut commands = CommandBuffer::default();
        let in_a = commands.generic(GenericDeclaration { bounds: &[] });

        let img_input = commands.input_generic(in_a)?;

        let img_idx = commands.bilinear(
            Descriptor::with_texel(Texel::new_u8(SampleParts::RgbA), 2048, 2048).unwrap(),
            Bilinear {
                u_min: [0.0, 0.0, 0.0, 0.0],
                u_max: [0.0, 0.0, 0.0, 0.0],
                v_min: [0.0, 0.0, 0.0, 0.0],
                v_max: [0.0, 0.0, 0.0, 0.0],
                uv_min: [0.0, 0.0, 0.0, 0.0],
                uv_max: [1.0, 1.0, 0.0, 0.0],
            },
        )?;

        let img_palette = commands.palette(
            img_input,
            Palette {
                height: Some(ColorChannel::R),
                width: Some(ColorChannel::G),
                height_base: 0,
                width_base: 0,
            },
            img_idx,
        )?;

        commands.output(img_palette)?;

        Ok::<_, CommandError>(commands)
    })()
    .expect("build generic inner function sequence");

    let fixed_palette_sig = fixed_palette.computed_signature();
    let (main, output, img_input) = (move || {
        let mut commands = CommandBuffer::default();
        let converter = commands.function(fixed_palette_sig)?;

        let target = image::DynamicImage::ImageRgba8(image::RgbaImage::new(512, 512));
        let srgb = Descriptor::with_srgb_image(&target);

        let img_input = commands.input(srgb)?;
        let img_input_ty = commands.register_descriptor(img_input)?;

        let img_output = commands.invoke(
            converter,
            command::InvocationArguments {
                generics: &[img_input_ty],
                arguments: &[img_input],
            },
        )?;

        let [img_output] = img_output[..] else {
            panic!("Mismatched function arguments count");
        };

        let (output, _outformat) = commands.output(img_output).expect("Valid for output");

        Ok::<_, CommandError>((commands, output, img_input))
    })()
    .expect("build generic inner function sequence");

    let plan = main
        .link(&[], &[fixed_palette], &[&[1], &[]])
        .expect("compile full function sequence");

    let capabilities = Capabilities::from({
        let mut devices = pool.iter_devices();
        devices.next().expect("the pool to contain a device")
    });

    let background = image::open(BACKGROUND).expect("Background image opened");

    let pool_background = {
        let entry = pool.insert_srgb(&background);
        entry.key()
    };

    let executable = plan
        .lower_to(capabilities)
        .expect("No extras beyond device required");

    let generic = run_executable_with_output(
        &executable,
        &mut pool,
        vec![(img_input, pool_background)],
        [],
        retire_with_one_image(output),
    );

    let image = pool.entry(generic).unwrap();
    util::assert_reference(image.into(), "generic.crc.png");
}
