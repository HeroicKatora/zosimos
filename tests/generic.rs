//! Check that the generics interface kind of works.
#[path = "util.rs"]
mod util;

use stealth_paint::buffer::{self, ColorChannel, Descriptor, SampleParts, Texel};
use stealth_paint::command::{self, Bilinear, CommandBuffer, CommandError, Palette, ShaderCommand};
use stealth_paint::pool::Pool;
use stealth_paint::program::{Capabilities, Program};

use self::util::{retire_with_one_image, run_executable_with_output};

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
        let img_input = commands.input(todo!())?;

        let img_idx = commands.bilinear(
            Descriptor::with_texel(Texel::new_u8(SampleParts::Rgb), 2048, 2048).unwrap(),
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

    let (main, output) = (move || {
        let mut commands = CommandBuffer::default();
        let img_input = commands.input(todo!())?;

        let target = image::DynamicImage::ImageRgba8(image::RgbaImage::new(512, 512));
        let srgb = Descriptor::with_srgb_image(&target);

        let srgb = todo!();
        let (output, _outformat) = commands.output(srgb).expect("Valid for output");

        Ok::<_, CommandError>((commands, output))
    })()
    .expect("build generic inner function sequence");

    let plan = main
        .link(&[fixed_palette], &[])
        .expect("compile full function sequence");

    let capabilities = Capabilities::from({
        let mut devices = pool.iter_devices();
        devices.next().expect("the pool to contain a device")
    });

    let executable = plan
        .lower_to(capabilities)
        .expect("No extras beyond device required");

    run_executable_with_output(
        &executable,
        &mut pool,
        vec![],
        retire_with_one_image(output),
    );
}
