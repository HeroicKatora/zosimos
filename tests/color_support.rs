//! File just verifies the different input / output color spaces that are truly supported.
//!
//! Here's a non-exhaustive list of goals:
//!
//! - `RGB`-ish spaces with some list of known primaries and EO-transfer functions.
//! - `YUV`-ish modifications of the above.
//! - WIP: `CMYK` modifications of the above.
//! - `Lab` color spaces and in particular Oklab and (WIP): CIELAB.
//! - WIP: ICC profiles for these inputs into a linear one.
//!
//! What we do not test here is any modification possible in such color spaces. Firstly most
//! operations are invoked on the linear equivalent which is always an rgb under a certain
//! whitepoint.
use zosimos::{
    buffer::{Descriptor, ImageBuffer},
    command::CommandBuffer,
    pool::Pool,
};

#[test]
fn test_descriptor_as_gpu_texture() {
    let mut pool = Pool::new();

    const ANY: wgpu::Backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: ANY,
        ..Default::default()
    });

    let descriptors: &[Descriptor] = &[
        { Descriptor::with_srgb_image(&image::DynamicImage::new_rgba8(4, 4)) },
        { Descriptor::with_srgb_image(&image::DynamicImage::new_rgba16(4, 4)) },
    ];

    for descriptor in descriptors {
        check_input_zeros_as(descriptor.clone(), &mut pool, &instance);
    }
}

fn check_input_zeros_as(descriptor: Descriptor, pool: &mut Pool, instance: &wgpu::Instance) {
    let buffer = ImageBuffer::with_descriptor(&descriptor);
    let key = pool.insert(buffer, descriptor.clone()).key();

    let mut command = CommandBuffer::default();
    let input = command
        .input(descriptor.clone())
        .expect("Valid descriptors can be used as inputs");
    let (output, out_descriptor) = command
        .output(input)
        .expect("Valid descriptors can be used as outputs");

    debug_assert_eq!(
        out_descriptor.as_concrete(),
        Some(descriptor),
        "Descriptor changed while writing to output, this is odd"
    );

    let program = command.compile().expect("Valid full program");

    let adapter = program
        .choose_adapter(
            instance
                .enumerate_adapters(wgpu::Backends::all())
                .into_iter(),
        )
        .unwrap();

    let mut run = program
        .launch(pool)
        .bind(input, key)
        .expect("Valid input register")
        .launch(&adapter)
        .expect("Valid launcher to run");

    while run.is_running() {
        let mut step = run.step().unwrap();
        step.block_on().unwrap();
    }

    {
        let mut retire = run.retire_gracefully(pool);
        retire.output(output).unwrap();
        retire.finish();
    }
}
