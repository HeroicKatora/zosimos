// This is almost certainly not all used in all tests.
#![allow(dead_code)]
use image::GenericImageView;
use std::hash::Hasher;
use std::path::Path;

use stealth_paint::command::{CommandBuffer, Register};
use stealth_paint::pool::PoolImage;
use stealth_paint::pool::{Pool, PoolKey};
use stealth_paint::program::Capabilities;
use stealth_paint::run::{Executable, Retire};

const CRC: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/reference");
const DEBUG: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/debug");

pub fn assert_reference(image: PoolImage, key: &str) {
    let image = image
        .to_image()
        .expect("Invalid image, must be convertible to `image` image");
    assert_reference_image(image, key);
}

pub fn assert_reference_image(image: image::DynamicImage, key: &str) {
    let mut crc = crc32fast::Hasher::new();
    let (width, height) = image.dimensions();
    crc.write_u32(width);
    crc.write_u32(height);

    crc.write(image.as_bytes());
    let crc = crc.finish();

    let output = Path::new(CRC).join(key);
    let debug_path = Path::new(DEBUG).join(key);

    if std::env::var_os("STEALTH_PAINT_BLESS").is_some() {
        eprintln!("{}: {:?}", key, image.color());
        std::fs::write(&output, format!("{}", crc)).expect("Failed to bless result");
        image
            .save_with_format(&debug_path, image::ImageFormat::Png)
            .expect("Failed to read result file");
    }

    let expected = std::fs::read(&output).expect("Failed to read result file");

    let expected: u64 = ::core::str::from_utf8(&expected)
        .expect("Failed to read result file")
        .parse()
        .expect("Failed to parse result file as 64-bit CRC");

    if expected != crc {
        image
            .save_with_format(&debug_path, image::ImageFormat::Png)
            .expect("Failed to read result file");

        panic!(
            "Reference CRC-32 comparison failed: {} vs. {}\
            An image has been saved to {}",
            expected,
            crc,
            debug_path.display(),
        );
    }
}

pub fn run_once_with_output<T>(
    commands: CommandBuffer,
    pool: &mut Pool,
    binds: impl IntoIterator<Item = (Register, PoolKey)>,
    output: impl FnOnce(&mut Retire) -> T,
) -> T {
    let plan = commands.compile().expect("Could build command buffer");
    let capabilities = Capabilities::from({
        let mut devices = pool.iter_devices();
        devices.next().expect("the pool to contain a device")
    });

    let executable = plan
        .lower_to(capabilities)
        .expect("No extras beyond device required");

    run_executable_with_output(&executable, pool, binds, output)
}

pub fn run_executable_with_output<T>(
    executable: &Executable,
    pool: &mut Pool,
    binds: impl IntoIterator<Item = (Register, PoolKey)>,
    output: impl FnOnce(&mut Retire) -> T,
) -> T {
    let mut environment = executable.from_pool(pool).expect("no device found in pool");

    for (target, key) in binds {
        environment.bind(target, key).unwrap();
    }

    let mut execution = executable.launch(environment).expect("Launching failed");

    while execution.is_running() {
        let _wait_point = execution.step().expect("Shouldn't fail but");
    }

    let mut retire = execution.retire_gracefully(pool);
    let result = output(&mut retire);
    retire.finish();
    result
}

pub fn retire_with_one_image(reg: Register) -> impl FnOnce(&mut Retire) -> PoolKey {
    move |retire: &mut Retire| retire.output(reg).expect("Valid for output").key()
}
