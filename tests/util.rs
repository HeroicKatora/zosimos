// This is almost certainly not all used in all tests.
#![allow(dead_code)]
use std::path::Path;

use zosimos::command::{CommandBuffer, Register};
use zosimos::pool::PoolImage;
use zosimos::pool::{Pool, PoolKey};
use zosimos::program::{Capabilities, Knob};
use zosimos::run::{Executable, Retire};

const CRC: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/reference");
const DEBUG: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/debug");

pub fn assert_reference(image: PoolImage, key: &str) {
    let image = image
        .to_image()
        .expect("Invalid image, must be convertible to `image` image");
    assert_reference_image(image, key);
}

pub fn assert_reference_image(image: image::DynamicImage, key: &str) {
    let hash = blockhash::blockhash256(&image);

    let output = Path::new(CRC).join(key);
    let debug_path = Path::new(DEBUG).join(key);

    if std::env::var_os("ZOSIMOS_BLESS").is_some() {
        let pre: String = std::fs::read_to_string(&output).unwrap_or_default();

        let mut lines = pre.lines().map(String::from).collect::<Vec<_>>();
        let hash = hash.to_string();

        if !lines.contains(&hash) {
            lines.push(hash);
        }

        eprintln!("{}: {:?}", key, image.color());
        std::fs::write(&output, lines.join("\n")).expect("Failed to bless result");

        image
            .save_with_format(&debug_path, image::ImageFormat::Png)
            .expect("Failed to read result file");
    }

    let expected = std::fs::read_to_string(&output).expect("Failed to read result file");

    if !expected.lines().any(|line| line == hash.to_string()) {
        image
            .save_with_format(&debug_path, image::ImageFormat::Png)
            .expect("Failed to read result file");

        panic!(
            "Reference CRC-32 comparison failed: {} vs. {}\
            An image has been saved to {}",
            expected,
            hash,
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

    run_executable_with_output(&executable, pool, binds, [], output)
}

pub fn run_executable_with_output<'knob, T>(
    executable: &Executable,
    pool: &mut Pool,
    binds: impl IntoIterator<Item = (Register, PoolKey)>,
    knobs: impl IntoIterator<Item = (Knob, &'knob [u8])>,
    output: impl FnOnce(&mut Retire) -> T,
) -> T {
    let mut environment = executable.from_pool(pool).expect("no device found in pool");

    for (knob, data) in knobs {
        environment.knob(knob, data).expect("no knob data set");
    }

    for (target, key) in binds {
        environment.bind(target, key).unwrap();
    }

    let _ = environment.recover_buffers();
    let mut execution = executable.launch(environment).expect("Launching failed");
    // Prepare pool to be clear for cache.
    pool.clear_cache();

    while execution.is_running() {
        let _wait_point = execution.step().expect("Shouldn't fail but");
    }

    let mut retire = execution.retire_gracefully(pool);
    let result = output(&mut retire);
    let _ = retire.retire_buffers();
    retire.finish();
    result
}

pub fn retire_with_one_image(reg: Register) -> impl FnOnce(&mut Retire) -> PoolKey {
    move |retire: &mut Retire| retire.output(reg).expect("Valid for output").key()
}
