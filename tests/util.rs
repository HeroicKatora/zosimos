use std::path::Path;
use std::hash::{Hash, Hasher};
use stealth_paint::pool::PoolImage;
use image::GenericImageView;

const CRC: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/reference");
const DEBUG: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/debug");

pub fn assert_reference(image: PoolImage, key: &str) {
    let image = image
        .to_image()
        .expect("Invalid image, must be convertible to `image` image");

    let mut crc = crc32fast::Hasher::new();
    let (width, height) = image.dimensions();
    crc.write_u32(width);
    crc.write_u32(height);
    core::mem::discriminant(&image).hash(&mut crc);
    crc.write(image.as_bytes());
    let crc = crc.finish();

    let output = Path::new(CRC).join(key);
    let debug_path = Path::new(DEBUG).join(key);

    if std::env::var_os("STEALTH_PAINT_BLESS").is_some() {
        std::fs::write(&output, format!("{}", crc))
            .expect("Failed to bless result");
        image.save_with_format(&debug_path, image::ImageFormat::Png)
            .expect("Failed to read result file");
    }

    let expected = std::fs::read(&output)
        .expect("Failed to read result file");

    let expected: u64 = ::core::str::from_utf8(&expected)
        .expect("Failed to read result file")
        .parse()
        .expect("Failed to parse result file as 64-bit CRC");

    if expected != crc {
        image.save_with_format(&debug_path, image::ImageFormat::Png)
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
