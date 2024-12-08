use serde::Deserialize;

use image_canvas::color;
use zosimos::{buffer, command};

#[derive(Deserialize)]
#[serde(remote = "command::ChromaticAdaptationMethod")]
pub enum ChromaticAdaptationMethod {
    Xyz,
    VonKries,
    BradfordVonKries,
    BradfordNonLinear,
}

#[derive(Deserialize)]
#[serde(remote = "color::Color")]
pub enum Color {
    Rgb {
        #[serde(with = "Primaries")]
        primary: color::Primaries,
        #[serde(with = "Transfer")]
        transfer: color::Transfer,
        #[serde(with = "Whitepoint")]
        whitepoint: color::Whitepoint,
        #[serde(with = "Luminance")]
        luminance: color::Luminance,
    },
    Oklab,
    Scalars {
        #[serde(with = "Transfer")]
        transfer: color::Transfer,
    },
}

#[derive(Deserialize)]
pub struct Descriptor {
    pub width: u32,
    pub height: u32,
    #[serde(with = "Color")]
    pub color: image_canvas::color::Color,
    #[serde(with = "Texel")]
    pub texel: buffer::Texel,
}

#[derive(Deserialize)]
#[serde(remote = "command::Rectangle")]
pub struct RectangleU32 {
    pub x: u32,
    pub y: u32,
    pub max_x: u32,
    pub max_y: u32,
}

#[derive(Deserialize)]
#[serde(remote = "buffer::Texel")]
pub struct Texel {
    #[serde(with = "Block")]
    block: buffer::Block,
    #[serde(with = "SampleBits")]
    bits: buffer::SampleBits,
    #[serde(with = "SampleParts")]
    parts: buffer::SampleParts,
}

#[derive(Deserialize)]
#[serde(remote = "color::Primaries")]
pub enum Primaries {
    Xyz,
    Bt601_525,
    Bt601_625,
    Bt709,
    Smpte240,
    Bt2020,
    Bt2100,
}

#[derive(Deserialize)]
#[serde(remote = "color::Transfer")]
pub enum Transfer {
    Bt709,
    Bt601,
    Smpte240,
    Linear,
    Srgb,
    Bt2020_10bit,
    Smpte2084,
}

#[derive(Deserialize)]
#[serde(remote = "color::Luminance")]
pub enum Luminance {
    Sdr,
    Hdr,
    AdobeRgb,
    DciP3,
}

#[derive(Deserialize)]
#[serde(remote = "buffer::Whitepoint")]
pub enum Whitepoint {
    A,
    B,
    C,
    D55,
    D65,
    D75,
    E,
    F2,
    F7,
    F11,
}

#[derive(Deserialize)]
#[serde(remote = "buffer::Block")]
pub enum Block {
    Pixel,
}

#[derive(Deserialize)]
#[serde(remote = "buffer::SampleBits")]
pub enum SampleBits {}

#[derive(Deserialize)]
#[serde(remote = "buffer::SampleParts")]
pub enum SampleParts {}
