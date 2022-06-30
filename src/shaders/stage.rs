use crate::buffer::{SampleBits, SampleParts, Transfer as RgbTransfer};
/// Detailed structs for the stage shader.
use core::num::NonZeroU32;
use wgpu::TextureFormat;

/// A potentially non-linear transform we apply to the linear values before we store them and
/// before we load them. This is not necessarily a opto-electrical transfer function because we
/// might do this even for non-RGB color spaces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Transfer {
    Rgb(RgbTransfer),
    LabLch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct XyzParameter {
    pub bits: SampleBits,
    pub parts: SampleParts,
    pub transfer: Transfer,
}

/// Defines the bit representation we use for our own coding of texels and pixels.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum StageKind {
    /// Each texel is 8 bit and we operate on 4 texels horizontally at the same time.
    R8uiX4 = 0,
    /// Each texel is 16 bit and we operate on 2 texels horizontally at the same time.
    R16uiX2 = 1,
    /// Each texel is 32 bit.
    R32ui = 2,
    /// Each texel is 64 bit and we decode it from 16-bit RGBA.
    Rgba16ui = 3,
    /// Each texel is 128 bit and we decode it from 32-bit RGBA.
    /// That's scarily large.
    Rgba32ui = 4,
}

impl XyzParameter {
    pub(crate) fn serialize_std140(&self) -> [u32; 4] {
        [
            self.transfer.as_u32(),
            Self::serialize_parts(self.parts),
            Self::serialize_bits(self.bits),
            // Upper bits are still reserved for texel block size.
            self.horizontal_subfactor() & 0xff,
        ]
    }

    pub(crate) fn stage_kind(&self) -> Option<StageKind> {
        Some(match self.bits.bytes() {
            1 => StageKind::R8uiX4,
            2 => StageKind::R16uiX2,
            4 => StageKind::R32ui,
            8 => StageKind::Rgba16ui,
            16 => StageKind::Rgba32ui,
            _ => return None,
        })
    }

    pub(crate) fn serialize_parts(parts: SampleParts) -> u32 {
        use SampleParts as S;
        match parts {
            S::A => 0,
            S::R => 1,
            S::G => 2,
            S::B => 3,
            S::Luma => 4,
            S::LumaA => 5,
            S::Rgb => 6,
            S::Bgr => 7,
            S::RgbA => 8,
            S::BgrA => 10,
            S::ARgb => 12,
            S::ABgr => 14,
            S::Yuv => 16,
            S::Lab => 17,
            S::LabA => 18,
            S::Lch => 19,
            S::LchA => 20,
            _ => todo!("{:?}", parts),
        }
    }

    pub(crate) fn serialize_bits(bits: SampleBits) -> u32 {
        use SampleBits as S;
        match bits {
            S::UInt8 => 0,
            S::UInt332 => 1,
            S::UInt233 => 2,
            S::UInt16 => 3,
            S::UInt4x4 => 4,
            S::UInt565 => 7,
            S::UInt8x2 => 8,
            S::UInt8x3 => 9,
            S::UInt8x4 => 10,
            S::UInt16x2 => 11,
            S::UInt16x3 => 12,
            S::UInt16x4 => 13,
            S::UInt2101010 => 14,
            S::UInt1010102 => 15,
            S::Float16x4 => 18,
            S::Float32x4 => 19,
            _ => todo!("{:?}", bits),
        }
    }

    pub(crate) fn horizontal_subfactor(&self) -> u32 {
        match self.bits.bytes() {
            1 => 4,
            2 => 2,
            _ => 1,
        }
    }

    pub(crate) fn linear_format(&self) -> TextureFormat {
        TextureFormat::Rgba16Float
    }
}

impl StageKind {
    pub const ALL: [Self; 5] = [
        Self::R8uiX4,
        Self::R16uiX2,
        Self::R32ui,
        Self::Rgba16ui,
        Self::Rgba32ui,
    ];

    pub(crate) fn encode_entry_point(self) -> &'static str {
        match self {
            Self::R8uiX4 => "encode_r8ui",
            Self::R16uiX2 => "encode_r16ui",
            Self::R32ui => "encode_r32ui",
            Self::Rgba16ui => "encode_rgba16ui",
            Self::Rgba32ui => "encode_rgba32ui",
        }
    }

    pub(crate) fn decode_entry_point(self) -> &'static str {
        match self {
            Self::R8uiX4 => "decode_r8ui",
            Self::R16uiX2 => "decode_r16ui",
            Self::R32ui => "decode_r32ui",
            Self::Rgba16ui => "decode_rgba16ui",
            Self::Rgba32ui => "decode_rgba32ui",
        }
    }

    pub(crate) fn decode_src(self) -> &'static [u8] {
        match self {
            Self::R8uiX4 => include_bytes!(concat!(env!("OUT_DIR"), "/spirv/stage_d8ui.frag.v")),
            Self::R16uiX2 => include_bytes!(concat!(env!("OUT_DIR"), "/spirv/stage_d16ui.frag.v")),
            Self::R32ui => include_bytes!(concat!(env!("OUT_DIR"), "/spirv/stage_d32ui.frag.v")),
            _ => todo!("{:?}", self),
        }
    }

    pub(crate) fn encode_src(self) -> &'static [u8] {
        match self {
            Self::R8uiX4 => include_bytes!(concat!(env!("OUT_DIR"), "/spirv/stage_e8ui.frag.v")),
            Self::R16uiX2 => include_bytes!(concat!(env!("OUT_DIR"), "/spirv/stage_e16ui.frag.v")),
            Self::R32ui => include_bytes!(concat!(env!("OUT_DIR"), "/spirv/stage_e32ui.frag.v")),
            _ => todo!("{:?}", self),
        }
    }

    pub(crate) fn texture_format(self) -> TextureFormat {
        // Chosen with compatibility to:
        // <https://www.w3.org/TR/webgpu/#plain-color-formats>
        match self {
            // We could also use Rgba8Uint here, however, there is no 16-bit/2 channel equivalent
            // so we rather use a more uniform choice.
            Self::R8uiX4 => TextureFormat::R32Uint,
            Self::R16uiX2 => TextureFormat::R32Uint,
            Self::R32ui => TextureFormat::R32Uint,
            Self::Rgba16ui => TextureFormat::Rgba16Uint,
            Self::Rgba32ui => TextureFormat::Rgba32Uint,
        }
    }

    pub(crate) fn decode_binding(self) -> u32 {
        self as u32
    }

    pub(crate) fn encode_binding(self) -> u32 {
        (self as u32) + 16
    }

    pub(crate) fn horizontal_subfactor(self) -> u32 {
        match self {
            Self::R8uiX4 => 4,
            Self::R16uiX2 => 2,
            _ => 1,
        }
    }

    pub(crate) fn stage_size(self, (w, h): (NonZeroU32, NonZeroU32)) -> (NonZeroU32, NonZeroU32) {
        let sub = self.horizontal_subfactor();
        let w = w.get() / sub + u32::from(w.get() % sub > 0);
        (NonZeroU32::new(w).unwrap(), h)
    }
}

impl Transfer {
    pub fn as_u32(self) -> u32 {
        match self {
            Transfer::Rgb(t) => t as u32,
            Transfer::LabLch => 0x100,
        }
    }
}

impl From<image_canvas::color::Transfer> for Transfer {
    fn from(t: RgbTransfer) -> Self {
        Transfer::Rgb(t)
    }
}
