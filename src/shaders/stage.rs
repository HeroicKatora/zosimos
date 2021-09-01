/// Detailed structs for the stage shader.
use core::num::NonZeroU32;
use crate::buffer::{SampleBits, SampleParts, Transfer};
use wgpu::TextureFormat;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct XyzParameter {
    pub bits: SampleBits,
    pub parts: SampleParts,
    pub transfer: Transfer,
}

/// Defines the bit representation we use for our own coding of texels and pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
            self.transfer as u32,
            self.parts as u32,
            self.bits as u32,
            self.horizontal_subfactor(),
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
            Self::R32ui => include_bytes!(concat!(env!("OUT_DIR"), "/spirv/stage_d32ui.frag.v")),
            _ => todo!("{:?}", self),
        }
    }

    pub(crate) fn encode_src(self) -> &'static [u8] {
        match self {
            Self::R8uiX4 => include_bytes!(concat!(env!("OUT_DIR"), "/spirv/stage_e8ui.frag.v")),
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

    pub(crate) fn stage_size(self, (w, h): (NonZeroU32, NonZeroU32))
        -> (NonZeroU32, NonZeroU32)
    {
        let sub = self.horizontal_subfactor();
        let w = w.get() / sub + u32::from(w.get() % sub > 0);
        (NonZeroU32::new(w).unwrap(), h)
    }
}
