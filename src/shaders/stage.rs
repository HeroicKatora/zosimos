/// Detailed structs for the stage shader.
use crate::buffer::{SampleBits, SampleParts, Transfer};
use wgpu::TextureFormat;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct XyzParameter {
    pub bits: SampleBits,
    pub parts: SampleParts,
    pub transfer: Transfer,
}

/// Defines the bit representation we use for our own encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum StageKind {
    R8ui = 0,
    R16ui = 1,
    R32ui = 2,
    Rgba16ui = 3,
    Rgba32ui = 4,
}

impl XyzParameter {
    pub(crate) fn serialize_std140(&self) -> [u32; 4] {
        [
            self.transfer as u32,
            self.parts as u32,
            self.bits as u32,
            0,
        ]
    }

    pub(crate) fn stage_kind(&self) -> Option<StageKind> {
        Some(match self.bits.bytes() {
            1 => StageKind::R8ui,
            2 => StageKind::R16ui,
            4 => StageKind::R32ui,
            8 => StageKind::Rgba16ui,
            16 => StageKind::Rgba32ui,
            _ => return None,
        })
    }

    pub(crate) fn linear_format(&self) -> TextureFormat {
        TextureFormat::Rgba16Float
    }
}

impl StageKind {
    pub const ALL: [Self; 5] = [
        Self::R8ui,
        Self::R16ui,
        Self::R32ui,
        Self::Rgba16ui,
        Self::Rgba32ui,
    ];

    pub(crate) fn encode_entry_point(self) -> &'static str {
        match self {
            Self::R8ui => "encode_r8ui",
            Self::R16ui => "encode_r16ui",
            Self::R32ui => "encode_r32ui",
            Self::Rgba16ui => "encode_rgba16ui",
            Self::Rgba32ui => "encode_rgba32ui",
        }
    }

    pub(crate) fn decode_entry_point(self) -> &'static str {
        match self {
            Self::R8ui => "decode_r8ui",
            Self::R16ui => "decode_r16ui",
            Self::R32ui => "decode_r32ui",
            Self::Rgba16ui => "decode_rgba16ui",
            Self::Rgba32ui => "decode_rgba32ui",
        }
    }

    pub(crate) fn texture_format(self) -> TextureFormat {
        match self {
            Self::R8ui => TextureFormat::R8Uint,
            Self::R16ui => TextureFormat::R16Uint,
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
}
