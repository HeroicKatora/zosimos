use super::{BufferInitContent, FragmentShaderData, FragmentShaderKey};
use std::borrow::Cow;

/// a linear transformation on rgb color.
pub const SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spirv/fractal_noise.frag.v"));

#[derive(Clone, Debug, PartialEq)]
pub struct Shader {
    pub num_octaves: u32,
    pub initial_amplitude: f32,
    pub amplitude_damping: f32,
    pub grid_scale: [f32; 2],
}

impl Shader {
    /// Construct a fractal noise pattern with a specific number of octaves.
    pub fn with_octaves(num_octaves: u32) -> Self {
        let amplitude_damping = 1.0f32;
        let initial_amplitude = 1.0 / (num_octaves as f64) as f32;
        let grid_scale = [100.0, 100.0];

        Self {
            num_octaves,
            grid_scale,
            initial_amplitude,
            amplitude_damping,
        }
    }

    /// Set damping and correct the initial amplitude such that the amplitudes over
    /// all octaves sum to 1.
    pub fn set_damping(&mut self, damping: f32) {
        // post condition: initial_amplitude * [ \Sum_i=0^{num_octaves} {damping}^i ] = 1
        let n = self.num_octaves as f64 as f32;
        // The usual summation trick for geometric series
        // total_factor = [ \Sum_i=0^{num_octaves} {damping}^i ] * (1 - damping)
        //              = 1 - {damping}^{num_octaves}
        let total_factor = 1.0 - damping.powf(n);
        self.initial_amplitude = if total_factor.abs() < 1e-7 {
            1.0
        } else {
            (1.0 - damping) / total_factor
        };
        self.amplitude_damping = damping;
    }
}

#[test]
fn test_set_damping() {
    let octaves = 8;
    let mut params = Shader::with_octaves(octaves);
    params.set_damping(0.5);

    let mut amp = params.initial_amplitude;
    let mut summed_amps = 0.0;
    for _ in 0..octaves {
        summed_amps += amp;
        amp *= params.amplitude_damping;
    }
    assert!((1. - summed_amps).abs() < 1e-6);
}

impl FragmentShaderData for Shader {
    fn key(&self) -> Option<FragmentShaderKey> {
        Some(FragmentShaderKey::FractalNoise)
    }

    fn spirv_source(&self) -> Cow<'static, [u8]> {
        Cow::Borrowed(SHADER)
    }

    fn binary_data(&self, buffer: &mut Vec<u8>) -> Option<BufferInitContent> {
        let Self {
            num_octaves,
            initial_amplitude,
            amplitude_damping,
            grid_scale,
        } = self;
        let mut buffer_content = BufferInitContent::builder(buffer);
        buffer_content.extend_from_pods(&[grid_scale[0], grid_scale[1]]);
        buffer_content.extend_from_pods(&[*initial_amplitude]);
        buffer_content.extend_from_pods(&[*amplitude_damping]);
        buffer_content.extend_from_pods(&[*num_octaves]);
        buffer_content.align_by_exponent(3);

        Some(buffer_content.build())
    }

    fn num_args(&self) -> u32 {
        0
    }
}
