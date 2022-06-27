use crate::winit::Window;

use stealth_paint::buffer::{Color, Descriptor, SampleParts, Texel, Transfer};
use stealth_paint::pool::{GpuKey, Pool, PoolKey};
use stealth_paint::program::Program;
use wgpu::{Adapter, Instance, Surface as WgpuSurface, SurfaceConfiguration};

pub struct Surface {
    /// The adapter for accessing devices.
    adapter: Adapter,
    /// Mirrored configuration of the surface.
    config: SurfaceConfiguration,
    /// The driver instance used for drawing.
    instance: Instance,
    /// The surface drawing into the window.
    inner: WgpuSurface,
    /// Our private resource pool of the surface.
    pool: Pool,
    /// The pool entry.
    entry: PoolEntry,
}

/// The pool entry of our surface declarator.
struct PoolEntry {
    gpu: Option<GpuKey>,
    key: Option<PoolKey>,
    descriptor: Descriptor,
}

impl Surface {
    pub fn new(window: &Window) -> Self {
        const ANY: wgpu::Backends = wgpu::Backends::VULKAN;

        let instance = wgpu::Instance::new(ANY);
        let inner = window.create_surface(&instance);

        let adapter = Program::request_compatible_adapter(
            &instance,
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&inner),
                force_fallback_adapter: false,
            },
        )
        .expect("to get an adapter");

        let (width, height) = window.inner_size();
        let (color, texel);
        let config = SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: match inner.get_preferred_format(&adapter) {
                None => {
                    color = Color::SRGB;
                    texel = Texel::new_u8(SampleParts::RgbA);
                    wgpu::TextureFormat::Rgba8UnormSrgb
                }
                Some(wgpu::TextureFormat::Rgba8Unorm) => {
                    color = match Color::SRGB {
                        Color::Rgb {
                            luminance,
                            transfer: _,
                            primary,
                            whitepoint,
                        } => Color::Rgb {
                            luminance,
                            primary,
                            whitepoint,
                            transfer: Transfer::Linear,
                        },
                        _ => unreachable!(),
                    };

                    texel = Texel::new_u8(SampleParts::RgbA);
                    wgpu::TextureFormat::Rgba8UnormSrgb
                }
                Some(wgpu::TextureFormat::Rgba8UnormSrgb) | _ => {
                    color = Color::SRGB;
                    texel = Texel::new_u8(SampleParts::RgbA);
                    wgpu::TextureFormat::Rgba8UnormSrgb
                }
            },
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
        };

        let descriptor = Descriptor {
            color,
            ..Descriptor::with_texel(texel, width, height).unwrap()
        };

        let mut that = Surface {
            adapter,
            config,
            inner,
            instance,
            pool: Pool::new(),
            entry: PoolEntry {
                gpu: None,
                key: None,
                descriptor,
            },
        };

        let mut pool = Pool::new();
        let gpu = that.configure_pool(&mut pool);
        that.entry.gpu = Some(gpu);
        let texture = pool.declare(that.descriptor());
        that.entry.key = Some(texture.key());
        that.pool = pool;
        that.lost();

        that
    }

    pub fn configure_pool(&self, pool: &mut Pool) -> GpuKey {
        pool.request_device(&self.adapter, Program::minimal_device_descriptor())
            .expect("to get a device")
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        todo!()
    }

    pub fn get_current_texture(&mut self) -> Result<wgpu::SurfaceTexture, wgpu::SurfaceError> {
        self.inner.get_current_texture()
    }

    pub fn descriptor(&self) -> Descriptor {
        self.entry.descriptor.clone()
    }

    pub fn lost(&mut self) {
        let dev = match self.pool.iter_devices().next() {
            Some(dev) => dev,
            None => {
                eprintln!("Lost device for screen rendering");
                return;
            }
        };

        self.inner.configure(dev, &self.config);
    }
}
