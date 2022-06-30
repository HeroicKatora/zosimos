use crate::winit::Window;

use stealth_paint::buffer::{Color, Descriptor, SampleParts, Texel, Transfer};
use stealth_paint::command;
use stealth_paint::pool::{GpuKey, Pool, PoolKey};
use stealth_paint::program::{Capabilities, Program};
use stealth_paint::run::Executable;
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
    /// The runtime state from stealth paint.
    runtimes: Runtimes,
}

/// The pool entry of our surface declarator.
struct PoolEntry {
    gpu: Option<GpuKey>,
    key: Option<PoolKey>,
    presentable: Option<PoolKey>,
    descriptor: Descriptor,
}

#[derive(Default)]
struct Runtimes {
    /// An executable color normalizing the chosen output picture into the output texture, then
    /// writing it as an output.
    normalizing: Option<NormalizingExe>,
}

struct NormalizingExe {
    exe: Executable,
    in_reg: command::Register,
    out_reg: command::Register,
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_DST,
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
                Some(wgpu::TextureFormat::Rgba8UnormSrgb) => {
                    color = Color::SRGB;
                    texel = Texel::new_u8(SampleParts::RgbA);
                    wgpu::TextureFormat::Rgba8UnormSrgb
                }
                Some(wgpu::TextureFormat::Bgra8UnormSrgb) | _ => {
                    color = Color::SRGB;
                    texel = Texel::new_u8(SampleParts::BgrA);
                    wgpu::TextureFormat::Bgra8UnormSrgb
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
                presentable: None,
                descriptor,
            },
            runtimes: Runtimes::default(),
        };

        let mut pool = Pool::new();
        let gpu = that.configure_pool(&mut pool);
        that.entry.gpu = Some(gpu);
        let surface = pool.declare(that.descriptor());
        that.entry.key = Some(surface.key());
        that.pool = pool;
        that.lost();

        that
    }

    pub fn configure_pool(&self, pool: &mut Pool) -> GpuKey {
        pool.request_device(&self.adapter, Program::minimal_device_descriptor())
            .expect("to get a device")
    }

    pub fn set_image(&mut self, image: &image::DynamicImage) {
        if let Some(key) = self.entry.presentable {
            let mut entry = self.pool.entry(key).unwrap();
            entry.set_srgb(&image);
        } else {
            let entry = self.pool.insert_srgb(image);
            self.entry.presentable = Some(entry.key());
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        todo!()
    }

    pub fn get_current_texture(&mut self) -> Result<wgpu::SurfaceTexture, wgpu::SurfaceError> {
        self.inner.get_current_texture()
    }

    pub fn present_to_texture(&mut self, surface_tex: &mut wgpu::SurfaceTexture) {
        let gpu = match self.entry.gpu {
            Some(key) => key,
            None => {
                log::warn!("No gpu to paint with.");
                return
            },
        };

        let surface = match self.entry.key {
            Some(key) => key,
            None => {
                log::warn!("No surface to paint to.");
                return
            },
        };

        let present = match self.entry.presentable {
            Some(key) => key,
            None => {
                log::warn!("No image to paint from.");
                return
            },
        };

        let present_desc = self.pool.entry(present).unwrap().descriptor();
        let surface_desc = self.pool.entry(surface).unwrap().descriptor();

        let device = self.pool.iter_devices().next().unwrap();
        let capabilities = Capabilities::from(device);

        let normalize = self
            .runtimes
            .get_or_insert_normalizing_exe(present_desc, surface_desc, capabilities)
            .expect("Should be able to build resize");

        let in_reg = normalize.in_reg;
        let out_reg = normalize.out_reg;

        self.pool.entry(surface).unwrap().replace_texture_unguarded(&mut surface_tex.texture, gpu);

        let mut run = normalize
            .exe
            .from_pool(&mut self.pool)
            .expect("Valid pool for our own executable");

        // Bind the input.
        run.bind(in_reg, present)
            .expect("Valid binding for our executable input");
        // Bind the output.
        run.bind_output(out_reg, surface)
            .expect("Valid binding for our executable output");
        log::warn!("{:?}", run.recover_buffers());

        let mut running = normalize
            .exe
            .launch(run)
            .expect("Valid binding to start our executable");

        // Ensure our cache does not grow infinitely.
        self.pool.clear_cache();

        while running.is_running() {
            let mut step = running
                .step()
                .expect("Valid binding to start our executable");
            step.block_on()
                .expect("Valid binding to block on our execution");
        }

        log::warn!("{:?}", running.resources_used());
        let mut retire = running.retire_gracefully(&mut self.pool);
        retire
            .input(in_reg)
            .expect("Valid to retire input of our executable");
        retire
            .output(out_reg)
            .expect("Valid to retire outputof our executable");

        retire.prune();
        log::warn!("{:?}", retire.retire_buffers());
        retire.finish();

        self.pool.entry(surface).unwrap().replace_texture_unguarded(&mut surface_tex.texture, gpu);
        log::info!("Presented!");
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

impl Runtimes {
    pub(crate) fn get_or_insert_normalizing_exe(
        &mut self,
        input: Descriptor,
        surface: Descriptor,
        caps: Capabilities,
    ) -> Option<&mut NormalizingExe> {
        if self.normalizing.is_some() {
            // FIXME: Polonius NLL.
            return self.normalizing.as_mut();
        }

        let mut cmd = command::CommandBuffer::default();
        let in_reg = cmd.input(input).ok()?;
        let resized = cmd.resize(in_reg, surface.size()).ok()?;
        let converted = cmd
            .color_convert(resized, surface.color.clone(), surface.texel.clone())
            .ok()?;
        let (out_reg, _desc) = cmd.output(converted).ok()?;
        eprintln!("{:?}", _desc);

        let program = cmd.compile().ok()?;
        let exe = program.lower_to(caps).ok()?;

        Some(self.normalizing.get_or_insert(NormalizingExe {
            exe,
            in_reg,
            out_reg,
        }))
    }
}
