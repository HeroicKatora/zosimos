use std::sync::Arc;

use crate::compute::{Compute, ComputeTailCommands};
use crate::winit::{WindowSurface, WindowedSurface};

use zosimos::buffer::{Color, Descriptor, SampleParts, Texel, Transfer};
use zosimos::command;
use zosimos::pool::{GpuKey, Pool, PoolKey, SwapChain};
use zosimos::program::{Capabilities, CompileError, LaunchError, Program};
use zosimos::run::{Executable, StepLimits};

use wgpu::{Adapter, Instance, SurfaceConfiguration};

pub struct Surface {
    /// The adapter for accessing devices.
    adapter: Adapter,
    /// Mirrored configuration of the surface.
    config: SurfaceConfiguration,
    /// The driver instance used for drawing.
    instance: Instance,
    /// The surface drawing into the window.
    inner: WindowSurface,
    /// Our private resource pool of the surface.
    pool: Pool,
    /// The pool entry.
    entry: PoolEntry,
    /// The runtime state from stealth paint.
    runtimes: Runtimes,
    ///
    commands: ComputeTailCommands,
}

#[derive(Debug)]
pub enum PresentationError {
    GpuDeviceLost,
}

#[derive(Debug)]
struct NormalizingError {
    fail: String,
}

/// The pool entry of our surface declarator.
struct PoolEntry {
    gpu: Option<GpuKey>,
    key: Option<PoolKey>,
    presentable: PoolKey,
    descriptor: Descriptor,
}

#[derive(Default)]
struct Runtimes {
    /// An executable color normalizing the chosen output picture into the output texture, then
    /// writing it as an output.
    normalizing: Option<NormalizingExe>,
}

/// Another compiled program, which puts the image onto the screen.
struct NormalizingExe {
    exe: Arc<Executable>,
    in_descriptor: Descriptor,
    out_descriptor: Descriptor,
    in_reg: command::Register,
    out_reg: command::Register,
}

impl Surface {
    pub fn new(window: &WindowSurface) -> Self {
        const ANY: wgpu::Backends = wgpu::Backends::all();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

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

        eprintln!("Rendering on {:?}", adapter.get_info());

        let (color, texel);
        let capabilities = inner.get_capabilities(&adapter);

        let preferred_format = match capabilities.formats.get(0) {
            None => {
                log::warn!("No supported surface formats …");
                color = Color::SRGB;
                texel = Texel::new_u8(SampleParts::RgbA);
                wgpu::TextureFormat::Rgba8Unorm
            }
            Some(wgpu::TextureFormat::Rgba8Unorm) => {
                log::warn!("Using format {:?}", wgpu::TextureFormat::Rgba8Unorm);
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
                    _ => unreachable!("That's not the right color"),
                };

                texel = Texel::new_u8(SampleParts::RgbA);
                wgpu::TextureFormat::Rgba8Unorm
            }
            Some(wgpu::TextureFormat::Rgba8UnormSrgb) => {
                log::warn!("Using format {:?}", wgpu::TextureFormat::Rgba8UnormSrgb);

                color = Color::SRGB;
                texel = Texel::new_u8(SampleParts::RgbA);
                wgpu::TextureFormat::Rgba8UnormSrgb
            }
            Some(wgpu::TextureFormat::Bgra8UnormSrgb) | _ => {
                log::warn!("Using format {:?}", wgpu::TextureFormat::Bgra8UnormSrgb);

                color = Color::SRGB;
                texel = Texel::new_u8(SampleParts::BgrA);
                wgpu::TextureFormat::Bgra8UnormSrgb
            }
        };

        let (width, height) = window.inner_size();

        let config = SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: preferred_format,
            width,
            height,
            present_mode: wgpu::PresentMode::AutoVsync,
            view_formats: [preferred_format].to_vec(),
            alpha_mode: Default::default(),
            desired_maximum_frame_latency: 0,
        };

        let descriptor = Descriptor {
            color,
            ..Descriptor::with_texel(texel, width, height).unwrap()
        };

        let empty = image::DynamicImage::new_rgba16(0, 0);

        let mut that = Surface {
            adapter,
            config,
            inner,
            instance,
            pool: Pool::new(),
            entry: PoolEntry {
                gpu: None,
                key: None,
                presentable: PoolKey::null(),
                descriptor,
            },
            runtimes: Runtimes::default(),
            commands: ComputeTailCommands::default(),
        };

        let gpu = that.reconfigure_gpu();
        that.entry.gpu = Some(gpu);
        let surface = that.pool.declare(that.descriptor());
        that.entry.key = Some(surface.key());
        that.reconfigure_surface().unwrap();
        // Create a nul image to ''present'' while booting.
        let presentable = that.pool.declare(Descriptor::with_srgb_image(&empty)).key();
        that.entry.presentable = presentable;

        that
    }

    /// Create a pool that shares the device with this surface.
    ///
    /// The pool can separately render textures which the surface's pool can then display.
    pub fn configure_pool(&mut self, pool: &mut Pool) -> GpuKey {
        log::info!("Surface reconfiguring pool device");
        let internal_key = self.reconfigure_gpu();

        self.pool
            .share_device(internal_key, pool)
            .expect("maintained incorrect gpu key")
    }

    /// Create a swap chain in our pool, for the presented texture.
    pub fn configure_swap_chain(&mut self, n: usize) -> SwapChain {
        self.pool.swap_chain(self.entry.presentable, n)
    }

    pub(crate) fn reconfigure_compute(&mut self, compute: &Compute) {
        compute.acquire(&mut self.commands);
    }

    /// Change the base device.
    pub(crate) fn reconfigure_gpu(&mut self) -> GpuKey {
        if let Some(gpu) = self.entry.gpu {
            gpu
        } else {
            log::info!("No gpu key, device lost or not initialized?");
            let mut descriptor = Program::minimal_device_descriptor();
            descriptor.required_limits.max_texture_dimension_1d = 4096;
            descriptor.required_limits.max_texture_dimension_2d = 4096;

            let gpu = self
                .pool
                .request_device(&self.adapter, descriptor)
                .expect("to get a device");

            gpu
        }
    }

    /// Get a new presentable image from the CPU host.
    pub fn set_image(&mut self, image: &image::DynamicImage) {
        log::info!("Uploading DynamicImage {:?} to GPU buffer", image.color());
        let gpu = self.reconfigure_gpu();
        let key = self.entry.presentable;
        let mut entry = self.pool.entry(key).unwrap();
        entry.set_srgb(&image);
        self.pool.upload(key, gpu).unwrap();
    }

    /// Get a new presentable image from the swap chain.
    pub fn set_from_swap_chain(&mut self, chain: &mut SwapChain) {
        chain.present(&mut self.pool)
    }

    pub fn get_current_texture(&mut self) -> Result<wgpu::SurfaceTexture, wgpu::SurfaceError> {
        self.inner.get_current_texture()
    }

    pub fn present_to_texture(&mut self, surface_tex: &mut wgpu::SurfaceTexture) {
        let gpu = match self.entry.gpu {
            Some(key) => key,
            None => {
                log::warn!("No gpu to paint with.");
                return;
            }
        };

        let surface = match self.entry.key {
            Some(key) => key,
            None => {
                log::warn!("No surface to paint to.");
                return;
            }
        };

        let present = self.entry.presentable;

        #[cfg(not(target_arch = "wasm32"))]
        let start = std::time::Instant::now();

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

        self.pool
            .entry(surface)
            .unwrap()
            .replace_texture_unguarded(&mut surface_tex.texture, gpu);

        let mut run = normalize
            .exe
            .from_pool(&mut self.pool)
            .expect("Valid pool for our own executable");

        #[cfg(not(target_arch = "wasm32"))]
        let end = std::time::Instant::now();
        #[cfg(not(target_arch = "wasm32"))]
        log::warn!("Time setup {:?}", end.saturating_duration_since(start));
        #[cfg(not(target_arch = "wasm32"))]
        let start = end;

        // Bind the input.
        run.bind(in_reg, present)
            .expect("Valid binding for our executable input");
        // Bind the output.
        run.bind_render(out_reg, surface)
            .expect("Valid binding for our executable output");
        log::warn!("Sub- optimality: {:?}", surface_tex.suboptimal);
        let recovered = run.recover_buffers();
        log::warn!("{:?}", recovered);

        let mut running = normalize
            .exe
            .launch(run)
            .expect("Valid binding to start our executable");

        #[cfg(not(target_arch = "wasm32"))]
        let end = std::time::Instant::now();
        #[cfg(not(target_arch = "wasm32"))]
        log::warn!("Time launch {:?}", end.saturating_duration_since(start));
        #[cfg(not(target_arch = "wasm32"))]
        let start = end;

        // Ensure our cache does not grow infinitely.
        self.pool.clear_cache();

        // FIXME: No. Async. Luckily this is straightforward.
        while running.is_running() {
            let limits = StepLimits::new().with_steps(usize::MAX);
            let mut step = running
                .step_to(limits)
                .expect("Valid binding to start our executable");
            step.block_on()
                .expect("Valid binding to block on our execution");
        }

        #[cfg(not(target_arch = "wasm32"))]
        let end = std::time::Instant::now();
        #[cfg(not(target_arch = "wasm32"))]
        log::warn!("Time run {:?}", end.saturating_duration_since(start));
        #[cfg(not(target_arch = "wasm32"))]
        let start = end;

        log::warn!("{:?}", running.resources_used());
        let mut retire = running.retire_gracefully(&mut self.pool);
        retire
            .input(in_reg)
            .expect("Valid to retire input of our executable");
        retire
            .render(out_reg)
            .expect("Valid to retire outputof our executable");

        retire.prune();
        let retired = retire.retire_buffers();
        log::warn!("{:?}", retired);
        retire.finish();

        self.pool
            .entry(surface)
            .unwrap()
            .replace_texture_unguarded(&mut surface_tex.texture, gpu);

        #[cfg(not(target_arch = "wasm32"))]
        let end = std::time::Instant::now();
        #[cfg(not(target_arch = "wasm32"))]
        log::warn!("Time finish {:?}", end.saturating_duration_since(start));
    }

    pub fn descriptor(&self) -> Descriptor {
        self.entry.descriptor.clone()
    }

    pub fn lost(&mut self) -> Result<(), PresentationError> {
        self.reconfigure_surface()
    }

    pub fn outdated(&mut self) -> Result<(), PresentationError> {
        self.reconfigure_surface()
    }

    fn reconfigure_surface(&mut self) -> Result<(), PresentationError> {
        let surface = match self.entry.key {
            Some(key) => key,
            None => {
                log::warn!("No surface to paint to.");
                return Err(PresentationError::GpuDeviceLost);
            }
        };

        let phys = self.inner.window().inner_size();
        self.config.width = phys.width;
        self.config.height = phys.height;

        let mut surface = self.pool.entry(surface).unwrap();
        let old_descriptor = surface.descriptor().clone();
        surface.declare(Descriptor {
            color: old_descriptor.color,
            ..Descriptor::with_texel(old_descriptor.texel, phys.width, phys.height).unwrap()
        });

        // Could also be done in `get_or_insert_normalizing_exe` by storing the relevant input
        // parameters or by assigning version increments to each configuration.
        // FIXME: this would help reuse partial state.
        self.runtimes.normalizing = None;

        let dev = match self.pool.iter_devices().next() {
            Some(dev) => dev,
            None => {
                eprintln!("Lost device for screen rendering");
                return Err(PresentationError::GpuDeviceLost);
            }
        };

        log::info!("Reconfigured surface {:?}", &self.config);
        self.inner.surface().configure(dev, &self.config);

        Ok(())
    }
}

impl WindowedSurface for Surface {
    fn recreate(&mut self) {}

    fn from_window(window: WindowSurface) -> Self {
        Surface::new(&window)
    }
}

impl Runtimes {
    pub(crate) fn get_or_insert_normalizing_exe(
        &mut self,
        // The descriptor of the to-render output.
        input: Descriptor,
        // The surface descriptor.
        surface: Descriptor,
        // Capabilities to use for conversion.
        caps: Capabilities,
    ) -> Result<&mut NormalizingExe, NormalizingError> {
        if let Some(normalize) = &self.normalizing {
            if input == normalize.in_descriptor && surface == normalize.out_descriptor {
                return Ok(self.normalizing.as_mut().unwrap());
            }
        }

        let mut cmd = command::CommandBuffer::default();
        let in_reg = cmd.input(input.clone())?;
        let resized = cmd.resize(in_reg, surface.size())?;
        let converted = cmd.color_convert(resized, surface.color.clone(), surface.texel.clone())?;
        let (out_reg, _desc) = cmd.render(converted)?;

        let program = cmd.compile()?;
        let exe = program.lower_to(caps)?;

        log::info!("{}", exe.dot());

        Ok(self.normalizing.get_or_insert(NormalizingExe {
            exe: Arc::new(exe),
            in_descriptor: input,
            out_descriptor: surface,
            in_reg,
            out_reg,
        }))
    }
}

impl From<CompileError> for NormalizingError {
    #[track_caller]
    fn from(err: CompileError) -> Self {
        let location = core::panic::Location::caller();
        NormalizingError {
            fail: format!("At {:?}: {:?}", location, err),
        }
    }
}

impl From<command::CommandError> for NormalizingError {
    #[track_caller]
    fn from(err: command::CommandError) -> Self {
        let location = core::panic::Location::caller();
        NormalizingError {
            fail: format!("At {:?}: {:?}", location, err),
        }
    }
}

impl From<LaunchError> for NormalizingError {
    #[track_caller]
    fn from(err: LaunchError) -> Self {
        let location = core::panic::Location::caller();
        NormalizingError {
            fail: format!("At {:?}: {:?}", location, err),
        }
    }
}
