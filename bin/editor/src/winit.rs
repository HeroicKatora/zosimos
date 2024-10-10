/// The winit related state of our application.
///
/// We have two responsibilities:
/// - rewrite input events into program semantics, based on the its current input state. The callee
///     will cooperatively indicate when it changes.
use winit::{event::*, event_loop::EventLoop, window::WindowAttributes};

use std::sync::Arc;
use std::thread::Scope;

pub struct Window {
    event_loop: EventLoop<()>,
    window: Option<Arc<winit::window::Window>>,
}

/// Combination of window/surface that fulfills safety property of `create_surface`.
pub struct WindowSurface {
    surface: Option<wgpu::Surface<'static>>,
    window: Arc<winit::window::Window>,
}

/// The input modality.
pub enum ModalContext {
    Main,
}

#[derive(Debug)]
pub enum ModalEvent {
    ExitPressed,
    MainEventsCleared,
    RedrawRequested,
    Resized,
}

/// The redraw errors handled by the window.
#[derive(Debug)]
pub enum RedrawError {
    Lost,
    Outdated,
}

pub trait ModalEditor {
    type Compute;
    type Surface;
    /// Handle one input event.
    fn event(&mut self, _: ModalEvent, _: &mut ModalContext);
    /// Redraw based on the current state.
    fn redraw_request(&mut self, _: &mut Self::Surface) -> Result<(), RedrawError>;
    /// Issued after a lost surface was recovered.
    fn lost(&mut self, _: &mut Self::Surface);
    /// Issued after an outdated surface was recovered.
    fn outdated(&mut self, _: &mut Self::Surface);
    /// Issued to query if an exit is required.
    fn exit(&self) -> bool;
    /// Update the surface with changes to the compute instructions.
    fn reconfigure_compute(&mut self, _: &mut Self::Compute);
}

pub trait ScopedCompute {
    fn maybe_launch<'a>(&mut self, _: &'a Scope<'a, '_>);
}

pub trait WindowedSurface {
    fn from_window(window: WindowSurface) -> Self;
    fn recreate(&mut self);
}

pub fn build() -> Window {
    let event_loop = EventLoop::new().unwrap();
    Window::new(event_loop)
}

impl Window {
    pub(crate) fn new(event_loop: EventLoop<()>) -> Self {
        Window {
            event_loop,
            window: None,
        }
    }

    pub fn run_on_main<Ed>(
        self,
        mut ed: Ed,
        mut on_surface: impl FnMut(&mut Ed::Surface) -> Ed::Compute,
    ) -> !
    where
        Ed: ModalEditor + 'static,
        Ed::Surface: WindowedSurface,
        Ed::Compute: ScopedCompute,
    {
        let Window {
            event_loop,
            mut window,
        } = self;

        std::thread::scope(move |scope| {
            let mut surface = None;
            let mut modal = ModalContext::Main;

            event_loop
                .run(move |ev, app| {
                    Self::threaded_event_loop_callback::<Ed>(
                        ev,
                        app,
                        &mut ed,
                        &mut window,
                        &mut surface,
                        &mut modal,
                        &mut on_surface,
                        scope,
                    )
                })
                .unwrap()
        });

        std::process::exit(0);
    }

    // FIXME: turn these into a `App` struct and pass by `&mut self`?
    fn threaded_event_loop_callback<'scope, Ed>(
        ev: winit::event::Event<()>,
        app: &winit::event_loop::ActiveEventLoop,
        ed: &mut Ed,
        window: &mut Option<Arc<winit::window::Window>>,
        surface: &mut Option<(Ed::Surface, Ed::Compute)>,
        modal: &mut ModalContext,
        on_surface: &mut impl FnMut(&mut Ed::Surface) -> Ed::Compute,
        // FIXME: generalize this so we can also use an async loop or on the web, the browser's
        // executor here.
        scope: &'scope Scope<'scope, '_>,
    ) where
        Ed: ModalEditor + 'static,
        Ed::Surface: WindowedSurface,
        Ed::Compute: ScopedCompute,
    {
        log::info!("Window event: {:?}", ev);
        let window = window.get_or_insert_with(|| {
            Arc::new(app.create_window(WindowAttributes::default()).unwrap())
        });

        let empty = WindowSurface {
            surface: None,
            window: window.clone(),
        };

        let (surface, compute) = surface.get_or_insert_with(|| {
            let mut preinit = Ed::Surface::from_window(empty);
            let compute = on_surface(&mut preinit);
            (preinit, compute)
        });

        log::info!("Window event: {:?}", ev);
        match Self::input(&window, &modal, ev) {
            None => {}
            Some(ModalEvent::MainEventsCleared) => {
                ed.event(ModalEvent::MainEventsCleared, modal);
                window.request_redraw();
            }
            Some(ModalEvent::Resized) => {
                surface.recreate();
                ed.lost(surface);
                ed.reconfigure_compute(compute);
            }
            Some(ModalEvent::RedrawRequested) => {
                compute.maybe_launch(scope);
                ed.reconfigure_compute(compute);
                log::warn!("Redraw requested");

                ed.event(ModalEvent::RedrawRequested, modal);
                match ed.redraw_request(surface) {
                    Ok(()) => {}
                    Err(RedrawError::Lost) => {
                        surface.recreate();
                        ed.lost(surface)
                    }
                    Err(RedrawError::Outdated) => {
                        surface.recreate();
                        ed.outdated(surface);
                    }
                }
            }
            Some(ev) => {
                ed.event(ev, modal);
            }
        }

        if ed.exit() {
            log::info!("Editor requested close, closing window");
            app.exit();
        }
    }

    #[allow(unreachable_patterns)]
    fn input(
        window: &winit::window::Window,
        _: &ModalContext,
        ev: Event<()>,
    ) -> Option<ModalEvent> {
        Some(match ev {
            Event::WindowEvent { window_id, event } if window.id() == window_id => match event {
                WindowEvent::CloseRequested => ModalEvent::ExitPressed,
                WindowEvent::RedrawRequested if window.id() == window_id => {
                    ModalEvent::RedrawRequested
                }
                WindowEvent::Resized(_) => ModalEvent::Resized,
                _ => return None,
                WindowEvent::Moved(_) => todo!(),
                WindowEvent::Destroyed => todo!(),
                WindowEvent::DroppedFile(_) => todo!(),
                WindowEvent::HoveredFile(_) => todo!(),
                WindowEvent::HoveredFileCancelled => todo!(),
                WindowEvent::Focused(_) => todo!(),
                WindowEvent::KeyboardInput {
                    device_id: _,
                    event: _,
                    is_synthetic: _,
                } => todo!(),
                WindowEvent::ModifiersChanged(_) => todo!(),
                WindowEvent::CursorMoved {
                    device_id: _,
                    position: _,
                } => todo!(),
                WindowEvent::CursorEntered { device_id: _ } => todo!(),
                WindowEvent::CursorLeft { device_id: _ } => todo!(),
                WindowEvent::MouseWheel {
                    device_id: _,
                    delta: _,
                    phase: _,
                } => todo!(),
                WindowEvent::MouseInput {
                    device_id: _,
                    state: _,
                    button: _,
                } => todo!(),
                WindowEvent::TouchpadPressure {
                    device_id: _,
                    pressure: _,
                    stage: _,
                } => todo!(),
                WindowEvent::AxisMotion {
                    device_id: _,
                    axis: _,
                    value: _,
                } => todo!(),
                WindowEvent::Touch(_) => todo!(),
                WindowEvent::ScaleFactorChanged {
                    scale_factor: _,
                    inner_size_writer: _,
                } => todo!(),
                WindowEvent::ThemeChanged(_) => todo!(),
            },
            _ => return None,
        })
    }
}

impl WindowSurface {
    pub fn recreate(&mut self, instance: &wgpu::Instance) {
        self.surface = None;
        let surface = instance.create_surface(self.window.clone()).unwrap();
        self.surface = Some(surface);
    }

    pub fn window(&self) -> &winit::window::Window {
        &*self.window
    }

    pub fn surface(&self) -> &wgpu::Surface {
        &*self
    }

    pub fn inner_size(&self) -> (u32, u32) {
        let phys = self.window.inner_size();
        (phys.width, phys.height)
    }

    pub fn create_surface(&self, instance: &wgpu::Instance) -> WindowSurface {
        let surface = instance.create_surface(self.window.clone()).unwrap();

        WindowSurface {
            surface: Some(surface),
            window: self.window.clone(),
        }
    }
}

impl core::ops::Deref for WindowSurface {
    type Target = wgpu::Surface<'static>;
    fn deref(&self) -> &wgpu::Surface<'static> {
        // Only the private copy of `Window` doesn't have this attribute.
        self.surface.as_ref().unwrap()
    }
}
