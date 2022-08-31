/// The winit related state of our application.
///
/// We have two responsibilities:
/// - rewrite input events into program semantics, based on the its current input state. The callee
///     will cooperatively indicate when it changes.
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use std::sync::Arc;

pub struct Window {
    event_loop: EventLoop<()>,
    inner: WindowSurface,
}

/// Combination of window/surface that fulfills safety property of `create_surface`.
pub struct WindowSurface {
    surface: Option<wgpu::Surface>,
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
}

pub trait WindowedSurface {
    fn recreate(&mut self);
}

pub fn build() -> Window {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();
    Window::new(window, event_loop)
}

impl Window {
    pub(crate) fn new(window: winit::window::Window, event_loop: EventLoop<()>) -> Self {
        let inner = WindowSurface {
            window: Arc::new(window),
            surface: None,
        };

        Window { event_loop, inner }
    } 

    pub fn create_surface(&self, instance: &wgpu::Instance) -> WindowSurface {
        let window = self.inner.window.clone();
        let surface = unsafe { instance.create_surface(&*window) };
        WindowSurface {
            surface: Some(surface),
            window,
        }
    }

    pub fn inner_size(&self) -> (u32, u32) {
        let phys = self.inner.window.inner_size();
        (phys.width, phys.height)
    }

    pub fn run_on_main<Ed>(
        self,
        mut ed: Ed,
        mut compute: Ed::Compute,
        mut surface: Ed::Surface,
    ) -> !
    where
        Ed: ModalEditor + 'static,
        Ed::Surface: WindowedSurface,
    {
        let Window { event_loop, inner } = self;
        let mut modal = ModalContext::Main;
        event_loop.run(move |ev, _, ctrl| {
            let recreate = |surface: &mut Ed::Surface| {
                surface.recreate();
            };

            log::info!("Window event: {:?}", ev);
            match Self::input(&inner.window, &modal, ev) {
                None => {}
                Some(ModalEvent::MainEventsCleared) => {
                    ed.event(ModalEvent::MainEventsCleared, &mut modal);
                    inner.window.request_redraw();
                }
                Some(ModalEvent::RedrawRequested) => {
                    ed.event(ModalEvent::RedrawRequested, &mut modal);
                    match ed.redraw_request(&mut surface) {
                        Ok(()) => {}
                        Err(RedrawError::Lost) => {
                            recreate(&mut surface);
                            ed.lost(&mut surface)
                        }
                        Err(RedrawError::Outdated) => {
                            recreate(&mut surface);
                            ed.outdated(&mut surface);
                        }
                    }
                }
                Some(ev) => {
                    ed.event(ev, &mut modal);
                }
            }

            if ed.exit() {
                log::info!("Editor requested close, closing window");
                *ctrl = ControlFlow::Exit;
            }
        })
    }

    #[allow(unreachable_patterns)]
    fn input(
        window: &winit::window::Window,
        _: &ModalContext,
        ev: Event<()>,
    ) -> Option<ModalEvent> {
        Some(match ev {
            Event::MainEventsCleared => ModalEvent::MainEventsCleared,
            Event::RedrawRequested(wid) if window.id() == wid => ModalEvent::RedrawRequested,
            Event::WindowEvent { window_id, event } if window.id() == window_id => {
                match event {
                    WindowEvent::CloseRequested => ModalEvent::ExitPressed,
                    _ => return None,
                    WindowEvent::Resized(_) => todo!(),
                    WindowEvent::Moved(_) => todo!(),
                    WindowEvent::Destroyed => todo!(),
                    WindowEvent::DroppedFile(_) => todo!(),
                    WindowEvent::HoveredFile(_) => todo!(),
                    WindowEvent::HoveredFileCancelled => todo!(),
                    WindowEvent::ReceivedCharacter(_) => todo!(),
                    WindowEvent::Focused(_) => todo!(),
                    WindowEvent::KeyboardInput { device_id, input, is_synthetic } => todo!(),
                    WindowEvent::ModifiersChanged(_) => todo!(),
                    WindowEvent::CursorMoved { device_id, position, modifiers } => todo!(),
                    WindowEvent::CursorEntered { device_id } => todo!(),
                    WindowEvent::CursorLeft { device_id } => todo!(),
                    WindowEvent::MouseWheel { device_id, delta, phase, modifiers } => todo!(),
                    WindowEvent::MouseInput { device_id, state, button, modifiers } => todo!(),
                    WindowEvent::TouchpadPressure { device_id, pressure, stage } => todo!(),
                    WindowEvent::AxisMotion { device_id, axis, value } => todo!(),
                    WindowEvent::Touch(_) => todo!(),
                    WindowEvent::ScaleFactorChanged { scale_factor, new_inner_size } => todo!(),
                    WindowEvent::ThemeChanged(_) => todo!(),
                }
            }
            _ => return None,
        })
    }
}

impl WindowSurface {
    pub fn recreate(&mut self, instance: &wgpu::Instance) {
        self.surface = None;
        let surface = unsafe { instance.create_surface(&*self.window) };
        self.surface = Some(surface);
    }

    pub fn window(&self) -> &winit::window::Window {
        &*self.window
    }

    pub fn surface(&self) -> &wgpu::Surface {
        &*self
    }
}

impl core::ops::Deref for WindowSurface {
    type Target = wgpu::Surface;
    fn deref(&self) -> &wgpu::Surface {
        // Only the private copy of `Window` doesn't have this attribute.
        self.surface.as_ref().unwrap()
    }
}
