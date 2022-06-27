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

pub struct Window {
    event_loop: EventLoop<()>,
    window: winit::window::Window,
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

pub trait ModalEditor {
    type Compute;
    type Surface;
    fn event(&mut self, _: ModalEvent, _: &mut ModalContext);
    fn redraw_request(&mut self, _: &mut Self::Surface);
    fn exit(&self) -> bool;
}

pub fn build() -> Window {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();
    Window { event_loop, window }
}

impl Window {
    pub fn create_surface(&self, instance: &wgpu::Instance) -> wgpu::Surface {
        unsafe { instance.create_surface(&self.window) }
    }

    pub fn inner_size(&self) -> (u32, u32) {
        let phys = self.window.inner_size();
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
    {
        let Window { event_loop, window } = self;
        let mut modal = ModalContext::Main;
        event_loop.run(move |ev, _, ctrl| {
            log::info!("Window event: {:?}", ev);
            match Self::input(&window, &modal, ev) {
                None => {}
                Some(ModalEvent::MainEventsCleared) => {
                    ed.event(ModalEvent::MainEventsCleared, &mut modal);
                    window.request_redraw();
                }
                Some(ModalEvent::RedrawRequested) => {
                    ed.event(ModalEvent::RedrawRequested, &mut modal);
                    ed.redraw_request(&mut surface);
                }
                Some(ev) => {
                    ed.event(ev, &mut modal);
                }
            }

            std::thread::sleep(std::time::Duration::from_millis(10));

            if ed.exit() {
                *ctrl = ControlFlow::Exit;
            }
        })
    }

    fn input(
        window: &winit::window::Window,
        _: &ModalContext,
        ev: Event<()>,
    ) -> Option<ModalEvent> {
        Some(match ev {
            Event::MainEventsCleared => ModalEvent::MainEventsCleared,
            Event::RedrawRequested(wid) if window.id() == wid => ModalEvent::RedrawRequested,
            _ => return None,
        })
    }
}
