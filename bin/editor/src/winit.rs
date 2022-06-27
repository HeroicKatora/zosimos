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
    event_loop: EventLoop::<()>,
    window: winit::window::Window,
}

/// The input modality.
pub enum ModalContext {
    Main,
}

pub enum ModalEvent {
    ExitPressed,
}

pub trait ModalEditor {
    fn event(&mut self, _: ModalEvent, _: &mut ModalContext);
    fn exit(&self) -> bool;
}

pub fn build() -> Window {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();
    Window { event_loop, window }
}

impl Window {
    pub fn run_on_main(self, mut ed: impl ModalEditor + 'static) -> ! {
        let Window { event_loop, window } = self;
        let mut modal = ModalContext::Main;
        event_loop.run(move |ev, _, ctrl| {
            if let Some(ev) = Self::input(&modal, ev) {
                ed.event(ev, &mut modal);
            }

            if ed.exit() {
                *ctrl = ControlFlow::Exit;
            }
        })
    }

    fn input(_: &ModalContext, ev: Event<()>) -> Option<ModalEvent> {
        match ev {
            _ => return None,
        }
    }
}
