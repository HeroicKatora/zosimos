use wasm_bindgen::prelude::*;
use web_sys;
use crate::{compute, editor, surface, winit::Window};

#[wasm_bindgen(start)]
pub async fn run() {
    // TODO: setup panic hook here?
    // TODO: schedule async IO-tasks here?
    let winit = Window::new_wasm();

    let mut surface = surface::Surface::new(&winit);
    let editor = editor::Editor::default();
    let compute = compute::Compute::new(&surface);

    surface.set_image(&{
        let img = include_bytes!(
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/input/background.png")
        );

        let img = std::io::Cursor::new(img);
        image::io::Reader::new(img).with_guessed_format().unwrap().decode().unwrap()
    });

    winit.run_on_main(editor, compute, surface)
}

impl Window {
    fn new_wasm() -> Self {
        use winit::platform::web::WindowExtWebSys;

        let event_loop = winit::event_loop::EventLoop::new();
        let mut builder = winit::window::WindowBuilder::new();
        let window = builder.build(&event_loop).unwrap();

        let level: log::Level = log::Level::Info;
        console_log::init_with_level(level).expect("could not initialize logger");
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        // On wasm, append the canvas to the document body
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| doc.body())
            .and_then(|body| {
                body.append_child(&web_sys::Element::from(window.canvas()))
                    .ok()
            })
            .expect("couldn't append canvas to document body");
        log::info!("Hello, from stealth-paint");

        Window::new(window, event_loop)
    }
}
