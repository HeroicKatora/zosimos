use crate::{compute, editor, surface, winit::Window};
use wasm_bindgen::prelude::*;
use web_sys;

#[cfg(all(not(target_os = "wasi"), target_arch = "wasm32",))]
pub async fn run() {
    let level: log::Level = log::Level::Info;
    console_log::init_with_level(level).expect("could not initialize logger");
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));

    run_sync();
}

pub fn run_sync() {
    // TODO: setup panic hook here?
    // TODO: schedule async IO-tasks here?
    let winit = Window::new_wasm();

    let mut surface = surface::Surface::new(&winit);
    let editor = editor::Editor::default();
    let compute = compute::Compute::new(&mut surface);

    surface.set_image(&{
        let img = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/input/background.png"
        ));

        let img = std::io::Cursor::new(img);
        image::io::Reader::new(img)
            .with_guessed_format()
            .unwrap()
            .decode()
            .unwrap()
    });

    winit.run_on_main(editor, compute, surface)
}

impl Window {
    fn new_wasm() -> Self {
        use winit::platform::web::{WindowBuilderExtWebSys, WindowExtWebSys};

        let canvas: web_sys::HtmlCanvasElement = web_sys::window()
            .and_then(|w| w.document())
            .expect("find a document")
            .query_selector("canvas")
            .expect("find something")
            .expect("find a canvas")
            .dyn_into()
            .expect("find a real canvas");
        eprintln!("{}×{}", canvas.width(), canvas.height());

        let event_loop = winit::event_loop::EventLoop::new();
        let mut builder = winit::window::WindowBuilder::new();
        builder = builder.with_canvas(Some(canvas));
        let window = builder.build(&event_loop).unwrap();

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
