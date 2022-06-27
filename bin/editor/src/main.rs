mod compute;
mod editor;
mod surface;
#[cfg(target_arch = "wasm32")]
mod wasm32;
mod winit;

fn main() {
    env_logger::init();

    let winit = winit::build();
    let surface = surface::Surface::new(&winit);
    let editor = editor::Editor::default();
    let compute = compute::Compute::new(&surface);

    winit.run_on_main(editor, compute, surface)
}
