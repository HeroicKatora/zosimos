use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub async fn run() {
    // TODO: setup panic hook here?
    // TODO: schedule async IO-tasks here?
    let winit = winit::build();
    let surface = surface::Surface::new(&winit);
    let editor = editor::Editor::default();
    let compute = compute::Compute::new(&surface);

    winit.run_on_main(editor, compute, surface)
}
