mod compute;
mod editor;
mod surface;
#[cfg(target_arch = "wasm32")]
mod wasm32;
mod winit;

#[cfg(not(target_os = "wasi"))]
fn main() {
    env_logger::init();

    let winit = winit::build();
    let editor = editor::Editor::default();

    winit.run_on_main(editor, |surface: &mut surface::Surface| {
        surface.set_image(&{
            let img = concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../tests/input/background.png"
            );
            eprintln!("{}", img);
            image::ImageReader::open(img).unwrap().decode().unwrap()
        });

        compute::Compute::new(surface)
    })
}

#[cfg(target_os = "wasi")]
fn main() {
    env_logger::init();
    wasm32::run_sync();
}
