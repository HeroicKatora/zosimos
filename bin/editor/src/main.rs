mod compute;
mod editor;
mod surface;
mod winit;

fn main() {
    env_logger::init();

    let winit = winit::build();
    let mut surface = surface::Surface::new(&winit);
    let editor = editor::Editor::default();
    let compute = compute::Compute::new(&mut surface);

    surface.set_image(&{
        let img = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/input/background.png"
        );
        eprintln!("{}", img);
        image::io::Reader::open(img).unwrap().decode().unwrap()
    });

    winit.run_on_main(editor, compute, surface)
}
