use stealth_paint::command::{CommandBuffer, Rectangle};
use stealth_paint::pool::Pool;

const BACKGROUND: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/background.png");
const FOREGROUND: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/foreground.png");
const OUTPUT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/composed.png");

#[test]
fn run_blending() {
    const ANY: wgpu::BackendBit = wgpu::BackendBit::all();
    let instance = wgpu::Instance::new(ANY);
    let adapters = instance.enumerate_adapters(ANY);

    let mut pool = Pool::new();
    let mut commands = CommandBuffer::default();

    let background = image::open(BACKGROUND)
        .expect("Background image opened");
    let foreground = image::open(FOREGROUND)
        .expect("Background image opened");

    let (bg_key, background) = {
        let entry = pool.insert_srgb(&background);
        // TODO: more configuration.
        (entry.key(), entry.descriptor())
    };

    let (fg_key, foreground) = {
        let entry = pool.insert_srgb(&foreground);
        // TODO: more configuration.
        (entry.key(), entry.descriptor())
    };

    let placement = Rectangle {
        x: 0,
        y: 0,
        max_x: foreground.layout.width(),
        max_y: foreground.layout.height(),
    };

    // Describe the pipeline:
    // 0: in (background)
    // 1: in (foreground)
    // 2: inscribe(0, placement, 1)
    // 3: out(3)
    let background = commands.input(background).unwrap();
    let foreground = commands.input(foreground).unwrap();
    let result = commands.inscribe(background, placement, foreground)
        .expect("Valid to inscribe");
    let outformat = commands.output(result)
        .expect("Valid for output");

    let plan = commands.compile()
        .expect("Could build command buffer");
    let adapter = plan.choose_adapter(adapters)
        .expect("Did not find any adapter for executing the blend operation");

    let mut execution = plan.launch(&mut pool)
        .bind(background, bg_key).unwrap()
        .bind(foreground, fg_key).unwrap()
        .launch(&adapter)
        .expect("Launching failed");

    while execution.is_running() {
        let _wait_point = execution.step().expect("No failure");
    }

    let retire = execution
        .retire_gracefully(&mut pool);
    let image = retire.output(result)
        .expect("A valid image output")
        .to_image()
        .expect("An `image` image");

    image.save(OUTPUT)
        .expect("Successfully saved");
}
