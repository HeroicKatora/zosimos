use stealth_paint::command::CommandBuffer;

#[test]
fn run_blending() {
    let mut commands = CommandBuffer::default();
    commands.input();

    let mut instance = wgpu::Instance::new(wgpu::BackendBit::all());
}
