use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

mod window;
use window::window::Window;

mod loader;

fn main() {
    let event_loop = EventLoop::new();

    let custom_window = Window::open("My new window", false, 800, 600);

    let _winit_window = winit::window::WindowBuilder::new()
        .with_title(&custom_window.title)
        .with_inner_size(winit::dpi::LogicalSize::new(
            custom_window.width as f64,
            custom_window.height as f64,
        ))
        .build(&event_loop)
        .unwrap();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            _ => {}
        }
    });
}
