use winit::{dpi::LogicalSize, window::WindowBuilder};

pub struct Window{
    pub title: String,
    pub width: u32,
    pub height: u32,
}

impl Window {
    pub fn open(
        title: &str,
        hide: bool,
        width: u32,
        height: u32,
    ) -> Self {
        let _ = WindowBuilder::new()
            .with_title(title)
            .with_inner_size(LogicalSize::new(width as f64, height as f64))
            .with_visible(!hide);

        Window {
            title: title.to_string(),
            width,
            height,
        }
    }
}