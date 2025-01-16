use std::sync::mpsc::Sender;

use crate::event::window_event::{Action, Key, MouseButton, WindowEvent};
#[cfg(not(target_arch = "wasm32"))]
use crate::window::gl_canvas::GLCanvas as CanvasImpl;
#[cfg(target_arch = "wasm32")]
use crate::window::WebGLCanvas as CanvasImpl;
use image::{GenericImage, Pixel};

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NumSamples {
    Zero = 0,
    One = 1,
    Two = 2,
    Four = 4,
    Eight = 8,
    Sixteen = 16,
}

impl NumSamples {
    pub fn from_u32(i: u32) -> Option<NumSamples> {
        match i {
            0 => Some(NumSamples::Zero),
            1 => Some(NumSamples::One),
            2 => Some(NumSamples::Two),
            4 => Some(NumSamples::Four),
            8 => Some(NumSamples::Eight),
            16 => Some(NumSamples::Sixteen),
            _ => None,
        }
    }
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanvasSetup {
    pub vsync: bool,
    pub samples: NumSamples,
}

pub struct Canvas {
    canvas: CanvasImpl,
}

impl Canvas {
    pub fn open(
        title: &str,
        hide: bool,
        width: u32,
        height: u32,
        canvas_setup: Option<CanvasSetup>,
        out_events: Sender<WindowEvent>,
    ) -> Self {
        Canvas {
            canvas: CanvasImpl::open(title, hide, width, height, canvas_setup, out_events),
        }
    }

    pub fn render_loop(data: impl FnMut(f64) -> bool + 'static) {
        CanvasImpl::render_loop(data)
    }

    pub fn poll_events(&mut self) {
        self.canvas.poll_events()
    }

    pub fn swap_buffers(&mut self) {
        self.canvas.swap_buffers()
    }

    pub fn size(&self) -> (u32, u32) {
        self.canvas.size()
    }

    pub fn cursor_pos(&self) -> Option<(f64, f64)> {
        self.canvas.cursor_pos()
    }

    pub fn scale_factor(&self) -> f64 {
        self.canvas.scale_factor()
    }

    pub fn set_title(&mut self, title: &str) {
        self.canvas.set_title(title)
    }

    pub fn set_icon(&mut self, icon: impl GenericImage<Pixel = impl Pixel<Subpixel = u8>>) {
        self.canvas.set_icon(icon)
    }

    pub fn set_cursor_grab(&self, grab: bool) {
        self.canvas.set_cursor_grab(grab);
    }

    pub fn set_cursor_position(&self, x: f64, y: f64) {
        self.canvas.set_cursor_position(x, y);
    }

    pub fn hide_cursor(&self, hide: bool) {
        self.canvas.hide_cursor(hide);
    }

    pub fn hide(&mut self) {
        self.canvas.hide()
    }

    pub fn show(&mut self) {
        self.canvas.show()
    }

    pub fn get_mouse_button(&self, button: MouseButton) -> Action {
        self.canvas.get_mouse_button(button)
    }

    pub fn get_key(&self, key: Key) -> Action {
        self.canvas.get_key(key)
    }
}

pub(crate) trait AbstractCanvas {
    fn open(
        title: &str,
        hide: bool,
        width: u32,
        height: u32,
        window_setup: Option<CanvasSetup>,
        out_events: Sender<WindowEvent>,
    ) -> Self;
    fn render_loop(data: impl FnMut(f64) -> bool + 'static);
    fn poll_events(&mut self);
    fn swap_buffers(&mut self);
    fn size(&self) -> (u32, u32);
    fn cursor_pos(&self) -> Option<(f64, f64)>;
    fn scale_factor(&self) -> f64;

    fn set_title(&mut self, title: &str);
    fn set_icon(&mut self, icon: impl GenericImage<Pixel = impl Pixel<Subpixel = u8>>);
    fn set_cursor_grab(&self, grab: bool);
    fn set_cursor_position(&self, x: f64, y: f64);
    fn hide_cursor(&self, hide: bool);
    fn hide(&mut self);
    fn show(&mut self);

    fn get_mouse_button(&self, button: MouseButton) -> Action;
    fn get_key(&self, key: Key) -> Action;
}
