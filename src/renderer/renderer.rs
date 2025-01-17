use crate::camera::camera::Camera;

pub trait Renderer {
    fn render(&mut self, pass: usize, camera: &mut dyn Camera);
}
