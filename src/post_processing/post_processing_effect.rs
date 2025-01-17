use crate::resource::framebuffer_manager::RenderTarget;

pub trait PostProcessingEffect {
    fn update(&mut self, dt: f32, w: f32, h: f32, znear: f32, zfar: f32);

    fn draw(&mut self, target: &RenderTarget);
}
