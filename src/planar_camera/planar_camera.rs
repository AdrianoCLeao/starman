use crate::event::window_event::WindowEvent;
use crate::resource::effect::ShaderUniform;
use crate::window::canvas::Canvas;
use nalgebra::{Matrix3, Point2, Vector2};

pub trait PlanarCamera {
    fn handle_event(&mut self, canvas: &Canvas, event: &WindowEvent);

    fn update(&mut self, canvas: &Canvas);

    fn upload(
        &self,
        proj: &mut ShaderUniform<Matrix3<f32>>,
        view: &mut ShaderUniform<Matrix3<f32>>,
    );

    fn unproject(&self, window_coord: &Point2<f32>, window_size: &Vector2<f32>) -> Point2<f32>;
}
