use crate::event::window_event::WindowEvent;
use crate::resource::effect::ShaderUniform;
use crate::window::canvas::Canvas;
use nalgebra::{Isometry3, Matrix4, Point2, Point3, Point4, Vector2, Vector3};

pub trait Camera {
    fn handle_event(&mut self, canvas: &Canvas, event: &WindowEvent);
    fn eye(&self) -> Point3<f32>; 
    fn view_transform(&self) -> Isometry3<f32>;
    fn transformation(&self) -> Matrix4<f32>;
    fn inverse_transformation(&self) -> Matrix4<f32>;
    fn clip_planes(&self) -> (f32, f32);

    fn update(&mut self, canvas: &Canvas);

    fn upload(
        &self,
        pass: usize,
        proj: &mut ShaderUniform<Matrix4<f32>>,
        view: &mut ShaderUniform<Matrix4<f32>>,
    );

    #[inline]
    fn num_passes(&self) -> usize {
        1usize
    }

    #[inline]
    fn start_pass(&self, _pass: usize, _canvas: &Canvas) {}

    #[inline]
    fn render_complete(&self, _canvas: &Canvas) {}

    fn project(&self, world_coord: &Point3<f32>, size: &Vector2<f32>) -> Vector2<f32> {
        let h_world_coord = world_coord.to_homogeneous();
        let h_normalized_coord = self.transformation() * h_world_coord;

        let normalized_coord = Point3::from_homogeneous(h_normalized_coord).unwrap();

        Vector2::new(
            (1.0 + normalized_coord.x) * size.x / 2.0,
            (1.0 + normalized_coord.y) * size.y / 2.0,
        )
    }

    fn unproject(
        &self,
        window_coord: &Point2<f32>,
        size: &Vector2<f32>,
    ) -> (Point3<f32>, Vector3<f32>) {
        let normalized_coord = Point2::new(
            2.0 * window_coord.x / size.x - 1.0,
            2.0 * -window_coord.y / size.y + 1.0,
        );

        let normalized_begin = Point4::new(normalized_coord.x, normalized_coord.y, -1.0, 1.0);
        let normalized_end = Point4::new(normalized_coord.x, normalized_coord.y, 1.0, 1.0);

        let cam = self.inverse_transformation();

        let h_unprojected_begin = cam * normalized_begin;
        let h_unprojected_end = cam * normalized_end;

        let unprojected_begin = Point3::from_homogeneous(h_unprojected_begin.coords).unwrap();
        let unprojected_end = Point3::from_homogeneous(h_unprojected_end.coords).unwrap();

        (
            unprojected_begin,
            (unprojected_end - unprojected_begin).normalize(),
        )
    }
}
