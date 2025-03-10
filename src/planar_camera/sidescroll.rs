use crate::event::window_event::{Action, MouseButton, WindowEvent};
use crate::planar_camera::PlanarCamera;
use crate::resource::effect::ShaderUniform;
use crate::window::canvas::Canvas;
use nalgebra::{self as na, Matrix3, Point2, Translation2, Vector2};
use num_traits::Pow;
use std::f32;

#[derive(Clone, Debug)]
pub struct Sidescroll {
    at: Point2<f32>,
    zoom: f32,

    zoom_step: f32,
    drag_button: Option<MouseButton>,

    view: Matrix3<f32>,
    proj: Matrix3<f32>,
    scaled_proj: Matrix3<f32>,
    inv_scaled_proj: Matrix3<f32>,
    last_cursor_pos: Vector2<f32>,
}

impl Sidescroll {
    pub fn new() -> Sidescroll {
        let mut res = Sidescroll {
            at: Point2::origin(),
            zoom: 1.0,
            zoom_step: 0.9,
            drag_button: Some(MouseButton::Button2),
            view: na::one(),
            proj: na::one(),
            scaled_proj: na::one(),
            inv_scaled_proj: na::one(),
            last_cursor_pos: na::zero(),
        };

        res.update_projviews();

        res
    }

    pub fn at(&self) -> Point2<f32> {
        self.at
    }

    pub fn set_at(&mut self, at: Point2<f32>) {
        self.at = at;
        self.update_projviews();
    }

    pub fn zoom(&self) -> f32 {
        self.zoom
    }

    pub fn set_zoom(&mut self, zoom: f32) {
        self.zoom = zoom;

        self.update_restrictions();
        self.update_projviews();
    }

    pub fn look_at(&mut self, at: Point2<f32>, zoom: f32) {
        self.at = at;
        self.zoom = zoom;
        self.update_projviews();
    }

    fn update_restrictions(&mut self) {
        if self.zoom < 0.00001 {
            self.zoom = 0.00001
        }
    }

    pub fn drag_button(&self) -> Option<MouseButton> {
        self.drag_button
    }

    pub fn rebind_drag_button(&mut self, new_button: Option<MouseButton>) {
        self.drag_button = new_button;
    }

    fn handle_right_button_displacement(&mut self, dpos: &Vector2<f32>) {
        self.at.x -= dpos.x / self.zoom;
        self.at.y += dpos.y / self.zoom;
        self.update_projviews();
    }

    fn handle_scroll(&mut self, off: f32) {
        self.zoom /= self.zoom_step.pow(off / 120.0);
        self.update_restrictions();
        self.update_projviews();
    }

    fn update_projviews(&mut self) {
        self.view = Translation2::new(-self.at.x, -self.at.y).to_homogeneous();
        self.scaled_proj = self.proj;
        self.scaled_proj.m11 *= self.zoom;
        self.scaled_proj.m22 *= self.zoom;

        self.inv_scaled_proj.m11 = 1.0 / self.scaled_proj.m11;
        self.inv_scaled_proj.m22 = 1.0 / self.scaled_proj.m22;
    }
}

impl PlanarCamera for Sidescroll {
    fn handle_event(&mut self, canvas: &Canvas, event: &WindowEvent) {
        let scale = 1.0; 

        match *event {
            WindowEvent::CursorPos(x, y, _) => {
                let curr_pos = Vector2::new(x as f32, y as f32);

                if let Some(drag_button) = self.drag_button {
                    if canvas.get_mouse_button(drag_button) == Action::Press {
                        let dpos = curr_pos - self.last_cursor_pos;
                        self.handle_right_button_displacement(&dpos)
                    }
                }

                self.last_cursor_pos = curr_pos;
            }
            WindowEvent::Scroll(_, off, _) => self.handle_scroll(off as f32),
            WindowEvent::FramebufferSize(w, h) => {
                self.proj = Matrix3::new(
                    2.0 * (scale as f32) / (w as f32),
                    0.0,
                    0.0,
                    0.0,
                    2.0 * (scale as f32) / (h as f32),
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                );
                self.update_projviews();
            }
            _ => {}
        }
    }

    #[inline]
    fn upload(
        &self,
        proj: &mut ShaderUniform<Matrix3<f32>>,
        view: &mut ShaderUniform<Matrix3<f32>>,
    ) {
        proj.upload(&self.scaled_proj);
        view.upload(&self.view);
    }

    fn update(&mut self, _: &Canvas) {}

    fn unproject(&self, window_coord: &Point2<f32>, size: &Vector2<f32>) -> Point2<f32> {
        let normalized_coords = Point2::new(
            2.0 * window_coord.x / size.x - 1.0,
            2.0 * -window_coord.y / size.y + 1.0,
        );

        let unprojected_hom = self.inv_scaled_proj * normalized_coords.to_homogeneous();

        Point2::from_homogeneous(unprojected_hom).unwrap() + self.at.coords
    }
}
