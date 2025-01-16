use crate::camera::camera::Camera;
use crate::event::window_event::{Action, Key, Modifiers, MouseButton, WindowEvent};
use crate::resource::effect::ShaderUniform;
use crate::window::canvas::Canvas;
use nalgebra::{self, Isometry3, Matrix4, Perspective3, Point3, Unit, UnitQuaternion, Vector2, Vector3};
use std::f32;

#[derive(Clone, Debug)]
pub struct ArcBall {
    at: Point3<f32>,    
    yaw: f32,
    pitch: f32,
    dist: f32,
    min_dist: f32,
    max_dist: f32,
    yaw_step: f32,
    pitch_step: f32,
    min_pitch: f32,
    max_pitch: f32,
    dist_step: f32,
    rotate_button: Option<MouseButton>,
    rotate_modifiers: Option<Modifiers>,
    drag_button: Option<MouseButton>,
    drag_modifiers: Option<Modifiers>,
    reset_key: Option<Key>,

    projection: Perspective3<f32>,
    view: Matrix4<f32>,
    proj: Matrix4<f32>,
    proj_view: Matrix4<f32>,
    inverse_proj_view: Matrix4<f32>,
    last_cursor_pos: Vector2<f32>,
    last_framebuffer_size: Vector2<f32>,
    coord_system: CoordSystemRh,
}

impl ArcBall {
    pub fn new(eye: Point3<f32>, at: Point3<f32>) -> ArcBall {
        ArcBall::new_with_frustrum(f32::consts::PI / 4.0, 0.001, 1024.0, eye, at)
    }

    pub fn new_with_frustrum(
        fov: f32,
        znear: f32,
        zfar: f32,
        eye: Point3<f32>,
        at: Point3<f32>,
    ) -> ArcBall {
        let mut res = ArcBall {
            at: Point3::new(0.0, 0.0, 0.0),
            yaw: 0.0,
            pitch: 0.0,
            dist: 0.0,
            min_dist: 0.00001,
            max_dist: 1.0e4,
            yaw_step: 0.005,
            pitch_step: 0.005,
            min_pitch: 0.01,
            max_pitch: std::f32::consts::PI - 0.01,
            dist_step: 1.01,
            rotate_button: Some(MouseButton::Button1),
            rotate_modifiers: None,
            drag_button: Some(MouseButton::Button2),
            drag_modifiers: None,
            reset_key: Some(Key::Return),
            projection: Perspective3::new(800.0 / 600.0, fov, znear, zfar),
            view: nalgebra::zero(),
            proj: nalgebra::zero(),
            proj_view: nalgebra::zero(),
            inverse_proj_view: nalgebra::zero(),
            last_framebuffer_size: Vector2::new(800.0, 600.0),
            last_cursor_pos: nalgebra::zero(),
            coord_system: CoordSystemRh::from_up_axis(Vector3::y_axis()),
        };

        res.look_at(eye, at);

        res
    }

    pub fn at(&self) -> Point3<f32> {
        self.at
    }

    pub fn set_at(&mut self, at: Point3<f32>) {
        self.at = at;
        self.update_projviews();
    }

    pub fn yaw(&self) -> f32 {
        self.yaw
    }

    pub fn set_yaw(&mut self, yaw: f32) {
        self.yaw = yaw;

        self.update_restrictions();
        self.update_projviews();
    }

    pub fn pitch(&self) -> f32 {
        self.pitch
    }

    pub fn set_pitch(&mut self, pitch: f32) {
        self.pitch = pitch;

        self.update_restrictions();
        self.update_projviews();
    }

    pub fn min_pitch(&self) -> f32 {
        self.min_pitch
    }

    pub fn set_min_pitch(&mut self, min_pitch: f32) {
        self.min_pitch = min_pitch;
    }

    pub fn max_pitch(&self) -> f32 {
        self.max_pitch
    }

    pub fn set_max_pitch(&mut self, max_pitch: f32) {
        self.max_pitch = max_pitch;
    }

    pub fn dist(&self) -> f32 {
        self.dist
    }

    pub fn set_dist(&mut self, dist: f32) {
        self.dist = dist;

        self.update_restrictions();
        self.update_projviews();
    }

    pub fn min_dist(&self) -> f32 {
        self.min_dist
    }

    pub fn set_min_dist(&mut self, min_dist: f32) {
        self.min_dist = min_dist;
    }

    pub fn max_dist(&self) -> f32 {
        self.max_dist
    }

    pub fn set_max_dist(&mut self, max_dist: f32) {
        self.max_dist = max_dist;
    }

    pub fn set_dist_step(&mut self, dist_step: f32) {
        self.dist_step = dist_step;
    }

    pub fn look_at(&mut self, eye: Point3<f32>, at: Point3<f32>) {
        let dist = (eye - at).norm();

        let view_eye = self.coord_system.rotation_to_y_up * eye;
        let view_at = self.coord_system.rotation_to_y_up * at;
        let pitch = ((view_eye.y - view_at.y) / dist).acos();
        let yaw = (view_eye.z - view_at.z).atan2(view_eye.x - view_at.x);

        self.at = at;
        self.dist = dist;
        self.yaw = yaw;
        self.pitch = pitch;

        self.update_restrictions();
        self.update_projviews();
    }

    fn update_restrictions(&mut self) {
        if self.dist < self.min_dist {
            self.dist = self.min_dist
        }

        if self.dist > self.max_dist {
            self.dist = self.max_dist
        }

        if self.pitch <= self.min_pitch {
            self.pitch = self.min_pitch
        }

        if self.pitch > self.max_pitch {
            self.pitch = self.max_pitch
        }
    }

    pub fn rotate_button(&self) -> Option<MouseButton> {
        self.rotate_button
    }

    pub fn rebind_rotate_button(&mut self, new_button: Option<MouseButton>) {
        self.rotate_button = new_button;
    }

    pub fn rotate_modifiers(&self) -> Option<Modifiers> {
        self.rotate_modifiers
    }

    pub fn set_rotate_modifiers(&mut self, modifiers: Option<Modifiers>) {
        self.rotate_modifiers = modifiers
    }

    pub fn drag_modifiers(&self) -> Option<Modifiers> {
        self.drag_modifiers
    }

    pub fn set_drag_modifiers(&mut self, modifiers: Option<Modifiers>) {
        self.drag_modifiers = modifiers
    }

    pub fn drag_button(&self) -> Option<MouseButton> {
        self.drag_button
    }

    pub fn rebind_drag_button(&mut self, new_button: Option<MouseButton>) {
        self.drag_button = new_button;
    }

    pub fn reset_key(&self) -> Option<Key> {
        self.reset_key
    }

    pub fn rebind_reset_key(&mut self, new_key: Option<Key>) {
        self.reset_key = new_key;
    }

    fn handle_left_button_displacement(&mut self, dpos: &Vector2<f32>) {
        self.yaw += dpos.x * self.yaw_step;
        self.pitch -= dpos.y * self.pitch_step;

        self.update_restrictions();
        self.update_projviews();
    }

    fn handle_right_button_displacement(&mut self, dpos_norm: &Vector2<f32>) {
        let eye = self.eye();
        let dir = (self.at - eye).normalize();
        let tangent = self.coord_system.up_axis.cross(&dir).normalize();
        let bitangent = dir.cross(&tangent);
        self.at =
            self.at + tangent * (dpos_norm.x * self.dist) + bitangent * (dpos_norm.y * self.dist);
        self.update_projviews();
    }

    fn handle_scroll(&mut self, off: f32) {
        let mut dpos = Vector2::new(
            0.5 - self.last_cursor_pos.x / self.last_framebuffer_size.x,
            0.5 - self.last_cursor_pos.y / self.last_framebuffer_size.y,
        );
        self.handle_right_button_displacement(&dpos);

        self.dist *= self.dist_step.powf(off);
        self.update_restrictions();
        self.update_projviews();

        dpos = -dpos;
        self.handle_right_button_displacement(&dpos);
    }

    fn update_projviews(&mut self) {
        self.proj = *self.projection.as_matrix();
        self.view = self.view_transform().to_homogeneous();
        self.proj_view = self.proj * self.view;
        self.inverse_proj_view = self.proj_view.try_inverse().unwrap();
    }

    #[inline]
    pub fn set_up_axis(&mut self, up_axis: Vector3<f32>) {
        self.set_up_axis_dir(Unit::new_normalize(up_axis));
    }

    #[inline]
    pub fn set_up_axis_dir(&mut self, up_axis: Unit<Vector3<f32>>) {
        if self.coord_system.up_axis != up_axis {
            let new_coord_system = CoordSystemRh::from_up_axis(up_axis);

            let old_eye = self.eye();
            self.coord_system = new_coord_system;
            self.look_at(old_eye, self.at);
        }
    }
}

impl Camera for ArcBall {
    fn clip_planes(&self) -> (f32, f32) {
        (self.projection.znear(), self.projection.zfar())
    }

    fn view_transform(&self) -> Isometry3<f32> {
        Isometry3::look_at_rh(&self.eye(), &self.at, &self.coord_system.up_axis)
    }

    fn eye(&self) -> Point3<f32> {
        let view_at = self.coord_system.rotation_to_y_up * self.at;
        let px = view_at.x + self.dist * self.yaw.cos() * self.pitch.sin();
        let py = view_at.y + self.dist * self.pitch.cos();
        let pz = view_at.z + self.dist * self.yaw.sin() * self.pitch.sin();
        self.coord_system.rotation_to_y_up.inverse() * Point3::new(px, py, pz)
    }

    fn handle_event(&mut self, canvas: &Canvas, event: &WindowEvent) {
        match *event {
            WindowEvent::CursorPos(x, y, modifiers) => {
                let curr_pos = Vector2::new(x as f32, y as f32);

                if let Some(rotate_button) = self.rotate_button {
                    if canvas.get_mouse_button(rotate_button) == Action::Press
                        && self
                            .rotate_modifiers
                            .map(|m| m == modifiers)
                            .unwrap_or(true)
                    {
                        let dpos = curr_pos - self.last_cursor_pos;
                        self.handle_left_button_displacement(&dpos)
                    }
                }

                if let Some(drag_button) = self.drag_button {
                    if canvas.get_mouse_button(drag_button) == Action::Press
                        && self.drag_modifiers.map(|m| m == modifiers).unwrap_or(true)
                    {
                        let dpos = curr_pos - self.last_cursor_pos;
                        let dpos_norm = dpos.component_div(&self.last_framebuffer_size);
                        self.handle_right_button_displacement(&dpos_norm)
                    }
                }

                self.last_cursor_pos = curr_pos;
            }
            WindowEvent::Key(key, Action::Press, _) if Some(key) == self.reset_key => {
                self.at = Point3::origin();
                self.update_projviews();
            }
            WindowEvent::Scroll(_, off, _) => self.handle_scroll(off as f32),
            WindowEvent::FramebufferSize(w, h) => {
                self.last_framebuffer_size = Vector2::new(w as f32, h as f32);
                self.projection.set_aspect(w as f32 / h as f32);
                self.update_projviews();
            }
            _ => {}
        }
    }

    #[inline]
    fn upload(
        &self,
        _: usize,
        proj: &mut ShaderUniform<Matrix4<f32>>,
        view: &mut ShaderUniform<Matrix4<f32>>,
    ) {
        proj.upload(&self.proj);
        view.upload(&self.view);
    }

    fn transformation(&self) -> Matrix4<f32> {
        self.proj_view
    }

    fn inverse_transformation(&self) -> Matrix4<f32> {
        self.inverse_proj_view
    }

    fn update(&mut self, _: &Canvas) {}
}

#[derive(Clone, Copy, Debug)]
struct CoordSystemRh {
    up_axis: Unit<Vector3<f32>>,
    rotation_to_y_up: UnitQuaternion<f32>,
}

impl CoordSystemRh {
    #[inline]
    fn from_up_axis(up_axis: Unit<Vector3<f32>>) -> Self {
        let rotation_to_y_up = UnitQuaternion::rotation_between_axis(&up_axis, &Vector3::y_axis())
            .unwrap_or_else(|| {
                UnitQuaternion::from_axis_angle(&Vector3::x_axis(), std::f32::consts::PI)
            });
        Self {
            up_axis,
            rotation_to_y_up,
        }
    }
}
