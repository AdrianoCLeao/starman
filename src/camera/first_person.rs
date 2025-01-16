use crate::camera::camera::Camera;
use crate::event::window_event::{Action, Key, MouseButton, WindowEvent};
use crate::resource::effect::ShaderUniform;
use crate::window::canvas::Canvas;
use nalgebra::{
    self, Isometry3, Matrix4, Perspective3, Point3, Translation3, Unit, UnitQuaternion, Vector2,
    Vector3,
};
use num_traits::Zero;
use std::f32;

#[derive(Debug, Clone)]
pub struct FirstPerson {
    eye: Point3<f32>,
    yaw: f32,
    pitch: f32,

    yaw_step: f32,
    pitch_step: f32,
    move_step: f32,
    rotate_button: Option<MouseButton>,
    drag_button: Option<MouseButton>,
    up_key: Option<Key>,
    down_key: Option<Key>,
    left_key: Option<Key>,
    right_key: Option<Key>,

    projection: Perspective3<f32>,
    proj: Matrix4<f32>,
    view: Matrix4<f32>,
    proj_view: Matrix4<f32>,
    inverse_proj_view: Matrix4<f32>,
    last_cursor_pos: Vector2<f32>,
    coord_system: CoordSystemRh,
}

impl FirstPerson {
    pub fn new(eye: Point3<f32>, at: Point3<f32>) -> FirstPerson {
        FirstPerson::new_with_frustrum(f32::consts::PI / 4.0, 0.1, 1024.0, eye, at)
    }

    pub fn new_with_frustrum(
        fov: f32,
        znear: f32,
        zfar: f32,
        eye: Point3<f32>,
        at: Point3<f32>,
    ) -> FirstPerson {
        let mut res = FirstPerson {
            eye: Point3::new(0.0, 0.0, 0.0),
            yaw: 0.0,
            pitch: 0.0,
            yaw_step: 0.005,
            pitch_step: 0.005,
            move_step: 0.5,
            rotate_button: Some(MouseButton::Button1),
            drag_button: Some(MouseButton::Button2),
            up_key: Some(Key::Up),
            down_key: Some(Key::Down),
            left_key: Some(Key::Left),
            right_key: Some(Key::Right),
            projection: Perspective3::new(800.0 / 600.0, fov, znear, zfar),
            proj: nalgebra::zero(),
            view: nalgebra::zero(),
            proj_view: nalgebra::zero(),
            inverse_proj_view: nalgebra::zero(),
            last_cursor_pos: nalgebra::zero(),
            coord_system: CoordSystemRh::from_up_axis(Vector3::y_axis()),
        };

        res.look_at(eye, at);

        res
    }

    #[inline]
    pub fn set_move_step(&mut self, step: f32) {
        self.move_step = step;
    }

    #[inline]
    pub fn set_pitch_step(&mut self, step: f32) {
        self.pitch_step = step;
    }

    #[inline]
    pub fn set_yaw_step(&mut self, step: f32) {
        self.yaw_step = step;
    }

    #[inline]
    pub fn move_step(&self) -> f32 {
        self.move_step
    }

    #[inline]
    pub fn pitch_step(&self) -> f32 {
        self.pitch_step
    }

    #[inline]
    pub fn yaw_step(&self) -> f32 {
        self.yaw_step
    }

    pub fn look_at(&mut self, eye: Point3<f32>, at: Point3<f32>) {
        let dist = (eye - at).norm();

        let view_eye = self.coord_system.rotation_to_y_up * eye;
        let view_at = self.coord_system.rotation_to_y_up * at;
        let pitch = ((view_at.y - view_eye.y) / dist).acos();
        let yaw = (view_at.z - view_eye.z).atan2(view_at.x - view_eye.x);

        self.eye = eye;
        self.yaw = yaw;
        self.pitch = pitch;
        self.update_projviews();
    }

    pub fn at(&self) -> Point3<f32> {
        let view_eye = self.coord_system.rotation_to_y_up * self.eye;
        let ax = view_eye.x + self.yaw.cos() * self.pitch.sin();
        let ay = view_eye.y + self.pitch.cos();
        let az = view_eye.z + self.yaw.sin() * self.pitch.sin();
        self.coord_system.rotation_to_y_up.inverse() * Point3::new(ax, ay, az)
    }

    fn update_restrictions(&mut self) {
        if self.pitch <= 0.01 {
            self.pitch = 0.01
        }

        let _pi: f32 = f32::consts::PI;
        if self.pitch > _pi - 0.01 {
            self.pitch = _pi - 0.01
        }
    }

    pub fn rotate_button(&self) -> Option<MouseButton> {
        self.rotate_button
    }

    pub fn rebind_rotate_button(&mut self, new_button: Option<MouseButton>) {
        self.rotate_button = new_button;
    }

    pub fn drag_button(&self) -> Option<MouseButton> {
        self.drag_button
    }

    pub fn rebind_drag_button(&mut self, new_button: Option<MouseButton>) {
        self.drag_button = new_button;
    }

    pub fn up_key(&self) -> Option<Key> {
        self.up_key
    }

    pub fn down_key(&self) -> Option<Key> {
        self.down_key
    }

    pub fn left_key(&self) -> Option<Key> {
        self.left_key
    }

    pub fn right_key(&self) -> Option<Key> {
        self.right_key
    }

    pub fn rebind_up_key(&mut self, new_key: Option<Key>) {
        self.up_key = new_key;
    }

    pub fn rebind_down_key(&mut self, new_key: Option<Key>) {
        self.down_key = new_key;
    }

    pub fn rebind_left_key(&mut self, new_key: Option<Key>) {
        self.left_key = new_key;
    }

    pub fn rebind_right_key(&mut self, new_key: Option<Key>) {
        self.right_key = new_key;
    }

    pub fn unbind_movement_keys(&mut self) {
        self.up_key = None;
        self.down_key = None;
        self.left_key = None;
        self.right_key = None;
    }

    #[doc(hidden)]
    pub fn handle_left_button_displacement(&mut self, dpos: &Vector2<f32>) {
        self.yaw = self.yaw + dpos.x * self.yaw_step;
        self.pitch = self.pitch + dpos.y * self.pitch_step;

        self.update_restrictions();
        self.update_projviews();
    }

    #[doc(hidden)]
    pub fn handle_right_button_displacement(&mut self, dpos: &Vector2<f32>) {
        let at = self.at();
        let dir = (at - self.eye).normalize();
        let tangent = self.coord_system.up_axis.cross(&dir).normalize();
        let bitangent = dir.cross(&tangent);

        self.eye = self.eye + tangent * (0.01 * dpos.x / 10.0) + bitangent * (0.01 * dpos.y / 10.0);
        self.update_restrictions();
        self.update_projviews();
    }

    #[doc(hidden)]
    pub fn handle_scroll(&mut self, yoff: f32) {
        let front = self.observer_frame() * Vector3::z();

        self.eye = self.eye + front * (self.move_step * yoff);

        self.update_restrictions();
        self.update_projviews();
    }

    fn update_projviews(&mut self) {
        self.view = self.view_transform().to_homogeneous();
        self.proj = *self.projection.as_matrix();
        self.proj_view = self.proj * self.view;
        let _ = self
            .proj_view
            .try_inverse()
            .map(|inverse_proj| self.inverse_proj_view = inverse_proj);
    }

    pub fn eye_dir(&self) -> Vector3<f32> {
        (self.at() - self.eye).normalize()
    }

    pub fn move_dir(&self, up: bool, down: bool, right: bool, left: bool) -> Vector3<f32> {
        let t = self.observer_frame();
        let frontv = t * Vector3::z();
        let rightv = t * Vector3::x();

        let mut movement = nalgebra::zero::<Vector3<f32>>();

        if up {
            movement = movement + frontv
        }

        if down {
            movement = movement - frontv
        }

        if right {
            movement = movement - rightv
        }

        if left {
            movement = movement + rightv
        }

        if movement.is_zero() {
            movement
        } else {
            movement.normalize()
        }
    }

    #[inline]
    pub fn translate_mut(&mut self, t: &Translation3<f32>) {
        let new_eye = t * self.eye;

        self.set_eye(new_eye);
    }

    #[inline]
    pub fn translate(&self, t: &Translation3<f32>) -> FirstPerson {
        let mut res = self.clone();
        res.translate_mut(t);
        res
    }

    #[inline]
    fn set_eye(&mut self, eye: Point3<f32>) {
        self.eye = eye;
        self.update_restrictions();
        self.update_projviews();
    }

    #[inline]
    pub fn set_up_axis(&mut self, up_axis: Vector3<f32>) {
        self.set_up_axis_dir(Unit::new_normalize(up_axis));
    }

    #[inline]
    pub fn set_up_axis_dir(&mut self, up_axis: Unit<Vector3<f32>>) {
        if self.coord_system.up_axis != up_axis {
            let new_coord_system = CoordSystemRh::from_up_axis(up_axis);
            let old_at = self.at();
            self.coord_system = new_coord_system;
            self.look_at(self.eye, old_at);
        }
    }

    fn observer_frame(&self) -> Isometry3<f32> {
        Isometry3::face_towards(&self.eye, &self.at(), &self.coord_system.up_axis)
    }
}

impl Camera for FirstPerson {
    fn clip_planes(&self) -> (f32, f32) {
        (self.projection.znear(), self.projection.zfar())
    }

    fn view_transform(&self) -> Isometry3<f32> {
        Isometry3::look_at_rh(&self.eye, &self.at(), &self.coord_system.up_axis)
    }

    fn handle_event(&mut self, canvas: &Canvas, event: &WindowEvent) {
        match *event {
            WindowEvent::CursorPos(x, y, _) => {
                let curr_pos = Vector2::new(x as f32, y as f32);

                if let Some(rotate_button) = self.rotate_button {
                    if canvas.get_mouse_button(rotate_button) == Action::Press {
                        let dpos = curr_pos - self.last_cursor_pos;
                        self.handle_left_button_displacement(&dpos)
                    }
                }

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
                self.projection.set_aspect(w as f32 / h as f32);
                self.update_projviews();
            }
            _ => {}
        }
    }

    fn eye(&self) -> Point3<f32> {
        self.eye
    }

    fn transformation(&self) -> Matrix4<f32> {
        self.proj_view
    }

    fn inverse_transformation(&self) -> Matrix4<f32> {
        self.inverse_proj_view
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

    fn update(&mut self, canvas: &Canvas) {
        let up = check_optional_key_state(canvas, self.up_key, Action::Press);
        let down = check_optional_key_state(canvas, self.down_key, Action::Press);
        let right = check_optional_key_state(canvas, self.right_key, Action::Press);
        let left = check_optional_key_state(canvas, self.left_key, Action::Press);
        let dir = self.move_dir(up, down, right, left);

        let move_amount = dir * self.move_step;
        self.translate_mut(&Translation3::from(move_amount));
    }
}

fn check_optional_key_state(canvas: &Canvas, key: Option<Key>, key_state: Action) -> bool {
    if let Some(actual_key) = key {
        canvas.get_key(actual_key) == key_state
    } else {
        false
    }
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
