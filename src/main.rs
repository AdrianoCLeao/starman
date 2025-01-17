#[macro_use]
extern crate serde_derive;
extern crate serde;
extern crate bitflags;

use light::Light;
use nalgebra::{Translation3, UnitQuaternion, Vector3};
use std::f32;
use std::path::Path;

mod window;
use window::window::Window;

mod loader;
mod light;
mod resource;
mod context;
mod error;
mod event;
mod text;
mod camera;
mod scene;
mod builtin;
mod renderer;
mod planar_camera;
mod post_processing;
mod planar_line_renderer;

fn main() {
    let mut window = Window::new("Kiss3d: obj");

    let obj_path = Path::new("assets/teapot/teapot.obj");
    let mtl_path = Path::new("assets/teapot");
    let mut teapot = window.add_obj(obj_path, mtl_path, Vector3::new(0.001, 0.001, 0.001));
    teapot.append_translation(&Translation3::new(0.0, -0.05, -0.2));

    let obj_path = Path::new("assets/rust_logo/rust_logo.obj");
    let mtl_path = Path::new("assets/rust_logo");
    let mut rust = window.add_obj(obj_path, mtl_path, Vector3::new(0.05, 0.05, 0.05));
    rust.prepend_to_local_rotation(&UnitQuaternion::from_axis_angle(
        &Vector3::x_axis(),
        -f32::consts::FRAC_PI_2,
    ));
    rust.set_color(0.0, 0.0, 1.0);

    window.set_light(Light::StickToCamera);

    let rot_teapot = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.014);
    let rot_rust = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), -0.014);

    while window.render() {
        teapot.prepend_to_local_rotation(&rot_teapot);
        rust.prepend_to_local_rotation(&rot_rust);
    }
}

