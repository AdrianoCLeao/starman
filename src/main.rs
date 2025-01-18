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
    let mut window = Window::new("Starman Project");

    let obj_path = Path::new("assets/rocket/rocket.obj");
    let mtl_path = Path::new("assets/rocket/rocket");
    let mut rocket = window.add_obj(obj_path, mtl_path, Vector3::new(0.1, 0.1, 0.1));

    window.set_light(Light::StickToCamera);

    let rot_rocket = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.014);

    while window.render() {
        rocket.prepend_to_local_rotation(&rot_rocket);
    }
}

