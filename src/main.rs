#[macro_use]
extern crate serde_derive;
extern crate bitflags;
extern crate serde;

use light::Light;
use nalgebra::{Point2, Point3, Translation3, UnitQuaternion, Vector3};
use std::f32;
use std::path::Path;

mod window;
use window::window::Window;

mod builtin;
mod camera;
mod context;
mod error;
mod event;
mod light;
mod loader;
mod planar_camera;
mod planar_line_renderer;
mod post_processing;
mod renderer;
mod resource;
mod scene;
mod text;

fn main() {
    let mut window = Window::new("Starman Project");

    let obj_path = Path::new("assets/rocket/rocket.obj");
    let mtl_path = Path::new("assets/rocket/rocket.mtl");
    let glb_path = Path::new("assets/box.glb");

    let mut glb = window.add_glb(glb_path, Vector3::new(0.1, 0.1, 0.1));
    glb.append_translation(&Translation3::new(0.0, 0.0, 0.0));

    let mut rocket = window.add_obj(obj_path, mtl_path, Vector3::new(0.1, 0.1, 0.1));
    rocket.append_translation(&Translation3::new(0.3, -0.1, 0.0));

    window.set_light(Light::StickToCamera);

    let rot_rocket = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.014);

    while window.render() {
        rocket.prepend_to_local_rotation(&rot_rocket);
        glb.prepend_to_local_rotation(&rot_rocket);

        window.draw_bar();

        window.draw_line(
            &Point3::origin(),
            &Point3::new(1.0, 0.0, 0.0),
            &Point3::new(1.0, 0.0, 0.0),
        );
        window.draw_line(
            &Point3::origin(),
            &Point3::new(0.0, 1.0, 0.0),
            &Point3::new(0.0, 1.0, 0.0),
        );
        window.draw_line(
            &Point3::origin(),
            &Point3::new(0.0, 0.0, 1.0),
            &Point3::new(0.0, 0.0, 1.0),
        );
    }
}
