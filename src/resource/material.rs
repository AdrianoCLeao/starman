use crate::camera::camera::Camera;
use crate::light::Light;
use crate::resource::mesh::{Mesh};
use crate::scene::object::{ObjectData};
use nalgebra::{Isometry2, Isometry3, Vector2, Vector3};

pub trait Material {
    fn render(
        &mut self,
        pass: usize,
        transform: &Isometry3<f32>,
        scale: &Vector3<f32>,
        camera: &mut dyn Camera,
        light: &Light,           
        data: &ObjectData,
        mesh: &mut Mesh,
    );
}