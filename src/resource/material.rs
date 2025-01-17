use crate::camera::camera::Camera;
use crate::light::Light;
use crate::planar_camera::PlanarCamera;
use crate::resource::mesh::Mesh;
use crate::scene::object::ObjectData;
use crate::scene::planar_object::PlanarObjectData;
use nalgebra::{Isometry2, Isometry3, Vector2, Vector3};

use super::planar_mesh::PlanarMesh;

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

pub trait PlanarMaterial {
    fn render(
        &mut self,
        transform: &Isometry2<f32>,
        scale: &Vector2<f32>,
        camera: &mut dyn PlanarCamera,
        data: &PlanarObjectData,
        mesh: &mut PlanarMesh,
    );
}