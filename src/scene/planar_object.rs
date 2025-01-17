//! Data structure of a scene node.

use crate::planar_camera::PlanarCamera;
use crate::resource::vertex_index::VertexIndex;
use crate::resource::planar_mesh::PlanarMesh;
use crate::resource::texture_manager::TextureManager;
use crate::context::context::Texture;
use crate::resource::material::PlanarMaterial;
use nalgebra::{Isometry2, Point2, Point3, Vector2};
use std::any::Any;
use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;

pub struct PlanarObjectData {
    material: Rc<RefCell<Box<dyn PlanarMaterial + 'static>>>,
    texture: Rc<Texture>,
    color: Point3<f32>,
    lines_color: Option<Point3<f32>>,
    wlines: f32,
    wpoints: f32,
    draw_surface: bool,
    cull: bool,
    user_data: Box<dyn Any + 'static>,
}

impl PlanarObjectData {
    #[inline]
    pub fn texture(&self) -> &Rc<Texture> {
        &self.texture
    }

    #[inline]
    pub fn color(&self) -> &Point3<f32> {
        &self.color
    }

    #[inline]
    pub fn lines_width(&self) -> f32 {
        self.wlines
    }

    #[inline]
    pub fn lines_color(&self) -> Option<&Point3<f32>> {
        self.lines_color.as_ref()
    }

    #[inline]
    pub fn points_size(&self) -> f32 {
        self.wpoints
    }

    #[inline]
    pub fn surface_rendering_active(&self) -> bool {
        self.draw_surface
    }

    #[inline]
    pub fn backface_culling_enabled(&self) -> bool {
        self.cull
    }

    #[inline]
    pub fn user_data(&self) -> &dyn Any {
        &*self.user_data
    }
}

pub struct PlanarObject {
    data: PlanarObjectData,
    mesh: Rc<RefCell<PlanarMesh>>,
}

impl PlanarObject {
    #[doc(hidden)]
    pub fn new(
        mesh: Rc<RefCell<PlanarMesh>>,
        r: f32,
        g: f32,
        b: f32,
        texture: Rc<Texture>,
        material: Rc<RefCell<Box<dyn PlanarMaterial + 'static>>>,
    ) -> PlanarObject {
        let user_data = ();
        let data = PlanarObjectData {
            color: Point3::new(r, g, b),
            lines_color: None,
            texture,
            wlines: 0.0,
            wpoints: 0.0,
            draw_surface: true,
            cull: true,
            material,
            user_data: Box::new(user_data),
        };

        PlanarObject { data, mesh }
    }

    #[doc(hidden)]
    pub fn render(
        &self,
        transform: &Isometry2<f32>,
        scale: &Vector2<f32>,
        camera: &mut dyn PlanarCamera,
    ) {
        self.data.material.borrow_mut().render(
            transform,
            scale,
            camera,
            &self.data,
            &mut *self.mesh.borrow_mut(),
        );
    }

    #[inline]
    pub fn data(&self) -> &PlanarObjectData {
        &self.data
    }

    #[inline]
    pub fn data_mut(&mut self) -> &mut PlanarObjectData {
        &mut self.data
    }

    #[inline]
    pub fn enable_backface_culling(&mut self, active: bool) {
        self.data.cull = active;
    }

    #[inline]
    pub fn set_user_data(&mut self, user_data: Box<dyn Any + 'static>) {
        self.data.user_data = user_data;
    }

    #[inline]
    pub fn material(&self) -> Rc<RefCell<Box<dyn PlanarMaterial + 'static>>> {
        self.data.material.clone()
    }

    #[inline]
    pub fn set_material(&mut self, material: Rc<RefCell<Box<dyn PlanarMaterial + 'static>>>) {
        self.data.material = material;
    }

    #[inline]
    pub fn set_lines_width(&mut self, width: f32) {
        self.data.wlines = width
    }

    #[inline]
    pub fn lines_width(&self) -> f32 {
        self.data.wlines
    }

    #[inline]
    pub fn set_lines_color(&mut self, color: Option<Point3<f32>>) {
        self.data.lines_color = color
    }

    #[inline]
    pub fn lines_color(&self) -> Option<&Point3<f32>> {
        self.data.lines_color()
    }

    #[inline]
    pub fn set_points_size(&mut self, size: f32) {
        self.data.wpoints = size
    }

    #[inline]
    pub fn points_size(&self) -> f32 {
        self.data.wpoints
    }

    #[inline]
    pub fn set_surface_rendering_activation(&mut self, active: bool) {
        self.data.draw_surface = active
    }

    #[inline]
    pub fn surface_rendering_activation(&self) -> bool {
        self.data.draw_surface
    }

    #[inline]
    pub fn mesh(&self) -> &Rc<RefCell<PlanarMesh>> {
        &self.mesh
    }

    #[inline(always)]
    pub fn modify_vertices<F: FnMut(&mut Vec<Point2<f32>>)>(&mut self, f: &mut F) {
        let bmesh = self.mesh.borrow_mut();
        let _ = bmesh
            .coords()
            .write()
            .unwrap()
            .data_mut()
            .as_mut()
            .map(|coords| f(coords));
    }

    #[inline(always)]
    pub fn read_vertices<F: FnMut(&[Point2<f32>])>(&self, f: &mut F) {
        let bmesh = self.mesh.borrow();
        let _ = bmesh
            .coords()
            .read()
            .unwrap()
            .data()
            .as_ref()
            .map(|coords| f(&coords[..]));
    }

    #[inline(always)]
    pub fn modify_faces<F: FnMut(&mut Vec<Point3<VertexIndex>>)>(&mut self, f: &mut F) {
        let bmesh = self.mesh.borrow_mut();
        let _ = bmesh
            .faces()
            .write()
            .unwrap()
            .data_mut()
            .as_mut()
            .map(|faces| f(faces));
    }

    #[inline(always)]
    pub fn read_faces<F: FnMut(&[Point3<VertexIndex>])>(&self, f: &mut F) {
        let bmesh = self.mesh.borrow();
        let _ = bmesh
            .faces()
            .read()
            .unwrap()
            .data()
            .as_ref()
            .map(|faces| f(&faces[..]));
    }

    #[inline(always)]
    pub fn modify_uvs<F: FnMut(&mut Vec<Point2<f32>>)>(&mut self, f: &mut F) {
        let bmesh = self.mesh.borrow_mut();
        let _ = bmesh
            .uvs()
            .write()
            .unwrap()
            .data_mut()
            .as_mut()
            .map(|uvs| f(uvs));
    }

    #[inline(always)]
    pub fn read_uvs<F: FnMut(&[Point2<f32>])>(&self, f: &mut F) {
        let bmesh = self.mesh.borrow();
        let _ = bmesh
            .uvs()
            .read()
            .unwrap()
            .data()
            .as_ref()
            .map(|uvs| f(&uvs[..]));
    }

    #[inline]
    pub fn set_color(&mut self, r: f32, g: f32, b: f32) {
        self.data.color.x = r;
        self.data.color.y = g;
        self.data.color.z = b;
    }

    #[inline]
    pub fn set_texture_from_file(&mut self, path: &Path, name: &str) {
        let texture = TextureManager::get_global_manager(|tm| tm.add(path, name));

        self.set_texture(texture)
    }

    #[inline]
    pub fn set_texture_with_name(&mut self, name: &str) {
        let texture = TextureManager::get_global_manager(|tm| {
            tm.get(name).unwrap_or_else(|| {
                panic!("Invalid attempt to use the unregistered texture: {}", name)
            })
        });

        self.set_texture(texture)
    }

    #[inline]
    pub fn set_texture(&mut self, texture: Rc<Texture>) {
        self.data.texture = texture
    }
}
