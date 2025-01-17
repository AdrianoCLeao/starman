use nalgebra::{self as na, Isometry2, Point2, Point3, Translation2, UnitComplex, Vector2};

use crate::planar_camera::PlanarCamera;
use crate::resource::planar_material_manager::PlanarMaterialManager;
use crate::resource::planar_mesh::PlanarMesh;
use crate::resource::planar_mesh_manager::PlanarMeshManager;
use crate::resource::vertex_index::VertexIndex;
use crate::resource::material::PlanarMaterial;
use crate::resource::texture_manager::TextureManager;
use crate::context::context::Texture;
use crate::scene::planar_object::PlanarObject;
use std::cell::{Ref, RefCell, RefMut};
use std::f32;
use std::mem;
use std::path::Path;
use std::rc::Rc;

pub struct PlanarSceneNodeData {
    local_scale: Vector2<f32>,
    local_transform: Isometry2<f32>,
    world_scale: Vector2<f32>,
    world_transform: Isometry2<f32>,
    visible: bool,
    up_to_date: bool,
    children: Vec<PlanarSceneNode>,
    object: Option<PlanarObject>,
    parent: Option<*const RefCell<PlanarSceneNodeData>>,
}

#[derive(Clone)]
pub struct PlanarSceneNode {
    data: Rc<RefCell<PlanarSceneNodeData>>,
}

impl PlanarSceneNodeData {
    fn set_parent(&mut self, parent: *const RefCell<PlanarSceneNodeData>) {
        self.parent = Some(parent);
    }

    fn remove_from_parent(&mut self, to_remove: &PlanarSceneNode) {
        let _ = self.parent.as_ref().map(|p| unsafe {
            let mut bp = (**p).borrow_mut();
            bp.remove(to_remove)
        });
    }

    fn remove(&mut self, o: &PlanarSceneNode) {
        if let Some(i) = self
            .children
            .iter()
            .rposition(|e| std::ptr::eq(&*o.data, &*e.data))
        {
            let _ = self.children.swap_remove(i);
        }
    }

    #[inline]
    pub fn has_object(&self) -> bool {
        self.object.is_some()
    }

    #[inline]
    pub fn is_root(&self) -> bool {
        self.parent.is_none()
    }

    pub fn render(&mut self, camera: &mut dyn PlanarCamera) {
        if self.visible {
            self.do_render(&na::one(), &Vector2::from_element(1.0), camera)
        }
    }

    fn do_render(
        &mut self,
        transform: &Isometry2<f32>,
        scale: &Vector2<f32>,
        camera: &mut dyn PlanarCamera,
    ) {
        if !self.up_to_date {
            self.up_to_date = true;
            self.world_transform = *transform * self.local_transform;
            self.world_scale = scale.component_mul(&self.local_scale);
        }

        if let Some(ref o) = self.object {
            o.render(&self.world_transform, &self.world_scale, camera)
        }

        for c in self.children.iter_mut() {
            let mut bc = c.data_mut();
            if bc.visible {
                bc.do_render(&self.world_transform, &self.world_scale, camera)
            }
        }
    }

    #[inline]
    pub fn object(&self) -> Option<&PlanarObject> {
        self.object.as_ref()
    }

    #[inline]
    pub fn object_mut(&mut self) -> Option<&mut PlanarObject> {
        self.object.as_mut()
    }

    #[inline]
    pub fn get_object(&self) -> &PlanarObject {
        self.object()
            .expect("This scene node does not contain an PlanarObject.")
    }

    #[inline]
    pub fn get_object_mut(&mut self) -> &mut PlanarObject {
        self.object_mut()
            .expect("This scene node does not contain an PlanarObject.")
    }

    #[inline]
    pub fn set_material(&mut self, material: Rc<RefCell<Box<dyn PlanarMaterial + 'static>>>) {
        self.apply_to_objects_mut(&mut |o| o.set_material(material.clone()))
    }

    #[inline]
    pub fn set_material_with_name(&mut self, name: &str) {
        let material = PlanarMaterialManager::get_global_manager(|tm| {
            tm.get(name).unwrap_or_else(|| {
                panic!("Invalid attempt to use the unregistered material: {}", name)
            })
        });

        self.set_material(material)
    }

    #[inline]
    pub fn set_lines_width(&mut self, width: f32) {
        self.apply_to_objects_mut(&mut |o| o.set_lines_width(width))
    }

    #[inline]
    pub fn set_lines_color(&mut self, color: Option<Point3<f32>>) {
        self.apply_to_objects_mut(&mut |o| o.set_lines_color(color))
    }

    #[inline]
    pub fn set_points_size(&mut self, size: f32) {
        self.apply_to_objects_mut(&mut |o| o.set_points_size(size))
    }

    #[inline]
    pub fn set_surface_rendering_activation(&mut self, active: bool) {
        self.apply_to_objects_mut(&mut |o| o.set_surface_rendering_activation(active))
    }

    #[inline]
    pub fn enable_backface_culling(&mut self, active: bool) {
        self.apply_to_objects_mut(&mut |o| o.enable_backface_culling(active))
    }

    #[inline(always)]
    pub fn modify_vertices<F: FnMut(&mut Vec<Point2<f32>>)>(&mut self, f: &mut F) {
        self.apply_to_objects_mut(&mut |o| o.modify_vertices(f))
    }

    #[inline(always)]
    pub fn read_vertices<F: FnMut(&[Point2<f32>])>(&self, f: &mut F) {
        self.apply_to_objects(&mut |o| o.read_vertices(f))
    }

    #[inline(always)]
    pub fn modify_faces<F: FnMut(&mut Vec<Point3<VertexIndex>>)>(&mut self, f: &mut F) {
        self.apply_to_objects_mut(&mut |o| o.modify_faces(f))
    }

    #[inline(always)]
    pub fn read_faces<F: FnMut(&[Point3<VertexIndex>])>(&self, f: &mut F) {
        self.apply_to_objects(&mut |o| o.read_faces(f))
    }

    #[inline(always)]
    pub fn modify_uvs<F: FnMut(&mut Vec<Point2<f32>>)>(&mut self, f: &mut F) {
        self.apply_to_objects_mut(&mut |o| o.modify_uvs(f))
    }

    #[inline(always)]
    pub fn read_uvs<F: FnMut(&[Point2<f32>])>(&self, f: &mut F) {
        self.apply_to_objects(&mut |o| o.read_uvs(f))
    }

    #[inline]
    pub fn is_visible(&self) -> bool {
        self.visible
    }

    #[inline]
    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    #[inline]
    pub fn set_color(&mut self, r: f32, g: f32, b: f32) {
        self.apply_to_objects_mut(&mut |o| o.set_color(r, g, b))
    }

    #[inline]
    pub fn set_texture_from_file(&mut self, path: &Path, name: &str) {
        let texture = TextureManager::get_global_manager(|tm| tm.add(path, name));

        self.set_texture(texture)
    }

    #[inline]
    pub fn set_texture_from_memory(&mut self, image_data: &[u8], name: &str) {
        let texture =
            TextureManager::get_global_manager(|tm| tm.add_image_from_memory(image_data, name));

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

    pub fn set_texture(&mut self, texture: Rc<Texture>) {
        self.apply_to_objects_mut(&mut |o| o.set_texture(texture.clone()))
    }

    #[inline]
    pub fn apply_to_objects_mut<F: FnMut(&mut PlanarObject)>(&mut self, f: &mut F) {
        if let Some(ref mut o) = self.object {
            f(o)
        }

        for c in self.children.iter_mut() {
            c.data_mut().apply_to_objects_mut(f)
        }
    }

    #[inline]
    pub fn apply_to_objects<F: FnMut(&PlanarObject)>(&self, f: &mut F) {
        if let Some(ref o) = self.object {
            f(o)
        }

        for c in self.children.iter() {
            c.data().apply_to_objects(f)
        }
    }

    #[inline]
    pub fn set_local_scale(&mut self, sx: f32, sy: f32) {
        self.invalidate();
        self.local_scale = Vector2::new(sx, sy)
    }

    #[inline]
    pub fn local_scale(&self) -> Vector2<f32> {
        self.local_scale
    }

    #[inline]
    pub fn local_transformation(&self) -> Isometry2<f32> {
        self.local_transform
    }

    #[inline]
    pub fn inverse_local_transformation(&self) -> Isometry2<f32> {
        self.local_transform.inverse()
    }

    #[inline]
    #[allow(mutable_transmutes)]
    pub fn world_transformation(&self) -> Isometry2<f32> {
        unsafe {
            let mself: &mut PlanarSceneNodeData = mem::transmute(self);
            mself.update();
        }
        self.world_transform
    }

    #[inline]
    #[allow(mutable_transmutes)]
    pub fn inverse_world_transformation(&self) -> Isometry2<f32> {
        unsafe {
            let mself: &mut PlanarSceneNodeData = mem::transmute(self);
            mself.update();
        }
        self.local_transform.inverse()
    }

    #[inline]
    pub fn append_transformation(&mut self, t: &Isometry2<f32>) {
        self.invalidate();
        self.local_transform = t * self.local_transform
    }

    #[inline]
    pub fn prepend_to_local_transformation(&mut self, t: &Isometry2<f32>) {
        self.invalidate();
        self.local_transform *= t;
    }

    #[inline]
    pub fn set_local_transformation(&mut self, t: Isometry2<f32>) {
        self.invalidate();
        self.local_transform = t
    }

    #[inline]
    pub fn local_translation(&self) -> Translation2<f32> {
        self.local_transform.translation
    }

    #[inline]
    pub fn inverse_local_translation(&self) -> Translation2<f32> {
        self.local_transform.translation.inverse()
    }

    #[inline]
    pub fn append_translation(&mut self, t: &Translation2<f32>) {
        self.invalidate();
        self.local_transform = t * self.local_transform
    }

    #[inline]
    pub fn prepend_to_local_translation(&mut self, t: &Translation2<f32>) {
        self.invalidate();
        self.local_transform *= t
    }

    #[inline]
    pub fn set_local_translation(&mut self, t: Translation2<f32>) {
        self.invalidate();
        self.local_transform.translation = t
    }

    #[inline]
    pub fn local_rotation(&self) -> UnitComplex<f32> {
        self.local_transform.rotation
    }

    #[inline]
    pub fn inverse_local_rotation(&self) -> UnitComplex<f32> {
        self.local_transform.rotation.inverse()
    }

    #[inline]
    pub fn append_rotation(&mut self, r: &UnitComplex<f32>) {
        self.invalidate();
        self.local_transform = r * self.local_transform
    }

    #[inline]
    pub fn append_rotation_wrt_center(&mut self, r: &UnitComplex<f32>) {
        self.invalidate();
        self.local_transform.append_rotation_wrt_center_mut(r)
    }

    #[inline]
    pub fn prepend_to_local_rotation(&mut self, r: &UnitComplex<f32>) {
        self.invalidate();
        self.local_transform *= r
    }

    #[inline]
    pub fn set_local_rotation(&mut self, r: UnitComplex<f32>) {
        self.invalidate();
        self.local_transform.rotation = r
    }

    fn invalidate(&mut self) {
        self.up_to_date = false;

        for c in self.children.iter_mut() {
            let mut dm = c.data_mut();

            if dm.up_to_date {
                dm.invalidate()
            }
        }
    }

    fn update(&mut self) {
        if !self.up_to_date {
            match self.parent {
                Some(ref mut p) => unsafe {
                    let mut dp = (**p).borrow_mut();

                    dp.update();
                    self.world_transform = self.local_transform * dp.world_transform;
                    self.world_scale = self.local_scale.component_mul(&dp.local_scale);
                    self.up_to_date = true;
                    return;
                },
                None => {}
            }

            self.world_transform = self.local_transform;
            self.world_scale = self.local_scale;
            self.up_to_date = true;
        }
    }
}

impl PlanarSceneNode {
    pub fn new(
        local_scale: Vector2<f32>,
        local_transform: Isometry2<f32>,
        object: Option<PlanarObject>,
    ) -> PlanarSceneNode {
        let data = PlanarSceneNodeData {
            local_scale,
            local_transform,
            world_transform: local_transform,
            world_scale: local_scale,
            visible: true,
            up_to_date: false,
            children: Vec::new(),
            object,
            parent: None,
        };

        PlanarSceneNode {
            data: Rc::new(RefCell::new(data)),
        }
    }

    pub fn new_empty() -> PlanarSceneNode {
        PlanarSceneNode::new(Vector2::from_element(1.0), na::one(), None)
    }

    pub fn unlink(&mut self) {
        let self_self = self.clone();
        self.data_mut().remove_from_parent(&self_self);
        self.data_mut().parent = None
    }

    pub fn data(&self) -> Ref<PlanarSceneNodeData> {
        self.data.borrow()
    }

    pub fn data_mut(&mut self) -> RefMut<PlanarSceneNodeData> {
        self.data.borrow_mut()
    }

    pub fn add_group(&mut self) -> PlanarSceneNode {
        let node = PlanarSceneNode::new_empty();

        self.add_child(node.clone());

        node
    }

    pub fn add_child(&mut self, node: PlanarSceneNode) {
        assert!(
            node.data().is_root(),
            "The added node must not have a parent yet."
        );

        let mut node = node;
        node.data_mut().set_parent(&*self.data);
        self.data_mut().children.push(node)
    }

    pub fn add_object(
        &mut self,
        local_scale: Vector2<f32>,
        local_transform: Isometry2<f32>,
        object: PlanarObject,
    ) -> PlanarSceneNode {
        let node = PlanarSceneNode::new(local_scale, local_transform, Some(object));

        self.add_child(node.clone());

        node
    }

    pub fn add_rectangle(&mut self, wx: f32, wy: f32) -> PlanarSceneNode {
        let res = self.add_geom_with_name("rectangle", Vector2::new(wx, wy));

        res.expect("Unable to load the default rectangle geometry.")
    }

    pub fn add_circle(&mut self, r: f32) -> PlanarSceneNode {
        let res = self.add_geom_with_name("circle", Vector2::new(r * 2.0, r * 2.0));

        res.expect("Unable to load the default circle geometry.")
    }

    pub fn add_capsule(&mut self, r: f32, h: f32) -> PlanarSceneNode {
        let name = format!("capsule_{}_{}", r, h);

        let mesh = PlanarMeshManager::get_global_manager(|mm| {
            if let Some(geom) = mm.get(&name) {
                geom
            } else {
                let mut capsule_vtx = vec![Point2::origin()];
                let mut capsule_ids = Vec::new();
                let nsamples = 50;

                for i in 0..=nsamples {
                    let ang = (i as f32) / (nsamples as f32) * f32::consts::PI;
                    capsule_vtx.push(Point2::new(ang.cos() * r, ang.sin() * r + h / 2.0));
                    capsule_ids.push(Point3::new(
                        0,
                        capsule_vtx.len() as VertexIndex - 2,
                        capsule_vtx.len() as VertexIndex - 1,
                    ));
                }

                for i in nsamples..=nsamples * 2 {
                    let ang = (i as f32) / (nsamples as f32) * f32::consts::PI;
                    capsule_vtx.push(Point2::new(ang.cos() * r, ang.sin() * r - h / 2.0));
                    capsule_ids.push(Point3::new(
                        0,
                        capsule_vtx.len() as VertexIndex - 2,
                        capsule_vtx.len() as VertexIndex - 1,
                    ));
                }

                capsule_ids.push(Point3::new(0, capsule_vtx.len() as VertexIndex - 1, 1));

                let capsule = PlanarMesh::new(capsule_vtx, capsule_ids, None, false);
                let mesh = Rc::new(RefCell::new(capsule));
                mm.add(mesh.clone(), &name);
                mesh
            }
        });

        self.add_mesh(mesh, Vector2::repeat(1.0))
    }

    pub fn add_geom_with_name(
        &mut self,
        geometry_name: &str,
        scale: Vector2<f32>,
    ) -> Option<PlanarSceneNode> {
        PlanarMeshManager::get_global_manager(|mm| mm.get(geometry_name))
            .map(|g| self.add_mesh(g, scale))
    }

    pub fn add_mesh(
        &mut self,
        mesh: Rc<RefCell<PlanarMesh>>,
        scale: Vector2<f32>,
    ) -> PlanarSceneNode {
        let tex = TextureManager::get_global_manager(|tm| tm.get_default());
        let mat = PlanarMaterialManager::get_global_manager(|mm| mm.get_default());
        let object = PlanarObject::new(mesh, 1.0, 1.0, 1.0, tex, mat);

        self.add_object(scale, na::one(), object)
    }

    pub fn add_convex_polygon(
        &mut self,
        polygon: Vec<Point2<f32>>,
        scale: Vector2<f32>,
    ) -> PlanarSceneNode {
        let mut indices = Vec::new();

        for i in 1..polygon.len() - 1 {
            indices.push(Point3::new(0, i as VertexIndex, i as VertexIndex + 1));
        }

        let mesh = PlanarMesh::new(polygon, indices, None, false);
        let tex = TextureManager::get_global_manager(|tm| tm.get_default());
        let mat = PlanarMaterialManager::get_global_manager(|mm| mm.get_default());
        let object = PlanarObject::new(Rc::new(RefCell::new(mesh)), 1.0, 1.0, 1.0, tex, mat);

        self.add_object(scale, na::one(), object)
    }

    #[inline]
    pub fn apply_to_scene_nodes_mut<F: FnMut(&mut PlanarSceneNode)>(&mut self, f: &mut F) {
        f(self);

        for c in self.data_mut().children.iter_mut() {
            c.apply_to_scene_nodes_mut(f)
        }
    }

    #[inline]
    pub fn apply_to_scene_nodes<F: FnMut(&PlanarSceneNode)>(&self, f: &mut F) {
        f(self);

        for c in self.data().children.iter() {
            c.apply_to_scene_nodes(f)
        }
    }

    pub fn render(&mut self, camera: &mut dyn PlanarCamera) {
        self.data_mut().render(camera)
    }

    #[inline]
    pub fn set_material(&mut self, material: Rc<RefCell<Box<dyn PlanarMaterial + 'static>>>) {
        self.data_mut().set_material(material)
    }

    #[inline]
    pub fn set_material_with_name(&mut self, name: &str) {
        self.data_mut().set_material_with_name(name)
    }

    #[inline]
    pub fn set_lines_width(&mut self, width: f32) {
        self.data_mut().set_lines_width(width)
    }

    #[inline]
    pub fn set_lines_color(&mut self, color: Option<Point3<f32>>) {
        self.data_mut().set_lines_color(color)
    }

    #[inline]
    pub fn set_points_size(&mut self, size: f32) {
        self.data_mut().set_points_size(size)
    }

    #[inline]
    pub fn set_surface_rendering_activation(&mut self, active: bool) {
        self.data_mut().set_surface_rendering_activation(active)
    }

    #[inline]
    pub fn enable_backface_culling(&mut self, active: bool) {
        self.data_mut().enable_backface_culling(active)
    }

    #[inline(always)]
    pub fn modify_vertices<F: FnMut(&mut Vec<Point2<f32>>)>(&mut self, f: &mut F) {
        self.data_mut().modify_vertices(f)
    }

    #[inline(always)]
    pub fn read_vertices<F: FnMut(&[Point2<f32>])>(&self, f: &mut F) {
        self.data().read_vertices(f)
    }

    #[inline(always)]
    pub fn modify_faces<F: FnMut(&mut Vec<Point3<VertexIndex>>)>(&mut self, f: &mut F) {
        self.data_mut().modify_faces(f)
    }

    #[inline(always)]
    pub fn read_faces<F: FnMut(&[Point3<VertexIndex>])>(&self, f: &mut F) {
        self.data().read_faces(f)
    }

    #[inline(always)]
    pub fn modify_uvs<F: FnMut(&mut Vec<Point2<f32>>)>(&mut self, f: &mut F) {
        self.data_mut().modify_uvs(f)
    }

    #[inline(always)]
    pub fn read_uvs<F: FnMut(&[Point2<f32>])>(&self, f: &mut F) {
        self.data().read_uvs(f)
    }

    #[inline]
    pub fn is_visible(&self) -> bool {
        self.data().is_visible()
    }

    #[inline]
    pub fn set_visible(&mut self, visible: bool) {
        self.data_mut().set_visible(visible)
    }

    #[inline]
    pub fn set_color(&mut self, r: f32, g: f32, b: f32) {
        self.data_mut().set_color(r, g, b)
    }

    #[inline]
    pub fn set_texture_from_file(&mut self, path: &Path, name: &str) {
        self.data_mut().set_texture_from_file(path, name)
    }

    pub fn set_texture_from_memory(&mut self, image_data: &[u8], name: &str) {
        self.data_mut().set_texture_from_memory(image_data, name)
    }

    #[inline]
    pub fn set_texture_with_name(&mut self, name: &str) {
        self.data_mut().set_texture_with_name(name)
    }

    pub fn set_texture(&mut self, texture: Rc<Texture>) {
        self.data_mut().set_texture(texture)
    }

    #[inline]
    pub fn set_local_scale(&mut self, sx: f32, sy: f32) {
        self.data_mut().set_local_scale(sx, sy)
    }

    #[inline]
    pub fn append_transformation(&mut self, t: &Isometry2<f32>) {
        self.data_mut().append_transformation(t)
    }

    #[inline]
    pub fn prepend_to_local_transformation(&mut self, t: &Isometry2<f32>) {
        self.data_mut().prepend_to_local_transformation(t)
    }

    #[inline]
    pub fn set_local_transformation(&mut self, t: Isometry2<f32>) {
        self.data_mut().set_local_transformation(t)
    }

    #[inline]
    pub fn append_translation(&mut self, t: &Translation2<f32>) {
        self.data_mut().append_translation(t)
    }

    #[inline]
    pub fn prepend_to_local_translation(&mut self, t: &Translation2<f32>) {
        self.data_mut().prepend_to_local_translation(t)
    }

    #[inline]
    pub fn set_local_translation(&mut self, t: Translation2<f32>) {
        self.data_mut().set_local_translation(t)
    }

    #[inline]
    pub fn append_rotation(&mut self, r: &UnitComplex<f32>) {
        self.data_mut().append_rotation(r)
    }

    #[inline]
    pub fn append_rotation_wrt_center(&mut self, r: &UnitComplex<f32>) {
        (*self.data_mut()).append_rotation_wrt_center(r)
    }

    #[inline]
    pub fn prepend_to_local_rotation(&mut self, r: &UnitComplex<f32>) {
        self.data_mut().prepend_to_local_rotation(r)
    }

    #[inline]
    pub fn set_local_rotation(&mut self, r: UnitComplex<f32>) {
        self.data_mut().set_local_rotation(r)
    }
}
