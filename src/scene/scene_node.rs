use crate::camera::camera::Camera;
use crate::context::context::Texture;
use crate::light::Light;
use crate::resource::material::Material;
use crate::resource::material_manager::MaterialManager;
use crate::resource::mesh::Mesh;
use crate::resource::mesh_manager::MeshManager;
use crate::resource::texture_manager::TextureManager;
use crate::resource::vertex_index::VertexIndex;
use crate::scene::object::Object;
use nalgebra::{self as na, Isometry3, Point2, Point3, Translation3, UnitQuaternion, Vector3};
use ncollide3d::procedural;
use ncollide3d::procedural::TriMesh;
use std::cell::{Ref, RefCell, RefMut};
use std::mem;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::rc::Weak;

pub struct SceneNodeData {
    local_scale: Vector3<f32>,
    local_transform: Isometry3<f32>,
    world_scale: Vector3<f32>,
    world_transform: Isometry3<f32>,
    visible: bool,
    up_to_date: bool,
    children: Vec<SceneNode>,
    object: Option<Object>,
    parent: Option<Weak<RefCell<SceneNodeData>>>,
}

#[derive(Clone)]
pub struct SceneNode {
    data: Rc<RefCell<SceneNodeData>>,
}

impl SceneNodeData {
    fn set_parent(&mut self, parent: Weak<RefCell<SceneNodeData>>) {
        self.parent = Some(parent);
    }

    fn remove_from_parent(&mut self, to_remove: &SceneNode) {
        let _ = self.parent.as_ref().map(|p| {
            if let Some(bp) = p.upgrade() {
                bp.borrow_mut().remove(to_remove);
            }
        });
    }

    fn remove(&mut self, o: &SceneNode) {
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

    pub fn render(&mut self, pass: usize, camera: &mut dyn Camera, light: &Light) {
        if self.visible {
            self.do_render(&na::one(), &Vector3::from_element(1.0), pass, camera, light)
        }
    }

    fn do_render(
        &mut self,
        transform: &Isometry3<f32>,
        scale: &Vector3<f32>,
        pass: usize,
        camera: &mut dyn Camera,
        light: &Light,
    ) {
        if !self.up_to_date {
            self.up_to_date = true;
            self.world_transform = *transform * self.local_transform;
            self.world_scale = scale.component_mul(&self.local_scale);
        }

        if let Some(ref o) = self.object {
            o.render(
                &self.world_transform,
                &self.world_scale,
                pass,
                camera,
                light,
            )
        }

        for c in self.children.iter_mut() {
            let mut bc = c.data_mut();
            if bc.visible {
                bc.do_render(
                    &self.world_transform,
                    &self.world_scale,
                    pass,
                    camera,
                    light,
                )
            }
        }
    }

    #[inline]
    pub fn object(&self) -> Option<&Object> {
        self.object.as_ref()
    }

    #[inline]
    pub fn object_mut(&mut self) -> Option<&mut Object> {
        self.object.as_mut()
    }

    #[inline]
    pub fn get_object(&self) -> &Object {
        self.object()
            .expect("This scene node does not contain an Object.")
    }

    #[inline]
    pub fn get_object_mut(&mut self) -> &mut Object {
        self.object_mut()
            .expect("This scene node does not contain an Object.")
    }

    #[inline]
    pub fn set_material(&mut self, material: Rc<RefCell<Box<dyn Material + 'static>>>) {
        self.apply_to_objects_mut(&mut |o| o.set_material(material.clone()))
    }

    #[inline]
    pub fn set_material_with_name(&mut self, name: &str) {
        let material = MaterialManager::get_global_manager(|tm| {
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
    pub fn modify_vertices<F: FnMut(&mut Vec<Point3<f32>>)>(&mut self, f: &mut F) {
        self.apply_to_objects_mut(&mut |o| o.modify_vertices(f))
    }

    #[inline(always)]
    pub fn read_vertices<F: FnMut(&[Point3<f32>])>(&self, f: &mut F) {
        self.apply_to_objects(&mut |o| o.read_vertices(f))
    }

    #[inline]
    pub fn recompute_normals(&mut self) {
        self.apply_to_objects_mut(&mut |o| o.recompute_normals())
    }

    #[inline(always)]
    pub fn modify_normals<F: FnMut(&mut Vec<Vector3<f32>>)>(&mut self, f: &mut F) {
        self.apply_to_objects_mut(&mut |o| o.modify_normals(f))
    }

    #[inline(always)]
    pub fn read_normals<F: FnMut(&[Vector3<f32>])>(&self, f: &mut F) {
        self.apply_to_objects(&mut |o| o.read_normals(f))
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
    pub fn apply_to_objects_mut<F: FnMut(&mut Object)>(&mut self, f: &mut F) {
        if let Some(ref mut o) = self.object {
            f(o)
        }

        for c in self.children.iter_mut() {
            c.data_mut().apply_to_objects_mut(f)
        }
    }

    #[inline]
    pub fn apply_to_objects<F: FnMut(&Object)>(&self, f: &mut F) {
        if let Some(ref o) = self.object {
            f(o)
        }

        for c in self.children.iter() {
            c.data().apply_to_objects(f)
        }
    }

    #[inline]
    pub fn set_local_scale(&mut self, sx: f32, sy: f32, sz: f32) {
        self.invalidate();
        self.local_scale = Vector3::new(sx, sy, sz)
    }

    #[inline]
    pub fn local_scale(&self) -> Vector3<f32> {
        self.local_scale
    }

    #[inline]
    pub fn reorient(&mut self, eye: &Point3<f32>, at: &Point3<f32>, up: &Vector3<f32>) {
        self.invalidate();
        self.local_transform = Isometry3::face_towards(eye, at, up)
    }

    #[inline]
    pub fn local_transformation(&self) -> Isometry3<f32> {
        self.local_transform
    }

    #[inline]
    pub fn inverse_local_transformation(&self) -> Isometry3<f32> {
        self.local_transform.inverse()
    }

    #[inline]
    #[allow(mutable_transmutes)]
    pub fn world_transformation(&self) -> Isometry3<f32> {
        unsafe {
            let mself: &mut SceneNodeData = mem::transmute(self);
            mself.update();
        }
        self.world_transform
    }

    #[inline]
    #[allow(mutable_transmutes)]
    pub fn inverse_world_transformation(&self) -> Isometry3<f32> {
        unsafe {
            let mself: &mut SceneNodeData = mem::transmute(self);
            mself.update();
        }
        self.local_transform.inverse()
    }

    #[inline]
    pub fn append_transformation(&mut self, t: &Isometry3<f32>) {
        self.invalidate();
        self.local_transform = t * self.local_transform
    }

    #[inline]
    pub fn prepend_to_local_transformation(&mut self, t: &Isometry3<f32>) {
        self.invalidate();
        self.local_transform *= t;
    }

    #[inline]
    pub fn set_local_transformation(&mut self, t: Isometry3<f32>) {
        self.invalidate();
        self.local_transform = t
    }

    #[inline]
    pub fn local_translation(&self) -> Translation3<f32> {
        self.local_transform.translation
    }

    #[inline]
    pub fn inverse_local_translation(&self) -> Translation3<f32> {
        self.local_transform.translation.inverse()
    }

    #[inline]
    pub fn append_translation(&mut self, t: &Translation3<f32>) {
        self.invalidate();
        self.local_transform = t * self.local_transform
    }

    #[inline]
    pub fn prepend_to_local_translation(&mut self, t: &Translation3<f32>) {
        self.invalidate();
        self.local_transform *= t
    }

    #[inline]
    pub fn set_local_translation(&mut self, t: Translation3<f32>) {
        self.invalidate();
        self.local_transform.translation = t
    }

    #[inline]
    pub fn local_rotation(&self) -> UnitQuaternion<f32> {
        self.local_transform.rotation
    }

    #[inline]
    pub fn inverse_local_rotation(&self) -> UnitQuaternion<f32> {
        self.local_transform.rotation.inverse()
    }

    #[inline]
    pub fn append_rotation(&mut self, r: &UnitQuaternion<f32>) {
        self.invalidate();
        self.local_transform = r * self.local_transform
    }

    #[inline]
    pub fn append_rotation_wrt_center(&mut self, r: &UnitQuaternion<f32>) {
        self.invalidate();
        self.local_transform.append_rotation_wrt_center_mut(r)
    }

    #[inline]
    pub fn prepend_to_local_rotation(&mut self, r: &UnitQuaternion<f32>) {
        self.invalidate();
        self.local_transform *= r
    }

    #[inline]
    pub fn set_local_rotation(&mut self, r: UnitQuaternion<f32>) {
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
                Some(ref mut p) => {
                    if let Some(dp) = p.upgrade() {
                        let mut dp = dp.borrow_mut();
                        dp.update();
                        self.world_transform = self.local_transform * dp.world_transform;
                        self.world_scale = self.local_scale.component_mul(&dp.local_scale);
                        self.up_to_date = true;
                        return;
                    }
                }
                None => {}
            }

            self.world_transform = self.local_transform;
            self.world_scale = self.local_scale;
            self.up_to_date = true;
        }
    }
}

impl Default for SceneNode {
    fn default() -> SceneNode {
        SceneNode::new_empty()
    }
}

impl SceneNode {
    pub fn new(
        local_scale: Vector3<f32>,
        local_transform: Isometry3<f32>,
        object: Option<Object>,
    ) -> SceneNode {
        let data = SceneNodeData {
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

        SceneNode {
            data: Rc::new(RefCell::new(data)),
        }
    }

    pub fn new_empty() -> SceneNode {
        SceneNode::new(Vector3::from_element(1.0), na::one(), None)
    }

    pub fn unlink(&mut self) {
        let self_self = self.clone();
        self.data_mut().remove_from_parent(&self_self);
        self.data_mut().parent = None
    }

    pub fn data(&self) -> Ref<SceneNodeData> {
        self.data.borrow()
    }

    pub fn data_mut(&mut self) -> RefMut<SceneNodeData> {
        self.data.borrow_mut()
    }

    pub fn add_group(&mut self) -> SceneNode {
        let node = SceneNode::new_empty();

        self.add_child(node.clone());

        node
    }

    pub fn add_child(&mut self, node: SceneNode) {
        assert!(
            node.data().is_root(),
            "The added node must not have a parent yet."
        );

        let mut node = node;
        let selfweakpointer = Rc::downgrade(&self.data);
        node.data_mut().set_parent(selfweakpointer);
        self.data_mut().children.push(node)
    }

    pub fn add_object(
        &mut self,
        local_scale: Vector3<f32>,
        local_transform: Isometry3<f32>,
        object: Object,
    ) -> SceneNode {
        let node = SceneNode::new(local_scale, local_transform, Some(object));

        self.add_child(node.clone());

        node
    }

    pub fn add_cube(&mut self, wx: f32, wy: f32, wz: f32) -> SceneNode {
        let res = self.add_geom_with_name("cube", Vector3::new(wx, wy, wz));

        res.expect("Unable to load the default cube geometry.")
    }

    pub fn add_sphere(&mut self, r: f32) -> SceneNode {
        let res = self.add_geom_with_name("sphere", Vector3::new(r * 2.0, r * 2.0, r * 2.0));

        res.expect("Unable to load the default sphere geometry.")
    }

    pub fn add_cone(&mut self, r: f32, h: f32) -> SceneNode {
        let res = self.add_geom_with_name("cone", Vector3::new(r * 2.0, h, r * 2.0));

        res.expect("Unable to load the default cone geometry.")
    }

    pub fn add_cylinder(&mut self, r: f32, h: f32) -> SceneNode {
        let res = self.add_geom_with_name("cylinder", Vector3::new(r * 2.0, h, r * 2.0));

        res.expect("Unable to load the default cylinder geometry.")
    }

    pub fn add_capsule(&mut self, r: f32, h: f32) -> SceneNode {
        self.add_trimesh(
            procedural::capsule(&(r * 2.0), &h, 50, 50),
            Vector3::from_element(1.0),
        )
    }

    pub fn add_quad(&mut self, w: f32, h: f32, usubdivs: usize, vsubdivs: usize) -> SceneNode {
        let mut node = self.add_trimesh(
            procedural::quad(w, h, usubdivs, vsubdivs),
            Vector3::from_element(1.0),
        );
        node.enable_backface_culling(false);

        node
    }

    pub fn add_quad_with_vertices(
        &mut self,
        vertices: &[Point3<f32>],
        nhpoints: usize,
        nvpoints: usize,
    ) -> SceneNode {
        let geom = procedural::quad_with_vertices(vertices, nhpoints, nvpoints);

        let mut node = self.add_trimesh(geom, Vector3::from_element(1.0));
        node.enable_backface_culling(false);

        node
    }

    pub fn add_geom_with_name(
        &mut self,
        geometry_name: &str,
        scale: Vector3<f32>,
    ) -> Option<SceneNode> {
        MeshManager::get_global_manager(|mm| mm.get(geometry_name)).map(|g| self.add_mesh(g, scale))
    }

    pub fn add_mesh(&mut self, mesh: Rc<RefCell<Mesh>>, scale: Vector3<f32>) -> SceneNode {
        let tex = TextureManager::get_global_manager(|tm| tm.get_default());
        let mat = MaterialManager::get_global_manager(|mm| mm.get_default());
        let object = Object::new(mesh, 1.0, 1.0, 1.0, tex, mat);

        self.add_object(scale, na::one(), object)
    }

    pub fn add_trimesh(&mut self, descr: TriMesh<f32>, scale: Vector3<f32>) -> SceneNode {
        self.add_mesh(
            Rc::new(RefCell::new(Mesh::from_trimesh(descr, false))),
            scale,
        )
    }

    pub fn add_obj(&mut self, path: &Path, mtl_dir: &Path, scale: Vector3<f32>) -> SceneNode {
        let tex = TextureManager::get_global_manager(|tm: &mut TextureManager| tm.get_default());
        let mat = MaterialManager::get_global_manager(|mm| mm.get_default());

        let result = MeshManager::load_obj(path, mtl_dir, path.to_str().unwrap()).map(|objs| {
            let mut root;

            let self_root = objs.len() == 1;
            let child_scale;

            if self_root {
                root = self.clone();
                child_scale = scale;
            } else {
                root = SceneNode::new(scale, na::one(), None);
                self.add_child(root.clone());
                child_scale = Vector3::from_element(1.0);
            }

            for (_, mesh, mtl) in objs.into_iter() {
                let mut object = Object::new(mesh, 1.0, 1.0, 1.0, tex.clone(), mat.clone());

                match mtl {
                    None => {}
                    Some(mtl) => {
                        object.set_color(mtl.diffuse.x, mtl.diffuse.y, mtl.diffuse.z);

                        for t in mtl.diffuse_texture.iter() {
                            let mut tpath = PathBuf::new();
                            tpath.push(mtl_dir);
                            tpath.push(&t[..]);
                            object.set_texture_from_file(&tpath, tpath.to_str().unwrap())
                        }

                        for t in mtl.ambiant_texture.iter() {
                            let mut tpath = PathBuf::new();
                            tpath.push(mtl_dir);
                            tpath.push(&t[..]);
                            object.set_texture_from_file(&tpath, tpath.to_str().unwrap())
                        }
                    }
                }

                let _ = root.add_object(child_scale, na::one(), object);
            }

            if self_root {
                root.data()
                    .children
                    .last()
                    .expect("There was nothing on this obj file.")
                    .clone()
            } else {
                root
            }
        });

        result.unwrap()
    }

    #[inline]
    pub fn apply_to_scene_nodes_mut<F: FnMut(&mut SceneNode)>(&mut self, f: &mut F) {
        f(self);

        for c in self.data_mut().children.iter_mut() {
            c.apply_to_scene_nodes_mut(f)
        }
    }

    #[inline]
    pub fn apply_to_scene_nodes<F: FnMut(&SceneNode)>(&self, f: &mut F) {
        f(self);

        for c in self.data().children.iter() {
            c.apply_to_scene_nodes(f)
        }
    }

    pub fn render(&mut self, pass: usize, camera: &mut dyn Camera, light: &Light) {
        self.data_mut().render(pass, camera, light)
    }

    #[inline]
    pub fn set_material(&mut self, material: Rc<RefCell<Box<dyn Material + 'static>>>) {
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
    pub fn modify_vertices<F: FnMut(&mut Vec<Point3<f32>>)>(&mut self, f: &mut F) {
        self.data_mut().modify_vertices(f)
    }

    #[inline(always)]
    pub fn read_vertices<F: FnMut(&[Point3<f32>])>(&self, f: &mut F) {
        self.data().read_vertices(f)
    }

    #[inline]
    pub fn recompute_normals(&mut self) {
        self.data_mut().recompute_normals()
    }

    #[inline(always)]
    pub fn modify_normals<F: FnMut(&mut Vec<Vector3<f32>>)>(&mut self, f: &mut F) {
        self.data_mut().modify_normals(f)
    }

    #[inline(always)]
    pub fn read_normals<F: FnMut(&[Vector3<f32>])>(&self, f: &mut F) {
        self.data().read_normals(f)
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
    pub fn set_local_scale(&mut self, sx: f32, sy: f32, sz: f32) {
        self.data_mut().set_local_scale(sx, sy, sz)
    }

    #[inline]
    pub fn reorient(&mut self, eye: &Point3<f32>, at: &Point3<f32>, up: &Vector3<f32>) {
        self.data_mut().reorient(eye, at, up)
    }

    #[inline]
    pub fn append_transformation(&mut self, t: &Isometry3<f32>) {
        self.data_mut().append_transformation(t)
    }

    #[inline]
    pub fn prepend_to_local_transformation(&mut self, t: &Isometry3<f32>) {
        self.data_mut().prepend_to_local_transformation(t)
    }

    #[inline]
    pub fn set_local_transformation(&mut self, t: Isometry3<f32>) {
        self.data_mut().set_local_transformation(t)
    }

    #[inline]
    pub fn append_translation(&mut self, t: &Translation3<f32>) {
        self.data_mut().append_translation(t)
    }

    #[inline]
    pub fn prepend_to_local_translation(&mut self, t: &Translation3<f32>) {
        self.data_mut().prepend_to_local_translation(t)
    }

    #[inline]
    pub fn set_local_translation(&mut self, t: Translation3<f32>) {
        self.data_mut().set_local_translation(t)
    }

    #[inline]
    pub fn append_rotation(&mut self, r: &UnitQuaternion<f32>) {
        self.data_mut().append_rotation(r)
    }

    #[inline]
    pub fn append_rotation_wrt_center(&mut self, r: &UnitQuaternion<f32>) {
        (*self.data_mut()).append_rotation_wrt_center(r)
    }

    #[inline]
    pub fn prepend_to_local_rotation(&mut self, r: &UnitQuaternion<f32>) {
        self.data_mut().prepend_to_local_rotation(r)
    }

    #[inline]
    pub fn set_local_rotation(&mut self, r: UnitQuaternion<f32>) {
        self.data_mut().set_local_rotation(r)
    }
}
