//! Data structure of a scene node geometry.
use std::iter;
use std::sync::{Arc, RwLock};

use crate::resource::gpu_vector::{AllocationType, BufferType, GPUVec};
use crate::resource::vertex_index::VertexIndex;
use crate::resource::effect::ShaderAttribute;
use nalgebra::{self, Point2, Point3, Vector3};
use ncollide3d::procedural::{IndexBuffer, TriMesh};
use num_traits::Zero;

pub struct Mesh {
    coords: Arc<RwLock<GPUVec<Point3<f32>>>>,
    faces: Arc<RwLock<GPUVec<Point3<VertexIndex>>>>,
    normals: Arc<RwLock<GPUVec<Vector3<f32>>>>,
    uvs: Arc<RwLock<GPUVec<Point2<f32>>>>,
    edges: Option<Arc<RwLock<GPUVec<Point2<VertexIndex>>>>>,
}

impl Mesh {
    pub fn new(
        coords: Vec<Point3<f32>>,
        faces: Vec<Point3<VertexIndex>>,
        normals: Option<Vec<Vector3<f32>>>,
        uvs: Option<Vec<Point2<f32>>>,
        dynamic_draw: bool,
    ) -> Mesh {
        let normals = match normals {
            Some(ns) => ns,
            None => Mesh::compute_normals_array(&coords[..], &faces[..]),
        };

        let uvs = match uvs {
            Some(us) => us,
            None => iter::repeat(Point2::origin()).take(coords.len()).collect(),
        };

        let location = if dynamic_draw {
            AllocationType::DynamicDraw
        } else {
            AllocationType::StaticDraw
        };
        let cs = Arc::new(RwLock::new(GPUVec::new(
            coords,
            BufferType::Array,
            location,
        )));
        let fs = Arc::new(RwLock::new(GPUVec::new(
            faces,
            BufferType::ElementArray,
            location,
        )));
        let ns = Arc::new(RwLock::new(GPUVec::new(
            normals,
            BufferType::Array,
            location,
        )));
        let us = Arc::new(RwLock::new(GPUVec::new(uvs, BufferType::Array, location)));

        Mesh::new_with_gpu_vectors(cs, fs, ns, us)
    }

    pub fn from_trimesh(mesh: TriMesh<f32>, dynamic_draw: bool) -> Mesh {
        let mut mesh = mesh;

        mesh.unify_index_buffer();

        let TriMesh {
            coords,
            normals,
            uvs,
            indices,
        } = mesh;

        Mesh::new(
            coords,
            indices
                .unwrap_unified()
                .into_iter()
                .map(nalgebra::convert)
                .collect(),
            normals,
            uvs,
            dynamic_draw,
        )
    }

    pub fn to_trimesh(&self) -> Option<TriMesh<f32>> {
        if !self.coords.read().unwrap().is_on_ram()
            || !self.faces.read().unwrap().is_on_ram()
            || !self.normals.read().unwrap().is_on_ram()
            || !self.uvs.read().unwrap().is_on_ram()
        {
            return None;
        }

        let coords = self.coords.read().unwrap().to_owned();
        let faces = self.faces.read().unwrap().to_owned();
        let normals = self.normals.read().unwrap().to_owned();
        let uvs = self.uvs.read().unwrap().to_owned();

        Some(TriMesh::new(
            coords.unwrap(),
            normals,
            uvs,
            Some(IndexBuffer::Unified(
                faces
                    .unwrap()
                    .into_iter()
                    .map(|e| Point3::new(e.x as u32, e.y as u32, e.z as u32))
                    .collect(),
            )),
        ))

        /*
        let unload_coords = !self.coords.read().unwrap().is_on_ram();
        let unload_faces = !self.faces.read().unwrap().is_on_ram();
        let unload_normals = !self.normals.read().unwrap().is_on_ram();
        let unload_uvs = !self.uvs.read().unwrap().is_on_ram();

        self.coords.write().unwrap().load_to_ram();
        self.faces.write().unwrap().load_to_ram();
        self.normals.write().unwrap().load_to_ram();
        self.uvs.write().unwrap().load_to_ram();

        let coords = self.coords.read().unwrap().to_owned();
        let faces = self.faces.read().unwrap().to_owned();
        let normals = self.normals.read().unwrap().to_owned();
        let uvs = self.uvs.read().unwrap().to_owned();

        if unload_coords {
            self.coords.write().unwrap().unload_from_ram();
        }
        if unload_faces {
            self.coords.write().unwrap().unload_from_ram();
        }
        if unload_normals {
            self.coords.write().unwrap().unload_from_ram();
        }
        if unload_uvs {
            self.coords.write().unwrap().unload_from_ram();
        }

        if coords.is_none() || faces.is_none() {
            None
        } else {
            Some(TriMesh::new(
                coords.unwrap(),
                normals,
                uvs,
                Some(IndexBuffer::Unified(
                    faces
                        .unwrap()
                        .into_iter()
                        .map(|e| Point3::new(e.x as u32, e.y as u32, e.z as u32))
                        .collect(),
                )),
            ))
        }*/
    }

    pub fn new_with_gpu_vectors(
        coords: Arc<RwLock<GPUVec<Point3<f32>>>>,
        faces: Arc<RwLock<GPUVec<Point3<VertexIndex>>>>,
        normals: Arc<RwLock<GPUVec<Vector3<f32>>>>,
        uvs: Arc<RwLock<GPUVec<Point2<f32>>>>,
    ) -> Mesh {
        Mesh {
            coords,
            faces,
            normals,
            uvs,
            edges: None,
        }
    }

    pub fn bind_coords(&mut self, coords: &mut ShaderAttribute<Point3<f32>>) {
        coords.bind(&mut *self.coords.write().unwrap());
    }

    pub fn bind_normals(&mut self, normals: &mut ShaderAttribute<Vector3<f32>>) {
        normals.bind(&mut *self.normals.write().unwrap());
    }

    pub fn bind_uvs(&mut self, uvs: &mut ShaderAttribute<Point2<f32>>) {
        uvs.bind(&mut *self.uvs.write().unwrap());
    }

    pub fn bind_faces(&mut self) {
        self.faces.write().unwrap().bind();
    }

    pub fn bind(
        &mut self,
        coords: &mut ShaderAttribute<Point3<f32>>,
        normals: &mut ShaderAttribute<Vector3<f32>>,
        uvs: &mut ShaderAttribute<Point2<f32>>,
    ) {
        self.bind_coords(coords);
        self.bind_normals(normals);
        self.bind_uvs(uvs);
        self.bind_faces();
    }

    pub fn bind_edges(&mut self) {
        if self.edges.is_none() {
            let mut edges = Vec::new();
            for face in self.faces.read().unwrap().data().as_ref().unwrap() {
                edges.push(Point2::new(face.x, face.y));
                edges.push(Point2::new(face.y, face.z));
                edges.push(Point2::new(face.z, face.x));
            }
            let gpu_edges =
                GPUVec::new(edges, BufferType::ElementArray, AllocationType::StaticDraw);
            self.edges = Some(Arc::new(RwLock::new(gpu_edges)));
        }

        self.edges.as_mut().unwrap().write().unwrap().bind();
    }

    pub fn unbind(&self) {
        self.coords.write().unwrap().unbind();
        self.normals.write().unwrap().unbind();
        self.uvs.write().unwrap().unbind();
        self.faces.write().unwrap().unbind();
    }

    pub fn num_pts(&self) -> usize {
        self.faces.read().unwrap().len() * 3
    }

    pub fn recompute_normals(&mut self) {
        Mesh::compute_normals(
            &self.coords.read().unwrap().data().as_ref().unwrap()[..],
            &self.faces.read().unwrap().data().as_ref().unwrap()[..],
            self.normals.write().unwrap().data_mut().as_mut().unwrap(),
        );
    }

    pub fn faces(&self) -> &Arc<RwLock<GPUVec<Point3<VertexIndex>>>> {
        &self.faces
    }

    pub fn normals(&self) -> &Arc<RwLock<GPUVec<Vector3<f32>>>> {
        &self.normals
    }

    pub fn coords(&self) -> &Arc<RwLock<GPUVec<Point3<f32>>>> {
        &self.coords
    }

    pub fn uvs(&self) -> &Arc<RwLock<GPUVec<Point2<f32>>>> {
        &self.uvs
    }

    pub fn compute_normals_array(
        coordinates: &[Point3<f32>],
        faces: &[Point3<VertexIndex>],
    ) -> Vec<Vector3<f32>> {
        let mut res = Vec::new();

        Mesh::compute_normals(coordinates, faces, &mut res);

        res
    }

    pub fn compute_normals(
        coordinates: &[Point3<f32>],
        faces: &[Point3<VertexIndex>],
        normals: &mut Vec<Vector3<f32>>,
    ) {
        let mut divisor: Vec<f32> = iter::repeat(0f32).take(coordinates.len()).collect();

        normals.clear();
        normals.extend(iter::repeat(Vector3::<f32>::zero()).take(coordinates.len()));

        for f in faces.iter() {
            let edge1 = coordinates[f.y as usize] - coordinates[f.x as usize];
            let edge2 = coordinates[f.z as usize] - coordinates[f.x as usize];
            let cross = edge1.cross(&edge2);
            let normal;

            if !cross.is_zero() {
                normal = cross.normalize()
            } else {
                normal = cross
            }

            normals[f.x as usize] += normal;
            normals[f.y as usize] += normal;
            normals[f.z as usize] += normal;

            divisor[f.x as usize] += 1.0;
            divisor[f.y as usize] += 1.0;
            divisor[f.z as usize] += 1.0;
        }

        for (n, divisor) in normals.iter_mut().zip(divisor.iter()) {
            *n /= *divisor
        }
    }
}
