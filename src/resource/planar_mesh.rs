//! Data structure of a scene node geometry.
use std::iter;
use std::sync::{Arc, RwLock};

use crate::resource::gpu_vector::{AllocationType, BufferType, GPUVec};
use crate::resource::vertex_index::VertexIndex;
use crate::resource::effect::ShaderAttribute;
use nalgebra::{Point2, Point3};

pub struct PlanarMesh {
    coords: Arc<RwLock<GPUVec<Point2<f32>>>>,
    faces: Arc<RwLock<GPUVec<Point3<VertexIndex>>>>,
    uvs: Arc<RwLock<GPUVec<Point2<f32>>>>,
    edges: Option<Arc<RwLock<GPUVec<Point2<VertexIndex>>>>>,
}

impl PlanarMesh {
    pub fn new(
        coords: Vec<Point2<f32>>,
        faces: Vec<Point3<VertexIndex>>,
        uvs: Option<Vec<Point2<f32>>>,
        dynamic_draw: bool,
    ) -> PlanarMesh {
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
        let us = Arc::new(RwLock::new(GPUVec::new(uvs, BufferType::Array, location)));

        PlanarMesh::new_with_gpu_vectors(cs, fs, us)
    }

    pub fn new_with_gpu_vectors(
        coords: Arc<RwLock<GPUVec<Point2<f32>>>>,
        faces: Arc<RwLock<GPUVec<Point3<VertexIndex>>>>,
        uvs: Arc<RwLock<GPUVec<Point2<f32>>>>,
    ) -> PlanarMesh {
        PlanarMesh {
            coords,
            faces,
            uvs,
            edges: None,
        }
    }

    pub fn bind_coords(&mut self, coords: &mut ShaderAttribute<Point2<f32>>) {
        coords.bind(&mut *self.coords.write().unwrap());
    }

    pub fn bind_uvs(&mut self, uvs: &mut ShaderAttribute<Point2<f32>>) {
        uvs.bind(&mut *self.uvs.write().unwrap());
    }

    pub fn bind_faces(&mut self) {
        self.faces.write().unwrap().bind();
    }

    pub fn bind(
        &mut self,
        coords: &mut ShaderAttribute<Point2<f32>>,
        uvs: &mut ShaderAttribute<Point2<f32>>,
    ) {
        self.bind_coords(coords);
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
        self.uvs.write().unwrap().unbind();
        self.faces.write().unwrap().unbind();
    }

    pub fn num_pts(&self) -> usize {
        self.faces.read().unwrap().len() * 3
    }

    pub fn faces(&self) -> &Arc<RwLock<GPUVec<Point3<VertexIndex>>>> {
        &self.faces
    }

    pub fn coords(&self) -> &Arc<RwLock<GPUVec<Point2<f32>>>> {
        &self.coords
    }

    pub fn uvs(&self) -> &Arc<RwLock<GPUVec<Point2<f32>>>> {
        &self.uvs
    }
}
