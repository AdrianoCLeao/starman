use std::sync::{Arc, RwLock};

use crate::resource::gpu_vector::GPUVec;
use crate::resource::vertex_index::VertexIndex;
use nalgebra::{self, Point2, Point3, Vector3};


pub struct Mesh {
    coords: Arc<RwLock<GPUVec<Point3<f32>>>>,
    faces: Arc<RwLock<GPUVec<Point3<VertexIndex>>>>,
    normals: Arc<RwLock<GPUVec<Vector3<f32>>>>,
    uvs: Arc<RwLock<GPUVec<Point2<f32>>>>,
    edges: Option<Arc<RwLock<GPUVec<Point2<VertexIndex>>>>>,
}

