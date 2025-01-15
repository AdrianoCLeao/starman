use std::iter;
use std::sync::{Arc, RwLock};

use crate::resource::gpu_vector::{AllocationType, BufferType, GPUVec};
use crate::resource::vertex_index::VertexIndex;
//use crate::resource::ShaderAttribute;
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
    
    pub fn compute_normals_array(
        coordinates: &[Point3<f32>],
        faces: &[Point3<VertexIndex>],
    ) -> Vec<Vector3<f32>> {
        let mut res = Vec::new();

        Mesh::compute_normals(coordinates, faces, &mut res);

        res
    }

    /// Computes normals from a set of faces.
    pub fn compute_normals(
        coordinates: &[Point3<f32>],
        faces: &[Point3<VertexIndex>],
        normals: &mut Vec<Vector3<f32>>,
    ) {
        let mut divisor: Vec<f32> = iter::repeat(0f32).take(coordinates.len()).collect();

        normals.clear();
        normals.extend(iter::repeat(Vector3::<f32>::zero()).take(coordinates.len()));

        // Accumulate normals ...
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

        // ... and compute the mean
        for (n, divisor) in normals.iter_mut().zip(divisor.iter()) {
            *n /= *divisor
        }
    }
}

