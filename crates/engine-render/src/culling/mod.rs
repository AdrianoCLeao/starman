//! Frustum culling and 3D batch/instance keys.

use engine_math::glam::Vec4;
use engine_math::{Mat4, Vec3};

use crate::extract::{Aabb, ExtractedMesh};

#[derive(Clone, Copy, Debug)]
pub struct Frustum {
    /// Planes as (nx, ny, nz, d) with Ax+By+Cz+D >= 0 inside.
    pub planes: [[f32; 4]; 6],
}

impl Frustum {
    pub fn from_view_proj(view_proj: Mat4) -> Self {
        let m = view_proj;
        let rows = [
            Vec4::new(m.x_axis.x, m.y_axis.x, m.z_axis.x, m.w_axis.x),
            Vec4::new(m.x_axis.y, m.y_axis.y, m.z_axis.y, m.w_axis.y),
            Vec4::new(m.x_axis.z, m.y_axis.z, m.z_axis.z, m.w_axis.z),
            Vec4::new(m.x_axis.w, m.y_axis.w, m.z_axis.w, m.w_axis.w),
        ];
        // Standard Gribb/Hartmann extraction.
        let raw = [
            rows[3] + rows[0], // left
            rows[3] - rows[0], // right
            rows[3] + rows[1], // bottom
            rows[3] - rows[1], // top
            rows[3] + rows[2], // near
            rows[3] - rows[2], // far
        ];
        let mut planes = [[0.0; 4]; 6];
        for (i, plane) in raw.iter().enumerate() {
            let len = Vec3::new(plane.x, plane.y, plane.z).length().max(1e-6);
            planes[i] = [plane.x / len, plane.y / len, plane.z / len, plane.w / len];
        }
        Self { planes }
    }

    pub fn intersects_aabb(&self, aabb: Aabb) -> bool {
        for plane in &self.planes {
            let n = Vec3::new(plane[0], plane[1], plane[2]);
            let positive = Vec3::new(
                if n.x >= 0.0 { aabb.max[0] } else { aabb.min[0] },
                if n.y >= 0.0 { aabb.max[1] } else { aabb.min[1] },
                if n.z >= 0.0 { aabb.max[2] } else { aabb.min[2] },
            );
            if n.dot(positive) + plane[3] < 0.0 {
                return false;
            }
        }
        true
    }
}

pub fn frustum_cull_meshes(view_proj: Mat4, meshes: &[ExtractedMesh]) -> Vec<usize> {
    let frustum = Frustum::from_view_proj(view_proj);
    meshes
        .iter()
        .enumerate()
        .filter(|(_, m)| frustum.intersects_aabb(m.aabb))
        .map(|(i, _)| i)
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BatchKey3d {
    pub mesh: u64,
    pub texture: u64,
    pub material: u64,
}

pub fn batch_key(mesh: &ExtractedMesh) -> BatchKey3d {
    BatchKey3d {
        mesh: mesh.mesh.id().value(),
        texture: mesh.texture.id().value(),
        material: mesh.material.id().value(),
    }
}

/// Group visible mesh indices into contiguous batches sharing mesh/texture/material.
pub fn build_batches_3d(
    meshes: &[ExtractedMesh],
    visible: &[usize],
) -> Vec<(BatchKey3d, Vec<usize>)> {
    let mut sorted: Vec<usize> = visible.to_vec();
    sorted.sort_by_key(|i| batch_key(&meshes[*i]));
    let mut batches = Vec::new();
    let mut current: Option<(BatchKey3d, Vec<usize>)> = None;
    for index in sorted {
        let key = batch_key(&meshes[index]);
        match current.as_mut() {
            Some((k, items)) if *k == key => items.push(index),
            _ => {
                if let Some(batch) = current.take() {
                    batches.push(batch);
                }
                current = Some((key, vec![index]));
            }
        }
    }
    if let Some(batch) = current {
        batches.push(batch);
    }
    batches
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aabb_inside_identity_frustum() {
        let vp = Mat4::orthographic_rh(-10.0, 10.0, -10.0, 10.0, 0.1, 100.0);
        let frustum = Frustum::from_view_proj(vp);
        let aabb = Aabb::from_center_extents(Vec3::ZERO, Vec3::splat(0.5));
        assert!(frustum.intersects_aabb(aabb));
        let far = Aabb::from_center_extents(Vec3::new(0.0, 0.0, -1000.0), Vec3::splat(0.5));
        assert!(!frustum.intersects_aabb(far));
    }
}
