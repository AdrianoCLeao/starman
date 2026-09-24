//! `ColliderShape3D` → Rapier shapes, including mesh-derived colliders.

use engine_assets::{Assets, LoadState, MeshData};
use rapier3d::prelude::{Point, Real, SharedShape};

use crate::components::ColliderShape3D;

const MIN_EXTENT: f32 = 0.001;

pub enum ShapeBuild {
    Ready(SharedShape),
    /// A mesh asset is still loading; retry next step.
    Pending,
    Failed(String),
}

pub fn build_shape(shape: &ColliderShape3D, assets: Option<&Assets>) -> ShapeBuild {
    let e = |v: f32| v.max(MIN_EXTENT);
    ShapeBuild::Ready(match shape {
        ColliderShape3D::Box { half_extents } => {
            SharedShape::cuboid(e(half_extents.x), e(half_extents.y), e(half_extents.z))
        }
        ColliderShape3D::Sphere { radius } => SharedShape::ball(e(*radius)),
        ColliderShape3D::Capsule {
            half_height,
            radius,
        } => SharedShape::capsule_y(e(*half_height), e(*radius)),
        ColliderShape3D::Cylinder {
            half_height,
            radius,
        } => SharedShape::cylinder(e(*half_height), e(*radius)),
        ColliderShape3D::Cone {
            half_height,
            radius,
        } => SharedShape::cone(e(*half_height), e(*radius)),
        ColliderShape3D::Trimesh => SharedShape::trimesh(
            vec![
                Point::new(-0.5, 0.0, -0.5),
                Point::new(0.5, 0.0, -0.5),
                Point::new(0.5, 0.0, 0.5),
                Point::new(-0.5, 0.0, 0.5),
            ],
            vec![[0, 1, 2], [0, 2, 3]],
        ),
        ColliderShape3D::Mesh { mesh, convex } => {
            let Some(assets) = assets else {
                return ShapeBuild::Failed("mesh colliders need the asset store".to_owned());
            };
            if mesh.is_empty() {
                return ShapeBuild::Failed("mesh collider has no mesh".to_owned());
            }
            let handle = assets.request::<MeshData>(mesh);
            match assets.state(handle) {
                Some(LoadState::Loaded) => {}
                Some(LoadState::Failed(reason)) => {
                    return ShapeBuild::Failed(format!(
                        "mesh '{}' failed to load: {reason}",
                        mesh.path
                    ))
                }
                _ => return ShapeBuild::Pending,
            }
            let Some(data) = assets.get(handle) else {
                return ShapeBuild::Pending;
            };
            return mesh_shape(&data, *convex);
        }
    })
}

/// Builds a triangle mesh or convex hull from mesh data.
pub fn mesh_shape(data: &MeshData, convex: bool) -> ShapeBuild {
    let points: Vec<Point<Real>> = data
        .vertices
        .iter()
        .map(|v| Point::new(v.position[0], v.position[1], v.position[2]))
        .collect();
    if points.len() < 3 {
        return ShapeBuild::Failed(format!("mesh '{}' has too few vertices", data.name));
    }
    if convex {
        return match SharedShape::convex_hull(&points) {
            Some(shape) => ShapeBuild::Ready(shape),
            None => {
                ShapeBuild::Failed(format!("mesh '{}' has a degenerate convex hull", data.name))
            }
        };
    }
    let count = points.len() as u32;
    let triangles: Vec<[u32; 3]> = data
        .indices
        .chunks_exact(3)
        .map(|t| [t[0], t[1], t[2]])
        .filter(|t| t.iter().all(|i| *i < count) && t[0] != t[1] && t[1] != t[2] && t[0] != t[2])
        .collect();
    if triangles.is_empty() {
        return ShapeBuild::Failed(format!("mesh '{}' has no valid triangles", data.name));
    }
    ShapeBuild::Ready(SharedShape::trimesh(points, triangles))
}
