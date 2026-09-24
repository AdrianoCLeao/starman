//! Conversions between engine transforms (local, hierarchical) and Rapier
//! isometries (world space).

use bevy_ecs::prelude::{Entity, Query};
use engine_core::{Parent, Transform};
use engine_math::{Affine3A, Quat, Vec3};
use rapier3d::na::{Isometry3, Quaternion, Translation3, UnitQuaternion};
use rapier3d::prelude::{Real, Vector};

pub type Pose = (Vec3, Quat);

/// Upper bound on hierarchy depth walked (cycle guard).
const MAX_DEPTH: usize = 256;

pub fn to_isometry((translation, rotation): Pose) -> Isometry3<Real> {
    Isometry3::from_parts(
        Translation3::new(translation.x, translation.y, translation.z),
        quat_to_na(rotation),
    )
}

pub fn from_isometry(iso: &Isometry3<Real>) -> Pose {
    let t = iso.translation.vector;
    (Vec3::new(t.x, t.y, t.z), na_to_quat(&iso.rotation))
}

pub fn quat_to_na(quat: Quat) -> UnitQuaternion<Real> {
    UnitQuaternion::from_quaternion(Quaternion::new(quat.w, quat.x, quat.y, quat.z))
}

pub fn na_to_quat(rotation: &UnitQuaternion<Real>) -> Quat {
    let q = rotation.quaternion();
    Quat::from_xyzw(q.i, q.j, q.k, q.w)
}

pub fn to_vector(v: Vec3) -> Vector<Real> {
    Vector::new(v.x, v.y, v.z)
}

pub fn from_vector(v: &Vector<Real>) -> Vec3 {
    Vec3::new(v.x, v.y, v.z)
}

/// World affine of `entity`, composing local transforms up the hierarchy
/// (independent of `GlobalTransform`, which is only propagated before
/// rendering).
pub fn world_affine(entity: Entity, transforms: &Query<(&Transform, Option<&Parent>)>) -> Affine3A {
    let mut affine = Affine3A::IDENTITY;
    let mut current = Some(entity);
    for _ in 0..MAX_DEPTH {
        let Some(node) = current else {
            break;
        };
        let Ok((transform, parent)) = transforms.get(node) else {
            break;
        };
        affine = transform.to_affine() * affine;
        current = parent.map(|parent| parent.0);
    }
    affine
}

/// World affine of `entity`'s parent (identity for roots).
pub fn parent_affine(
    entity: Entity,
    transforms: &Query<(&Transform, Option<&Parent>)>,
) -> Affine3A {
    match transforms.get(entity) {
        Ok((_, Some(parent))) => world_affine(parent.0, transforms),
        _ => Affine3A::IDENTITY,
    }
}

/// Position and rotation of an affine (scale discarded).
pub fn pose_of(affine: &Affine3A) -> Pose {
    let (_, rotation, translation) = affine.to_scale_rotation_translation();
    (translation, rotation.normalize())
}

/// World pose of `entity` (scale discarded).
pub fn world_pose(entity: Entity, transforms: &Query<(&Transform, Option<&Parent>)>) -> Pose {
    pose_of(&world_affine(entity, transforms))
}

/// Converts a world pose to a local pose under `parent_world`.
pub fn world_to_local(parent_world: &Affine3A, world: Pose) -> Pose {
    if *parent_world == Affine3A::IDENTITY {
        return world;
    }
    let local = parent_world.inverse() * Affine3A::from_rotation_translation(world.1, world.0);
    pose_of(&local)
}
