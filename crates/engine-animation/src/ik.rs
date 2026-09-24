//! Inverse kinematics on model-space poses: analytic two-bone IK with a
//! pole target, and look-at with an angle limit.

use engine_math::{Affine3A, Quat, Vec3};

use crate::skeleton::{Pose, Skeleton};

fn rotation_of(affine: &Affine3A) -> Quat {
    affine.to_scale_rotation_translation().1.normalize()
}

fn position_of(affine: &Affine3A) -> Vec3 {
    Vec3::from(affine.translation)
}

fn parent_rotation(skeleton: &Skeleton, model: &[Affine3A], bone: usize) -> Quat {
    match skeleton.bones[bone].parent {
        Some(parent) => rotation_of(&model[parent]),
        None => rotation_of(&Affine3A::from_mat4(skeleton.root_transform)),
    }
}

/// Applies a model-space rotation `delta` to `bone` (blended by `weight`)
/// by rewriting its local rotation.
fn rotate_bone(
    skeleton: &Skeleton,
    pose: &mut Pose,
    model: &[Affine3A],
    bone: usize,
    delta: Quat,
    weight: f32,
) {
    let current = rotation_of(&model[bone]);
    let target = (delta * current).normalize();
    let parent = parent_rotation(skeleton, model, bone);
    let local = (parent.inverse() * target).normalize();
    let old = pose.locals[bone].rotation;
    pose.locals[bone].rotation = old.slerp(local, weight.clamp(0.0, 1.0)).normalize();
}

/// Two-bone IK: rotates `upper` and `middle` so `end` reaches `target`
/// (model space), bending towards `pole`. Recomputes `model`.
#[allow(clippy::too_many_arguments)]
pub fn solve_two_bone(
    skeleton: &Skeleton,
    pose: &mut Pose,
    model: &mut Vec<Affine3A>,
    upper: usize,
    middle: usize,
    end: usize,
    target: Vec3,
    pole: Vec3,
    weight: f32,
) {
    if weight <= 0.0 {
        return;
    }
    let a = position_of(&model[upper]);
    let b = position_of(&model[middle]);
    let c = position_of(&model[end]);
    let l1 = (b - a).length();
    let l2 = (c - b).length();
    if l1 < 1e-5 || l2 < 1e-5 {
        return;
    }
    let to_target = target - a;
    let distance = to_target
        .length()
        .clamp((l1 - l2).abs() + 1e-4, l1 + l2 - 1e-4);
    let direction = to_target.normalize_or_zero();
    if direction == Vec3::ZERO {
        return;
    }
    // Bend direction: towards the pole, perpendicular to the reach axis.
    let mut bend = pole - a;
    bend -= direction * bend.dot(direction);
    if bend.length_squared() < 1e-8 {
        bend = (b - a) - direction * (b - a).dot(direction);
    }
    let bend = bend.normalize_or_zero();
    let cos_a =
        ((l1 * l1 + distance * distance - l2 * l2) / (2.0 * l1 * distance)).clamp(-1.0, 1.0);
    let sin_a = (1.0 - cos_a * cos_a).max(0.0).sqrt();
    let desired_mid = a + (direction * cos_a + bend * sin_a) * l1;

    let upper_delta = Quat::from_rotation_arc((b - a).normalize(), (desired_mid - a).normalize());
    rotate_bone(skeleton, pose, model, upper, upper_delta, weight);
    pose.model_space(skeleton, model);

    let b = position_of(&model[middle]);
    let c = position_of(&model[end]);
    let reach = a + direction * distance;
    let middle_delta = Quat::from_rotation_arc(
        (c - b).normalize(),
        (reach - b).normalize_or_zero().normalize(),
    );
    rotate_bone(skeleton, pose, model, middle, middle_delta, weight);
    pose.model_space(skeleton, model);
}

/// Look-at: rotates `bone` so its local `forward` axis points at `target`
/// (model space), limited to `max_angle` radians from the animated pose.
#[allow(clippy::too_many_arguments)]
pub fn solve_look_at(
    skeleton: &Skeleton,
    pose: &mut Pose,
    model: &mut Vec<Affine3A>,
    bone: usize,
    forward: Vec3,
    target: Vec3,
    max_angle: f32,
    weight: f32,
) {
    if weight <= 0.0 {
        return;
    }
    let position = position_of(&model[bone]);
    let current = (rotation_of(&model[bone]) * forward).normalize_or_zero();
    let desired = (target - position).normalize_or_zero();
    if current == Vec3::ZERO || desired == Vec3::ZERO {
        return;
    }
    let mut delta = Quat::from_rotation_arc(current, desired);
    let (axis, angle) = delta.to_axis_angle();
    if angle > max_angle && max_angle > 0.0 {
        delta = Quat::from_axis_angle(axis, max_angle);
    }
    rotate_bone(skeleton, pose, model, bone, delta, weight);
    pose.model_space(skeleton, model);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton::{Bone, BoneTransform};
    use engine_math::Mat4;

    fn arm() -> Skeleton {
        let bone = |name: &str, parent, x: f32| Bone {
            name: name.into(),
            parent,
            rest: BoneTransform {
                translation: Vec3::new(x, 0.0, 0.0),
                ..BoneTransform::IDENTITY
            },
        };
        Skeleton::new(
            "arm",
            vec![
                bone("shoulder", None, 0.0),
                bone("elbow", Some(0), 1.0),
                bone("hand", Some(1), 1.0),
            ],
            Mat4::IDENTITY,
            vec![],
            vec![],
        )
    }

    #[test]
    fn two_bone_reaches_reachable_targets_and_bends_to_the_pole() {
        let skeleton = arm();
        let mut pose = skeleton.rest_pose();
        let mut model = Vec::new();
        pose.model_space(&skeleton, &mut model);
        let target = Vec3::new(1.0, 1.0, 0.0);
        solve_two_bone(
            &skeleton,
            &mut pose,
            &mut model,
            0,
            1,
            2,
            target,
            Vec3::new(0.0, 0.0, 5.0),
            1.0,
        );
        let hand = Vec3::from(model[2].translation);
        assert!(hand.distance(target) < 1e-3, "{hand}");
        let elbow = Vec3::from(model[1].translation);
        assert!(elbow.z > 0.1, "bends towards the pole: {elbow}");
        assert!(
            (elbow.length() - 1.0).abs() < 1e-3,
            "bone lengths preserved"
        );

        // Unreachable: stretches straight towards it.
        solve_two_bone(
            &skeleton,
            &mut pose,
            &mut model,
            0,
            1,
            2,
            Vec3::new(0.0, 10.0, 0.0),
            Vec3::Z,
            1.0,
        );
        let hand = Vec3::from(model[2].translation);
        assert!(hand.normalize().dot(Vec3::Y) > 0.99, "{hand}");
    }

    #[test]
    fn look_at_respects_the_angle_limit() {
        let skeleton = arm();
        let mut pose = skeleton.rest_pose();
        let mut model = Vec::new();
        pose.model_space(&skeleton, &mut model);
        solve_look_at(
            &skeleton,
            &mut pose,
            &mut model,
            2,
            Vec3::X,
            Vec3::new(2.0, 5.0, 0.0),
            0.5,
            1.0,
        );
        let forward = model[2].to_scale_rotation_translation().1 * Vec3::X;
        assert!((forward.angle_between(Vec3::X) - 0.5).abs() < 1e-3);
    }
}
