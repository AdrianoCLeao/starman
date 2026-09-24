//! Pose evaluation: samples clip instances, blends them per layer
//! (override or additive, through bone masks) and extracts root motion.

use std::sync::Arc;

use engine_math::{Quat, Vec3, Vec4};

use crate::clip::{AnimationClip, ClipBinding, RootMotionSettings};
use crate::graph::LayerBlend;
use crate::skeleton::{BoneTransform, Pose, Skeleton};
use crate::state_machine::ClipInstance;

/// The clip instances of one layer for this frame.
#[derive(Clone)]
pub struct LayerOutput {
    pub blend: LayerBlend,
    pub weight: f32,
    /// Per-bone weight (1 = fully affected); `None` = every bone.
    pub mask: Option<Arc<Vec<f32>>>,
    pub instances: Vec<ClipInstance>,
    /// Whether root motion is extracted from this layer.
    pub root_motion: bool,
}

/// Root motion produced during one evaluation (model space).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RootMotionDelta {
    pub translation: Vec3,
    pub rotation: Quat,
}

impl Default for RootMotionDelta {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
        }
    }
}

/// Reusable buffers (one per animated entity avoids reallocations).
#[derive(Default, Clone, Debug)]
pub struct PoseScratch {
    sample: Vec<BoneTransform>,
    reference: Vec<BoneTransform>,
    translation: Vec<Vec3>,
    rotation: Vec<Vec4>,
    scale: Vec<Vec3>,
    animated: Vec<bool>,
}

/// Per-bone weights for a mask on `skeleton`.
pub fn mask_weights(skeleton: &Skeleton, bones: &[String], include_descendants: bool) -> Vec<f32> {
    let roots: Vec<usize> = bones
        .iter()
        .filter_map(|b| skeleton.bone_index(b))
        .collect();
    (0..skeleton.len())
        .map(|bone| {
            let hit = if include_descendants {
                roots.iter().any(|root| skeleton.is_descendant(bone, *root))
            } else {
                roots.contains(&bone)
            };
            if hit {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

/// Rotation about Y contained in `rotation` (swing-twist decomposition).
pub fn yaw_of(rotation: Quat) -> Quat {
    let twist = Quat::from_xyzw(0.0, rotation.y, 0.0, rotation.w);
    if twist.length_squared() < 1e-10 {
        Quat::IDENTITY
    } else {
        twist.normalize()
    }
}

/// Model-space rotation of `bone`'s parent in the rest pose.
fn parent_rest_rotation(skeleton: &Skeleton, bone: usize) -> Quat {
    let mut rotation = Quat::IDENTITY;
    let mut current = skeleton.bones.get(bone).and_then(|b| b.parent);
    while let Some(index) = current {
        rotation = skeleton.bones[index].rest.rotation * rotation;
        current = skeleton.bones[index].parent;
    }
    let (_, root, _) = skeleton.root_transform.to_scale_rotation_translation();
    root * rotation
}

struct RootSample {
    translation: Vec3,
    yaw: Quat,
}

fn root_sample(
    clip: &AnimationClip,
    binding: &ClipBinding,
    bone: usize,
    time: f32,
    rest: BoneTransform,
) -> RootSample {
    let local = clip.sample_bone(binding, bone, time, rest);
    RootSample {
        translation: local.translation,
        yaw: yaw_of(local.rotation),
    }
}

/// Motion of the root bone between the instance's previous and current
/// time (parent space), accounting for one loop wrap.
fn root_delta(
    instance: &ClipInstance,
    settings: &RootMotionSettings,
    bone: usize,
    rest: BoneTransform,
) -> (Vec3, Quat) {
    let clip = &instance.clip.clip;
    let binding = &instance.clip.binding;
    let at = |time: f32| root_sample(clip, binding, bone, time, rest);
    let (from, to) = (instance.previous_time, instance.time);
    let (translation, yaw) = if clip.looping && to < from {
        let (a, end, start, b) = (at(from), at(clip.length()), at(0.0), at(to));
        (
            (end.translation - a.translation) + (b.translation - start.translation),
            (b.yaw * start.yaw.inverse()) * (end.yaw * a.yaw.inverse()),
        )
    } else {
        let (a, b) = (at(from), at(to));
        (b.translation - a.translation, b.yaw * a.yaw.inverse())
    };
    let mut translation = translation;
    if !settings.horizontal {
        translation.x = 0.0;
        translation.z = 0.0;
    }
    if !settings.vertical {
        translation.y = 0.0;
    }
    let yaw = if settings.yaw { yaw } else { Quat::IDENTITY };
    (translation, yaw)
}

/// Removes the extracted root motion from a sampled root bone so the
/// character animates in place.
fn make_in_place(
    local: &mut BoneTransform,
    settings: &RootMotionSettings,
    clip: &AnimationClip,
    binding: &ClipBinding,
    bone: usize,
    rest: BoneTransform,
) {
    let start = clip.sample_bone(binding, bone, 0.0, rest);
    if settings.horizontal {
        local.translation.x = start.translation.x;
        local.translation.z = start.translation.z;
    }
    if settings.vertical {
        local.translation.y = start.translation.y;
    }
    if settings.yaw {
        let yaw = yaw_of(local.rotation);
        local.rotation = (yaw_of(start.rotation) * yaw.inverse() * local.rotation).normalize();
    }
}

fn accumulate_rotation(sum: &mut Vec4, rotation: Quat, weight: f32) {
    let mut v = Vec4::from(rotation);
    if sum.dot(v) < 0.0 {
        v = -v;
    }
    *sum += v * weight;
}

/// Evaluates `layers` into `pose` (starting from the rest pose) and
/// returns the extracted root motion.
pub fn evaluate_pose(
    skeleton: &Skeleton,
    layers: &[LayerOutput],
    pose: &mut Pose,
    scratch: &mut PoseScratch,
) -> RootMotionDelta {
    let bones = skeleton.len();
    pose.locals.clear();
    pose.locals
        .extend(skeleton.bones.iter().map(|bone| bone.rest));
    let mut root_motion = RootMotionDelta::default();
    let mut root_weight = 0.0;

    for layer in layers {
        if layer.weight <= 1e-4 || layer.instances.is_empty() {
            continue;
        }
        let additive = layer.blend == LayerBlend::Additive;
        scratch.translation.clear();
        scratch.translation.resize(bones, Vec3::ZERO);
        scratch.rotation.clear();
        scratch.rotation.resize(bones, Vec4::ZERO);
        scratch.scale.clear();
        scratch.scale.resize(bones, Vec3::ZERO);
        scratch.animated.clear();
        scratch.animated.resize(bones, false);
        let mut total = 0.0;

        for instance in &layer.instances {
            let clip = &instance.clip.clip;
            let binding = &instance.clip.binding;
            let weight = instance.weight;
            if weight <= 1e-5 {
                continue;
            }
            total += weight;
            scratch.sample.clear();
            scratch
                .sample
                .extend(skeleton.bones.iter().map(|bone| bone.rest));
            clip.sample_bones(binding, instance.time, &mut scratch.sample);
            for bone in binding.bones.iter().flatten() {
                if let Some(flag) = scratch.animated.get_mut(*bone) {
                    *flag = true;
                }
            }

            if layer.root_motion {
                if let (Some(settings), Some(bone)) = (&clip.root_motion, binding.root_bone) {
                    let rest = skeleton.bones[bone].rest;
                    let (translation, yaw) = root_delta(instance, settings, bone, rest);
                    let parent = parent_rest_rotation(skeleton, bone);
                    root_motion.translation += parent * translation * weight;
                    root_motion.rotation = root_motion.rotation.slerp(
                        root_motion.rotation * yaw,
                        weight / (root_weight + weight).max(1e-6),
                    );
                    root_weight += weight;
                    make_in_place(
                        &mut scratch.sample[bone],
                        settings,
                        clip,
                        binding,
                        bone,
                        rest,
                    );
                }
            }

            if additive {
                scratch.reference.clear();
                scratch
                    .reference
                    .extend(skeleton.bones.iter().map(|bone| bone.rest));
                clip.sample_bones(binding, 0.0, &mut scratch.reference);
                for bone in 0..bones {
                    let (sample, reference) = (scratch.sample[bone], scratch.reference[bone]);
                    scratch.translation[bone] +=
                        (sample.translation - reference.translation) * weight;
                    accumulate_rotation(
                        &mut scratch.rotation[bone],
                        reference.rotation.inverse() * sample.rotation,
                        weight,
                    );
                    scratch.scale[bone] +=
                        (sample.scale / reference.scale.max(Vec3::splat(1e-6))) * weight;
                }
            } else {
                for bone in 0..bones {
                    let sample = scratch.sample[bone];
                    scratch.translation[bone] += sample.translation * weight;
                    accumulate_rotation(&mut scratch.rotation[bone], sample.rotation, weight);
                    scratch.scale[bone] += sample.scale * weight;
                }
            }
        }
        if total <= 1e-5 {
            continue;
        }
        // A layer fading towards an empty state fades out.
        let layer_weight = layer.weight.clamp(0.0, 1.0) * total.min(1.0);
        for bone in 0..bones {
            if !scratch.animated[bone] {
                continue;
            }
            let mask = layer
                .mask
                .as_ref()
                .map_or(1.0, |m| m.get(bone).copied().unwrap_or(0.0));
            let factor = layer_weight * mask;
            if factor <= 1e-5 {
                continue;
            }
            let translation = scratch.translation[bone] / total;
            let rotation = {
                let v = scratch.rotation[bone];
                if v.length_squared() > 1e-12 {
                    Quat::from_vec4(v).normalize()
                } else {
                    Quat::IDENTITY
                }
            };
            let scale = scratch.scale[bone] / total;
            let local = &mut pose.locals[bone];
            if additive {
                local.translation += translation * factor;
                local.rotation =
                    (local.rotation * Quat::IDENTITY.slerp(rotation, factor)).normalize();
                local.scale *= Vec3::ONE.lerp(scale, factor);
            } else {
                *local = local.lerp(
                    &BoneTransform {
                        translation,
                        rotation,
                        scale,
                    },
                    factor,
                );
            }
        }
    }
    // A partial root-motion weight (fading in from a state without root
    // motion) scales the motion down; it is deliberately not renormalized.
    root_motion
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::{BoneTrack, ClipBinding};
    use crate::skeleton::Bone;
    use crate::state_machine::ResolvedClip;
    use engine_math::curve::Curve;
    use engine_math::Mat4;

    fn skeleton() -> Skeleton {
        let bone = |name: &str, parent| Bone {
            name: name.into(),
            parent,
            rest: BoneTransform::IDENTITY,
        };
        Skeleton::new(
            "s",
            vec![
                bone("hips", None),
                bone("spine", Some(0)),
                bone("arm", Some(1)),
                bone("leg", Some(0)),
            ],
            Mat4::IDENTITY,
            vec![],
            vec![],
        )
    }

    fn resolved(skeleton: &Skeleton, clip: AnimationClip) -> ResolvedClip {
        let binding = clip.bind(skeleton);
        ResolvedClip {
            clip: Arc::new(clip),
            binding: Arc::new(binding),
        }
    }

    fn rotating(bone: &str, angle: f32) -> AnimationClip {
        AnimationClip {
            duration: 1.0,
            bone_tracks: vec![BoneTrack {
                bone: bone.into(),
                rotation: Some(Curve::constant(Quat::from_rotation_x(angle))),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn instance(clip: ResolvedClip, weight: f32, time: f32, previous_time: f32) -> ClipInstance {
        ClipInstance {
            clip,
            time,
            previous_time,
            weight,
            fire_events: true,
        }
    }

    #[test]
    fn override_layers_blend_and_respect_masks() {
        let skeleton = skeleton();
        let base = resolved(&skeleton, rotating("spine", 1.0));
        let mut upper = rotating("spine", -1.0);
        upper.bone_tracks.push(BoneTrack {
            bone: "leg".into(),
            rotation: Some(Curve::constant(Quat::from_rotation_x(-1.0))),
            ..Default::default()
        });
        let upper = resolved(&skeleton, upper);
        let mask = Arc::new(mask_weights(&skeleton, &["spine".into()], true));
        assert_eq!(*mask, vec![0.0, 1.0, 1.0, 0.0]);
        let layers = vec![
            LayerOutput {
                blend: LayerBlend::Override,
                weight: 1.0,
                mask: None,
                instances: vec![instance(base, 1.0, 0.0, 0.0)],
                root_motion: false,
            },
            LayerOutput {
                blend: LayerBlend::Override,
                weight: 0.5,
                mask: Some(mask),
                instances: vec![instance(upper, 1.0, 0.0, 0.0)],
                root_motion: false,
            },
        ];
        let mut pose = Pose::default();
        evaluate_pose(&skeleton, &layers, &mut pose, &mut PoseScratch::default());
        assert!(
            pose.locals[1].rotation.angle_between(Quat::IDENTITY) < 1e-4,
            "half way between ±1"
        );
        assert_eq!(pose.locals[3], BoneTransform::IDENTITY, "leg masked out");
        assert_eq!(
            pose.locals[2],
            BoneTransform::IDENTITY,
            "unanimated bones keep rest"
        );
    }

    #[test]
    fn additive_layers_add_relative_to_the_first_frame() {
        let skeleton = skeleton();
        let base = resolved(&skeleton, rotating("spine", 0.5));
        let mut nod = AnimationClip {
            duration: 1.0,
            bone_tracks: vec![BoneTrack {
                bone: "spine".into(),
                rotation: Some(Curve::linear([
                    (0.0, Quat::from_rotation_x(2.0)),
                    (1.0, Quat::from_rotation_x(2.3)),
                ])),
                ..Default::default()
            }],
            ..Default::default()
        };
        nod.looping = false;
        let nod = resolved(&skeleton, nod);
        let layers = vec![
            LayerOutput {
                blend: LayerBlend::Override,
                weight: 1.0,
                mask: None,
                instances: vec![instance(base, 1.0, 0.0, 0.0)],
                root_motion: false,
            },
            LayerOutput {
                blend: LayerBlend::Additive,
                weight: 1.0,
                mask: None,
                instances: vec![instance(nod, 1.0, 1.0, 1.0)],
                root_motion: false,
            },
        ];
        let mut pose = Pose::default();
        evaluate_pose(&skeleton, &layers, &mut pose, &mut PoseScratch::default());
        assert!(
            pose.locals[1]
                .rotation
                .angle_between(Quat::from_rotation_x(0.8))
                < 1e-4
        );
    }

    #[test]
    fn root_motion_is_extracted_and_removed_from_the_pose() {
        let skeleton = skeleton();
        let mut walk = AnimationClip {
            duration: 1.0,
            bone_tracks: vec![BoneTrack {
                bone: "hips".into(),
                translation: Some(Curve::linear([
                    (0.0, Vec3::new(0.0, 1.0, 0.0)),
                    (1.0, Vec3::new(0.0, 1.2, 2.0)),
                ])),
                ..Default::default()
            }],
            root_motion: Some(RootMotionSettings {
                bone: "hips".into(),
                horizontal: true,
                vertical: false,
                yaw: false,
            }),
            ..Default::default()
        };
        walk.looping = true;
        let walk = resolved(&skeleton, walk);
        assert_eq!(walk.binding.root_bone, Some(0));
        let layers = |time, previous| {
            vec![LayerOutput {
                blend: LayerBlend::Override,
                weight: 1.0,
                mask: None,
                instances: vec![instance(walk.clone(), 1.0, time, previous)],
                root_motion: true,
            }]
        };
        let mut pose = Pose::default();
        let mut scratch = PoseScratch::default();
        let delta = evaluate_pose(&skeleton, &layers(0.5, 0.25), &mut pose, &mut scratch);
        assert!(
            (delta.translation - Vec3::new(0.0, 0.0, 0.5)).length() < 1e-4,
            "{}",
            delta.translation
        );
        let hips = pose.locals[0].translation;
        assert!(
            (hips - Vec3::new(0.0, 1.1, 0.0)).length() < 1e-4,
            "in place, vertical kept: {hips}"
        );
        // Wrapping from 0.9 to 0.1 covers 0.2s of motion.
        let delta = evaluate_pose(&skeleton, &layers(0.1, 0.9), &mut pose, &mut scratch);
        assert!(
            (delta.translation.z - 0.4).abs() < 1e-4,
            "{}",
            delta.translation
        );
        let _ = ClipBinding::default();
    }
}
