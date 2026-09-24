//! The animation pipeline:
//!
//! * `UpdateSet::AnimationGraph` — attach runtime state, resolve assets,
//!   step animators/players, fire timeline events, gather property writes.
//! * `UpdateSet::AnimationSample` — sample and blend poses (parallel).
//! * `UpdateSet::AnimationApply` — IK, root motion, property tracks, bone
//!   attachments.
//! * `PreRenderSet::SkinPalette` — skinning matrices and bounds.

use std::collections::HashMap;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_assets::Assets;
use engine_core::{FrameTime, GlobalTransform, Parent, SkinPalette, Transform};
use engine_math::{Affine3A, Mat4, Quat, Vec3};
use engine_physics::CharacterInput;

use crate::blend::{evaluate_pose, mask_weights, LayerOutput};
use crate::clip::{AnimationClip, ClipBinding};
use crate::components::{
    AnimationEvent, AnimationOutput, AnimationPlayer, Animator, AnimatorRuntime, BoneAttachment,
    ClipSlot, InverseKinematics, PlaybackLoop, PlayerRuntime, SkeletonInstance, SkinnedMesh,
};
use crate::graph::LayerBlend;
use crate::ik::{solve_look_at, solve_two_bone};
use crate::property::{PropertyKey, PropertyWrites};
use crate::skeleton::Skeleton;
use crate::state_machine::{ClipInstance, MachineRuntime, Parameters, ResolvedClip};

/// Adds runtime companions to new animation components and resets them
/// when their asset references change.
pub fn attach_runtime_state(
    mut commands: Commands,
    skinned: Query<(Entity, Option<&SkeletonInstance>, &SkinnedMesh), Changed<SkinnedMesh>>,
    animators: Query<(Entity, Option<&AnimatorRuntime>, &Animator), Changed<Animator>>,
    players: Query<(Entity, Option<&PlayerRuntime>, &AnimationPlayer), Changed<AnimationPlayer>>,
    outputs: Query<(), With<AnimationOutput>>,
) {
    for (entity, instance, mesh) in &skinned {
        if instance.is_none_or(|i| i.reference != mesh.skeleton) {
            commands.entity(entity).insert((
                SkeletonInstance {
                    reference: mesh.skeleton.clone(),
                    ..Default::default()
                },
                SkinPalette::default(),
            ));
        }
    }
    for (entity, runtime, animator) in &animators {
        if runtime.is_none_or(|r| r.reference != animator.graph) {
            let params = runtime.map(|r| r.params.clone()).unwrap_or_default();
            commands.entity(entity).insert(AnimatorRuntime {
                reference: animator.graph.clone(),
                params,
                ..Default::default()
            });
        }
        if !outputs.contains(entity) {
            commands.entity(entity).insert(AnimationOutput::default());
        }
    }
    for (entity, runtime, player) in &players {
        if runtime.is_none_or(|r| r.slot.reference != player.clip) {
            commands.entity(entity).insert(PlayerRuntime {
                slot: ClipSlot {
                    reference: player.clip.clone(),
                    ..Default::default()
                },
                ..Default::default()
            });
        }
        if !outputs.contains(entity) {
            commands.entity(entity).insert(AnimationOutput::default());
        }
    }
}

/// Picks up a (re)loaded clip; binds it to `skeleton`.
fn refresh_clip(slot: &mut ClipSlot, assets: &Assets, skeleton: Option<(&Arc<Skeleton>, u64)>) {
    if slot.reference.is_empty() {
        slot.resolved = None;
        return;
    }
    let handle = *slot
        .handle
        .get_or_insert_with(|| assets.request::<AnimationClip>(&slot.reference));
    let revision = assets.revision(handle);
    let generation = skeleton.map_or(0, |(_, generation)| generation);
    if slot.resolved.is_some()
        && revision == slot.revision
        && generation == slot.skeleton_generation
    {
        return;
    }
    let Some(clip) = assets.get(handle) else {
        return;
    };
    let binding = match skeleton {
        Some((skeleton, _)) => clip.bind(skeleton),
        None => ClipBinding::default(),
    };
    slot.revision = revision;
    slot.skeleton_generation = generation;
    slot.resolved = Some(ResolvedClip {
        clip,
        binding: Arc::new(binding),
    });
}

/// Resolves skeletons through the asset store.
pub fn resolve_skeletons(assets: Option<Res<Assets>>, mut skeletons: Query<&mut SkeletonInstance>) {
    let Some(assets) = assets else {
        return;
    };
    for mut instance in &mut skeletons {
        if instance.reference.is_empty() {
            continue;
        }
        let handle = match instance.handle {
            Some(handle) => handle,
            None => {
                let handle = assets.request::<Skeleton>(&instance.reference);
                instance.handle = Some(handle);
                handle
            }
        };
        let revision = assets.revision(handle);
        if instance.skeleton.is_some() && revision == instance.revision {
            continue;
        }
        if let Some(skeleton) = assets.get(handle) {
            instance.revision = revision;
            instance.pose = skeleton.rest_pose();
            let mut model = Vec::new();
            instance.pose.model_space(&skeleton, &mut model);
            instance.model = model;
            instance.skeleton = Some(skeleton);
            instance.generation += 1;
        }
    }
}

/// Resolves animator graphs and their clips (bound to the skeleton).
pub fn resolve_animator_assets(
    assets: Option<Res<Assets>>,
    mut animators: Query<(&mut AnimatorRuntime, Option<&SkeletonInstance>)>,
) {
    let Some(assets) = assets else {
        return;
    };
    for (mut runtime, instance) in &mut animators {
        let runtime = &mut *runtime;
        if runtime.reference.is_empty() {
            continue;
        }
        let handle = *runtime
            .handle
            .get_or_insert_with(|| assets.request(&runtime.reference));
        let revision = assets.revision(handle);
        if runtime.graph.is_none() || revision != runtime.revision {
            if let Some(graph) = assets.get(handle) {
                runtime.revision = revision;
                runtime.params = Parameters::from_graph(&graph, &runtime.params);
                runtime.machines = graph
                    .layers
                    .iter()
                    .map(|layer| MachineRuntime::new(&layer.state_machine))
                    .collect();
                let mut slots = HashMap::new();
                for reference in graph.clips() {
                    let key = reference.request_key();
                    let slot = runtime.clips.remove(&key).unwrap_or_else(|| ClipSlot {
                        reference: reference.clone(),
                        ..Default::default()
                    });
                    slots.insert(key, slot);
                }
                runtime.clips = slots;
                runtime.masks.clear();
                runtime.mask_generation = u64::MAX;
                runtime.graph = Some(graph);
            }
        }
        let skeleton = instance.and_then(|i| i.skeleton.as_ref().map(|s| (s, i.generation)));
        for (key, slot) in &mut runtime.clips {
            refresh_clip(slot, &assets, skeleton);
            match &slot.resolved {
                Some(resolved) => {
                    runtime.resolved.insert(key.clone(), resolved.clone());
                }
                None => {
                    runtime.resolved.remove(key);
                }
            }
        }
        let generation = skeleton.map_or(0, |(_, g)| g);
        if let Some(graph) = &runtime.graph {
            if runtime.mask_generation != generation {
                runtime.mask_generation = generation;
                runtime.masks = graph
                    .layers
                    .iter()
                    .map(|layer| {
                        let (mask, (skeleton, _)) = (layer.mask.as_ref()?, skeleton?);
                        Some(Arc::new(mask_weights(
                            skeleton,
                            &mask.bones,
                            mask.include_descendants,
                        )))
                    })
                    .collect();
            }
        }
    }
}

/// Resolves player clips.
pub fn resolve_player_assets(
    assets: Option<Res<Assets>>,
    mut players: Query<(&mut PlayerRuntime, Option<&SkeletonInstance>)>,
) {
    let Some(assets) = assets else {
        return;
    };
    for (mut runtime, instance) in &mut players {
        let skeleton = instance.and_then(|i| i.skeleton.as_ref().map(|s| (s, i.generation)));
        refresh_clip(&mut runtime.slot, &assets, skeleton);
    }
}

fn emit_events(
    entity: Entity,
    instances: &[ClipInstance],
    events: &mut EventWriter<AnimationEvent>,
) {
    for instance in instances.iter().filter(|i| i.fire_events) {
        let clip = &instance.clip.clip;
        for event in clip.events_between(instance.previous_time, instance.time) {
            events.send(AnimationEvent {
                entity,
                clip: clip.name.clone(),
                name: event.name.clone(),
                payload: event.payload.clone(),
                time: event.time,
            });
        }
    }
}

fn gather_properties(
    entity: Entity,
    instances: &[ClipInstance],
    layer_weight: f32,
    writes: &mut PropertyWrites,
) {
    for instance in instances {
        let clip = &instance.clip.clip;
        for track in &clip.property_tracks {
            let Some(value) = track.curve.sample(instance.time) else {
                continue;
            };
            writes.push(
                PropertyKey {
                    entity,
                    target: track.target.clone(),
                    component: track.component.clone(),
                    field: track.field.clone(),
                },
                value,
                instance.weight * layer_weight,
            );
        }
    }
}

/// Steps animators: state machines, blend trees, events, property tracks.
pub fn evaluate_animators(
    time: Option<Res<FrameTime>>,
    mut animators: Query<(
        Entity,
        &Animator,
        &mut AnimatorRuntime,
        &mut AnimationOutput,
    )>,
    mut events: EventWriter<AnimationEvent>,
    mut writes: ResMut<PropertyWrites>,
) {
    let dt = time.map_or(0.0, |t| t.delta_seconds);
    for (entity, animator, mut runtime, mut output) in &mut animators {
        let runtime = &mut *runtime;
        output.layers.clear();
        let Some(graph) = runtime.graph.clone() else {
            continue;
        };
        let step = if animator.paused {
            0.0
        } else {
            dt * animator.speed.max(0.0)
        };
        for (index, layer) in graph.layers.iter().enumerate() {
            let Some(machine) = runtime.machines.get_mut(index) else {
                continue;
            };
            machine.step(
                &layer.state_machine,
                &mut runtime.params,
                &runtime.resolved,
                step,
            );
            let mut instances = Vec::new();
            machine.collect(
                &layer.state_machine,
                &runtime.params,
                &runtime.resolved,
                1.0,
                &mut instances,
            );
            let weight = layer
                .weight_parameter
                .as_ref()
                .map_or(layer.weight, |p| runtime.params.float(p))
                .clamp(0.0, 1.0);
            if weight > 1e-4 {
                emit_events(entity, &instances, &mut events);
                gather_properties(entity, &instances, weight, &mut writes);
            }
            output.layers.push(LayerOutput {
                blend: layer.blend,
                weight,
                mask: runtime.masks.get(index).cloned().flatten(),
                instances,
                root_motion: index == 0 && animator.apply_root_motion,
            });
        }
    }
}

/// Advances single-clip players.
pub fn evaluate_players(
    time: Option<Res<FrameTime>>,
    mut players: Query<(
        Entity,
        &mut AnimationPlayer,
        &mut PlayerRuntime,
        &mut AnimationOutput,
    )>,
    mut events: EventWriter<AnimationEvent>,
    mut writes: ResMut<PropertyWrites>,
) {
    let dt = time.map_or(0.0, |t| t.delta_seconds);
    for (entity, mut player, mut runtime, mut output) in &mut players {
        output.layers.clear();
        let Some(resolved) = runtime.slot.resolved.clone() else {
            continue;
        };
        let clip = &resolved.clip;
        let length = clip.length();
        let looping = match player.looping {
            PlaybackLoop::ClipDefault => clip.looping,
            PlaybackLoop::Loop => true,
            PlaybackLoop::Once => false,
        };
        let previous = if runtime.started {
            runtime.previous_time
        } else {
            -1e-4
        };
        if player.playing && dt > 0.0 {
            let mut next = player.time + dt * player.speed;
            if looping && length > 0.0 {
                next = next.rem_euclid(length);
            } else if next >= length || next <= 0.0 {
                next = next.clamp(0.0, length);
                player.playing = false;
            }
            player.time = next;
        }
        let current = player.time;
        runtime.previous_time = current;
        runtime.started = true;
        let instances = vec![ClipInstance {
            clip: resolved.clone(),
            time: current,
            previous_time: previous,
            weight: 1.0,
            fire_events: true,
        }];
        if previous != current {
            emit_events(entity, &instances, &mut events);
        }
        gather_properties(entity, &instances, 1.0, &mut writes);
        output.layers.push(LayerOutput {
            blend: LayerBlend::Override,
            weight: 1.0,
            mask: None,
            instances,
            root_motion: false,
        });
    }
}

/// Samples and blends poses (in parallel over entities).
pub fn sample_poses(mut skeletons: Query<(&mut SkeletonInstance, Option<&AnimationOutput>)>) {
    skeletons.par_iter_mut().for_each(|(mut instance, output)| {
        let instance = &mut *instance;
        let Some(skeleton) = instance.skeleton.clone() else {
            return;
        };
        let Some(output) = output else {
            return;
        };
        instance.root_motion = evaluate_pose(
            &skeleton,
            &output.layers,
            &mut instance.pose,
            &mut instance.scratch,
        );
        instance.pose.model_space(&skeleton, &mut instance.model);
    });
}

/// Applies IK chains (targets in world space).
pub fn apply_inverse_kinematics(
    mut query: Query<(
        &mut SkeletonInstance,
        &InverseKinematics,
        Option<&GlobalTransform>,
    )>,
) {
    for (mut instance, ik, global) in &mut query {
        let instance = &mut *instance;
        let Some(skeleton) = instance.skeleton.clone() else {
            continue;
        };
        let world_to_model = global.map_or(Affine3A::IDENTITY, |g| g.0.inverse());
        let to_model = |p: Vec3| world_to_model.transform_point3(p);
        for chain in &ik.two_bone {
            let bones = (
                skeleton.bone_index(&chain.upper),
                skeleton.bone_index(&chain.middle),
                skeleton.bone_index(&chain.end),
            );
            if let (Some(upper), Some(middle), Some(end)) = bones {
                solve_two_bone(
                    &skeleton,
                    &mut instance.pose,
                    &mut instance.model,
                    upper,
                    middle,
                    end,
                    to_model(chain.target),
                    to_model(chain.pole),
                    chain.weight,
                );
            }
        }
        for chain in &ik.look_at {
            if let Some(bone) = skeleton.bone_index(&chain.bone) {
                solve_look_at(
                    &skeleton,
                    &mut instance.pose,
                    &mut instance.model,
                    bone,
                    chain.forward,
                    to_model(chain.target),
                    chain.max_angle,
                    chain.weight,
                );
            }
        }
    }
}

/// Moves entities by their extracted root motion: through the character
/// controller when present, else directly.
pub fn apply_root_motion(
    mut query: Query<(
        &Animator,
        &mut SkeletonInstance,
        &mut Transform,
        Option<&mut CharacterInput>,
    )>,
) {
    for (animator, mut instance, mut transform, input) in &mut query {
        let delta = std::mem::take(&mut instance.root_motion);
        if !animator.apply_root_motion
            || (delta.translation == Vec3::ZERO && delta.rotation == Quat::IDENTITY)
        {
            continue;
        }
        let world = transform.rotation * (delta.translation * transform.scale);
        match input {
            Some(mut input) => {
                input.root_motion += world;
                input.root_rotation = (delta.rotation * input.root_rotation).normalize();
            }
            None => {
                transform.translation += world;
                transform.rotation = (transform.rotation * delta.rotation).normalize();
            }
        }
    }
}

/// Places bone attachments on their parent's bones.
pub fn update_bone_attachments(
    mut attachments: Query<(&BoneAttachment, &Parent, &mut Transform)>,
    skeletons: Query<&SkeletonInstance>,
) {
    for (attachment, parent, mut transform) in &mut attachments {
        let Ok(instance) = skeletons.get(parent.0) else {
            continue;
        };
        let Some(bone) = instance.bone_model(&attachment.bone) else {
            continue;
        };
        let local =
            bone * Affine3A::from_rotation_translation(attachment.rotation, attachment.offset);
        let (scale, rotation, translation) = local.to_scale_rotation_translation();
        let rotation = rotation.normalize();
        if transform.translation != translation
            || transform.rotation != rotation
            || transform.scale != scale
        {
            transform.translation = translation;
            transform.rotation = rotation;
            transform.scale = scale;
        }
    }
}

/// Skinning matrices (`model[joint] * inverse_bind`) and posed bounds.
pub fn compute_skin_palettes(
    mut query: Query<(&SkeletonInstance, &SkinnedMesh, &mut SkinPalette)>,
) {
    query
        .par_iter_mut()
        .for_each(|(instance, mesh, mut palette)| {
            let Some(skeleton) = &instance.skeleton else {
                return;
            };
            if instance.model.len() != skeleton.len() {
                return;
            }
            let palette = &mut *palette;
            palette.joint_matrices.clear();
            let mut min = Vec3::splat(f32::MAX);
            let mut max = Vec3::splat(f32::MIN);
            for (joint, inverse_bind) in skeleton.joints.iter().zip(&skeleton.inverse_bind) {
                let model = instance.model[*joint];
                palette
                    .joint_matrices
                    .push(Mat4::from(model) * *inverse_bind);
                let position = Vec3::from(model.translation);
                min = min.min(position);
                max = max.max(position);
            }
            palette.bounds = (!skeleton.joints.is_empty()).then(|| {
                let pad = Vec3::splat(mesh.bounds_padding.max(0.0));
                (min - pad, max + pad)
            });
        });
}
