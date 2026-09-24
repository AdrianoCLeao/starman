//! Animation components (reflected, authored in scenes) and their runtime
//! companions (attached automatically).

use std::collections::HashMap;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use bevy_reflect::Reflect;
use engine_assets::{AssetRef, Handle};
use engine_math::{Affine3A, Quat, Vec3};

use crate::blend::{LayerOutput, PoseScratch, RootMotionDelta};
use crate::clip::AnimationClip;
use crate::graph::{AnimGraph, ParameterValue};
use crate::skeleton::{Pose, Skeleton};
use crate::state_machine::{MachineRuntime, MachineStatus, Parameters, ResolvedClip};

/// Deforms the entity's mesh with a skeleton (`file.glb#skin:<i>`).
#[derive(Component, Clone, Debug, PartialEq, Reflect, engine_reflect::RegisterReflect)]
pub struct SkinnedMesh {
    pub skeleton: AssetRef,
    /// Added around the joints' bounds for culling (meters).
    #[engine_reflect(range(min = 0.0, max = 10.0))]
    pub bounds_padding: f32,
}

impl Default for SkinnedMesh {
    fn default() -> Self {
        Self {
            skeleton: AssetRef::default(),
            bounds_padding: 0.5,
        }
    }
}

/// Drives the entity with an animation graph (`*.animgraph.ron`).
#[derive(Component, Clone, Debug, PartialEq, Reflect, engine_reflect::RegisterReflect)]
pub struct Animator {
    pub graph: AssetRef,
    #[engine_reflect(range(min = 0.0, max = 10.0))]
    pub speed: f32,
    /// Moves the entity (or its character controller) by the root motion
    /// of the base layer.
    pub apply_root_motion: bool,
    pub paused: bool,
}

impl Default for Animator {
    fn default() -> Self {
        Self {
            graph: AssetRef::default(),
            speed: 1.0,
            apply_root_motion: false,
            paused: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Reflect)]
pub enum PlaybackLoop {
    /// Use the clip's own looping flag.
    #[default]
    ClipDefault,
    Loop,
    Once,
}

/// Plays one clip (props, doors, lights, UI-less cutscene bits).
#[derive(Component, Clone, Debug, PartialEq, Reflect, engine_reflect::RegisterReflect)]
pub struct AnimationPlayer {
    pub clip: AssetRef,
    #[engine_reflect(range(min = -10.0, max = 10.0))]
    pub speed: f32,
    pub playing: bool,
    pub looping: PlaybackLoop,
    /// Playhead in seconds (write it to seek).
    pub time: f32,
}

impl Default for AnimationPlayer {
    fn default() -> Self {
        Self {
            clip: AssetRef::default(),
            speed: 1.0,
            playing: true,
            looping: PlaybackLoop::ClipDefault,
            time: 0.0,
        }
    }
}

/// Places a child entity on a bone of its parent's skeleton (weapons,
/// props in hand, effects).
#[derive(Component, Clone, Debug, PartialEq, Reflect, engine_reflect::RegisterReflect)]
pub struct BoneAttachment {
    pub bone: String,
    pub offset: Vec3,
    pub rotation: Quat,
}

impl Default for BoneAttachment {
    fn default() -> Self {
        Self {
            bone: String::new(),
            offset: Vec3::ZERO,
            rotation: Quat::IDENTITY,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Reflect)]
pub struct TwoBoneChain {
    pub name: String,
    pub upper: String,
    pub middle: String,
    pub end: String,
    /// World-space target and pole.
    pub target: Vec3,
    pub pole: Vec3,
    pub weight: f32,
}

impl Default for TwoBoneChain {
    fn default() -> Self {
        Self {
            name: String::new(),
            upper: String::new(),
            middle: String::new(),
            end: String::new(),
            target: Vec3::ZERO,
            pole: Vec3::Z,
            weight: 0.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Reflect)]
pub struct LookAtChain {
    pub name: String,
    pub bone: String,
    /// Bone-local axis that should face the target.
    pub forward: Vec3,
    /// World-space target.
    pub target: Vec3,
    /// Radians from the animated pose.
    pub max_angle: f32,
    pub weight: f32,
}

impl Default for LookAtChain {
    fn default() -> Self {
        Self {
            name: String::new(),
            bone: String::new(),
            forward: Vec3::Z,
            target: Vec3::ZERO,
            max_angle: 1.2,
            weight: 0.0,
        }
    }
}

/// IK chains applied after the animation pose (weights are animatable).
#[derive(Component, Clone, Debug, Default, PartialEq, Reflect, engine_reflect::RegisterReflect)]
pub struct InverseKinematics {
    pub two_bone: Vec<TwoBoneChain>,
    pub look_at: Vec<LookAtChain>,
}

impl InverseKinematics {
    pub fn set_target(&mut self, chain: &str, target: Vec3, weight: f32) -> bool {
        if let Some(c) = self.two_bone.iter_mut().find(|c| c.name == chain) {
            c.target = target;
            c.weight = weight;
            return true;
        }
        if let Some(c) = self.look_at.iter_mut().find(|c| c.name == chain) {
            c.target = target;
            c.weight = weight;
            return true;
        }
        false
    }
}

/// Runtime skeleton state (pose buffers) of a [`SkinnedMesh`].
#[derive(Component, Default)]
pub struct SkeletonInstance {
    pub(crate) reference: AssetRef,
    pub(crate) handle: Option<Handle<Skeleton>>,
    pub(crate) revision: u64,
    pub skeleton: Option<Arc<Skeleton>>,
    pub pose: Pose,
    /// Model-space bone transforms of the final pose.
    pub model: Vec<Affine3A>,
    pub(crate) scratch: PoseScratch,
    /// Root motion extracted this frame (model space).
    pub root_motion: RootMotionDelta,
    /// Bumped when the skeleton (re)loads.
    pub generation: u64,
}

impl SkeletonInstance {
    /// Model-space transform of `bone`.
    pub fn bone_model(&self, bone: &str) -> Option<Affine3A> {
        let index = self.skeleton.as_ref()?.bone_index(bone)?;
        self.model.get(index).copied()
    }
}

#[derive(Clone, Default)]
pub(crate) struct ClipSlot {
    pub reference: AssetRef,
    pub handle: Option<Handle<AnimationClip>>,
    pub revision: u64,
    pub skeleton_generation: u64,
    pub resolved: Option<ResolvedClip>,
}

/// This frame's clip instances per layer (graph → sampler hand-off).
#[derive(Component, Default, Clone)]
pub struct AnimationOutput {
    pub layers: Vec<LayerOutput>,
}

/// Runtime state of an [`Animator`]: parameters, state machines, clips.
#[derive(Component, Default)]
pub struct AnimatorRuntime {
    pub(crate) reference: AssetRef,
    pub(crate) handle: Option<Handle<AnimGraph>>,
    pub(crate) revision: u64,
    pub(crate) graph: Option<Arc<AnimGraph>>,
    pub(crate) params: Parameters,
    pub(crate) machines: Vec<MachineRuntime>,
    pub(crate) clips: HashMap<String, ClipSlot>,
    pub(crate) resolved: HashMap<String, ResolvedClip>,
    pub(crate) masks: Vec<Option<Arc<Vec<f32>>>>,
    pub(crate) mask_generation: u64,
}

impl AnimatorRuntime {
    pub fn set_parameter(&mut self, name: &str, value: ParameterValue) {
        self.params.set(name, value);
    }

    pub fn set_float(&mut self, name: &str, value: f32) {
        self.params.set(name, ParameterValue::Float(value));
    }

    pub fn set_int(&mut self, name: &str, value: i32) {
        self.params.set(name, ParameterValue::Int(value));
    }

    pub fn set_bool(&mut self, name: &str, value: bool) {
        self.params.set(name, ParameterValue::Bool(value));
    }

    pub fn set_trigger(&mut self, name: &str) {
        self.params.set(name, ParameterValue::Trigger(true));
    }

    pub fn reset_trigger(&mut self, name: &str) {
        self.params.set(name, ParameterValue::Trigger(false));
    }

    pub fn parameter(&self, name: &str) -> Option<ParameterValue> {
        self.params.get(name)
    }

    pub fn parameters(&self) -> impl Iterator<Item = (&String, &ParameterValue)> {
        self.params.iter()
    }

    /// Whether the graph has loaded.
    pub fn is_ready(&self) -> bool {
        self.graph.is_some()
    }

    pub fn graph(&self) -> Option<&Arc<AnimGraph>> {
        self.graph.as_ref()
    }

    /// Status of `layer`'s state machine.
    pub fn status(&self, layer: usize) -> Option<MachineStatus> {
        let graph = self.graph.as_ref()?;
        let machine = &graph.layers.get(layer)?.state_machine;
        Some(self.machines.get(layer)?.status(machine))
    }

    /// Name of the current state of `layer`.
    pub fn current_state(&self, layer: usize) -> Option<String> {
        self.status(layer).map(|status| status.state)
    }

    /// Forces `layer` into `state` (crossfading over `fade` seconds).
    pub fn play(&mut self, layer: usize, state: &str, fade: f32) -> bool {
        let Some(graph) = self.graph.clone() else {
            return false;
        };
        let Some(machine) = graph.layers.get(layer).map(|l| &l.state_machine) else {
            return false;
        };
        let (Some(index), Some(runtime)) =
            (machine.state_index(state), self.machines.get_mut(layer))
        else {
            return false;
        };
        runtime.play(machine, index, fade);
        true
    }
}

/// Runtime state of an [`AnimationPlayer`].
#[derive(Component, Default)]
pub struct PlayerRuntime {
    pub(crate) slot: ClipSlot,
    pub(crate) previous_time: f32,
    pub(crate) started: bool,
}

/// A timeline event fired by a playing clip.
#[derive(Event, Clone, Debug, PartialEq)]
pub struct AnimationEvent {
    pub entity: Entity,
    pub clip: String,
    pub name: String,
    pub payload: String,
    pub time: f32,
}
