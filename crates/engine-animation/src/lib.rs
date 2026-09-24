//! Animation (ADR 0013): skeletons and skinning palettes, clips with bone,
//! property and event tracks, animation graphs (layers, bone masks,
//! additive blending, state machines, blend trees), root motion and IK.

// ECS query tuples are inherently long; aliasing each one hurts more than helps.
#![allow(clippy::type_complexity)]

pub mod blend;
pub mod clip;
pub mod components;
pub mod curve;
pub mod gltf_import;
pub mod graph;
pub mod ik;
pub mod plugin;
pub mod property;
pub mod skeleton;
pub mod state_machine;
pub mod systems;

pub use blend::{evaluate_pose, mask_weights, LayerOutput, PoseScratch, RootMotionDelta};
pub use clip::{
    AnimationClip, AnimationClipLoader, BoneTrack, ClipBinding, ClipEvent, PropertyCurve,
    PropertyTrack, PropertyValue, RootMotionSettings, ANIMATION_CLIP_VERSION,
};
pub use components::{
    AnimationEvent, AnimationOutput, AnimationPlayer, Animator, AnimatorRuntime, BoneAttachment,
    InverseKinematics, LookAtChain, PlaybackLoop, PlayerRuntime, SkeletonInstance, SkinnedMesh,
    TwoBoneChain,
};
pub use curve::{Animatable, Curve, Interpolation};
pub use graph::{
    AnimGraph, AnimGraphLoader, Blend2DMode, BlendChild1D, BlendChild2D, BoneMask, Condition,
    GraphLayer, LayerBlend, Motion, ParameterDef, ParameterValue, State, StateMachine, Transition,
    TransitionCurve, ANIM_GRAPH_VERSION,
};
pub use plugin::AnimationPlugin;
pub use property::PropertyWrites;
pub use skeleton::{Bone, BoneTransform, Pose, Skeleton, SkeletonLoader};
pub use state_machine::{MachineStatus, Parameters};
