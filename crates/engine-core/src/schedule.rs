//! Engine schedules and the canonical system sets that order subsystems
//! inside them (ADR 0012).
//!
//! Every engine subsystem registers its systems into one of these sets
//! instead of relying on ad-hoc `.chain()` calls, so gameplay code (Lua,
//! Rust plugins, project systems) can be ordered relative to physics,
//! animation, UI and friends without knowing their internals.

use bevy_ecs::schedule::{ExecutorKind, IntoSystemSetConfigs, Schedule, ScheduleLabel, SystemSet};

#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Startup;

/// Runs once per frame after the OS pump and before any fixed step, so
/// input actions (and anything else sampled per frame) are current when
/// the simulation reads them.
#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
pub struct First;

#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FixedUpdate;

#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Update;

#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
pub struct PreRender;

/// Which engine schedule a system (or a scripted callback) runs in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScheduleKind {
    Startup,
    First,
    FixedUpdate,
    Update,
    PreRender,
}

/// Ordered sets of the per-frame [`First`] schedule.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FirstSet {
    /// Input action evaluation, device assignment, rebinding.
    Input,
    /// Consumers of fresh per-frame input (UI pointer routing, …).
    Late,
}

/// Ordered sets of the fixed-step simulation schedule.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FixedSet {
    /// Fixed-rate input sampling (action state is already current).
    Input,
    /// Script (`on_fixed_update`) and plugin fixed callbacks.
    Scripts,
    /// Engine-side gameplay helpers (character motors, damage, …).
    Gameplay,
    /// ECS → physics world synchronisation (new bodies, kinematic targets).
    PhysicsSync,
    /// The physics solver step.
    PhysicsStep,
    /// Physics world → ECS transform write-back.
    PhysicsWriteback,
    /// Collision/trigger event emission.
    PhysicsEvents,
    /// Navigation agents, crowd and obstacle updates.
    Navigation,
    /// Behavior trees and perception.
    Ai,
}

/// Ordered sets of the per-frame update schedule.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UpdateSet {
    /// Raw input → action state, local players, rebinding sessions.
    InputActions,
    /// Script (`on_update`) and plugin update callbacks.
    Scripts,
    /// Engine-side gameplay reactions (interaction, pickups, …).
    Gameplay,
    /// Animation graph evaluation (state machines, transitions, params).
    AnimationGraph,
    /// Clip sampling, blending, layers and masks.
    AnimationSample,
    /// Post-process: IK, root motion extraction, property tracks, write-back.
    AnimationApply,
    /// Particle simulation (CPU) and GPU dispatch preparation.
    Particles,
    /// Audio listeners, emitters, zones, mixer snapshots.
    Audio,
    /// UI layout, text shaping and bindings.
    UiLayout,
    /// UI hit-testing, focus navigation, widget events.
    UiInteraction,
    /// Save/load requests.
    Save,
}

/// Ordered sets of the pre-render schedule.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PreRenderSet {
    /// Local → global transform propagation.
    TransformPropagate,
    /// Joint palettes for skinned meshes (after propagation).
    SkinPalette,
    /// World-space bounds and visibility inputs.
    VisibilityBounds,
    /// Anything that must observe the final frame state (debug draw, …).
    Late,
}

pub struct EngineSchedules {
    pub startup: Schedule,
    pub first: Schedule,
    pub fixed_update: Schedule,
    pub update: Schedule,
    pub pre_render: Schedule,
}

impl Default for EngineSchedules {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineSchedules {
    pub fn new() -> Self {
        let mut startup = Schedule::new(Startup);
        startup.set_executor_kind(ExecutorKind::MultiThreaded);

        let mut first = Schedule::new(First);
        first.set_executor_kind(ExecutorKind::MultiThreaded);
        first.configure_sets((FirstSet::Input, FirstSet::Late).chain());

        let mut fixed_update = Schedule::new(FixedUpdate);
        fixed_update.set_executor_kind(ExecutorKind::MultiThreaded);
        fixed_update.configure_sets(
            (
                FixedSet::Input,
                FixedSet::Scripts,
                FixedSet::Gameplay,
                FixedSet::PhysicsSync,
                FixedSet::PhysicsStep,
                FixedSet::PhysicsWriteback,
                FixedSet::PhysicsEvents,
                FixedSet::Navigation,
                FixedSet::Ai,
            )
                .chain(),
        );

        let mut update = Schedule::new(Update);
        update.set_executor_kind(ExecutorKind::MultiThreaded);
        update.configure_sets(
            (
                UpdateSet::InputActions,
                UpdateSet::Scripts,
                UpdateSet::Gameplay,
                UpdateSet::AnimationGraph,
                UpdateSet::AnimationSample,
                UpdateSet::AnimationApply,
                UpdateSet::Particles,
                UpdateSet::Audio,
                UpdateSet::UiLayout,
                UpdateSet::UiInteraction,
                UpdateSet::Save,
            )
                .chain(),
        );

        let mut pre_render = Schedule::new(PreRender);
        pre_render.set_executor_kind(ExecutorKind::MultiThreaded);
        pre_render.configure_sets(
            (
                PreRenderSet::TransformPropagate,
                PreRenderSet::SkinPalette,
                PreRenderSet::VisibilityBounds,
                PreRenderSet::Late,
            )
                .chain(),
        );

        Self {
            startup,
            first,
            fixed_update,
            update,
            pre_render,
        }
    }

    pub fn get_mut(&mut self, kind: ScheduleKind) -> &mut Schedule {
        match kind {
            ScheduleKind::Startup => &mut self.startup,
            ScheduleKind::First => &mut self.first,
            ScheduleKind::FixedUpdate => &mut self.fixed_update,
            ScheduleKind::Update => &mut self.update,
            ScheduleKind::PreRender => &mut self.pre_render,
        }
    }
}
