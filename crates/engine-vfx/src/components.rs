//! Particle emitter component, commands and runtime state.

use std::sync::Arc;

use bevy_ecs::prelude::*;
use bevy_reflect::Reflect;
use engine_assets::{AssetRef, Handle};

use crate::effect::ParticleEffect;
use crate::sim::{CpuEmitter, MeshPoints, SpawnScheduler};

/// Plays a particle effect (`*.vfx.ron`) at the entity.
#[derive(Component, Clone, Debug, PartialEq, Reflect, engine_reflect::RegisterReflect)]
pub struct ParticleEmitter {
    pub effect: AssetRef,
    /// The simulation advances.
    pub playing: bool,
    /// New particles spawn (stop emitting to let the effect die out).
    pub emitting: bool,
    /// Spawn-rate and burst multiplier.
    #[engine_reflect(range(min = 0.0, max = 10.0))]
    pub intensity: f32,
    /// Multiplies every particle's color.
    #[engine_reflect(color)]
    pub tint: [f32; 4],
    #[engine_reflect(range(min = 0.0, max = 10.0))]
    pub time_scale: f32,
    /// Varies the random stream between instances of the same effect.
    pub seed: u32,
    /// Despawn the entity when a non-looping effect has finished.
    pub despawn_when_finished: bool,
}

impl Default for ParticleEmitter {
    fn default() -> Self {
        Self {
            effect: AssetRef::default(),
            playing: true,
            emitting: true,
            intensity: 1.0,
            tint: [1.0; 4],
            time_scale: 1.0,
            seed: 0,
            despawn_when_finished: false,
        }
    }
}

/// Imperative control of emitters (scripts, gameplay).
#[derive(Event, Clone, Debug, PartialEq)]
pub enum ParticleCommand {
    /// Clears particles and restarts the effect timeline.
    Restart(Entity),
    /// Spawns `count` extra particles from one emitter (all when `None`).
    Burst {
        entity: Entity,
        emitter: Option<String>,
        count: u32,
    },
}

/// Which backend an emitter instance runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActiveBackend {
    Cpu,
    Gpu,
}

/// One queued GPU simulation step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuStep {
    pub spawn: u32,
    pub dt: f32,
    pub time: f32,
}

pub(crate) const MAX_PENDING_GPU_STEPS: usize = 120;

#[derive(Clone, Debug)]
pub struct EmitterState {
    pub backend: ActiveBackend,
    pub(crate) scheduler: SpawnScheduler,
    pub cpu: CpuEmitter,
    pub(crate) mesh_handle: Option<Handle<engine_assets::MeshData>>,
    pub mesh_points: Option<MeshPoints>,
    /// Steps the GPU backend has yet to simulate (consumed by the renderer).
    pub pending_gpu: Vec<GpuStep>,
    /// Clear the GPU particle buffers before the next step.
    pub gpu_reset: bool,
    /// Capacity actually granted (budget-clamped).
    pub capacity: u32,
    /// Live particles last reported (CPU exact; GPU estimated).
    pub alive: u32,
}

/// Runtime state of a [`ParticleEmitter`].
#[derive(Component, Default)]
pub struct ParticleEffectState {
    pub(crate) reference: AssetRef,
    pub(crate) handle: Option<Handle<ParticleEffect>>,
    pub(crate) revision: u64,
    pub effect: Option<Arc<ParticleEffect>>,
    pub emitters: Vec<EmitterState>,
    /// Effect time in seconds.
    pub time: f32,
    pub(crate) prewarmed: bool,
    /// Bumped whenever GPU buffers must be (re)created.
    pub generation: u64,
    pub finished: bool,
}

impl ParticleEffectState {
    /// Live particles over every emitter.
    pub fn alive(&self) -> u32 {
        self.emitters.iter().map(|e| e.alive).sum()
    }
}
