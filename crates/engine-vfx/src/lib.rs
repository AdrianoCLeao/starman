//! Particle effects (ADR 0014): authored `*.vfx.ron` effects whose
//! emitters simulate on the GPU (compute: spawn/update/compaction, bitonic
//! sort, indirect draws) or on the CPU (sub-emitters, physics collision,
//! devices without compute), rendered by one render-extension node.

// ECS query tuples are inherently long; aliasing each one hurts more than helps.
#![allow(clippy::type_complexity)]

pub mod components;
pub mod effect;
pub mod plugin;
pub mod render;
pub mod sim;
pub mod systems;

pub use components::{
    ActiveBackend, EmitterState, GpuStep, ParticleCommand, ParticleEffectState, ParticleEmitter,
};
pub use effect::{
    BlendMode, Burst, EmitterDef, Flipbook, Init, Module, ParticleEffect, ParticleEffectLoader,
    Range, Render, RenderMode, Shape, SimulationBackend, SimulationSpace, Spawn, SubEmitter,
    SubEmitterTrigger, VelocityDirection, PARTICLE_EFFECT_VERSION,
};
pub use plugin::VfxPlugin;
pub use render::ParticleRenderer;
pub use sim::{CpuEmitter, CpuParticle, GpuParticle, SpawnScheduler};
pub use systems::ParticleSettings;
