//! Per-frame particle update (`UpdateSet::Particles`): effect resolution,
//! spawn scheduling, CPU simulation, GPU step queues, budgets, commands.

use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_assets::{Assets, LoadState, MeshData};
use engine_core::{FrameTime, GlobalTransform, Transform};
use engine_math::{Affine3A, Vec3, Vec4};
use engine_physics::{PhysicsWorld3D, SpatialQuery, SpatialQueryFilter};

use crate::components::{
    ActiveBackend, EmitterState, GpuStep, ParticleCommand, ParticleEffectState, ParticleEmitter,
    MAX_PENDING_GPU_STEPS,
};
use crate::effect::{EmitterDef, ParticleEffect, Shape, SimulationBackend};
use crate::sim::{sample_mesh_surface, CpuEmitter, SpawnScheduler, StepContext, SubSpawn};

/// Backend availability and budgets (updated by the renderer from the
/// device tier and quality preset).
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct ParticleSettings {
    /// Compute shaders are available (GPU backend usable).
    pub gpu_compute: bool,
    pub max_cpu_particles: u32,
    pub max_gpu_particles: u32,
}

impl Default for ParticleSettings {
    fn default() -> Self {
        Self {
            gpu_compute: false,
            max_cpu_particles: 10_000,
            max_gpu_particles: 100_000,
        }
    }
}

const MESH_SAMPLES: usize = 1024;
const PREWARM_STEP: f32 = 1.0 / 30.0;

fn backend_for(
    def: &EmitterDef,
    effect: &ParticleEffect,
    settings: &ParticleSettings,
) -> ActiveBackend {
    let gpu_ok = settings.gpu_compute && def.gpu_capable(effect);
    let collision = def
        .modules
        .iter()
        .any(|m| matches!(m, crate::effect::Module::Collision { .. }));
    match def.backend {
        SimulationBackend::Cpu => ActiveBackend::Cpu,
        SimulationBackend::Gpu if gpu_ok => ActiveBackend::Gpu,
        SimulationBackend::Gpu => ActiveBackend::Cpu,
        // Auto: physics collision is exact only on the CPU; big emitters
        // go to the GPU.
        SimulationBackend::Auto if gpu_ok && !(collision && def.max_particles <= 2048) => {
            ActiveBackend::Gpu
        }
        SimulationBackend::Auto => ActiveBackend::Cpu,
    }
}

fn build_states(
    effect: &ParticleEffect,
    seed: u32,
    settings: &ParticleSettings,
) -> Vec<EmitterState> {
    let mut cpu_budget = settings.max_cpu_particles;
    let mut gpu_budget = settings.max_gpu_particles;
    effect
        .emitters
        .iter()
        .enumerate()
        .map(|(index, def)| {
            let backend = backend_for(def, effect, settings);
            let budget = match backend {
                ActiveBackend::Cpu => &mut cpu_budget,
                ActiveBackend::Gpu => &mut gpu_budget,
            };
            let capacity = def.max_particles.min(*budget);
            *budget -= capacity;
            let seed = effect
                .seed
                .wrapping_mul(0x9E37_79B9)
                .wrapping_add(seed)
                .wrapping_add(index as u32 * 7919);
            EmitterState {
                backend,
                scheduler: SpawnScheduler::default(),
                cpu: CpuEmitter::new(
                    if backend == ActiveBackend::Cpu {
                        capacity as usize
                    } else {
                        0
                    },
                    seed,
                ),
                mesh_handle: None,
                mesh_points: None,
                pending_gpu: Vec::new(),
                gpu_reset: true,
                capacity,
                alive: 0,
            }
        })
        .collect()
}

fn queue_gpu_step(state: &mut EmitterState, step: GpuStep) {
    if state.pending_gpu.len() >= MAX_PENDING_GPU_STEPS {
        // Nobody is rendering: merge into the last step to stay bounded.
        if let Some(last) = state.pending_gpu.last_mut() {
            last.spawn = last.spawn.saturating_add(step.spawn);
            last.dt += step.dt;
            last.time = step.time;
            return;
        }
    }
    state.pending_gpu.push(step);
}

/// Advances every particle effect by one frame.
#[allow(clippy::too_many_arguments)]
pub fn update_particles(
    mut commands: Commands,
    time: Option<Res<FrameTime>>,
    assets: Option<Res<Assets>>,
    settings: Res<ParticleSettings>,
    physics: Option<Res<PhysicsWorld3D>>,
    spatial: SpatialQuery,
    mut command_events: EventReader<ParticleCommand>,
    mut emitters: Query<(
        Entity,
        &ParticleEmitter,
        Option<&mut ParticleEffectState>,
        Option<&GlobalTransform>,
        Option<&Transform>,
    )>,
) {
    let dt = time.map_or(0.0, |t| t.delta_seconds);
    let gravity = physics
        .as_ref()
        .map(|p| Vec3::new(p.gravity.x, p.gravity.y, p.gravity.z))
        .unwrap_or(Vec3::new(0.0, -9.81, 0.0));
    let queries = spatial.get();
    let raycast = |from: Vec3, to: Vec3| -> Option<(Vec3, Vec3)> {
        let queries = queries.as_ref()?;
        let delta = to - from;
        let distance = delta.length();
        let hit = queries.raycast(from, delta, distance, &SpatialQueryFilter::default())?;
        Some((hit.point, hit.normal))
    };
    let commands_list: Vec<ParticleCommand> = command_events.read().cloned().collect();

    for (entity, emitter, state, global, local) in &mut emitters {
        let Some(mut state) = state else {
            commands.entity(entity).insert(ParticleEffectState {
                reference: emitter.effect.clone(),
                ..Default::default()
            });
            continue;
        };
        let state = &mut *state;
        let Some(assets) = assets.as_deref() else {
            continue;
        };

        // Effect (re)load.
        if state.reference != emitter.effect {
            *state = ParticleEffectState {
                reference: emitter.effect.clone(),
                generation: state.generation + 1,
                ..Default::default()
            };
        }
        if state.reference.is_empty() {
            continue;
        }
        let handle = *state
            .handle
            .get_or_insert_with(|| assets.request::<ParticleEffect>(&state.reference));
        let revision = assets.revision(handle);
        let restart = commands_list
            .iter()
            .any(|c| matches!(c, ParticleCommand::Restart(e) if *e == entity));
        if state.effect.is_none() || revision != state.revision || restart {
            let Some(effect) = assets.get(handle) else {
                continue;
            };
            state.revision = revision;
            state.emitters = build_states(&effect, emitter.seed, &settings);
            state.effect = Some(effect);
            state.time = 0.0;
            state.prewarmed = false;
            state.finished = false;
            state.generation += 1;
        }
        let effect = state.effect.clone().expect("loaded above");

        // Backend changes (renderer came up, device lost) restart emitters.
        let backends_changed = effect
            .emitters
            .iter()
            .zip(&state.emitters)
            .any(|(def, s)| backend_for(def, &effect, &settings) != s.backend);
        if backends_changed {
            state.emitters = build_states(&effect, emitter.seed, &settings);
            state.generation += 1;
        }

        // Mesh shapes.
        for (def, emitter_state) in effect.emitters.iter().zip(&mut state.emitters) {
            let Shape::Mesh { mesh } = &def.shape else {
                continue;
            };
            if emitter_state.mesh_points.is_some() {
                continue;
            }
            let handle = *emitter_state
                .mesh_handle
                .get_or_insert_with(|| assets.request::<MeshData>(mesh));
            match assets.state(handle) {
                Some(LoadState::Loaded) => {
                    if let Some(data) = assets.get(handle) {
                        let positions: Vec<Vec3> = data
                            .vertices
                            .iter()
                            .map(|v| Vec3::from(v.position))
                            .collect();
                        emitter_state.mesh_points = Some(Arc::new(sample_mesh_surface(
                            &positions,
                            &data.indices,
                            MESH_SAMPLES,
                            effect.seed ^ emitter.seed,
                        )));
                    }
                }
                Some(LoadState::Failed(reason)) => {
                    log::warn!(target: "engine::vfx", "emitter '{}' mesh shape: {reason}", def.name);
                    emitter_state.mesh_points = Some(Arc::new(vec![(Vec3::ZERO, Vec3::Y)]));
                }
                _ => {}
            }
        }

        let transform = global
            .map(|g| g.0)
            .or_else(|| local.map(|t| t.to_affine()))
            .unwrap_or(Affine3A::IDENTITY);
        let tint = Vec4::from_array(emitter.tint);
        let mut extra: Vec<(Option<String>, u32)> = commands_list
            .iter()
            .filter_map(|c| match c {
                ParticleCommand::Burst {
                    entity: e,
                    emitter,
                    count,
                } if *e == entity => Some((emitter.clone(), *count)),
                _ => None,
            })
            .collect();

        let mut steps: Vec<f32> = Vec::new();
        if !state.prewarmed {
            state.prewarmed = true;
            let mut remaining = effect.prewarm.max(0.0);
            while remaining > 1e-4 {
                let step = remaining.min(PREWARM_STEP);
                steps.push(step);
                remaining -= step;
            }
        }
        if emitter.playing {
            steps.push(dt * emitter.time_scale.max(0.0));
        }

        for step_dt in steps {
            if step_dt <= 0.0 && extra.is_empty() {
                continue;
            }
            let start_time = state.time;
            state.time += step_dt;
            let mut subs: Vec<SubSpawn> = Vec::new();
            let raycast_ref: &dyn Fn(Vec3, Vec3) -> Option<(Vec3, Vec3)> = &raycast;
            for (index, def) in effect.emitters.iter().enumerate() {
                let emitter_state = &mut state.emitters[index];
                let mut count = if def.sub_emitter_only {
                    0
                } else {
                    emitter_state.scheduler.step(
                        def.spawn.rate,
                        def.spawn.per_distance,
                        &def.spawn.bursts,
                        effect.duration,
                        effect.looping,
                        emitter.emitting,
                        Vec3::from(transform.translation),
                        step_dt,
                        emitter.intensity,
                    )
                };
                for (target, extra_count) in &extra {
                    if target.as_ref().is_none_or(|name| *name == def.name) {
                        count += extra_count;
                    }
                }
                let needs_mesh = matches!(def.shape, Shape::Mesh { .. });
                if needs_mesh && emitter_state.mesh_points.is_none() {
                    count = 0;
                }
                match emitter_state.backend {
                    ActiveBackend::Cpu => {
                        let points = emitter_state.mesh_points.clone();
                        emitter_state.cpu.spawn(
                            def,
                            points.as_ref(),
                            &transform,
                            count,
                            None,
                            tint,
                        );
                        let ctx = StepContext {
                            dt: step_dt,
                            time: start_time,
                            gravity,
                            transform,
                            raycast: Some(raycast_ref),
                        };
                        emitter_state.cpu.update(def, &effect, &ctx, &mut subs);
                        emitter_state.alive = emitter_state.cpu.particles.len() as u32;
                    }
                    ActiveBackend::Gpu => {
                        queue_gpu_step(
                            emitter_state,
                            GpuStep {
                                spawn: count.min(emitter_state.capacity),
                                dt: step_dt,
                                time: start_time,
                            },
                        );
                    }
                }
            }
            extra.clear();
            for sub in subs {
                let (Some(def), Some(target)) = (
                    effect.emitters.get(sub.emitter),
                    state.emitters.get_mut(sub.emitter),
                ) else {
                    continue;
                };
                if target.backend != ActiveBackend::Cpu {
                    continue;
                }
                let points = target.mesh_points.clone();
                target.cpu.spawn(
                    def,
                    points.as_ref(),
                    &transform,
                    sub.count,
                    Some((sub.position, sub.velocity)),
                    tint,
                );
                target.alive = target.cpu.particles.len() as u32;
            }
        }

        // Completion of one-shot effects.
        if effect.duration > 0.0 && !effect.looping && !state.finished {
            let longest = effect
                .emitters
                .iter()
                .map(|e| e.init.lifetime.max)
                .fold(0.0, f32::max);
            let cpu_idle = state
                .emitters
                .iter()
                .all(|e| e.backend == ActiveBackend::Gpu || e.cpu.particles.is_empty());
            // GPU particle counts are not read back: wait out the longest
            // lifetime instead.
            let gpu_done = state
                .emitters
                .iter()
                .all(|e| e.backend == ActiveBackend::Cpu)
                || state.time >= effect.duration + longest;
            if state.time >= effect.duration && cpu_idle && gpu_done {
                state.finished = true;
                if emitter.despawn_when_finished {
                    commands.entity(entity).despawn();
                }
            }
        }
    }
}
