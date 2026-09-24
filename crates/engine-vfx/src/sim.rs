//! Simulation shared by both backends: deterministic random numbers and
//! noise (mirrored in `shaders/particles_common.wgsl`), spawn scheduling,
//! shape sampling, and the CPU simulator.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use engine_math::{Affine3A, Quat, Vec3, Vec4};

use crate::effect::{
    sample_lut, Burst, EmitterDef, Module, ParticleEffect, Shape, SimulationSpace,
    SubEmitterTrigger, VelocityDirection, LUT_SIZE,
};

/// PCG hash (same constants as the WGSL version).
pub fn pcg(value: u32) -> u32 {
    let state = value.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28).wrapping_add(4))) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

/// Uniform float in `[0, 1)` from `seed` and a channel.
pub fn random(seed: u32, channel: u32) -> f32 {
    (pcg(seed ^ pcg(channel.wrapping_add(0x9E37_79B9))) >> 8) as f32 / (1u32 << 24) as f32
}

fn lattice(x: i32, y: i32, z: i32, seed: u32) -> f32 {
    let h = pcg((x as u32).wrapping_mul(73_856_093)
        ^ (y as u32).wrapping_mul(19_349_663)
        ^ (z as u32).wrapping_mul(83_492_791)
        ^ seed);
    (h >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
}

/// Smooth value noise in [-1, 1].
pub fn value_noise(p: Vec3, seed: u32) -> f32 {
    let i = p.floor();
    let f = p - i;
    let u = f * f * (Vec3::splat(3.0) - 2.0 * f);
    let (x, y, z) = (i.x as i32, i.y as i32, i.z as i32);
    let mut result = 0.0;
    for dz in 0..2 {
        for dy in 0..2 {
            for dx in 0..2 {
                let w = (if dx == 1 { u.x } else { 1.0 - u.x })
                    * (if dy == 1 { u.y } else { 1.0 - u.y })
                    * (if dz == 1 { u.z } else { 1.0 - u.z });
                result += w * lattice(x + dx, y + dy, z + dz, seed);
            }
        }
    }
    result
}

/// Divergence-free noise: the curl of a noise vector potential.
pub fn curl_noise(p: Vec3) -> Vec3 {
    const E: f32 = 0.01;
    let potential = |q: Vec3| {
        Vec3::new(
            value_noise(q, 11),
            value_noise(q + Vec3::new(31.4, 17.1, 5.9), 23),
            value_noise(q + Vec3::new(-9.2, 44.7, 13.3), 37),
        )
    };
    let dx = Vec3::new(E, 0.0, 0.0);
    let dy = Vec3::new(0.0, E, 0.0);
    let dz = Vec3::new(0.0, 0.0, E);
    let (px0, px1) = (potential(p - dx), potential(p + dx));
    let (py0, py1) = (potential(p - dy), potential(p + dy));
    let (pz0, pz1) = (potential(p - dz), potential(p + dz));
    Vec3::new(
        (py1.z - py0.z) - (pz1.y - pz0.y),
        (pz1.x - pz0.x) - (px1.z - px0.z),
        (px1.y - px0.y) - (py1.x - py0.x),
    ) / (2.0 * E)
}

/// One particle as uploaded to the GPU (both backends).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GpuParticle {
    /// xyz position, w age.
    pub position_age: [f32; 4],
    /// xyz velocity, w lifetime.
    pub velocity_lifetime: [f32; 4],
    /// Initial linear RGBA.
    pub color: [f32; 4],
    /// Initial size, rotation, angular velocity, seed (bit pattern).
    pub size_rotation: [f32; 4],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CpuParticle {
    pub position: Vec3,
    pub velocity: Vec3,
    pub age: f32,
    pub lifetime: f32,
    pub size: f32,
    pub rotation: f32,
    pub angular_velocity: f32,
    pub color: Vec4,
    pub seed: u32,
}

impl CpuParticle {
    pub fn to_gpu(&self) -> GpuParticle {
        GpuParticle {
            position_age: self.position.extend(self.age).to_array(),
            velocity_lifetime: self.velocity.extend(self.lifetime).to_array(),
            color: self.color.to_array(),
            size_rotation: [
                self.size,
                self.rotation,
                self.angular_velocity,
                f32::from_bits(self.seed),
            ],
        }
    }
}

/// Burst firings whose time lies in `[start, end)`, repeating every
/// `duration` seconds for looping effects.
pub fn burst_firings(burst: &Burst, start: f32, end: f32, duration: f32, looping: bool) -> u32 {
    if end <= start {
        return 0;
    }
    let interval = burst.interval.max(0.0);
    let cycles = match (burst.cycles, interval > 0.0) {
        (0, true) => u32::MAX,
        (0, false) => 1,
        (n, _) => n,
    };
    let cycle_length = if duration > 0.0 && looping {
        duration
    } else {
        f32::INFINITY
    };
    let (first, last) = if cycle_length.is_finite() {
        (
            (start / cycle_length).floor() as i64,
            (end / cycle_length).floor() as i64,
        )
    } else {
        (0, 0)
    };
    let mut total: u64 = 0;
    for cycle in first..=last {
        let base = if cycle_length.is_finite() {
            cycle as f32 * cycle_length
        } else {
            0.0
        };
        let cycle_end = base + cycle_length;
        let origin = base + burst.time;
        if interval <= 0.0 {
            if origin >= start && origin < end && origin < cycle_end {
                total += 1;
            }
            continue;
        }
        let k_min = ((start - origin) / interval).ceil().max(0.0) as u64;
        let upper = end.min(cycle_end);
        let k_end = ((upper - origin) / interval).ceil().max(0.0) as u64;
        let k_end = k_end.min(cycles as u64);
        total += k_end.saturating_sub(k_min);
    }
    total.min(u32::MAX as u64) as u32
}

/// How many particles to spawn per frame (rate, distance, bursts).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SpawnScheduler {
    accumulator: f32,
    time: f32,
    last_position: Option<Vec3>,
}

impl SpawnScheduler {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn time(&self) -> f32 {
        self.time
    }

    /// Particles to spawn over `dt` seconds while `emitting`, with the
    /// effect cycle `duration` (0 = continuous) and emitter `position`.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        rate: f32,
        per_distance: f32,
        bursts: &[Burst],
        duration: f32,
        looping: bool,
        emitting: bool,
        position: Vec3,
        dt: f32,
        intensity: f32,
    ) -> u32 {
        let start = self.time;
        self.time += dt;
        let moved = self
            .last_position
            .map_or(0.0, |last| last.distance(position));
        self.last_position = Some(position);
        let finished = duration > 0.0 && !looping && start >= duration;
        if !emitting || finished {
            return 0;
        }
        let intensity = intensity.max(0.0);
        self.accumulator += (rate * dt + per_distance * moved) * intensity;
        let mut count = self.accumulator.floor();
        self.accumulator -= count;
        for burst in bursts {
            let fired = burst_firings(burst, start, self.time, duration, looping);
            count += (fired as f32 * burst.count as f32 * intensity).round();
        }
        count.min(u32::MAX as f32) as u32
    }
}

/// Pre-sampled mesh surface points (position, normal) for `Shape::Mesh`.
pub type MeshPoints = Arc<Vec<(Vec3, Vec3)>>;

/// Area-weighted surface samples of a triangle mesh.
pub fn sample_mesh_surface(
    positions: &[Vec3],
    indices: &[u32],
    count: usize,
    seed: u32,
) -> Vec<(Vec3, Vec3)> {
    let triangles: Vec<[Vec3; 3]> = indices
        .chunks_exact(3)
        .filter_map(|t| {
            Some([
                *positions.get(t[0] as usize)?,
                *positions.get(t[1] as usize)?,
                *positions.get(t[2] as usize)?,
            ])
        })
        .collect();
    let areas: Vec<f32> = triangles
        .iter()
        .map(|[a, b, c]| (*b - *a).cross(*c - *a).length() * 0.5)
        .collect();
    let total: f32 = areas.iter().sum();
    if triangles.is_empty() || total <= 0.0 {
        return vec![(Vec3::ZERO, Vec3::Y)];
    }
    let mut cumulative = Vec::with_capacity(areas.len());
    let mut sum = 0.0;
    for area in &areas {
        sum += area / total;
        cumulative.push(sum);
    }
    (0..count as u32)
        .map(|i| {
            let r = random(seed ^ 0xA511_E9B3, i * 3);
            let index = cumulative
                .partition_point(|c| *c < r)
                .min(triangles.len() - 1);
            let [a, b, c] = triangles[index];
            let (mut u, mut v) = (random(seed, i * 3 + 1), random(seed, i * 3 + 2));
            if u + v > 1.0 {
                u = 1.0 - u;
                v = 1.0 - v;
            }
            let normal = (b - a).cross(c - a).normalize_or_zero();
            (a + (b - a) * u + (c - a) * v, normal)
        })
        .collect()
}

fn unit_sphere(seed: u32, channel: u32) -> Vec3 {
    let z = random(seed, channel) * 2.0 - 1.0;
    let angle = random(seed, channel + 1) * std::f32::consts::TAU;
    let r = (1.0 - z * z).max(0.0).sqrt();
    Vec3::new(r * angle.cos(), z, r * angle.sin())
}

/// Local position and outward direction of a new particle.
pub fn sample_shape(shape: &Shape, points: Option<&MeshPoints>, seed: u32) -> (Vec3, Vec3) {
    match shape {
        Shape::Point => (Vec3::ZERO, unit_sphere(seed, 10)),
        Shape::Sphere { radius, surface } => {
            let direction = unit_sphere(seed, 10);
            let distance = if *surface {
                *radius
            } else {
                radius * random(seed, 12).cbrt()
            };
            (direction * distance, direction)
        }
        Shape::Cone { angle, radius } => {
            let spin = random(seed, 10) * std::f32::consts::TAU;
            let spread = random(seed, 11).sqrt();
            let tilt = angle.clamp(0.0, std::f32::consts::PI) * spread;
            let direction = Vec3::new(tilt.sin() * spin.cos(), tilt.cos(), tilt.sin() * spin.sin());
            let base = Vec3::new(spin.cos(), 0.0, spin.sin()) * radius * spread;
            (base, direction)
        }
        Shape::Box { half_extents } => {
            let p =
                Vec3::new(random(seed, 10), random(seed, 11), random(seed, 12)) * 2.0 - Vec3::ONE;
            (p * *half_extents, Vec3::Y)
        }
        Shape::Mesh { .. } => match points.filter(|p| !p.is_empty()) {
            Some(points) => {
                let index = (random(seed, 10) * points.len() as f32) as usize % points.len();
                points[index]
            }
            None => (Vec3::ZERO, Vec3::Y),
        },
    }
}

/// Scene inputs for one simulation step.
pub struct StepContext<'a> {
    pub dt: f32,
    pub time: f32,
    pub gravity: Vec3,
    /// Emitter local → world.
    pub transform: Affine3A,
    /// Casts `from → to` (world); returns hit point and normal.
    pub raycast: Option<&'a dyn Fn(Vec3, Vec3) -> Option<(Vec3, Vec3)>>,
}

/// A request for a sub-emitter burst (world space).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubSpawn {
    pub emitter: usize,
    pub position: Vec3,
    pub velocity: Vec3,
    pub count: u32,
}

/// CPU simulation of one emitter.
#[derive(Clone, Debug, Default)]
pub struct CpuEmitter {
    pub particles: Vec<CpuParticle>,
    next_seed: u32,
    pub capacity: usize,
}

impl CpuEmitter {
    pub fn new(capacity: usize, seed: u32) -> Self {
        Self {
            particles: Vec::with_capacity(capacity.min(4096)),
            next_seed: seed,
            capacity,
        }
    }

    pub fn clear(&mut self) {
        self.particles.clear();
    }

    /// Spawns `count` particles from the emitter shape (plus optional
    /// explicit origins in world space for sub-emitters).
    pub fn spawn(
        &mut self,
        def: &EmitterDef,
        points: Option<&MeshPoints>,
        transform: &Affine3A,
        count: u32,
        origin: Option<(Vec3, Vec3)>,
        tint: Vec4,
    ) -> usize {
        let world = def.space == SimulationSpace::World;
        let rotation = transform.to_scale_rotation_translation().1;
        let mut spawned = 0;
        for _ in 0..count {
            if self.particles.len() >= self.capacity {
                break;
            }
            let seed = pcg(self.next_seed);
            self.next_seed = self.next_seed.wrapping_add(1);
            let (local_position, outward) = sample_shape(&def.shape, points, seed);
            let direction = match def.init.direction {
                VelocityDirection::Shape => outward,
                VelocityDirection::Direction(d) => d.normalize_or_zero(),
            };
            let speed = def.init.speed.lerp(random(seed, 1));
            let (mut position, mut velocity) = (local_position, direction * speed);
            if world {
                position = transform.transform_point3(position);
                velocity = rotation * velocity;
            }
            if let Some((origin, inherited)) = origin {
                // Sub-emitter: shape relative to the parent particle.
                position = if world {
                    origin + (rotation * local_position)
                } else {
                    transform.inverse().transform_point3(origin) + local_position
                };
                velocity += if world {
                    inherited
                } else {
                    rotation.inverse() * inherited
                };
            }
            self.particles.push(CpuParticle {
                position,
                velocity,
                age: 0.0,
                lifetime: def.init.lifetime.lerp(random(seed, 2)).max(1e-3),
                size: def.init.size.lerp(random(seed, 3)),
                rotation: def.init.rotation.lerp(random(seed, 4)),
                angular_velocity: def.init.angular_velocity.lerp(random(seed, 5)),
                color: def.init.color * tint,
                seed,
            });
            spawned += 1;
        }
        spawned
    }

    /// Advances every particle; returns sub-emitter requests.
    pub fn update(
        &mut self,
        def: &EmitterDef,
        effect: &ParticleEffect,
        ctx: &StepContext<'_>,
        subs: &mut Vec<SubSpawn>,
    ) {
        let dt = ctx.dt;
        let world = def.space == SimulationSpace::World;
        let to_world = |p: Vec3| {
            if world {
                p
            } else {
                ctx.transform.transform_point3(p)
            }
        };
        let rotation = ctx.transform.to_scale_rotation_translation().1;
        let gravity = if world {
            ctx.gravity
        } else {
            rotation.inverse() * ctx.gravity
        };
        let speed_lut = def.speed_lut();
        let has_speed_curve = def
            .modules
            .iter()
            .any(|m| matches!(m, Module::SpeedOverLife(_)));
        let death_subs: Vec<(usize, u32, f32)> = def
            .sub_emitters
            .iter()
            .filter(|s| s.trigger == SubEmitterTrigger::Death)
            .filter_map(|s| {
                Some((
                    effect.emitter_index(&s.emitter)?,
                    s.count,
                    s.inherit_velocity,
                ))
            })
            .collect();

        let mut index = 0;
        while index < self.particles.len() {
            let particle = &mut self.particles[index];
            particle.age += dt;
            if particle.age >= particle.lifetime {
                let dead = self.particles.swap_remove(index);
                for (emitter, count, inherit) in &death_subs {
                    subs.push(SubSpawn {
                        emitter: *emitter,
                        position: to_world(dead.position),
                        velocity: (if world {
                            dead.velocity
                        } else {
                            rotation * dead.velocity
                        }) * *inherit,
                        count: *count,
                    });
                }
                continue;
            }
            let life = particle.age / particle.lifetime;
            let mut kill = false;
            for module in &def.modules {
                match module {
                    Module::Gravity(scale) => particle.velocity += gravity * *scale * dt,
                    Module::Acceleration(a) => particle.velocity += *a * dt,
                    Module::Drag(k) => particle.velocity *= (1.0 - k * dt).max(0.0),
                    Module::CurlNoise {
                        strength,
                        frequency,
                        scroll_speed,
                    } => {
                        let p =
                            particle.position * *frequency + Vec3::splat(ctx.time * scroll_speed);
                        particle.velocity += curl_noise(p) * *strength * dt;
                    }
                    Module::ColorOverLife(_)
                    | Module::SizeOverLife(_)
                    | Module::SpeedOverLife(_) => {}
                    Module::Collision { .. } => {}
                }
            }
            let speed_scale = if has_speed_curve {
                sample_lut(&speed_lut, life)
            } else {
                1.0
            };
            let step = particle.velocity * speed_scale * dt;
            let next = particle.position + step;
            let collision = def.modules.iter().find_map(|m| match m {
                Module::Collision {
                    bounce,
                    friction,
                    kill,
                } => Some((*bounce, *friction, *kill)),
                _ => None,
            });
            match (collision, ctx.raycast) {
                (Some((bounce, friction, kill_on_hit)), Some(raycast))
                    if step.length_squared() > 0.0 =>
                {
                    match raycast(to_world(particle.position), to_world(next)) {
                        Some((point, normal)) => {
                            if kill_on_hit {
                                kill = true;
                            } else {
                                let normal = if world {
                                    normal
                                } else {
                                    rotation.inverse() * normal
                                };
                                let v = particle.velocity;
                                let normal_part = normal * v.dot(normal);
                                let tangent = v - normal_part;
                                particle.velocity = tangent * (1.0 - friction.clamp(0.0, 1.0))
                                    - normal_part * bounce;
                                let point = if world {
                                    point
                                } else {
                                    ctx.transform.inverse().transform_point3(point)
                                };
                                particle.position = point + normal * 1e-3;
                            }
                        }
                        None => particle.position = next,
                    }
                }
                _ => particle.position = next,
            }
            particle.rotation += particle.angular_velocity * dt;
            if kill {
                self.particles.swap_remove(index);
                continue;
            }
            index += 1;
        }
    }

    /// Particles as GPU records, sorted back to front for `camera` when
    /// requested (positions stay in the emitter's simulation space).
    pub fn gpu_records(
        &self,
        sort_from: Option<Vec3>,
        to_world: &Affine3A,
        local: bool,
    ) -> Vec<GpuParticle> {
        let mut records: Vec<(f32, GpuParticle)> = self
            .particles
            .iter()
            .map(|p| {
                let distance = sort_from.map_or(0.0, |camera| {
                    let world = if local {
                        to_world.transform_point3(p.position)
                    } else {
                        p.position
                    };
                    world.distance_squared(camera)
                });
                (distance, p.to_gpu())
            })
            .collect();
        if sort_from.is_some() {
            records.sort_by(|a, b| b.0.total_cmp(&a.0));
        }
        records.into_iter().map(|(_, r)| r).collect()
    }
}

/// Number of over-life LUT entries (re-exported for the GPU uniform).
pub const LUT: usize = LUT_SIZE;

/// Rotation helper for tests and tools.
pub fn yaw(angle: f32) -> Quat {
    Quat::from_rotation_y(angle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effect::{EmitterDef, Range};

    #[test]
    fn random_is_deterministic_and_uniformish() {
        assert_eq!(random(42, 1), random(42, 1));
        assert_ne!(random(42, 1), random(42, 2));
        let mean: f32 = (0..10_000).map(|i| random(i, 0)).sum::<f32>() / 10_000.0;
        assert!((mean - 0.5).abs() < 0.02, "{mean}");
        assert!((0..1000).all(|i| (0.0..1.0).contains(&random(i, 7))));
    }

    #[test]
    fn curl_noise_is_nearly_divergence_free() {
        let p = Vec3::new(0.3, 1.7, -2.2);
        let e = 0.01;
        let div = (curl_noise(p + Vec3::X * e).x - curl_noise(p - Vec3::X * e).x
            + curl_noise(p + Vec3::Y * e).y
            - curl_noise(p - Vec3::Y * e).y
            + curl_noise(p + Vec3::Z * e).z
            - curl_noise(p - Vec3::Z * e).z)
            / (2.0 * e);
        let magnitude = curl_noise(p).length();
        assert!(magnitude > 0.01);
        assert!(
            div.abs() < magnitude * 2.0 + 0.5,
            "div {div} vs {magnitude}"
        );
    }

    #[test]
    fn scheduler_handles_rate_bursts_and_cycles() {
        let mut scheduler = SpawnScheduler::default();
        let bursts = [Burst {
            time: 0.0,
            count: 10,
            cycles: 1,
            interval: 0.0,
        }];
        let mut total = 0;
        for _ in 0..60 {
            total += scheduler.step(
                30.0,
                0.0,
                &bursts,
                0.0,
                true,
                true,
                Vec3::ZERO,
                1.0 / 60.0,
                1.0,
            );
        }
        assert_eq!(total, 40, "one second at 30/s plus a burst of 10");

        let mut scheduler = SpawnScheduler::default();
        let looping = [Burst {
            time: 0.5,
            count: 5,
            cycles: 1,
            interval: 0.0,
        }];
        let mut total = 0;
        for _ in 0..30 {
            total += scheduler.step(0.0, 0.0, &looping, 1.0, true, true, Vec3::ZERO, 0.1, 1.0);
        }
        assert_eq!(total, 15, "a burst per one-second cycle over three seconds");

        let mut scheduler = SpawnScheduler::default();
        let moved: u32 = (0..10)
            .map(|i| {
                scheduler.step(
                    0.0,
                    2.0,
                    &[],
                    0.0,
                    true,
                    true,
                    Vec3::new(i as f32, 0.0, 0.0),
                    0.1,
                    1.0,
                )
            })
            .sum();
        assert_eq!(moved, 18, "two per meter over nine meters");
    }

    #[test]
    fn cpu_simulation_is_deterministic_and_applies_modules() {
        let mut def = EmitterDef::new("dust", 100);
        def.init.lifetime = Range::new(1.0, 1.0);
        def.init.speed = Range::constant(0.0);
        def.modules = vec![Module::Gravity(1.0), Module::Drag(0.0)];
        let run = || {
            let mut sim = CpuEmitter::new(100, 7);
            let transform = Affine3A::from_translation(Vec3::new(0.0, 5.0, 0.0));
            sim.spawn(&def, None, &transform, 50, None, Vec4::ONE);
            let effect = ParticleEffect {
                emitters: vec![def.clone()],
                ..ParticleEffect::template()
            };
            let ctx = StepContext {
                dt: 0.1,
                time: 0.0,
                gravity: Vec3::new(0.0, -10.0, 0.0),
                transform,
                raycast: None,
            };
            let mut subs = Vec::new();
            for _ in 0..5 {
                sim.update(&def, &effect, &ctx, &mut subs);
            }
            sim
        };
        let (a, b) = (run(), run());
        assert_eq!(a.particles, b.particles);
        assert_eq!(a.particles.len(), 50, "capacity respected, none died yet");
        let p = a.particles[0];
        assert!((p.velocity.y + 5.0).abs() < 1e-4, "{:?}", p.velocity);
        assert!(p.position.y < 5.0);
        assert_eq!(a.gpu_records(None, &Affine3A::IDENTITY, false).len(), 50);
    }

    #[test]
    fn collision_bounces_and_death_spawns_sub_emitters() {
        let mut def = EmitterDef::new("ball", 4);
        def.init.lifetime = Range::constant(0.35);
        def.init.speed = Range::constant(0.0);
        def.modules = vec![
            Module::Gravity(1.0),
            Module::Collision {
                bounce: 1.0,
                friction: 0.0,
                kill: false,
            },
        ];
        def.sub_emitters.push(crate::effect::SubEmitter {
            trigger: SubEmitterTrigger::Death,
            emitter: "spark".into(),
            count: 3,
            inherit_velocity: 1.0,
        });
        let mut effect = ParticleEffect::template();
        effect.emitters = vec![def.clone(), EmitterDef::new("spark", 16)];
        let floor = |from: Vec3, to: Vec3| {
            (to.y < 0.0 && from.y >= 0.0).then(|| {
                let t = from.y / (from.y - to.y);
                (from + (to - from) * t, Vec3::Y)
            })
        };
        let transform = Affine3A::from_translation(Vec3::new(0.0, 0.5, 0.0));
        let mut sim = CpuEmitter::new(4, 1);
        sim.spawn(&def, None, &transform, 1, None, Vec4::ONE);
        let ctx = StepContext {
            dt: 0.05,
            time: 0.0,
            gravity: Vec3::new(0.0, -10.0, 0.0),
            transform,
            raycast: Some(&floor),
        };
        let mut subs = Vec::new();
        for _ in 0..6 {
            sim.update(&def, &effect, &ctx, &mut subs);
        }
        let p = sim.particles[0];
        assert!(p.position.y >= 0.0, "never below the floor: {p:?}");
        assert!(p.velocity.y > 0.0, "bounced upwards: {p:?}");
        for _ in 0..2 {
            sim.update(&def, &effect, &ctx, &mut subs);
        }
        assert!(sim.particles.is_empty());
        assert_eq!(subs.len(), 1);
        assert_eq!((subs[0].emitter, subs[0].count), (1, 3));
    }

    #[test]
    fn mesh_surface_samples_lie_on_triangles() {
        let positions = [Vec3::ZERO, Vec3::X, Vec3::Z];
        let samples = sample_mesh_surface(&positions, &[0, 1, 2], 64, 3);
        assert_eq!(samples.len(), 64);
        assert!(samples
            .iter()
            .all(|(p, n)| p.y.abs() < 1e-6 && p.x + p.z <= 1.0 + 1e-5 && n.y.abs() > 0.99));
    }
}
