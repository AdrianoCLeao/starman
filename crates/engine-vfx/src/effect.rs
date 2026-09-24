//! The `ParticleEffect` asset (`*.vfx.ron`): one or more emitters, each a
//! stack of spawn, shape, init, update and render modules. The same
//! description drives both simulation backends.

use std::collections::HashSet;

use engine_assets::{Asset, AssetLoader, AssetRef, LoadContext};
use engine_core::Result;
use engine_math::{Curve, Interpolation, Vec3, Vec4};
use serde::{Deserialize, Serialize};

pub const PARTICLE_EFFECT_VERSION: u32 = 1;

/// Inclusive random range.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Range {
    pub min: f32,
    pub max: f32,
}

impl Range {
    pub const fn constant(value: f32) -> Self {
        Self {
            min: value,
            max: value,
        }
    }

    pub const fn new(min: f32, max: f32) -> Self {
        Self { min, max }
    }

    pub fn lerp(self, t: f32) -> f32 {
        self.min + (self.max - self.min) * t
    }
}

impl Default for Range {
    fn default() -> Self {
        Self::constant(1.0)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimulationBackend {
    /// GPU when the device supports compute, CPU otherwise (and when the
    /// emitter needs CPU-only features such as sub-emitters or physics
    /// collision).
    #[default]
    Auto,
    Gpu,
    Cpu,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimulationSpace {
    /// Particles move with the emitter entity.
    Local,
    /// Particles stay where they were spawned.
    #[default]
    World,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Burst {
    /// Seconds after the effect starts.
    pub time: f32,
    pub count: u32,
    /// Repetitions (0 = forever while looping).
    #[serde(default = "one_u32")]
    pub cycles: u32,
    #[serde(default)]
    pub interval: f32,
}

fn one_u32() -> u32 {
    1
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Spawn {
    /// Particles per second.
    #[serde(default)]
    pub rate: f32,
    /// Particles per meter the emitter travels.
    #[serde(default)]
    pub per_distance: f32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bursts: Vec<Burst>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum Shape {
    #[default]
    Point,
    Sphere {
        radius: f32,
        #[serde(default)]
        surface: bool,
    },
    /// Opening along +Y: `angle` (radians) from the axis, base `radius`.
    Cone {
        angle: f32,
        #[serde(default)]
        radius: f32,
    },
    Box {
        half_extents: Vec3,
    },
    /// Points on the surface of a mesh (area-weighted, pre-sampled).
    Mesh {
        mesh: AssetRef,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum VelocityDirection {
    /// Out of the shape (sphere normal, cone direction, mesh normal).
    #[default]
    Shape,
    /// A fixed local direction.
    Direction(Vec3),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Init {
    pub lifetime: Range,
    pub speed: Range,
    #[serde(default)]
    pub direction: VelocityDirection,
    pub size: Range,
    #[serde(default = "zero_range")]
    pub rotation: Range,
    #[serde(default = "zero_range")]
    pub angular_velocity: Range,
    /// Linear RGBA (HDR allowed).
    #[serde(default = "white")]
    pub color: Vec4,
}

fn zero_range() -> Range {
    Range::constant(0.0)
}

fn white() -> Vec4 {
    Vec4::ONE
}

impl Default for Init {
    fn default() -> Self {
        Self {
            lifetime: Range::new(1.0, 2.0),
            speed: Range::new(1.0, 2.0),
            direction: VelocityDirection::Shape,
            size: Range::new(0.1, 0.2),
            rotation: zero_range(),
            angular_velocity: zero_range(),
            color: Vec4::ONE,
        }
    }
}

/// Per-frame behaviour, applied in order.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Module {
    /// Scaled scene gravity (the physics world's gravity vector).
    Gravity(f32),
    /// A constant acceleration.
    Acceleration(Vec3),
    /// Linear drag coefficient (per second).
    Drag(f32),
    /// Divergence-free turbulence.
    CurlNoise {
        strength: f32,
        frequency: f32,
        #[serde(default)]
        scroll_speed: f32,
    },
    /// Multiplies the initial color over normalized age.
    ColorOverLife(Curve<Vec4>),
    /// Multiplies the initial size over normalized age.
    SizeOverLife(Curve<f32>),
    /// Multiplies velocity over normalized age.
    SpeedOverLife(Curve<f32>),
    /// Bounces off geometry: the depth buffer on the GPU, physics
    /// colliders on the CPU.
    Collision {
        #[serde(default = "half")]
        bounce: f32,
        #[serde(default)]
        friction: f32,
        /// Kill particles on their first contact.
        #[serde(default)]
        kill: bool,
    },
}

fn half() -> f32 {
    0.5
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BlendMode {
    Additive,
    #[default]
    Alpha,
    Premultiplied,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum RenderMode {
    /// Camera-facing quads.
    #[default]
    Billboard,
    /// Quads stretched along velocity (sparks, rain).
    Stretched { length_scale: f32 },
    /// Instanced mesh per particle.
    Mesh { mesh: AssetRef },
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Flipbook {
    pub columns: u32,
    pub rows: u32,
    /// Frames per second; 0 = spread the frames over the lifetime.
    #[serde(default)]
    pub fps: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Render {
    #[serde(default)]
    pub mode: RenderMode,
    #[serde(default)]
    pub blend: BlendMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub texture: Option<AssetRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flipbook: Option<Flipbook>,
    /// Soft-particle fade distance against scene depth (meters, 0 = off).
    #[serde(default)]
    pub soft_distance: f32,
    /// Sort back to front (alpha blending).
    #[serde(default)]
    pub sort: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubEmitterTrigger {
    Spawn,
    Death,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SubEmitter {
    pub trigger: SubEmitterTrigger,
    /// Name of another emitter of this effect.
    pub emitter: String,
    pub count: u32,
    /// Fraction of the parent's velocity added to the children.
    #[serde(default)]
    pub inherit_velocity: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmitterDef {
    pub name: String,
    #[serde(default)]
    pub backend: SimulationBackend,
    pub max_particles: u32,
    #[serde(default)]
    pub space: SimulationSpace,
    #[serde(default)]
    pub spawn: Spawn,
    #[serde(default)]
    pub shape: Shape,
    #[serde(default)]
    pub init: Init,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub modules: Vec<Module>,
    #[serde(default)]
    pub render: Render,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sub_emitters: Vec<SubEmitter>,
    /// Emitters fed only by sub-emitter events do not spawn on their own.
    #[serde(default)]
    pub sub_emitter_only: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParticleEffect {
    #[serde(default = "effect_version")]
    pub version: u32,
    pub emitters: Vec<EmitterDef>,
    /// Seconds of simulation run when the effect starts.
    #[serde(default)]
    pub prewarm: f32,
    /// Seconds per cycle; 0 = continuous.
    #[serde(default)]
    pub duration: f32,
    #[serde(default = "yes")]
    pub looping: bool,
    /// Local-space bounds used for culling (derived when absent).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bounds: Option<(Vec3, Vec3)>,
    #[serde(default)]
    pub seed: u32,
}

fn effect_version() -> u32 {
    PARTICLE_EFFECT_VERSION
}

fn yes() -> bool {
    true
}

impl Asset for ParticleEffect {
    const TYPE_NAME: &'static str = "ParticleEffect";
}

/// Number of samples in the over-life lookup tables shared by both
/// backends.
pub const LUT_SIZE: usize = 16;

impl EmitterDef {
    pub fn new(name: impl Into<String>, max_particles: u32) -> Self {
        Self {
            name: name.into(),
            backend: SimulationBackend::Auto,
            max_particles,
            space: SimulationSpace::World,
            spawn: Spawn::default(),
            shape: Shape::Point,
            init: Init::default(),
            modules: Vec::new(),
            render: Render::default(),
            sub_emitters: Vec::new(),
            sub_emitter_only: false,
        }
    }

    /// Whether this emitter can run on the GPU backend.
    pub fn gpu_capable(&self, effect: &ParticleEffect) -> bool {
        self.sub_emitters.is_empty()
            && !effect
                .emitters
                .iter()
                .any(|other| other.sub_emitters.iter().any(|s| s.emitter == self.name))
    }

    /// Color multiplier over normalized age, as a lookup table.
    pub fn color_lut(&self) -> [Vec4; LUT_SIZE] {
        let curve = self.modules.iter().find_map(|m| match m {
            Module::ColorOverLife(curve) => Some(curve),
            _ => None,
        });
        std::array::from_fn(|i| {
            let t = i as f32 / (LUT_SIZE - 1) as f32;
            curve.and_then(|c| c.sample(t)).unwrap_or(Vec4::ONE)
        })
    }

    fn scalar_lut(&self, pick: impl Fn(&Module) -> Option<&Curve<f32>>) -> [f32; LUT_SIZE] {
        let curve = self.modules.iter().find_map(pick);
        std::array::from_fn(|i| {
            let t = i as f32 / (LUT_SIZE - 1) as f32;
            curve.and_then(|c| c.sample(t)).unwrap_or(1.0)
        })
    }

    pub fn size_lut(&self) -> [f32; LUT_SIZE] {
        self.scalar_lut(|m| match m {
            Module::SizeOverLife(curve) => Some(curve),
            _ => None,
        })
    }

    pub fn speed_lut(&self) -> [f32; LUT_SIZE] {
        self.scalar_lut(|m| match m {
            Module::SpeedOverLife(curve) => Some(curve),
            _ => None,
        })
    }
}

/// Samples a lookup table at normalized age `t` (linear).
pub fn sample_lut<T>(lut: &[T; LUT_SIZE], t: f32) -> T
where
    T: Copy + std::ops::Mul<f32, Output = T> + std::ops::Add<Output = T>,
{
    let x = t.clamp(0.0, 1.0) * (LUT_SIZE - 1) as f32;
    let i = (x.floor() as usize).min(LUT_SIZE - 2);
    let f = x - i as f32;
    lut[i] * (1.0 - f) + lut[i + 1] * f
}

impl ParticleEffect {
    pub fn emitter_index(&self, name: &str) -> Option<usize> {
        self.emitters.iter().position(|e| e.name == name)
    }

    pub fn validate(&self) -> std::result::Result<(), Vec<String>> {
        let mut errors = Vec::new();
        if self.version > PARTICLE_EFFECT_VERSION {
            errors.push(format!(
                "effect version {} is newer than supported {PARTICLE_EFFECT_VERSION}",
                self.version
            ));
        }
        if self.emitters.is_empty() {
            errors.push("an effect needs at least one emitter".to_owned());
        }
        let mut names = HashSet::new();
        for emitter in &self.emitters {
            let context = &emitter.name;
            if !names.insert(emitter.name.as_str()) {
                errors.push(format!("emitter '{context}' is declared twice"));
            }
            if emitter.max_particles == 0 || emitter.max_particles > 1 << 20 {
                errors.push(format!("{context}: max_particles must be in 1..=1048576"));
            }
            if emitter.init.lifetime.min <= 0.0
                || emitter.init.lifetime.max < emitter.init.lifetime.min
            {
                errors.push(format!(
                    "{context}: lifetime must be positive with min <= max"
                ));
            }
            if emitter.spawn.rate < 0.0 || emitter.spawn.per_distance < 0.0 {
                errors.push(format!("{context}: spawn rates must be non-negative"));
            }
            if emitter.backend == SimulationBackend::Gpu && !emitter.gpu_capable(self) {
                errors.push(format!(
                    "{context}: sub-emitters run on the CPU backend; use Auto or Cpu"
                ));
            }
            for module in &emitter.modules {
                let curve_error = match module {
                    Module::ColorOverLife(c) => c.validate().err(),
                    Module::SizeOverLife(c) | Module::SpeedOverLife(c) => c.validate().err(),
                    _ => None,
                };
                if let Some(error) = curve_error {
                    errors.push(format!("{context}: {error}"));
                }
            }
            if let Some(flipbook) = emitter.render.flipbook {
                if flipbook.columns == 0 || flipbook.rows == 0 {
                    errors.push(format!(
                        "{context}: flipbook needs at least one column and row"
                    ));
                }
            }
            for sub in &emitter.sub_emitters {
                match self.emitter_index(&sub.emitter) {
                    None => errors.push(format!(
                        "{context}: sub-emitter '{}' does not exist",
                        sub.emitter
                    )),
                    Some(index) if self.emitters[index].name == emitter.name => {
                        errors.push(format!("{context}: an emitter cannot feed itself"))
                    }
                    _ => {}
                }
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    pub fn to_ron(&self) -> String {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default().struct_names(false))
            .unwrap_or_default()
    }

    /// A small default effect (editor "Create → Particle Effect").
    pub fn template() -> Self {
        let mut emitter = EmitterDef::new("sparks", 512);
        emitter.spawn.rate = 40.0;
        emitter.shape = Shape::Cone {
            angle: 0.4,
            radius: 0.05,
        };
        emitter.modules = vec![
            Module::Gravity(0.5),
            Module::Drag(0.5),
            Module::ColorOverLife(Curve {
                interpolation: Interpolation::Linear,
                times: vec![0.0, 1.0],
                values: vec![Vec4::new(1.0, 0.8, 0.4, 1.0), Vec4::new(1.0, 0.2, 0.0, 0.0)],
                in_tangents: vec![],
                out_tangents: vec![],
            }),
        ];
        emitter.render.blend = BlendMode::Additive;
        Self {
            version: PARTICLE_EFFECT_VERSION,
            emitters: vec![emitter],
            prewarm: 0.0,
            duration: 0.0,
            looping: true,
            bounds: None,
            seed: 0,
        }
    }
}

/// Loads and validates `*.vfx.ron`.
pub struct ParticleEffectLoader;

impl AssetLoader for ParticleEffectLoader {
    type Asset = ParticleEffect;

    fn extensions(&self) -> &'static [&'static str] {
        &["vfx.ron"]
    }

    fn load(&self, bytes: &[u8], ctx: &mut LoadContext<'_>) -> Result<ParticleEffect> {
        let text = std::str::from_utf8(bytes).map_err(|_| ctx.error("effect is not UTF-8"))?;
        let effect: ParticleEffect =
            ron::from_str(text).map_err(|error| ctx.error(format!("invalid effect: {error}")))?;
        effect
            .validate()
            .map_err(|errors| ctx.error(errors.join("; ")))?;
        Ok(effect)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_round_trips_and_validates() {
        let effect = ParticleEffect::template();
        effect.validate().unwrap();
        let parsed: ParticleEffect = ron::from_str(&effect.to_ron()).unwrap();
        assert_eq!(parsed, effect);
        let lut = effect.emitters[0].color_lut();
        assert_eq!(lut[0], Vec4::new(1.0, 0.8, 0.4, 1.0));
        assert_eq!(lut[LUT_SIZE - 1].w, 0.0);
        assert!((sample_lut(&lut, 0.5).w - 0.5).abs() < 1e-5);
    }

    #[test]
    fn validation_reports_problems_and_gpu_constraints() {
        let mut effect = ParticleEffect::template();
        let mut child = EmitterDef::new("smoke", 0);
        child.init.lifetime = Range::new(2.0, 1.0);
        effect.emitters[0].sub_emitters.push(SubEmitter {
            trigger: SubEmitterTrigger::Death,
            emitter: "smoke".into(),
            count: 2,
            inherit_velocity: 0.0,
        });
        effect.emitters[0].backend = SimulationBackend::Gpu;
        effect.emitters.push(child);
        let errors = effect.validate().unwrap_err().join("\n");
        assert!(errors.contains("max_particles"), "{errors}");
        assert!(errors.contains("lifetime"), "{errors}");
        assert!(errors.contains("sub-emitters run on the CPU"), "{errors}");
        assert!(
            !effect.emitters[1].gpu_capable(&effect),
            "fed by a sub-emitter"
        );
    }
}
