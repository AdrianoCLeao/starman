//! Audio components (sources, listener, zones), commands and events.

use bevy_ecs::prelude::*;
use bevy_reflect::Reflect;
use engine_assets::AssetRef;
use engine_math::Vec3;

use crate::backend::{Rolloff, VoiceId};

/// Plays a clip at (or, when not spatial, for) the entity.
#[derive(Component, Clone, Debug, PartialEq, Reflect, engine_reflect::RegisterReflect)]
pub struct AudioSource {
    pub clip: AssetRef,
    /// Mixer bus the voice routes to.
    pub bus: String,
    #[engine_reflect(range(min = -80.0, max = 12.0))]
    pub volume_db: f32,
    #[engine_reflect(range(min = 0.1, max = 4.0))]
    pub pitch: f32,
    /// Random ± dB applied per play.
    #[engine_reflect(range(min = 0.0, max = 12.0))]
    pub volume_variation_db: f32,
    /// Random ± pitch applied per play.
    #[engine_reflect(range(min = 0.0, max = 1.0))]
    pub pitch_variation: f32,
    pub looping: bool,
    /// Starts when the clip is loaded (and again after a scene reload).
    pub autoplay: bool,
    /// Seconds to fade in.
    #[engine_reflect(range(min = 0.0, max = 10.0))]
    pub fade_in: f32,
    /// Positional playback from the entity's world position.
    pub spatial: bool,
    #[engine_reflect(range(min = 0.0, max = 1000.0))]
    pub min_distance: f32,
    #[engine_reflect(range(min = 0.0, max = 1000.0))]
    pub max_distance: f32,
    pub rolloff: Rolloff,
    /// 0 = plain 2D, 1 = fully positional.
    #[engine_reflect(range(min = 0.0, max = 1.0))]
    pub spatial_blend: f32,
    /// Higher priorities steal voices from lower ones at the limits.
    pub priority: i32,
    /// Concurrent voices of this clip (0 = unlimited).
    pub max_instances: u32,
}

impl Default for AudioSource {
    fn default() -> Self {
        Self {
            clip: AssetRef::default(),
            bus: "sfx".to_owned(),
            volume_db: 0.0,
            pitch: 1.0,
            volume_variation_db: 0.0,
            pitch_variation: 0.0,
            looping: false,
            autoplay: false,
            fade_in: 0.0,
            spatial: true,
            min_distance: 1.0,
            max_distance: 30.0,
            rolloff: Rolloff::Logarithmic,
            spatial_blend: 1.0,
            priority: 0,
            max_instances: 0,
        }
    }
}

/// Where the player hears from; the first enabled one wins.
#[derive(Component, Clone, Debug, PartialEq, Reflect, engine_reflect::RegisterReflect)]
pub struct AudioListener {
    pub enabled: bool,
}

impl Default for AudioListener {
    fn default() -> Self {
        Self { enabled: true }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
pub enum ZoneShape {
    Box { half_extents: Vec3 },
    Sphere { radius: f32 },
}

impl Default for ZoneShape {
    fn default() -> Self {
        Self::Box {
            half_extents: Vec3::splat(5.0),
        }
    }
}

impl ZoneShape {
    /// Distance from a local-space point to the shape (0 inside).
    pub fn distance(&self, local: Vec3) -> f32 {
        match self {
            Self::Box { half_extents } => {
                (local.abs() - half_extents.abs()).max(Vec3::ZERO).length()
            }
            Self::Sphere { radius } => (local.length() - radius.abs()).max(0.0),
        }
    }
}

/// A region that, while the listener is in it, applies a mixer snapshot,
/// routes a bus into a reverb send and plays an ambient loop, all faded
/// by the listener's distance to the shape.
#[derive(Component, Clone, Debug, PartialEq, Reflect, engine_reflect::RegisterReflect)]
pub struct AudioZone {
    pub shape: ZoneShape,
    /// Blend distance outside the shape.
    #[engine_reflect(range(min = 0.0, max = 100.0))]
    pub fade_distance: f32,
    /// Mixer snapshot applied at the zone's weight (empty: none).
    pub snapshot: String,
    /// Send bus (e.g. `reverb`) that `send_bus` feeds while inside.
    pub reverb_send: String,
    pub send_bus: String,
    #[engine_reflect(range(min = -80.0, max = 12.0))]
    pub send_db: f32,
    /// Ambient loop (empty: none).
    pub ambient: AssetRef,
    pub ambient_bus: String,
    #[engine_reflect(range(min = -80.0, max = 12.0))]
    pub ambient_volume_db: f32,
    /// Order among overlapping zones (higher applies last).
    pub priority: i32,
}

impl Default for AudioZone {
    fn default() -> Self {
        Self {
            shape: ZoneShape::default(),
            fade_distance: 2.0,
            snapshot: String::new(),
            reverb_send: String::new(),
            send_bus: "sfx".to_owned(),
            send_db: -6.0,
            ambient: AssetRef::default(),
            ambient_bus: "ambience".to_owned(),
            ambient_volume_db: 0.0,
            priority: 0,
        }
    }
}

impl AudioZone {
    /// The shape scaled by a transform's scale.
    pub fn scaled_shape(&self, scale: Vec3) -> ZoneShape {
        match self.shape {
            ZoneShape::Box { half_extents } => ZoneShape::Box {
                half_extents: half_extents * scale.abs(),
            },
            ZoneShape::Sphere { radius } => ZoneShape::Sphere {
                radius: radius * scale.abs().max_element(),
            },
        }
    }

    /// Weight for a world-space point: 1 inside the (scaled) shape,
    /// fading linearly to 0 at `fade_distance` (world units) outside.
    pub fn weight_at(&self, transform: &engine_core::GlobalTransform, point: Vec3) -> f32 {
        let t = transform.compute_transform();
        let local = t.rotation.inverse() * (point - t.translation);
        self.weight_for_distance(self.scaled_shape(t.scale).distance(local))
    }

    fn weight_for_distance(&self, distance: f32) -> f32 {
        if distance <= 0.0 {
            1.0
        } else if self.fade_distance <= 0.0 {
            0.0
        } else {
            (1.0 - distance / self.fade_distance).clamp(0.0, 1.0)
        }
    }
}

/// A fire-and-forget sound.
#[derive(Clone, Debug, PartialEq)]
pub struct OneShot {
    pub clip: AssetRef,
    pub bus: String,
    pub volume_db: f32,
    pub pitch: f32,
    pub volume_variation_db: f32,
    pub pitch_variation: f32,
    /// World position (None: 2D).
    pub position: Option<Vec3>,
    pub min_distance: f32,
    pub max_distance: f32,
    pub rolloff: Rolloff,
    pub priority: i32,
    /// Echoed back in [`AudioFinished`].
    pub tag: Option<String>,
}

impl OneShot {
    pub fn new(clip: AssetRef, bus: impl Into<String>) -> Self {
        Self {
            clip,
            bus: bus.into(),
            volume_db: 0.0,
            pitch: 1.0,
            volume_variation_db: 0.0,
            pitch_variation: 0.0,
            position: None,
            min_distance: 1.0,
            max_distance: 30.0,
            rolloff: Rolloff::Logarithmic,
            priority: 0,
            tag: None,
        }
    }

    pub fn at(mut self, position: Vec3) -> Self {
        self.position = Some(position);
        self
    }
}

/// Imperative audio control (gameplay, scripts, UI).
#[derive(Event, Clone, Debug, PartialEq)]
pub enum AudioCommand {
    /// (Re)starts the entity's [`AudioSource`].
    Play(Entity),
    Stop {
        entity: Entity,
        fade: f32,
    },
    SetPaused {
        entity: Entity,
        paused: bool,
    },
    OneShot(OneShot),
    /// Fades a mixer snapshot in (stacked by priority).
    PushSnapshot(String),
    /// Fades a mixer snapshot out.
    PopSnapshot(String),
    /// User volume (linear 0..=1), persisted.
    SetUserVolume {
        bus: String,
        volume: f32,
    },
    SetMuted(bool),
    /// Pauses or resumes every voice on `bus` and its child buses.
    PauseBus {
        bus: String,
        paused: bool,
    },
    /// Stops every voice (on `bus` and children when given).
    StopAll {
        bus: Option<String>,
        fade: f32,
    },
}

/// A voice ended (clip finished or stopped).
#[derive(Event, Clone, Debug, PartialEq)]
pub struct AudioFinished {
    pub voice: VoiceId,
    pub entity: Option<Entity>,
    pub tag: Option<String>,
}

/// Runtime playback state of an [`AudioSource`] (not serialized).
#[derive(Component, Clone, Debug, Default)]
pub struct AudioSourceState {
    pub voice: Option<VoiceId>,
    /// Autoplay already happened for this clip.
    pub autoplayed: bool,
    /// Waiting for the clip to load before starting.
    pub pending_play: bool,
    pub paused: bool,
    /// Per-play random offsets.
    pub volume_offset_db: f32,
    pub pitch_scale: f32,
    pub(crate) clip_key: String,
}

/// Runtime state of an [`AudioZone`].
#[derive(Component, Clone, Debug, Default)]
pub struct AudioZoneState {
    pub weight: f32,
    pub ambient_voice: Option<VoiceId>,
}
