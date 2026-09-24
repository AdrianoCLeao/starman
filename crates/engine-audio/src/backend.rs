//! The backend seam: the ECS side resolves *what* should sound (voices,
//! mix, listener); an [`AudioBackend`] turns that into output. Kira drives
//! real devices; [`NullBackend`](crate::NullBackend) is a deterministic
//! stand-in for CI, tests and headless playthroughs.

use std::sync::Arc;

use engine_math::{Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::clip::AudioClip;
use crate::mixer::{AudioMixer, MixState};

/// Backend-assigned voice identifier (never reused within a backend).
pub type VoiceId = u64;

/// How volume falls off between `min_distance` and `max_distance`.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, bevy_reflect::Reflect,
)]
pub enum Rolloff {
    Linear,
    /// Steep near the source, flat far away (natural-sounding default).
    #[default]
    Logarithmic,
    /// No distance attenuation (panning only).
    None,
}

/// Spatialization of one voice.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpatialParams {
    pub position: Vec3,
    pub min_distance: f32,
    pub max_distance: f32,
    pub rolloff: Rolloff,
    /// 0 = fully 2D, 1 = fully positional.
    pub spatial_blend: f32,
}

/// Distance attenuation as linear gain (shared by every backend and by
/// debug views so they agree on audibility).
pub fn attenuation(distance: f32, min_distance: f32, max_distance: f32, rolloff: Rolloff) -> f32 {
    let min = min_distance.max(0.0);
    let max = max_distance.max(min + 1e-3);
    if distance <= min {
        return 1.0;
    }
    if distance >= max {
        return match rolloff {
            Rolloff::None => 1.0,
            _ => 0.0,
        };
    }
    let t = (distance - min) / (max - min);
    match rolloff {
        Rolloff::Linear => 1.0 - t,
        Rolloff::Logarithmic => (1.0 - t).powi(3),
        Rolloff::None => 1.0,
    }
}

/// Everything needed to start one voice.
#[derive(Clone, Debug)]
pub struct PlayRequest {
    pub clip: Arc<AudioClip>,
    pub bus: String,
    pub volume_db: f32,
    /// Playback rate (1 = original pitch).
    pub pitch: f32,
    pub looping: bool,
    /// Seconds into the clip to start from.
    pub start_offset: f32,
    /// Seconds to fade in.
    pub fade_in: f32,
    pub spatial: Option<SpatialParams>,
}

impl PlayRequest {
    pub fn new(clip: Arc<AudioClip>, bus: impl Into<String>) -> Self {
        Self {
            clip,
            bus: bus.into(),
            volume_db: 0.0,
            pitch: 1.0,
            looping: false,
            start_offset: 0.0,
            fade_in: 0.0,
            spatial: None,
        }
    }
}

/// Changes to a playing voice; `None` fields stay as they are.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VoiceUpdate {
    pub volume_db: Option<f32>,
    pub pitch: Option<f32>,
    pub position: Option<Vec3>,
}

/// A voice as the backend sees it (debugging, tests, the editor mixer).
#[derive(Clone, Debug, PartialEq)]
pub struct VoiceInfo {
    pub id: VoiceId,
    pub bus: String,
    pub volume_db: f32,
    pub pitch: f32,
    pub position: Option<Vec3>,
    pub paused: bool,
    /// Seconds played (wraps for loops).
    pub time: f32,
}

/// An audio output implementation.
pub trait AudioBackend: Send + Sync + 'static {
    fn name(&self) -> &'static str;

    /// (Re)builds buses, sends and effects. Voices already playing keep
    /// playing on the previous graph until they finish.
    fn configure(&mut self, mixer: &AudioMixer);

    /// Moves bus volumes, send levels and effect parameters towards `mix`
    /// over `fade` seconds. Only changed values need to reach the device.
    fn apply_mix(&mut self, mix: &MixState, fade: f32);

    fn set_listener(&mut self, position: Vec3, rotation: Quat);

    /// Starts a voice; `None` when the backend could not (resource limits,
    /// unknown bus is routed to master instead).
    fn play(&mut self, request: PlayRequest) -> Option<VoiceId>;

    fn update_voice(&mut self, id: VoiceId, update: VoiceUpdate, fade: f32);

    fn set_paused(&mut self, id: VoiceId, paused: bool, fade: f32);

    /// Stops after fading out over `fade` seconds.
    fn stop(&mut self, id: VoiceId, fade: f32);

    fn is_playing(&self, id: VoiceId) -> bool;

    /// Advances time and reaps finished voices.
    fn update(&mut self, dt: f32);

    fn voices(&self) -> Vec<VoiceInfo>;
}
