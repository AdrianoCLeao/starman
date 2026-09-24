//! Audio: clips, a mixer asset (bus tree, sends, effects, snapshots),
//! spatial sources, a listener and zones, driven through a pluggable
//! [`AudioBackend`] — kira for devices, [`NullBackend`] for CI and
//! headless runs (ADR 0016).

mod backend;
mod clip;
mod components;
mod engine;
mod kira_backend;
mod mixer;
mod null;
mod plugin;
mod settings;
mod systems;

pub use backend::{
    attenuation, AudioBackend, PlayRequest, Rolloff, SpatialParams, VoiceId, VoiceInfo, VoiceUpdate,
};
pub use clip::{encode_wav, AudioClip, AudioClipLoader};
pub use components::{
    AudioCommand, AudioFinished, AudioListener, AudioSource, AudioSourceState, AudioZone,
    AudioZoneState, OneShot, ZoneShape,
};
pub use engine::{
    ActiveSnapshot, AudioEngine, AudioMixerSource, VoiceRecord, VoiceStart, MIX_TWEEN_SECONDS,
    STEAL_FADE_SECONDS,
};
pub use kira_backend::KiraBackend;
pub use mixer::{
    gain_to_db, resolve_mix, AudioMixer, AudioMixerLoader, BusDef, EffectDef, EffectOverride,
    EffectParam, EqKind, FilterKind, MixState, SendDef, SendLevel, SnapshotDef,
    AUDIO_MIXER_VERSION, MASTER, SILENCE_DB,
};
pub use null::NullBackend;
pub use plugin::{open_device_backend, AudioPlugin};
pub use settings::{
    AudioUserSettings, AudioUserSettingsStore, AUDIO_USER_SETTINGS_VERSION, SAVE_DEBOUNCE_SECONDS,
};
pub use systems::{spatial_gain, ListenerPose, PendingOneShots, ONE_SHOT_LOAD_TIMEOUT};

/// Stable module identifier for logs and reports.
pub fn module_name() -> &'static str {
    "engine-audio"
}
