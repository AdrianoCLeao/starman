//! Per-user audio preferences (bus volumes, global mute) persisted as
//! `audio.ron` in the user settings directory, versioned with forward
//! migration like the input overrides.

use std::collections::BTreeMap;
use std::path::PathBuf;

use bevy_ecs::prelude::Resource;
use engine_core::{EngineError, Result};
use serde::{Deserialize, Serialize};

pub const AUDIO_USER_SETTINGS_VERSION: u32 = 1;

/// Seconds without changes before a dirty store is written (slider drags
/// produce a change per frame).
pub const SAVE_DEBOUNCE_SECONDS: f32 = 0.5;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AudioUserSettings {
    #[serde(default)]
    pub version: u32,
    /// Bus → linear volume (0..=1); absent buses play at 1.
    #[serde(default)]
    pub bus_volumes: BTreeMap<String, f32>,
    #[serde(default)]
    pub muted: bool,
}

impl Default for AudioUserSettings {
    fn default() -> Self {
        Self {
            version: AUDIO_USER_SETTINGS_VERSION,
            bus_volumes: BTreeMap::new(),
            muted: false,
        }
    }
}

impl AudioUserSettings {
    pub fn volume(&self, bus: &str) -> f32 {
        self.bus_volumes.get(bus).copied().unwrap_or(1.0)
    }

    /// Effective per-bus gains for the mixer (mute folds into master).
    pub fn gains(&self) -> BTreeMap<String, f32> {
        let mut gains = self.bus_volumes.clone();
        if self.muted {
            gains.insert(crate::mixer::MASTER.to_owned(), 0.0);
        }
        gains
    }

    pub fn parse(text: &str) -> std::result::Result<Self, String> {
        let mut settings: AudioUserSettings =
            ron::from_str(text).map_err(|error| format!("invalid audio settings: {error}"))?;
        match settings.version {
            0 | AUDIO_USER_SETTINGS_VERSION => {
                settings.version = AUDIO_USER_SETTINGS_VERSION;
                for volume in settings.bus_volumes.values_mut() {
                    *volume = if volume.is_finite() {
                        volume.clamp(0.0, 1.0)
                    } else {
                        1.0
                    };
                }
                Ok(settings)
            }
            other => Err(format!(
                "audio settings version {other} is newer than supported ({AUDIO_USER_SETTINGS_VERSION})"
            )),
        }
    }

    pub fn to_ron(&self) -> String {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default())
            .expect("audio settings serialize")
    }
}

/// The active audio preferences and where they persist.
#[derive(Resource, Clone, Debug, Default)]
pub struct AudioUserSettingsStore {
    pub settings: AudioUserSettings,
    pub path: Option<PathBuf>,
    pub dirty: bool,
    /// Seconds since the last change (drives the debounced save).
    pub idle: f32,
}

impl AudioUserSettingsStore {
    /// Loads from `path` (missing → defaults; corrupt → moved aside to
    /// `audio.ron.corrupt` and defaults used).
    pub fn load(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let settings = match std::fs::read_to_string(&path) {
            Ok(text) => AudioUserSettings::parse(&text).unwrap_or_else(|error| {
                log::warn!(
                    target: "engine::audio",
                    "ignoring '{}': {error}; using default volumes",
                    path.display()
                );
                let _ = std::fs::rename(&path, path.with_extension("ron.corrupt"));
                AudioUserSettings::default()
            }),
            Err(_) => AudioUserSettings::default(),
        };
        Self {
            settings,
            path: Some(path),
            dirty: false,
            idle: 0.0,
        }
    }

    pub fn set_volume(&mut self, bus: impl Into<String>, volume: f32) {
        let volume = if volume.is_finite() {
            volume.clamp(0.0, 1.0)
        } else {
            1.0
        };
        let bus = bus.into();
        if self.settings.bus_volumes.get(&bus) != Some(&volume) {
            self.settings.bus_volumes.insert(bus, volume);
            self.touch();
        }
    }

    pub fn set_muted(&mut self, muted: bool) {
        if self.settings.muted != muted {
            self.settings.muted = muted;
            self.touch();
        }
    }

    fn touch(&mut self) {
        self.dirty = true;
        self.idle = 0.0;
    }

    /// Advances the debounce timer; saves once changes settle.
    pub fn tick(&mut self, dt: f32) {
        if !self.dirty {
            return;
        }
        self.idle += dt;
        if self.idle >= SAVE_DEBOUNCE_SECONDS {
            if let Err(error) = self.save() {
                log::warn!(target: "engine::audio", "cannot save audio settings: {error}");
                self.dirty = false;
            }
        }
    }

    /// Writes atomically (temp file + rename).
    pub fn save(&mut self) -> Result<()> {
        let Some(path) = self.path.clone() else {
            self.dirty = false;
            return Ok(());
        };
        let io = |error: std::io::Error, path: &std::path::Path| EngineError::AssetLoad {
            path: path.display().to_string(),
            reason: error.to_string(),
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| io(e, parent))?;
        }
        engine_assets::import::write_atomic(&path, self.settings.to_ron().as_bytes())
            .map_err(|e| io(e, &path))?;
        self.dirty = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn volumes_clamp_migrate_and_persist_after_debounce() {
        let parsed = AudioUserSettings::parse("(bus_volumes: {\"music\": 3.0})").unwrap();
        assert_eq!(parsed.version, AUDIO_USER_SETTINGS_VERSION);
        assert_eq!(parsed.volume("music"), 1.0);
        assert_eq!(parsed.volume("sfx"), 1.0);
        assert!(AudioUserSettings::parse("(version: 7)").is_err());

        let dir = std::env::temp_dir().join(format!(
            "starman-audio-settings-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let path = dir.join("audio.ron");
        let mut store = AudioUserSettingsStore::load(&path);
        store.set_volume("music", 0.25);
        store.set_muted(true);
        store.tick(0.2);
        assert!(!path.exists(), "debounced");
        store.tick(0.4);
        assert!(!store.dirty);
        let loaded = AudioUserSettingsStore::load(&path);
        assert_eq!(loaded.settings.volume("music"), 0.25);
        assert_eq!(loaded.settings.gains()[crate::mixer::MASTER], 0.0);
        std::fs::write(&path, "(((").unwrap();
        assert_eq!(
            AudioUserSettingsStore::load(&path).settings,
            AudioUserSettings::default()
        );
        assert!(path.with_extension("ron.corrupt").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
