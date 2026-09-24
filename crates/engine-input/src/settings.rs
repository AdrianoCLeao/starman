//! Per-user input preferences: binding overrides, look sensitivity and
//! invert-Y, persisted as `input.ron` in the user settings directory.
//! Versioned with forward migration (ADR 0006).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use bevy_ecs::prelude::Resource;
use engine_core::{EngineError, Result};
use serde::{Deserialize, Serialize};

use crate::actions::{ActionDef, Binding};

pub const INPUT_USER_SETTINGS_VERSION: u32 = 1;

fn default_sensitivity() -> f32 {
    1.0
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InputUserSettings {
    #[serde(default)]
    pub version: u32,
    /// `"context/action"` → full replacement binding list.
    #[serde(default)]
    pub overrides: BTreeMap<String, Vec<Binding>>,
    #[serde(default = "default_sensitivity")]
    pub mouse_sensitivity: f32,
    #[serde(default = "default_sensitivity")]
    pub gamepad_look_sensitivity: f32,
    #[serde(default)]
    pub invert_y: bool,
}

impl Default for InputUserSettings {
    fn default() -> Self {
        Self {
            version: INPUT_USER_SETTINGS_VERSION,
            overrides: BTreeMap::new(),
            mouse_sensitivity: 1.0,
            gamepad_look_sensitivity: 1.0,
            invert_y: false,
        }
    }
}

pub fn override_key(context: &str, action: &str) -> String {
    format!("{context}/{action}")
}

impl InputUserSettings {
    /// Effective bindings of `def` in `context` (override or authored).
    pub fn bindings_for<'a>(&'a self, context: &str, def: &'a ActionDef) -> &'a [Binding] {
        self.overrides
            .get(&override_key(context, &def.name))
            .map(Vec::as_slice)
            .unwrap_or(&def.bindings)
    }

    pub fn set_bindings(&mut self, context: &str, action: &str, bindings: Vec<Binding>) {
        self.overrides
            .insert(override_key(context, action), bindings);
    }

    pub fn reset(&mut self, context: &str, action: &str) {
        self.overrides.remove(&override_key(context, action));
    }

    pub fn reset_all(&mut self) {
        self.overrides.clear();
    }

    /// Parses any supported version, migrating to the current one.
    pub fn parse(text: &str) -> std::result::Result<Self, String> {
        let mut settings: InputUserSettings =
            ron::from_str(text).map_err(|error| format!("invalid input settings: {error}"))?;
        match settings.version {
            // v0 (no version field): same shape, defaults fill the rest.
            0 | INPUT_USER_SETTINGS_VERSION => {
                settings.version = INPUT_USER_SETTINGS_VERSION;
                settings.mouse_sensitivity = settings.mouse_sensitivity.clamp(0.01, 100.0);
                settings.gamepad_look_sensitivity =
                    settings.gamepad_look_sensitivity.clamp(0.01, 100.0);
                Ok(settings)
            }
            other => Err(format!(
                "input settings version {other} is newer than supported ({INPUT_USER_SETTINGS_VERSION})"
            )),
        }
    }

    pub fn to_ron(&self) -> String {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default())
            .expect("input settings serialize")
    }
}

/// The active user settings and where they persist.
#[derive(Resource, Clone, Debug, Default)]
pub struct InputUserSettingsStore {
    pub settings: InputUserSettings,
    pub path: Option<PathBuf>,
    /// Changed since the last save.
    pub dirty: bool,
}

impl InputUserSettingsStore {
    /// Loads from `path` (missing file → defaults). A corrupt file is kept
    /// as `input.ron.corrupt` and defaults are used, so a bad edit never
    /// locks the player out of their controls.
    pub fn load(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let settings = match std::fs::read_to_string(&path) {
            Ok(text) => match InputUserSettings::parse(&text) {
                Ok(settings) => settings,
                Err(error) => {
                    log::warn!(
                        target: "engine::input",
                        "ignoring '{}': {error}; using default bindings",
                        path.display()
                    );
                    let _ = std::fs::rename(&path, path.with_extension("ron.corrupt"));
                    InputUserSettings::default()
                }
            },
            Err(_) => InputUserSettings::default(),
        };
        Self {
            settings,
            path: Some(path),
            dirty: false,
        }
    }

    /// Writes the settings atomically (temp file + rename).
    pub fn save(&mut self) -> Result<()> {
        let Some(path) = self.path.clone() else {
            return Ok(());
        };
        write_atomic(&path, self.settings.to_ron().as_bytes())?;
        self.dirty = false;
        Ok(())
    }
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| EngineError::AssetLoad {
            path: parent.display().to_string(),
            reason: error.to_string(),
        })?;
    }
    engine_assets::import::write_atomic(path, bytes).map_err(|error| EngineError::AssetLoad {
        path: path.display().to_string(),
        reason: error.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actions::{ActionKind, InputSource};
    use winit::keyboard::KeyCode;

    #[test]
    fn overrides_replace_authored_bindings_and_round_trip() {
        let def = ActionDef::new("jump", ActionKind::Button)
            .bind(Binding::new(InputSource::Key(KeyCode::Space)));
        let mut settings = InputUserSettings::default();
        assert_eq!(settings.bindings_for("gameplay", &def).len(), 1);
        settings.set_bindings(
            "gameplay",
            "jump",
            vec![Binding::new(InputSource::Key(KeyCode::KeyJ))],
        );
        assert_eq!(
            settings.bindings_for("gameplay", &def)[0].source,
            InputSource::Key(KeyCode::KeyJ)
        );
        let parsed = InputUserSettings::parse(&settings.to_ron()).unwrap();
        assert_eq!(parsed, settings);
        settings.reset("gameplay", "jump");
        assert_eq!(
            settings.bindings_for("gameplay", &def)[0].source,
            InputSource::Key(KeyCode::Space)
        );
    }

    #[test]
    fn unversioned_files_migrate_and_future_versions_are_rejected() {
        let migrated = InputUserSettings::parse("(invert_y: true)").unwrap();
        assert_eq!(migrated.version, INPUT_USER_SETTINGS_VERSION);
        assert!(migrated.invert_y);
        assert_eq!(migrated.mouse_sensitivity, 1.0);
        assert!(InputUserSettings::parse("(version: 99)").is_err());
    }

    #[test]
    fn store_saves_atomically_and_recovers_from_corruption() {
        let dir = std::env::temp_dir().join(format!(
            "starman-input-settings-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let path = dir.join("input.ron");
        let mut store = InputUserSettingsStore::load(&path);
        store.settings.invert_y = true;
        store.save().unwrap();
        assert!(InputUserSettingsStore::load(&path).settings.invert_y);
        std::fs::write(&path, "not ron at all (").unwrap();
        let recovered = InputUserSettingsStore::load(&path);
        assert!(!recovered.settings.invert_y);
        assert!(path.with_extension("ron.corrupt").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
