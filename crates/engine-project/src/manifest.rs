//! The project manifest: the versioned RON file (`project.ron`) at a
//! project's root that declares its identity, entry point, plugins, and
//! settings, per ADR 0001.

use std::collections::BTreeMap;
use std::path::PathBuf;

use engine_core::ProjectId;
use serde::{Deserialize, Serialize};

use crate::settings::GameSettings;

/// An extensible, order-stable bag of project-level settings. Values are
/// arbitrary RON so new settings can be introduced without a manifest
/// version bump; a setting's *meaning* is owned by whatever subsystem reads
/// it, not by this crate.
pub type ProjectSettings = BTreeMap<String, ron::Value>;

/// A declared dependency on an engine or third-party plugin.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct PluginRef {
    pub name: String,
    /// A loose version requirement string (e.g. `"^0.1"`). Not yet enforced.
    #[serde(default)]
    pub version_req: Option<String>,
    /// Project-relative path to the plugin directory or library file.
    #[serde(default)]
    pub path: Option<String>,
}

/// Lua script roots and optional entry chunk.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Default)]
pub struct ScriptsConfig {
    #[serde(default)]
    pub entry: Option<String>,
    #[serde(default)]
    pub roots: Vec<String>,
}

impl ScriptsConfig {
    pub fn has_scripts(&self) -> bool {
        self.entry.is_some() || !self.roots.is_empty()
    }

    pub fn effective_roots(&self) -> Vec<String> {
        if self.roots.is_empty() {
            if self.entry.is_some() {
                vec!["scripts/".to_owned()]
            } else {
                Vec::new()
            }
        } else {
            self.roots.clone()
        }
    }
}

/// Deny-by-default permission grants for scripts and plugins.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct PermissionsConfig {
    #[serde(default = "default_filesystem_grants")]
    pub filesystem: Vec<String>,
    #[serde(default)]
    pub process: bool,
    #[serde(default)]
    pub network: bool,
}

fn default_filesystem_grants() -> Vec<String> {
    vec![
        "assets/**".to_owned(),
        "scripts/**".to_owned(),
        "saves/**".to_owned(),
    ]
}

impl Default for PermissionsConfig {
    fn default() -> Self {
        Self {
            filesystem: default_filesystem_grants(),
            process: false,
            network: false,
        }
    }
}

impl PermissionsConfig {
    pub fn to_plugin_permissions(&self) -> engine_plugin::ProjectPermissions {
        engine_plugin::ProjectPermissions {
            filesystem: self.filesystem.clone(),
            process: self.process,
            network: self.network,
        }
    }
}

/// The project manifest, persisted as `project.ron` at the project root.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ProjectManifest {
    pub version: u32,
    pub id: ProjectId,
    pub name: String,
    /// Path to the entry scene, relative to the project's source assets
    /// directory (see [`crate::ProjectPaths::assets_dir`]).
    pub entry_scene: String,
    #[serde(default)]
    pub plugins: Vec<PluginRef>,
    #[serde(default)]
    pub scripts: ScriptsConfig,
    #[serde(default)]
    pub permissions: PermissionsConfig,
    /// Typed engine settings (physics layers, input, audio, localization,
    /// UI, saves, rendering). New in manifest v2.
    #[serde(default)]
    pub game: GameSettings,
    /// Free-form settings owned by project plugins.
    #[serde(default)]
    pub settings: ProjectSettings,
    /// Build target identifiers this project targets (e.g. `"windows"`,
    /// `"linux"`, `"macos"`). Empty means "all supported targets".
    #[serde(default)]
    pub targets: Vec<String>,
}

impl ProjectManifest {
    /// The manifest format version this build writes. Older versions are
    /// migrated forward on read (see [`crate::migration`]); a bump requires
    /// a migration step and a golden fixture (ADR 0006).
    pub const CURRENT_VERSION: u32 = 2;

    pub fn new(name: impl Into<String>, entry_scene: impl Into<String>) -> Self {
        Self {
            version: Self::CURRENT_VERSION,
            id: ProjectId::new_v4(),
            name: name.into(),
            entry_scene: entry_scene.into(),
            plugins: Vec::new(),
            scripts: ScriptsConfig::default(),
            permissions: PermissionsConfig::default(),
            game: GameSettings::default(),
            settings: ProjectSettings::new(),
            targets: Vec::new(),
        }
    }

    /// Resolve a plugin's on-disk library path relative to the project root.
    pub fn resolve_plugin_dir(
        &self,
        project_root: &std::path::Path,
        plugin: &PluginRef,
    ) -> PathBuf {
        match &plugin.path {
            Some(path) => project_root.join(path),
            None => project_root.join("plugins").join(&plugin.name),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_ron() {
        let mut manifest = ProjectManifest::new("My Game", "scenes/main.scene.ron");
        manifest.scripts.entry = Some("scripts/main.lua".to_owned());
        manifest.scripts.roots = vec!["scripts/".to_owned()];
        manifest.plugins.push(PluginRef {
            name: "example_gameplay".to_owned(),
            version_req: Some("^0.1".to_owned()),
            path: Some("plugins/example_gameplay".to_owned()),
        });
        let serialized = ron::ser::to_string_pretty(&manifest, ron::ser::PrettyConfig::default())
            .expect("manifest should serialize");
        let deserialized: ProjectManifest =
            ron::from_str(&serialized).expect("manifest should deserialize");
        assert_eq!(manifest, deserialized);
    }

    #[test]
    fn missing_optional_fields_default_to_empty() {
        let source = r#"(
            version: 2,
            id: "00000000-0000-0000-0000-000000000000",
            name: "Minimal",
            entry_scene: "scenes/main.scene.ron",
        )"#;
        let manifest: ProjectManifest = ron::from_str(source).expect("manifest should parse");
        assert!(manifest.plugins.is_empty());
        assert!(manifest.settings.is_empty());
        assert!(manifest.targets.is_empty());
        assert!(!manifest.scripts.has_scripts());
        assert!(!manifest.permissions.process);
        assert!(!manifest.permissions.network);
    }
}
