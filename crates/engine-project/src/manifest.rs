//! The project manifest: the versioned RON file (`project.ron`) at a
//! project's root that declares its identity, entry point, plugins, and
//! settings, per ADR 0001.

use std::collections::BTreeMap;

use engine_core::ProjectId;
use serde::{Deserialize, Serialize};

/// An extensible, order-stable bag of project-level settings. Values are
/// arbitrary RON so new settings can be introduced without a manifest
/// version bump; a setting's *meaning* is owned by whatever subsystem reads
/// it, not by this crate.
pub type ProjectSettings = BTreeMap<String, ron::Value>;

/// A declared dependency on an engine or third-party plugin. The ABI to
/// actually load plugins is M3 work (ADR 0005); for now this is only a
/// declaration recorded in the manifest.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct PluginRef {
    pub name: String,
    /// A loose version requirement string (e.g. `"^0.1"`). Not yet enforced.
    #[serde(default)]
    pub version_req: Option<String>,
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
    pub settings: ProjectSettings,
    /// Build target identifiers this project targets (e.g. `"windows"`,
    /// `"linux"`, `"macos"`). Empty means "all supported targets".
    #[serde(default)]
    pub targets: Vec<String>,
}

impl ProjectManifest {
    /// The only manifest format version this build knows how to read.
    /// Bumping this requires a migration path, mirroring scene versioning
    /// (ADR 0006).
    pub const CURRENT_VERSION: u32 = 1;

    pub fn new(name: impl Into<String>, entry_scene: impl Into<String>) -> Self {
        Self {
            version: Self::CURRENT_VERSION,
            id: ProjectId::new_v4(),
            name: name.into(),
            entry_scene: entry_scene.into(),
            plugins: Vec::new(),
            settings: ProjectSettings::new(),
            targets: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_ron() {
        let manifest = ProjectManifest::new("My Game", "scenes/main.scene.ron");
        let serialized = ron::ser::to_string_pretty(&manifest, ron::ser::PrettyConfig::default())
            .expect("manifest should serialize");
        let deserialized: ProjectManifest =
            ron::from_str(&serialized).expect("manifest should deserialize");
        assert_eq!(manifest, deserialized);
    }

    #[test]
    fn missing_optional_fields_default_to_empty() {
        let source = r#"(
            version: 1,
            id: "00000000-0000-0000-0000-000000000000",
            name: "Minimal",
            entry_scene: "scenes/main.scene.ron",
        )"#;
        let manifest: ProjectManifest = ron::from_str(source).expect("manifest should parse");
        assert!(manifest.plugins.is_empty());
        assert!(manifest.settings.is_empty());
        assert!(manifest.targets.is_empty());
    }
}
