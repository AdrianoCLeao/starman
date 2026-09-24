//! Forward-only project manifest migrations (ADR 0006).
//!
//! Reading never rewrites `project.ron`; [`crate::Project::persist_migration`]
//! writes a `project.ron.v<N>.bak` backup and then the migrated manifest.

use serde::Deserialize;

use crate::manifest::ProjectManifest;

/// A parsed manifest and the version it was migrated from, if any.
#[derive(Debug, Clone)]
pub struct ParsedManifest {
    pub manifest: ProjectManifest,
    pub migrated_from: Option<u32>,
}

#[derive(Deserialize)]
struct VersionProbe {
    version: u32,
}

/// Parses any supported manifest version into the current one.
pub fn parse_manifest(source: &str) -> Result<ParsedManifest, String> {
    let probe: VersionProbe = ron::from_str(source)
        .map_err(|error| format!("failed to read the manifest version: {error}"))?;
    match probe.version {
        1 => {
            // v1 → v2: additive. `game` settings are new and default; the
            // free-form `settings` bag is carried over untouched.
            let mut manifest: ProjectManifest =
                ron::from_str(source).map_err(|error| format!("invalid v1 manifest: {error}"))?;
            manifest.version = ProjectManifest::CURRENT_VERSION;
            Ok(ParsedManifest {
                manifest,
                migrated_from: Some(1),
            })
        }
        ProjectManifest::CURRENT_VERSION => {
            let manifest: ProjectManifest =
                ron::from_str(source).map_err(|error| format!("invalid manifest: {error}"))?;
            Ok(ParsedManifest {
                manifest,
                migrated_from: None,
            })
        }
        other if other > ProjectManifest::CURRENT_VERSION => Err(format!(
            "unsupported project manifest version {other}: it is newer than this engine \
             supports ({}); update Starman to open this project",
            ProjectManifest::CURRENT_VERSION
        )),
        other => Err(format!(
            "unsupported project manifest version {other}; expected 1..={}",
            ProjectManifest::CURRENT_VERSION
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const V1_FIXTURE: &str = include_str!("../tests/fixtures/project_v1.ron");

    #[test]
    fn v1_fixture_migrates_to_current_with_defaults() {
        let parsed = parse_manifest(V1_FIXTURE).expect("v1 parses");
        assert_eq!(parsed.migrated_from, Some(1));
        let manifest = parsed.manifest;
        assert_eq!(manifest.version, ProjectManifest::CURRENT_VERSION);
        assert_eq!(manifest.name, "Fixture v1");
        assert_eq!(manifest.game, Default::default());
        assert_eq!(manifest.plugins.len(), 1);
        assert!(manifest.settings.contains_key("custom_plugin_setting"));
    }

    #[test]
    fn current_version_is_not_migrated() {
        let parsed = parse_manifest(V1_FIXTURE).unwrap();
        let written =
            ron::ser::to_string_pretty(&parsed.manifest, ron::ser::PrettyConfig::default())
                .unwrap();
        let reparsed = parse_manifest(&written).unwrap();
        assert_eq!(reparsed.migrated_from, None);
        assert_eq!(reparsed.manifest, parsed.manifest);
    }

    #[test]
    fn future_versions_are_rejected_with_an_actionable_error() {
        let error = parse_manifest("(version: 99)").unwrap_err();
        assert!(error.contains("newer than this engine"), "{error}");
    }
}
