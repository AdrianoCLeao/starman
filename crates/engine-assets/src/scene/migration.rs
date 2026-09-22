//! Forward-only migration of scene files to the current version, per ADR
//! 0006 (migrations are forward-only and must preserve a backup of the
//! original, authored data before it is rewritten).

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use engine_core::{EngineError, EntityId, Result};
use serde::{Deserialize, Serialize};

use super::{write_scene_ron, EntityData, SceneFile, SceneValue};

/// The most recent scene format version this build knows how to migrate
/// forward from. Scenes older than this (or newer than
/// [`SceneFile::CURRENT_VERSION`]) fail with an explicit, actionable error
/// rather than being silently reinterpreted.
pub(crate) const LEGACY_VERSION_V1: u32 = 1;

/// Cheap, tolerant peek at a scene file's `version` field, ignoring every
/// other field, so the deserializer can pick the right concrete type to
/// parse the rest of the document with.
#[derive(Deserialize)]
pub(crate) struct VersionProbe {
    pub version: u32,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub(crate) struct LegacySceneFileV1 {
    pub version: u32,
    pub name: String,
    pub entities: Vec<LegacyEntityDataV1>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub(crate) struct LegacyEntityDataV1 {
    pub name: Option<String>,
    pub components: HashMap<String, SceneValue>,
    pub children: Vec<LegacyEntityDataV1>,
}

/// Converts a version-1 scene into the current version, assigning a fresh
/// [`EntityId`] to every entity. Assignment order matches the entities'
/// traversal order in the source file, so the *relative* result is stable
/// for a given input; because ids are UUID v4, migrating the same file twice
/// independently still yields two different (but each internally valid)
/// sets of ids — this is expected and harmless, since migration is meant to
/// run once per file, after which the written-back v2 file is the source of
/// truth for identity.
pub(crate) fn migrate_v1_to_v2(legacy: LegacySceneFileV1) -> SceneFile {
    SceneFile {
        version: SceneFile::CURRENT_VERSION,
        name: legacy.name,
        entities: legacy.entities.into_iter().map(migrate_entity).collect(),
    }
}

fn migrate_entity(legacy: LegacyEntityDataV1) -> EntityData {
    EntityData {
        id: EntityId::new_v4(),
        name: legacy.name,
        components: legacy.components,
        children: legacy.children.into_iter().map(migrate_entity).collect(),
    }
}

/// Migrates a legacy v1 scene, backs up the original file next to it, then
/// overwrites `path` with the migrated (v2) contents.
pub(crate) fn migrate_and_persist(
    path: &Path,
    original_source: &str,
    legacy: LegacySceneFileV1,
) -> Result<SceneFile> {
    let migrated = migrate_v1_to_v2(legacy);

    let backup_path = backup_path_for(path);
    fs::write(&backup_path, original_source).map_err(|error| EngineError::AssetLoad {
        path: backup_path.display().to_string(),
        reason: format!("failed to write pre-migration backup: {error}"),
    })?;

    write_scene_ron(path, &migrated)?;

    log::info!(
        target: "engine::assets",
        "Migrated scene '{}' from version {} to {} (backup: '{}')",
        path.display(),
        LEGACY_VERSION_V1,
        SceneFile::CURRENT_VERSION,
        backup_path.display(),
    );

    Ok(migrated)
}

fn backup_path_for(path: &Path) -> PathBuf {
    let mut backup = path.as_os_str().to_owned();
    backup.push(format!(".v{LEGACY_VERSION_V1}.bak"));
    PathBuf::from(backup)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrates_entities_in_traversal_order_and_assigns_ids() {
        let legacy = LegacySceneFileV1 {
            version: LEGACY_VERSION_V1,
            name: "Legacy".to_owned(),
            entities: vec![LegacyEntityDataV1 {
                name: Some("Root".to_owned()),
                components: HashMap::new(),
                children: vec![LegacyEntityDataV1 {
                    name: Some("Child".to_owned()),
                    components: HashMap::new(),
                    children: Vec::new(),
                }],
            }],
        };

        let migrated = migrate_v1_to_v2(legacy);

        assert_eq!(migrated.version, SceneFile::CURRENT_VERSION);
        assert_eq!(migrated.entities.len(), 1);
        assert_eq!(migrated.entities[0].name.as_deref(), Some("Root"));
        assert_eq!(migrated.entities[0].children.len(), 1);

        let root_id = migrated.entities[0].id;
        let child_id = migrated.entities[0].children[0].id;
        assert_ne!(root_id, child_id, "each entity must get its own id");
    }
}
