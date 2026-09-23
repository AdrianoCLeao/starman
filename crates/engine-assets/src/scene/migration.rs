//! Forward-only migration of scene files to the current version, per ADR
//! 0006 (migrations are forward-only and must preserve a backup of the
//! original, authored data before it is rewritten).

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use engine_core::{EngineError, EntityId, Result};
use serde::{Deserialize, Serialize};

use super::{write_scene_ron, EntityData, SceneFile, SceneValue};

/// Scene format version 1 (no entity ids).
pub(crate) const LEGACY_VERSION_V1: u32 = 1;
/// Scene format version 2 (entity ids, no nested instances).
pub(crate) const LEGACY_VERSION_V2: u32 = 2;

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

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub(crate) struct LegacySceneFileV2 {
    pub version: u32,
    pub name: String,
    pub entities: Vec<LegacyEntityDataV2>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub(crate) struct LegacyEntityDataV2 {
    pub id: EntityId,
    pub name: Option<String>,
    pub components: HashMap<String, SceneValue>,
    pub children: Vec<LegacyEntityDataV2>,
}

/// Converts a version-1 scene into version 2 (ids), then into the current
/// version.
pub(crate) fn migrate_v1_to_current(legacy: LegacySceneFileV1) -> SceneFile {
    migrate_v2_to_current(migrate_v1_to_v2(legacy))
}

pub(crate) fn migrate_v1_to_v2(legacy: LegacySceneFileV1) -> LegacySceneFileV2 {
    LegacySceneFileV2 {
        version: LEGACY_VERSION_V2,
        name: legacy.name,
        entities: legacy.entities.into_iter().map(migrate_entity_v1).collect(),
    }
}

fn migrate_entity_v1(legacy: LegacyEntityDataV1) -> LegacyEntityDataV2 {
    LegacyEntityDataV2 {
        id: EntityId::new_v4(),
        name: legacy.name,
        components: legacy.components,
        children: legacy.children.into_iter().map(migrate_entity_v1).collect(),
    }
}

/// Version 2 → 3 is structurally a no-op: instances are optional and default
/// to absent. Only the version number changes.
pub(crate) fn migrate_v2_to_current(legacy: LegacySceneFileV2) -> SceneFile {
    SceneFile {
        version: SceneFile::CURRENT_VERSION,
        name: legacy.name,
        entities: legacy.entities.into_iter().map(migrate_entity_v2).collect(),
    }
}

fn migrate_entity_v2(legacy: LegacyEntityDataV2) -> EntityData {
    EntityData {
        id: legacy.id,
        name: legacy.name,
        components: legacy.components,
        children: legacy.children.into_iter().map(migrate_entity_v2).collect(),
        instance: None,
    }
}

pub(crate) fn migrate_v1_and_persist(
    path: &Path,
    original_source: &str,
    legacy: LegacySceneFileV1,
) -> Result<SceneFile> {
    let migrated = migrate_v1_to_current(legacy);
    persist_migration(path, original_source, LEGACY_VERSION_V1, &migrated)?;
    Ok(migrated)
}

pub(crate) fn migrate_v2_and_persist(
    path: &Path,
    original_source: &str,
    legacy: LegacySceneFileV2,
) -> Result<SceneFile> {
    let migrated = migrate_v2_to_current(legacy);
    persist_migration(path, original_source, LEGACY_VERSION_V2, &migrated)?;
    Ok(migrated)
}

fn persist_migration(
    path: &Path,
    original_source: &str,
    from_version: u32,
    migrated: &SceneFile,
) -> Result<()> {
    let backup_path = backup_path_for(path, from_version);
    fs::write(&backup_path, original_source).map_err(|error| EngineError::AssetLoad {
        path: backup_path.display().to_string(),
        reason: format!("failed to write pre-migration backup: {error}"),
    })?;

    write_scene_ron(path, migrated)?;

    log::info!(
        target: "engine::assets",
        "Migrated scene '{}' from version {} to {} (backup: '{}')",
        path.display(),
        from_version,
        SceneFile::CURRENT_VERSION,
        backup_path.display(),
    );

    Ok(())
}

fn backup_path_for(path: &Path, from_version: u32) -> PathBuf {
    let mut backup = path.as_os_str().to_owned();
    backup.push(format!(".v{from_version}.bak"));
    PathBuf::from(backup)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrates_v1_entities_in_traversal_order_and_assigns_ids() {
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

        let migrated = migrate_v1_to_current(legacy);

        assert_eq!(migrated.version, SceneFile::CURRENT_VERSION);
        assert_eq!(migrated.entities.len(), 1);
        assert_eq!(migrated.entities[0].name.as_deref(), Some("Root"));
        assert_eq!(migrated.entities[0].children.len(), 1);
        assert!(migrated.entities[0].instance.is_none());

        let root_id = migrated.entities[0].id;
        let child_id = migrated.entities[0].children[0].id;
        assert_ne!(root_id, child_id, "each entity must get its own id");
    }

    #[test]
    fn migrates_v2_preserving_ids() {
        let id = EntityId::new_v4();
        let legacy = LegacySceneFileV2 {
            version: LEGACY_VERSION_V2,
            name: "V2".to_owned(),
            entities: vec![LegacyEntityDataV2 {
                id,
                name: Some("Root".to_owned()),
                components: HashMap::new(),
                children: Vec::new(),
            }],
        };

        let migrated = migrate_v2_to_current(legacy);
        assert_eq!(migrated.version, SceneFile::CURRENT_VERSION);
        assert_eq!(migrated.entities[0].id, id);
        assert!(migrated.entities[0].instance.is_none());
    }
}
