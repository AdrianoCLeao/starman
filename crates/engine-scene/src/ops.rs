//! Apply / Revert / Promote operations on scene documents (pure data +
//! optional world resync hooks).

use std::path::Path;

use engine_assets::{
    write_scene_ron, LocalAddedEntity, LocalParent, OverrideEntry, SceneEntityData, SceneFile,
    SceneInstanceData, SceneValue,
};
use engine_core::{EngineError, EntityId, Result, SourceAssetId};

/// Removes matching overrides from `instance`. An empty `field_path` matches
/// whole-component overrides; `component == "*"` reverts everything for
/// `target` (or all targets when `target` is `None`).
pub fn revert_overrides(
    instance: &mut SceneInstanceData,
    target: Option<EntityId>,
    component: Option<&str>,
    field_path: Option<&str>,
) -> usize {
    let before = instance.overrides.len();
    instance.overrides.retain(|entry| {
        if let Some(target) = target {
            if entry.target != target {
                return true;
            }
        }
        if let Some(component) = component {
            if component != "*" && entry.component != component {
                return true;
            }
        }
        if let Some(field_path) = field_path {
            if entry.field_path != field_path {
                return true;
            }
        }
        false
    });
    instance.normalize();
    before - instance.overrides.len()
}

/// Writes `overrides` into the source `SceneFile` (mutating template
/// entities), then clears those overrides from `instance`.
pub fn apply_overrides_to_source(
    source: &mut SceneFile,
    instance: &mut SceneInstanceData,
    overrides: &[OverrideEntry],
) -> Result<usize> {
    let mut applied = 0;
    for entry in overrides {
        if apply_one_to_source(source, entry)? {
            applied += 1;
        }
    }
    let targets: Vec<_> = overrides.to_vec();
    for entry in &targets {
        revert_overrides(
            instance,
            Some(entry.target),
            Some(&entry.component),
            Some(&entry.field_path),
        );
    }
    Ok(applied)
}

fn apply_one_to_source(source: &mut SceneFile, entry: &OverrideEntry) -> Result<bool> {
    let entity = find_entity_mut(&mut source.entities, entry.target).ok_or_else(|| {
        EngineError::AssetLoad {
            path: entry.target.to_string(),
            reason: "apply failed: override target entity does not exist in source scene"
                .to_owned(),
        }
    })?;

    if entry.field_path.is_empty() {
        entity
            .components
            .insert(entry.component.clone(), entry.value.clone());
        return Ok(true);
    }

    let component = entity
        .components
        .entry(entry.component.clone())
        .or_insert_with(|| SceneValue::Map(ron::Map::new()));
    set_field(component, &entry.field_path, entry.value.clone());
    Ok(true)
}

/// Moves a locally-added entity into the source scene document, parented
/// under `source_parent` (`None` = root of the source scene). Removes it
/// from `instance.added`.
pub fn promote_local_entity(
    source: &mut SceneFile,
    instance: &mut SceneInstanceData,
    local_id: EntityId,
    source_parent: Option<EntityId>,
) -> Result<()> {
    let index = instance
        .added
        .iter()
        .position(|added| added.entity.id == local_id)
        .ok_or_else(|| EngineError::AssetLoad {
            path: local_id.to_string(),
            reason: "promote failed: entity is not in this instance's local-added list".to_owned(),
        })?;
    let LocalAddedEntity { entity, .. } = instance.added.remove(index);

    match source_parent {
        None => source.entities.push(entity),
        Some(parent_id) => {
            let parent = find_entity_mut(&mut source.entities, parent_id).ok_or_else(|| {
                EngineError::AssetLoad {
                    path: parent_id.to_string(),
                    reason: "promote failed: source parent entity not found".to_owned(),
                }
            })?;
            parent.children.push(entity);
        }
    }
    Ok(())
}

/// Loads a scene file from disk (current version only — callers should
/// migrate via [`engine_assets::SceneDeserializer`] first if needed).
pub fn load_scene_file(path: &Path) -> Result<SceneFile> {
    let source = std::fs::read_to_string(path).map_err(|error| EngineError::AssetLoad {
        path: path.display().to_string(),
        reason: error.to_string(),
    })?;
    ron::from_str(&source).map_err(|error| EngineError::AssetLoad {
        path: path.display().to_string(),
        reason: format!("failed to parse scene: {error}"),
    })
}

pub fn save_scene_file(path: &Path, scene: &SceneFile) -> Result<()> {
    write_scene_ron(path, scene)
}

/// Records a field override on `instance` (replacing any existing override
/// for the same target/component/field).
pub fn set_override(
    instance: &mut SceneInstanceData,
    target: EntityId,
    component: impl Into<String>,
    field_path: impl Into<String>,
    value: SceneValue,
) {
    let component = component.into();
    let field_path = field_path.into();
    instance.overrides.retain(|entry| {
        !(entry.target == target && entry.component == component && entry.field_path == field_path)
    });
    instance.overrides.push(OverrideEntry {
        target,
        component,
        field_path,
        value,
    });
    instance.normalize();
}

pub fn add_local_entity(instance: &mut SceneInstanceData, added: LocalAddedEntity) {
    instance.added.push(added);
}

pub fn mark_removed(instance: &mut SceneInstanceData, template_id: EntityId) {
    if !instance.removed.contains(&template_id) {
        instance.removed.push(template_id);
    }
    instance.normalize();
}

pub fn new_instance(scene: SourceAssetId, scene_path: Option<String>) -> SceneInstanceData {
    let mut data = SceneInstanceData::new(scene);
    data.scene_path = scene_path;
    data
}

fn find_entity_mut(entities: &mut [SceneEntityData], id: EntityId) -> Option<&mut SceneEntityData> {
    for entity in entities.iter_mut() {
        if entity.id == id {
            return Some(entity);
        }
        if let Some(found) = find_entity_mut(&mut entity.children, id) {
            return Some(found);
        }
    }
    None
}

fn set_field(value: &mut SceneValue, field_path: &str, new_value: SceneValue) {
    let parts: Vec<&str> = field_path.split('.').filter(|p| !p.is_empty()).collect();
    if parts.is_empty() {
        *value = new_value;
        return;
    }
    if parts.len() == 1 {
        if let SceneValue::Map(map) = value {
            let key = parts[0];
            let exists = map
                .iter()
                .any(|(k, _)| matches!(k, SceneValue::String(s) if s == key));
            if exists {
                if let Some((_, slot)) = map
                    .iter_mut()
                    .find(|(k, _)| matches!(k, SceneValue::String(s) if s == key))
                {
                    *slot = new_value;
                }
            } else {
                map.insert(SceneValue::String(key.to_owned()), new_value);
            }
        }
        return;
    }
    if let SceneValue::Map(map) = value {
        let key = parts[0];
        if let Some((_, slot)) = map
            .iter_mut()
            .find(|(k, _)| matches!(k, SceneValue::String(s) if s == key))
        {
            set_field(slot, &parts[1..].join("."), new_value);
        }
    }
}

/// Helper used by promote when the local was parented to a template entity.
pub fn local_parent_template(parent: LocalParent) -> Option<EntityId> {
    match parent {
        LocalParent::Template(id) => Some(id),
        LocalParent::InstanceRoot => None,
    }
}
