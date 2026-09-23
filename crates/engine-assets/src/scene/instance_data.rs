//! Serializable scene-instance payload persisted on instance-root entities
//! in SceneFile v3 (nested scenes / prefabs).

use engine_core::{EntityId, SourceAssetId};
use serde::{Deserialize, Serialize};

use super::{EntityData, SceneValue};

/// One field-level override relative to a template entity inside the
/// referenced scene asset.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct OverrideEntry {
    /// Template entity id inside the referenced [`SceneFile`].
    pub target: EntityId,
    pub component: String,
    /// Dot-separated reflect field path (empty string = whole component).
    #[serde(default)]
    pub field_path: String,
    pub value: SceneValue,
}

/// Local attachment point for an entity added only on this instance.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub enum LocalParent {
    /// Parent is the instance root itself.
    InstanceRoot,
    /// Parent is an inherited template entity (by its template [`EntityId`]).
    Template(EntityId),
}

/// An entity (sub)tree that exists only on this instance, not in the source
/// scene asset.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct LocalAddedEntity {
    pub parent: LocalParent,
    pub entity: EntityData,
}

/// Prefab / nested-scene instance data stored on the instance-root entity
/// in the parent scene document.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct SceneInstanceData {
    /// Stable id of the nested scene asset.
    pub scene: SourceAssetId,
    /// Optional legacy path used only when resolving without a database.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scene_path: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub overrides: Vec<OverrideEntry>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub removed: Vec<EntityId>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub added: Vec<LocalAddedEntity>,
}

impl SceneInstanceData {
    pub fn new(scene: SourceAssetId) -> Self {
        Self {
            scene,
            scene_path: None,
            overrides: Vec::new(),
            removed: Vec::new(),
            added: Vec::new(),
        }
    }

    /// Sorts overrides and removed ids for deterministic serialization.
    pub fn normalize(&mut self) {
        self.overrides.sort_by(|a, b| {
            (&a.target, a.component.as_str(), a.field_path.as_str()).cmp(&(
                &b.target,
                b.component.as_str(),
                b.field_path.as_str(),
            ))
        });
        self.removed.sort();
        self.removed.dedup();
    }
}
