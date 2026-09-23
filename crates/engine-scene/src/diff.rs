//! Deterministic structural diff between a template scene and an instance
//! payload (overrides / added / removed).

use std::fmt;

use engine_assets::{OverrideEntry, SceneEntityData, SceneFile, SceneInstanceData, SceneValue};
use engine_core::EntityId;

/// One ordered, human-readable change in a structural diff.
#[derive(Debug, Clone, PartialEq)]
pub enum DiffChange {
    Added {
        id: EntityId,
        name: Option<String>,
    },
    Removed {
        id: EntityId,
        name: Option<String>,
    },
    FieldChanged {
        target: EntityId,
        component: String,
        field_path: String,
        from: SceneValue,
        to: SceneValue,
    },
    ComponentChanged {
        target: EntityId,
        component: String,
        from: Option<SceneValue>,
        to: Option<SceneValue>,
    },
}

/// Deterministic diff of an instance against its template.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StructuralDiff {
    pub changes: Vec<DiffChange>,
}

impl StructuralDiff {
    /// Diffs `instance` against `template` (the source scene asset).
    pub fn between(template: &SceneFile, instance: &SceneInstanceData) -> Self {
        let mut changes = Vec::new();
        let template_ids = collect_entity_meta(&template.entities);

        for removed in &instance.removed {
            let name = template_ids
                .iter()
                .find(|(id, _)| id == removed)
                .and_then(|(_, name)| name.clone());
            changes.push(DiffChange::Removed { id: *removed, name });
        }

        for added in &instance.added {
            changes.push(DiffChange::Added {
                id: added.entity.id,
                name: added.entity.name.clone(),
            });
        }

        for entry in &instance.overrides {
            if entry.field_path.is_empty() {
                let from = find_component(&template.entities, entry.target, &entry.component);
                changes.push(DiffChange::ComponentChanged {
                    target: entry.target,
                    component: entry.component.clone(),
                    from,
                    to: Some(entry.value.clone()),
                });
            } else {
                let from = find_component(&template.entities, entry.target, &entry.component)
                    .and_then(|value| field_at(&value, &entry.field_path))
                    .unwrap_or(SceneValue::Unit);
                changes.push(DiffChange::FieldChanged {
                    target: entry.target,
                    component: entry.component.clone(),
                    field_path: entry.field_path.clone(),
                    from,
                    to: entry.value.clone(),
                });
            }
        }

        changes.sort_by_key(|change| change.sort_key());
        Self { changes }
    }

    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }
}

impl DiffChange {
    fn sort_key(&self) -> (u8, String, String, String) {
        match self {
            DiffChange::Removed { id, .. } => (0, id.to_string(), String::new(), String::new()),
            DiffChange::Added { id, .. } => (1, id.to_string(), String::new(), String::new()),
            DiffChange::ComponentChanged {
                target, component, ..
            } => (2, target.to_string(), component.clone(), String::new()),
            DiffChange::FieldChanged {
                target,
                component,
                field_path,
                ..
            } => (3, target.to_string(), component.clone(), field_path.clone()),
        }
    }
}

impl fmt::Display for StructuralDiff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.changes.is_empty() {
            return writeln!(f, "(no differences)");
        }
        for change in &self.changes {
            match change {
                DiffChange::Added { id, name } => {
                    writeln!(
                        f,
                        "+ added {} ({})",
                        id,
                        name.as_deref().unwrap_or("unnamed")
                    )?;
                }
                DiffChange::Removed { id, name } => {
                    writeln!(
                        f,
                        "- removed {} ({})",
                        id,
                        name.as_deref().unwrap_or("unnamed")
                    )?;
                }
                DiffChange::ComponentChanged {
                    target,
                    component,
                    to,
                    ..
                } => {
                    writeln!(
                        f,
                        "~ {target}.{component} => {to:?}",
                        to = to.as_ref().unwrap_or(&SceneValue::Unit)
                    )?;
                }
                DiffChange::FieldChanged {
                    target,
                    component,
                    field_path,
                    to,
                    ..
                } => {
                    writeln!(f, "~ {target}.{component}.{field_path} => {to:?}")?;
                }
            }
        }
        Ok(())
    }
}

fn collect_entity_meta(entities: &[SceneEntityData]) -> Vec<(EntityId, Option<String>)> {
    let mut out = Vec::new();
    fn walk(entities: &[SceneEntityData], out: &mut Vec<(EntityId, Option<String>)>) {
        for entity in entities {
            out.push((entity.id, entity.name.clone()));
            walk(&entity.children, out);
            if let Some(instance) = &entity.instance {
                for added in &instance.added {
                    walk(std::slice::from_ref(&added.entity), out);
                }
            }
        }
    }
    walk(entities, &mut out);
    out
}

fn find_component(
    entities: &[SceneEntityData],
    id: EntityId,
    component: &str,
) -> Option<SceneValue> {
    fn walk(entities: &[SceneEntityData], id: EntityId, component: &str) -> Option<SceneValue> {
        for entity in entities {
            if entity.id == id {
                return entity.components.get(component).cloned();
            }
            if let Some(found) = walk(&entity.children, id, component) {
                return Some(found);
            }
        }
        None
    }
    walk(entities, id, component)
}

fn field_at(value: &SceneValue, field_path: &str) -> Option<SceneValue> {
    let mut current = value.clone();
    for part in field_path.split('.').filter(|p| !p.is_empty()) {
        let SceneValue::Map(map) = current else {
            return None;
        };
        current = map.iter().find_map(|(k, v)| match k {
            SceneValue::String(s) if s == part => Some(v.clone()),
            _ => None,
        })?;
    }
    Some(current)
}

/// Convenience: build a diff from raw override lists (tests).
pub fn diff_from_overrides(template: &SceneFile, overrides: Vec<OverrideEntry>) -> StructuralDiff {
    let mut instance = SceneInstanceData::new(engine_core::SourceAssetId::new_v4());
    instance.overrides = overrides;
    StructuralDiff::between(template, &instance)
}
