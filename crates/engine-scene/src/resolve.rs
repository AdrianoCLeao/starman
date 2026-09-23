//! Live instance expansion: spawn / resync inherited entities from scene
//! assets into the world.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use bevy_ecs::prelude::Entity;
use bevy_ecs::world::World;
use engine_assets::{
    AssetServer, InheritedEntity, InstanceLocalEntity, LocalAddedEntity, LocalParent,
    OverrideEntry, SceneDeserializer, SceneEntityData, SceneExternalComponents, SceneFile,
    SceneInstance, SceneInstanceData, SceneValue,
};
use engine_core::{Children, EngineError, EntityId, EntityName, Parent, PersistentId, Result};
use engine_reflect::{ComponentRegistry, ReflectTypeRegistry};

use crate::InstancePath;

/// Expands every [`SceneInstance`] currently present in `world` (depth-first,
/// unlimited nesting). Safe to call after a flat [`SceneDeserializer`] load.
pub fn expand_all_instances(
    world: &mut World,
    component_registry: &ComponentRegistry,
    type_registry: &ReflectTypeRegistry,
    asset_server: &mut AssetServer,
    external: Option<&dyn SceneExternalComponents>,
) -> Result<usize> {
    let mut resolver = InstanceResolver {
        component_registry,
        type_registry,
        external,
    };
    resolver.expand_pending(world, asset_server)
}

/// Tears down inherited/local children of `instance_root` and re-expands
/// from the current [`SceneInstance`] payload (hot-reload / apply).
pub fn resync_instance(
    world: &mut World,
    instance_root: Entity,
    component_registry: &ComponentRegistry,
    type_registry: &ReflectTypeRegistry,
    asset_server: &mut AssetServer,
    external: Option<&dyn SceneExternalComponents>,
) -> Result<()> {
    let mut resolver = InstanceResolver {
        component_registry,
        type_registry,
        external,
    };
    resolver.resync(world, instance_root, asset_server)
}

pub struct InstanceResolver<'a> {
    pub component_registry: &'a ComponentRegistry,
    pub type_registry: &'a ReflectTypeRegistry,
    pub external: Option<&'a dyn SceneExternalComponents>,
}

impl<'a> InstanceResolver<'a> {
    pub fn expand_pending(
        &mut self,
        world: &mut World,
        asset_server: &mut AssetServer,
    ) -> Result<usize> {
        let mut expanded = 0;
        // Loop until no pending instance lacks inherited children. Nested
        // instances appear as we expand parents.
        loop {
            let pending: Vec<Entity> = world
                .iter_entities()
                .filter(|entity| entity.get::<SceneInstance>().is_some())
                .filter(|entity| !instance_already_expanded(world, entity.id()))
                .map(|entity| entity.id())
                .collect();

            if pending.is_empty() {
                break;
            }

            for root in pending {
                self.expand_one(world, root, asset_server)?;
                expanded += 1;
            }
        }
        Ok(expanded)
    }

    pub fn resync(
        &mut self,
        world: &mut World,
        instance_root: Entity,
        asset_server: &mut AssetServer,
    ) -> Result<()> {
        despawn_instance_contents(world, instance_root);
        self.expand_one(world, instance_root, asset_server)?;
        // Nested instances created during expand need a full pass.
        self.expand_pending(world, asset_server)?;
        Ok(())
    }

    fn expand_one(
        &mut self,
        world: &mut World,
        instance_root: Entity,
        asset_server: &mut AssetServer,
    ) -> Result<()> {
        let data = world
            .get::<SceneInstance>(instance_root)
            .map(|value| value.0.clone())
            .ok_or_else(|| EngineError::AssetLoad {
                path: format!("{instance_root:?}"),
                reason: "entity is not a SceneInstance root".to_owned(),
            })?;

        let template = load_template_scene(&data, asset_server)?;
        let removed: HashSet<EntityId> = data.removed.iter().copied().collect();
        let overrides_by_target = group_overrides(&data.overrides);

        let mut template_map: HashMap<EntityId, Entity> = HashMap::new();

        for root in &template.entities {
            if removed.contains(&root.id) {
                continue;
            }
            self.spawn_inherited_recursive(
                world,
                asset_server,
                root,
                instance_root,
                instance_root,
                &removed,
                &overrides_by_target,
                &mut template_map,
            )?;
        }

        for added in &data.added {
            self.spawn_local_added(world, asset_server, instance_root, added, &template_map)?;
        }

        let _ = InstancePath::new(
            world
                .get::<PersistentId>(instance_root)
                .map(|id| vec![id.0])
                .unwrap_or_default(),
        );

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn spawn_inherited_recursive(
        &mut self,
        world: &mut World,
        asset_server: &mut AssetServer,
        data: &SceneEntityData,
        instance_root: Entity,
        parent: Entity,
        removed: &HashSet<EntityId>,
        overrides: &HashMap<EntityId, Vec<&OverrideEntry>>,
        template_map: &mut HashMap<EntityId, Entity>,
    ) -> Result<Entity> {
        // Spawn via a one-off deserializer for component payloads, then tag.
        let entity = {
            let mut deserializer = SceneDeserializer::new(
                world,
                self.component_registry,
                self.type_registry,
                asset_server,
            );
            if let Some(external) = self.external {
                deserializer = deserializer.with_external_components(external);
            }
            // Temporary: spawn without parent, then reparent.
            let stub = SceneEntityData {
                id: data.id,
                name: data.name.clone(),
                components: apply_override_map(&data.components, overrides.get(&data.id)),
                children: Vec::new(),
                instance: data.instance.clone(),
            };
            deserializer
                .load_scene(&SceneFile {
                    version: SceneFile::CURRENT_VERSION,
                    name: "_instance_stub".to_owned(),
                    entities: vec![stub],
                })?
                .into_iter()
                .next()
                .ok_or_else(|| EngineError::AssetLoad {
                    path: data.id.to_string(),
                    reason: "failed to spawn inherited entity".to_owned(),
                })?
        };

        // Replace PersistentId semantics: inherited entities keep template id
        // only via InheritedEntity — remove PersistentId so parent saves do
        // not treat them as native. (They still need a runtime handle.)
        if let Ok(mut entity_ref) = world.get_entity_mut(entity) {
            entity_ref.remove::<PersistentId>();
            entity_ref.insert(InheritedEntity {
                template_id: data.id,
                instance_root,
            });
        }

        attach_child(world, parent, entity);
        template_map.insert(data.id, entity);

        for child in &data.children {
            if removed.contains(&child.id) {
                continue;
            }
            self.spawn_inherited_recursive(
                world,
                asset_server,
                child,
                instance_root,
                entity,
                removed,
                overrides,
                template_map,
            )?;
        }

        Ok(entity)
    }

    fn spawn_local_added(
        &mut self,
        world: &mut World,
        asset_server: &mut AssetServer,
        instance_root: Entity,
        added: &LocalAddedEntity,
        template_map: &HashMap<EntityId, Entity>,
    ) -> Result<()> {
        let parent = match &added.parent {
            LocalParent::InstanceRoot => instance_root,
            LocalParent::Template(template_id) => {
                *template_map
                    .get(template_id)
                    .ok_or_else(|| EngineError::AssetLoad {
                        path: template_id.to_string(),
                        reason: "local-added entity parents to a missing/removed template entity"
                            .to_owned(),
                    })?
            }
        };

        let entity = {
            let mut deserializer = SceneDeserializer::new(
                world,
                self.component_registry,
                self.type_registry,
                asset_server,
            );
            if let Some(external) = self.external {
                deserializer = deserializer.with_external_components(external);
            }
            let stub = SceneEntityData {
                id: added.entity.id,
                name: added.entity.name.clone(),
                components: added.entity.components.clone(),
                children: Vec::new(),
                instance: added.entity.instance.clone(),
            };
            deserializer
                .load_scene(&SceneFile {
                    version: SceneFile::CURRENT_VERSION,
                    name: "_local_stub".to_owned(),
                    entities: vec![stub],
                })?
                .into_iter()
                .next()
                .ok_or_else(|| EngineError::AssetLoad {
                    path: added.entity.id.to_string(),
                    reason: "failed to spawn local-added entity".to_owned(),
                })?
        };

        if let Ok(mut entity_ref) = world.get_entity_mut(entity) {
            entity_ref.insert(InstanceLocalEntity { instance_root });
        }
        attach_child(world, parent, entity);

        // Recurse into local children stored on the EntityData tree.
        for (index, child) in added.entity.children.iter().enumerate() {
            let nested = LocalAddedEntity {
                parent: LocalParent::Template(
                    // Use a synthetic approach: parent is the just-spawned local.
                    // Locals parent to locals via children list, not LocalParent.
                    EntityId::deterministic_fallback(&format!("unused/{index}")),
                ),
                entity: child.clone(),
            };
            let _ = nested;
            self.spawn_local_child_tree(world, asset_server, instance_root, entity, child)?;
        }

        Ok(())
    }

    fn spawn_local_child_tree(
        &mut self,
        world: &mut World,
        asset_server: &mut AssetServer,
        instance_root: Entity,
        parent: Entity,
        data: &SceneEntityData,
    ) -> Result<()> {
        let entity = {
            let mut deserializer = SceneDeserializer::new(
                world,
                self.component_registry,
                self.type_registry,
                asset_server,
            );
            if let Some(external) = self.external {
                deserializer = deserializer.with_external_components(external);
            }
            let stub = SceneEntityData {
                id: data.id,
                name: data.name.clone(),
                components: data.components.clone(),
                children: Vec::new(),
                instance: data.instance.clone(),
            };
            deserializer
                .load_scene(&SceneFile {
                    version: SceneFile::CURRENT_VERSION,
                    name: "_local_child_stub".to_owned(),
                    entities: vec![stub],
                })?
                .into_iter()
                .next()
                .ok_or_else(|| EngineError::AssetLoad {
                    path: data.id.to_string(),
                    reason: "failed to spawn local child".to_owned(),
                })?
        };
        if let Ok(mut entity_ref) = world.get_entity_mut(entity) {
            entity_ref.insert(InstanceLocalEntity { instance_root });
        }
        attach_child(world, parent, entity);
        for child in &data.children {
            self.spawn_local_child_tree(world, asset_server, instance_root, entity, child)?;
        }
        Ok(())
    }
}

fn instance_already_expanded(world: &World, instance_root: Entity) -> bool {
    world.iter_entities().any(|entity| {
        entity
            .get::<InheritedEntity>()
            .is_some_and(|inherited| inherited.instance_root == instance_root)
            || entity
                .get::<InstanceLocalEntity>()
                .is_some_and(|local| local.instance_root == instance_root)
    })
}

fn despawn_instance_contents(world: &mut World, instance_root: Entity) {
    let to_despawn: Vec<Entity> = world
        .iter_entities()
        .filter(|entity| {
            entity
                .get::<InheritedEntity>()
                .is_some_and(|inherited| inherited.instance_root == instance_root)
                || entity
                    .get::<InstanceLocalEntity>()
                    .is_some_and(|local| local.instance_root == instance_root)
        })
        .map(|entity| entity.id())
        .collect();

    for entity in to_despawn {
        detach_from_parent(world, entity);
        world.despawn(entity);
    }

    // Clear children list on the root of anything we removed.
    let remaining: Vec<Entity> = world
        .get::<Children>(instance_root)
        .map(|children| {
            children
                .0
                .iter()
                .copied()
                .filter(|child| world.get_entity(*child).is_ok())
                .collect()
        })
        .unwrap_or_default();
    if let Some(mut children) = world.get_mut::<Children>(instance_root) {
        children.0 = remaining;
    }
}

fn attach_child(world: &mut World, parent: Entity, child: Entity) {
    if let Ok(mut entity_ref) = world.get_entity_mut(child) {
        entity_ref.insert(Parent(parent));
    }
    if let Some(mut children) = world.get_mut::<Children>(parent) {
        if !children.0.contains(&child) {
            children.0.push(child);
        }
    } else if let Ok(mut entity_ref) = world.get_entity_mut(parent) {
        entity_ref.insert(Children(vec![child]));
    }
}

fn detach_from_parent(world: &mut World, child: Entity) {
    let parent = world.get::<Parent>(child).map(|p| p.0);
    if let Some(parent) = parent {
        if let Some(mut children) = world.get_mut::<Children>(parent) {
            children.0.retain(|id| *id != child);
        }
    }
}

fn load_template_scene(
    data: &SceneInstanceData,
    asset_server: &mut AssetServer,
) -> Result<SceneFile> {
    let relative = if let Some(database) = asset_server.database() {
        database
            .resolve_relative_path(data.scene)
            .map(str::to_owned)
            .or_else(|| data.scene_path.clone())
    } else {
        data.scene_path.clone()
    }
    .ok_or_else(|| EngineError::AssetLoad {
        path: data.scene.to_string(),
        reason: "cannot resolve nested scene asset (no path and no database entry)".to_owned(),
    })?;

    let asset_path = asset_server.resolve_path(&relative)?;
    let path = Path::new(asset_path.as_str());
    let source = std::fs::read_to_string(path).map_err(|error| EngineError::AssetLoad {
        path: path.display().to_string(),
        reason: error.to_string(),
    })?;
    let mut scene: SceneFile = ron::from_str(&source).map_err(|error| EngineError::AssetLoad {
        path: path.display().to_string(),
        reason: format!("failed to parse nested scene: {error}"),
    })?;
    scene.version = SceneFile::CURRENT_VERSION;
    Ok(scene)
}

fn group_overrides(overrides: &[OverrideEntry]) -> HashMap<EntityId, Vec<&OverrideEntry>> {
    let mut map: HashMap<EntityId, Vec<&OverrideEntry>> = HashMap::new();
    for entry in overrides {
        map.entry(entry.target).or_default().push(entry);
    }
    map
}

fn apply_override_map(
    base: &HashMap<String, SceneValue>,
    overrides: Option<&Vec<&OverrideEntry>>,
) -> HashMap<String, SceneValue> {
    let mut components = base.clone();
    let Some(overrides) = overrides else {
        return components;
    };
    for entry in overrides {
        if entry.field_path.is_empty() {
            components.insert(entry.component.clone(), entry.value.clone());
        } else {
            let current = components
                .entry(entry.component.clone())
                .or_insert_with(|| SceneValue::Map(ron::Map::new()));
            apply_field_override(current, &entry.field_path, &entry.value);
        }
    }
    components
}

fn apply_field_override(value: &mut SceneValue, field_path: &str, new_value: &SceneValue) {
    let mut parts = field_path.split('.').filter(|p| !p.is_empty());
    let Some(first) = parts.next() else {
        *value = new_value.clone();
        return;
    };
    let rest: Vec<&str> = parts.collect();
    if rest.is_empty() {
        if let SceneValue::Map(map) = value {
            let exists = map
                .iter()
                .any(|(k, _)| matches!(k, SceneValue::String(s) if s == first));
            if exists {
                if let Some((_, slot)) = map
                    .iter_mut()
                    .find(|(k, _)| matches!(k, SceneValue::String(s) if s == first))
                {
                    *slot = new_value.clone();
                }
            } else {
                map.insert(SceneValue::String(first.to_owned()), new_value.clone());
            }
        }
        return;
    }
    if let SceneValue::Map(map) = value {
        if let Some((_, slot)) = map
            .iter_mut()
            .find(|(k, _)| matches!(k, SceneValue::String(s) if s == first))
        {
            apply_field_override(slot, &rest.join("."), new_value);
        }
    }
}

/// Helper for tests / diagnostics: count entities that carry a name.
pub fn count_named(world: &World, name: &str) -> usize {
    world
        .iter_entities()
        .filter(|e| e.get::<EntityName>().is_some_and(|n| n.0 == name))
        .count()
}
