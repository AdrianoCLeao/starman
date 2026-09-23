use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::Path;

use bevy_ecs::entity::Entity;
use bevy_ecs::prelude::Component;
use bevy_ecs::world::World;
use engine_core::{
    Children, EngineError, EntityId, EntityName, Parent, PersistentId, Result, SourceAssetId,
};
use engine_reflect::bevy_reflect::{
    DynamicEnum, DynamicStruct, DynamicTuple, PartialReflect, ReflectMut, ReflectRef, VariantType,
};
use engine_reflect::{
    ComponentDescriptor, ComponentRegistry, ReflectMetadataRegistry, ReflectTypeRegistry,
};
use serde::{Deserialize, Serialize};

use crate::AssetServer;

mod instance_data;
mod migration;

pub use instance_data::{
    LocalAddedEntity, LocalParent, OverrideEntry, SceneInstanceData,
};

/// Live ECS marker for a nested-scene instance root. The authored payload is
/// [`SceneInstanceData`]; expansion of inherited children is performed by
/// `engine-scene`'s resolver after deserialize.
#[derive(Component, Clone, Debug, PartialEq)]
pub struct SceneInstance(pub SceneInstanceData);

/// Marks an entity that was spawned from a template inside a
/// [`SceneInstance`]. These entities are never written back into the parent
/// scene document — only their overrides / local adds are.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct InheritedEntity {
    pub template_id: EntityId,
    pub instance_root: Entity,
}

/// Marks an entity that was added locally on an instance (not part of the
/// source scene asset). Serialized under `SceneInstanceData::added`.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct InstanceLocalEntity {
    pub instance_root: Entity,
}

pub type SceneValue = ron::Value;
pub type SceneEntityData = EntityData;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct SceneFile {
    pub version: u32,
    pub name: String,
    pub entities: Vec<EntityData>,
}

impl SceneFile {
    /// Scene format version 3: nested scene instances with overrides
    /// (ADR 0007). Versions 1 and 2 migrate automatically on load, with a
    /// backup preserved next to the original file (ADR 0006).
    pub const CURRENT_VERSION: u32 = 3;

    /// Builds an empty, valid, current-version scene (no entities). Useful
    /// for scaffolding a project's entry scene on creation.
    pub fn empty(name: impl Into<String>) -> Self {
        Self {
            version: Self::CURRENT_VERSION,
            name: name.into(),
            entities: Vec::new(),
        }
    }

    /// Writes this scene to `path` using the same canonical, deterministic
    /// RON rendering as [`SceneSerializer::save_file`].
    pub fn write_to(&self, path: &Path) -> Result<()> {
        write_scene_ron(path, self)
    }

    /// Collects every nested scene [`SourceAssetId`] referenced by instance
    /// roots in this document (non-recursive into those assets).
    pub fn direct_scene_dependencies(&self) -> Vec<SourceAssetId> {
        let mut ids = Vec::new();
        collect_scene_deps(&self.entities, &mut ids);
        ids.sort();
        ids.dedup();
        ids
    }
}

fn collect_scene_deps(entities: &[EntityData], out: &mut Vec<SourceAssetId>) {
    for entity in entities {
        if let Some(instance) = &entity.instance {
            out.push(instance.scene);
            for added in &instance.added {
                collect_scene_deps(std::slice::from_ref(&added.entity), out);
            }
        }
        collect_scene_deps(&entity.children, out);
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct EntityData {
    pub id: EntityId,
    pub name: Option<String>,
    pub components: HashMap<String, SceneValue>,
    pub children: Vec<EntityData>,
    /// When set, this entity is an instance root of another scene asset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instance: Option<SceneInstanceData>,
}

#[derive(Serialize)]
struct StableSceneFile {
    version: u32,
    name: String,
    entities: Vec<StableEntityData>,
}

#[derive(Serialize)]
struct StableEntityData {
    id: EntityId,
    name: Option<String>,
    components: BTreeMap<String, SceneValue>,
    children: Vec<StableEntityData>,
    #[serde(skip_serializing_if = "Option::is_none")]
    instance: Option<SceneInstanceData>,
}

pub trait SceneExternalComponents {
    fn serialize_entity_components(
        &self,
        _world: &World,
        _entity: Entity,
        _asset_server: &AssetServer,
        _out: &mut HashMap<String, SceneValue>,
    ) -> Result<()> {
        Ok(())
    }

    fn deserialize_entity_component(
        &self,
        _world: &mut World,
        _entity: Entity,
        _component_name: &str,
        _component_value: &SceneValue,
        _asset_server: &mut AssetServer,
    ) -> Result<bool> {
        Ok(false)
    }
}

pub struct SceneSerializer<'w, 'a> {
    world: &'w World,
    component_registry: &'w ComponentRegistry,
    _type_registry: &'w ReflectTypeRegistry,
    _metadata_registry: Option<&'w ReflectMetadataRegistry>,
    asset_server: Option<&'a AssetServer>,
    external_components: Option<&'a dyn SceneExternalComponents>,
}

impl<'w, 'a> SceneSerializer<'w, 'a> {
    pub fn new(
        world: &'w World,
        component_registry: &'w ComponentRegistry,
        type_registry: &'w ReflectTypeRegistry,
    ) -> Self {
        Self {
            world,
            component_registry,
            _type_registry: type_registry,
            _metadata_registry: None,
            asset_server: None,
            external_components: None,
        }
    }

    pub fn with_metadata_registry(
        mut self,
        metadata_registry: &'w ReflectMetadataRegistry,
    ) -> Self {
        self._metadata_registry = Some(metadata_registry);
        self
    }

    pub fn with_asset_server(mut self, asset_server: &'a AssetServer) -> Self {
        self.asset_server = Some(asset_server);
        self
    }

    pub fn with_external_components(
        mut self,
        external_components: &'a dyn SceneExternalComponents,
    ) -> Self {
        self.external_components = Some(external_components);
        self
    }

    pub fn serialize_world(&self, scene_name: &str) -> Result<SceneFile> {
        let mut root_entities = Vec::new();

        for (index, entity_ref) in self.world.iter_entities().enumerate() {
            let entity = entity_ref.id();
            if entity_ref.get::<Parent>().is_some() {
                continue;
            }
            // Inherited entities are owned by a SceneInstance expansion —
            // never materialize them as native roots.
            if entity_ref.get::<InheritedEntity>().is_some() {
                continue;
            }
            if entity_ref.get::<InstanceLocalEntity>().is_some() {
                // Locals are serialized via the owning SceneInstance.added
                // list, not as independent roots.
                continue;
            }

            root_entities.push(self.serialize_entity(entity, &index.to_string())?);
        }

        Ok(SceneFile {
            version: SceneFile::CURRENT_VERSION,
            name: scene_name.to_owned(),
            entities: root_entities,
        })
    }

    pub fn save_file(&self, path: &Path, scene_name: &str) -> Result<SceneFile> {
        let scene = self.serialize_world(scene_name)?;
        write_scene_ron(path, &scene)?;
        Ok(scene)
    }

    fn serialize_entity(&self, entity: Entity, path: &str) -> Result<EntityData> {
        // Never walk into inherited children — they belong to the template.
        if self.world.get::<InheritedEntity>(entity).is_some() {
            return Err(EngineError::AssetLoad {
                path: path.to_owned(),
                reason: "internal error: attempted to serialize an inherited entity as native"
                    .to_owned(),
            });
        }

        let mut components = HashMap::new();

        for descriptor in self.component_registry.all() {
            if !descriptor.has(entity, self.world) {
                continue;
            }

            let component_name = descriptor_scene_name(descriptor);
            if should_skip_default_component(component_name) {
                continue;
            }

            let Some(reflect_value) = descriptor.get_reflect(entity, self.world) else {
                continue;
            };

            let serialized_value = match reflect_to_scene_value(reflect_value.as_partial_reflect())
            {
                Ok(value) => value,
                Err(error) => {
                    log::warn!(
                        target: "engine::assets",
                        "Skipping component '{}' on entity {:?} during scene serialization: {}",
                        component_name,
                        entity,
                        error
                    );
                    continue;
                }
            };
            components.insert(component_name.to_owned(), serialized_value);
        }

        if let (Some(external_components), Some(asset_server)) =
            (self.external_components, self.asset_server)
        {
            external_components.serialize_entity_components(
                self.world,
                entity,
                asset_server,
                &mut components,
            )?;
        }

        let name = self
            .world
            .get::<EntityName>(entity)
            .map(|value| value.0.clone());

        let instance = match self.world.get::<SceneInstance>(entity) {
            Some(value) => {
                let mut data = value.0.clone();
                data.added = collect_local_added(self, entity, path)?;
                data.normalize();
                Some(data)
            }
            None => None,
        };

        let children = self
            .world
            .get::<Children>(entity)
            .map(|children| {
                children
                    .0
                    .iter()
                    .copied()
                    .enumerate()
                    .filter_map(|(index, child)| {
                        // Skip inherited and instance-local children; locals
                        // are captured into SceneInstance.added above.
                        if self.world.get::<InheritedEntity>(child).is_some() {
                            return None;
                        }
                        if self.world.get::<InstanceLocalEntity>(child).is_some() {
                            return None;
                        }
                        Some(self.serialize_entity(child, &format!("{path}/{index}")))
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default();

        let id = self
            .world
            .get::<PersistentId>(entity)
            .map(|persistent_id| persistent_id.0)
            .unwrap_or_else(|| EntityId::deterministic_fallback(path));

        Ok(EntityData {
            id,
            name,
            components,
            children,
            instance,
        })
    }

    /// Serializes a local-added entity and its local-only descendants.
    fn serialize_local_entity(&self, entity: Entity, path: &str) -> Result<EntityData> {
        let mut components = HashMap::new();

        for descriptor in self.component_registry.all() {
            if !descriptor.has(entity, self.world) {
                continue;
            }

            let component_name = descriptor_scene_name(descriptor);
            if should_skip_default_component(component_name) {
                continue;
            }

            let Some(reflect_value) = descriptor.get_reflect(entity, self.world) else {
                continue;
            };

            let serialized_value = match reflect_to_scene_value(reflect_value.as_partial_reflect()) {
                Ok(value) => value,
                Err(error) => {
                    log::warn!(
                        target: "engine::assets",
                        "Skipping component '{}' on local entity {:?}: {}",
                        component_name,
                        entity,
                        error
                    );
                    continue;
                }
            };
            components.insert(component_name.to_owned(), serialized_value);
        }

        if let (Some(external_components), Some(asset_server)) =
            (self.external_components, self.asset_server)
        {
            external_components.serialize_entity_components(
                self.world,
                entity,
                asset_server,
                &mut components,
            )?;
        }

        let name = self
            .world
            .get::<EntityName>(entity)
            .map(|value| value.0.clone());

        let children = self
            .world
            .get::<Children>(entity)
            .map(|children| {
                children
                    .0
                    .iter()
                    .copied()
                    .enumerate()
                    .filter(|(_, child)| self.world.get::<InstanceLocalEntity>(*child).is_some())
                    .map(|(index, child)| {
                        self.serialize_local_entity(child, &format!("{path}/{index}"))
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default();

        let id = self
            .world
            .get::<PersistentId>(entity)
            .map(|persistent_id| persistent_id.0)
            .unwrap_or_else(|| EntityId::deterministic_fallback(path));

        Ok(EntityData {
            id,
            name,
            components,
            children,
            instance: None,
        })
    }
}

fn collect_local_added(
    serializer: &SceneSerializer<'_, '_>,
    instance_root: Entity,
    path: &str,
) -> Result<Vec<LocalAddedEntity>> {
    let world = serializer.world;
    let mut added = Vec::new();

    if let Some(children) = world.get::<Children>(instance_root) {
        for (index, child) in children.0.iter().copied().enumerate() {
            if world.get::<InstanceLocalEntity>(child).is_none() {
                continue;
            }
            let entity_data =
                serializer.serialize_local_entity(child, &format!("{path}/local{index}"))?;
            added.push(LocalAddedEntity {
                parent: LocalParent::InstanceRoot,
                entity: entity_data,
            });
        }
    }

    for entity_ref in world.iter_entities() {
        let entity = entity_ref.id();
        let Some(local) = entity_ref.get::<InstanceLocalEntity>() else {
            continue;
        };
        if local.instance_root != instance_root {
            continue;
        }
        let Some(parent) = entity_ref.get::<Parent>() else {
            continue;
        };
        if parent.0 == instance_root {
            continue;
        }
        let Some(inherited) = world.get::<InheritedEntity>(parent.0) else {
            continue;
        };
        if added.iter().any(|entry| {
            world
                .get::<PersistentId>(entity)
                .is_some_and(|id| id.0 == entry.entity.id)
        }) {
            continue;
        }
        let entity_data =
            serializer.serialize_local_entity(entity, &format!("{path}/local{}", entity.index()))?;
        if added.iter().any(|entry| entry.entity.id == entity_data.id) {
            continue;
        }
        added.push(LocalAddedEntity {
            parent: LocalParent::Template(inherited.template_id),
            entity: entity_data,
        });
    }

    Ok(added)
}

/// Renders a scene as the canonical, deterministically-ordered RON
/// representation and writes it to `path` via an atomic temp+rename.
/// Shared by [`SceneSerializer`] and scene migration so both produce
/// byte-identical output for the same data.
pub fn write_scene_ron(path: &Path, scene: &SceneFile) -> Result<()> {
    let mut normalized = scene.clone();
    normalize_scene_instances(&mut normalized);
    let stable_scene = to_stable_scene_file(&normalized);
    let pretty = ron::ser::PrettyConfig::default();
    let serialized = ron::ser::to_string_pretty(&stable_scene, pretty).map_err(|error| {
        EngineError::AssetLoad {
            path: path.display().to_string(),
            reason: format!("failed to serialize scene: {error}"),
        }
    })?;

    crate::import::write_atomic(path, serialized.as_bytes()).map_err(|error| {
        EngineError::AssetLoad {
            path: path.display().to_string(),
            reason: error.to_string(),
        }
    })
}

fn normalize_scene_instances(scene: &mut SceneFile) {
    fn walk(entities: &mut [EntityData]) {
        for entity in entities {
            if let Some(instance) = &mut entity.instance {
                instance.normalize();
            }
            walk(&mut entity.children);
            if let Some(instance) = &mut entity.instance {
                for added in &mut instance.added {
                    walk(std::slice::from_mut(&mut added.entity));
                }
            }
        }
    }
    walk(&mut scene.entities);
}

pub struct SceneDeserializer<'w, 'a> {
    world: &'w mut World,
    component_registry: &'w ComponentRegistry,
    _type_registry: &'w ReflectTypeRegistry,
    asset_server: &'w mut AssetServer,
    external_components: Option<&'a dyn SceneExternalComponents>,
}

impl<'w, 'a> SceneDeserializer<'w, 'a> {
    pub fn new(
        world: &'w mut World,
        component_registry: &'w ComponentRegistry,
        type_registry: &'w ReflectTypeRegistry,
        asset_server: &'w mut AssetServer,
    ) -> Self {
        Self {
            world,
            component_registry,
            _type_registry: type_registry,
            asset_server,
            external_components: None,
        }
    }

    pub fn with_external_components(
        mut self,
        external_components: &'a dyn SceneExternalComponents,
    ) -> Self {
        self.external_components = Some(external_components);
        self
    }

    pub fn load_file(&mut self, path: &Path) -> Result<Vec<Entity>> {
        let source = fs::read_to_string(path).map_err(|error| EngineError::AssetLoad {
            path: path.display().to_string(),
            reason: error.to_string(),
        })?;

        let probe: migration::VersionProbe =
            ron::from_str(&source).map_err(|error| EngineError::AssetLoad {
                path: path.display().to_string(),
                reason: format!("failed to parse scene file: {error}"),
            })?;

        let scene = if probe.version == SceneFile::CURRENT_VERSION {
            ron::from_str::<SceneFile>(&source).map_err(|error| EngineError::AssetLoad {
                path: path.display().to_string(),
                reason: format!("failed to parse scene file: {error}"),
            })?
        } else if probe.version == migration::LEGACY_VERSION_V1 {
            let legacy: migration::LegacySceneFileV1 =
                ron::from_str(&source).map_err(|error| EngineError::AssetLoad {
                    path: path.display().to_string(),
                    reason: format!("failed to parse legacy (v1) scene file: {error}"),
                })?;
            migration::migrate_v1_and_persist(path, &source, legacy)?
        } else if probe.version == migration::LEGACY_VERSION_V2 {
            let legacy: migration::LegacySceneFileV2 =
                ron::from_str(&source).map_err(|error| EngineError::AssetLoad {
                    path: path.display().to_string(),
                    reason: format!("failed to parse legacy (v2) scene file: {error}"),
                })?;
            migration::migrate_v2_and_persist(path, &source, legacy)?
        } else {
            return Err(EngineError::AssetLoad {
                path: path.display().to_string(),
                reason: format!(
                    "unsupported scene version {}; expected {} (or {} / {} for automatic migration)",
                    probe.version,
                    SceneFile::CURRENT_VERSION,
                    migration::LEGACY_VERSION_V1,
                    migration::LEGACY_VERSION_V2
                ),
            });
        };

        self.load_scene(&scene)
    }

    pub fn load_scene(&mut self, scene: &SceneFile) -> Result<Vec<Entity>> {
        if scene.version != SceneFile::CURRENT_VERSION {
            return Err(EngineError::AssetLoad {
                path: scene.name.clone(),
                reason: format!(
                    "unsupported scene version {}; expected {}",
                    scene.version,
                    SceneFile::CURRENT_VERSION
                ),
            });
        }

        let mut roots = Vec::with_capacity(scene.entities.len());
        for root in &scene.entities {
            roots.push(self.spawn_entity_recursive(root, None)?);
        }

        Ok(roots)
    }

    fn spawn_entity_recursive(
        &mut self,
        data: &EntityData,
        parent: Option<Entity>,
    ) -> Result<Entity> {
        let entity = self.world.spawn_empty().id();

        if let Ok(mut entity_ref) = self.world.get_entity_mut(entity) {
            entity_ref.insert(PersistentId(data.id));
        }

        if let Some(name) = &data.name {
            if let Ok(mut entity_ref) = self.world.get_entity_mut(entity) {
                entity_ref.insert(EntityName::new(name.clone()));
            }
        }

        for (component_name, component_value) in &data.components {
            if should_skip_default_component(component_name) {
                continue;
            }

            let handled_by_external =
                self.try_deserialize_external_component(entity, component_name, component_value)?;
            if handled_by_external {
                continue;
            }

            self.apply_registered_component(entity, component_name, component_value)?;
        }

        if let Some(instance) = &data.instance {
            if let Ok(mut entity_ref) = self.world.get_entity_mut(entity) {
                entity_ref.insert(SceneInstance(instance.clone()));
            }
        }

        if let Some(parent_entity) = parent {
            if let Ok(mut entity_ref) = self.world.get_entity_mut(entity) {
                entity_ref.insert(Parent(parent_entity));
            }
        }

        let mut child_entities = Vec::with_capacity(data.children.len());
        for child in &data.children {
            let child_entity = self.spawn_entity_recursive(child, Some(entity))?;
            child_entities.push(child_entity);
        }

        if !child_entities.is_empty() {
            if let Ok(mut entity_ref) = self.world.get_entity_mut(entity) {
                entity_ref.insert(Children(child_entities));
            }
        }

        Ok(entity)
    }

    fn try_deserialize_external_component(
        &mut self,
        entity: Entity,
        component_name: &str,
        component_value: &SceneValue,
    ) -> Result<bool> {
        let Some(external_components) = self.external_components else {
            return Ok(false);
        };

        external_components.deserialize_entity_component(
            self.world,
            entity,
            component_name,
            component_value,
            self.asset_server,
        )
    }

    fn apply_registered_component(
        &mut self,
        entity: Entity,
        component_name: &str,
        component_value: &SceneValue,
    ) -> Result<()> {
        let Some(descriptor) =
            find_descriptor_by_scene_name(self.component_registry, component_name)
        else {
            log::warn!(
                target: "engine::assets",
                "Scene references unknown component '{}' ; skipping",
                component_name,
            );
            return Ok(());
        };

        if !descriptor.insert_default(entity, self.world) {
            log::warn!(
                target: "engine::assets",
                "Failed to insert default component '{}' on entity {:?}",
                component_name,
                entity
            );
            return Ok(());
        }

        let Some(reflect_component) = descriptor.get_reflect_mut(entity, self.world) else {
            log::warn!(
                target: "engine::assets",
                "Inserted component '{}' is not reflect-mutable on entity {:?}",
                component_name,
                entity
            );
            return Ok(());
        };

        if let Err(error) = apply_scene_value_to_partial_reflect(
            reflect_component,
            component_value,
            descriptor.name,
        ) {
            log::warn!(
                target: "engine::assets",
                "Skipping component '{}' on entity {:?}: {}",
                component_name,
                entity,
                error
            );
            let _ = descriptor.remove(entity, self.world);
        }

        Ok(())
    }
}

fn descriptor_scene_name(descriptor: &ComponentDescriptor) -> &'static str {
    let short_name = short_type_name(descriptor.name);
    match short_name {
        "MeshRenderable3d" => "MeshRenderer",
        "SpriteRenderable2d" => "Sprite",
        _ => short_name,
    }
}

fn should_skip_default_component(component_name: &str) -> bool {
    matches!(
        component_name,
        "EntityName"
            | "Parent"
            | "Children"
            | "GlobalTransform"
            | "RigidBodyHandle3D"
            | "ColliderHandle3D"
    )
}

fn short_type_name(type_path: &str) -> &str {
    type_path.rsplit("::").next().unwrap_or(type_path)
}

fn find_descriptor_by_scene_name<'a>(
    component_registry: &'a ComponentRegistry,
    component_name: &str,
) -> Option<&'a ComponentDescriptor> {
    component_registry.all().iter().find(|descriptor| {
        descriptor.name == component_name || descriptor_scene_name(descriptor) == component_name
    })
}

fn reflect_to_scene_value(value: &dyn PartialReflect) -> Result<SceneValue> {
    if let Some(v) = value.try_downcast_ref::<bool>() {
        return Ok(SceneValue::Bool(*v));
    }
    if let Some(v) = value.try_downcast_ref::<String>() {
        return Ok(SceneValue::String(v.clone()));
    }
    if let Some(v) = value.try_downcast_ref::<f32>() {
        return Ok(SceneValue::Number(ron::Number::new(*v as f64)));
    }
    if let Some(v) = value.try_downcast_ref::<f64>() {
        return Ok(SceneValue::Number(ron::Number::new(*v)));
    }
    if let Some(v) = value.try_downcast_ref::<i32>() {
        return Ok(SceneValue::Number(ron::Number::new(*v as i64)));
    }
    if let Some(v) = value.try_downcast_ref::<i64>() {
        return Ok(SceneValue::Number(ron::Number::new(*v)));
    }
    if let Some(v) = value.try_downcast_ref::<u32>() {
        return Ok(SceneValue::Number(ron::Number::new(*v as i64)));
    }
    if let Some(v) = value.try_downcast_ref::<u64>() {
        let value_i64 = i64::try_from(*v).map_err(|_| EngineError::AssetLoad {
            path: "scene".to_owned(),
            reason: format!("u64 value '{}' exceeds i64 range in scene value", v),
        })?;
        return Ok(SceneValue::Number(ron::Number::new(value_i64)));
    }

    match value.reflect_ref() {
        ReflectRef::Struct(data) => {
            let mut map = ron::Map::new();
            for index in 0..data.field_len() {
                let Some(field_name) = data.name_at(index) else {
                    continue;
                };
                let Some(field_value) = data.field_at(index) else {
                    continue;
                };

                map.insert(
                    SceneValue::String(field_name.to_owned()),
                    reflect_to_scene_value(field_value)?,
                );
            }

            Ok(SceneValue::Map(map))
        }
        ReflectRef::TupleStruct(data) => {
            let mut seq = Vec::with_capacity(data.field_len());
            for index in 0..data.field_len() {
                let Some(field_value) = data.field(index) else {
                    continue;
                };
                seq.push(reflect_to_scene_value(field_value)?);
            }
            Ok(SceneValue::Seq(seq))
        }
        ReflectRef::Tuple(data) => {
            let mut seq = Vec::with_capacity(data.field_len());
            for index in 0..data.field_len() {
                let Some(field_value) = data.field(index) else {
                    continue;
                };
                seq.push(reflect_to_scene_value(field_value)?);
            }
            Ok(SceneValue::Seq(seq))
        }
        ReflectRef::List(data) => {
            let mut seq = Vec::with_capacity(data.len());
            for index in 0..data.len() {
                let Some(entry) = data.get(index) else {
                    continue;
                };
                seq.push(reflect_to_scene_value(entry)?);
            }
            Ok(SceneValue::Seq(seq))
        }
        ReflectRef::Array(data) => {
            let mut seq = Vec::with_capacity(data.len());
            for index in 0..data.len() {
                let Some(entry) = data.get(index) else {
                    continue;
                };
                seq.push(reflect_to_scene_value(entry)?);
            }
            Ok(SceneValue::Seq(seq))
        }
        ReflectRef::Map(data) => {
            let mut map = ron::Map::new();
            for (key, entry) in data.iter() {
                map.insert(reflect_to_scene_value(key)?, reflect_to_scene_value(entry)?);
            }
            Ok(SceneValue::Map(map))
        }
        ReflectRef::Set(data) => {
            let mut seq = Vec::with_capacity(data.len());
            for value in data.iter() {
                seq.push(reflect_to_scene_value(value)?);
            }
            Ok(SceneValue::Seq(seq))
        }
        ReflectRef::Enum(data) => enum_to_scene_value(data),
        ReflectRef::Opaque(_) => Err(EngineError::AssetLoad {
            path: "scene".to_owned(),
            reason: format!(
                "unsupported opaque reflected value '{}'",
                value.reflect_type_path()
            ),
        }),
    }
}

fn enum_to_scene_value(value: &dyn engine_reflect::bevy_reflect::Enum) -> Result<SceneValue> {
    match value.variant_type() {
        VariantType::Unit => Ok(SceneValue::String(value.variant_name().to_owned())),
        VariantType::Tuple => {
            let mut tuple_values = Vec::with_capacity(value.field_len());
            for index in 0..value.field_len() {
                let Some(field_value) = value.field_at(index) else {
                    continue;
                };
                tuple_values.push(reflect_to_scene_value(field_value)?);
            }

            Ok(map_value(vec![
                (
                    "$variant",
                    SceneValue::String(value.variant_name().to_owned()),
                ),
                ("$tuple", SceneValue::Seq(tuple_values)),
            ]))
        }
        VariantType::Struct => {
            let mut field_map = ron::Map::new();
            for index in 0..value.field_len() {
                let Some(field_name) = value.name_at(index) else {
                    continue;
                };
                let Some(field_value) = value.field_at(index) else {
                    continue;
                };
                field_map.insert(
                    SceneValue::String(field_name.to_owned()),
                    reflect_to_scene_value(field_value)?,
                );
            }

            Ok(map_value(vec![
                (
                    "$variant",
                    SceneValue::String(value.variant_name().to_owned()),
                ),
                ("$fields", SceneValue::Map(field_map)),
            ]))
        }
    }
}

fn apply_scene_value_to_partial_reflect(
    target: &mut dyn PartialReflect,
    value: &SceneValue,
    context: &str,
) -> Result<()> {
    if let Some(target) = target.try_downcast_mut::<bool>() {
        let SceneValue::Bool(source) = value else {
            return Err(type_mismatch_error(context, "bool", value));
        };
        *target = *source;
        return Ok(());
    }

    if let Some(target) = target.try_downcast_mut::<String>() {
        let SceneValue::String(source) = value else {
            return Err(type_mismatch_error(context, "String", value));
        };
        *target = source.clone();
        return Ok(());
    }

    if let Some(target) = target.try_downcast_mut::<f32>() {
        *target = scene_number_to_f64(value, context)? as f32;
        return Ok(());
    }
    if let Some(target) = target.try_downcast_mut::<f64>() {
        *target = scene_number_to_f64(value, context)?;
        return Ok(());
    }
    if let Some(target) = target.try_downcast_mut::<i32>() {
        *target = i32::try_from(scene_number_to_i64(value, context)?).map_err(|_| {
            EngineError::AssetLoad {
                path: context.to_owned(),
                reason: "number is out of range for i32".to_owned(),
            }
        })?;
        return Ok(());
    }
    if let Some(target) = target.try_downcast_mut::<i64>() {
        *target = scene_number_to_i64(value, context)?;
        return Ok(());
    }
    if let Some(target) = target.try_downcast_mut::<u32>() {
        *target = u32::try_from(scene_number_to_i64(value, context)?).map_err(|_| {
            EngineError::AssetLoad {
                path: context.to_owned(),
                reason: "number is out of range for u32".to_owned(),
            }
        })?;
        return Ok(());
    }
    if let Some(target) = target.try_downcast_mut::<u64>() {
        *target = u64::try_from(scene_number_to_i64(value, context)?).map_err(|_| {
            EngineError::AssetLoad {
                path: context.to_owned(),
                reason: "number is out of range for u64".to_owned(),
            }
        })?;
        return Ok(());
    }

    match target.reflect_mut() {
        ReflectMut::Struct(data) => {
            let SceneValue::Map(map) = value else {
                return Err(type_mismatch_error(context, "struct/map", value));
            };

            for (key, map_value) in map.iter() {
                let SceneValue::String(field_name) = key else {
                    continue;
                };

                if let Some(field) = data.field_mut(field_name) {
                    apply_scene_value_to_partial_reflect(field, map_value, context)?;
                }
            }

            Ok(())
        }
        ReflectMut::TupleStruct(data) => {
            let SceneValue::Seq(values) = value else {
                return Err(type_mismatch_error(context, "tuple struct/seq", value));
            };

            for (index, source_value) in values.iter().enumerate() {
                if let Some(field) = data.field_mut(index) {
                    apply_scene_value_to_partial_reflect(field, source_value, context)?;
                }
            }

            Ok(())
        }
        ReflectMut::Tuple(data) => {
            let SceneValue::Seq(values) = value else {
                return Err(type_mismatch_error(context, "tuple/seq", value));
            };

            for (index, source_value) in values.iter().enumerate() {
                if let Some(field) = data.field_mut(index) {
                    apply_scene_value_to_partial_reflect(field, source_value, context)?;
                }
            }

            Ok(())
        }
        ReflectMut::List(data) => {
            let SceneValue::Seq(values) = value else {
                return Err(type_mismatch_error(context, "list/seq", value));
            };

            for (index, source_value) in values.iter().enumerate() {
                if let Some(field) = data.get_mut(index) {
                    apply_scene_value_to_partial_reflect(field, source_value, context)?;
                }
            }

            Ok(())
        }
        ReflectMut::Array(data) => {
            let SceneValue::Seq(values) = value else {
                return Err(type_mismatch_error(context, "array/seq", value));
            };

            for (index, source_value) in values.iter().enumerate() {
                if let Some(field) = data.get_mut(index) {
                    apply_scene_value_to_partial_reflect(field, source_value, context)?;
                }
            }

            Ok(())
        }
        ReflectMut::Map(_) => Err(EngineError::AssetLoad {
            path: context.to_owned(),
            reason: "map reflection targets are not yet supported by scene deserializer".to_owned(),
        }),
        ReflectMut::Set(_) => Err(EngineError::AssetLoad {
            path: context.to_owned(),
            reason: "set reflection targets are not yet supported by scene deserializer".to_owned(),
        }),
        ReflectMut::Enum(enum_reflect) => {
            let enum_value = scene_value_to_dynamic_enum(value, context)?;
            enum_reflect.apply(enum_value.as_partial_reflect());
            Ok(())
        }
        ReflectMut::Opaque(_) => Err(EngineError::AssetLoad {
            path: context.to_owned(),
            reason: "opaque reflection target is not supported by scene deserializer".to_owned(),
        }),
    }
}

fn scene_value_to_dynamic_enum(value: &SceneValue, context: &str) -> Result<DynamicEnum> {
    let (variant_name, variant_payload) = match value {
        SceneValue::String(variant_name) => {
            return Ok(DynamicEnum::new(variant_name.clone(), ()));
        }
        SceneValue::Map(map) => {
            let Some(SceneValue::String(variant_name)) = map_get(map, "$variant") else {
                return Err(EngineError::AssetLoad {
                    path: context.to_owned(),
                    reason: "enum map value is missing '$variant' string".to_owned(),
                });
            };
            (variant_name.clone(), map)
        }
        _ => {
            return Err(EngineError::AssetLoad {
                path: context.to_owned(),
                reason: format!("unsupported enum scene value '{value:?}'"),
            });
        }
    };

    if let Some(SceneValue::Map(fields)) = map_get(variant_payload, "$fields") {
        let mut dynamic_struct = DynamicStruct::default();
        for (key, field_value) in fields.iter() {
            let SceneValue::String(field_name) = key else {
                continue;
            };
            dynamic_struct.insert_boxed(field_name, scene_value_to_dynamic(field_value, context)?);
        }

        return Ok(DynamicEnum::new(variant_name, dynamic_struct));
    }

    if let Some(SceneValue::Seq(fields)) = map_get(variant_payload, "$tuple") {
        let mut dynamic_tuple = DynamicTuple::default();
        for field_value in fields {
            dynamic_tuple.insert_boxed(scene_value_to_dynamic(field_value, context)?);
        }

        return Ok(DynamicEnum::new(variant_name, dynamic_tuple));
    }

    Ok(DynamicEnum::new(variant_name, ()))
}

fn scene_value_to_dynamic(value: &SceneValue, context: &str) -> Result<Box<dyn PartialReflect>> {
    match value {
        SceneValue::Bool(value) => Ok(Box::new(*value)),
        SceneValue::String(value) => Ok(Box::new(value.clone())),
        SceneValue::Number(value) => {
            if let Some(v) = (*value).as_f64() {
                return Ok(Box::new(v as f32));
            }
            Ok(Box::new((*value).into_f64() as f32))
        }
        SceneValue::Seq(values) => {
            let mut dynamic_tuple = DynamicTuple::default();
            for value in values {
                dynamic_tuple.insert_boxed(scene_value_to_dynamic(value, context)?);
            }
            Ok(Box::new(dynamic_tuple))
        }
        SceneValue::Map(values) => {
            let mut dynamic_struct = DynamicStruct::default();
            for (key, value) in values.iter() {
                let SceneValue::String(field_name) = key else {
                    continue;
                };
                dynamic_struct.insert_boxed(field_name, scene_value_to_dynamic(value, context)?);
            }
            Ok(Box::new(dynamic_struct))
        }
        SceneValue::Option(None) => Err(EngineError::AssetLoad {
            path: context.to_owned(),
            reason: "optional values are not currently supported in dynamic enum conversion"
                .to_owned(),
        }),
        SceneValue::Option(Some(inner)) => scene_value_to_dynamic(inner, context),
        SceneValue::Char(value) => Ok(Box::new(*value)),
        SceneValue::Unit => Ok(Box::new(())),
    }
}

fn scene_number_to_f64(value: &SceneValue, context: &str) -> Result<f64> {
    let SceneValue::Number(number) = value else {
        return Err(type_mismatch_error(context, "number", value));
    };

    Ok((*number).into_f64())
}

fn scene_number_to_i64(value: &SceneValue, context: &str) -> Result<i64> {
    let SceneValue::Number(number) = value else {
        return Err(type_mismatch_error(context, "number", value));
    };

    if let Some(value) = (*number).as_i64() {
        return Ok(value);
    }

    Err(EngineError::AssetLoad {
        path: context.to_owned(),
        reason: format!("expected integer number but found '{number:?}'"),
    })
}

fn type_mismatch_error(context: &str, expected: &str, found: &SceneValue) -> EngineError {
    EngineError::AssetLoad {
        path: context.to_owned(),
        reason: format!("expected {expected} but found '{found:?}'"),
    }
}

fn map_value(entries: Vec<(&str, SceneValue)>) -> SceneValue {
    let mut map = ron::Map::new();
    for (key, value) in entries {
        map.insert(SceneValue::String(key.to_owned()), value);
    }
    SceneValue::Map(map)
}

fn map_get<'a>(map: &'a ron::Map, key: &str) -> Option<&'a SceneValue> {
    map.iter().find_map(|(current_key, value)| {
        let SceneValue::String(current_key) = current_key else {
            return None;
        };

        if current_key == key {
            return Some(value);
        }

        None
    })
}

fn to_stable_scene_file(scene: &SceneFile) -> StableSceneFile {
    StableSceneFile {
        version: scene.version,
        name: scene.name.clone(),
        entities: scene.entities.iter().map(to_stable_entity_data).collect(),
    }
}

fn to_stable_entity_data(entity: &EntityData) -> StableEntityData {
    let instance = entity.instance.as_ref().map(|data| {
        let mut normalized = data.clone();
        normalized.normalize();
        normalized
    });

    StableEntityData {
        id: entity.id,
        name: entity.name.clone(),
        components: entity
            .components
            .iter()
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect(),
        children: entity.children.iter().map(to_stable_entity_data).collect(),
        instance,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_version_constant_is_stable() {
        assert_eq!(SceneFile::CURRENT_VERSION, 3);
    }
}
