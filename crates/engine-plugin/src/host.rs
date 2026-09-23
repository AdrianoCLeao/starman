//! Broad host command/query bus shared by native plugins and Lua.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;
use engine_core::{Children, EntityId, EntityName, Parent, PersistentId, Transform};
use engine_reflect::{ComponentRegistry, ReflectTypeRegistry};
use serde_json::Value as JsonValue;
use starman_plugin_sdk::EntityHandle;
use thiserror::Error;

use crate::permissions::PermissionGuard;
use crate::registry::DynamicRegistration;
use crate::PluginId;

#[derive(Debug, Error)]
pub enum HostError {
    #[error("{0}")]
    Message(String),
    #[error("permission denied: {0}")]
    PermissionDenied(String),
    #[error("entity handle {0} is invalid")]
    InvalidEntity(u64),
    #[error("unknown component '{0}'")]
    UnknownComponent(String),
}

pub type HostResult<T> = std::result::Result<T, HostError>;

/// JSON-compatible host value (broad reflect bridge).
pub type HostValue = JsonValue;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleName {
    Startup,
    FixedUpdate,
    Update,
    PreRender,
}

impl ScheduleName {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Startup => "Startup",
            Self::FixedUpdate => "FixedUpdate",
            Self::Update => "Update",
            Self::PreRender => "PreRender",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "Startup" | "startup" => Some(Self::Startup),
            "FixedUpdate" | "fixed_update" => Some(Self::FixedUpdate),
            "Update" | "update" => Some(Self::Update),
            "PreRender" | "pre_render" => Some(Self::PreRender),
            _ => None,
        }
    }
}

pub enum HostCommand {
    Spawn {
        name: Option<String>,
    },
    Despawn {
        entity: EntityHandle,
    },
    SetParent {
        entity: EntityHandle,
        parent: Option<EntityHandle>,
    },
    InsertComponent {
        entity: EntityHandle,
        component: String,
    },
    RemoveComponent {
        entity: EntityHandle,
        component: String,
    },
    SetField {
        entity: EntityHandle,
        component: String,
        field_path: String,
        value: HostValue,
    },
    Log {
        level: u32,
        message: String,
    },
    RegisterSystem {
        owner: PluginId,
        schedule: ScheduleName,
        name: String,
        /// Opaque callback id interpreted by the caller (Lua or plugin).
        callback_id: u64,
    },
    RegisterComponentMeta {
        owner: PluginId,
        component: String,
    },
}

pub enum HostQuery {
    EntityCount,
    FindByPersistentId(EntityId),
    GetField {
        entity: EntityHandle,
        component: String,
        field_path: String,
    },
    ListComponentNames,
    HasComponent {
        entity: EntityHandle,
        component: String,
    },
}

/// Shared host state accessible from FFI callbacks and Lua.
pub struct HostBus {
    pub permissions: PermissionGuard,
    pub handles: HandleTable,
    pub registrations: DynamicRegistration,
    pub logs: Vec<(u32, String)>,
    /// System callbacks queued by plugins/Lua: (owner, schedule, name, callback_id).
    pub system_callbacks: Vec<(PluginId, ScheduleName, String, u64)>,
}

impl HostBus {
    pub fn new(permissions: PermissionGuard) -> Self {
        Self {
            permissions,
            handles: HandleTable::default(),
            registrations: DynamicRegistration::default(),
            logs: Vec::new(),
            system_callbacks: Vec::new(),
        }
    }

    pub fn execute(
        &mut self,
        world: &mut World,
        component_registry: &ComponentRegistry,
        command: HostCommand,
    ) -> HostResult<HostValue> {
        match command {
            HostCommand::Spawn { name } => {
                let mut entity = world.spawn(Transform::IDENTITY);
                if let Some(name) = name {
                    entity.insert(EntityName::new(name));
                }
                let id = EntityId::new_v4();
                entity.insert(PersistentId(id));
                let entity = entity.id();
                let handle = self.handles.alloc(entity);
                Ok(JsonValue::Number(handle.0.into()))
            }
            HostCommand::Despawn { entity } => {
                let entity = self.handles.resolve(entity)?;
                world.despawn(entity);
                self.handles.invalidate(entity);
                Ok(JsonValue::Bool(true))
            }
            HostCommand::SetParent { entity, parent } => {
                let entity = self.handles.resolve(entity)?;
                if let Some(parent_handle) = parent {
                    let parent = self.handles.resolve(parent_handle)?;
                    if let Ok(mut e) = world.get_entity_mut(entity) {
                        e.insert(Parent(parent));
                    }
                    if let Some(mut children) = world.get_mut::<Children>(parent) {
                        if !children.0.contains(&entity) {
                            children.0.push(entity);
                        }
                    } else if let Ok(mut p) = world.get_entity_mut(parent) {
                        p.insert(Children(vec![entity]));
                    }
                } else {
                    let old_parent = world.get::<Parent>(entity).map(|p| p.0);
                    if let Some(old_parent) = old_parent {
                        if let Some(mut children) = world.get_mut::<Children>(old_parent) {
                            children.0.retain(|c| *c != entity);
                        }
                    }
                    if let Ok(mut e) = world.get_entity_mut(entity) {
                        e.remove::<Parent>();
                    }
                }
                Ok(JsonValue::Bool(true))
            }
            HostCommand::InsertComponent { entity, component } => {
                let entity = self.handles.resolve(entity)?;
                let descriptor = find_component(component_registry, &component)
                    .ok_or(HostError::UnknownComponent(component))?;
                if !descriptor.insert_default(entity, world) {
                    return Err(HostError::Message(format!(
                        "failed to insert component '{}'",
                        descriptor.name
                    )));
                }
                Ok(JsonValue::Bool(true))
            }
            HostCommand::RemoveComponent { entity, component } => {
                let entity = self.handles.resolve(entity)?;
                let descriptor = find_component(component_registry, &component)
                    .ok_or(HostError::UnknownComponent(component))?;
                let _ = descriptor.remove(entity, world);
                Ok(JsonValue::Bool(true))
            }
            HostCommand::SetField {
                entity,
                component,
                field_path,
                value,
            } => {
                let entity = self.handles.resolve(entity)?;
                set_field_json(
                    world,
                    component_registry,
                    entity,
                    &component,
                    &field_path,
                    &value,
                )?;
                Ok(JsonValue::Bool(true))
            }
            HostCommand::Log { level, message } => {
                match level {
                    1 => log::error!(target: "engine::plugin", "{message}"),
                    2 => log::warn!(target: "engine::plugin", "{message}"),
                    3 => log::info!(target: "engine::plugin", "{message}"),
                    _ => log::debug!(target: "engine::plugin", "{message}"),
                }
                self.logs.push((level, message));
                Ok(JsonValue::Null)
            }
            HostCommand::RegisterSystem {
                owner,
                schedule,
                name,
                callback_id,
            } => {
                self.registrations
                    .record_system(owner, schedule.as_str(), &name);
                self.system_callbacks
                    .push((owner, schedule, name, callback_id));
                Ok(JsonValue::Bool(true))
            }
            HostCommand::RegisterComponentMeta { owner, component } => {
                self.registrations.record_component(owner, &component);
                Ok(JsonValue::Bool(true))
            }
        }
    }

    pub fn query(
        &mut self,
        world: &World,
        component_registry: &ComponentRegistry,
        _type_registry: &ReflectTypeRegistry,
        query: HostQuery,
    ) -> HostResult<HostValue> {
        match query {
            HostQuery::EntityCount => Ok(JsonValue::Number(world.iter_entities().count().into())),
            HostQuery::FindByPersistentId(id) => {
                for entity_ref in world.iter_entities() {
                    if entity_ref.get::<PersistentId>().is_some_and(|p| p.0 == id) {
                        let handle = self.handles.alloc(entity_ref.id());
                        return Ok(JsonValue::Number(handle.0.into()));
                    }
                }
                Ok(JsonValue::Null)
            }
            HostQuery::GetField {
                entity,
                component,
                field_path,
            } => {
                let entity = self.handles.resolve(entity)?;
                get_field_json(world, component_registry, entity, &component, &field_path)
            }
            HostQuery::ListComponentNames => {
                let names: Vec<JsonValue> = component_registry
                    .all()
                    .iter()
                    .map(|d| JsonValue::String(short_name(d.name).to_owned()))
                    .collect();
                Ok(JsonValue::Array(names))
            }
            HostQuery::HasComponent { entity, component } => {
                let entity = self.handles.resolve(entity)?;
                let Some(descriptor) = find_component(component_registry, &component) else {
                    return Ok(JsonValue::Bool(false));
                };
                Ok(JsonValue::Bool(descriptor.has(entity, world)))
            }
        }
    }
}

#[derive(Default)]
pub struct HandleTable {
    next: u64,
    to_entity: HashMap<u64, Entity>,
    to_handle: HashMap<Entity, u64>,
}

impl HandleTable {
    pub fn alloc(&mut self, entity: Entity) -> EntityHandle {
        if let Some(existing) = self.to_handle.get(&entity) {
            return EntityHandle(*existing);
        }
        self.next = self.next.saturating_add(1);
        let id = self.next;
        self.to_entity.insert(id, entity);
        self.to_handle.insert(entity, id);
        EntityHandle(id)
    }

    pub fn resolve(&self, handle: EntityHandle) -> HostResult<Entity> {
        self.to_entity
            .get(&handle.0)
            .copied()
            .ok_or(HostError::InvalidEntity(handle.0))
    }

    pub fn invalidate(&mut self, entity: Entity) {
        if let Some(handle) = self.to_handle.remove(&entity) {
            self.to_entity.remove(&handle);
        }
    }
}

/// Thread-safe wrapper used from FFI.
pub type SharedHostBus = Arc<Mutex<HostBus>>;

fn find_component<'a>(
    registry: &'a ComponentRegistry,
    name: &str,
) -> Option<&'a engine_reflect::ComponentDescriptor> {
    registry
        .all()
        .iter()
        .find(|d| d.name == name || short_name(d.name) == name)
}

fn short_name(type_path: &str) -> &str {
    type_path.rsplit("::").next().unwrap_or(type_path)
}

fn set_field_json(
    world: &mut World,
    registry: &ComponentRegistry,
    entity: Entity,
    component: &str,
    field_path: &str,
    value: &HostValue,
) -> HostResult<()> {
    let descriptor = find_component(registry, component)
        .ok_or_else(|| HostError::UnknownComponent(component.to_owned()))?;
    if !descriptor.has(entity, world) && !descriptor.insert_default(entity, world) {
        return Err(HostError::Message(format!(
            "component '{component}' missing and could not be inserted"
        )));
    }
    let Some(reflect) = descriptor.get_reflect_mut(entity, world) else {
        return Err(HostError::Message(format!(
            "component '{component}' is not reflect-mutable"
        )));
    };

    if field_path.is_empty() {
        return Err(HostError::Message(
            "whole-component JSON apply is not supported; provide a field_path".to_owned(),
        ));
    }

    apply_json_to_reflect(reflect.as_partial_reflect_mut(), field_path, value)
}

fn get_field_json(
    world: &World,
    registry: &ComponentRegistry,
    entity: Entity,
    component: &str,
    field_path: &str,
) -> HostResult<HostValue> {
    let descriptor = find_component(registry, component)
        .ok_or_else(|| HostError::UnknownComponent(component.to_owned()))?;
    let Some(reflect) = descriptor.get_reflect(entity, world) else {
        return Ok(JsonValue::Null);
    };
    if field_path.is_empty() {
        return reflect_to_json(reflect.as_partial_reflect());
    }
    let field = walk_field(reflect.as_partial_reflect(), field_path)?;
    reflect_to_json(field)
}

fn walk_field<'a>(
    value: &'a dyn engine_reflect::bevy_reflect::PartialReflect,
    path: &str,
) -> HostResult<&'a dyn engine_reflect::bevy_reflect::PartialReflect> {
    use engine_reflect::bevy_reflect::ReflectRef;
    let mut current = value;
    for part in path.split('.').filter(|p| !p.is_empty()) {
        current = match current.reflect_ref() {
            ReflectRef::Struct(data) => data
                .field(part)
                .ok_or_else(|| HostError::Message(format!("missing field '{part}'")))?,
            ReflectRef::TupleStruct(data) => {
                let index: usize = part
                    .parse()
                    .map_err(|_| HostError::Message(format!("invalid tuple index '{part}'")))?;
                data.field(index)
                    .ok_or_else(|| HostError::Message(format!("missing tuple field {index}")))?
            }
            _ => {
                return Err(HostError::Message(format!(
                    "cannot walk into field '{part}'"
                )));
            }
        };
    }
    Ok(current)
}

fn apply_json_to_reflect(
    root: &mut dyn engine_reflect::bevy_reflect::PartialReflect,
    path: &str,
    value: &HostValue,
) -> HostResult<()> {
    use engine_reflect::bevy_reflect::ReflectMut;

    let parts: Vec<&str> = path.split('.').filter(|p| !p.is_empty()).collect();
    if parts.is_empty() {
        return apply_json_leaf(root, value);
    }

    // Navigate to parent of leaf.
    let mut indices = Vec::new();
    {
        let mut current = root.as_partial_reflect();
        for (i, part) in parts.iter().enumerate() {
            if i + 1 == parts.len() {
                break;
            }
            match current.reflect_ref() {
                engine_reflect::bevy_reflect::ReflectRef::Struct(data) => {
                    let idx = (0..data.field_len())
                        .find(|&i| data.name_at(i) == Some(*part))
                        .ok_or_else(|| HostError::Message(format!("missing field '{part}'")))?;
                    indices.push(FieldNav::Struct(idx));
                    current = data.field_at(idx).unwrap();
                }
                engine_reflect::bevy_reflect::ReflectRef::TupleStruct(data) => {
                    let idx: usize = part
                        .parse()
                        .map_err(|_| HostError::Message(format!("invalid tuple index '{part}'")))?;
                    indices.push(FieldNav::Tuple(idx));
                    current = data
                        .field(idx)
                        .ok_or_else(|| HostError::Message(format!("missing tuple field {idx}")))?;
                }
                _ => {
                    return Err(HostError::Message(format!(
                        "cannot navigate through '{part}'"
                    )));
                }
            }
        }
    }

    // Re-walk mutably.
    let leaf_name = parts[parts.len() - 1];
    let mut target = root;
    for nav in indices {
        target = match (target.reflect_mut(), nav) {
            (ReflectMut::Struct(data), FieldNav::Struct(idx)) => data
                .field_at_mut(idx)
                .ok_or_else(|| HostError::Message("struct field vanished".to_owned()))?,
            (ReflectMut::TupleStruct(data), FieldNav::Tuple(idx)) => data
                .field_mut(idx)
                .ok_or_else(|| HostError::Message("tuple field vanished".to_owned()))?,
            _ => {
                return Err(HostError::Message("navigation mismatch".to_owned()));
            }
        };
    }

    match target.reflect_mut() {
        ReflectMut::Struct(data) => {
            let field = data
                .field_mut(leaf_name)
                .ok_or_else(|| HostError::Message(format!("missing field '{leaf_name}'")))?;
            apply_json_leaf(field, value)
        }
        ReflectMut::TupleStruct(data) => {
            let idx: usize = leaf_name
                .parse()
                .map_err(|_| HostError::Message(format!("invalid tuple index '{leaf_name}'")))?;
            let field = data
                .field_mut(idx)
                .ok_or_else(|| HostError::Message(format!("missing tuple field {idx}")))?;
            apply_json_leaf(field, value)
        }
        _ => apply_json_leaf(target, value),
    }
}

enum FieldNav {
    Struct(usize),
    Tuple(usize),
}

fn apply_json_leaf(
    target: &mut dyn engine_reflect::bevy_reflect::PartialReflect,
    value: &HostValue,
) -> HostResult<()> {
    if let Some(slot) = target.try_downcast_mut::<f32>() {
        *slot = value
            .as_f64()
            .ok_or_else(|| HostError::Message("expected number".into()))? as f32;
        return Ok(());
    }
    if let Some(slot) = target.try_downcast_mut::<f64>() {
        *slot = value
            .as_f64()
            .ok_or_else(|| HostError::Message("expected number".into()))?;
        return Ok(());
    }
    if let Some(slot) = target.try_downcast_mut::<i32>() {
        *slot = value
            .as_i64()
            .ok_or_else(|| HostError::Message("expected integer".into()))? as i32;
        return Ok(());
    }
    if let Some(slot) = target.try_downcast_mut::<i64>() {
        *slot = value
            .as_i64()
            .ok_or_else(|| HostError::Message("expected integer".into()))?;
        return Ok(());
    }
    if let Some(slot) = target.try_downcast_mut::<u32>() {
        *slot = value
            .as_u64()
            .ok_or_else(|| HostError::Message("expected u32".into()))? as u32;
        return Ok(());
    }
    if let Some(slot) = target.try_downcast_mut::<bool>() {
        *slot = value
            .as_bool()
            .ok_or_else(|| HostError::Message("expected bool".into()))?;
        return Ok(());
    }
    if let Some(slot) = target.try_downcast_mut::<String>() {
        *slot = value
            .as_str()
            .ok_or_else(|| HostError::Message("expected string".into()))?
            .to_owned();
        return Ok(());
    }

    // glam Vec3 / Quat as object maps.
    if let Some(obj) = value.as_object() {
        if let engine_reflect::bevy_reflect::ReflectMut::Struct(data) = target.reflect_mut() {
            for (key, val) in obj {
                if let Some(field) = data.field_mut(key) {
                    apply_json_leaf(field, val)?;
                }
            }
            return Ok(());
        }
    }

    Err(HostError::Message(format!(
        "unsupported reflect leaf for value {value}"
    )))
}

fn reflect_to_json(
    value: &dyn engine_reflect::bevy_reflect::PartialReflect,
) -> HostResult<HostValue> {
    use engine_reflect::bevy_reflect::ReflectRef;

    if let Some(v) = value.try_downcast_ref::<f32>() {
        return Ok(JsonValue::from(*v));
    }
    if let Some(v) = value.try_downcast_ref::<f64>() {
        return Ok(JsonValue::from(*v));
    }
    if let Some(v) = value.try_downcast_ref::<i32>() {
        return Ok(JsonValue::from(*v));
    }
    if let Some(v) = value.try_downcast_ref::<i64>() {
        return Ok(JsonValue::from(*v));
    }
    if let Some(v) = value.try_downcast_ref::<u32>() {
        return Ok(JsonValue::from(*v));
    }
    if let Some(v) = value.try_downcast_ref::<bool>() {
        return Ok(JsonValue::from(*v));
    }
    if let Some(v) = value.try_downcast_ref::<String>() {
        return Ok(JsonValue::String(v.clone()));
    }

    match value.reflect_ref() {
        ReflectRef::Struct(data) => {
            let mut map = serde_json::Map::new();
            for i in 0..data.field_len() {
                let Some(name) = data.name_at(i) else {
                    continue;
                };
                let Some(field) = data.field_at(i) else {
                    continue;
                };
                map.insert(name.to_owned(), reflect_to_json(field)?);
            }
            Ok(JsonValue::Object(map))
        }
        ReflectRef::TupleStruct(data) => {
            let mut arr = Vec::new();
            for i in 0..data.field_len() {
                if let Some(field) = data.field(i) {
                    arr.push(reflect_to_json(field)?);
                }
            }
            Ok(JsonValue::Array(arr))
        }
        ReflectRef::Enum(data) => Ok(JsonValue::String(data.variant_name().to_owned())),
        _ => Ok(JsonValue::Null),
    }
}
