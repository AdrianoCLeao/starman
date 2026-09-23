//! Editor clipboard for entity subtrees (copy / cut / paste).

use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;
use engine_assets::{InstanceLocalEntity, SceneInstance};
use engine_core::{Children, EntityId, EntityName, Parent, PersistentId};
use engine_reflect::ComponentRegistry;
use engine_render::{MeshRenderable3d, SpriteRenderable2d};
use engine_reflect::bevy_reflect::PartialReflect;

pub struct ClipboardEntity {
    pub name: Option<String>,
    pub components: Vec<(String, Box<dyn PartialReflect>)>,
    pub mesh_renderable: Option<MeshRenderable3d>,
    pub sprite_renderable: Option<SpriteRenderable2d>,
    pub scene_instance: Option<SceneInstance>,
    pub children: Vec<ClipboardEntity>,
}

#[derive(Default)]
pub struct EntityClipboard {
    pub entries: Vec<ClipboardEntity>,
    pub cut: bool,
}

impl EntityClipboard {
    pub fn clear(&mut self) {
        self.entries.clear();
        self.cut = false;
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

pub fn capture_clipboard_entity(
    world: &World,
    component_registry: &ComponentRegistry,
    entity: Entity,
) -> Option<ClipboardEntity> {
    if world.get_entity(entity).is_err() {
        return None;
    }

    let mut components = Vec::new();
    for descriptor in component_registry.all() {
        let short = descriptor.name.rsplit("::").next().unwrap_or(descriptor.name);
        if matches!(
            short,
            "EntityName"
                | "Parent"
                | "Children"
                | "GlobalTransform"
                | "RigidBodyHandle3D"
                | "ColliderHandle3D"
        ) {
            continue;
        }
        if !descriptor.has(entity, world) {
            continue;
        }
        if let Some(component) = descriptor.get_reflect(entity, world) {
            components.push((
                descriptor.name.to_owned(),
                component.as_partial_reflect().clone_value(),
            ));
        }
    }

    let children_ids = world
        .get::<Children>(entity)
        .map(|c| c.0.clone())
        .unwrap_or_default();
    let mut children = Vec::new();
    for child in children_ids {
        // Do not copy inherited children — they come from the instance.
        if world.get::<engine_assets::InheritedEntity>(child).is_some() {
            continue;
        }
        if let Some(captured) = capture_clipboard_entity(world, component_registry, child) {
            children.push(captured);
        }
    }

    Some(ClipboardEntity {
        name: world.get::<EntityName>(entity).map(|n| n.0.clone()),
        components,
        mesh_renderable: world.get::<MeshRenderable3d>(entity).copied(),
        sprite_renderable: world.get::<SpriteRenderable2d>(entity).copied(),
        scene_instance: world.get::<SceneInstance>(entity).cloned(),
        children,
    })
}

pub fn paste_clipboard_entity(
    world: &mut World,
    component_registry: &ComponentRegistry,
    entry: &ClipboardEntity,
    parent: Option<Entity>,
    as_instance_local: Option<Entity>,
) -> Option<Entity> {
    let entity = world.spawn_empty().id();
    if let Ok(mut entity_ref) = world.get_entity_mut(entity) {
        entity_ref.insert(PersistentId(EntityId::new_v4()));
        if let Some(name) = &entry.name {
            entity_ref.insert(EntityName::new(name.clone()));
        }
        if let Some(mesh) = entry.mesh_renderable {
            entity_ref.insert(mesh);
        }
        if let Some(sprite) = entry.sprite_renderable {
            entity_ref.insert(sprite);
        }
        if let Some(instance) = entry.scene_instance.clone() {
            entity_ref.insert(instance);
        }
        if let Some(instance_root) = as_instance_local {
            entity_ref.insert(InstanceLocalEntity { instance_root });
        }
    }

    for (component_name, value) in &entry.components {
        let Some(descriptor) = component_registry
            .all()
            .iter()
            .find(|d| d.name == component_name.as_str())
        else {
            continue;
        };
        if !descriptor.insert_default(entity, world) {
            continue;
        }
        if let Some(target) = descriptor.get_reflect_mut(entity, world) {
            let _ = target
                .as_partial_reflect_mut()
                .try_apply(value.as_ref());
        }
    }

    let mut child_entities = Vec::new();
    for child in &entry.children {
        if let Some(child_entity) =
            paste_clipboard_entity(world, component_registry, child, Some(entity), as_instance_local)
        {
            child_entities.push(child_entity);
        }
    }
    if !child_entities.is_empty() {
        if let Ok(mut entity_ref) = world.get_entity_mut(entity) {
            entity_ref.insert(Children(child_entities.clone()));
        }
        for child in &child_entities {
            if let Ok(mut child_ref) = world.get_entity_mut(*child) {
                child_ref.insert(Parent(entity));
            }
        }
    }

    if let Some(parent) = parent {
        if let Ok(mut entity_ref) = world.get_entity_mut(entity) {
            entity_ref.insert(Parent(parent));
        }
        if let Some(mut children) = world.get_mut::<Children>(parent) {
            children.0.push(entity);
        } else if let Ok(mut parent_ref) = world.get_entity_mut(parent) {
            parent_ref.insert(Children(vec![entity]));
        }
    }

    Some(entity)
}
