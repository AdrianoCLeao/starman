use std::collections::{HashSet, VecDeque};

use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;
use engine_core::{Children, EditorEntityBundle, EntityName, Parent};
use engine_reflect::bevy_reflect::{PartialReflect, ReflectMut};
use engine_reflect::{ComponentDescriptor, ComponentRegistry};
use engine_render::{MeshRenderable3d, SpriteRenderable2d};

pub trait EditorCommand: Send + Sync {
    fn execute(&mut self, world: &mut World);
    fn undo(&mut self, world: &mut World);
    fn description(&self) -> &str;
    fn selection_hint(&self) -> Option<Entity> {
        None
    }
}

pub struct CommandHistory {
    undo_stack: VecDeque<Box<dyn EditorCommand>>,
    redo_stack: VecDeque<Box<dyn EditorCommand>>,
    max_size: usize,
}

impl CommandHistory {
    pub fn new(max_size: usize) -> Self {
        Self {
            undo_stack: VecDeque::new(),
            redo_stack: VecDeque::new(),
            max_size,
        }
    }

    pub fn execute(
        &mut self,
        mut cmd: Box<dyn EditorCommand>,
        world: &mut World,
    ) -> Option<Entity> {
        cmd.execute(world);
        let selection_hint = cmd.selection_hint();
        self.undo_stack.push_back(cmd);
        self.redo_stack.clear();

        while self.undo_stack.len() > self.max_size {
            let _ = self.undo_stack.pop_front();
        }

        selection_hint
    }

    pub fn undo(&mut self, world: &mut World) -> Option<Entity> {
        if let Some(mut cmd) = self.undo_stack.pop_back() {
            cmd.undo(world);
            let selection_hint = cmd.selection_hint();
            self.redo_stack.push_back(cmd);
            return selection_hint;
        }

        None
    }

    pub fn redo(&mut self, world: &mut World) -> Option<Entity> {
        if let Some(mut cmd) = self.redo_stack.pop_back() {
            cmd.execute(world);
            let selection_hint = cmd.selection_hint();
            self.undo_stack.push_back(cmd);
            return selection_hint;
        }

        None
    }

    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    pub fn clear(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
    }
}

pub struct SetFieldCommand {
    pub entity: Entity,
    pub component_name: String,
    pub field_path: String,
    pub old_value: Box<dyn PartialReflect>,
    pub new_value: Box<dyn PartialReflect>,
    pub desc: String,
}

impl EditorCommand for SetFieldCommand {
    fn execute(&mut self, world: &mut World) {
        apply_field_value(
            world,
            self.entity,
            &self.component_name,
            &self.field_path,
            self.new_value.as_ref(),
        );
    }

    fn undo(&mut self, world: &mut World) {
        apply_field_value(
            world,
            self.entity,
            &self.component_name,
            &self.field_path,
            self.old_value.as_ref(),
        );
    }

    fn description(&self) -> &str {
        &self.desc
    }
}

pub struct SetComponentCommand {
    pub entity: Entity,
    pub component_name: String,
    pub old_value: Box<dyn PartialReflect>,
    pub new_value: Box<dyn PartialReflect>,
    pub desc: String,
}

impl EditorCommand for SetComponentCommand {
    fn execute(&mut self, world: &mut World) {
        apply_component_value(
            world,
            self.entity,
            &self.component_name,
            self.new_value.as_ref(),
        );
    }

    fn undo(&mut self, world: &mut World) {
        apply_component_value(
            world,
            self.entity,
            &self.component_name,
            self.old_value.as_ref(),
        );
    }

    fn description(&self) -> &str {
        &self.desc
    }
}

pub struct AddComponentCommand {
    pub entity: Entity,
    pub component_name: String,
    desc: String,
}

impl AddComponentCommand {
    pub fn new(entity: Entity, component_name: impl Into<String>) -> Self {
        let component_name = component_name.into();
        Self {
            entity,
            desc: format!("Add {}", short_type_name(&component_name)),
            component_name,
        }
    }
}

impl EditorCommand for AddComponentCommand {
    fn execute(&mut self, world: &mut World) {
        with_component_registry(world, |world, component_registry| {
            if let Some(descriptor) = find_descriptor(component_registry, &self.component_name) {
                let _ = descriptor.insert_default(self.entity, world);
            }
        });
    }

    fn undo(&mut self, world: &mut World) {
        with_component_registry(world, |world, component_registry| {
            if let Some(descriptor) = find_descriptor(component_registry, &self.component_name) {
                let _ = descriptor.remove(self.entity, world);
            }
        });
    }

    fn description(&self) -> &str {
        &self.desc
    }
}

pub struct RemoveComponentCommand {
    pub entity: Entity,
    pub component_name: String,
    cached_value: Option<Box<dyn PartialReflect>>,
    desc: String,
}

impl RemoveComponentCommand {
    pub fn new(entity: Entity, component_name: impl Into<String>) -> Self {
        let component_name = component_name.into();
        Self {
            entity,
            desc: format!("Remove {}", short_type_name(&component_name)),
            component_name,
            cached_value: None,
        }
    }
}

impl EditorCommand for RemoveComponentCommand {
    fn execute(&mut self, world: &mut World) {
        with_component_registry(world, |world, component_registry| {
            if let Some(descriptor) = find_descriptor(component_registry, &self.component_name) {
                if self.cached_value.is_none() {
                    if let Some(value) = descriptor.get_reflect(self.entity, world) {
                        self.cached_value = Some(value.as_partial_reflect().clone_value());
                    }
                }

                let _ = descriptor.remove(self.entity, world);
            }
        });
    }

    fn undo(&mut self, world: &mut World) {
        with_component_registry(world, |world, component_registry| {
            if let Some(descriptor) = find_descriptor(component_registry, &self.component_name) {
                if !descriptor.insert_default(self.entity, world) {
                    return;
                }

                if let Some(value) = self.cached_value.as_deref() {
                    if let Some(target) = descriptor.get_reflect_mut(self.entity, world) {
                        let _ = target
                            .as_partial_reflect_mut()
                            .try_apply(value.as_partial_reflect());
                    }
                }
            }
        });
    }

    fn description(&self) -> &str {
        &self.desc
    }
}

struct ComponentSnapshot {
    component_name: String,
    value: Box<dyn PartialReflect>,
}

struct EntitySnapshot {
    name: Option<String>,
    components: Vec<ComponentSnapshot>,
    mesh_renderable: Option<MeshRenderable3d>,
    sprite_renderable: Option<SpriteRenderable2d>,
    children: Vec<EntitySnapshot>,
}

pub struct DeleteEntityCommand {
    target_entity: Entity,
    parent: Option<Entity>,
    parent_index: Option<usize>,
    snapshot: Option<EntitySnapshot>,
    desc: String,
}

impl DeleteEntityCommand {
    pub fn new(entity: Entity) -> Self {
        Self {
            target_entity: entity,
            parent: None,
            parent_index: None,
            snapshot: None,
            desc: "Delete entity".to_owned(),
        }
    }
}

impl EditorCommand for DeleteEntityCommand {
    fn execute(&mut self, world: &mut World) {
        with_component_registry(world, |world, component_registry| {
            if world.get_entity(self.target_entity).is_err() {
                return;
            }

            if self.snapshot.is_none() {
                self.parent = world
                    .get::<Parent>(self.target_entity)
                    .map(|parent| parent.0);
                self.parent_index = self
                    .parent
                    .and_then(|parent| child_index_of(world, parent, self.target_entity));
                self.snapshot =
                    capture_entity_snapshot(world, component_registry, self.target_entity);
            }

            if let Some(parent) = self.parent {
                let _ = remove_child_from_parent(world, parent, self.target_entity);
            }

            despawn_entity_subtree(world, self.target_entity);
        });
    }

    fn undo(&mut self, world: &mut World) {
        with_component_registry(world, |world, component_registry| {
            let Some(snapshot) = self.snapshot.as_ref() else {
                return;
            };

            let Some(restored_root) = restore_entity_snapshot(world, component_registry, snapshot)
            else {
                return;
            };

            if let Some(parent) = self.parent {
                if world.get_entity(parent).is_ok() {
                    attach_child_to_parent(world, parent, restored_root, self.parent_index);
                }
            }

            self.target_entity = restored_root;
        });
    }

    fn description(&self) -> &str {
        &self.desc
    }

    fn selection_hint(&self) -> Option<Entity> {
        self.parent
    }
}

pub struct DuplicateEntityCommand {
    source_entity: Entity,
    parent: Option<Entity>,
    insert_index: Option<usize>,
    snapshot: Option<EntitySnapshot>,
    duplicate_entity: Option<Entity>,
    desc: String,
}

impl DuplicateEntityCommand {
    pub fn new(entity: Entity) -> Self {
        Self {
            source_entity: entity,
            parent: None,
            insert_index: None,
            snapshot: None,
            duplicate_entity: None,
            desc: "Duplicate entity".to_owned(),
        }
    }
}

impl EditorCommand for DuplicateEntityCommand {
    fn execute(&mut self, world: &mut World) {
        with_component_registry(world, |world, component_registry| {
            if self.snapshot.is_none() {
                if world.get_entity(self.source_entity).is_err() {
                    return;
                }

                self.parent = world
                    .get::<Parent>(self.source_entity)
                    .map(|parent| parent.0);
                self.insert_index = self.parent.and_then(|parent| {
                    child_index_of(world, parent, self.source_entity).map(|index| index + 1)
                });
                self.snapshot =
                    capture_entity_snapshot(world, component_registry, self.source_entity);
            }

            let Some(snapshot) = self.snapshot.as_ref() else {
                return;
            };

            let Some(duplicated_root) =
                restore_entity_snapshot(world, component_registry, snapshot)
            else {
                return;
            };

            if let Some(parent) = self.parent {
                if world.get_entity(parent).is_ok() {
                    attach_child_to_parent(world, parent, duplicated_root, self.insert_index);
                }
            }

            self.duplicate_entity = Some(duplicated_root);
        });
    }

    fn undo(&mut self, world: &mut World) {
        let Some(duplicate_entity) = self.duplicate_entity.take() else {
            return;
        };

        with_component_registry(world, |world, _component_registry| {
            if let Some(parent) = self.parent {
                let _ = remove_child_from_parent(world, parent, duplicate_entity);
            }
            despawn_entity_subtree(world, duplicate_entity);
        });
    }

    fn description(&self) -> &str {
        &self.desc
    }

    fn selection_hint(&self) -> Option<Entity> {
        self.duplicate_entity
    }
}

pub struct RenameEntityCommand {
    entity: Entity,
    old_name: String,
    new_name: String,
    desc: String,
}

impl RenameEntityCommand {
    pub fn new(entity: Entity, old_name: String, new_name: String) -> Self {
        Self {
            entity,
            old_name,
            new_name,
            desc: "Rename entity".to_owned(),
        }
    }
}

impl EditorCommand for RenameEntityCommand {
    fn execute(&mut self, world: &mut World) {
        set_entity_name(world, self.entity, &self.new_name);
    }

    fn undo(&mut self, world: &mut World) {
        set_entity_name(world, self.entity, &self.old_name);
    }

    fn description(&self) -> &str {
        &self.desc
    }

    fn selection_hint(&self) -> Option<Entity> {
        Some(self.entity)
    }
}

pub struct SpawnEntityCommand {
    parent: Option<Entity>,
    insert_index: Option<usize>,
    spawned_entity: Option<Entity>,
    desc: String,
}

impl SpawnEntityCommand {
    pub fn new_root() -> Self {
        Self {
            parent: None,
            insert_index: None,
            spawned_entity: None,
            desc: "Create entity".to_owned(),
        }
    }

    pub fn new_child(parent: Entity) -> Self {
        Self {
            parent: Some(parent),
            insert_index: None,
            spawned_entity: None,
            desc: "Create child entity".to_owned(),
        }
    }
}

impl EditorCommand for SpawnEntityCommand {
    fn execute(&mut self, world: &mut World) {
        if let Some(parent) = self.parent {
            if world.get_entity(parent).is_err() {
                log::warn!(
                    target: "engine::editor",
                    "Skipping spawn child command: parent entity {:?} no longer exists",
                    parent
                );
                return;
            }

            if self.insert_index.is_none() {
                self.insert_index = world
                    .get::<Children>(parent)
                    .map(|children| children.0.len());
            }
        }

        let entity = world.spawn(EditorEntityBundle::default()).id();

        if let Some(parent) = self.parent {
            attach_child_to_parent(world, parent, entity, self.insert_index);
        }

        self.spawned_entity = Some(entity);
    }

    fn undo(&mut self, world: &mut World) {
        let Some(entity) = self.spawned_entity.take() else {
            return;
        };

        if let Some(parent) = self.parent {
            let _ = remove_child_from_parent(world, parent, entity);
        }

        if let Ok(mut entity_ref) = world.get_entity_mut(entity) {
            let _ = entity_ref.remove::<Parent>();
        }

        despawn_entity_subtree(world, entity);
    }

    fn description(&self) -> &str {
        &self.desc
    }

    fn selection_hint(&self) -> Option<Entity> {
        self.spawned_entity.or(self.parent)
    }
}

pub struct ReparentEntityCommand {
    entity: Entity,
    new_parent: Option<Entity>,
    old_parent: Option<Entity>,
    old_parent_index: Option<usize>,
    captured_original: bool,
    desc: String,
}

impl ReparentEntityCommand {
    pub fn new(entity: Entity, new_parent: Option<Entity>) -> Self {
        Self {
            entity,
            new_parent,
            old_parent: None,
            old_parent_index: None,
            captured_original: false,
            desc: "Reparent entity".to_owned(),
        }
    }
}

impl EditorCommand for ReparentEntityCommand {
    fn execute(&mut self, world: &mut World) {
        if world.get_entity(self.entity).is_err() {
            log::warn!(
                target: "engine::editor",
                "Skipping reparent command: entity {:?} no longer exists",
                self.entity
            );
            return;
        }

        if let Some(new_parent) = self.new_parent {
            if new_parent == self.entity {
                log::warn!(
                    target: "engine::editor",
                    "Rejected reparent command: entity {:?} cannot be parent of itself",
                    self.entity
                );
                return;
            }

            if world.get_entity(new_parent).is_err() {
                log::warn!(
                    target: "engine::editor",
                    "Skipping reparent command: target parent {:?} no longer exists",
                    new_parent
                );
                return;
            }

            if is_descendant_of(world, new_parent, self.entity) {
                log::warn!(
                    target: "engine::editor",
                    "Rejected reparent command: entity {:?} cannot become child of descendant {:?}",
                    self.entity,
                    new_parent
                );
                return;
            }
        }

        if !self.captured_original {
            self.old_parent = world.get::<Parent>(self.entity).map(|parent| parent.0);
            self.old_parent_index = self
                .old_parent
                .and_then(|parent| child_index_of(world, parent, self.entity));
            self.captured_original = true;
        }

        let current_parent = world.get::<Parent>(self.entity).map(|parent| parent.0);
        if current_parent == self.new_parent {
            return;
        }

        if let Some(parent) = current_parent {
            let _ = remove_child_from_parent(world, parent, self.entity);
        }

        if let Some(parent) = self.new_parent {
            attach_child_to_parent(world, parent, self.entity, None);
        } else if let Ok(mut entity_ref) = world.get_entity_mut(self.entity) {
            let _ = entity_ref.remove::<Parent>();
        }
    }

    fn undo(&mut self, world: &mut World) {
        if world.get_entity(self.entity).is_err() {
            log::warn!(
                target: "engine::editor",
                "Skipping undo reparent command: entity {:?} no longer exists",
                self.entity
            );
            return;
        }

        let current_parent = world.get::<Parent>(self.entity).map(|parent| parent.0);
        if let Some(parent) = current_parent {
            let _ = remove_child_from_parent(world, parent, self.entity);
        }

        if let Some(parent) = self.old_parent {
            if world.get_entity(parent).is_ok() {
                attach_child_to_parent(world, parent, self.entity, self.old_parent_index);
            } else if let Ok(mut entity_ref) = world.get_entity_mut(self.entity) {
                let _ = entity_ref.remove::<Parent>();
            }
        } else if let Ok(mut entity_ref) = world.get_entity_mut(self.entity) {
            let _ = entity_ref.remove::<Parent>();
        }
    }

    fn description(&self) -> &str {
        &self.desc
    }

    fn selection_hint(&self) -> Option<Entity> {
        Some(self.entity)
    }
}

fn apply_component_value(
    world: &mut World,
    entity: Entity,
    component_name: &str,
    value: &dyn PartialReflect,
) {
    with_component_registry(world, |world, component_registry| {
        let Some(descriptor) = find_descriptor(component_registry, component_name) else {
            return;
        };

        let Some(target) = descriptor.get_reflect_mut(entity, world) else {
            return;
        };

        let _ = target.as_partial_reflect_mut().try_apply(value);
    });
}

fn apply_field_value(
    world: &mut World,
    entity: Entity,
    component_name: &str,
    field_path: &str,
    value: &dyn PartialReflect,
) {
    with_component_registry(world, |world, component_registry| {
        let Some(descriptor) = find_descriptor(component_registry, component_name) else {
            return;
        };

        let Some(component) = descriptor.get_reflect_mut(entity, world) else {
            return;
        };

        if field_path.is_empty() {
            let _ = component.as_partial_reflect_mut().try_apply(value);
            return;
        }

        let path: Vec<&str> = field_path.split('.').collect();
        if let Some(field) = traverse_field_path_mut(component.as_partial_reflect_mut(), &path) {
            let _ = field.try_apply(value);
        }
    });
}

fn traverse_field_path_mut<'a>(
    current: &'a mut dyn PartialReflect,
    path: &[&str],
) -> Option<&'a mut dyn PartialReflect> {
    if path.is_empty() {
        return Some(current);
    }

    let (head, tail) = path.split_first()?;

    let child = match current.reflect_mut() {
        ReflectMut::Struct(data) => data.field_mut(head),
        ReflectMut::TupleStruct(data) => head
            .parse::<usize>()
            .ok()
            .and_then(|index| data.field_mut(index)),
        ReflectMut::Tuple(data) => head
            .parse::<usize>()
            .ok()
            .and_then(|index| data.field_mut(index)),
        ReflectMut::Enum(data) => {
            if let Ok(index) = head.parse::<usize>() {
                data.field_at_mut(index)
            } else {
                data.field_mut(head)
            }
        }
        ReflectMut::List(data) => head
            .parse::<usize>()
            .ok()
            .and_then(|index| data.get_mut(index)),
        ReflectMut::Array(data) => head
            .parse::<usize>()
            .ok()
            .and_then(|index| data.get_mut(index)),
        _ => None,
    }?;

    traverse_field_path_mut(child, tail)
}

fn find_descriptor<'a>(
    component_registry: &'a ComponentRegistry,
    component_name: &str,
) -> Option<&'a ComponentDescriptor> {
    component_registry.all().iter().find(|descriptor| {
        descriptor.name == component_name || short_type_name(descriptor.name) == component_name
    })
}

fn capture_entity_snapshot(
    world: &World,
    component_registry: &ComponentRegistry,
    entity: Entity,
) -> Option<EntitySnapshot> {
    if world.get_entity(entity).is_err() {
        return None;
    }

    let mut components = Vec::new();
    for descriptor in component_registry.all() {
        if should_skip_snapshot_component(descriptor) {
            continue;
        }

        if !descriptor.has(entity, world) {
            continue;
        }

        if let Some(component) = descriptor.get_reflect(entity, world) {
            components.push(ComponentSnapshot {
                component_name: descriptor.name.to_owned(),
                value: component.as_partial_reflect().clone_value(),
            });
        }
    }

    let children_ids = world
        .get::<Children>(entity)
        .map(|children| children.0.clone())
        .unwrap_or_default();
    let mut children = Vec::new();
    for child in children_ids {
        if let Some(snapshot) = capture_entity_snapshot(world, component_registry, child) {
            children.push(snapshot);
        }
    }

    Some(EntitySnapshot {
        name: world.get::<EntityName>(entity).map(|value| value.0.clone()),
        components,
        mesh_renderable: world.get::<MeshRenderable3d>(entity).copied(),
        sprite_renderable: world.get::<SpriteRenderable2d>(entity).copied(),
        children,
    })
}

fn restore_entity_snapshot(
    world: &mut World,
    component_registry: &ComponentRegistry,
    snapshot: &EntitySnapshot,
) -> Option<Entity> {
    let entity = world.spawn_empty().id();

    if let Ok(mut entity_ref) = world.get_entity_mut(entity) {
        if let Some(name) = &snapshot.name {
            entity_ref.insert(EntityName::new(name.clone()));
        }
        if let Some(mesh_renderable) = snapshot.mesh_renderable {
            entity_ref.insert(mesh_renderable);
        }
        if let Some(sprite_renderable) = snapshot.sprite_renderable {
            entity_ref.insert(sprite_renderable);
        }
    }

    for component in &snapshot.components {
        let Some(descriptor) = find_descriptor(component_registry, &component.component_name)
        else {
            continue;
        };

        if !descriptor.insert_default(entity, world) {
            continue;
        }

        if let Some(target) = descriptor.get_reflect_mut(entity, world) {
            let _ = target
                .as_partial_reflect_mut()
                .try_apply(component.value.as_ref());
        }
    }

    let mut restored_children = Vec::new();
    for child_snapshot in &snapshot.children {
        let Some(child_entity) = restore_entity_snapshot(world, component_registry, child_snapshot)
        else {
            continue;
        };

        if let Ok(mut child_ref) = world.get_entity_mut(child_entity) {
            child_ref.insert(Parent(entity));
        }

        restored_children.push(child_entity);
    }

    if !restored_children.is_empty() {
        if let Ok(mut entity_ref) = world.get_entity_mut(entity) {
            entity_ref.insert(Children(restored_children));
        }
    }

    Some(entity)
}

fn should_skip_snapshot_component(descriptor: &ComponentDescriptor) -> bool {
    matches!(
        short_type_name(descriptor.name),
        "EntityName"
            | "Parent"
            | "Children"
            | "GlobalTransform"
            | "RigidBodyHandle3D"
            | "ColliderHandle3D"
    )
}

fn child_index_of(world: &World, parent: Entity, child: Entity) -> Option<usize> {
    world
        .get::<Children>(parent)
        .and_then(|children| children.0.iter().position(|current| *current == child))
}

fn remove_child_from_parent(world: &mut World, parent: Entity, child: Entity) -> Option<usize> {
    let Ok(mut parent_ref) = world.get_entity_mut(parent) else {
        return None;
    };

    let mut children = parent_ref.get_mut::<Children>()?;

    let index = children.0.iter().position(|current| *current == child)?;
    children.0.remove(index);
    Some(index)
}

fn attach_child_to_parent(world: &mut World, parent: Entity, child: Entity, index: Option<usize>) {
    if world.get_entity(parent).is_err() || world.get_entity(child).is_err() {
        return;
    }

    if let Ok(mut child_ref) = world.get_entity_mut(child) {
        child_ref.insert(Parent(parent));
    }

    let Ok(mut parent_ref) = world.get_entity_mut(parent) else {
        return;
    };

    if let Some(mut children) = parent_ref.get_mut::<Children>() {
        if children.0.contains(&child) {
            return;
        }

        let insert_at = index.unwrap_or(children.0.len()).min(children.0.len());
        children.0.insert(insert_at, child);
    } else {
        parent_ref.insert(Children(vec![child]));
    }
}

fn set_entity_name(world: &mut World, entity: Entity, name: &str) {
    if let Ok(mut entity_ref) = world.get_entity_mut(entity) {
        if let Some(mut current_name) = entity_ref.get_mut::<EntityName>() {
            current_name.0 = name.to_owned();
        } else {
            entity_ref.insert(EntityName::new(name.to_owned()));
        }
    }
}

fn collect_subtree_entities(world: &World, entity: Entity, out: &mut Vec<Entity>) {
    let children = world
        .get::<Children>(entity)
        .map(|children| children.0.clone())
        .unwrap_or_default();

    for child in children {
        collect_subtree_entities(world, child, out);
    }

    out.push(entity);
}

fn despawn_entity_subtree(world: &mut World, root: Entity) {
    if world.get_entity(root).is_err() {
        return;
    }

    let mut entities = Vec::new();
    collect_subtree_entities(world, root, &mut entities);

    for entity in entities {
        let _ = world.despawn(entity);
    }
}

fn is_descendant_of(world: &World, candidate: Entity, ancestor: Entity) -> bool {
    let mut visited = HashSet::new();
    let mut current = Some(candidate);

    while let Some(entity) = current {
        if !visited.insert(entity) {
            break;
        }

        if entity == ancestor {
            return true;
        }

        current = world.get::<Parent>(entity).map(|parent| parent.0);
    }

    false
}

fn with_component_registry<R>(
    world: &mut World,
    f: impl FnOnce(&mut World, &ComponentRegistry) -> R,
) -> R {
    let component_registry = world
        .remove_resource::<ComponentRegistry>()
        .unwrap_or_default();

    let result = f(world, &component_registry);

    world.insert_resource(component_registry);

    result
}

pub fn short_type_name(type_path: &str) -> &str {
    type_path.rsplit("::").next().unwrap_or(type_path)
}
