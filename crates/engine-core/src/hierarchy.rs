//! Parent/child hierarchy maintenance.
//!
//! `Parent` and `Children` are plain components; these helpers keep both
//! sides coherent (no dangling child lists, no duplicate entries, no
//! cycles) and are the only supported way to reparent or recursively
//! despawn at runtime.

use bevy_ecs::entity::Entity;
use bevy_ecs::system::{Commands, EntityCommands};
use bevy_ecs::world::{Command, World};

use crate::{Children, GlobalTransform, Parent, Transform};

/// Returns `true` when `ancestor` is `entity` or one of its ancestors.
pub fn is_ancestor_of(world: &World, ancestor: Entity, entity: Entity) -> bool {
    let mut current = Some(entity);
    let mut guard = 0usize;
    while let Some(node) = current {
        if node == ancestor {
            return true;
        }
        guard += 1;
        if guard > 65_536 {
            return true;
        }
        current = world.get::<Parent>(node).map(|parent| parent.0);
    }
    false
}

/// Reparents `child` under `parent` (or detaches it with `None`), updating
/// both `Parent` and `Children`. Refuses (returns `false`) to create a cycle
/// or to touch despawned entities.
///
/// When `keep_world_transform` is set, the child's local `Transform` is
/// recomputed so its world placement does not jump.
pub fn set_parent_with(
    world: &mut World,
    child: Entity,
    parent: Option<Entity>,
    keep_world_transform: bool,
) -> bool {
    if world.get_entity(child).is_err() {
        return false;
    }
    if let Some(parent) = parent {
        if world.get_entity(parent).is_err() || is_ancestor_of(world, child, parent) {
            return false;
        }
    }

    let child_world = keep_world_transform
        .then(|| world.get::<GlobalTransform>(child).map(|g| g.0))
        .flatten();

    if let Some(old_parent) = world.get::<Parent>(child).map(|p| p.0) {
        if let Some(mut children) = world.get_mut::<Children>(old_parent) {
            children.0.retain(|entity| *entity != child);
        }
    }

    match parent {
        Some(parent) => {
            world.entity_mut(child).insert(Parent(parent));
            let mut parent_mut = world.entity_mut(parent);
            if let Some(mut children) = parent_mut.get_mut::<Children>() {
                if !children.0.contains(&child) {
                    children.0.push(child);
                }
            } else {
                parent_mut.insert(Children(vec![child]));
            }
        }
        None => {
            world.entity_mut(child).remove::<Parent>();
        }
    }

    if let Some(child_world) = child_world {
        let parent_world = parent
            .and_then(|parent| world.get::<GlobalTransform>(parent).map(|g| g.0))
            .unwrap_or(engine_math::glam::Affine3A::IDENTITY);
        let local = parent_world.inverse() * child_world;
        let (scale, rotation, translation) = local.to_scale_rotation_translation();
        if let Some(mut transform) = world.get_mut::<Transform>(child) {
            *transform = Transform {
                translation,
                rotation,
                scale,
            };
        }
    }
    true
}

/// Reparents keeping the local transform (the common scripted case).
pub fn set_parent(world: &mut World, child: Entity, parent: Option<Entity>) -> bool {
    set_parent_with(world, child, parent, false)
}

/// Collects `root` and all of its descendants (depth-first, root first).
pub fn collect_descendants(world: &World, root: Entity) -> Vec<Entity> {
    let mut out = Vec::new();
    let mut stack = vec![root];
    while let Some(entity) = stack.pop() {
        if out.contains(&entity) {
            continue;
        }
        out.push(entity);
        if let Some(children) = world.get::<Children>(entity) {
            stack.extend(children.0.iter().rev().copied());
        }
    }
    out
}

/// Despawns `root` and all descendants, detaching it from its parent.
pub fn despawn_recursive(world: &mut World, root: Entity) {
    if world.get_entity(root).is_err() {
        return;
    }
    if let Some(parent) = world.get::<Parent>(root).map(|p| p.0) {
        if let Some(mut children) = world.get_mut::<Children>(parent) {
            children.0.retain(|entity| *entity != root);
        }
    }
    for entity in collect_descendants(world, root).into_iter().rev() {
        let _ = world.despawn(entity);
    }
}

struct SetParentCommand {
    child: Entity,
    parent: Option<Entity>,
    keep_world_transform: bool,
}

impl Command for SetParentCommand {
    fn apply(self, world: &mut World) {
        if !set_parent_with(world, self.child, self.parent, self.keep_world_transform) {
            log::warn!(
                target: "engine::ecs",
                "set_parent({:?} -> {:?}) rejected (missing entity or cycle)",
                self.child,
                self.parent
            );
        }
    }
}

struct DespawnRecursiveCommand(Entity);

impl Command for DespawnRecursiveCommand {
    fn apply(self, world: &mut World) {
        despawn_recursive(world, self.0);
    }
}

/// Deferred hierarchy operations for systems.
pub trait HierarchyCommandsExt {
    fn set_parent(&mut self, child: Entity, parent: Option<Entity>);
    fn set_parent_keep_world(&mut self, child: Entity, parent: Option<Entity>);
    fn despawn_recursive(&mut self, root: Entity);
}

impl HierarchyCommandsExt for Commands<'_, '_> {
    fn set_parent(&mut self, child: Entity, parent: Option<Entity>) {
        self.queue(SetParentCommand {
            child,
            parent,
            keep_world_transform: false,
        });
    }

    fn set_parent_keep_world(&mut self, child: Entity, parent: Option<Entity>) {
        self.queue(SetParentCommand {
            child,
            parent,
            keep_world_transform: true,
        });
    }

    fn despawn_recursive(&mut self, root: Entity) {
        self.queue(DespawnRecursiveCommand(root));
    }
}

/// Convenience on entity commands.
pub trait EntityHierarchyExt {
    fn set_parent(&mut self, parent: Option<Entity>) -> &mut Self;
}

impl EntityHierarchyExt for EntityCommands<'_> {
    fn set_parent(&mut self, parent: Option<Entity>) -> &mut Self {
        let child = self.id();
        self.commands().queue(SetParentCommand {
            child,
            parent,
            keep_world_transform: false,
        });
        self
    }
}

#[cfg(test)]
#[path = "hierarchy_tests.rs"]
mod tests;
