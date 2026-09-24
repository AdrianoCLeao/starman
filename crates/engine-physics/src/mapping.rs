use bevy_ecs::prelude::{Entity, Resource};
use rapier3d::prelude::{ColliderHandle, RigidBodyHandle};
use std::collections::HashMap;

/// Collider → entity lookup (queries, events, picking).
#[derive(Resource, Default)]
pub struct ColliderEntityMap3D {
    by_collider: HashMap<ColliderHandle, Entity>,
    /// Colliders removed since the last event flush: removal produces
    /// "stopped" events one step later, which still need their entity.
    recently_removed: HashMap<ColliderHandle, Entity>,
}

impl ColliderEntityMap3D {
    pub fn insert(&mut self, collider: ColliderHandle, entity: Entity) {
        self.by_collider.insert(collider, entity);
    }

    pub fn get(&self, collider: &ColliderHandle) -> Option<Entity> {
        self.by_collider.get(collider).copied()
    }

    /// Like [`Self::get`], also resolving colliders removed this step.
    pub fn resolve(&self, collider: &ColliderHandle) -> Option<Entity> {
        self.get(collider)
            .or_else(|| self.recently_removed.get(collider).copied())
    }

    pub fn remove(&mut self, collider: ColliderHandle) {
        if let Some(entity) = self.by_collider.remove(&collider) {
            self.recently_removed.insert(collider, entity);
        }
    }

    pub(crate) fn clear_recently_removed(&mut self) {
        self.recently_removed.clear();
    }

    pub fn as_map(&self) -> &HashMap<ColliderHandle, Entity> {
        &self.by_collider
    }

    pub fn len(&self) -> usize {
        self.by_collider.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_collider.is_empty()
    }
}

/// Entity ↔ Rapier handle bookkeeping for bodies and colliders.
#[derive(Resource, Default)]
pub struct PhysicsEntityHandles3D {
    bodies: HashMap<Entity, RigidBodyHandle>,
    body_entities: HashMap<RigidBodyHandle, Entity>,
    colliders: HashMap<Entity, ColliderHandle>,
}

impl PhysicsEntityHandles3D {
    pub fn insert_body(&mut self, entity: Entity, body: RigidBodyHandle) {
        self.bodies.insert(entity, body);
        self.body_entities.insert(body, entity);
    }

    pub fn insert_collider(&mut self, entity: Entity, collider: ColliderHandle) {
        self.colliders.insert(entity, collider);
    }

    pub fn remove_body(&mut self, entity: Entity) -> Option<RigidBodyHandle> {
        let body = self.bodies.remove(&entity)?;
        self.body_entities.remove(&body);
        Some(body)
    }

    pub fn remove_collider(&mut self, entity: Entity) -> Option<ColliderHandle> {
        self.colliders.remove(&entity)
    }

    pub fn body(&self, entity: Entity) -> Option<RigidBodyHandle> {
        self.bodies.get(&entity).copied()
    }

    pub fn collider(&self, entity: Entity) -> Option<ColliderHandle> {
        self.colliders.get(&entity).copied()
    }

    pub fn body_entity(&self, body: RigidBodyHandle) -> Option<Entity> {
        self.body_entities.get(&body).copied()
    }

    pub fn body_entities(&self) -> impl Iterator<Item = Entity> + '_ {
        self.bodies.keys().copied()
    }

    pub fn collider_entities(&self) -> impl Iterator<Item = Entity> + '_ {
        self.colliders.keys().copied()
    }

    /// Every entity with a body or collider.
    pub fn entities(&self) -> impl Iterator<Item = Entity> + '_ {
        self.bodies
            .keys()
            .chain(
                self.colliders
                    .keys()
                    .filter(|e| !self.bodies.contains_key(e)),
            )
            .copied()
    }
}
