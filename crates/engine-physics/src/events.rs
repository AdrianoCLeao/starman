//! Typed collision and trigger events, published after every step.

use bevy_ecs::prelude::*;
use rapier3d::prelude::{CollisionEvent, CollisionEventFlags};

use crate::mapping::{ColliderEntityMap3D, PhysicsEntityHandles3D};
use crate::world3d::PhysicsWorld3D;

/// Two solid colliders started touching. `body_*` is the entity owning
/// the collider's rigid body (the collider entity itself for a body's own
/// collider; `None` for standalone static colliders).
#[derive(Event, Clone, Copy, Debug, PartialEq, Eq)]
pub struct CollisionStarted {
    pub a: Entity,
    pub b: Entity,
    pub body_a: Option<Entity>,
    pub body_b: Option<Entity>,
}

#[derive(Event, Clone, Copy, Debug, PartialEq, Eq)]
pub struct CollisionStopped {
    pub a: Entity,
    pub b: Entity,
    pub body_a: Option<Entity>,
    pub body_b: Option<Entity>,
    /// One of the colliders was removed (despawned, disabled).
    pub removed: bool,
}

/// Something entered a [`crate::Sensor`] volume.
#[derive(Event, Clone, Copy, Debug, PartialEq, Eq)]
pub struct TriggerEntered {
    pub trigger: Entity,
    /// The collider that entered.
    pub other: Entity,
    /// Its rigid body's entity, when it has one (usually what gameplay
    /// wants: the player, not the player's foot collider).
    pub other_body: Option<Entity>,
}

#[derive(Event, Clone, Copy, Debug, PartialEq, Eq)]
pub struct TriggerExited {
    pub trigger: Entity,
    pub other: Entity,
    pub other_body: Option<Entity>,
    pub removed: bool,
}

/// Translates the step's Rapier events into typed ECS events.
#[allow(clippy::too_many_arguments)]
pub fn publish_collision_events(
    physics: Option<Res<PhysicsWorld3D>>,
    mut colliders: Option<ResMut<ColliderEntityMap3D>>,
    handles: Option<Res<PhysicsEntityHandles3D>>,
    sensors: Query<(), With<crate::Sensor>>,
    mut started: EventWriter<CollisionStarted>,
    mut stopped: EventWriter<CollisionStopped>,
    mut entered: EventWriter<TriggerEntered>,
    mut exited: EventWriter<TriggerExited>,
) {
    let (Some(physics), Some(colliders), Some(handles)) =
        (physics, colliders.as_deref_mut(), handles)
    else {
        return;
    };
    for event in physics.events.drain() {
        let (h1, h2, flags) = match event {
            CollisionEvent::Started(a, b, flags) | CollisionEvent::Stopped(a, b, flags) => {
                (a, b, flags)
            }
        };
        let (Some(e1), Some(e2)) = (colliders.resolve(&h1), colliders.resolve(&h2)) else {
            continue;
        };
        let body_of = |handle, entity| {
            physics
                .collider_set
                .get(handle)
                .and_then(|collider| collider.parent())
                .and_then(|body| handles.body_entity(body))
                .or_else(|| handles.body(entity).map(|_| entity))
        };
        let (b1, b2) = (body_of(h1, e1), body_of(h2, e2));
        let removed = flags.contains(CollisionEventFlags::REMOVED);
        if flags.contains(CollisionEventFlags::SENSOR) {
            // Report once per sensor involved (two overlapping sensors
            // both get an event).
            let pairs = [(e1, e2, b2), (e2, e1, b1)];
            for (trigger, other, other_body) in pairs {
                let is_sensor = sensors.contains(trigger)
                    || physics
                        .collider_set
                        .get(if trigger == e1 { h1 } else { h2 })
                        .is_some_and(|c| c.is_sensor());
                if !is_sensor {
                    continue;
                }
                if event.started() {
                    entered.send(TriggerEntered {
                        trigger,
                        other,
                        other_body,
                    });
                } else {
                    exited.send(TriggerExited {
                        trigger,
                        other,
                        other_body,
                        removed,
                    });
                }
            }
        } else if event.started() {
            started.send(CollisionStarted {
                a: e1,
                b: e2,
                body_a: b1,
                body_b: b2,
            });
        } else {
            stopped.send(CollisionStopped {
                a: e1,
                b: e2,
                body_a: b1,
                body_b: b2,
                removed,
            });
        }
    }
    colliders.clear_recently_removed();
}
