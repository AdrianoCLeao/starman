//! 3D physics on Rapier (ADR 0012 runtime plugin): bodies, compound
//! colliders, sensors and collision/trigger events, collision layers,
//! joints, a kinematic character controller, spatial queries, render
//! interpolation and debug drawing.

// ECS query tuples are inherently long; aliasing each one hurts more than helps.
#![allow(clippy::type_complexity)]

pub mod character;
pub mod components;
pub mod debug;
pub mod events;
pub mod joints;
pub mod layers;
pub mod mapping;
pub mod plugin;
pub mod pose;
pub mod query;
pub mod reflect;
pub mod shapes;
pub mod systems;
pub mod world3d;

pub use character::{CharacterController, CharacterInput, CharacterState};
pub use components::{
    ColliderHandle3D, ColliderShape3D, CollisionLayer, ExternalForce, ExternalImpulse,
    JointHandle3D, PhysicsMaterial, PhysicsPose, PhysicsRejected, RigidBody3DBundle,
    RigidBodyHandle3D, RigidBodySettings, RigidBodyType, Sensor, Velocity,
};
pub use debug::PhysicsDebugSettings;
pub use events::{CollisionStarted, CollisionStopped, TriggerEntered, TriggerExited};
pub use joints::{JointKind, JointMotor, PhysicsJoint};
pub use layers::{LayerMask, PhysicsLayers, MAX_LAYERS};
pub use mapping::{ColliderEntityMap3D, PhysicsEntityHandles3D};
pub use plugin::PhysicsPlugin;
pub use query::{
    raycast, PhysicsQueries, QueryShape, RaycastHit, ShapeCastHit, SpatialQuery, SpatialQueryFilter,
};
pub use reflect::register_physics_reflection_types;
pub use systems::{
    cleanup_orphaned_bodies, physics_fixed_update_systems_3d, physics_sync_systems,
    step_physics_world, sync_kinematic_bodies_from_transforms, sync_new_bodies, sync_new_colliders,
    write_back_transforms, PhysicsStepConfig3D,
};
pub use world3d::PhysicsWorld3D;

use engine_core::Result;

#[derive(Default)]
pub struct PhysicsModule;

impl PhysicsModule {
    pub fn new() -> Self {
        Self
    }

    pub fn step(&self, dt_seconds: f32) -> Result<()> {
        log::trace!(
            target: "engine::physics",
            "Physics module fixed_update tick: {:.4}",
            dt_seconds
        );
        Ok(())
    }
}

pub fn module_name() -> &'static str {
    "engine-physics"
}

pub fn dimensions_supported() -> (&'static str, &'static str) {
    (
        std::any::type_name::<rapier2d::prelude::RigidBodySet>(),
        std::any::type_name::<rapier3d::prelude::RigidBodySet>(),
    )
}
