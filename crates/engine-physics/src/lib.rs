pub mod components;
pub mod mapping;
pub mod plugin;
pub mod query;
pub mod reflect;
pub mod systems;
pub mod world3d;

pub use components::{
    ColliderHandle3D, ColliderShape3D, PhysicsMaterial, RigidBody3DBundle, RigidBodyHandle3D,
    RigidBodyType,
};
pub use mapping::{ColliderEntityMap3D, PhysicsEntityHandles3D};
pub use plugin::PhysicsPlugin;
pub use query::{raycast, RaycastHit};
pub use reflect::register_physics_reflection_types;
pub use systems::{
    cleanup_orphaned_bodies, physics_fixed_update_systems_3d, step_physics_world,
    sync_kinematic_bodies_from_transforms, sync_new_bodies, write_back_transforms,
    PhysicsStepConfig3D,
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
