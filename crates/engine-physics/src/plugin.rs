//! Runtime plugin wiring 3D physics into a [`GameRuntime`].

use bevy_ecs::schedule::IntoSystemConfigs;
use engine_core::{FixedSet, GameRuntime, RuntimePlugin, ScheduleKind};

use crate::mapping::{ColliderEntityMap3D, PhysicsEntityHandles3D};
use crate::reflect::register_physics_reflection_types;
use crate::systems::{
    cleanup_orphaned_bodies, step_physics_world, sync_kinematic_bodies_from_transforms,
    sync_new_bodies, write_back_transforms, PhysicsStepConfig3D,
};
use crate::world3d::PhysicsWorld3D;

/// Installs the physics world, its resources and the fixed-step systems in
/// the canonical `FixedSet::Physics*` sets.
#[derive(Default)]
pub struct PhysicsPlugin;

impl RuntimePlugin for PhysicsPlugin {
    fn name(&self) -> &'static str {
        "engine::physics"
    }

    fn build(&self, runtime: &mut GameRuntime) {
        let dt = runtime.fixed_timestep_seconds();
        runtime
            .insert_resource(PhysicsWorld3D::with_timestep(dt))
            .insert_resource(PhysicsStepConfig3D::new(dt))
            .init_resource::<ColliderEntityMap3D>()
            .init_resource::<PhysicsEntityHandles3D>()
            .add_systems(
                ScheduleKind::FixedUpdate,
                (
                    (
                        cleanup_orphaned_bodies,
                        sync_new_bodies,
                        sync_kinematic_bodies_from_transforms,
                    )
                        .chain()
                        .in_set(FixedSet::PhysicsSync),
                    step_physics_world.in_set(FixedSet::PhysicsStep),
                    write_back_transforms.in_set(FixedSet::PhysicsWriteback),
                ),
            );
        engine_reflect::with_reflection_registries(&mut runtime.world, |t, c, m| {
            register_physics_reflection_types(t, c, m)
        });
    }
}
