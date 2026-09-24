//! Runtime plugin wiring 3D physics into a [`GameRuntime`].

use bevy_ecs::schedule::IntoSystemConfigs;
use engine_core::{FixedSet, GameRuntime, PreRenderSet, RuntimePlugin, ScheduleKind};

use crate::debug::{draw_physics_debug, PhysicsDebugSettings};
use crate::events::{
    publish_collision_events, CollisionStarted, CollisionStopped, TriggerEntered, TriggerExited,
};
use crate::layers::PhysicsLayers;
use crate::mapping::{ColliderEntityMap3D, PhysicsEntityHandles3D};
use crate::reflect::register_physics_reflection_types;
use crate::systems::{
    interpolate_transforms, physics_sync_systems, step_physics_world, write_back_transforms,
    PhysicsStepConfig3D,
};
use crate::world3d::PhysicsWorld3D;

/// Installs the physics world, its resources, events and the fixed-step
/// systems in the canonical `FixedSet::Physics*` sets.
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
            .init_resource::<PhysicsLayers>()
            .init_resource::<PhysicsDebugSettings>()
            .add_event::<CollisionStarted>()
            .add_event::<CollisionStopped>()
            .add_event::<TriggerEntered>()
            .add_event::<TriggerExited>()
            .add_systems(
                ScheduleKind::FixedUpdate,
                (
                    physics_sync_systems().in_set(FixedSet::PhysicsSync),
                    step_physics_world.in_set(FixedSet::PhysicsStep),
                    write_back_transforms.in_set(FixedSet::PhysicsWriteback),
                    publish_collision_events.in_set(FixedSet::PhysicsEvents),
                ),
            )
            .add_systems(
                ScheduleKind::PreRender,
                (
                    interpolate_transforms.before(PreRenderSet::TransformPropagate),
                    draw_physics_debug.in_set(PreRenderSet::Late),
                ),
            );
        engine_reflect::with_reflection_registries(&mut runtime.world, |t, c, m| {
            register_physics_reflection_types(t, c, m)
        });
    }
}
