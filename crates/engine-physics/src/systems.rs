//! ECS ↔ Rapier synchronisation.
//!
//! Fixed-step order (see [`crate::PhysicsPlugin`]):
//! `PhysicsSync`: cleanup → new bodies → new colliders → component
//! changes → transforms/velocities/forces → joints → character
//! controllers; `PhysicsStep`: the solver; `PhysicsWriteback`: body poses
//! and velocities back to the ECS; `PhysicsEvents`: typed events.
//! Pre-render interpolates body transforms between the last two steps.

use bevy_ecs::prelude::*;
use bevy_ecs::schedule::IntoSystemConfigs;
use engine_assets::Assets;
use engine_core::{FrameTime, Parent, Transform, DEFAULT_FIXED_TIMESTEP_SECONDS};
use engine_math::Vec3;
use rapier3d::prelude::{
    ActiveCollisionTypes, ActiveEvents, ColliderBuilder, InteractionGroups, LockedAxes, RigidBody,
    RigidBodyActivation, RigidBodyBuilder,
};

use crate::components::{
    ColliderHandle3D, ColliderShape3D, CollisionLayer, ExternalForce, ExternalImpulse,
    PhysicsMaterial, PhysicsPose, PhysicsRejected, RigidBodyHandle3D, RigidBodySettings,
    RigidBodyType, Sensor, Velocity,
};
use crate::layers::PhysicsLayers;
use crate::mapping::{ColliderEntityMap3D, PhysicsEntityHandles3D};
use crate::pose::{
    from_isometry, from_vector, parent_affine, to_isometry, to_vector, world_affine, world_pose,
    world_to_local, Pose,
};
use crate::shapes::{build_shape, ShapeBuild};
use crate::world3d::PhysicsWorld3D;

type Transforms<'w, 's> = Query<'w, 's, (&'static Transform, Option<&'static Parent>)>;

#[derive(Resource, Clone, Copy, Debug)]
pub struct PhysicsStepConfig3D {
    pub fixed_dt_seconds: f32,
}

impl PhysicsStepConfig3D {
    pub fn new(fixed_dt_seconds: f32) -> Self {
        Self {
            fixed_dt_seconds: fixed_dt_seconds.max(0.000_001),
        }
    }
}

impl Default for PhysicsStepConfig3D {
    fn default() -> Self {
        Self::new(DEFAULT_FIXED_TIMESTEP_SECONDS as f32)
    }
}

/// The complete fixed-step physics pipeline as one chained set (tests and
/// hosts that do not use [`crate::PhysicsPlugin`]'s system sets).
pub fn physics_fixed_update_systems_3d() -> impl IntoSystemConfigs<()> {
    (
        physics_sync_systems(),
        step_physics_world,
        write_back_transforms,
    )
        .chain()
}

/// The `PhysicsSync` systems, chained.
pub fn physics_sync_systems() -> impl IntoSystemConfigs<()> {
    (
        clear_rejections,
        cleanup_orphaned_bodies,
        sync_new_bodies,
        sync_new_colliders,
        sync_changed_components,
        sync_kinematic_bodies_from_transforms,
        apply_velocities_and_forces,
        crate::joints::sync_joints,
        crate::character::ensure_character_components,
        crate::character::move_character_controllers,
    )
        .chain()
}

fn rapier_body_type(body_type: RigidBodyType) -> rapier3d::prelude::RigidBodyType {
    match body_type {
        RigidBodyType::Dynamic => rapier3d::prelude::RigidBodyType::Dynamic,
        RigidBodyType::Kinematic => rapier3d::prelude::RigidBodyType::KinematicPositionBased,
        RigidBodyType::Static => rapier3d::prelude::RigidBodyType::Fixed,
    }
}

fn apply_body_settings(body: &mut RigidBody, settings: &RigidBodySettings) {
    body.set_gravity_scale(settings.gravity_scale, true);
    body.set_linear_damping(settings.linear_damping.max(0.0));
    body.set_angular_damping(settings.angular_damping.max(0.0));
    body.enable_ccd(settings.ccd);
    let mut locked = LockedAxes::empty();
    for (flag, on) in [
        (
            LockedAxes::TRANSLATION_LOCKED_X,
            settings.lock_translation_x,
        ),
        (
            LockedAxes::TRANSLATION_LOCKED_Y,
            settings.lock_translation_y,
        ),
        (
            LockedAxes::TRANSLATION_LOCKED_Z,
            settings.lock_translation_z,
        ),
        (LockedAxes::ROTATION_LOCKED_X, settings.lock_rotation_x),
        (LockedAxes::ROTATION_LOCKED_Y, settings.lock_rotation_y),
        (LockedAxes::ROTATION_LOCKED_Z, settings.lock_rotation_z),
    ] {
        if on {
            locked |= flag;
        }
    }
    body.set_locked_axes(locked, true);
    let can_sleep = body.activation().normalized_linear_threshold >= 0.0;
    if can_sleep != settings.can_sleep {
        *body.activation_mut() = if settings.can_sleep {
            RigidBodyActivation::active()
        } else {
            RigidBodyActivation::cannot_sleep()
        };
    }
}

fn collider_groups(
    layers: Option<&PhysicsLayers>,
    layer: Option<&CollisionLayer>,
) -> InteractionGroups {
    match layers {
        Some(layers) => layers.groups(layer.map_or(0, |layer| layer.layer)),
        None => InteractionGroups::all(),
    }
}

/// Lets rejected entities retry after their physics components change.
pub fn clear_rejections(
    mut commands: Commands,
    changed: Query<
        Entity,
        (
            With<PhysicsRejected>,
            Or<(Changed<RigidBodyType>, Changed<ColliderShape3D>)>,
        ),
    >,
) {
    for entity in &changed {
        commands.entity(entity).remove::<PhysicsRejected>();
    }
}

/// Removes Rapier objects whose entity (or component) is gone.
pub fn cleanup_orphaned_bodies(
    mut commands: Commands,
    entities: Query<(
        Has<RigidBodyHandle3D>,
        Has<RigidBodyType>,
        Has<ColliderHandle3D>,
        Has<ColliderShape3D>,
    )>,
    mut physics: Option<ResMut<PhysicsWorld3D>>,
    mut collider_entity_map: Option<ResMut<ColliderEntityMap3D>>,
    mut entity_handles: Option<ResMut<PhysicsEntityHandles3D>>,
) {
    let (Some(physics), Some(collider_entity_map), Some(entity_handles)) = (
        physics.as_deref_mut(),
        collider_entity_map.as_deref_mut(),
        entity_handles.as_deref_mut(),
    ) else {
        return;
    };
    let PhysicsWorld3D {
        rigid_body_set,
        collider_set,
        island_manager,
        impulse_joint_set,
        multibody_joint_set,
        ..
    } = physics;

    let stale_bodies: Vec<Entity> = entity_handles
        .body_entities()
        .filter(|entity| !matches!(entities.get(*entity), Ok((true, true, _, _))))
        .collect();
    for entity in stale_bodies {
        if let Some(handle) = entity_handles.remove_body(entity) {
            rigid_body_set.remove(
                handle,
                island_manager,
                collider_set,
                impulse_joint_set,
                multibody_joint_set,
                true,
            );
        }
        if let Some(mut commands) = commands.get_entity(entity) {
            commands.remove::<(
                RigidBodyHandle3D,
                PhysicsPose,
                crate::components::JointHandle3D,
            )>();
        }
    }

    let stale_colliders: Vec<Entity> = entity_handles
        .collider_entities()
        .filter(|entity| {
            let alive = matches!(entities.get(*entity), Ok((_, _, true, true)));
            let handle = entity_handles.collider(*entity);
            !alive || handle.is_none_or(|handle| !collider_set.contains(handle))
        })
        .collect();
    for entity in stale_colliders {
        if let Some(handle) = entity_handles.remove_collider(entity) {
            collider_set.remove(handle, island_manager, rigid_body_set, true);
            collider_entity_map.remove(handle);
        }
        if let Some(mut commands) = commands.get_entity(entity) {
            // Recreated next step if the shape is still there (e.g. its
            // owning body was removed and it becomes standalone).
            commands.remove::<ColliderHandle3D>();
        }
    }
}

type NewBodyData = (
    Entity,
    &'static RigidBodyType,
    Option<&'static ColliderShape3D>,
    Option<&'static RigidBodySettings>,
    Option<&'static Velocity>,
);

/// Creates Rapier bodies for new `RigidBodyType` entities.
pub fn sync_new_bodies(
    mut commands: Commands,
    query: Query<NewBodyData, (Without<RigidBodyHandle3D>, Without<PhysicsRejected>)>,
    transforms: Transforms,
    mut physics: Option<ResMut<PhysicsWorld3D>>,
    mut entity_handles: Option<ResMut<PhysicsEntityHandles3D>>,
) {
    let (Some(physics), Some(entity_handles)) =
        (physics.as_deref_mut(), entity_handles.as_deref_mut())
    else {
        return;
    };

    for (entity, body_type, shape, settings, velocity) in &query {
        let concave_ok = match shape {
            Some(ColliderShape3D::Trimesh) => *body_type == RigidBodyType::Static,
            Some(shape) if shape.is_concave() => *body_type != RigidBodyType::Dynamic,
            _ => true,
        };
        if !concave_ok {
            let reason = format!(
                "{:?} bodies cannot use concave colliders; use a convex mesh or a Static body",
                body_type
            );
            log::warn!(target: "engine::physics", "rejected physics body for {entity:?}: {reason}");
            commands.entity(entity).insert(PhysicsRejected(reason));
            continue;
        }

        let world = world_pose(entity, &transforms);
        let local = transforms
            .get(entity)
            .map(|(t, _)| (t.translation, t.rotation))
            .unwrap_or(world);
        let settings = settings.copied().unwrap_or_default();
        let mut body = RigidBodyBuilder::new(rapier_body_type(*body_type))
            .position(to_isometry(world))
            .build();
        apply_body_settings(&mut body, &settings);
        if let Some(velocity) = velocity {
            body.set_linvel(to_vector(velocity.linear), true);
            body.set_angvel(to_vector(velocity.angular), true);
        }
        let handle = physics.rigid_body_set.insert(body);
        entity_handles.insert_body(entity, handle);
        let interpolate = settings.interpolate && *body_type == RigidBodyType::Dynamic;
        commands.entity(entity).insert((
            RigidBodyHandle3D(handle),
            PhysicsPose::new(world, local, interpolate),
        ));
    }
}

type NewColliderData = (
    Entity,
    &'static ColliderShape3D,
    Option<&'static PhysicsMaterial>,
    Option<&'static CollisionLayer>,
    Has<Sensor>,
    Has<RigidBodyType>,
);

/// Creates colliders: on the entity's own body, on the nearest ancestor
/// body, or standalone (static) when no ancestor has a body.
#[allow(clippy::too_many_arguments)]
pub fn sync_new_colliders(
    mut commands: Commands,
    query: Query<NewColliderData, (Without<ColliderHandle3D>, Without<PhysicsRejected>)>,
    body_markers: Query<(), With<RigidBodyType>>,
    transforms: Transforms,
    assets: Option<Res<Assets>>,
    layers: Option<Res<PhysicsLayers>>,
    mut physics: Option<ResMut<PhysicsWorld3D>>,
    mut collider_entity_map: Option<ResMut<ColliderEntityMap3D>>,
    mut entity_handles: Option<ResMut<PhysicsEntityHandles3D>>,
) {
    let (Some(physics), Some(collider_entity_map), Some(entity_handles)) = (
        physics.as_deref_mut(),
        collider_entity_map.as_deref_mut(),
        entity_handles.as_deref_mut(),
    ) else {
        return;
    };

    let mut created = false;
    for (entity, shape, material, layer, sensor, has_body) in &query {
        // Find the owning body.
        let owner = if has_body {
            match entity_handles.body(entity) {
                Some(body) => Some((entity, body)),
                None => continue, // body pending or rejected
            }
        } else {
            let mut owner = None;
            let mut pending = false;
            let mut current = transforms
                .get(entity)
                .ok()
                .and_then(|(_, p)| p.map(|p| p.0));
            for _ in 0..256 {
                let Some(ancestor) = current else {
                    break;
                };
                if let Some(body) = entity_handles.body(ancestor) {
                    owner = Some((ancestor, body));
                    break;
                }
                if body_markers.contains(ancestor) {
                    pending = true;
                    break;
                }
                current = transforms
                    .get(ancestor)
                    .ok()
                    .and_then(|(_, p)| p.map(|p| p.0));
            }
            if pending {
                continue;
            }
            owner
        };

        let shared = match build_shape(shape, assets.as_deref()) {
            ShapeBuild::Ready(shape) => shape,
            ShapeBuild::Pending => continue,
            ShapeBuild::Failed(reason) => {
                log::warn!(target: "engine::physics", "rejected collider for {entity:?}: {reason}");
                commands.entity(entity).insert(PhysicsRejected(reason));
                continue;
            }
        };
        let material = material.copied().unwrap_or_default();
        let groups = collider_groups(layers.as_deref(), layer);
        let mut builder = ColliderBuilder::new(shared)
            .friction(material.friction.clamp(0.0, 1.0))
            .restitution(material.restitution.clamp(0.0, 1.0))
            .density(material.density.max(0.000_001))
            .collision_groups(groups)
            .solver_groups(groups)
            .sensor(sensor)
            .active_events(ActiveEvents::COLLISION_EVENTS);
        if sensor {
            builder = builder.active_collision_types(ActiveCollisionTypes::all());
        }

        let handle = match owner {
            Some((owner_entity, body)) => {
                if owner_entity != entity {
                    let owner_world = world_affine(owner_entity, &transforms);
                    let relative = world_to_local(&owner_world, world_pose(entity, &transforms));
                    builder = builder.position(to_isometry(relative));
                }
                let PhysicsWorld3D {
                    rigid_body_set,
                    collider_set,
                    ..
                } = &mut *physics;
                collider_set.insert_with_parent(builder.build(), body, rigid_body_set)
            }
            None => {
                builder = builder.position(to_isometry(world_pose(entity, &transforms)));
                physics.collider_set.insert(builder.build())
            }
        };
        collider_entity_map.insert(handle, entity);
        entity_handles.insert_collider(entity, handle);
        commands.entity(entity).insert(ColliderHandle3D(handle));
        created = true;
    }
    if created {
        // Queries and character controllers see new geometry right away.
        physics.update_query_pipeline();
    }
}

type ChangedBodyData = (
    &'static RigidBodyHandle3D,
    Ref<'static, RigidBodyType>,
    Option<Ref<'static, RigidBodySettings>>,
    Option<&'static mut PhysicsPose>,
);

type ChangedColliderData = (
    Entity,
    &'static ColliderHandle3D,
    Ref<'static, ColliderShape3D>,
    Option<Ref<'static, PhysicsMaterial>>,
    Option<Ref<'static, CollisionLayer>>,
);

/// Applies edits of body type/settings, materials, layers, sensors and
/// shapes to existing Rapier objects.
#[allow(clippy::too_many_arguments)]
pub fn sync_changed_components(
    mut commands: Commands,
    mut bodies: Query<ChangedBodyData>,
    colliders: Query<ChangedColliderData>,
    added_sensors: Query<Entity, Added<Sensor>>,
    mut removed_sensors: RemovedComponents<Sensor>,
    layer_components: Query<&CollisionLayer>,
    layers: Option<Res<PhysicsLayers>>,
    mut physics: Option<ResMut<PhysicsWorld3D>>,
    mut collider_entity_map: Option<ResMut<ColliderEntityMap3D>>,
    mut entity_handles: Option<ResMut<PhysicsEntityHandles3D>>,
) {
    let (Some(physics), Some(collider_entity_map), Some(entity_handles)) = (
        physics.as_deref_mut(),
        collider_entity_map.as_deref_mut(),
        entity_handles.as_deref_mut(),
    ) else {
        return;
    };

    for (handle, body_type, settings, pose) in &mut bodies {
        let Some(body) = physics.rigid_body_set.get_mut(handle.0) else {
            continue;
        };
        if body_type.is_changed() && !body_type.is_added() {
            body.set_body_type(rapier_body_type(*body_type), true);
        }
        if let Some(settings) = settings {
            if settings.is_changed() && !settings.is_added() {
                apply_body_settings(body, &settings);
            }
            if let Some(mut pose) = pose {
                pose.interpolate = settings.interpolate && *body_type == RigidBodyType::Dynamic;
            }
        }
    }

    for (entity, handle, shape, material, layer) in &colliders {
        if shape.is_changed() && !shape.is_added() {
            // New geometry: rebuild the collider next step.
            let PhysicsWorld3D {
                rigid_body_set,
                collider_set,
                island_manager,
                ..
            } = &mut *physics;
            collider_set.remove(handle.0, island_manager, rigid_body_set, true);
            collider_entity_map.remove(handle.0);
            entity_handles.remove_collider(entity);
            commands.entity(entity).remove::<ColliderHandle3D>();
            continue;
        }
        let Some(collider) = physics.collider_set.get_mut(handle.0) else {
            continue;
        };
        if let Some(material) = material.filter(|m| m.is_changed() && !m.is_added()) {
            collider.set_friction(material.friction.clamp(0.0, 1.0));
            collider.set_restitution(material.restitution.clamp(0.0, 1.0));
            collider.set_density(material.density.max(0.000_001));
        }
        let layer_changed = layer
            .as_ref()
            .is_some_and(|l| l.is_changed() && !l.is_added());
        if layer_changed {
            let groups = collider_groups(layers.as_deref(), layer.as_deref());
            collider.set_collision_groups(groups);
            collider.set_solver_groups(groups);
        }
    }

    let mut set_sensor = |entity: Entity, on: bool| {
        if let Some(handle) = entity_handles.collider(entity) {
            if let Some(collider) = physics.collider_set.get_mut(handle) {
                collider.set_sensor(on);
                collider.set_active_collision_types(if on {
                    ActiveCollisionTypes::all()
                } else {
                    ActiveCollisionTypes::default()
                });
            }
        }
    };
    for entity in &added_sensors {
        set_sensor(entity, true);
    }
    for entity in removed_sensors.read() {
        set_sensor(entity, false);
    }

    if let Some(layers) = layers.as_ref().filter(|l| l.is_changed() && !l.is_added()) {
        for (handle, entity) in collider_entity_map.as_map() {
            if let Some(collider) = physics.collider_set.get_mut(*handle) {
                let groups = layers.groups(layer_components.get(*entity).map_or(0, |l| l.layer));
                collider.set_collision_groups(groups);
                collider.set_solver_groups(groups);
            }
        }
    }
}

type MovedBodyData = (
    Entity,
    &'static RigidBodyHandle3D,
    &'static RigidBodyType,
    Option<&'static mut PhysicsPose>,
);

/// Applies gameplay edits of `Transform` to bodies and colliders:
/// kinematic bodies get a kinematic target, dynamic/static bodies are
/// teleported (translation and rotation independently, so rotating a
/// character does not snap it to its interpolated position), colliders
/// follow their entity.
pub fn sync_kinematic_bodies_from_transforms(
    mut bodies: Query<MovedBodyData>,
    changed: Query<Entity, Changed<Transform>>,
    colliders: Query<(Entity, &ColliderHandle3D), Without<RigidBodyHandle3D>>,
    transforms: Transforms,
    mut physics: Option<ResMut<PhysicsWorld3D>>,
    entity_handles: Option<Res<PhysicsEntityHandles3D>>,
) {
    let (Some(physics), Some(entity_handles)) = (physics.as_deref_mut(), entity_handles) else {
        return;
    };

    for (entity, handle, body_type, pose) in &mut bodies {
        if !changed.contains(entity) {
            continue;
        }
        let Ok((transform, _)) = transforms.get(entity) else {
            continue;
        };
        let local = (transform.translation, transform.rotation);
        let (moved_translation, moved_rotation) = match &pose {
            Some(pose) => (
                local.0 != pose.last_written.0,
                local.1 != pose.last_written.1,
            ),
            None => (true, true),
        };
        if !moved_translation && !moved_rotation {
            continue;
        }
        let Some(body) = physics.rigid_body_set.get_mut(handle.0) else {
            continue;
        };
        let target = world_pose(entity, &transforms);
        let (current_t, current_r) = from_isometry(body.next_position());
        let world: Pose = (
            if moved_translation {
                target.0
            } else {
                current_t
            },
            if moved_rotation { target.1 } else { current_r },
        );
        match body_type {
            RigidBodyType::Kinematic => body.set_next_kinematic_position(to_isometry(world)),
            RigidBodyType::Dynamic | RigidBodyType::Static => {
                body.set_position(to_isometry(world), true)
            }
        }
        if let Some(mut pose) = pose {
            pose.last_written = local;
            if moved_translation {
                // A teleport must not be smeared by interpolation.
                pose.previous = world;
            } else {
                pose.previous.1 = world.1;
            }
            pose.current = world;
        }
    }

    for (entity, handle) in &colliders {
        if !changed.contains(entity) {
            continue;
        }
        let Some(collider) = physics.collider_set.get_mut(handle.0) else {
            continue;
        };
        match collider.parent() {
            None => collider.set_position(to_isometry(world_pose(entity, &transforms))),
            Some(body) => {
                let Some(owner) = entity_handles.body_entity(body) else {
                    continue;
                };
                let owner_world = world_affine(owner, &transforms);
                let relative = world_to_local(&owner_world, world_pose(entity, &transforms));
                collider.set_position_wrt_parent(to_isometry(relative));
            }
        }
    }
}

/// Gameplay-set velocities, continuous forces and one-shot impulses.
pub fn apply_velocities_and_forces(
    velocities: Query<(&RigidBodyHandle3D, Ref<Velocity>)>,
    forces: Query<(&RigidBodyHandle3D, &ExternalForce)>,
    mut removed_forces: RemovedComponents<ExternalForce>,
    mut impulses: Query<(&RigidBodyHandle3D, &mut ExternalImpulse)>,
    mut physics: Option<ResMut<PhysicsWorld3D>>,
    entity_handles: Option<Res<PhysicsEntityHandles3D>>,
) {
    let Some(physics) = physics.as_deref_mut() else {
        return;
    };
    for (handle, velocity) in &velocities {
        if velocity.is_changed() && !velocity.is_added() {
            if let Some(body) = physics.rigid_body_set.get_mut(handle.0) {
                body.set_linvel(to_vector(velocity.linear), true);
                body.set_angvel(to_vector(velocity.angular), true);
            }
        }
    }
    for (handle, force) in &forces {
        if let Some(body) = physics.rigid_body_set.get_mut(handle.0) {
            let active = force.force != Vec3::ZERO || force.torque != Vec3::ZERO;
            body.reset_forces(false);
            body.reset_torques(false);
            body.add_force(to_vector(force.force), active);
            body.add_torque(to_vector(force.torque), active);
        }
    }
    if let Some(entity_handles) = entity_handles {
        for entity in removed_forces.read() {
            if let Some(body) = entity_handles
                .body(entity)
                .and_then(|handle| physics.rigid_body_set.get_mut(handle))
            {
                body.reset_forces(false);
                body.reset_torques(false);
            }
        }
    }
    for (handle, mut impulse) in &mut impulses {
        if impulse.impulse == Vec3::ZERO && impulse.torque_impulse == Vec3::ZERO {
            continue;
        }
        if let Some(body) = physics.rigid_body_set.get_mut(handle.0) {
            body.apply_impulse(to_vector(impulse.impulse), true);
            body.apply_torque_impulse(to_vector(impulse.torque_impulse), true);
        }
        *impulse = ExternalImpulse::default();
    }
}

pub fn step_physics_world(
    mut physics: Option<ResMut<PhysicsWorld3D>>,
    step_config: Option<Res<PhysicsStepConfig3D>>,
) {
    let Some(physics) = physics.as_deref_mut() else {
        return;
    };

    if let Some(step_config) = step_config {
        physics.set_timestep(step_config.fixed_dt_seconds);
    }

    physics.step();
}

type WritebackData = (
    Entity,
    &'static RigidBodyHandle3D,
    &'static RigidBodyType,
    &'static mut Transform,
    Option<&'static mut PhysicsPose>,
    Option<&'static mut Velocity>,
);

/// Dynamic body poses (converted into the entity's parent space) and
/// velocities back to the ECS.
pub fn write_back_transforms(
    mut set: ParamSet<(Transforms, Query<WritebackData>)>,
    parents: Query<&Parent>,
    physics: Option<Res<PhysicsWorld3D>>,
) {
    let Some(physics) = physics.as_deref() else {
        return;
    };

    // Parent spaces first (read-only pass).
    let with_parents: Vec<Entity> = set
        .p1()
        .iter()
        .filter(|(entity, _, body_type, _, _, _)| {
            **body_type == RigidBodyType::Dynamic && parents.contains(*entity)
        })
        .map(|(entity, ..)| entity)
        .collect();
    let parent_spaces: std::collections::HashMap<Entity, engine_math::Affine3A> = {
        let transforms = set.p0();
        with_parents
            .into_iter()
            .map(|entity| (entity, parent_affine(entity, &transforms)))
            .collect()
    };

    let mut bodies = set.p1();
    for (entity, handle, body_type, mut transform, pose, velocity) in &mut bodies {
        let Some(body) = physics.rigid_body_set.get(handle.0) else {
            continue;
        };
        if let Some(mut velocity) = velocity {
            let linear = from_vector(body.linvel());
            let angular = from_vector(body.angvel());
            if velocity.linear != linear || velocity.angular != angular {
                let velocity = velocity.bypass_change_detection();
                velocity.linear = linear;
                velocity.angular = angular;
            }
        }
        if *body_type != RigidBodyType::Dynamic || body.is_sleeping() {
            continue;
        }
        let world = from_isometry(body.position());
        let local = match parent_spaces.get(&entity) {
            Some(parent) => world_to_local(parent, world),
            None => world,
        };
        transform.translation = local.0;
        transform.rotation = local.1;
        if let Some(mut pose) = pose {
            pose.previous = pose.current;
            pose.current = world;
            pose.last_written = local;
        }
    }
}

/// Pre-render: moves interpolated bodies to their pose between the last
/// two fixed steps, so motion is smooth at any frame rate.
pub fn interpolate_transforms(
    mut set: ParamSet<(
        Transforms,
        Query<(Entity, &mut Transform, &mut PhysicsPose)>,
    )>,
    parents: Query<&Parent>,
    time: Option<Res<FrameTime>>,
) {
    let alpha = time.map_or(1.0, |time| time.alpha).clamp(0.0, 1.0);
    let with_parents: Vec<Entity> = set
        .p1()
        .iter()
        .filter(|(entity, _, pose)| pose.interpolate && parents.contains(*entity))
        .map(|(entity, _, _)| entity)
        .collect();
    let parent_spaces: std::collections::HashMap<Entity, engine_math::Affine3A> = {
        let transforms = set.p0();
        with_parents
            .into_iter()
            .map(|entity| (entity, parent_affine(entity, &transforms)))
            .collect()
    };
    for (entity, mut transform, mut pose) in &mut set.p1() {
        if !pose.interpolate || pose.previous == pose.current {
            continue;
        }
        let world = pose.interpolated(alpha);
        let local = match parent_spaces.get(&entity) {
            Some(parent) => world_to_local(parent, world),
            None => world,
        };
        if (transform.translation, transform.rotation) != local {
            transform.translation = local.0;
            transform.rotation = local.1;
        }
        pose.last_written = local;
    }
}

#[cfg(test)]
#[path = "systems_tests.rs"]
mod tests;
