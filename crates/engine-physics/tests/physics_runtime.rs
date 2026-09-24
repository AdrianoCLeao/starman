//! Physics through the shared runtime: events, layers, compound
//! colliders, joints, character controller, queries, teleports.

use bevy_ecs::prelude::*;
use engine_assets::{AssetRef, Assets, MeshData, MeshVertex};
use engine_core::{
    DebugCategory, DebugDraw, EntityId, GameRuntime, Parent, PersistentId, Transform,
};
use engine_math::{Quat, Vec3};
use engine_physics::*;

const DT: f32 = 1.0 / 60.0;

fn runtime() -> GameRuntime {
    let mut runtime = GameRuntime::with_fixed_timestep(DT as f64);
    runtime.add_plugin(PhysicsPlugin);
    runtime
}

fn run(runtime: &mut GameRuntime, steps: usize) {
    for _ in 0..steps {
        runtime.step(DT);
    }
}

fn floor(runtime: &mut GameRuntime) -> Entity {
    runtime
        .world
        .spawn((
            RigidBodyType::Static,
            ColliderShape3D::Box {
                half_extents: Vec3::new(30.0, 0.5, 30.0),
            },
            Transform::from_xyz(0.0, -0.5, 0.0),
        ))
        .id()
}

fn events<E: Event + Clone>(runtime: &GameRuntime) -> Vec<E> {
    let events = runtime.world.resource::<Events<E>>();
    events.get_cursor().read(events).cloned().collect()
}

#[derive(Resource, Default)]
struct Collected {
    entered: Vec<TriggerEntered>,
    exited: Vec<TriggerExited>,
    started: Vec<CollisionStarted>,
}

fn collect(
    mut collected: ResMut<Collected>,
    mut entered: EventReader<TriggerEntered>,
    mut exited: EventReader<TriggerExited>,
    mut started: EventReader<CollisionStarted>,
) {
    collected.entered.extend(entered.read().copied());
    collected.exited.extend(exited.read().copied());
    collected.started.extend(started.read().copied());
}

fn collecting_runtime() -> GameRuntime {
    let mut runtime = runtime();
    runtime.init_resource::<Collected>();
    runtime.add_systems(engine_core::ScheduleKind::Update, collect);
    runtime
}

fn character(runtime: &mut GameRuntime, at: Vec3) -> Entity {
    runtime
        .world
        .spawn((
            RigidBodyType::Kinematic,
            ColliderShape3D::Capsule {
                half_height: 0.5,
                radius: 0.3,
            },
            CharacterController::default(),
            Transform::from_translation(at),
        ))
        .id()
}

#[test]
fn dynamic_body_hits_the_floor_and_reports_the_collision() {
    let mut runtime = collecting_runtime();
    let floor = floor(&mut runtime);
    let ball = runtime
        .world
        .spawn((
            RigidBodyType::Dynamic,
            ColliderShape3D::Sphere { radius: 0.5 },
            Velocity::default(),
            Transform::from_xyz(0.0, 3.0, 0.0),
        ))
        .id();
    run(&mut runtime, 90);
    let collected = runtime.world.resource::<Collected>();
    assert!(collected
        .started
        .iter()
        .any(|e| (e.a == ball && e.b == floor) || (e.a == floor && e.b == ball)));
    let y = runtime.world.get::<Transform>(ball).unwrap().translation.y;
    assert!((y - 0.5).abs() < 0.1, "resting on the floor, y = {y}");
    let velocity = runtime.world.get::<Velocity>(ball).unwrap();
    assert!(velocity.linear.length() < 0.5);
}

#[test]
fn character_controller_walks_jumps_and_triggers_sensors() {
    let mut runtime = collecting_runtime();
    floor(&mut runtime);
    let trigger = runtime
        .world
        .spawn((
            ColliderShape3D::Box {
                half_extents: Vec3::new(1.0, 1.0, 1.0),
            },
            Sensor,
            Transform::from_xyz(4.0, 1.0, 0.0),
        ))
        .id();
    let player = character(&mut runtime, Vec3::new(0.0, 1.0, 0.0));
    run(&mut runtime, 30);
    let state = *runtime.world.get::<CharacterState>(player).unwrap();
    assert!(state.grounded, "settles on the floor: {state:?}");

    runtime
        .world
        .get_mut::<CharacterInput>(player)
        .unwrap()
        .velocity = Vec3::new(3.0, 0.0, 0.0);
    run(&mut runtime, 150);
    let x = runtime
        .world
        .get::<Transform>(player)
        .unwrap()
        .translation
        .x;
    assert!(x > 6.0, "walked through the trigger, x = {x}");
    let collected = runtime.world.resource::<Collected>();
    assert!(collected
        .entered
        .iter()
        .any(|e| e.trigger == trigger && e.other == player && e.other_body == Some(player)));
    assert!(collected.exited.iter().any(|e| e.trigger == trigger));

    {
        let mut input = runtime.world.get_mut::<CharacterInput>(player).unwrap();
        input.velocity = Vec3::ZERO;
        input.jump_speed = 5.0;
    }
    run(&mut runtime, 10);
    let state = *runtime.world.get::<CharacterState>(player).unwrap();
    assert!(!state.grounded);
    assert!(
        runtime
            .world
            .get::<Transform>(player)
            .unwrap()
            .translation
            .y
            > 1.3
    );
    run(&mut runtime, 90);
    let state = *runtime.world.get::<CharacterState>(player).unwrap();
    assert!(state.grounded, "landed again");
    assert_eq!(
        runtime
            .world
            .get::<CharacterInput>(player)
            .unwrap()
            .jump_speed,
        0.0
    );
}

#[test]
fn character_controller_climbs_steps_and_is_blocked_by_walls() {
    let mut runtime = runtime();
    floor(&mut runtime);
    runtime.world.spawn((
        RigidBodyType::Static,
        ColliderShape3D::Box {
            half_extents: Vec3::new(1.0, 0.1, 3.0),
        },
        Transform::from_xyz(3.0, 0.1, 0.0),
    ));
    runtime.world.spawn((
        RigidBodyType::Static,
        ColliderShape3D::Box {
            half_extents: Vec3::new(0.5, 3.0, 3.0),
        },
        Transform::from_xyz(-3.0, 3.0, 0.0),
    ));
    let player = character(&mut runtime, Vec3::new(0.0, 0.9, 0.0));
    run(&mut runtime, 20);
    runtime
        .world
        .get_mut::<CharacterInput>(player)
        .unwrap()
        .velocity = Vec3::new(2.0, 0.0, 0.0);
    run(&mut runtime, 90);
    let t = runtime.world.get::<Transform>(player).unwrap().translation;
    assert!(t.x > 2.5 && t.y > 0.95, "stepped onto the 0.2m ledge: {t}");

    runtime
        .world
        .get_mut::<CharacterInput>(player)
        .unwrap()
        .velocity = Vec3::new(-4.0, 0.0, 0.0);
    run(&mut runtime, 180);
    let t = runtime.world.get::<Transform>(player).unwrap().translation;
    assert!(t.x > -2.6 && t.x < -2.0, "stopped by the wall: {t}");
}

#[test]
fn collision_layers_filter_contacts_and_queries() {
    let mut runtime = runtime();
    {
        let names: Vec<String> = ["default", "ghost"].iter().map(|s| s.to_string()).collect();
        let mut layers = PhysicsLayers::new(&names, &[]);
        layers.set_collides(0, 1, false);
        runtime.world.insert_resource(layers);
    }
    floor(&mut runtime);
    let ghost = runtime
        .world
        .spawn((
            RigidBodyType::Dynamic,
            ColliderShape3D::Sphere { radius: 0.5 },
            CollisionLayer { layer: 1 },
            Transform::from_xyz(0.0, 2.0, 0.0),
        ))
        .id();
    run(&mut runtime, 60);
    assert!(
        runtime.world.get::<Transform>(ghost).unwrap().translation.y < -1.0,
        "ghost falls through the default-layer floor"
    );

    // Queries filter by membership, not by the ghost's matrix row.
    let world = &runtime.world;
    let queries = PhysicsQueries::new(
        world.resource::<PhysicsWorld3D>(),
        world.resource::<ColliderEntityMap3D>(),
        world.resource::<PhysicsEntityHandles3D>(),
    );
    // Queries see the simulated pose (the transform is interpolated).
    let ghost_y = world.get::<PhysicsPose>(ghost).unwrap().current.0.y;
    let down = SpatialQueryFilter::default().with_layers(LayerMask::from_layer(1));
    let hit = queries
        .raycast(Vec3::new(0.0, 10.0, 0.0), -Vec3::Y, 100.0, &down)
        .expect("ghost is hit when querying its layer");
    assert_eq!(hit.entity, ghost);
    assert!((hit.point.y - (ghost_y + 0.5)).abs() < 0.05);
    let hit = queries
        .raycast(
            Vec3::new(0.0, 10.0, 0.0),
            -Vec3::Y,
            100.0,
            &SpatialQueryFilter::default().excluding(ghost),
        )
        .unwrap();
    assert_ne!(hit.entity, ghost, "excluded");
}

#[test]
fn child_colliders_form_one_compound_body() {
    let mut runtime = runtime();
    floor(&mut runtime);
    let body = runtime
        .world
        .spawn((
            RigidBodyType::Dynamic,
            RigidBodySettings::default(),
            Transform::from_xyz(0.0, 3.0, 0.0),
        ))
        .id();
    let mut children = Vec::new();
    for x in [-1.0, 1.0] {
        let child = runtime
            .world
            .spawn((
                ColliderShape3D::Box {
                    half_extents: Vec3::splat(0.25),
                },
                Transform::from_xyz(x, 0.0, 0.0),
                Parent(body),
            ))
            .id();
        children.push(child);
    }
    run(&mut runtime, 120);
    let physics = runtime.world.resource::<PhysicsWorld3D>();
    let handle = runtime.world.get::<RigidBodyHandle3D>(body).unwrap().0;
    assert_eq!(physics.rigid_body_set[handle].colliders().len(), 2);
    let y = runtime.world.get::<Transform>(body).unwrap().translation.y;
    assert!(
        (y - 0.25).abs() < 0.1,
        "compound rests on its two boxes: {y}"
    );

    // Despawning a child removes its collider; the body keeps the other.
    runtime.world.despawn(children[0]);
    run(&mut runtime, 1);
    let physics = runtime.world.resource::<PhysicsWorld3D>();
    assert_eq!(physics.rigid_body_set[handle].colliders().len(), 1);
    assert_eq!(runtime.world.resource::<ColliderEntityMap3D>().len(), 2);
}

#[test]
fn joints_hold_bodies_to_the_world_and_to_each_other() {
    let mut runtime = runtime();
    // A pendulum hanging from the world, and a weight welded to it.
    let pendulum = runtime
        .world
        .spawn((
            RigidBodyType::Dynamic,
            ColliderShape3D::Sphere { radius: 0.2 },
            Transform::from_xyz(2.0, 5.0, 0.0),
            PhysicsJoint {
                kind: JointKind::Spherical,
                target_anchor: Vec3::new(0.0, 5.0, 0.0),
                local_anchor: Vec3::new(-2.0, 0.0, 0.0),
                ..Default::default()
            },
        ))
        .id();
    let pendulum_id = EntityId::new_v4();
    runtime
        .world
        .entity_mut(pendulum)
        .insert(PersistentId(pendulum_id));
    let weight = runtime
        .world
        .spawn((
            RigidBodyType::Dynamic,
            ColliderShape3D::Sphere { radius: 0.2 },
            Transform::from_xyz(2.0, 4.0, 0.0),
            PhysicsJoint {
                kind: JointKind::Fixed,
                target: pendulum_id.to_string(),
                local_anchor: Vec3::new(0.0, 1.0, 0.0),
                ..Default::default()
            },
        ))
        .id();
    run(&mut runtime, 120);
    let p = runtime
        .world
        .get::<Transform>(pendulum)
        .unwrap()
        .translation;
    let w = runtime.world.get::<Transform>(weight).unwrap().translation;
    assert!(
        (p.distance(Vec3::new(0.0, 5.0, 0.0)) - 2.0).abs() < 0.1,
        "pendulum stays on its radius: {p}"
    );
    assert!(p.y < 4.9, "and swings down: {p}");
    assert!(
        (p.distance(w) - 1.0).abs() < 0.1,
        "weight stays welded: {p} {w}"
    );
    assert!(runtime.world.get::<JointHandle3D>(weight).is_some());

    // Removing the joint component removes the Rapier joint.
    runtime.world.entity_mut(weight).remove::<PhysicsJoint>();
    run(&mut runtime, 1);
    assert!(runtime.world.get::<JointHandle3D>(weight).is_none());
    assert_eq!(
        runtime
            .world
            .resource::<PhysicsWorld3D>()
            .impulse_joint_set
            .len(),
        1
    );
}

#[test]
fn revolute_motor_opens_a_door() {
    let mut runtime = runtime();
    let door = runtime
        .world
        .spawn((
            RigidBodyType::Dynamic,
            ColliderShape3D::Box {
                half_extents: Vec3::new(0.5, 1.0, 0.05),
            },
            RigidBodySettings {
                gravity_scale: 0.0,
                ..Default::default()
            },
            Transform::from_xyz(0.5, 1.0, 0.0),
            PhysicsJoint {
                kind: JointKind::Revolute,
                local_anchor: Vec3::new(-0.5, 0.0, 0.0),
                target_anchor: Vec3::new(0.0, 1.0, 0.0),
                axis: Vec3::Y,
                motor: JointMotor {
                    enabled: true,
                    target_position: std::f32::consts::FRAC_PI_2,
                    stiffness: 200.0,
                    damping: 20.0,
                    ..Default::default()
                },
                ..Default::default()
            },
        ))
        .id();
    run(&mut runtime, 240);
    let rotation = runtime.world.get::<Transform>(door).unwrap().rotation;
    let (axis, angle) = rotation.to_axis_angle();
    assert!(axis.y.abs() > 0.9, "rotates about Y: {axis}");
    assert!(
        (angle - std::f32::consts::FRAC_PI_2).abs() < 0.2,
        "opened: {angle}"
    );
}

#[test]
fn teleporting_a_dynamic_body_moves_it_without_interpolation_smear() {
    let mut runtime = runtime();
    floor(&mut runtime);
    let crate_entity = runtime
        .world
        .spawn((
            RigidBodyType::Dynamic,
            ColliderShape3D::Box {
                half_extents: Vec3::splat(0.5),
            },
            Transform::from_xyz(0.0, 0.5, 0.0),
        ))
        .id();
    run(&mut runtime, 10);
    runtime
        .world
        .get_mut::<Transform>(crate_entity)
        .unwrap()
        .translation = Vec3::new(10.0, 0.5, 0.0);
    run(&mut runtime, 1);
    let t = runtime
        .world
        .get::<Transform>(crate_entity)
        .unwrap()
        .translation;
    assert!((t.x - 10.0).abs() < 0.05, "teleported: {t}");
    let pose = runtime.world.get::<PhysicsPose>(crate_entity).unwrap();
    assert!((pose.previous.0.x - 10.0).abs() < 0.05);

    // Rotating only keeps the simulated translation.
    runtime
        .world
        .get_mut::<Transform>(crate_entity)
        .unwrap()
        .rotation = Quat::from_rotation_y(1.0);
    run(&mut runtime, 1);
    let t = runtime.world.get::<Transform>(crate_entity).unwrap();
    assert!((t.translation.x - 10.0).abs() < 0.05);
    assert!(t.rotation.angle_between(Quat::from_rotation_y(1.0)) < 0.05);
}

#[test]
fn velocity_impulse_and_force_drive_bodies() {
    let mut runtime = runtime();
    let body = runtime
        .world
        .spawn((
            RigidBodyType::Dynamic,
            ColliderShape3D::Sphere { radius: 0.5 },
            RigidBodySettings {
                gravity_scale: 0.0,
                angular_damping: 0.0,
                ..Default::default()
            },
            Velocity::default(),
            ExternalImpulse::default(),
            Transform::default(),
        ))
        .id();
    run(&mut runtime, 1);
    runtime
        .world
        .get_mut::<ExternalImpulse>(body)
        .unwrap()
        .impulse = Vec3::new(1.0, 0.0, 0.0);
    run(&mut runtime, 2);
    let v = runtime.world.get::<Velocity>(body).unwrap().linear;
    assert!(v.x > 0.1 && v.y.abs() < 1e-3, "impulse applied once: {v}");
    assert_eq!(
        *runtime.world.get::<ExternalImpulse>(body).unwrap(),
        ExternalImpulse::default()
    );

    runtime.world.get_mut::<Velocity>(body).unwrap().linear = Vec3::new(0.0, 0.0, 2.0);
    run(&mut runtime, 1);
    let v = runtime.world.get::<Velocity>(body).unwrap().linear;
    assert!(
        (v.z - 2.0).abs() < 1e-3 && v.x.abs() < 1e-3,
        "set velocity: {v}"
    );

    runtime.world.entity_mut(body).insert(ExternalForce {
        force: Vec3::new(0.0, 10.0, 0.0),
        torque: Vec3::ZERO,
    });
    run(&mut runtime, 30);
    assert!(runtime.world.get::<Velocity>(body).unwrap().linear.y > 0.5);
}

#[test]
fn mesh_colliders_load_from_assets() {
    let mut runtime = runtime();
    let assets = Assets::new();
    let reference = AssetRef::from_path("meshes/ramp.mesh");
    let handle = assets.request::<MeshData>(&reference);
    runtime.world.insert_resource(assets.clone());
    let vertex = |x: f32, y: f32, z: f32| MeshVertex {
        position: [x, y, z],
        normal: [0.0, 1.0, 0.0],
        uv: [0.0, 0.0],
    };
    let ground = runtime
        .world
        .spawn((
            RigidBodyType::Static,
            ColliderShape3D::Mesh {
                mesh: reference,
                convex: false,
            },
            Transform::default(),
        ))
        .id();
    run(&mut runtime, 2);
    assert!(
        runtime.world.get::<ColliderHandle3D>(ground).is_none(),
        "pending"
    );

    let mut mesh = MeshData::new(
        "ramp",
        vec![
            vertex(-5.0, 0.0, -5.0),
            vertex(5.0, 0.0, -5.0),
            vertex(5.0, 0.0, 5.0),
            vertex(-5.0, 0.0, 5.0),
        ],
        vec![0, 2, 1, 0, 3, 2],
    );
    mesh.name = "ramp".into();
    assert!(assets.replace(handle, mesh));
    let ball = runtime
        .world
        .spawn((
            RigidBodyType::Dynamic,
            ColliderShape3D::Sphere { radius: 0.5 },
            Transform::from_xyz(0.0, 2.0, 0.0),
        ))
        .id();
    run(&mut runtime, 90);
    assert!(runtime.world.get::<ColliderHandle3D>(ground).is_some());
    let y = runtime.world.get::<Transform>(ball).unwrap().translation.y;
    assert!((y - 0.5).abs() < 0.1, "ball rests on the mesh: {y}");
}

#[test]
fn overlap_and_shapecast_find_colliders() {
    let mut runtime = runtime();
    let wall = runtime
        .world
        .spawn((
            ColliderShape3D::Box {
                half_extents: Vec3::new(0.5, 2.0, 2.0),
            },
            Transform::from_xyz(5.0, 0.0, 0.0),
        ))
        .id();
    let sensor = runtime
        .world
        .spawn((
            ColliderShape3D::Sphere { radius: 1.0 },
            Sensor,
            Transform::default(),
        ))
        .id();
    run(&mut runtime, 1);
    let world = &runtime.world;
    let queries = PhysicsQueries::new(
        world.resource::<PhysicsWorld3D>(),
        world.resource::<ColliderEntityMap3D>(),
        world.resource::<PhysicsEntityHandles3D>(),
    );
    let shape = QueryShape::Sphere { radius: 0.5 };
    let hit = queries
        .shapecast(
            shape,
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::X,
            20.0,
            &SpatialQueryFilter::default(),
        )
        .expect("sweeps into the wall");
    assert_eq!(hit.entity, wall);
    assert!((hit.distance - 4.0).abs() < 0.05, "{}", hit.distance);
    assert!(hit.normal.x < -0.9);

    let found = queries.overlap(
        shape,
        Vec3::ZERO,
        Quat::IDENTITY,
        &SpatialQueryFilter::default(),
    );
    assert!(found.is_empty(), "sensors excluded by default");
    let found = queries.overlap(
        shape,
        Vec3::ZERO,
        Quat::IDENTITY,
        &SpatialQueryFilter::default().with_sensors(true),
    );
    assert_eq!(found, vec![sensor]);
    let all = queries.raycast_all(
        Vec3::new(-5.0, 0.0, 0.0),
        Vec3::X,
        20.0,
        &SpatialQueryFilter::default().with_sensors(true),
    );
    assert_eq!(
        all.iter().map(|h| h.entity).collect::<Vec<_>>(),
        vec![sensor, wall]
    );
}

#[test]
fn debug_draw_emits_lines_when_enabled() {
    let mut runtime = runtime();
    floor(&mut runtime);
    character(&mut runtime, Vec3::new(0.0, 1.0, 0.0));
    run(&mut runtime, 2);
    assert!(
        runtime.world.resource::<DebugDraw>().lines().is_empty(),
        "disabled by default"
    );
    runtime
        .world
        .resource_mut::<DebugDraw>()
        .set_enabled(DebugCategory::Physics, true);
    run(&mut runtime, 1);
    assert!(runtime.world.resource::<DebugDraw>().lines().len() > 20);
}

#[test]
fn dynamic_concave_mesh_is_rejected_until_fixed() {
    let mut runtime = runtime();
    let entity = runtime
        .world
        .spawn((
            RigidBodyType::Dynamic,
            ColliderShape3D::Trimesh,
            Transform::default(),
        ))
        .id();
    run(&mut runtime, 2);
    assert!(runtime.world.get::<PhysicsRejected>(entity).is_some());
    assert!(runtime.world.get::<RigidBodyHandle3D>(entity).is_none());
    *runtime.world.get_mut::<RigidBodyType>(entity).unwrap() = RigidBodyType::Static;
    run(&mut runtime, 2);
    assert!(runtime.world.get::<PhysicsRejected>(entity).is_none());
    assert!(runtime.world.get::<RigidBodyHandle3D>(entity).is_some());
    let _ = events::<CollisionStarted>(&runtime);
}
