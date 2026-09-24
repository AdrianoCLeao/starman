//! Skinned character through the real asset server and runtime: glTF
//! skeleton/clip import, `.anim.ron` overlays, graph transitions, events,
//! root motion, property tracks, bone attachments and skin palettes.

use std::path::PathBuf;

use bevy_ecs::prelude::*;
use content_gen::glb::{self, *};
use engine_animation::*;
use engine_assets::{AssetRef, AssetServer, AssetsPlugin};
use engine_core::{EntityName, GameRuntime, Parent, SkinPalette, Transform};
use engine_math::{Quat, Vec3};

const DT: f32 = 1.0 / 30.0;

fn character_glb() -> Vec<u8> {
    let mut armature = Node::new("Armature");
    armature.children = vec![1, 3];
    let mut root = Node::new("root");
    root.children = vec![2];
    let arm = Node::new("arm").at([0.0, 1.0, 0.0]);
    let mut body = Node::new("Body");
    body.mesh = Some(0);
    body.skin = Some(0);
    let primitive = MeshPrimitive {
        positions: vec![[0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.1, 1.0, 0.0]],
        normals: vec![[0.0, 0.0, 1.0]; 3],
        uvs: vec![[0.0, 0.0]; 3],
        indices: vec![0, 1, 2],
        joints: vec![[1, 0, 0, 0]; 3],
        weights: vec![[1.0, 0.0, 0.0, 0.0]; 3],
        material: None,
    };
    let rotation = |angle: f32| {
        let q = Quat::from_rotation_z(angle);
        [q.x, q.y, q.z, q.w]
    };
    let document = Document {
        nodes: vec![armature, root, arm, body],
        meshes: vec![Mesh {
            name: "Body".into(),
            primitives: vec![primitive],
        }],
        materials: vec![],
        skins: vec![Skin {
            name: "rig".into(),
            joints: vec![1, 2],
            inverse_bind_matrices: vec![
                translation_matrix([0.0; 3]),
                translation_matrix([0.0, -1.0, 0.0]),
            ],
            skeleton: Some(1),
        }],
        animations: vec![
            Animation {
                name: "wave".into(),
                channels: vec![Channel {
                    node: 2,
                    path: ChannelPath::Rotation,
                    interpolation: glb::Interpolation::Linear,
                    times: vec![0.0, 1.0],
                    values: [rotation(0.0), rotation(1.0)].concat(),
                }],
            },
            Animation {
                name: "walk".into(),
                channels: vec![Channel {
                    node: 1,
                    path: ChannelPath::Translation,
                    interpolation: glb::Interpolation::Linear,
                    times: vec![0.0, 1.0],
                    values: vec![0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
                }],
            },
        ],
        roots: vec![0],
    };
    write_glb(&document)
}

const WAVE: &str = r#"(
    source: Some((path: "char.glb#anim:0")),
    looping: false,
    events: [(time: 0.5, name: "swing", payload: "(power: 3)")],
    property_tracks: [
        (target: "Lamp", component: "Transform", field: "translation.y",
         curve: Float((times: [0.0, 1.0], values: [0.0, 4.0]))),
    ],
)"#;

const WALK: &str = r#"(
    source: Some((path: "char.glb#anim:1")),
    root_motion: Some((bone: "root")),
)"#;

const GRAPH: &str = r#"(
    parameters: [(name: "wave", default: Trigger(false))],
    layers: [(
        name: "base",
        state_machine: (
            entry: "walk",
            states: [
                (name: "walk", motion: Clip((path: "walk.anim.ron"))),
                (name: "wave", motion: Clip((path: "wave.anim.ron"))),
            ],
            transitions: [
                (from: "walk", to: "wave", conditions: [If("wave")]),
                (from: "wave", to: "walk", exit_time: Some(1.0), duration: 0.1),
            ],
        ),
    )],
)"#;

fn project() -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!("starman-anim-{nanos}"));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("char.glb"), character_glb()).unwrap();
    std::fs::write(root.join("wave.anim.ron"), WAVE).unwrap();
    std::fs::write(root.join("walk.anim.ron"), WALK).unwrap();
    std::fs::write(root.join("char.animgraph.ron"), GRAPH).unwrap();
    root
}

struct Harness {
    server: AssetServer,
    runtime: GameRuntime,
    root: PathBuf,
}

impl Harness {
    fn new() -> Self {
        let root = project();
        let server = AssetServer::new(root.to_string_lossy().to_string());
        let mut runtime = GameRuntime::new();
        runtime.insert_resource(server.assets().clone());
        runtime.add_plugin(AssetsPlugin);
        runtime.add_plugin(AnimationPlugin);
        Self {
            server,
            runtime,
            root,
        }
    }

    fn step(&mut self, frames: usize) {
        for _ in 0..frames {
            self.server.update_blocking();
            self.runtime.step(DT);
        }
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

#[derive(Resource, Default)]
struct Fired(Vec<AnimationEvent>);

fn record(mut fired: ResMut<Fired>, mut events: EventReader<AnimationEvent>) {
    fired.0.extend(events.read().cloned());
}

#[test]
fn skinned_character_plays_graph_with_events_root_motion_and_properties() {
    let mut harness = Harness::new();
    harness.runtime.init_resource::<Fired>();
    harness.runtime.add_systems(
        engine_core::ScheduleKind::Update,
        record.after(engine_core::UpdateSet::AnimationApply),
    );
    let world = &mut harness.runtime.world;
    let character = world
        .spawn((
            Transform::default(),
            SkinnedMesh {
                skeleton: AssetRef::from_path("char.glb#skin:0"),
                ..Default::default()
            },
            Animator {
                graph: AssetRef::from_path("char.animgraph.ron"),
                apply_root_motion: true,
                ..Default::default()
            },
        ))
        .id();
    let lamp = world
        .spawn((
            EntityName::new("Lamp"),
            Transform::default(),
            Parent(character),
        ))
        .id();
    let hand = world
        .spawn((
            Transform::default(),
            Parent(character),
            BoneAttachment {
                bone: "arm".into(),
                ..Default::default()
            },
        ))
        .id();
    engine_core::set_parent(world, lamp, Some(character));
    engine_core::set_parent(world, hand, Some(character));

    harness.step(31);
    let world = &harness.runtime.world;
    let runtime = world.get::<AnimatorRuntime>(character).unwrap();
    assert!(runtime.is_ready());
    assert_eq!(runtime.current_state(0).as_deref(), Some("walk"));
    let z = world.get::<Transform>(character).unwrap().translation.z;
    assert!((z - 2.0).abs() < 0.15, "a second of 2 m/s root motion: {z}");
    let instance = world.get::<SkeletonInstance>(character).unwrap();
    assert!(
        instance.pose.locals[0].translation.length() < 1e-4,
        "root bone animates in place"
    );
    let palette = world.get::<SkinPalette>(character).unwrap();
    assert_eq!(palette.joint_matrices.len(), 2);
    assert!(palette.bounds.is_some());

    harness
        .runtime
        .world
        .get_mut::<AnimatorRuntime>(character)
        .unwrap()
        .set_trigger("wave");
    harness.step(18);
    let world = &harness.runtime.world;
    assert_eq!(
        world
            .get::<AnimatorRuntime>(character)
            .unwrap()
            .current_state(0)
            .as_deref(),
        Some("wave")
    );
    let fired = &world.resource::<Fired>().0;
    assert_eq!(fired.len(), 1, "{fired:?}");
    assert_eq!(fired[0].name, "swing");
    assert_eq!(fired[0].payload, "(power: 3)");
    assert_eq!(fired[0].entity, character);

    // ~0.57s into the wave: arm rotated ~0.57 rad, lamp raised ~2.3.
    let instance = world.get::<SkeletonInstance>(character).unwrap();
    let angle = instance.pose.locals[1]
        .rotation
        .angle_between(Quat::IDENTITY);
    assert!((angle - 0.57).abs() < 0.08, "{angle}");
    let lamp_y = world.get::<Transform>(lamp).unwrap().translation.y;
    assert!((lamp_y - 2.3).abs() < 0.3, "{lamp_y}");
    let hand_transform = world.get::<Transform>(hand).unwrap();
    assert!((hand_transform.translation - Vec3::Y).length() < 1e-3);
    assert!(hand_transform.rotation.angle_between(Quat::IDENTITY) > 0.4);
    let palette = world.get::<SkinPalette>(character).unwrap();
    let tip = palette.joint_matrices[1].transform_point3(Vec3::new(0.0, 2.0, 0.0));
    assert!(tip.x < -0.4, "skinned vertex follows the arm: {tip}");

    // The wave ends and exit time returns to walking.
    harness.step(20);
    let world = &harness.runtime.world;
    assert_eq!(
        world
            .get::<AnimatorRuntime>(character)
            .unwrap()
            .current_state(0)
            .as_deref(),
        Some("walk")
    );
}

#[test]
fn animation_player_drives_properties_without_a_skeleton() {
    let mut harness = Harness::new();
    let world = &mut harness.runtime.world;
    let door = world
        .spawn((Transform::default(), EntityName::new("Door")))
        .id();
    let lamp = world
        .spawn((EntityName::new("Lamp"), Transform::default()))
        .id();
    engine_core::set_parent(world, lamp, Some(door));
    world.entity_mut(door).insert(AnimationPlayer {
        clip: AssetRef::from_path("wave.anim.ron"),
        ..Default::default()
    });
    harness.step(40);
    let world = &harness.runtime.world;
    let player = world.get::<AnimationPlayer>(door).unwrap();
    assert!(!player.playing, "non-looping clip stops at the end");
    assert!((player.time - 1.0).abs() < 1e-4);
    assert!((world.get::<Transform>(lamp).unwrap().translation.y - 4.0).abs() < 1e-3);
}
