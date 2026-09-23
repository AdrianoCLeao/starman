use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use bevy_ecs::world::World;
use engine_assets::{
    AssetServer, SceneDeserializer, SceneEntityData, SceneFile, SceneSerializer, SceneValue,
};
use engine_core::{
    register_core_reflection_types, Children, EntityId, EntityName, Parent, PersistentId, Transform,
};
use engine_physics::{
    register_physics_reflection_types, ColliderShape3D, PhysicsMaterial, RigidBodyType,
};
use engine_reflect::{ComponentRegistry, ReflectMetadataRegistry, ReflectTypeRegistry};

fn build_registries() -> (
    ReflectTypeRegistry,
    ComponentRegistry,
    ReflectMetadataRegistry,
) {
    let mut type_registry = ReflectTypeRegistry::default();
    let mut component_registry = ComponentRegistry::default();
    let mut metadata_registry = ReflectMetadataRegistry::default();

    register_core_reflection_types(
        &mut type_registry,
        &mut component_registry,
        &mut metadata_registry,
    );
    register_physics_reflection_types(
        &mut type_registry,
        &mut component_registry,
        &mut metadata_registry,
    );

    (type_registry, component_registry, metadata_registry)
}

fn unique_temp_path(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be valid")
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{nanos}.scene.ron"))
}

#[test]
fn scene_roundtrip_preserves_hierarchy_and_physics_components() {
    let (type_registry, component_registry, metadata_registry) = build_registries();

    let mut world = World::new();
    let root = world
        .spawn((
            EntityName::new("Root"),
            Transform::from_xyz(1.0, 2.0, 3.0),
            RigidBodyType::Static,
            ColliderShape3D::default(),
            PhysicsMaterial {
                restitution: 0.2,
                friction: 0.8,
                density: 3.0,
            },
            Children::default(),
        ))
        .id();

    let child = world
        .spawn((
            EntityName::new("Child"),
            Transform::from_xyz(0.0, 5.0, 0.0),
            Parent(root),
        ))
        .id();

    world.entity_mut(root).insert(Children(vec![child]));

    let serializer = SceneSerializer::new(&world, &component_registry, &type_registry)
        .with_metadata_registry(&metadata_registry);
    let scene = serializer
        .serialize_world("RoundTrip")
        .expect("scene should serialize");

    assert_eq!(scene.entities.len(), 1);
    assert_eq!(scene.entities[0].name.as_deref(), Some("Root"));
    assert_eq!(scene.entities[0].children.len(), 1);

    let mut loaded_world = World::new();
    let mut asset_server = AssetServer::new("assets");
    let mut deserializer = SceneDeserializer::new(
        &mut loaded_world,
        &component_registry,
        &type_registry,
        &mut asset_server,
    );

    let roots = deserializer
        .load_scene(&scene)
        .expect("scene should deserialize");

    assert_eq!(roots.len(), 1);
    let loaded_root = roots[0];

    let name = loaded_world
        .get::<EntityName>(loaded_root)
        .expect("root should have name");
    assert_eq!(name.0, "Root");

    let body_type = loaded_world
        .get::<RigidBodyType>(loaded_root)
        .expect("root should have rigid body type");
    assert_eq!(*body_type, RigidBodyType::Static);

    let children = loaded_world
        .get::<Children>(loaded_root)
        .expect("root should have children");
    assert_eq!(children.0.len(), 1);

    let loaded_child = children.0[0];
    let parent = loaded_world
        .get::<Parent>(loaded_child)
        .expect("child should have parent");
    assert_eq!(parent.0, loaded_root);
}

#[test]
fn scene_deserializer_skips_unknown_components() {
    let (type_registry, component_registry, _) = build_registries();

    let mut scene = SceneFile {
        version: SceneFile::CURRENT_VERSION,
        name: "UnknownComponent".to_owned(),
        entities: vec![SceneEntityData {
            id: EntityId::new_v4(),
            name: Some("OnlyEntity".to_owned()),
            components: HashMap::new(),
            children: Vec::new(),
            instance: None,
        }],
    };

    scene.entities[0].components.insert(
        "UnknownComponentType".to_owned(),
        SceneValue::String("ignored".to_owned()),
    );

    let mut world = World::new();
    let mut asset_server = AssetServer::new("assets");
    let mut deserializer = SceneDeserializer::new(
        &mut world,
        &component_registry,
        &type_registry,
        &mut asset_server,
    );

    let roots = deserializer
        .load_scene(&scene)
        .expect("unknown component should not fail scene load");

    assert_eq!(roots.len(), 1);
    let entity_name = world
        .get::<EntityName>(roots[0])
        .expect("entity name should be present");
    assert_eq!(entity_name.0, "OnlyEntity");
}

#[test]
fn scene_deserializer_rejects_version_mismatch() {
    let (type_registry, component_registry, _) = build_registries();

    let scene = SceneFile {
        version: 999,
        name: "Mismatch".to_owned(),
        entities: Vec::new(),
    };

    let mut world = World::new();
    let mut asset_server = AssetServer::new("assets");
    let mut deserializer = SceneDeserializer::new(
        &mut world,
        &component_registry,
        &type_registry,
        &mut asset_server,
    );

    let error = deserializer
        .load_scene(&scene)
        .expect_err("version mismatch should fail");
    assert!(error.to_string().contains("unsupported scene version"));
}

#[test]
fn scene_load_file_accepts_handwritten_ron() {
    let (type_registry, component_registry, _) = build_registries();

    let scene_path = unique_temp_path("starman-handwritten");
    let source = r#"(
    version: 2,
    name: "HandwrittenScene",
    entities: [
        (
            id: "5c0a9c2e-4b8a-4c2a-9c9a-9c9a9c9a9c9a",
            name: Some("ManualRoot"),
            components: {
                "Transform": {
                    "translation": {"x": 1.0, "y": 2.0, "z": 3.0},
                    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                    "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
                },
            },
            children: [],
        ),
    ],
)
"#;
    fs::write(&scene_path, source).expect("handwritten scene should be written");

    let mut world = World::new();
    let mut asset_server = AssetServer::new("assets");
    let mut deserializer = SceneDeserializer::new(
        &mut world,
        &component_registry,
        &type_registry,
        &mut asset_server,
    );

    let roots = deserializer
        .load_file(&scene_path)
        .expect("handwritten scene file should load");

    assert_eq!(roots.len(), 1);
    let root_name = world
        .get::<EntityName>(roots[0])
        .expect("root should carry entity name");
    assert_eq!(root_name.0, "ManualRoot");

    let transform = world
        .get::<Transform>(roots[0])
        .expect("root should include transform");
    assert_eq!(transform.translation, [1.0, 2.0, 3.0].into());

    let persistent_id = world
        .get::<PersistentId>(roots[0])
        .expect("root should carry the persisted entity id");
    assert_eq!(
        persistent_id.0.to_string(),
        "5c0a9c2e-4b8a-4c2a-9c9a-9c9a9c9a9c9a"
    );

    let _ = fs::remove_file(scene_path);
}

#[test]
fn scene_roundtrip_preserves_three_level_hierarchy() {
    let (type_registry, component_registry, metadata_registry) = build_registries();

    let mut world = World::new();
    let root = world
        .spawn((
            EntityName::new("Root"),
            Transform::from_xyz(0.0, 0.0, 0.0),
            Children::default(),
        ))
        .id();
    let child = world
        .spawn((
            EntityName::new("Child"),
            Transform::from_xyz(1.0, 0.0, 0.0),
            Parent(root),
            Children::default(),
        ))
        .id();
    let grandchild = world
        .spawn((
            EntityName::new("Grandchild"),
            Transform::from_xyz(2.0, 0.0, 0.0),
            Parent(child),
            Children::default(),
        ))
        .id();
    let great_grandchild = world
        .spawn((
            EntityName::new("GreatGrandchild"),
            Transform::from_xyz(3.0, 0.0, 0.0),
            Parent(grandchild),
        ))
        .id();

    world.entity_mut(root).insert(Children(vec![child]));
    world.entity_mut(child).insert(Children(vec![grandchild]));
    world
        .entity_mut(grandchild)
        .insert(Children(vec![great_grandchild]));

    let serializer = SceneSerializer::new(&world, &component_registry, &type_registry)
        .with_metadata_registry(&metadata_registry);
    let scene = serializer
        .serialize_world("DeepHierarchy")
        .expect("scene should serialize");

    assert_eq!(scene.entities.len(), 1);
    assert_eq!(scene.entities[0].children.len(), 1);
    assert_eq!(scene.entities[0].children[0].children.len(), 1);
    assert_eq!(scene.entities[0].children[0].children[0].children.len(), 1);

    let mut loaded_world = World::new();
    let mut asset_server = AssetServer::new("assets");
    let mut deserializer = SceneDeserializer::new(
        &mut loaded_world,
        &component_registry,
        &type_registry,
        &mut asset_server,
    );
    let roots = deserializer
        .load_scene(&scene)
        .expect("scene should deserialize");

    let loaded_root = roots[0];
    let loaded_child = loaded_world
        .get::<Children>(loaded_root)
        .expect("root should have one child")
        .0[0];
    let loaded_grandchild = loaded_world
        .get::<Children>(loaded_child)
        .expect("child should have one child")
        .0[0];
    let loaded_great_grandchild = loaded_world
        .get::<Children>(loaded_grandchild)
        .expect("grandchild should have one child")
        .0[0];

    assert_eq!(
        loaded_world
            .get::<EntityName>(loaded_great_grandchild)
            .expect("great-grandchild should be named")
            .0,
        "GreatGrandchild"
    );
}

#[test]
fn scene_deserializer_handles_fifty_entities() {
    let (type_registry, component_registry, metadata_registry) = build_registries();

    let mut world = World::new();
    for i in 0..50 {
        world.spawn((
            EntityName::new(format!("Entity-{i}")),
            Transform::from_xyz(i as f32, 0.0, 0.0),
        ));
    }

    let serializer = SceneSerializer::new(&world, &component_registry, &type_registry)
        .with_metadata_registry(&metadata_registry);
    let scene = serializer
        .serialize_world("FiftyEntities")
        .expect("scene should serialize");
    assert_eq!(scene.entities.len(), 50);

    let mut loaded_world = World::new();
    let mut asset_server = AssetServer::new("assets");
    let mut deserializer = SceneDeserializer::new(
        &mut loaded_world,
        &component_registry,
        &type_registry,
        &mut asset_server,
    );

    let roots = deserializer
        .load_scene(&scene)
        .expect("fifty-entity scene should deserialize");

    assert_eq!(roots.len(), 50);
}

#[test]
fn scene_save_file_output_is_deterministic() {
    let (type_registry, component_registry, metadata_registry) = build_registries();

    let mut world = World::new();
    world.spawn((
        EntityName::new("Deterministic-A"),
        Transform::from_xyz(1.0, 2.0, 3.0),
        RigidBodyType::Static,
    ));
    world.spawn((
        EntityName::new("Deterministic-B"),
        Transform::from_xyz(4.0, 5.0, 6.0),
        ColliderShape3D::default(),
        PhysicsMaterial::default(),
    ));

    let serializer = SceneSerializer::new(&world, &component_registry, &type_registry)
        .with_metadata_registry(&metadata_registry);

    let first_path = unique_temp_path("starman-deterministic-1");
    let second_path = unique_temp_path("starman-deterministic-2");
    serializer
        .save_file(&first_path, "DeterministicScene")
        .expect("first scene write should succeed");
    serializer
        .save_file(&second_path, "DeterministicScene")
        .expect("second scene write should succeed");

    let first_source =
        fs::read_to_string(&first_path).expect("first scene file should be readable");
    let second_source =
        fs::read_to_string(&second_path).expect("second scene file should be readable");

    assert_eq!(first_source, second_source);

    let _ = fs::remove_file(first_path);
    let _ = fs::remove_file(second_path);
}

#[test]
#[ignore = "performance evidence test for EP-08 target"]
fn scene_serializer_serializes_ten_entities_under_fifty_ms() {
    let (type_registry, component_registry, metadata_registry) = build_registries();

    let mut world = World::new();
    for i in 0..10 {
        world.spawn((
            EntityName::new(format!("Perf-{i}")),
            Transform::from_xyz(i as f32, i as f32 * 0.5, 0.0),
            RigidBodyType::Static,
        ));
    }

    let serializer = SceneSerializer::new(&world, &component_registry, &type_registry)
        .with_metadata_registry(&metadata_registry);

    let started = Instant::now();
    let scene = serializer
        .serialize_world("PerfScene")
        .expect("scene should serialize");
    assert_eq!(scene.entities.len(), 10);

    let elapsed_ms = started.elapsed().as_millis();
    assert!(
        elapsed_ms < 50,
        "expected serialization under 50ms, got {elapsed_ms}ms"
    );
}
