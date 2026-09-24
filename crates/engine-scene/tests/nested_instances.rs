//! Integration tests for nested scene expansion, overrides, and diffs.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use bevy_ecs::world::World;
use engine_assets::{
    AssetServer, OverrideEntry, SceneDeserializer, SceneEntityData, SceneFile, SceneInstanceData,
    SceneValue,
};
use engine_core::{register_core_reflection_types, EntityId, EntityName, SourceAssetId, Transform};
use engine_reflect::{ComponentRegistry, ReflectMetadataRegistry, ReflectTypeRegistry};
use engine_scene::{
    apply_overrides_to_source, expand_all_instances, promote_local_entity, revert_overrides,
    set_override, CycleDetector, StructuralDiff,
};
use std::collections::HashMap as StdHashMap;

fn registries() -> (
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
    (type_registry, component_registry, metadata_registry)
}

fn scratch_dir(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("{prefix}-{nanos}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_scene(path: &std::path::Path, scene: &SceneFile) {
    scene.write_to(path).expect("write scene");
}

#[test]
fn expands_nested_instance_and_applies_override() {
    let dir = scratch_dir("starman-nested");
    let crate_id = EntityId::new_v4();
    let prop_scene_id = SourceAssetId::new_v4();

    let mut transform_map = ron::Map::new();
    transform_map.insert(
        SceneValue::String("translation".into()),
        SceneValue::Map({
            let mut m = ron::Map::new();
            m.insert(
                SceneValue::String("x".into()),
                SceneValue::Number(0.0.into()),
            );
            m.insert(
                SceneValue::String("y".into()),
                SceneValue::Number(0.0.into()),
            );
            m.insert(
                SceneValue::String("z".into()),
                SceneValue::Number(0.0.into()),
            );
            m
        }),
    );
    transform_map.insert(
        SceneValue::String("rotation".into()),
        SceneValue::Map({
            let mut m = ron::Map::new();
            m.insert(
                SceneValue::String("x".into()),
                SceneValue::Number(0.0.into()),
            );
            m.insert(
                SceneValue::String("y".into()),
                SceneValue::Number(0.0.into()),
            );
            m.insert(
                SceneValue::String("z".into()),
                SceneValue::Number(0.0.into()),
            );
            m.insert(
                SceneValue::String("w".into()),
                SceneValue::Number(1.0.into()),
            );
            m
        }),
    );
    transform_map.insert(
        SceneValue::String("scale".into()),
        SceneValue::Map({
            let mut m = ron::Map::new();
            m.insert(
                SceneValue::String("x".into()),
                SceneValue::Number(1.0.into()),
            );
            m.insert(
                SceneValue::String("y".into()),
                SceneValue::Number(1.0.into()),
            );
            m.insert(
                SceneValue::String("z".into()),
                SceneValue::Number(1.0.into()),
            );
            m
        }),
    );

    let prop = SceneFile {
        version: SceneFile::CURRENT_VERSION,
        name: "crate".into(),
        entities: vec![SceneEntityData {
            id: crate_id,
            name: Some("Crate".into()),
            components: {
                let mut c = HashMap::new();
                c.insert("Transform".into(), SceneValue::Map(transform_map));
                c
            },
            children: vec![],
            instance: None,
        }],
    };
    let prop_path = dir.join("crate.scene.ron");
    write_scene(&prop_path, &prop);

    let mut instance = SceneInstanceData::new(prop_scene_id);
    instance.scene_path = Some("crate.scene.ron".into());
    set_override(
        &mut instance,
        crate_id,
        "Transform",
        "translation",
        SceneValue::Map({
            let mut m = ron::Map::new();
            m.insert(
                SceneValue::String("x".into()),
                SceneValue::Number(5.0.into()),
            );
            m.insert(
                SceneValue::String("y".into()),
                SceneValue::Number(0.0.into()),
            );
            m.insert(
                SceneValue::String("z".into()),
                SceneValue::Number(0.0.into()),
            );
            m
        }),
    );

    let level = SceneFile {
        version: SceneFile::CURRENT_VERSION,
        name: "level".into(),
        entities: vec![SceneEntityData {
            id: EntityId::new_v4(),
            name: Some("CrateInstance".into()),
            components: HashMap::new(),
            children: vec![],
            instance: Some(instance),
        }],
    };
    let level_path = dir.join("level.scene.ron");
    write_scene(&level_path, &level);

    let (type_registry, component_registry, _) = registries();
    let mut world = World::new();
    let mut asset_server = AssetServer::new(dir.to_string_lossy().to_string());
    let mut deserializer = SceneDeserializer::new(
        &mut world,
        &component_registry,
        &type_registry,
        &mut asset_server,
    );
    let roots = deserializer.load_file(&level_path).expect("load level");
    assert_eq!(roots.len(), 1);

    let expanded = expand_all_instances(
        &mut world,
        &component_registry,
        &type_registry,
        &mut asset_server,
        None,
    )
    .expect("expand");
    assert_eq!(expanded, 1);

    let crate_entities: Vec<_> = world
        .iter_entities()
        .filter(|e| e.get::<EntityName>().is_some_and(|n| n.0 == "Crate"))
        .map(|e| e.id())
        .collect();
    assert_eq!(crate_entities.len(), 1);
    let transform = world
        .get::<Transform>(crate_entities[0])
        .expect("inherited crate has transform");
    assert!((transform.translation.x - 5.0).abs() < 0.01);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn structural_diff_is_deterministic() {
    let target = EntityId::new_v4();
    let template = SceneFile {
        version: SceneFile::CURRENT_VERSION,
        name: "t".into(),
        entities: vec![SceneEntityData {
            id: target,
            name: Some("A".into()),
            components: HashMap::new(),
            children: vec![],
            instance: None,
        }],
    };
    let mut instance = SceneInstanceData::new(SourceAssetId::new_v4());
    instance.overrides = vec![
        OverrideEntry {
            target,
            component: "Transform".into(),
            field_path: "scale".into(),
            value: SceneValue::Unit,
        },
        OverrideEntry {
            target,
            component: "Transform".into(),
            field_path: "translation".into(),
            value: SceneValue::Unit,
        },
    ];
    let a = StructuralDiff::between(&template, &instance).to_string();
    let b = StructuralDiff::between(&template, &instance).to_string();
    assert_eq!(a, b);
    assert!(a.contains("translation"));
}

#[test]
fn apply_revert_promote_round_trip() {
    let template_entity = EntityId::new_v4();
    let local_id = EntityId::new_v4();
    let mut source = SceneFile {
        version: SceneFile::CURRENT_VERSION,
        name: "src".into(),
        entities: vec![SceneEntityData {
            id: template_entity,
            name: Some("Base".into()),
            components: HashMap::new(),
            children: vec![],
            instance: None,
        }],
    };
    let mut instance = SceneInstanceData::new(SourceAssetId::new_v4());
    set_override(
        &mut instance,
        template_entity,
        "Visible",
        "",
        SceneValue::Map(ron::Map::new()),
    );
    assert_eq!(instance.overrides.len(), 1);

    let overrides = instance.overrides.clone();
    apply_overrides_to_source(&mut source, &mut instance, &overrides).expect("apply");
    assert!(instance.overrides.is_empty());
    assert!(source.entities[0].components.contains_key("Visible"));

    set_override(
        &mut instance,
        template_entity,
        "Visible",
        "",
        SceneValue::Unit,
    );
    revert_overrides(
        &mut instance,
        Some(template_entity),
        Some("Visible"),
        Some(""),
    );
    assert!(instance.overrides.is_empty());

    instance.added.push(engine_assets::LocalAddedEntity {
        parent: engine_assets::LocalParent::InstanceRoot,
        entity: SceneEntityData {
            id: local_id,
            name: Some("Local".into()),
            components: HashMap::new(),
            children: vec![],
            instance: None,
        },
    });
    promote_local_entity(&mut source, &mut instance, local_id, None).expect("promote");
    assert!(instance.added.is_empty());
    assert_eq!(source.entities.len(), 2);
}

#[test]
fn cycle_detector_rejects_loops() {
    let a = SourceAssetId::new_v4();
    let b = SourceAssetId::new_v4();
    let mut edges = StdHashMap::new();
    edges.insert(a, vec![b]);
    edges.insert(b, vec![a]);
    assert!(CycleDetector::check(&edges).is_err());
}
