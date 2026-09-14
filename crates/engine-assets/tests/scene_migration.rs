use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use bevy_ecs::world::World;
use engine_assets::{AssetServer, SceneDeserializer, SceneFile};
use engine_core::{register_core_reflection_types, Children, EntityName, PersistentId, Transform};
use engine_reflect::{ComponentRegistry, ReflectMetadataRegistry, ReflectTypeRegistry};

const FIXTURE_V1_MINIMAL: &str = include_str!("fixtures/scene_v1_minimal.ron");

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

    (type_registry, component_registry, metadata_registry)
}

/// Copies the checked-in v1 fixture into a scratch file so each test
/// mutates its own private copy — migration rewrites the file in place and
/// writes a backup next to it.
fn scratch_copy_of_fixture(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be valid")
        .as_nanos();
    let path = std::env::temp_dir().join(format!("{prefix}-{nanos}.scene.ron"));
    fs::write(&path, FIXTURE_V1_MINIMAL).expect("fixture copy should be written");
    path
}

fn backup_path_for(path: &Path) -> PathBuf {
    let mut backup = path.as_os_str().to_owned();
    backup.push(".v1.bak");
    PathBuf::from(backup)
}

#[test]
fn loading_a_v1_scene_migrates_it_and_assigns_stable_ids() {
    let (type_registry, component_registry, _) = build_registries();
    let scene_path = scratch_copy_of_fixture("starman-migration");

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
        .expect("v1 fixture should migrate and load");

    assert_eq!(roots.len(), 1);
    let root = roots[0];
    assert_eq!(
        world
            .get::<EntityName>(root)
            .expect("root should keep its name")
            .0,
        "Root"
    );
    assert_eq!(
        world
            .get::<Transform>(root)
            .expect("root should keep its transform")
            .translation,
        [1.0, 2.0, 3.0].into()
    );

    let child = world
        .get::<Children>(root)
        .expect("root should keep its child")
        .0[0];

    let root_id = world
        .get::<PersistentId>(root)
        .expect("migrated root should carry a persistent id")
        .0;
    let child_id = world
        .get::<PersistentId>(child)
        .expect("migrated child should carry a persistent id")
        .0;
    assert_ne!(root_id, child_id, "every entity gets its own id");

    // The file on disk was rewritten to the current version, and the
    // original content was preserved as a backup.
    let rewritten = fs::read_to_string(&scene_path).expect("migrated file should be readable");
    assert!(rewritten.contains(&format!("version: {}", SceneFile::CURRENT_VERSION)));

    let backup_path = backup_path_for(&scene_path);
    let backup = fs::read_to_string(&backup_path).expect("backup file should exist");
    assert_eq!(backup, FIXTURE_V1_MINIMAL);

    let _ = fs::remove_file(&scene_path);
    let _ = fs::remove_file(&backup_path);
}

#[test]
fn reloading_a_migrated_scene_yields_the_same_ids_every_time() {
    let (type_registry, component_registry, _) = build_registries();
    let scene_path = scratch_copy_of_fixture("starman-migration-golden");

    // First load performs the migration and writes the ids to disk.
    let ids_from_first_load = {
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
            .expect("first load should migrate the fixture");
        collect_persistent_ids(&world, &roots)
    };

    // Every subsequent load reads the already-migrated (v2) file, so the
    // exact same ids must come back every time: this is the golden-file
    // property migration exists to establish.
    for _ in 0..2 {
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
            .expect("reload of a v2 scene should not migrate again");
        let ids = collect_persistent_ids(&world, &roots);
        assert_eq!(ids, ids_from_first_load);
    }

    let _ = fs::remove_file(&scene_path);
    let _ = fs::remove_file(backup_path_for(&scene_path));
}

fn collect_persistent_ids(
    world: &World,
    roots: &[bevy_ecs::entity::Entity],
) -> Vec<engine_core::EntityId> {
    fn visit(
        world: &World,
        entity: bevy_ecs::entity::Entity,
        out: &mut Vec<engine_core::EntityId>,
    ) {
        out.push(
            world
                .get::<PersistentId>(entity)
                .expect("every migrated entity should carry a persistent id")
                .0,
        );
        if let Some(children) = world.get::<Children>(entity) {
            for child in children.0.iter().copied() {
                visit(world, child, out);
            }
        }
    }

    let mut ids = Vec::new();
    for root in roots.iter().copied() {
        visit(world, root, &mut ids);
    }

    let unique: HashSet<_> = ids.iter().collect();
    assert_eq!(unique.len(), ids.len(), "ids must be unique per entity");

    ids
}
