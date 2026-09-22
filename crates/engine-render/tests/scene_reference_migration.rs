//! Proves the M1 gate ("mover ou renomear um asset não quebra referências")
//! at the full scene+asset level: a scene authored in the old, path-based
//! reference format is vulnerable to a rename until it is saved at least
//! once through an `AssetServer` with a database attached — after that, it
//! survives the very same rename. This is the "migration on save" design
//! documented in `crates/engine-render/src/scene_adapter.rs` and
//! `docs/asset-pipeline.md`.

use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use bevy_ecs::world::World;
use engine_assets::{AssetDatabase, AssetServer, SceneDeserializer, SceneSerializer};
use engine_core::{register_core_reflection_types, EntityName};
use engine_reflect::{ComponentRegistry, ReflectMetadataRegistry, ReflectTypeRegistry};
use engine_render::{MeshRenderable3d, RenderSceneAdapter};

fn scratch_dir(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be valid")
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{nanos}"))
}

fn reference_assets_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("examples")
        .join("reference-project")
        .join("assets")
}

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

/// Sets up a scratch project directory with a copy of the reference
/// texture/mesh/material (the texture under the name `rock.png`, so it can
/// be renamed later) and a hand-written scene file in legacy path format.
fn scaffold_legacy_project(root: &std::path::Path) -> PathBuf {
    let assets = root.join("assets");
    let reference = reference_assets_root();
    for (source_relative, dest_relative) in [
        ("textures/placeholder.png", "textures/rock.png"),
        ("meshes/cube.glb", "meshes/cube.glb"),
        ("materials/default.ron", "materials/default.ron"),
    ] {
        let dest = assets.join(dest_relative);
        fs::create_dir_all(dest.parent().unwrap()).expect("dest dir should be created");
        fs::copy(reference.join(source_relative), &dest).expect("fixture should copy");
    }

    let scene_path = assets.join("scenes/level.scene.ron");
    fs::create_dir_all(scene_path.parent().unwrap()).expect("scenes dir should be created");
    fs::write(
        &scene_path,
        r#"(
    version: 2,
    name: "GateTest",
    entities: [
        (
            id: "5c0a9c2e-4b8a-4c2a-9c9a-9c9a9c9a9c9a",
            name: Some("Cube"),
            components: {
                "MeshRenderer": {
                    "mesh": "meshes/cube.glb",
                    "texture": "textures/rock.png",
                    "material": "materials/default.ron",
                },
            },
            children: [],
        ),
    ],
)
"#,
    )
    .expect("legacy scene should be written");

    scene_path
}

fn load_scene(
    assets_root: &std::path::Path,
    scene_path: &std::path::Path,
    database: AssetDatabase,
) -> engine_core::Result<(World, AssetServer)> {
    let (type_registry, component_registry, _) = build_registries();
    let mut world = World::new();
    let mut asset_server = AssetServer::new(assets_root.to_string_lossy().to_string());
    asset_server.attach_database(database);

    let adapter = RenderSceneAdapter;
    let mut deserializer = SceneDeserializer::new(
        &mut world,
        &component_registry,
        &type_registry,
        &mut asset_server,
    )
    .with_external_components(&adapter);
    deserializer.load_file(scene_path)?;

    Ok((world, asset_server))
}

#[test]
fn legacy_scene_breaks_on_rename_until_resaved_then_survives_it() {
    let root = scratch_dir("starman-gate-scene-reference-migration");
    let scene_path = scaffold_legacy_project(&root);
    let assets_root = root.join("assets");
    let cache_root = root.join(".starman/cache/imported");

    // --- Session 1: load the legacy (path-based) scene. Succeeds, and the
    // asset server learns the stable ids for everything it referenced.
    let database = AssetDatabase::open(&assets_root, &cache_root).expect("database should open");
    let (world_session_1, mut asset_server_session_1) =
        load_scene(&assets_root, &scene_path, database)
            .expect("legacy scene should load the first time");

    // Rename the texture the scene references, through the database (the
    // only way a rename is guaranteed to preserve the asset's id).
    asset_server_session_1
        .database_mut()
        .expect("database should be attached")
        .rename("textures/rock.png", "textures/boulder.png")
        .expect("rename should succeed");

    // --- Session 2: a brand new process/session reloading the exact same
    // on-disk scene file, which *still* says "textures/rock.png" (nothing
    // has saved it back yet). The scene load itself does not hard-fail —
    // `RenderSceneAdapter` treats an unresolved asset reference as a
    // recoverable, logged skip (same as a scene referencing any other
    // missing asset) — but the broken reference means the entity ends up
    // *without* its `MeshRenderable3d` component. The file has not been
    // migrated, so the gate's guarantee does not (and should not) apply to
    // it yet.
    let database_session_2 =
        AssetDatabase::open(&assets_root, &cache_root).expect("database should reopen");
    let (world_session_2, _) = load_scene(&assets_root, &scene_path, database_session_2)
        .expect("scene load itself does not hard-fail on an unresolved reference");
    let cube_session_2 = world_session_2
        .iter_entities()
        .find(|entity| {
            world_session_2
                .get::<EntityName>(entity.id())
                .is_some_and(|name| name.0 == "Cube")
        })
        .expect("cube entity should still spawn");
    assert!(
        world_session_2
            .get::<MeshRenderable3d>(cube_session_2.id())
            .is_none(),
        "a never-resaved legacy scene is expected to lose its mesh renderer after the rename"
    );

    // --- Migrate: save the world from session 1 (which already knows the
    // ids) back over the same scene file. This is what "migration on save"
    // means in practice — no separate migration tool, just the normal
    // authoring save path.
    let (type_registry, component_registry, metadata_registry) = build_registries();
    let adapter = RenderSceneAdapter;
    let serializer = SceneSerializer::new(&world_session_1, &component_registry, &type_registry)
        .with_metadata_registry(&metadata_registry)
        .with_asset_server(&asset_server_session_1)
        .with_external_components(&adapter);
    serializer
        .save_file(&scene_path, "GateTest")
        .expect("resaving the migrated scene should succeed");

    let resaved_source = fs::read_to_string(&scene_path).expect("resaved scene should be readable");
    assert!(
        !resaved_source.contains("rock.png") && !resaved_source.contains("boulder.png"),
        "resaved scene should reference the texture by id, not by path: {resaved_source}"
    );

    // --- Session 3: fresh session, reloading the now-migrated scene file.
    // The texture is still renamed (from session 1) — and this time it
    // resolves, because the reference is an id, not the old path.
    let database_session_3 =
        AssetDatabase::open(&assets_root, &cache_root).expect("database should reopen again");
    let (world_session_3, _asset_server_session_3) =
        load_scene(&assets_root, &scene_path, database_session_3)
            .expect("migrated scene should survive the earlier rename");

    let mut found_cube = false;
    for entity in world_session_3.iter_entities() {
        if let Some(name) = world_session_3.get::<EntityName>(entity.id()) {
            if name.0 == "Cube" {
                found_cube = true;
                assert!(world_session_3
                    .get::<MeshRenderable3d>(entity.id())
                    .is_some());
            }
        }
    }
    assert!(found_cube, "the cube entity should have reloaded");

    let _ = fs::remove_dir_all(&root);
}
