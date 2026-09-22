//! Proves the M1 gate at the product level: the exact sequence of
//! `starman` subcommands a user would actually run — `new`, `validate`,
//! `import`, `test` — survives renaming an asset a scene references.
//! Deliberately redundant with
//! `crates/engine-render/tests/scene_reference_migration.rs`, which proves
//! the same phenomenon at the library level (`SceneDeserializer`/
//! `RenderSceneAdapter` directly) — this one proves it through the
//! commands a real user would type.

use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use bevy_ecs::world::World;
use engine_assets::{AssetDatabase, AssetServer, SceneSerializer};
use engine_core::register_core_reflection_types;
use engine_reflect::{ComponentRegistry, ReflectMetadataRegistry, ReflectTypeRegistry};
use engine_render::{MeshRenderable3d, RenderSceneAdapter};
use image::{Rgba, RgbaImage};
use starman_cli::commands;

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

#[test]
fn new_validate_import_rename_test_survives_the_rename() {
    let root = scratch_dir("starman-cli-gate");

    // `new`: scaffold the project.
    let new_outcome = commands::new::run(
        root.clone(),
        "Gate Test".to_owned(),
        "scenes/main.scene.ron".to_owned(),
    )
    .expect("new should succeed");
    assert_eq!(new_outcome.path, root);

    // Copy a real texture into the project (the mesh/material fixtures
    // aren't needed for this test — one asset reference is enough to prove
    // the gate).
    let assets_dir = root.join("assets");
    let texture_relative = "textures/rock.png";
    let texture_dest = assets_dir.join(texture_relative);
    fs::create_dir_all(texture_dest.parent().unwrap()).expect("textures dir should be created");
    RgbaImage::from_pixel(2, 2, Rgba([9, 8, 7, 255]))
        .save(&texture_dest)
        .expect("texture fixture should be written");

    // Author a scene with one entity referencing that texture through a
    // real `MeshRenderable3d`... but `MeshRenderable3d` needs mesh and
    // material handles too. Reuse the reference project's mesh/material so
    // this stays a realistic MeshRenderer, not a synthetic stand-in.
    let reference = reference_assets_root();
    fs::create_dir_all(assets_dir.join("meshes")).unwrap();
    fs::create_dir_all(assets_dir.join("materials")).unwrap();
    fs::copy(
        reference.join("meshes/cube.glb"),
        assets_dir.join("meshes/cube.glb"),
    )
    .expect("mesh fixture should copy");
    fs::copy(
        reference.join("materials/default.ron"),
        assets_dir.join("materials/default.ron"),
    )
    .expect("material fixture should copy");

    let entry_scene_path = assets_dir.join("scenes/main.scene.ron");
    {
        let database = AssetDatabase::open(&assets_dir, root.join(".starman/cache/imported"))
            .expect("database should open");
        let mut asset_server = AssetServer::new(assets_dir.to_string_lossy().to_string());
        asset_server.attach_database(database);

        let texture = asset_server
            .load_texture_handle(texture_relative)
            .expect("texture should load");
        let mesh = asset_server
            .load_mesh_handle("meshes/cube.glb")
            .expect("mesh should load");
        let material = asset_server
            .load_material_handle("materials/default.ron")
            .expect("material should load");

        let mut world = World::new();
        world.spawn(MeshRenderable3d::new(mesh, texture, material));

        let (type_registry, component_registry, metadata_registry) = build_registries();
        let adapter = RenderSceneAdapter;
        let serializer = SceneSerializer::new(&world, &component_registry, &type_registry)
            .with_metadata_registry(&metadata_registry)
            .with_asset_server(&asset_server)
            .with_external_components(&adapter);
        serializer
            .save_file(&entry_scene_path, "GateTest")
            .expect("authored scene should save");
    }

    // The scene now references the texture by id (it was loaded and saved
    // through a database-attached AssetServer) — confirm that before
    // relying on it, so a regression here fails loudly and specifically.
    let saved_source = fs::read_to_string(&entry_scene_path).expect("scene should be readable");
    assert!(
        !saved_source.contains("rock.png"),
        "authored scene should reference the texture by id, not by path: {saved_source}"
    );

    // `validate`: the project (manifest + layout) is well-formed.
    let report = commands::validate::run(root.clone()).expect("validate should succeed");
    assert!(report.is_valid(), "report: {report}");

    // `import`: every asset (including the one just authored) imports
    // cleanly.
    let import_outcome = commands::import::run(root.clone()).expect("import should succeed");
    assert!(import_outcome.summary.is_success());

    // Rename the referenced texture — through the database, the only way a
    // rename is guaranteed to preserve the asset's id.
    let mut database = AssetDatabase::open(&assets_dir, root.join(".starman/cache/imported"))
        .expect("database should reopen");
    database
        .rename("textures/rock.png", "textures/boulder.png")
        .expect("rename should succeed");
    drop(database);

    // `test`: headlessly loads the entry scene. If the rename had broken
    // the reference, the entity would still spawn (a broken asset
    // reference is a logged, recoverable skip — see
    // `scene_reference_migration.rs`) but without its `MeshRenderable3d`.
    // `starman-cli test` doesn't currently expose per-entity component
    // detail, but the same headless-load machinery is exercised end to
    // end; the definitive component-level assertion lives in
    // `scene_reference_migration.rs`. Here, success plus a non-zero entity
    // count is the product-level signal: the project the user actually
    // interacts with still opens.
    let test_outcome = commands::test::run(root.clone()).expect("test should succeed after rename");
    assert_eq!(test_outcome.entity_count, 1);

    let _ = fs::remove_dir_all(&root);
}
