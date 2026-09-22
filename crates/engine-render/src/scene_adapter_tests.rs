use std::collections::HashMap;

use bevy_ecs::world::World;
use engine_assets::{AssetDatabase, AssetServer, SceneExternalComponents, SceneValue};

use crate::{MeshRenderable3d, RenderSceneAdapter, SpriteRenderable2d};

fn reference_assets_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("examples")
        .join("reference-project")
        .join("assets")
}

#[test]
fn render_scene_adapter_serializes_and_deserializes_mesh_and_sprite_components() {
    let assets_root = reference_assets_root();
    let mut asset_server = AssetServer::new(assets_root.to_string_lossy().to_string());
    let texture = asset_server
        .load_texture_handle("textures/placeholder.png")
        .expect("texture should load");
    let mesh = asset_server
        .load_mesh_handle("meshes/cube.glb")
        .expect("mesh should load");
    let material = asset_server
        .load_material_handle("materials/default.ron")
        .expect("material should load");

    let mut world = World::new();
    let entity = world
        .spawn((
            MeshRenderable3d::new(mesh, texture, material),
            SpriteRenderable2d::new(texture)
                .with_size(2.0, 3.0)
                .with_color([0.25, 0.5, 0.75, 1.0]),
        ))
        .id();

    let adapter = RenderSceneAdapter;
    let mut serialized_components = HashMap::new();
    adapter
        .serialize_entity_components(&world, entity, &asset_server, &mut serialized_components)
        .expect("render components should serialize");

    assert!(serialized_components.contains_key("MeshRenderer"));
    assert!(serialized_components.contains_key("Sprite"));

    let mut loaded_world = World::new();
    let loaded_entity = loaded_world.spawn_empty().id();
    for (component_name, component_value) in &serialized_components {
        let handled = adapter
            .deserialize_entity_component(
                &mut loaded_world,
                loaded_entity,
                component_name,
                component_value,
                &mut asset_server,
            )
            .expect("component should deserialize");
        assert!(handled);
    }

    let loaded_mesh_renderer = loaded_world
        .get::<MeshRenderable3d>(loaded_entity)
        .expect("mesh renderer should be present");
    assert_eq!(loaded_mesh_renderer.mesh.id(), mesh.id());
    assert_eq!(loaded_mesh_renderer.texture.id(), texture.id());
    assert_eq!(loaded_mesh_renderer.material.id(), material.id());

    let loaded_sprite = loaded_world
        .get::<SpriteRenderable2d>(loaded_entity)
        .expect("sprite should be present");
    assert_eq!(loaded_sprite.texture.id(), texture.id());
    assert_eq!(loaded_sprite.size, [2.0, 3.0]);
    assert_eq!(loaded_sprite.color, [0.25, 0.5, 0.75, 1.0]);
}

#[test]
fn render_scene_adapter_skips_mesh_renderer_with_missing_required_field() {
    let mut world = World::new();
    let entity = world.spawn_empty().id();
    let mut asset_server = AssetServer::new("assets");
    let adapter = RenderSceneAdapter;

    let payload = map_value(vec![
        (
            "texture",
            SceneValue::String("textures/placeholder.png".to_owned()),
        ),
        (
            "material",
            SceneValue::String("materials/default.ron".to_owned()),
        ),
    ]);

    let handled = adapter
        .deserialize_entity_component(
            &mut world,
            entity,
            "MeshRenderer",
            &payload,
            &mut asset_server,
        )
        .expect("payload should be handled without failing");

    assert!(handled);
    assert!(world.get::<MeshRenderable3d>(entity).is_none());
}

#[test]
fn render_scene_adapter_skips_sprite_when_texture_asset_is_missing() {
    let mut world = World::new();
    let entity = world.spawn_empty().id();
    let mut asset_server = AssetServer::new("assets");
    let adapter = RenderSceneAdapter;

    let payload = map_value(vec![
        (
            "texture",
            SceneValue::String("textures/does-not-exist.png".to_owned()),
        ),
        (
            "size",
            SceneValue::Seq(vec![
                SceneValue::Number(ron::Number::new(64.0)),
                SceneValue::Number(ron::Number::new(64.0)),
            ]),
        ),
    ]);

    let handled = adapter
        .deserialize_entity_component(&mut world, entity, "Sprite", &payload, &mut asset_server)
        .expect("missing texture should not fail scene loading");

    assert!(handled);
    assert!(world.get::<SpriteRenderable2d>(entity).is_none());
}

fn map_value(entries: Vec<(&str, SceneValue)>) -> SceneValue {
    let mut map = ron::Map::new();
    for (key, value) in entries {
        map.insert(SceneValue::String(key.to_owned()), value);
    }
    SceneValue::Map(map)
}

fn scratch_dir(prefix: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time should be valid")
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{nanos}"))
}

/// Copies the reference project's texture/mesh/material fixtures into a
/// scratch `assets/` directory, so tests that attach a database (which
/// writes `.meta.ron` sidecars next to each source asset as a side effect
/// of loading) never mutate the real, checked-in `assets/` directory.
fn scratch_assets_root(prefix: &str) -> std::path::PathBuf {
    let root = scratch_dir(prefix);
    let source = reference_assets_root();
    for relative in [
        "textures/placeholder.png",
        "meshes/cube.glb",
        "materials/default.ron",
    ] {
        let destination = root.join(relative);
        std::fs::create_dir_all(destination.parent().unwrap())
            .expect("scratch asset dir should be created");
        std::fs::copy(source.join(relative), &destination).expect("reference fixture should copy");
    }
    root
}

#[test]
fn render_scene_adapter_serializes_ids_when_a_database_is_attached() {
    let assets_root = scratch_assets_root("starman-scene-adapter-ids-assets");
    let cache_root = scratch_dir("starman-scene-adapter-ids-cache");
    let database = AssetDatabase::open(&assets_root, &cache_root).expect("database should open");

    let mut asset_server = AssetServer::new(assets_root.to_string_lossy().to_string());
    asset_server.attach_database(database);

    let texture = asset_server
        .load_texture_handle("textures/placeholder.png")
        .expect("texture should load");
    let mesh = asset_server
        .load_mesh_handle("meshes/cube.glb")
        .expect("mesh should load");
    let material = asset_server
        .load_material_handle("materials/default.ron")
        .expect("material should load");

    let mut world = World::new();
    let entity = world
        .spawn(MeshRenderable3d::new(mesh, texture, material))
        .id();

    let adapter = RenderSceneAdapter;
    let mut serialized_components = HashMap::new();
    adapter
        .serialize_entity_components(&world, entity, &asset_server, &mut serialized_components)
        .expect("render components should serialize");

    let SceneValue::Map(mesh_renderer) = serialized_components
        .get("MeshRenderer")
        .expect("MeshRenderer should have been serialized")
    else {
        panic!("MeshRenderer should serialize as a map");
    };

    for field in ["mesh", "texture", "material"] {
        let SceneValue::String(value) = mesh_renderer
            .iter()
            .find_map(|(key, value)| match key {
                SceneValue::String(key) if key == field => Some(value),
                _ => None,
            })
            .unwrap_or_else(|| panic!("field '{field}' should be present"))
        else {
            panic!("field '{field}' should be a string");
        };
        // Path-based references contain a '/'; id-based ones are bare
        // UUIDs and never do.
        assert!(
            !value.contains('/'),
            "field '{field}' should serialize as an id, got '{value}'"
        );
    }

    let _ = std::fs::remove_dir_all(&cache_root);
    let _ = std::fs::remove_dir_all(&assets_root);
}

#[test]
fn render_scene_adapter_round_trips_through_id_based_references() {
    let assets_root = scratch_assets_root("starman-scene-adapter-roundtrip-assets");
    let cache_root = scratch_dir("starman-scene-adapter-roundtrip-cache");
    let database = AssetDatabase::open(&assets_root, &cache_root).expect("database should open");

    let mut asset_server = AssetServer::new(assets_root.to_string_lossy().to_string());
    asset_server.attach_database(database);

    let texture = asset_server
        .load_texture_handle("textures/placeholder.png")
        .expect("texture should load");
    let mesh = asset_server
        .load_mesh_handle("meshes/cube.glb")
        .expect("mesh should load");
    let material = asset_server
        .load_material_handle("materials/default.ron")
        .expect("material should load");

    let mut world = World::new();
    let entity = world
        .spawn(MeshRenderable3d::new(mesh, texture, material))
        .id();

    let adapter = RenderSceneAdapter;
    let mut serialized_components = HashMap::new();
    adapter
        .serialize_entity_components(&world, entity, &asset_server, &mut serialized_components)
        .expect("render components should serialize");

    let mut loaded_world = World::new();
    let loaded_entity = loaded_world.spawn_empty().id();
    let handled = adapter
        .deserialize_entity_component(
            &mut loaded_world,
            loaded_entity,
            "MeshRenderer",
            serialized_components
                .get("MeshRenderer")
                .expect("MeshRenderer should have been serialized"),
            &mut asset_server,
        )
        .expect("id-based MeshRenderer should deserialize");
    assert!(handled);

    let loaded = loaded_world
        .get::<MeshRenderable3d>(loaded_entity)
        .expect("mesh renderer should be present");
    // Texture/material handles are memoized by path, so the same handle id
    // comes back. The mesh field, though, was serialized by its
    // *sub-asset* id (cube.glb has exactly one mesh, so it round-trips
    // through `load_mesh_handle_by_sub_id`, which — deliberately — is a
    // distinct registry entry from `load_mesh_handle`'s whole-file entry,
    // even for a single-mesh file (see scene_adapter.rs's module docs): a
    // fresh handle with byte-identical mesh content is expected here, not
    // the same handle id.
    assert_eq!(loaded.texture.id(), texture.id());
    assert_eq!(loaded.material.id(), material.id());
    let loaded_payload = asset_server
        .mesh_payload(loaded.mesh)
        .expect("loaded mesh payload should exist");
    let original_payload = asset_server
        .mesh_payload(mesh)
        .expect("original mesh payload should exist");
    assert_eq!(
        loaded_payload.vertices.len(),
        original_payload.vertices.len()
    );
    assert_eq!(loaded_payload.indices, original_payload.indices);

    let _ = std::fs::remove_dir_all(&cache_root);
    let _ = std::fs::remove_dir_all(&assets_root);
}

#[test]
fn render_scene_adapter_reads_legacy_path_references_and_learns_their_id() {
    let assets_root = scratch_assets_root("starman-scene-adapter-legacy-assets");
    let cache_root = scratch_dir("starman-scene-adapter-legacy-cache");
    let database = AssetDatabase::open(&assets_root, &cache_root).expect("database should open");

    let mut asset_server = AssetServer::new(assets_root.to_string_lossy().to_string());
    asset_server.attach_database(database);

    let adapter = RenderSceneAdapter;
    let payload = map_value(vec![
        ("mesh", SceneValue::String("meshes/cube.glb".to_owned())),
        (
            "texture",
            SceneValue::String("textures/placeholder.png".to_owned()),
        ),
        (
            "material",
            SceneValue::String("materials/default.ron".to_owned()),
        ),
    ]);

    let mut world = World::new();
    let entity = world.spawn_empty().id();
    let handled = adapter
        .deserialize_entity_component(
            &mut world,
            entity,
            "MeshRenderer",
            &payload,
            &mut asset_server,
        )
        .expect("legacy path payload should still deserialize");
    assert!(handled);

    let mesh_renderer = world
        .get::<MeshRenderable3d>(entity)
        .expect("mesh renderer should be present");

    // Loading by legacy path, with a database attached, should have taught
    // the asset server the mesh/texture/material's stable ids — proving
    // the "migration on save" mechanism: the next serialize would now
    // emit ids instead of paths for this same content.
    assert!(asset_server.mesh_source_id(mesh_renderer.mesh).is_some());
    assert!(asset_server
        .texture_source_id(mesh_renderer.texture)
        .is_some());
    assert!(asset_server
        .material_source_id(mesh_renderer.material)
        .is_some());

    let _ = std::fs::remove_dir_all(&cache_root);
    let _ = std::fs::remove_dir_all(&assets_root);
}
