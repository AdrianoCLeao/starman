use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use engine_assets::{AssetDatabase, AssetServer};

fn scratch_project(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be valid")
        .as_nanos();
    let root = std::env::temp_dir().join(format!("{prefix}-{nanos}"));
    fs::create_dir_all(root.join("assets/meshes")).expect("assets dir should be created");
    root
}

fn copy_two_mesh_fixture(assets_dir: &std::path::Path, relative: &str) -> String {
    let fixture =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/two_meshes.gltf");
    let dest = assets_dir.join(relative);
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).expect("parent dir should be created");
    }
    fs::copy(&fixture, &dest).expect("fixture should copy");
    relative.to_owned()
}

fn open_database(root: &std::path::Path) -> AssetDatabase {
    AssetDatabase::open(root.join("assets"), root.join(".starman/cache/imported"))
        .expect("database should open")
}

#[test]
fn importing_a_multi_mesh_file_registers_one_sub_asset_per_mesh() {
    let root = scratch_project("starman-sub-assets-import");
    let relative = copy_two_mesh_fixture(&root.join("assets"), "meshes/two.gltf");
    let mut database = open_database(&root);

    let id = database
        .ensure_imported(&relative)
        .expect("import should succeed");
    let sub_assets = database.sub_assets_of(id);

    assert_eq!(sub_assets.len(), 2);
    assert_eq!(sub_assets[0].key, "mesh:0");
    assert_eq!(sub_assets[1].key, "mesh:1");
    assert_eq!(sub_assets[0].label.as_deref(), Some("TriangleA"));
    assert_eq!(sub_assets[1].label.as_deref(), Some("TriangleB"));
    assert_ne!(sub_assets[0].id, sub_assets[1].id);

    let (resolved_parent, record) = database
        .resolve_sub_asset(sub_assets[0].id)
        .expect("sub-asset should resolve back to its parent");
    assert_eq!(resolved_parent, id);
    assert_eq!(record.key, "mesh:0");

    let _ = fs::remove_dir_all(&root);
}

#[test]
fn sub_asset_ids_are_deterministic_across_reimports() {
    let root = scratch_project("starman-sub-assets-deterministic");
    let relative = copy_two_mesh_fixture(&root.join("assets"), "meshes/two.gltf");
    let mut database = open_database(&root);

    let id = database
        .ensure_imported(&relative)
        .expect("first import should succeed");
    let first_sub_ids: Vec<_> = database.sub_assets_of(id).iter().map(|r| r.id).collect();

    // Touch the file with identical content and reimport: the parent id and
    // the derived sub-asset ids must be exactly the same as before.
    database.rescan().expect("rescan should succeed");
    let id_again = database
        .ensure_imported(&relative)
        .expect("reimport should succeed");
    let second_sub_ids: Vec<_> = database
        .sub_assets_of(id_again)
        .iter()
        .map(|r| r.id)
        .collect();

    assert_eq!(id, id_again);
    assert_eq!(first_sub_ids, second_sub_ids);

    let _ = fs::remove_dir_all(&root);
}

#[test]
fn load_mesh_handle_by_sub_id_loads_only_the_requested_mesh() {
    let root = scratch_project("starman-sub-assets-load-one");
    let relative = copy_two_mesh_fixture(&root.join("assets"), "meshes/two.gltf");
    let mut database = open_database(&root);
    let parent_id = database
        .ensure_imported(&relative)
        .expect("import should succeed");
    let sub_assets = database.sub_assets_of(parent_id).to_vec();

    let mut server = AssetServer::new(root.join("assets").to_string_lossy().to_string());
    server.attach_database(database);

    let handle_a = server
        .load_mesh_handle_by_sub_id(sub_assets[0].id)
        .expect("mesh 0 should load");
    let handle_b = server
        .load_mesh_handle_by_sub_id(sub_assets[1].id)
        .expect("mesh 1 should load");

    assert_ne!(handle_a.id(), handle_b.id());

    let payload_a = server.mesh_payload(handle_a).expect("payload should exist");
    let payload_b = server.mesh_payload(handle_b).expect("payload should exist");
    assert_eq!(payload_a.vertices.len(), 3);
    assert_eq!(payload_b.vertices.len(), 3);
    assert_eq!(payload_a.name, "TriangleA");
    assert_eq!(payload_b.name, "TriangleB");

    assert_eq!(server.mesh_sub_source_id(handle_a), Some(sub_assets[0].id));
    assert_eq!(server.mesh_source_id(handle_a), Some(parent_id));

    let _ = fs::remove_dir_all(&root);
}

#[test]
fn load_mesh_handle_merges_every_mesh_in_a_multi_mesh_file() {
    let root = scratch_project("starman-sub-assets-merge");
    let relative = copy_two_mesh_fixture(&root.join("assets"), "meshes/two.gltf");
    let mut database = open_database(&root);
    database
        .ensure_imported(&relative)
        .expect("import should succeed");

    let mut server = AssetServer::new(root.join("assets").to_string_lossy().to_string());
    server.attach_database(database);

    let handle = server
        .load_mesh_handle(&relative)
        .expect("merged mesh should load");
    let payload = server.mesh_payload(handle).expect("payload should exist");

    assert_eq!(payload.vertices.len(), 6, "both triangles combined");
    // A multi-mesh merge doesn't correspond to any single sub-asset.
    assert_eq!(server.mesh_sub_source_id(handle), None);
    assert!(server.mesh_source_id(handle).is_some());

    let _ = fs::remove_dir_all(&root);
}

#[test]
fn load_mesh_handle_on_a_single_mesh_file_also_learns_its_sub_asset_id() {
    let root = scratch_project("starman-sub-assets-single");
    let cube_source = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/reference-project/assets/meshes/cube.glb");
    let dest_relative = "meshes/cube.glb";
    fs::copy(&cube_source, root.join("assets").join(dest_relative))
        .expect("cube fixture should copy");

    let mut database = open_database(&root);
    let parent_id = database
        .ensure_imported(dest_relative)
        .expect("import should succeed");
    assert_eq!(
        database.sub_assets_of(parent_id).len(),
        1,
        "reference cube mesh has exactly one mesh"
    );

    let mut server = AssetServer::new(root.join("assets").to_string_lossy().to_string());
    server.attach_database(database);

    let handle = server
        .load_mesh_handle(dest_relative)
        .expect("mesh should load");

    assert!(server.mesh_source_id(handle).is_some());
    assert!(
        server.mesh_sub_source_id(handle).is_some(),
        "a single-mesh file's merged handle should also be known by its (only) sub-asset id"
    );

    let _ = fs::remove_dir_all(&root);
}
