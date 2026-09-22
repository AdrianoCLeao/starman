use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use engine_assets::{AssetDatabase, AssetServer};
use image::{Rgba, RgbaImage};

fn scratch_project(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be valid")
        .as_nanos();
    let root = std::env::temp_dir().join(format!("{prefix}-{nanos}"));
    fs::create_dir_all(root.join("assets/textures")).expect("assets dir should be created");
    fs::create_dir_all(root.join("assets/materials")).expect("assets dir should be created");
    root
}

fn write_png(path: &std::path::Path) {
    RgbaImage::from_pixel(2, 2, Rgba([1, 2, 3, 255]))
        .save(path)
        .expect("png should be saved");
}

const MATERIAL: &str = "(base_color_factor: [1.0, 1.0, 1.0, 1.0], metallic: 0.0, roughness: 1.0)";

#[test]
fn loading_with_a_database_attached_learns_the_source_id() {
    let root = scratch_project("starman-typed-refs-with-db");
    write_png(&root.join("assets/textures/a.png"));
    fs::write(root.join("assets/materials/m.ron"), MATERIAL).unwrap();

    let database = AssetDatabase::open(root.join("assets"), root.join(".starman/cache/imported"))
        .expect("database should open");
    let mut server = AssetServer::new(root.join("assets").to_string_lossy().to_string());
    server.attach_database(database);

    let texture = server.load_texture_handle("textures/a.png").unwrap();
    let material = server.load_material_handle("materials/m.ron").unwrap();

    assert!(server.texture_source_id(texture).is_some());
    assert!(server.material_source_id(material).is_some());

    let _ = fs::remove_dir_all(&root);
}

#[test]
fn loading_without_a_database_never_learns_a_source_id() {
    let root = scratch_project("starman-typed-refs-no-db");
    write_png(&root.join("assets/textures/a.png"));

    let mut server = AssetServer::new(root.join("assets").to_string_lossy().to_string());
    let texture = server.load_texture_handle("textures/a.png").unwrap();

    assert_eq!(server.texture_source_id(texture), None);

    let _ = fs::remove_dir_all(&root);
}

#[test]
fn load_by_id_round_trips_to_the_same_source_id() {
    let root = scratch_project("starman-typed-refs-by-id");
    write_png(&root.join("assets/textures/a.png"));

    let mut database =
        AssetDatabase::open(root.join("assets"), root.join(".starman/cache/imported"))
            .expect("database should open");
    let id = database.ensure_imported("textures/a.png").unwrap();

    let mut server = AssetServer::new(root.join("assets").to_string_lossy().to_string());
    server.attach_database(database);

    let handle = server.load_texture_handle_by_id(id).unwrap();
    assert_eq!(server.texture_source_id(handle), Some(id));

    let _ = fs::remove_dir_all(&root);
}
