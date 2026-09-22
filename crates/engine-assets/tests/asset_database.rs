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
    root
}

fn write_test_png(path: &std::path::Path, rgba: [u8; 4]) {
    let image = RgbaImage::from_pixel(2, 2, Rgba(rgba));
    image.save(path).expect("png should be saved");
}

/// Proves the M1 gate directly: a component that references a source asset
/// by its stable id (not by path) keeps resolving after the asset is
/// renamed, as long as the rename goes through the asset database.
#[test]
fn loading_a_texture_by_id_survives_a_rename() {
    let root = scratch_project("starman-asset-db-rename-e2e");
    let assets_dir = root.join("assets");
    write_test_png(&assets_dir.join("textures/rock.png"), [10, 20, 30, 255]);

    let mut database = AssetDatabase::open(&assets_dir, root.join(".starman/cache/imported"))
        .expect("database should open");
    let id = database
        .ensure_imported("textures/rock.png")
        .expect("texture should import");

    let mut server = AssetServer::new(assets_dir.to_string_lossy().to_string());
    server.attach_database(database);

    let handle_before = server
        .load_texture_handle_by_id(id)
        .expect("texture should load by id before rename");
    let payload_before = server
        .texture_payload(handle_before)
        .expect("payload should exist")
        .clone();

    server
        .database_mut()
        .expect("database should be attached")
        .rename("textures/rock.png", "textures/boulder.png")
        .expect("rename should succeed");

    let handle_after = server
        .load_texture_handle_by_id(id)
        .expect("texture should still load by the same id after rename");
    let payload_after = server
        .texture_payload(handle_after)
        .expect("payload should exist after rename");

    assert_eq!(payload_before.pixels_rgba8, payload_after.pixels_rgba8);

    let _ = fs::remove_dir_all(&root);
}

#[test]
fn loading_by_id_without_a_database_fails_with_an_actionable_error() {
    let mut server = AssetServer::new("assets");
    let bogus_id = engine_core::SourceAssetId::new_v4();

    let error = server
        .load_texture_handle_by_id(bogus_id)
        .expect_err("loading by id without a database should fail");

    assert!(error.to_string().contains("no asset database attached"));
}
