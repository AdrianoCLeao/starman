use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Sender};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use engine_assets::{
    AssetDatabase, AssetServer, AssetState, AssetWatcher, HotReloadReport, WatchConfig,
};
use image::{Rgba, RgbaImage};
use notify::{event::ModifyKind, Event, EventKind};

struct Fixture {
    root: PathBuf,
    assets: PathBuf,
    server: AssetServer,
    events: Sender<notify::Result<Event>>,
}

impl Fixture {
    fn new(prefix: &str) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be valid")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("{prefix}-{nanos}"));
        let assets = root.join("assets");
        fs::create_dir_all(assets.join("textures")).expect("assets dir should be created");
        fs::create_dir_all(assets.join("materials")).expect("assets dir should be created");

        let mut server = AssetServer::new(assets.to_string_lossy().to_string());
        let (events, rx) = mpsc::channel();
        server.replace_watcher(AssetWatcher::with_event_source(
            &assets,
            rx,
            WatchConfig {
                quiet_window: Duration::ZERO,
                worker_threads: 2,
            },
        ));

        Self {
            root,
            assets,
            server,
            events,
        }
    }

    fn attach_database(&mut self) {
        let database = AssetDatabase::open(&self.assets, self.root.join(".starman/cache/imported"))
            .expect("database should open");
        self.server.attach_database(database);
    }

    fn touch_event(&self, relative: &str) {
        let event =
            Event::new(EventKind::Modify(ModifyKind::Any)).add_path(self.assets.join(relative));
        self.events.send(Ok(event)).expect("event should be sent");
    }

    /// Polls until `done` accepts the accumulated report or a timeout hits.
    fn poll_until(&mut self, done: impl Fn(&HotReloadReport) -> bool) -> HotReloadReport {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut total = HotReloadReport::default();
        loop {
            let report = self.server.poll_hot_reload();
            total.textures.extend(report.textures);
            total.meshes.extend(report.meshes);
            total.materials.extend(report.materials);
            total.scenes.extend(report.scenes);
            total.removed.extend(report.removed);
            total.failed.extend(report.failed);
            if done(&total) {
                return total;
            }
            assert!(Instant::now() < deadline, "timed out; got {total:?}");
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn write_png(path: &Path, rgba: [u8; 4]) {
    RgbaImage::from_pixel(2, 2, Rgba(rgba))
        .save(path)
        .expect("png should be saved");
}

const MATERIAL_A: &str = "(base_color_factor: [1.0, 0.0, 0.0, 1.0], metallic: 0.1, roughness: 0.9)";
const MATERIAL_B: &str = "(base_color_factor: [0.0, 1.0, 0.0, 1.0], metallic: 0.5, roughness: 0.4)";

#[test]
fn changed_texture_is_reloaded_with_a_newer_revision() {
    let mut fixture = Fixture::new("starman-watch-texture");
    write_png(&fixture.assets.join("textures/a.png"), [255, 0, 0, 255]);
    let handle = fixture
        .server
        .load_texture_handle("textures/a.png")
        .unwrap();
    let before = fixture.server.texture_payload(handle).unwrap().clone();

    write_png(&fixture.assets.join("textures/a.png"), [0, 255, 0, 255]);
    fixture.touch_event("textures/a.png");
    let report = fixture.poll_until(|report| !report.textures.is_empty());

    assert_eq!(report.textures.len(), 1);
    let after = fixture.server.texture_payload(handle).unwrap();
    assert!(after.revision > before.revision);
    assert_ne!(after.pixels_rgba8, before.pixels_rgba8);
}

#[test]
fn a_burst_of_events_reloads_once() {
    let mut fixture = Fixture::new("starman-watch-burst");
    write_png(&fixture.assets.join("textures/a.png"), [255, 0, 0, 255]);
    fixture
        .server
        .load_texture_handle("textures/a.png")
        .unwrap();

    write_png(&fixture.assets.join("textures/a.png"), [0, 0, 255, 255]);
    for _ in 0..10 {
        fixture.touch_event("textures/a.png");
    }
    let report = fixture.poll_until(|report| !report.textures.is_empty());
    assert_eq!(report.textures.len(), 1);

    let extra = fixture.server.poll_hot_reload();
    assert!(extra.textures.is_empty());
}

#[test]
fn changed_material_is_parsed_and_revised() {
    let mut fixture = Fixture::new("starman-watch-material");
    let path = fixture.assets.join("materials/m.ron");
    fs::write(&path, MATERIAL_A).unwrap();
    let handle = fixture
        .server
        .load_material_handle("materials/m.ron")
        .unwrap();
    assert_eq!(
        fixture.server.material_payload(handle).unwrap().metallic,
        0.1
    );
    let revision_before = fixture.server.material_revision(handle).unwrap();

    fs::write(&path, MATERIAL_B).unwrap();
    fixture.touch_event("materials/m.ron");
    fixture.poll_until(|report| !report.materials.is_empty());

    let payload = fixture.server.material_payload(handle).unwrap();
    assert_eq!(payload.metallic, 0.5);
    assert_eq!(payload.base_color_factor, [0.0, 1.0, 0.0, 1.0]);
    assert!(fixture.server.material_revision(handle).unwrap() > revision_before);
}

#[test]
fn invalid_material_keeps_previous_payload_and_marks_failed() {
    let mut fixture = Fixture::new("starman-watch-bad-material");
    let path = fixture.assets.join("materials/m.ron");
    fs::write(&path, MATERIAL_A).unwrap();
    let handle = fixture
        .server
        .load_material_handle("materials/m.ron")
        .unwrap();

    fs::write(&path, "( this is not valid").unwrap();
    fixture.touch_event("materials/m.ron");
    let report = fixture.poll_until(|report| !report.failed.is_empty());

    assert!(report.materials.is_empty());
    assert_eq!(
        fixture.server.material_state(handle),
        Some(AssetState::Failed)
    );
    assert_eq!(
        fixture.server.material_payload(handle).unwrap().metallic,
        0.1
    );

    // Fixing the file recovers the asset.
    fs::write(&path, MATERIAL_B).unwrap();
    fixture.touch_event("materials/m.ron");
    fixture.poll_until(|report| !report.materials.is_empty());
    assert_eq!(
        fixture.server.material_state(handle),
        Some(AssetState::Loaded)
    );
    assert_eq!(
        fixture.server.material_payload(handle).unwrap().metallic,
        0.5
    );
}

#[test]
fn scene_change_is_only_reported() {
    let mut fixture = Fixture::new("starman-watch-scene");
    fs::create_dir_all(fixture.assets.join("scenes")).unwrap();
    fs::write(fixture.assets.join("scenes/level.scene.ron"), "()").unwrap();

    fixture.touch_event("scenes/level.scene.ron");
    let report = fixture.poll_until(|report| !report.scenes.is_empty());

    assert_eq!(report.scenes.len(), 1);
    assert_eq!(report.reloaded_count(), 0);
}

#[test]
fn removed_file_is_reported_and_payload_is_kept() {
    let mut fixture = Fixture::new("starman-watch-removed");
    let path = fixture.assets.join("textures/a.png");
    write_png(&path, [1, 2, 3, 255]);
    let handle = fixture
        .server
        .load_texture_handle("textures/a.png")
        .unwrap();

    fs::remove_file(&path).unwrap();
    let event = Event::new(EventKind::Remove(notify::event::RemoveKind::File)).add_path(path);
    fixture.events.send(Ok(event)).unwrap();
    let report = fixture.poll_until(|report| !report.removed.is_empty());

    assert_eq!(report.removed.len(), 1);
    assert!(fixture.server.texture_payload(handle).is_some());
}

#[test]
fn reimport_updates_meta_and_cache_and_suppresses_identical_saves() {
    let mut fixture = Fixture::new("starman-watch-database");
    write_png(&fixture.assets.join("textures/a.png"), [255, 0, 0, 255]);
    fixture.attach_database();
    let id = fixture
        .server
        .database_mut()
        .unwrap()
        .ensure_imported("textures/a.png")
        .unwrap();
    fixture
        .server
        .load_texture_handle("textures/a.png")
        .unwrap();
    let hash_before = fixture
        .server
        .database()
        .unwrap()
        .meta(id)
        .unwrap()
        .content_hash
        .clone();

    // Identical bytes re-saved: nothing to reload.
    fixture.touch_event("textures/a.png");
    std::thread::sleep(Duration::from_millis(100));
    let quiet = fixture.server.poll_hot_reload();
    assert!(quiet.textures.is_empty());

    // Real change: reloaded, meta hash updated, cache entry present.
    write_png(&fixture.assets.join("textures/a.png"), [0, 255, 0, 255]);
    fixture.touch_event("textures/a.png");
    fixture.poll_until(|report| !report.textures.is_empty());

    let database = fixture.server.database().unwrap();
    let hash_after = database.meta(id).unwrap().content_hash.clone();
    assert_ne!(hash_before, hash_after);
    assert!(database.cache_entry_path(id).unwrap().is_file());
}

#[test]
fn dependent_material_is_reported_when_its_texture_changes() {
    let mut fixture = Fixture::new("starman-watch-dependents");
    write_png(&fixture.assets.join("textures/a.png"), [255, 0, 0, 255]);
    fs::write(fixture.assets.join("materials/m.ron"), MATERIAL_A).unwrap();
    fixture.attach_database();

    {
        let database = fixture.server.database_mut().unwrap();
        let texture = database.ensure_imported("textures/a.png").unwrap();
        let material = database.ensure_imported("materials/m.ron").unwrap();
        database.set_dependencies(material, vec![texture]).unwrap();
    }
    fixture
        .server
        .load_texture_handle("textures/a.png")
        .unwrap();
    fixture
        .server
        .load_material_handle("materials/m.ron")
        .unwrap();

    write_png(&fixture.assets.join("textures/a.png"), [0, 0, 255, 255]);
    fixture.touch_event("textures/a.png");
    let report =
        fixture.poll_until(|report| !report.textures.is_empty() && !report.materials.is_empty());

    assert_eq!(report.textures.len(), 1);
    assert_eq!(report.materials.len(), 1);
}

#[test]
fn meta_and_temp_files_are_ignored() {
    let mut fixture = Fixture::new("starman-watch-ignored");
    write_png(&fixture.assets.join("textures/a.png"), [1, 1, 1, 255]);
    fixture
        .server
        .load_texture_handle("textures/a.png")
        .unwrap();

    fixture.touch_event("textures/a.png.meta.ron");
    fixture.touch_event("textures/a.png.tmp-1-2");
    std::thread::sleep(Duration::from_millis(50));
    let report = fixture.server.poll_hot_reload();
    assert_eq!(report, HotReloadReport::default());
}

/// Exercises the real OS watcher end to end. Sensitive to filesystem event
/// timing, so it is opt-in: `cargo test -p engine-assets -- --ignored`.
#[test]
#[ignore = "depends on OS filesystem event timing"]
fn real_filesystem_watcher_reloads_a_changed_texture() {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!("starman-watch-real-{nanos}"));
    fs::create_dir_all(&root).unwrap();
    write_png(&root.join("a.png"), [255, 0, 0, 255]);

    let mut server = AssetServer::new(root.to_string_lossy().to_string());
    let handle = server.load_texture_handle("a.png").unwrap();
    let before = server.texture_payload(handle).unwrap().revision;

    std::thread::sleep(Duration::from_millis(200));
    write_png(&root.join("a.png"), [0, 255, 0, 255]);

    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        if !server.poll_hot_reload().textures.is_empty() {
            assert!(server.texture_payload(handle).unwrap().revision > before);
            let _ = fs::remove_dir_all(&root);
            return;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let _ = fs::remove_dir_all(&root);
    panic!("real watcher did not report the change in time");
}
