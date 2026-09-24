use super::{Asset, AssetLoader, AssetRef, Assets, LoadContext, LoadState};
use crate::AssetServer;
use engine_core::Result;
use std::path::PathBuf;

#[derive(Debug, PartialEq)]
struct Note(String);

impl Asset for Note {
    const TYPE_NAME: &'static str = "Note";
}

struct NoteLoader;

impl AssetLoader for NoteLoader {
    type Asset = Note;

    fn extensions(&self) -> &'static [&'static str] {
        &["note.txt"]
    }

    fn load(&self, bytes: &[u8], ctx: &mut LoadContext<'_>) -> Result<Note> {
        let text = std::str::from_utf8(bytes).map_err(|_| ctx.error("not utf-8"))?;
        if text.starts_with("bad") {
            return Err(ctx.error("bad note"));
        }
        Ok(Note(match ctx.sub_key {
            Some(key) => format!("{}#{key}", text.trim()),
            None => text.trim().to_owned(),
        }))
    }
}

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "starman-typed-{tag}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn server(dir: &TempDir) -> AssetServer {
    let server = AssetServer::new(dir.0.to_string_lossy().to_string());
    server.assets().register_loader(NoteLoader);
    server
}

#[test]
fn requests_are_deduplicated_and_load_blocking() {
    let dir = TempDir::new("dedup");
    std::fs::write(dir.0.join("a.note.txt"), "hello").unwrap();
    let mut server = server(&dir);
    let first = server.assets().request_path::<Note>("a.note.txt");
    let second = server.assets().request_path::<Note>("a.note.txt");
    assert_eq!(first, second);
    assert_eq!(server.assets().state(first), Some(LoadState::Pending));
    let report = server.update_blocking();
    assert_eq!(report.loaded, vec![("Note", "a.note.txt".to_owned())]);
    assert_eq!(server.assets().get(first).unwrap().0, "hello");
    assert!(server.assets().revision(first) > 0);
}

#[test]
fn async_update_eventually_loads() {
    let dir = TempDir::new("async");
    std::fs::write(dir.0.join("b.note.txt"), "async").unwrap();
    let mut server = server(&dir);
    let handle = server.assets().request_path::<Note>("b.note.txt");
    let mut loaded = false;
    for _ in 0..500 {
        server.update();
        if server.assets().state(handle) == Some(LoadState::Loaded) {
            loaded = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    assert!(loaded);
}

#[test]
fn failures_are_reported_with_context() {
    let dir = TempDir::new("fail");
    std::fs::write(dir.0.join("c.note.txt"), "bad content").unwrap();
    let mut server = server(&dir);
    let error = server
        .load_blocking::<Note>("c.note.txt")
        .unwrap_err()
        .to_string();
    assert!(error.contains("bad note"), "{error}");
    let missing = server
        .load_blocking::<Note>("missing.note.txt")
        .unwrap_err()
        .to_string();
    assert!(missing.contains("does not exist"), "{missing}");
    let empty = server.assets().request::<Note>(&AssetRef::default());
    assert!(matches!(
        server.assets().state(empty),
        Some(LoadState::Failed(_))
    ));
}

#[test]
fn sub_keys_are_passed_to_loaders() {
    let dir = TempDir::new("sub");
    std::fs::write(dir.0.join("d.note.txt"), "base").unwrap();
    let mut server = server(&dir);
    let handle = server.load_blocking::<Note>("d.note.txt#part:1").unwrap();
    assert_eq!(server.assets().get(handle).unwrap().0, "base#part:1");
}

#[test]
fn inserted_assets_are_replaceable_and_versioned() {
    let assets = Assets::new();
    let handle = assets.insert("procedural", Note("one".into()));
    let rev = assets.revision(handle);
    assert!(assets.replace(handle, Note("two".into())));
    assert_eq!(assets.get(handle).unwrap().0, "two");
    assert!(assets.revision(handle) > rev);
    assert_eq!(assets.insert("procedural", Note("three".into())), handle);
}

#[test]
fn longest_suffix_decides_the_type() {
    let assets = Assets::new();
    assets.register_loader(NoteLoader);
    assert_eq!(assets.type_name_for_path("x/y.note.txt"), Some("Note"));
    assert_eq!(assets.type_name_for_path("x/y.txt"), None);
}

#[test]
fn ids_resolve_through_the_database_after_rename() {
    let dir = TempDir::new("rename");
    let assets_root = dir.0.join("assets");
    std::fs::create_dir_all(&assets_root).unwrap();
    std::fs::write(assets_root.join("e.note.txt"), "tracked").unwrap();
    let mut database =
        crate::AssetDatabase::open(&assets_root, dir.0.join("cache")).expect("database");
    let id = database.ensure_imported("e.note.txt").unwrap();
    database.rename("e.note.txt", "moved.note.txt").unwrap();
    let mut server = AssetServer::new(assets_root.to_string_lossy().to_string());
    server.assets().register_loader(NoteLoader);
    server.attach_database(database);
    let handle = server
        .load_ref_blocking::<Note>(&AssetRef::from_id(id).with_path("e.note.txt"))
        .unwrap();
    assert_eq!(server.assets().get(handle).unwrap().0, "tracked");
    assert_eq!(
        server.assets().source(handle).unwrap().relative_path,
        "moved.note.txt"
    );
}
