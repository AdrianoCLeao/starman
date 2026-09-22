//! Proves the M1 gate's asset-database side against the *real* reference
//! project (`examples/reference-project/assets`), not a synthetic fixture —
//! Definição de Pronto #5 ("foi usada em uma fatia real do jogo de
//! referência") applied to the import pipeline specifically.

use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use engine_assets::AssetDatabase;

fn reference_assets_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("examples")
        .join("reference-project")
        .join("assets")
}

fn scratch_dir(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be valid")
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{nanos}"))
}

/// Copies the real reference project's `assets/` into a scratch directory.
/// `AssetDatabase::ensure_imported` writes `.meta.ron` sidecars next to
/// every source asset as a side effect of importing it — this test must
/// never do that to the actual, checked-in project.
///
/// `include_meta` controls whether already-committed `.meta.ron` sidecars
/// (the reference project ships with these checked in — ADR 0003 treats
/// them as permanent output, not disposable cache) come along: `false`
/// simulates a from-scratch first import (e.g. a fresh clone before
/// `.meta.ron` files existed at all); `true` simulates the project exactly
/// as checked in today.
fn copy_reference_assets(destination: &std::path::Path, include_meta: bool) {
    let source = reference_assets_root();
    assert!(
        source.is_dir(),
        "reference project assets should exist at '{}'",
        source.display()
    );
    copy_dir_recursive(&source, destination, include_meta);
}

fn copy_dir_recursive(source: &std::path::Path, destination: &std::path::Path, include_meta: bool) {
    fs::create_dir_all(destination).expect("destination dir should be created");
    for entry in fs::read_dir(source).expect("source dir should be readable") {
        let entry = entry.expect("entry should be readable");
        let path = entry.path();
        let is_meta = path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".meta.ron"));
        if is_meta && !include_meta {
            continue;
        }
        let dest_path = destination.join(entry.file_name());
        if path.is_dir() {
            copy_dir_recursive(&path, &dest_path, include_meta);
        } else {
            fs::copy(&path, &dest_path).expect("file should copy");
        }
    }
}

#[test]
fn every_asset_in_the_reference_project_imports_cleanly_from_scratch() {
    let root = scratch_dir("starman-reference-project-import");
    let assets_root = root.join("assets");
    copy_reference_assets(&assets_root, false);

    // ATTRIBUTION.md is documentation, not an importable asset type, but
    // `import_all` still hashes/caches it (see its own doc comment) — so
    // this is every regular file under assets/, not a filtered "recognized
    // asset types only" count.
    let expected_files = count_files(&assets_root);

    let cache_root = root.join(".starman/cache/imported");
    let mut database =
        AssetDatabase::open(&assets_root, &cache_root).expect("database should open");

    let summary = database.import_all().expect("import_all should succeed");

    assert!(
        summary.is_success(),
        "every reference asset should import without failure, got failures: {:?}",
        summary.failed
    );
    assert_eq!(
        summary.imported.len(),
        expected_files,
        "expected one imported entry per file under {}",
        assets_root.display()
    );
    assert!(summary.unchanged.is_empty());

    let _ = fs::remove_dir_all(&root);
}

#[test]
fn the_committed_project_is_already_up_to_date_and_stays_that_way_on_reimport() {
    let root = scratch_dir("starman-reference-project-committed");
    let assets_root = root.join("assets");
    copy_reference_assets(&assets_root, true);
    let expected_files = count_files_excluding_meta(&assets_root);

    let cache_root = root.join(".starman/cache/imported");
    let mut database =
        AssetDatabase::open(&assets_root, &cache_root).expect("database should open");

    // The committed `.meta.ron` sidecars already match their source's
    // content hash, so even the *first* import here should find nothing to
    // do — proving the checked-in reference project is actually up to date
    // with itself, not just that reimporting twice is stable.
    let first = database.import_all().expect("first import should succeed");
    assert!(first.is_success());
    assert!(
        first.imported.is_empty(),
        "the committed project should already be fully imported, got: {:?}",
        first.imported
    );
    assert_eq!(first.unchanged.len(), expected_files);

    database.rescan().expect("rescan should succeed");
    let second = database.import_all().expect("second import should succeed");

    assert!(second.is_success());
    assert!(second.imported.is_empty());
    assert_eq!(second.unchanged.len(), expected_files);

    let _ = fs::remove_dir_all(&root);
}

fn count_files(dir: &std::path::Path) -> usize {
    count_files_impl(dir, true)
}

fn count_files_excluding_meta(dir: &std::path::Path) -> usize {
    count_files_impl(dir, false)
}

fn count_files_impl(dir: &std::path::Path, include_meta: bool) -> usize {
    let mut count = 0;
    for entry in fs::read_dir(dir).expect("dir should be readable") {
        let entry = entry.expect("entry should be readable");
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_str().unwrap_or_default();
        let is_hidden = name.starts_with('.');
        let is_meta = name.ends_with(".meta.ron");
        if is_hidden || (is_meta && !include_meta) {
            continue;
        }
        if path.is_dir() {
            count += count_files_impl(&path, include_meta);
        } else {
            count += 1;
        }
    }
    count
}
