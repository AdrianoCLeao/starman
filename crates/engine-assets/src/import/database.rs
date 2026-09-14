//! The asset database: the source of truth for "what source assets exist,
//! what stable id each one has, and what depends on what" (ADR 0003).
//!
//! References into a project should be by [`SourceAssetId`], resolved to a
//! current relative path through this database at load time — that
//! indirection is what lets a source asset be renamed or moved without
//! breaking anything that references it, as long as the rename goes
//! through [`AssetDatabase::rename`] (or, failing that, the asset can still
//! be re-associated by content hash on the next [`AssetDatabase::rescan`]).

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use engine_core::{EngineError, Result, SourceAssetId};

use super::cache::ImportedCache;
use super::dependency_graph::DependencyGraph;
use super::hash::hash_bytes;
use super::meta::{
    default_importer_key_for, meta_path_for, read_meta, source_path_for_meta, write_meta,
    AssetMeta, META_SUFFIX,
};

pub struct AssetDatabase {
    assets_root: PathBuf,
    cache: ImportedCache,
    by_id: HashMap<SourceAssetId, String>,
    by_relative_path: HashMap<String, SourceAssetId>,
    metas: HashMap<SourceAssetId, AssetMeta>,
    /// Metadata whose source file is currently missing (e.g. renamed
    /// outside the engine without its sidecar): kept around only long
    /// enough to re-associate the id with wherever the content resurfaces.
    dangling: HashMap<String, (PathBuf, AssetMeta)>,
    dependency_graph: DependencyGraph,
}

impl AssetDatabase {
    /// Opens (and immediately scans) the asset database for a project's
    /// source assets directory, caching imported output under `cache_root`.
    pub fn open(assets_root: impl Into<PathBuf>, cache_root: impl Into<PathBuf>) -> Result<Self> {
        let mut database = Self {
            assets_root: assets_root.into(),
            cache: ImportedCache::new(cache_root),
            by_id: HashMap::new(),
            by_relative_path: HashMap::new(),
            metas: HashMap::new(),
            dangling: HashMap::new(),
            dependency_graph: DependencyGraph::default(),
        };
        database.rescan()?;
        Ok(database)
    }

    pub fn assets_root(&self) -> &Path {
        &self.assets_root
    }

    /// Rebuilds every in-memory index by walking the source assets
    /// directory. Cheap enough at M1 scale; a real project would want this
    /// driven incrementally by file-watch events instead (a later slice).
    pub fn rescan(&mut self) -> Result<()> {
        self.by_id.clear();
        self.by_relative_path.clear();
        self.metas.clear();
        self.dangling.clear();
        self.dependency_graph = DependencyGraph::default();

        let mut files = Vec::new();
        collect_files(&self.assets_root, &mut files)?;

        for path in &files {
            let Some(raw) = path.as_os_str().to_str() else {
                continue;
            };
            if !raw.ends_with(META_SUFFIX) {
                continue;
            }

            let Some(source_path) = source_path_for_meta(path) else {
                continue;
            };
            let Some(meta) = read_meta(path)? else {
                continue;
            };

            if source_path.is_file() {
                let Some(relative) = relative_string(&self.assets_root, &source_path) else {
                    continue;
                };
                self.by_id.insert(meta.id, relative.clone());
                self.by_relative_path.insert(relative, meta.id);
                self.dependency_graph
                    .set_dependencies(meta.id, &meta.dependencies);
                self.metas.insert(meta.id, meta);
            } else {
                // The source moved or was deleted without its sidecar: keep
                // the metadata around so a future `ensure_imported` call for
                // matching content can recover the same id.
                self.dangling
                    .insert(meta.content_hash.clone(), (path.clone(), meta));
            }
        }

        Ok(())
    }

    /// Ensures `relative_path` (relative to [`Self::assets_root`]) is
    /// imported: assigns it a stable id on first import (recovering one
    /// from a dangling sidecar if the content matches an asset that moved
    /// without its metadata), and re-imports whenever its content hash has
    /// changed since the last import. Returns the asset's stable id.
    pub fn ensure_imported(&mut self, relative_path: &str) -> Result<SourceAssetId> {
        let source_path = self.assets_root.join(relative_path);
        if !source_path.is_file() {
            return Err(EngineError::AssetLoad {
                path: relative_path.to_owned(),
                reason: "source asset does not exist".to_owned(),
            });
        }

        let bytes = fs::read(&source_path).map_err(|error| EngineError::AssetLoad {
            path: relative_path.to_owned(),
            reason: format!("failed to read source asset: {error}"),
        })?;
        let content_hash = hash_bytes(&bytes);

        let meta_path = meta_path_for(&source_path);
        let existing_meta = read_meta(&meta_path)?;

        let (id, dependencies) = match &existing_meta {
            Some(meta) => (meta.id, meta.dependencies.clone()),
            None => match self.dangling.remove(&content_hash) {
                Some((old_meta_path, meta)) => {
                    let _ = fs::remove_file(&old_meta_path);
                    log::info!(
                        target: "engine::assets",
                        "Re-associated asset id {} with '{}' by content hash after its metadata went missing",
                        meta.id,
                        relative_path,
                    );
                    (meta.id, meta.dependencies)
                }
                None => (SourceAssetId::new_v4(), Vec::new()),
            },
        };

        let importer = default_importer_key_for(&source_path).to_owned();
        let up_to_date = existing_meta
            .as_ref()
            .is_some_and(|meta| meta.content_hash == content_hash && meta.id == id);

        if !up_to_date {
            let meta = AssetMeta {
                version: AssetMeta::CURRENT_VERSION,
                id,
                importer,
                dependencies: dependencies.clone(),
                content_hash: content_hash.clone(),
            };
            write_meta(&meta_path, &meta)?;
            self.metas.insert(id, meta);
        }

        self.cache.store(&content_hash, &bytes)?;

        self.by_id.insert(id, relative_path.to_owned());
        self.by_relative_path.insert(relative_path.to_owned(), id);
        self.dependency_graph.set_dependencies(id, &dependencies);

        Ok(id)
    }

    /// Declares which other assets `id` depends on (e.g. a material
    /// depending on a texture), for cascade invalidation. Overwrites any
    /// previously declared dependencies for `id`.
    pub fn set_dependencies(
        &mut self,
        id: SourceAssetId,
        dependencies: Vec<SourceAssetId>,
    ) -> Result<()> {
        if let Some(relative_path) = self.by_id.get(&id).cloned() {
            if let Some(meta) = self.metas.get_mut(&id) {
                meta.dependencies = dependencies.clone();
                let source_path = self.assets_root.join(&relative_path);
                write_meta(&meta_path_for(&source_path), meta)?;
            }
        }

        self.dependency_graph.set_dependencies(id, &dependencies);
        Ok(())
    }

    /// Moves a source asset (and its `.meta.ron` sidecar, if any) from
    /// `old_relative_path` to `new_relative_path`, preserving its stable id
    /// so every existing reference by id keeps resolving correctly.
    pub fn rename(&mut self, old_relative_path: &str, new_relative_path: &str) -> Result<()> {
        let old_source = self.assets_root.join(old_relative_path);
        let new_source = self.assets_root.join(new_relative_path);

        if !old_source.is_file() {
            return Err(EngineError::AssetLoad {
                path: old_relative_path.to_owned(),
                reason: "source asset does not exist".to_owned(),
            });
        }
        if new_source.is_file() {
            return Err(EngineError::AssetLoad {
                path: new_relative_path.to_owned(),
                reason: "a file already exists at the destination".to_owned(),
            });
        }

        if let Some(parent) = new_source.parent() {
            fs::create_dir_all(parent).map_err(|error| EngineError::AssetLoad {
                path: new_relative_path.to_owned(),
                reason: format!("failed to create destination directory: {error}"),
            })?;
        }

        fs::rename(&old_source, &new_source).map_err(|error| EngineError::AssetLoad {
            path: old_relative_path.to_owned(),
            reason: format!("failed to move source asset: {error}"),
        })?;

        let old_meta_path = meta_path_for(&old_source);
        let new_meta_path = meta_path_for(&new_source);
        if old_meta_path.is_file() {
            fs::rename(&old_meta_path, &new_meta_path).map_err(|error| EngineError::AssetLoad {
                path: old_relative_path.to_owned(),
                reason: format!("failed to move asset metadata: {error}"),
            })?;
        }

        if let Some(id) = self.by_relative_path.remove(old_relative_path) {
            self.by_id.insert(id, new_relative_path.to_owned());
            self.by_relative_path
                .insert(new_relative_path.to_owned(), id);
        }

        Ok(())
    }

    /// Resolves a stable id to its current relative path, or `None` if the
    /// database has never seen (or has since lost track of) that id.
    pub fn resolve_relative_path(&self, id: SourceAssetId) -> Option<&str> {
        self.by_id.get(&id).map(String::as_str)
    }

    /// Resolves a relative path to the stable id currently associated with
    /// it, if it has been imported.
    pub fn resolve_id(&self, relative_path: &str) -> Option<SourceAssetId> {
        self.by_relative_path.get(relative_path).copied()
    }

    pub fn meta(&self, id: SourceAssetId) -> Option<&AssetMeta> {
        self.metas.get(&id)
    }

    /// The path of the cached, imported output for `id`, if it has been
    /// imported at least once.
    pub fn cache_entry_path(&self, id: SourceAssetId) -> Option<PathBuf> {
        let meta = self.metas.get(&id)?;
        Some(self.cache.entry_path(&meta.content_hash))
    }

    pub fn direct_dependents_of(&self, id: SourceAssetId) -> Vec<SourceAssetId> {
        self.dependency_graph.direct_dependents_of(id)
    }

    /// Every asset that would need reconsidering if `id` changed, directly
    /// or transitively. Callers (e.g. a future file watcher) use this to
    /// decide what else to re-import.
    pub fn invalidate(&self, id: SourceAssetId) -> Vec<SourceAssetId> {
        self.dependency_graph.transitive_dependents_of(id)
    }
}

fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }

    let entries = fs::read_dir(dir).map_err(|error| EngineError::AssetLoad {
        path: dir.display().to_string(),
        reason: format!("failed to read directory: {error}"),
    })?;

    for entry in entries {
        let entry = entry.map_err(|error| EngineError::AssetLoad {
            path: dir.display().to_string(),
            reason: format!("failed to read directory entry: {error}"),
        })?;
        let path = entry.path();

        let is_hidden = path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with('.'));
        if is_hidden {
            continue;
        }

        if path.is_dir() {
            collect_files(&path, out)?;
        } else {
            out.push(path);
        }
    }

    Ok(())
}

fn relative_string(root: &Path, path: &Path) -> Option<String> {
    let stripped = path.strip_prefix(root).ok()?;
    Some(stripped.to_string_lossy().replace('\\', "/"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct Scratch {
        assets_root: PathBuf,
        cache_root: PathBuf,
    }

    impl Scratch {
        fn new(prefix: &str) -> Self {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time should be valid")
                .as_nanos();
            let root = std::env::temp_dir().join(format!("{prefix}-{nanos}"));
            let assets_root = root.join("assets");
            let cache_root = root.join("cache");
            fs::create_dir_all(&assets_root).expect("assets dir should be created");
            Self {
                assets_root,
                cache_root,
            }
        }

        fn write_asset(&self, relative_path: &str, contents: &[u8]) -> PathBuf {
            let path = self.assets_root.join(relative_path);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).expect("parent dir should be created");
            }
            fs::write(&path, contents).expect("asset should be written");
            path
        }

        fn database(&self) -> AssetDatabase {
            AssetDatabase::open(&self.assets_root, &self.cache_root).expect("database should open")
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            if let Some(root) = self.assets_root.parent() {
                let _ = fs::remove_dir_all(root);
            }
        }
    }

    #[test]
    fn ensure_imported_assigns_a_stable_id_across_reimports() {
        let scratch = Scratch::new("starman-db-stable-id");
        scratch.write_asset("textures/rock.png", b"pixels-v1");
        let mut db = scratch.database();

        let first = db
            .ensure_imported("textures/rock.png")
            .expect("first import should succeed");
        let second = db
            .ensure_imported("textures/rock.png")
            .expect("second import of unchanged content should succeed");

        assert_eq!(first, second);
    }

    #[test]
    fn ensure_imported_is_deterministic_for_identical_content() {
        let scratch = Scratch::new("starman-db-deterministic-hash");
        scratch.write_asset("a.png", b"same-bytes");
        scratch.write_asset("b.png", b"same-bytes");
        let mut db = scratch.database();

        let id_a = db.ensure_imported("a.png").expect("a should import");
        let id_b = db.ensure_imported("b.png").expect("b should import");

        // Different assets still get distinct ids...
        assert_ne!(id_a, id_b);
        // ...but collapse onto the very same cache entry, since content
        // addressing is keyed by hash, not by source path.
        assert_eq!(
            db.cache_entry_path(id_a).unwrap(),
            db.cache_entry_path(id_b).unwrap()
        );
    }

    #[test]
    fn reimporting_after_content_change_keeps_the_same_id() {
        let scratch = Scratch::new("starman-db-reimport");
        let path = scratch.write_asset("textures/rock.png", b"pixels-v1");
        let mut db = scratch.database();

        let id = db
            .ensure_imported("textures/rock.png")
            .expect("import should succeed");
        fs::write(&path, b"pixels-v2").expect("asset should be rewritten");
        let id_after_change = db
            .ensure_imported("textures/rock.png")
            .expect("reimport should succeed");

        assert_eq!(id, id_after_change);
        let meta = db.meta(id).expect("meta should exist");
        assert_eq!(meta.content_hash, hash_bytes(b"pixels-v2"));
    }

    #[test]
    fn rename_preserves_id_and_reference_resolution() {
        let scratch = Scratch::new("starman-db-rename");
        scratch.write_asset("textures/rock.png", b"pixels");
        let mut db = scratch.database();

        let id = db
            .ensure_imported("textures/rock.png")
            .expect("import should succeed");

        db.rename("textures/rock.png", "textures/boulder.png")
            .expect("rename should succeed");

        assert_eq!(db.resolve_relative_path(id), Some("textures/boulder.png"));
        assert_eq!(db.resolve_id("textures/rock.png"), None);
        assert_eq!(db.resolve_id("textures/boulder.png"), Some(id));
        assert!(scratch.assets_root.join("textures/boulder.png").is_file());
        assert!(scratch
            .assets_root
            .join("textures/boulder.png.meta.ron")
            .is_file());
    }

    #[test]
    fn plain_filesystem_rename_is_recovered_by_content_hash_on_rescan() {
        let scratch = Scratch::new("starman-db-orphan-recovery");
        scratch.write_asset("textures/rock.png", b"pixels");
        let mut db = scratch.database();
        let id = db
            .ensure_imported("textures/rock.png")
            .expect("import should succeed");

        // Simulate a rename done outside the engine: the sidecar is left
        // behind, dangling, next to a source file that no longer exists.
        fs::rename(
            scratch.assets_root.join("textures/rock.png"),
            scratch.assets_root.join("textures/boulder.png"),
        )
        .expect("plain filesystem rename should succeed");

        db.rescan().expect("rescan should succeed");
        let recovered_id = db
            .ensure_imported("textures/boulder.png")
            .expect("re-import at the new path should succeed");

        assert_eq!(id, recovered_id);
    }

    #[test]
    fn dependents_are_reported_directly_and_transitively() {
        let scratch = Scratch::new("starman-db-dependents");
        scratch.write_asset("textures/rock.png", b"pixels");
        scratch.write_asset("materials/rock.ron", b"material");
        let mut db = scratch.database();

        let texture_id = db
            .ensure_imported("textures/rock.png")
            .expect("texture import should succeed");
        let material_id = db
            .ensure_imported("materials/rock.ron")
            .expect("material import should succeed");

        db.set_dependencies(material_id, vec![texture_id])
            .expect("dependency should be recorded");

        assert_eq!(db.direct_dependents_of(texture_id), vec![material_id]);
        assert_eq!(db.invalidate(texture_id), vec![material_id]);
    }
}
