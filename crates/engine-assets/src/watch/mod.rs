//! Asset file watching: turns raw filesystem events under a project's
//! source assets directory into debounced, classified [`AssetChange`]s.
//!
//! Flow: `notify` event -> debounce (bursts collapse per path) -> reimport
//! job on a worker thread (read, hash, content-addressed cache store; only
//! when a database cache is attached) -> commit on the polling thread
//! (`.meta`/index updates) -> `AssetChange`s for the path and for every
//! asset that depends on it. Polling never blocks.

mod debounce;
mod jobs;

use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;
use std::time::{Duration, Instant};

use notify::{EventKind, RecommendedWatcher, RecursiveMode, Watcher};

use crate::import::{
    hash_bytes, is_temp_file, AssetDatabase, ImportOutcome, ImportedCache, META_SUFFIX,
};
use crate::pathing::normalize_disk_path;
use debounce::Debouncer;
use jobs::{Job, JobPool};

/// Tuning knobs for [`AssetWatcher`].
#[derive(Debug, Clone, Copy)]
pub struct WatchConfig {
    /// How long a path must stay quiet before its change is processed.
    /// Reload latency is this window plus the reimport job time.
    pub quiet_window: Duration,
    pub worker_threads: usize,
}

impl Default for WatchConfig {
    fn default() -> Self {
        Self {
            quiet_window: Duration::from_millis(150),
            worker_threads: 2,
        }
    }
}

/// A classified change to a source asset. Paths are normalized absolute
/// disk paths (the same form `AssetServer` resolves handles to).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AssetChange {
    Texture(PathBuf),
    Mesh(PathBuf),
    /// Any `.ron` data file that is not a scene (materials today).
    Material(PathBuf),
    Scene(PathBuf),
    /// Lua (or other) script under a project scripts root.
    Script(PathBuf),
    /// Native plugin dynamic library (`.dll` / `.so` / `.dylib`).
    Plugin(PathBuf),
    Other(PathBuf),
    Removed(PathBuf),
}

impl AssetChange {
    pub fn path(&self) -> &Path {
        match self {
            Self::Texture(path)
            | Self::Mesh(path)
            | Self::Material(path)
            | Self::Scene(path)
            | Self::Script(path)
            | Self::Plugin(path)
            | Self::Other(path)
            | Self::Removed(path) => path,
        }
    }
}

fn classify(path: PathBuf) -> AssetChange {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();

    if name.ends_with(".scene.ron") {
        return AssetChange::Scene(path);
    }

    match extension.as_str() {
        "png" | "jpg" | "jpeg" => AssetChange::Texture(path),
        "glb" | "gltf" => AssetChange::Mesh(path),
        "ron" => AssetChange::Material(path),
        "lua" => AssetChange::Script(path),
        "dll" | "so" | "dylib" => AssetChange::Plugin(path),
        _ => AssetChange::Other(path),
    }
}

pub struct AssetWatcher {
    root: PathBuf,
    _watcher: Option<RecommendedWatcher>,
    events: Receiver<notify::Result<notify::Event>>,
    debouncer: Debouncer,
    config: WatchConfig,
    pool: Option<JobPool>,
}

impl AssetWatcher {
    /// Watches `root` recursively with a real OS watcher. If the watcher
    /// cannot be created (or `root` does not exist) the returned watcher is
    /// inert but harmless: `poll` simply never reports anything.
    pub fn watch_root(root: &Path, config: WatchConfig) -> Self {
        let (tx, rx) = mpsc::channel();
        let normalized_root = normalize_disk_path(root);

        let watcher = match notify::recommended_watcher(move |event| {
            let _ = tx.send(event);
        }) {
            Ok(mut watcher) => {
                if normalized_root.is_dir() {
                    if let Err(error) = watcher.watch(&normalized_root, RecursiveMode::Recursive) {
                        log::warn!(
                            target: "engine::assets",
                            "failed to watch '{}' for hot-reload: {}",
                            normalized_root.display(),
                            error
                        );
                    }
                }
                Some(watcher)
            }
            Err(error) => {
                log::warn!(
                    target: "engine::assets",
                    "asset hot-reload watcher unavailable: {error}"
                );
                None
            }
        };

        Self::from_parts(normalized_root, watcher, rx, config)
    }

    /// Builds a watcher fed by an arbitrary event source instead of the OS,
    /// so behavior can be tested deterministically.
    pub fn with_event_source(
        root: &Path,
        events: Receiver<notify::Result<notify::Event>>,
        config: WatchConfig,
    ) -> Self {
        Self::from_parts(normalize_disk_path(root), None, events, config)
    }

    fn from_parts(
        root: PathBuf,
        watcher: Option<RecommendedWatcher>,
        events: Receiver<notify::Result<notify::Event>>,
        config: WatchConfig,
    ) -> Self {
        Self {
            root,
            _watcher: watcher,
            events,
            debouncer: Debouncer::new(config.quiet_window),
            config,
            pool: None,
        }
    }

    /// Enables off-thread reimport: changed files are read, hashed and
    /// stored in `cache` by worker threads before being reported.
    pub(crate) fn set_cache(&mut self, cache: ImportedCache) {
        let work = move |job: &Job| -> Result<ImportOutcome, String> {
            let bytes = std::fs::read(&job.path).map_err(|error| error.to_string())?;
            let content_hash = hash_bytes(&bytes);
            cache
                .store(&content_hash, &bytes)
                .map_err(|error| error.to_string())?;
            Ok(ImportOutcome {
                path: job.path.clone(),
                relative_path: job.relative_path.clone(),
                content_hash,
            })
        };
        self.pool = Some(JobPool::new(self.config.worker_threads, Arc::new(work)));
    }

    /// Drains pending events and finished jobs without blocking. When a
    /// `database` is given, finished reimports are committed to it, changes
    /// whose content hash did not actually change are suppressed, and
    /// assets that depend on a changed one are reported as well.
    pub fn poll(&mut self, database: Option<&mut AssetDatabase>) -> Vec<AssetChange> {
        self.poll_at(Instant::now(), database)
    }

    pub fn poll_at(
        &mut self,
        now: Instant,
        mut database: Option<&mut AssetDatabase>,
    ) -> Vec<AssetChange> {
        self.drain_events(now);

        let mut changes = Vec::new();

        for path in self.debouncer.drain_ready(now) {
            if !path.is_file() {
                changes.push(AssetChange::Removed(path));
                continue;
            }

            match (&self.pool, self.relative_path(&path)) {
                (Some(pool), Some(relative_path)) => pool.submit(path, relative_path),
                _ => changes.push(classify(path)),
            }
        }

        let Some(pool) = &self.pool else {
            return changes;
        };

        for result in pool.poll_results() {
            let outcome = match result.outcome {
                Ok(outcome) => outcome,
                Err(reason) => {
                    if result.path.is_file() {
                        log::warn!(
                            target: "engine::assets",
                            "reimport of '{}' failed: {}",
                            result.path.display(),
                            reason
                        );
                    } else {
                        changes.push(AssetChange::Removed(result.path));
                    }
                    continue;
                }
            };

            let mut dependents = Vec::new();
            if let Some(database) = database.as_deref_mut() {
                match database.apply_import_outcome(&outcome) {
                    Ok(applied) if !applied.changed => continue,
                    Ok(applied) => dependents = applied.dependents,
                    Err(error) => {
                        log::warn!(
                            target: "engine::assets",
                            "failed to record reimport of '{}': {}",
                            outcome.relative_path,
                            error
                        );
                    }
                }
            }

            changes.push(classify(outcome.path));
            for relative in dependents {
                changes.push(classify(self.root.join(relative)));
            }
        }

        dedup_preserving_order(&mut changes);
        changes
    }

    fn drain_events(&mut self, now: Instant) {
        while let Ok(event) = self.events.try_recv() {
            let event = match event {
                Ok(event) => event,
                Err(error) => {
                    log::warn!(target: "engine::assets", "asset watch event error: {error}");
                    continue;
                }
            };

            if !matches!(
                event.kind,
                EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_)
            ) {
                continue;
            }

            for path in event.paths {
                let path = normalize_event_path(&path);
                if self.is_watchable(&path) {
                    self.debouncer.push(path, now);
                }
            }
        }
    }

    fn is_watchable(&self, path: &Path) -> bool {
        let Ok(relative) = path.strip_prefix(&self.root) else {
            return false;
        };
        if is_temp_file(path) || path.is_dir() {
            return false;
        }
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(META_SUFFIX))
        {
            return false;
        }

        !relative.components().any(|component| {
            component
                .as_os_str()
                .to_str()
                .is_some_and(|part| part.starts_with('.'))
        })
    }

    fn relative_path(&self, path: &Path) -> Option<String> {
        let relative = path.strip_prefix(&self.root).ok()?;
        Some(relative.to_string_lossy().replace('\\', "/"))
    }
}

/// Canonicalizes an event path even when the file no longer exists (a
/// removed file cannot be canonicalized directly, so its parent is).
fn normalize_event_path(path: &Path) -> PathBuf {
    if path.exists() {
        return normalize_disk_path(path);
    }

    match (path.parent(), path.file_name()) {
        (Some(parent), Some(name)) => normalize_disk_path(parent).join(name),
        _ => path.to_path_buf(),
    }
}

fn dedup_preserving_order(changes: &mut Vec<AssetChange>) {
    let mut seen = std::collections::HashSet::new();
    changes.retain(|change| seen.insert(change.clone()));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classification_is_by_extension_with_scene_special_case() {
        assert!(matches!(
            classify("a/rock.PNG".into()),
            AssetChange::Texture(_)
        ));
        assert!(matches!(
            classify("a/cube.glb".into()),
            AssetChange::Mesh(_)
        ));
        assert!(matches!(
            classify("a/default.ron".into()),
            AssetChange::Material(_)
        ));
        assert!(matches!(
            classify("a/level.scene.ron".into()),
            AssetChange::Scene(_)
        ));
        assert!(matches!(
            classify("a/notes.txt".into()),
            AssetChange::Other(_)
        ));
        assert!(matches!(
            classify("scripts/main.lua".into()),
            AssetChange::Script(_)
        ));
        assert!(matches!(
            classify("plugins/libexample.dylib".into()),
            AssetChange::Plugin(_)
        ));
    }
}
