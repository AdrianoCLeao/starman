//! A content-addressed cache of imported asset output. For M1 this is a
//! deliberately simple pass-through (the "processed" output is just the
//! source bytes, addressed by their content hash) — real per-type
//! processing (texture decoding, mesh baking, ...) is layered on top of the
//! same addressing scheme in a later milestone.

use std::fs;
use std::path::PathBuf;

use engine_core::{EngineError, Result};

use super::atomic::write_atomic;

#[derive(Clone)]
pub(crate) struct ImportedCache {
    root: PathBuf,
}

impl ImportedCache {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The path a cache entry for `hash` lives (or would live) at, sharded
    /// by the first two hex characters so no single directory accumulates
    /// every entry.
    pub fn entry_path(&self, hash: &str) -> PathBuf {
        let shard = hash.get(0..2).unwrap_or("00");
        self.root.join(shard).join(hash)
    }

    pub fn contains(&self, hash: &str) -> bool {
        self.entry_path(hash).is_file()
    }

    /// Stores `bytes` under `hash`, if not already present. Content
    /// addressing makes this idempotent: writing the same hash twice is a
    /// no-op after the first time, and two different source paths with
    /// identical content collapse onto the same cache entry.
    pub fn store(&self, hash: &str, bytes: &[u8]) -> Result<PathBuf> {
        let entry_path = self.entry_path(hash);
        if self.contains(hash) {
            return Ok(entry_path);
        }

        if let Some(parent) = entry_path.parent() {
            fs::create_dir_all(parent).map_err(|error| EngineError::AssetLoad {
                path: entry_path.display().to_string(),
                reason: format!("failed to create import cache directory: {error}"),
            })?;
        }

        write_atomic(&entry_path, bytes).map_err(|error| EngineError::AssetLoad {
            path: entry_path.display().to_string(),
            reason: format!("failed to write import cache entry: {error}"),
        })?;

        Ok(entry_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn scratch_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be valid")
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{nanos}"))
    }

    #[test]
    fn store_is_idempotent_and_content_addressed() {
        let root = scratch_dir("starman-cache");
        let cache = ImportedCache::new(&root);

        let hash = "abcd1234";
        let first = cache
            .store(hash, b"hello")
            .expect("first store should succeed");
        let second = cache
            .store(hash, b"hello")
            .expect("second store should succeed");

        assert_eq!(first, second);
        assert!(cache.contains(hash));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn entries_are_sharded_by_hash_prefix() {
        let root = scratch_dir("starman-cache-shard");
        let cache = ImportedCache::new(&root);

        let path = cache.entry_path("abcd1234");
        assert!(path.starts_with(root.join("ab")));

        let _ = fs::remove_dir_all(&root);
    }
}
