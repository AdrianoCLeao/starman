//! Copy a plugin library to a generation-specific cache before loading.

use std::fs;
use std::path::{Path, PathBuf};

use engine_core::{EngineError, Result};

/// Copies `source` into `cache_root/<name>-<generation>/<filename>` and
/// returns the destination path. Callers load from the copy so the original
/// can be overwritten by a rebuild.
pub fn shadow_copy_library(
    source: &Path,
    cache_root: &Path,
    name: &str,
    generation: u64,
) -> Result<PathBuf> {
    if !source.is_file() {
        return Err(EngineError::AssetLoad {
            path: source.display().to_string(),
            reason: "plugin library file does not exist".to_owned(),
        });
    }

    let file_name = source
        .file_name()
        .ok_or_else(|| EngineError::AssetLoad {
            path: source.display().to_string(),
            reason: "plugin path has no file name".to_owned(),
        })?;

    let dest_dir = cache_root.join(format!("{name}-{generation}"));
    fs::create_dir_all(&dest_dir).map_err(|error| EngineError::AssetLoad {
        path: dest_dir.display().to_string(),
        reason: error.to_string(),
    })?;

    let dest = dest_dir.join(file_name);
    fs::copy(source, &dest).map_err(|error| EngineError::AssetLoad {
        path: dest.display().to_string(),
        reason: format!("shadow-copy failed: {error}"),
    })?;

    Ok(dest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn copies_into_generation_dir() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("starman-shadow-{nanos}"));
        let src_dir = root.join("src");
        fs::create_dir_all(&src_dir).unwrap();
        let src = src_dir.join("plugin.so");
        fs::write(&src, b"fake").unwrap();
        let cache = root.join("cache");
        let dest = shadow_copy_library(&src, &cache, "demo", 3).unwrap();
        assert!(dest.is_file());
        assert!(dest.to_string_lossy().contains("demo-3"));
        let _ = fs::remove_dir_all(&root);
    }
}
