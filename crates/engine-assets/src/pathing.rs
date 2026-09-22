use engine_core::{EngineError, Result};
use std::path::{Component, Path, PathBuf};

use crate::AssetPath;

pub(crate) fn resolve_disk_path(root: &AssetPath, relative_path: &str) -> Result<PathBuf> {
    if relative_path.trim().is_empty() {
        return Err(EngineError::AssetLoad {
            path: relative_path.to_owned(),
            reason: "path cannot be empty".to_owned(),
        });
    }

    let relative = Path::new(relative_path);
    if relative.is_absolute()
        || relative
            .components()
            .any(|component| matches!(component, Component::ParentDir | Component::RootDir))
    {
        return Err(EngineError::AssetLoad {
            path: relative_path.to_owned(),
            reason: "path must be root-relative without traversal segments".to_owned(),
        });
    }

    // Canonicalize the root once and join the (already-validated,
    // traversal-free) relative path onto it, rather than canonicalizing the
    // full joined path independently: the latter falls back to the raw,
    // un-resolved join whenever the leaf doesn't exist yet (a mesh that
    // hasn't been imported, a texture nobody has loaded yet, ...), which on
    // platforms where the root itself sits behind a symlink (e.g. macOS's
    // `/var` -> `/private/var`) makes a perfectly valid path look like it
    // "escaped" a root that resolved the symlink. Joining onto the
    // already-canonical root keeps the result consistently prefixed by it,
    // so no separate escape check is needed here.
    let normalized_root = normalize_disk_path(Path::new(root.as_str()));
    Ok(normalized_root.join(relative))
}

pub(crate) fn normalize_disk_path(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

pub(crate) fn to_asset_path(path: &Path) -> AssetPath {
    AssetPath::new(path.to_string_lossy().replace('\\', "/"))
}

pub(crate) fn to_relative_asset_path(root: &AssetPath, path: &AssetPath) -> Option<String> {
    let root_path = Path::new(root.as_str());
    let asset_path = Path::new(path.as_str());

    if !asset_path.is_absolute() {
        if let Ok(stripped) = asset_path.strip_prefix(root_path) {
            let relative = stripped.to_string_lossy().replace('\\', "/");
            return Some(relative.trim_start_matches('/').to_owned());
        }

        return Some(asset_path.to_string_lossy().replace('\\', "/"));
    }

    let normalized_root = normalize_disk_path(root_path);
    let normalized_asset = normalize_disk_path(asset_path);
    let stripped = normalized_asset.strip_prefix(&normalized_root).ok()?;
    let relative = stripped.to_string_lossy().replace('\\', "/");
    Some(relative.trim_start_matches('/').to_owned())
}
