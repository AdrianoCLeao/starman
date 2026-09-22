//! The `.meta.ron` sidecar recorded next to every imported source asset:
//! its stable id, the importer used, its declared dependencies, and the
//! content hash it had at the last successful import.

use std::fs;
use std::path::{Path, PathBuf};

use engine_core::{EngineError, Result, SourceAssetId, SubAssetId};

use super::atomic::write_atomic;
use serde::{Deserialize, Serialize};

pub const META_SUFFIX: &str = ".meta.ron";

/// One sub-resource within a source asset — e.g. one mesh within a
/// multi-mesh glTF file — addressed by its own stable [`SubAssetId`]
/// (ADR 0002: derived from the parent id plus a stable importer key).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct SubAssetRecord {
    pub id: SubAssetId,
    /// The stable key the id was derived from (e.g. `"mesh:0"`).
    pub key: String,
    /// A human-readable label from the source content, if any (e.g. the
    /// glTF mesh's own name).
    pub label: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct AssetMeta {
    pub version: u32,
    pub id: SourceAssetId,
    pub importer: String,
    #[serde(default)]
    pub dependencies: Vec<SourceAssetId>,
    pub content_hash: String,
    /// Sub-resources within this asset (e.g. one entry per mesh in a
    /// multi-mesh glTF file). Empty for single-resource asset types
    /// (textures, materials) and for older `.meta.ron` files written before
    /// this field existed — additive, no version bump needed.
    #[serde(default)]
    pub sub_assets: Vec<SubAssetRecord>,
}

impl AssetMeta {
    pub const CURRENT_VERSION: u32 = 1;
}

/// The `.meta.ron` path that sits next to `source_path`.
pub(crate) fn meta_path_for(source_path: &Path) -> PathBuf {
    let mut meta = source_path.as_os_str().to_owned();
    meta.push(META_SUFFIX);
    PathBuf::from(meta)
}

/// Recovers the source path a `.meta.ron` file describes, or `None` if
/// `path` isn't a meta file at all.
pub(crate) fn source_path_for_meta(meta_path: &Path) -> Option<PathBuf> {
    let raw = meta_path.as_os_str().to_str()?;
    raw.strip_suffix(META_SUFFIX).map(PathBuf::from)
}

pub(crate) fn read_meta(meta_path: &Path) -> Result<Option<AssetMeta>> {
    if !meta_path.is_file() {
        return Ok(None);
    }

    let source = fs::read_to_string(meta_path).map_err(|error| EngineError::AssetLoad {
        path: meta_path.display().to_string(),
        reason: format!("failed to read asset metadata: {error}"),
    })?;

    let meta: AssetMeta = ron::from_str(&source).map_err(|error| EngineError::AssetLoad {
        path: meta_path.display().to_string(),
        reason: format!("failed to parse asset metadata: {error}"),
    })?;

    if meta.version != AssetMeta::CURRENT_VERSION {
        return Err(EngineError::AssetLoad {
            path: meta_path.display().to_string(),
            reason: format!(
                "unsupported asset metadata version {}; expected {}",
                meta.version,
                AssetMeta::CURRENT_VERSION
            ),
        });
    }

    Ok(Some(meta))
}

pub(crate) fn write_meta(meta_path: &Path, meta: &AssetMeta) -> Result<()> {
    let pretty = ron::ser::PrettyConfig::default();
    let serialized =
        ron::ser::to_string_pretty(meta, pretty).map_err(|error| EngineError::AssetLoad {
            path: meta_path.display().to_string(),
            reason: format!("failed to serialize asset metadata: {error}"),
        })?;

    write_atomic(meta_path, serialized.as_bytes()).map_err(|error| EngineError::AssetLoad {
        path: meta_path.display().to_string(),
        reason: format!("failed to write asset metadata: {error}"),
    })
}

/// Guesses an importer key from a source asset's extension. This is a
/// placeholder classification — a real importer registry (ADR 0003) comes
/// later; for now it only needs to be stable so re-importing the same file
/// doesn't spuriously look "changed".
pub(crate) fn default_importer_key_for(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("png") | Some("jpg") | Some("jpeg") => "texture",
        Some("glb") | Some("gltf") => "mesh",
        Some("ron") => "ron-data",
        Some("ogg") | Some("wav") | Some("mp3") => "audio",
        _ => "opaque",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meta_path_appends_suffix() {
        let path = meta_path_for(Path::new("textures/rock.png"));
        assert_eq!(path, PathBuf::from("textures/rock.png.meta.ron"));
    }

    #[test]
    fn source_path_for_meta_strips_suffix() {
        let source = source_path_for_meta(Path::new("textures/rock.png.meta.ron"));
        assert_eq!(source, Some(PathBuf::from("textures/rock.png")));
    }

    #[test]
    fn source_path_for_meta_rejects_non_meta_paths() {
        assert_eq!(source_path_for_meta(Path::new("textures/rock.png")), None);
    }

    #[test]
    fn importer_key_is_derived_from_extension() {
        assert_eq!(default_importer_key_for(Path::new("a.png")), "texture");
        assert_eq!(default_importer_key_for(Path::new("a.glb")), "mesh");
        assert_eq!(default_importer_key_for(Path::new("a.ron")), "ron-data");
        assert_eq!(default_importer_key_for(Path::new("a.xyz")), "opaque");
    }
}
