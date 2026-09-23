//! Crash-safe autosave and recovery for open scene documents.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use engine_assets::write_scene_ron;
use engine_assets::SceneFile;
use engine_core::{EngineError, Result};
use engine_project::Project;

const AUTOSAVE_INTERVAL: Duration = Duration::from_secs(30);

pub struct AutosaveState {
    pub last_save: Instant,
    pub enabled: bool,
}

impl Default for AutosaveState {
    fn default() -> Self {
        Self {
            last_save: Instant::now(),
            enabled: true,
        }
    }
}

impl AutosaveState {
    pub fn due(&self) -> bool {
        self.enabled && self.last_save.elapsed() >= AUTOSAVE_INTERVAL
    }

    pub fn mark_saved(&mut self) {
        self.last_save = Instant::now();
    }
}

/// Directory: `<project>/.starman/autosave/`
pub fn autosave_dir(project: &Project) -> PathBuf {
    project.paths.generated_dir().join("autosave")
}

pub fn autosave_path_for_scene(project: &Project, scene_path: &Path) -> PathBuf {
    let stem = scene_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("scene");
    let id = scene_path
        .parent()
        .map(|p| {
            p.strip_prefix(project.paths.assets_dir())
                .unwrap_or(p)
                .to_string_lossy()
                .replace(['/', '\\'], "__")
        })
        .unwrap_or_default();
    let name = if id.is_empty() {
        format!("{stem}.scene.ron")
    } else {
        format!("{id}__{stem}.scene.ron")
    };
    autosave_dir(project).join(name)
}

pub fn write_autosave(project: &Project, scene_path: &Path, scene: &SceneFile) -> Result<PathBuf> {
    let dir = autosave_dir(project);
    fs::create_dir_all(&dir).map_err(|error| EngineError::AssetLoad {
        path: dir.display().to_string(),
        reason: error.to_string(),
    })?;
    let path = autosave_path_for_scene(project, scene_path);
    write_scene_ron(&path, scene)?;
    // Dirty marker so recovery can detect an unclean shutdown.
    let marker = dir.join("dirty.journal");
    fs::write(&marker, scene_path.display().to_string()).map_err(|error| {
        EngineError::AssetLoad {
            path: marker.display().to_string(),
            reason: error.to_string(),
        }
    })?;
    Ok(path)
}

pub fn clear_dirty_marker(project: &Project) {
    let marker = autosave_dir(project).join("dirty.journal");
    let _ = fs::remove_file(marker);
}

/// If a dirty journal exists and an autosave is newer than the scene file,
/// returns `(autosave_path, original_scene_path)`.
pub fn pending_recovery(project: &Project) -> Option<(PathBuf, PathBuf)> {
    let dir = autosave_dir(project);
    let marker = dir.join("dirty.journal");
    let original = PathBuf::from(fs::read_to_string(&marker).ok()?);
    let autosave = autosave_path_for_scene(project, &original);
    if !autosave.is_file() {
        return None;
    }
    let autosave_mtime = fs::metadata(&autosave).ok()?.modified().ok()?;
    let original_mtime = fs::metadata(&original)
        .ok()
        .and_then(|m| m.modified().ok());
    if original_mtime.is_some_and(|m| m >= autosave_mtime) {
        return None;
    }
    Some((autosave, original))
}
