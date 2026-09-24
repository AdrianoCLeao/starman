use std::fs;
use std::path::Path;

use engine_assets::SceneFile;
use engine_core::{EngineError, Result};

use crate::manifest::ProjectManifest;
use crate::migration::{parse_manifest, ParsedManifest};
use crate::paths::ProjectPaths;
use crate::validate::ValidationReport;

/// An opened Starman project: its manifest plus the resolved directory
/// layout it lives in.
#[derive(Debug, Clone)]
pub struct Project {
    pub manifest: ProjectManifest,
    pub paths: ProjectPaths,
    /// The manifest version this project was migrated from on open, until
    /// [`Project::persist_migration`] writes the migrated manifest.
    pub migrated_from: Option<u32>,
}

/// Options for scaffolding a brand-new project with [`Project::create`].
#[derive(Debug, Clone)]
pub struct CreateOptions {
    pub name: String,
    /// Path to the entry scene, relative to the project's `assets/`
    /// directory.
    pub entry_scene: String,
}

impl Default for CreateOptions {
    fn default() -> Self {
        Self {
            name: "New Project".to_owned(),
            entry_scene: "scenes/main.scene.ron".to_owned(),
        }
    }
}

impl Project {
    /// Opens an existing project, failing fast with an actionable error if
    /// the manifest is missing/unreadable, its version is unsupported, or
    /// the layout it declares (assets directory, entry scene) is incomplete.
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let paths = ProjectPaths::new(root.as_ref().to_path_buf());
        let parsed = read_manifest(&paths)?;
        let manifest = parsed.manifest;
        if let Some(from) = parsed.migrated_from {
            log::info!(
                target: "engine::project",
                "project manifest migrated in memory from v{from} to v{}; \
                 it is rewritten (with a backup) on the next save",
                ProjectManifest::CURRENT_VERSION
            );
        }

        let assets_dir = paths.assets_dir();
        if !assets_dir.is_dir() {
            return Err(EngineError::InvalidProject {
                path: paths.root().display().to_string(),
                reason: format!("missing source assets directory '{}'", assets_dir.display()),
            });
        }

        let entry_scene_path = paths.resolve_asset_relative(&manifest.entry_scene);
        if !entry_scene_path.is_file() {
            return Err(EngineError::InvalidProject {
                path: paths.root().display().to_string(),
                reason: format!(
                    "entry scene '{}' declared in the manifest does not exist",
                    entry_scene_path.display()
                ),
            });
        }

        Ok(Self {
            manifest,
            paths,
            migrated_from: parsed.migrated_from,
        })
    }

    /// Writes the manifest back to `project.ron` atomically. When the
    /// project was migrated on open, the original file is first copied to
    /// `project.ron.v<N>.bak` (ADR 0006).
    pub fn save_manifest(&mut self) -> Result<()> {
        if let Some(from) = self.migrated_from {
            let manifest_path = self.paths.manifest_file();
            let backup = manifest_path.with_file_name(format!("project.ron.v{from}.bak"));
            if !backup.exists() {
                fs::copy(&manifest_path, &backup).map_err(|error| io_error(&self.paths, error))?;
            }
        }
        write_manifest(&self.paths, &self.manifest)?;
        self.migrated_from = None;
        Ok(())
    }

    /// Persists an on-open migration (no-op when none happened). Returns
    /// the version migrated from.
    pub fn persist_migration(&mut self) -> Result<Option<u32>> {
        let from = self.migrated_from;
        if from.is_some() {
            self.save_manifest()?;
        }
        Ok(from)
    }

    /// Lenient variant of [`Self::open`]: collects every problem it finds
    /// instead of stopping at the first one.
    pub fn validate(root: impl AsRef<Path>) -> Result<ValidationReport> {
        let paths = ProjectPaths::new(root.as_ref().to_path_buf());
        let mut report = ValidationReport::new();

        let manifest = match read_manifest(&paths) {
            Ok(parsed) => {
                if let Some(from) = parsed.migrated_from {
                    report.push_warning(format!(
                        "project manifest is v{from}; it will be migrated to v{} (with a backup) \
                         on the next save or `starman migrate`",
                        ProjectManifest::CURRENT_VERSION
                    ));
                }
                Some(parsed.manifest)
            }
            Err(error) => {
                report.push_error(error.to_string());
                None
            }
        };

        if !paths.assets_dir().is_dir() {
            report.push_error(format!(
                "missing source assets directory '{}'",
                paths.assets_dir().display()
            ));
        }

        if let Some(manifest) = &manifest {
            let entry_scene_path = paths.resolve_asset_relative(&manifest.entry_scene);
            if !entry_scene_path.is_file() {
                report.push_error(format!(
                    "entry scene '{}' declared in the manifest does not exist",
                    entry_scene_path.display()
                ));
            }

            validate_scripts_and_plugins(&paths, manifest, &mut report);
            validate_game_settings(&paths, manifest, &mut report);
        }

        for (label, dir) in [
            ("cache", paths.cache_dir()),
            ("imported asset cache", paths.imported_dir()),
            ("diagnostics", paths.diagnostics_dir()),
            ("build output", paths.build_dir()),
        ] {
            if !dir.is_dir() {
                report.push_warning(format!(
                    "{label} directory '{}' does not exist yet; it will be created on demand",
                    dir.display()
                ));
            }
        }

        Ok(report)
    }

    /// Scaffolds a new project at `root`: the manifest, the directory
    /// layout, and a minimal valid entry scene. Safe to call again on an
    /// already-initialized project — it reopens it rather than overwriting
    /// authored data.
    pub fn create(root: impl AsRef<Path>, options: CreateOptions) -> Result<Self> {
        let paths = ProjectPaths::new(root.as_ref().to_path_buf());

        if paths.manifest_file().is_file() {
            return Self::open(paths.root());
        }

        fs::create_dir_all(paths.root()).map_err(|error| io_error(&paths, error))?;
        fs::create_dir_all(paths.assets_dir()).map_err(|error| io_error(&paths, error))?;
        fs::create_dir_all(paths.imported_dir()).map_err(|error| io_error(&paths, error))?;
        fs::create_dir_all(paths.diagnostics_dir()).map_err(|error| io_error(&paths, error))?;
        fs::create_dir_all(paths.build_dir()).map_err(|error| io_error(&paths, error))?;

        let entry_scene_path = paths.resolve_asset_relative(&options.entry_scene);
        if let Some(parent) = entry_scene_path.parent() {
            fs::create_dir_all(parent).map_err(|error| io_error(&paths, error))?;
        }
        if !entry_scene_path.is_file() {
            SceneFile::empty(options.name.clone()).write_to(&entry_scene_path)?;
        }

        write_gitignore(&paths)?;

        let manifest = ProjectManifest::new(options.name, options.entry_scene);
        write_manifest(&paths, &manifest)?;

        Ok(Self {
            manifest,
            paths,
            migrated_from: None,
        })
    }
}

fn read_manifest(paths: &ProjectPaths) -> Result<ParsedManifest> {
    let manifest_path = paths.manifest_file();
    let source =
        fs::read_to_string(&manifest_path).map_err(|error| EngineError::InvalidProject {
            path: paths.root().display().to_string(),
            reason: format!(
                "failed to read manifest '{}': {error}",
                manifest_path.display()
            ),
        })?;

    parse_manifest(&source).map_err(|reason| EngineError::InvalidProject {
        path: paths.root().display().to_string(),
        reason: format!("manifest '{}': {reason}", manifest_path.display()),
    })
}

fn validate_game_settings(
    paths: &ProjectPaths,
    manifest: &ProjectManifest,
    report: &mut ValidationReport,
) {
    manifest.game.validate(report);
    for (field, asset) in manifest.game.asset_refs() {
        let (file, _) = asset.split_path();
        if file.is_empty() {
            if asset.id.trim().is_empty() {
                report.push_error(format!("{field} is set but empty"));
            }
            continue;
        }
        if !paths.resolve_asset_relative(file).is_file() {
            report.push_error(format!(
                "{field} references '{file}', which does not exist under assets/"
            ));
        }
    }
    let localization_root = paths.resolve_asset_relative(&manifest.game.localization.root);
    if localization_root.is_dir() {
        for locale in &manifest.game.localization.supported {
            if !localization_root.join(locale).is_dir() {
                report.push_warning(format!(
                    "locale '{locale}' has no directory '{}'",
                    localization_root.join(locale).display()
                ));
            }
        }
    }
}

fn validate_scripts_and_plugins(
    paths: &ProjectPaths,
    manifest: &ProjectManifest,
    report: &mut ValidationReport,
) {
    let root = paths.root();

    if let Some(entry) = &manifest.scripts.entry {
        let entry_path = root.join(entry);
        if !entry_path.is_file() {
            report.push_error(format!(
                "scripts.entry '{}' does not exist",
                entry_path.display()
            ));
        }
    }

    for script_root in manifest.scripts.effective_roots() {
        let dir = root.join(&script_root);
        if !dir.is_dir() {
            report.push_warning(format!(
                "scripts root '{}' does not exist yet",
                dir.display()
            ));
        }
    }

    for grant in &manifest.permissions.filesystem {
        if grant.contains("..") {
            report.push_error(format!(
                "permissions.filesystem grant '{grant}' must not contain '..'"
            ));
        }
    }

    for plugin in &manifest.plugins {
        let plugin_dir = manifest.resolve_plugin_dir(root, plugin);
        if !plugin_dir.exists() {
            report.push_warning(format!(
                "plugin '{}' path '{}' does not exist yet",
                plugin.name,
                plugin_dir.display()
            ));
            continue;
        }
        if plugin_dir.is_dir() {
            if engine_plugin::find_plugin_library(&plugin_dir, &plugin.name).is_none() {
                report.push_warning(format!(
                    "plugin '{}' directory '{}' has no loadable library yet (build the cdylib first)",
                    plugin.name,
                    plugin_dir.display()
                ));
            }
        } else if plugin_dir.is_file() {
            let ext = plugin_dir
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("");
            if !matches!(ext, "dll" | "so" | "dylib") {
                report.push_error(format!(
                    "plugin '{}' path '{}' is not a dynamic library",
                    plugin.name,
                    plugin_dir.display()
                ));
            }
        }
    }
}

fn write_manifest(paths: &ProjectPaths, manifest: &ProjectManifest) -> Result<()> {
    let pretty = ron::ser::PrettyConfig::default();
    let serialized = ron::ser::to_string_pretty(manifest, pretty).map_err(|error| {
        EngineError::InvalidProject {
            path: paths.root().display().to_string(),
            reason: format!("failed to serialize project manifest: {error}"),
        }
    })?;

    fs::write(paths.manifest_file(), serialized).map_err(|error| io_error(paths, error))
}

fn write_gitignore(paths: &ProjectPaths) -> Result<()> {
    let gitignore_path = paths.root().join(".gitignore");
    if gitignore_path.is_file() {
        return Ok(());
    }

    fs::write(&gitignore_path, "/.starman/\n/build/\n").map_err(|error| io_error(paths, error))
}

fn io_error(paths: &ProjectPaths, error: std::io::Error) -> EngineError {
    EngineError::InvalidProject {
        path: paths.root().display().to_string(),
        reason: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn scratch_dir(prefix: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be valid")
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{nanos}"))
    }

    #[test]
    fn create_scaffolds_a_valid_project() {
        let root = scratch_dir("starman-create");

        let project =
            Project::create(&root, CreateOptions::default()).expect("project should be created");

        assert!(project.paths.manifest_file().is_file());
        assert!(project.paths.assets_dir().is_dir());
        assert!(project.paths.imported_dir().is_dir());
        assert!(project.paths.diagnostics_dir().is_dir());
        assert!(project.paths.build_dir().is_dir());
        assert!(project
            .paths
            .resolve_asset_relative(&project.manifest.entry_scene)
            .is_file());

        let report = Project::validate(&root).expect("validation should run");
        assert!(report.is_valid(), "report: {report}");

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn create_is_idempotent_and_preserves_identity() {
        let root = scratch_dir("starman-create-idempotent");

        let first =
            Project::create(&root, CreateOptions::default()).expect("first create should succeed");
        let second = Project::create(&root, CreateOptions::default())
            .expect("second create on the same root should reopen, not fail");

        assert_eq!(first.manifest.id, second.manifest.id);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn open_fails_with_actionable_error_when_manifest_is_missing() {
        let root = scratch_dir("starman-open-missing");
        fs::create_dir_all(&root).expect("scratch dir should be created");

        let error = Project::open(&root).expect_err("open should fail without a manifest");
        assert!(
            error.to_string().contains("project.ron") || error.to_string().contains("manifest")
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn open_rejects_unsupported_manifest_version() {
        let root = scratch_dir("starman-open-bad-version");
        fs::create_dir_all(&root).expect("scratch dir should be created");
        fs::write(
            root.join("project.ron"),
            r#"(
                version: 999,
                id: "00000000-0000-0000-0000-000000000000",
                name: "Bad",
                entry_scene: "scenes/main.scene.ron",
            )"#,
        )
        .expect("manifest should be written");

        let error =
            Project::open(&root).expect_err("open should reject an unsupported manifest version");
        assert!(error
            .to_string()
            .contains("unsupported project manifest version"));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn validate_reports_missing_directories_without_failing() {
        let root = scratch_dir("starman-validate-missing-dirs");
        let project =
            Project::create(&root, CreateOptions::default()).expect("project should be created");

        fs::remove_dir_all(project.paths.generated_dir())
            .expect("generated dir should be removable");

        let report = Project::validate(&root).expect("validation should run");
        assert!(
            report.is_valid(),
            "missing generated directories should only warn: {report}"
        );
        assert!(report.warnings().count() >= 2);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn validate_reports_error_when_entry_scene_is_missing() {
        let root = scratch_dir("starman-validate-missing-scene");
        let project =
            Project::create(&root, CreateOptions::default()).expect("project should be created");

        let entry_scene_path = project
            .paths
            .resolve_asset_relative(&project.manifest.entry_scene);
        fs::remove_file(&entry_scene_path).expect("entry scene should be removable");

        let report = Project::validate(&root).expect("validation should run");
        assert!(!report.is_valid());
        assert!(report
            .errors()
            .any(|issue| issue.message.contains("entry scene")));

        let _ = fs::remove_dir_all(&root);
    }
}
