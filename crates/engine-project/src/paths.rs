//! The directory layout of a Starman project, per ADR 0001: authored
//! source assets, generated import cache, local diagnostics, and build
//! output are kept in clearly separated locations so tooling never confuses
//! authored data with something it can safely regenerate.

use std::path::{Path, PathBuf};

/// Resolves the well-known locations inside a project root. This type does
/// not touch the filesystem by itself; callers decide when to create or
/// check for these paths (see [`crate::Project::create`] and
/// [`crate::Project::validate`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectPaths {
    root: PathBuf,
}

impl ProjectPaths {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The project manifest file.
    pub fn manifest_file(&self) -> PathBuf {
        self.root.join("project.ron")
    }

    /// Authored, source-controlled assets: meshes, textures, scenes,
    /// materials, etc. Never written to by generated tooling.
    pub fn assets_dir(&self) -> PathBuf {
        self.root.join("assets")
    }

    /// Root of all engine-generated, disposable local state: import cache
    /// and diagnostics. Safe to delete entirely; the engine recreates it.
    pub fn generated_dir(&self) -> PathBuf {
        self.root.join(".starman")
    }

    /// General-purpose cache (import cache, thumbnails, etc.).
    pub fn cache_dir(&self) -> PathBuf {
        self.generated_dir().join("cache")
    }

    /// Content-addressed cache of imported (processed) asset outputs.
    pub fn imported_dir(&self) -> PathBuf {
        self.cache_dir().join("imported")
    }

    /// Local, per-machine diagnostics: logs and crash reports.
    pub fn diagnostics_dir(&self) -> PathBuf {
        self.generated_dir().join("diagnostics")
    }

    /// Packaged build output (development and release profiles).
    pub fn build_dir(&self) -> PathBuf {
        self.root.join("build")
    }

    /// Save games written while developing (editor play, `starman run`).
    /// Distributed builds write to [`user_data_dir`] instead.
    pub fn saves_dir(&self) -> PathBuf {
        self.root.join("saves")
    }

    /// Per-user settings while developing (input rebinding, volumes); kept
    /// under the generated directory so they are never committed.
    pub fn dev_user_settings_dir(&self) -> PathBuf {
        self.generated_dir().join("user")
    }

    /// Resolves a scene path declared in the manifest (relative to
    /// [`Self::assets_dir`]) to an absolute path.
    pub fn resolve_asset_relative(&self, relative: &str) -> PathBuf {
        self.assets_dir().join(relative)
    }
}

/// The per-user, per-project data directory of a *distributed* game:
/// `<OS data dir>/Starman/<project-slug>-<id prefix>/`. Saves live in
/// `saves/` and user settings in `settings/` below it.
pub fn user_data_dir(project_name: &str, project_id: &str) -> Option<PathBuf> {
    let slug: String = project_name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>()
        .split('-')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("-");
    let prefix: String = project_id.chars().filter(|c| *c != '-').take(8).collect();
    let slug = if slug.is_empty() {
        "game".to_owned()
    } else {
        slug
    };
    dirs::data_dir().map(|dir| dir.join("Starman").join(format!("{slug}-{prefix}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_are_rooted_under_the_project_directory() {
        let paths = ProjectPaths::new("/tmp/my-project");
        assert_eq!(
            paths.manifest_file(),
            PathBuf::from("/tmp/my-project/project.ron")
        );
        assert_eq!(paths.assets_dir(), PathBuf::from("/tmp/my-project/assets"));
        assert_eq!(
            paths.imported_dir(),
            PathBuf::from("/tmp/my-project/.starman/cache/imported")
        );
        assert_eq!(
            paths.diagnostics_dir(),
            PathBuf::from("/tmp/my-project/.starman/diagnostics")
        );
        assert_eq!(paths.build_dir(), PathBuf::from("/tmp/my-project/build"));
    }

    #[test]
    fn user_data_dir_is_slugged_and_stable() {
        if let Some(dir) = user_data_dir("My Cool Game!", "bedc6c7d-6817-47c9") {
            assert!(
                dir.ends_with("Starman/my-cool-game-bedc6c7d"),
                "{}",
                dir.display()
            );
        }
        let paths = ProjectPaths::new("/tmp/my-project");
        assert_eq!(paths.saves_dir(), PathBuf::from("/tmp/my-project/saves"));
    }

    #[test]
    fn resolve_asset_relative_joins_under_assets_dir() {
        let paths = ProjectPaths::new("/tmp/my-project");
        assert_eq!(
            paths.resolve_asset_relative("scenes/main.scene.ron"),
            PathBuf::from("/tmp/my-project/assets/scenes/main.scene.ron")
        );
    }
}
