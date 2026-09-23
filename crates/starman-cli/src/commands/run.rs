use std::path::{Path, PathBuf};

use engine_assets::AssetDatabase;
use engine_core::Result as EngineResult;
use engine_project::Project;
use engine_runner::RunnerOptions;

use crate::error::CliError;

/// Opens the project, resolves its entry scene, attaches an asset database
/// (so hot-reload during the run keeps `.meta`/cache current), and runs it
/// windowed.
pub fn run(path: PathBuf) -> Result<(), CliError> {
    run_with(path, engine_runner::run_scene_windowed)
}

/// Same as [`run`], but the actual windowed run is delegated to `run_fn` —
/// production always passes [`engine_runner::run_scene_windowed`]; tests
/// pass a stub so project/path resolution is exercised without ever opening
/// a real window.
pub(crate) fn run_with(
    path: PathBuf,
    run_fn: impl FnOnce(&str, &Path, RunnerOptions) -> EngineResult<()>,
) -> Result<(), CliError> {
    let engine_error = |source| CliError::Engine {
        path: path.clone(),
        source,
    };

    let project = Project::open(&path).map_err(engine_error)?;
    let assets_root = project.paths.assets_dir().to_string_lossy().into_owned();
    let scene_path = project
        .paths
        .resolve_asset_relative(&project.manifest.entry_scene);
    let database = AssetDatabase::open(project.paths.assets_dir(), project.paths.imported_dir())
        .map_err(engine_error)?;

    let options = RunnerOptions::new(project.manifest.name.clone())
        .with_database(database)
        .with_project_root(project.paths.root());
    run_fn(&assets_root, &scene_path, options).map_err(engine_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_project::CreateOptions;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn scratch_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be valid")
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{nanos}"))
    }

    /// Confirms project/path resolution succeeds without ever invoking a
    /// real windowed run.
    #[test]
    fn resolves_project_and_scene_before_handing_off_to_the_runner() {
        let root = scratch_dir("starman-cli-run-resolve");
        Project::create(&root, CreateOptions::default()).expect("project should be created");

        let mut received: Option<(String, PathBuf)> = None;
        let result = run_with(root.clone(), |assets_root, scene_path, _options| {
            received = Some((assets_root.to_owned(), scene_path.to_path_buf()));
            Ok(())
        });

        assert!(result.is_ok());
        let (assets_root, scene_path) = received.expect("run_fn should have been called");
        assert!(PathBuf::from(&assets_root).is_dir());
        assert!(scene_path.is_file());

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn fails_before_ever_calling_run_fn_when_the_project_does_not_exist() {
        let root = scratch_dir("starman-cli-run-missing");
        let mut called = false;

        let result = run_with(root, |_, _, _| {
            called = true;
            Ok(())
        });

        assert!(result.is_err());
        assert!(!called, "run_fn must not be called for an invalid project");
    }
}
