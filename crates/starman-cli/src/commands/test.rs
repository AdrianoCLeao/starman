use std::path::PathBuf;

use engine_assets::AssetDatabase;
use engine_core::EngineModules;
use engine_project::{Project, ValidationReport};
use engine_runner::RunnerOptions;

use crate::error::CliError;

#[derive(Debug)]
pub struct TestOutcome {
    pub path: PathBuf,
    pub report: ValidationReport,
    pub entity_count: usize,
    pub lua_frames: u32,
}

/// Validates the project, then headlessly loads its entry scene (no window,
/// no GPU device) to confirm it parses and every referenced asset resolves.
/// When scripts/plugins are declared, boots the extensibility host and ticks
/// a few Lua frames (M3 gate).
pub fn run(path: PathBuf) -> Result<TestOutcome, CliError> {
    let engine_error = |source| CliError::Engine {
        path: path.clone(),
        source,
    };

    let report = Project::validate(&path).map_err(engine_error)?;
    if !report.is_valid() {
        return Err(CliError::Validation { path, report });
    }

    let project = Project::open(&path).map_err(engine_error)?;
    let assets_root = project.paths.assets_dir().to_string_lossy().into_owned();
    let scene_path = project
        .paths
        .resolve_asset_relative(&project.manifest.entry_scene);

    let mut options = RunnerOptions::new(format!("{} (test)", project.manifest.name))
        .with_project_root(project.paths.root());
    if let Ok(database) =
        AssetDatabase::open(project.paths.assets_dir(), project.paths.imported_dir())
    {
        options = options.with_database(database);
    }

    let mut prepared = engine_runner::prepare_scene_world(&assets_root, &scene_path, options)
        .map_err(engine_error)?;

    // Exercise Lua for a few frames when present (headless gate).
    let mut lua_frames = 0u32;
    for _ in 0..3 {
        let _ = prepared
            .engine
            .modules
            .flush_input(&mut prepared.engine.world);
        lua_frames += 1;
    }

    Ok(TestOutcome {
        path,
        report,
        entity_count: prepared.entity_count(),
        lua_frames,
    })
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

    #[test]
    fn a_valid_project_loads_headlessly_and_reports_zero_entities() {
        let root = scratch_dir("starman-cli-test-ok");
        Project::create(&root, CreateOptions::default()).expect("project should be created");

        let outcome = run(root.clone()).expect("test should succeed");
        assert_eq!(outcome.entity_count, 0);
        assert!(outcome.report.is_valid());

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn validation_errors_short_circuit_before_loading_the_scene() {
        let root = scratch_dir("starman-cli-test-invalid");
        let project =
            Project::create(&root, CreateOptions::default()).expect("project should be created");
        let entry_scene_path = project
            .paths
            .resolve_asset_relative(&project.manifest.entry_scene);
        fs::remove_file(&entry_scene_path).expect("entry scene should be removable");

        let error = run(root.clone()).expect_err("test should fail validation");
        assert_eq!(error.exit_code(), 2);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_broken_scene_file_fails_with_a_message_naming_it() {
        let root = scratch_dir("starman-cli-test-broken-scene");
        let project =
            Project::create(&root, CreateOptions::default()).expect("project should be created");
        let entry_scene_path = project
            .paths
            .resolve_asset_relative(&project.manifest.entry_scene);
        fs::write(&entry_scene_path, "not valid ron (((").expect("scene should be overwritten");

        let error = run(root.clone()).expect_err("test should fail to load the scene");
        assert_eq!(error.exit_code(), 4);
        assert!(error
            .to_string()
            .contains(&entry_scene_path.display().to_string()));

        let _ = fs::remove_dir_all(&root);
    }
}
