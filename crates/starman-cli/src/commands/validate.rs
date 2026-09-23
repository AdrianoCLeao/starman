use std::path::PathBuf;

use engine_project::{Project, ValidationReport};
use engine_scene::CompositionGraph;

use crate::error::CliError;

/// Returns the report (which may still carry warnings) on success; fails
/// only when the report contains at least one error-level issue.
pub fn run(path: PathBuf) -> Result<ValidationReport, CliError> {
    let report = Project::validate(&path).map_err(|source| CliError::Engine {
        path: path.clone(),
        source,
    })?;

    if !report.is_valid() {
        return Err(CliError::Validation { path, report });
    }

    // Nested-scene cycle / broken-ref check on the entry scene (M2).
    let project = Project::open(&path).map_err(|source| CliError::Engine {
        path: path.clone(),
        source,
    })?;
    let entry = project
        .paths
        .resolve_asset_relative(&project.manifest.entry_scene);
    if let Ok(database) =
        engine_assets::AssetDatabase::open(project.paths.assets_dir(), project.paths.imported_dir())
    {
        if let Err(error) = CompositionGraph::build_from_file(&entry, &database) {
            let mut report = ValidationReport::new();
            report.push_error(error.to_string());
            return Err(CliError::Validation { path, report });
        }
    }

    Ok(report)
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
    fn a_freshly_created_project_is_valid() {
        let root = scratch_dir("starman-cli-validate-ok");
        Project::create(&root, CreateOptions::default()).expect("project should be created");

        let report = run(root.clone()).expect("validate should succeed");
        assert!(report.is_valid());

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_broken_manifest_fails_with_an_actionable_error() {
        let root = scratch_dir("starman-cli-validate-broken");
        fs::create_dir_all(&root).expect("scratch dir should be created");
        fs::write(root.join("project.ron"), "not valid ron (((")
            .expect("manifest should be written");

        let error = run(root.clone()).expect_err("validate should fail");
        assert_eq!(error.exit_code(), 2);
        match error {
            CliError::Validation { path, report } => {
                assert_eq!(path, root);
                assert!(report
                    .errors()
                    .any(|issue| issue.message.contains("manifest")));
            }
            other => panic!("expected CliError::Validation, got {other:?}"),
        }

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_missing_entry_scene_reports_a_validation_error() {
        let root = scratch_dir("starman-cli-validate-missing-scene");
        let project =
            Project::create(&root, CreateOptions::default()).expect("project should be created");
        let entry_scene_path = project
            .paths
            .resolve_asset_relative(&project.manifest.entry_scene);
        fs::remove_file(&entry_scene_path).expect("entry scene should be removable");

        let error = run(root.clone()).expect_err("validate should fail");
        assert_eq!(error.exit_code(), 2);

        let _ = fs::remove_dir_all(&root);
    }
}
