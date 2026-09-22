use std::path::PathBuf;

use engine_assets::{AssetDatabase, ImportSummary};
use engine_project::Project;

use crate::error::CliError;

#[derive(Debug)]
pub struct ImportOutcome {
    pub path: PathBuf,
    pub summary: ImportSummary,
}

pub fn run(path: PathBuf) -> Result<ImportOutcome, CliError> {
    let engine_error = |source| CliError::Engine {
        path: path.clone(),
        source,
    };

    let project = Project::open(&path).map_err(engine_error)?;
    let mut database =
        AssetDatabase::open(project.paths.assets_dir(), project.paths.imported_dir())
            .map_err(engine_error)?;
    let summary = database.import_all().map_err(engine_error)?;

    if summary.is_success() {
        Ok(ImportOutcome { path, summary })
    } else {
        Err(CliError::Import {
            path,
            failed: summary.failed,
        })
    }
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
    fn imports_new_assets_and_is_stable_on_rerun() {
        let root = scratch_dir("starman-cli-import");
        let project =
            Project::create(&root, CreateOptions::default()).expect("project should be created");
        fs::write(project.paths.assets_dir().join("notes.txt"), b"hello world")
            .expect("asset should be written");

        let first = run(root.clone()).expect("first import should succeed");
        // The entry scene plus notes.txt.
        assert_eq!(first.summary.imported.len(), 2);
        assert!(first.summary.failed.is_empty());

        let second = run(root.clone()).expect("second import should succeed");
        assert!(second.summary.imported.is_empty());
        assert_eq!(second.summary.unchanged.len(), 2);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn fails_for_a_project_that_does_not_exist() {
        let root = scratch_dir("starman-cli-import-missing");
        let error = run(root.clone()).expect_err("import should fail for a missing project");
        assert_eq!(error.exit_code(), 4);
    }
}
