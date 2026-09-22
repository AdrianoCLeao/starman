use std::path::PathBuf;

use engine_project::{CreateOptions, Project};

use crate::error::CliError;

#[derive(Debug)]
pub struct NewOutcome {
    pub path: PathBuf,
    pub name: String,
    pub project_id: String,
}

pub fn run(path: PathBuf, name: String, entry_scene: String) -> Result<NewOutcome, CliError> {
    let project =
        Project::create(&path, CreateOptions { name, entry_scene }).map_err(|source| {
            CliError::Engine {
                path: path.clone(),
                source,
            }
        })?;

    Ok(NewOutcome {
        path: project.paths.root().to_path_buf(),
        name: project.manifest.name.clone(),
        project_id: project.manifest.id.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn creates_a_project_and_reports_its_identity() {
        let root = scratch_dir("starman-cli-new");

        let outcome = run(
            root.clone(),
            "Demo Game".to_owned(),
            "scenes/main.scene.ron".to_owned(),
        )
        .expect("new should succeed");

        assert_eq!(outcome.name, "Demo Game");
        assert!(!outcome.project_id.is_empty());
        assert!(root.join("project.ron").is_file());
        assert!(root.join("assets/scenes/main.scene.ron").is_file());

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn is_idempotent_and_preserves_identity() {
        let root = scratch_dir("starman-cli-new-idempotent");

        let first = run(
            root.clone(),
            "A".to_owned(),
            "scenes/main.scene.ron".to_owned(),
        )
        .expect("first new should succeed");
        let second = run(
            root.clone(),
            "A".to_owned(),
            "scenes/main.scene.ron".to_owned(),
        )
        .expect("second new on the same path should reopen, not fail");

        assert_eq!(first.project_id, second.project_id);

        let _ = fs::remove_dir_all(&root);
    }
}
