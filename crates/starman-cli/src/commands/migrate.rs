use std::path::PathBuf;

use engine_project::{Project, ProjectManifest};

use crate::error::CliError;

/// What `starman migrate` did.
pub struct MigrateOutcome {
    pub path: PathBuf,
    /// The manifest version the project was migrated from, if it was old.
    pub manifest_migrated_from: Option<u32>,
}

/// Persists every pending on-open migration of the project (currently the
/// manifest), writing `*.bak` backups first (ADR 0006).
pub fn run(path: PathBuf) -> Result<MigrateOutcome, CliError> {
    let mut project = Project::open(&path).map_err(|source| CliError::Engine {
        path: path.clone(),
        source,
    })?;
    let manifest_migrated_from =
        project
            .persist_migration()
            .map_err(|source| CliError::Engine {
                path: path.clone(),
                source,
            })?;
    Ok(MigrateOutcome {
        path,
        manifest_migrated_from,
    })
}

impl MigrateOutcome {
    pub fn message(&self) -> String {
        match self.manifest_migrated_from {
            Some(from) => format!(
                "Migrated '{}' manifest v{from} -> v{} (backup: project.ron.v{from}.bak).",
                self.path.display(),
                ProjectManifest::CURRENT_VERSION
            ),
            None => format!("Project '{}' is already up to date.", self.path.display()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn migrates_a_v1_manifest_with_a_backup() {
        let root = std::env::temp_dir().join(format!(
            "starman-cli-migrate-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("assets/scenes")).unwrap();
        fs::write(
            root.join("assets/scenes/main.scene.ron"),
            "(version: 3, name: \"m\", entities: [])",
        )
        .unwrap();
        let v1 = "(\n    version: 1,\n    id: \"00000000-0000-0000-0000-000000000001\",\n    name: \"Old\",\n    entry_scene: \"scenes/main.scene.ron\",\n)";
        fs::write(root.join("project.ron"), v1).unwrap();

        let outcome = run(root.clone()).expect("migrate succeeds");
        assert_eq!(outcome.manifest_migrated_from, Some(1));
        assert_eq!(
            fs::read_to_string(root.join("project.ron.v1.bak")).unwrap(),
            v1
        );
        let reopened = Project::open(&root).unwrap();
        assert_eq!(reopened.migrated_from, None);
        assert_eq!(reopened.manifest.version, ProjectManifest::CURRENT_VERSION);

        let again = run(root.clone()).unwrap();
        assert_eq!(again.manifest_migrated_from, None);
        let _ = fs::remove_dir_all(&root);
    }
}
