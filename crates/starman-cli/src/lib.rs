//! The Starman project CLI, as a thin-lib/thin-bin pair: `dispatch` does
//! all the work and is what tests call directly (no subprocess spawning);
//! `main.rs` only parses argv and maps the result to an exit code.

pub mod cli;
pub mod commands;
pub mod error;

pub use cli::{Cli, Command, L10nAction};
pub use error::CliError;

/// Executes the parsed command, returning a human-readable success message
/// to print, or an actionable [`CliError`] (see [`CliError::exit_code`]).
pub fn dispatch(cli: Cli) -> Result<String, CliError> {
    match cli.command {
        Command::New {
            path,
            name,
            entry_scene,
        } => {
            let outcome = commands::new::run(path, name, entry_scene)?;
            Ok(format!(
                "Created project '{}' (id {}) at '{}'.",
                outcome.name,
                outcome.project_id,
                outcome.path.display()
            ))
        }
        Command::Validate { path } => {
            let report = commands::validate::run(path.clone())?;
            Ok(format!("Project '{}' is valid.\n{report}", path.display()))
        }
        Command::Import { path } => {
            let outcome = commands::import::run(path)?;
            Ok(format!(
                "Imported {} asset(s), {} unchanged, 0 failed.",
                outcome.summary.imported.len(),
                outcome.summary.unchanged.len()
            ))
        }
        Command::Run { path } => {
            commands::run::run(path)?;
            Ok("Runner exited cleanly.".to_owned())
        }
        Command::Migrate { path } => Ok(commands::migrate::run(path)?.message()),
        Command::L10n {
            action: L10nAction::Check { path },
        } => commands::l10n::check(path),
        Command::Test { path } => {
            let outcome = commands::test::run(path)?;
            Ok(format!(
                "Project '{}' is valid; entry scene loaded with {} entities.",
                outcome.path.display(),
                outcome.entity_count
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_project::{CreateOptions, Project};
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn scratch_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be valid")
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{nanos}"))
    }

    #[test]
    fn dispatch_new_reports_the_created_project() {
        let root = scratch_dir("starman-cli-dispatch-new");

        let message = dispatch(Cli {
            command: Command::New {
                path: root.clone(),
                name: "Demo".to_owned(),
                entry_scene: "scenes/main.scene.ron".to_owned(),
            },
        })
        .expect("dispatch should succeed");

        assert!(message.contains("Demo"));
        assert!(message.contains(&root.display().to_string()));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn dispatch_validate_reports_success_for_a_fresh_project() {
        let root = scratch_dir("starman-cli-dispatch-validate");
        Project::create(&root, CreateOptions::default()).expect("project should be created");

        let message = dispatch(Cli {
            command: Command::Validate { path: root.clone() },
        })
        .expect("dispatch should succeed");

        assert!(message.contains("is valid"));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn dispatch_test_reports_entity_count() {
        let root = scratch_dir("starman-cli-dispatch-test");
        Project::create(&root, CreateOptions::default()).expect("project should be created");

        let message = dispatch(Cli {
            command: Command::Test { path: root.clone() },
        })
        .expect("dispatch should succeed");

        assert!(message.contains("0 entities"));

        let _ = std::fs::remove_dir_all(&root);
    }
}
