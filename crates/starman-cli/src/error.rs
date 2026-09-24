//! The CLI's own error type. Every subcommand reports failures through
//! this, so `main` has one place that maps a failure to an actionable
//! message and a stable exit code.
//!
//! Exit codes: `0` success (handled in `main`, not here); `2` validation
//! failed; `3` asset import had failures; `4` opening, creating, or running
//! a project/scene failed. (Argument-parsing errors are handled by `clap`
//! itself, before any of this runs, and also exit `2`.)

use std::fmt;
use std::path::PathBuf;

use engine_core::EngineError;
use engine_project::ValidationReport;

#[derive(Debug)]
pub enum CliError {
    /// The project failed validation (`validate`, or `test` before it even
    /// attempts to load the entry scene).
    Validation {
        path: PathBuf,
        report: ValidationReport,
    },
    /// One or more source assets failed to import.
    Import {
        path: PathBuf,
        failed: Vec<(String, String)>,
    },
    /// Creating, opening, or running a project/scene failed.
    Engine { path: PathBuf, source: EngineError },
    /// A content check found problems (`l10n check`, `save verify`,
    /// playthroughs).
    Check { path: PathBuf, report: String },
}

impl CliError {
    pub fn exit_code(&self) -> u8 {
        match self {
            CliError::Validation { .. } => 2,
            CliError::Import { .. } => 3,
            CliError::Engine { .. } => 4,
            CliError::Check { .. } => 5,
        }
    }
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CliError::Validation { path, report } => {
                write!(f, "validation failed for '{}':\n{report}", path.display())
            }
            CliError::Import { path, failed } => {
                writeln!(f, "import failed for '{}':", path.display())?;
                for (index, (relative_path, reason)) in failed.iter().enumerate() {
                    if index > 0 {
                        writeln!(f)?;
                    }
                    write!(f, "  {relative_path}: {reason}")?;
                }
                Ok(())
            }
            CliError::Engine { path, source } => {
                write!(f, "failed for '{}': {source}", path.display())
            }
            CliError::Check { path, report } => {
                write!(f, "check failed for '{}':\n{report}", path.display())
            }
        }
    }
}

impl std::error::Error for CliError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_codes_match_the_documented_scheme() {
        let validation = CliError::Validation {
            path: PathBuf::from("x"),
            report: ValidationReport::default(),
        };
        let import = CliError::Import {
            path: PathBuf::from("x"),
            failed: Vec::new(),
        };
        let engine = CliError::Engine {
            path: PathBuf::from("x"),
            source: EngineError::Config("boom".to_owned()),
        };

        assert_eq!(validation.exit_code(), 2);
        assert_eq!(import.exit_code(), 3);
        assert_eq!(engine.exit_code(), 4);
    }

    #[test]
    fn display_includes_the_offending_path() {
        let error = CliError::Engine {
            path: PathBuf::from("/tmp/my-project"),
            source: EngineError::Config("bad manifest".to_owned()),
        };
        let message = error.to_string();
        assert!(message.contains("/tmp/my-project"));
        assert!(message.contains("bad manifest"));
    }
}
