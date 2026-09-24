use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "starman", version, about = "Starman engine project CLI")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Create a new Starman project at `path`.
    New {
        path: PathBuf,
        /// The project's display name.
        #[arg(long, default_value = "New Project")]
        name: String,
        /// Entry scene path, relative to the project's `assets/` directory.
        #[arg(long, default_value = "scenes/main.scene.ron")]
        entry_scene: String,
    },
    /// Validate an existing project's manifest and directory layout.
    Validate { path: PathBuf },
    /// Import (or re-import changed) source assets under the project.
    Import { path: PathBuf },
    /// Open a window and run the project's entry scene.
    Run { path: PathBuf },
    /// Validate the project and headlessly load its entry scene.
    Test { path: PathBuf },
    /// Persist pending format migrations (with backups).
    Migrate { path: PathBuf },
    /// Localization tools.
    L10n {
        #[command(subcommand)]
        action: L10nAction,
    },
}

#[derive(Subcommand, Debug)]
pub enum L10nAction {
    /// Report parse errors and keys missing from (or unknown to) the
    /// default locale; fails on errors and missing keys.
    Check { path: PathBuf },
}
