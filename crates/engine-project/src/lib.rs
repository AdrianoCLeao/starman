//! The Starman project model (ADR 0001): a versioned manifest plus a fixed
//! directory layout that separates authored source assets from generated
//! cache, diagnostics, and build output.

mod manifest;
mod paths;
mod project;
mod validate;

pub use manifest::{
    PermissionsConfig, PluginRef, ProjectManifest, ProjectSettings, ScriptsConfig,
};
pub use paths::ProjectPaths;
pub use project::{CreateOptions, Project};
pub use validate::{Severity, ValidationIssue, ValidationReport};
