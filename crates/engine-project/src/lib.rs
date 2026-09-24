//! The Starman project model (ADR 0001): a versioned manifest plus a fixed
//! directory layout that separates authored source assets from generated
//! cache, diagnostics, and build output.

mod manifest;
pub mod migration;
mod paths;
mod project;
pub mod settings;
mod validate;

pub use manifest::{PermissionsConfig, PluginRef, ProjectManifest, ProjectSettings, ScriptsConfig};
pub use paths::{user_data_dir, ProjectPaths};
pub use project::{CreateOptions, Project};
pub use settings::{
    AudioSettings, GameSettings, InputSettings, LocalizationSettings, PhysicsSettings,
    RenderingSettings, SaveSettings, UiScaleMode, UiSettings, MAX_PHYSICS_LAYERS,
};
pub use validate::{Severity, ValidationIssue, ValidationReport};
