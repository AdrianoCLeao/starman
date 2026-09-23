pub mod app;
mod asset_browser;
mod autosave;
mod clipboard;
pub mod commands;
pub mod config;
pub mod inspector;
pub mod layout;
mod play_runtime;
mod prefab_context;
pub mod selection;
pub mod viewport;

#[cfg(test)]
mod app_tests;

#[cfg(test)]
mod commands_tests;

#[cfg(test)]
mod config_tests;

pub use app::EditorApp;
