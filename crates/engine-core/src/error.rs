use thiserror::Error;

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("Configuration error: {0}")]
    Config(String),
    #[error("Render error: {0}")]
    Render(String),
    #[error("Asset loading failed for '{path}': {reason}")]
    AssetLoad { path: String, reason: String },
    #[error("Physics world is not initialized")]
    PhysicsNotInitialized,
    #[error("Audio backend unavailable: {0}")]
    Audio(String),
    #[error("Windowing error: {0}")]
    Window(String),
    #[error("Invalid {kind} '{value}': {reason}")]
    InvalidId {
        kind: &'static str,
        value: String,
        reason: String,
    },
    #[error("Invalid project at '{path}': {reason}")]
    InvalidProject { path: String, reason: String },
}

pub type Result<T> = std::result::Result<T, EngineError>;
