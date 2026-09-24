//! Requests from the game to its host: quitting and cursor handling.

use bevy_ecs::prelude::Resource;

/// How the OS cursor is constrained to the window.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CursorGrab {
    #[default]
    None,
    /// Cursor stays inside the window.
    Confined,
    /// Cursor is locked in place (relative mouse look).
    Locked,
}

/// Desired cursor state; the windowed host applies changes each frame.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub struct CursorState {
    pub grab: CursorGrab,
    pub visible: bool,
}

impl Default for CursorState {
    fn default() -> Self {
        Self {
            grab: CursorGrab::None,
            visible: true,
        }
    }
}

/// Application lifecycle requests from gameplay (menus, scripts).
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AppControl {
    pub exit_requested: bool,
}

impl AppControl {
    pub fn request_exit(&mut self) {
        self.exit_requested = true;
    }
}
