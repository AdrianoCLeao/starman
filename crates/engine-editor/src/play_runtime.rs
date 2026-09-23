//! In-process play-mode helpers for the editor (M3).

use std::path::PathBuf;
use std::time::Instant;

use engine_assets::{AssetWatcher, WatchConfig};
use engine_core::Result;
use engine_lua::ExtensibilityHost;
use engine_project::Project;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlayKind {
    /// Lua/plugins tick inside the editor world (hot reload gate).
    InProcess,
    /// Spawns `game-runner` as a standalone preview (M8-oriented path).
    StandalonePreview,
}

pub struct InProcessPlay {
    pub host: ExtensibilityHost,
    pub script_watcher: Option<AssetWatcher>,
    pub last_tick: Instant,
    pub restore_snapshot: PathBuf,
}

impl InProcessPlay {
    pub fn start(project: &Project, restore_snapshot: PathBuf) -> Result<Self> {
        let mut host = ExtensibilityHost::bootstrap(project.paths.root(), &project.manifest)?;
        host.load_lua_entry()?;

        let script_watcher = host
            .script_watcher_roots()
            .first()
            .map(|root| AssetWatcher::watch_root(root, WatchConfig::default()));

        // Also watch plugin directories for binary rebuilds.
        for plugin in &project.manifest.plugins {
            let dir = project
                .manifest
                .resolve_plugin_dir(project.paths.root(), plugin);
            if dir.is_dir() {
                // Extra watchers are optional; primary is scripts. Plugin
                // reload is polled when a Plugin AssetChange appears on any
                // watcher the editor polls — for M3 we reload on script
                // watcher Other/Plugin if libraries live under scripts (rare).
                let _ = dir;
            }
        }

        Ok(Self {
            host,
            script_watcher,
            last_tick: Instant::now(),
            restore_snapshot,
        })
    }

    pub fn take_dt(&mut self) -> f32 {
        let now = Instant::now();
        let dt = now.duration_since(self.last_tick).as_secs_f32();
        self.last_tick = now;
        dt.clamp(0.0, 0.1)
    }
}
