//! Bootstrap plugins + Lua from a project manifest for editor/runner/CLI.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use engine_core::Result;
use engine_plugin::{
    find_plugin_library, CapabilityFlags, HostBus, PermissionGuard, PluginHost, PluginId,
    PluginState, SharedHostBus,
};
use engine_project::{PermissionsConfig, ProjectManifest};

use crate::runtime::{LuaRuntime, LuaRuntimeConfig};

/// Combined extensibility host: native plugins + Lua, sharing one [`HostBus`].
pub struct ExtensibilityHost {
    pub bus: SharedHostBus,
    pub plugins: PluginHost,
    pub lua: Option<LuaRuntime>,
    pub project_root: PathBuf,
    script_watcher_roots: Vec<PathBuf>,
}

impl ExtensibilityHost {
    pub fn bootstrap(project_root: &Path, manifest: &ProjectManifest) -> Result<Self> {
        let permissions =
            PermissionGuard::new(project_root, manifest.permissions.to_plugin_permissions());
        let bus = Arc::new(Mutex::new(HostBus::new(permissions.clone())));
        let cache_root = project_root.join(".starman/plugin-cache");
        let mut plugins = PluginHost::new(Arc::clone(&bus), cache_root);

        let mask = CapabilityFlags::ALL;
        let perms = &manifest.permissions;
        for plugin in &manifest.plugins {
            let dir = manifest.resolve_plugin_dir(project_root, plugin);
            let library = if dir.is_file() {
                Some(dir.clone())
            } else {
                find_plugin_library(&dir, &plugin.name)
            };
            let Some(library) = library else {
                log::warn!(
                    target: "engine::lua",
                    "skipping plugin '{}': no library at {}",
                    plugin.name,
                    dir.display()
                );
                continue;
            };
            match plugins.load(
                &plugin.name,
                &library,
                mask,
                !perms.filesystem.is_empty(),
                perms.network,
                perms.process,
            ) {
                Ok(id) => {
                    log::info!(
                        target: "engine::plugin",
                        "loaded plugin '{}' as {:?}",
                        plugin.name,
                        id
                    );
                }
                Err(error) => {
                    log::error!(
                        target: "engine::plugin",
                        "failed to load plugin '{}': {error}",
                        plugin.name
                    );
                }
            }
        }

        let lua = if manifest.scripts.has_scripts() {
            let roots = manifest.scripts.effective_roots();
            let scripts_root =
                project_root.join(roots.first().map(|s| s.as_str()).unwrap_or("scripts/"));
            let entry = manifest
                .scripts
                .entry
                .as_ref()
                .map(|e| project_root.join(e));
            let config = LuaRuntimeConfig {
                scripts_root: scripts_root.clone(),
                entry,
                persist_path: project_root.join(".starman/lua-persist.json"),
                coroutine_budget: 8,
            };
            let runtime = LuaRuntime::new(Arc::clone(&bus), config)?;
            runtime.install_bindings()?;
            Some(runtime)
        } else {
            None
        };

        let script_watcher_roots = manifest
            .scripts
            .effective_roots()
            .into_iter()
            .map(|r| project_root.join(r))
            .collect();

        Ok(Self {
            bus,
            plugins,
            lua,
            project_root: project_root.to_path_buf(),
            script_watcher_roots,
        })
    }

    pub fn script_watcher_roots(&self) -> &[PathBuf] {
        &self.script_watcher_roots
    }

    pub fn load_lua_entry(&mut self) -> Result<()> {
        if let Some(lua) = &mut self.lua {
            lua.load_entry()?;
        }
        Ok(())
    }

    pub fn hot_reload_lua(&mut self) -> Result<()> {
        if let Some(lua) = &mut self.lua {
            lua.hot_reload()?;
        }
        Ok(())
    }

    pub fn tick_lua(&mut self, dt: f32) -> Result<()> {
        if let Some(lua) = &mut self.lua {
            lua.tick(dt)?;
        }
        Ok(())
    }

    pub fn plugin_states(&self) -> Vec<(PluginId, String, PluginState)> {
        self.plugins
            .plugins()
            .iter()
            .map(|p| (p.id, p.name.clone(), p.state))
            .collect()
    }

    pub fn reload_plugin_by_name(
        &mut self,
        name: &str,
        source: &Path,
        permissions: &PermissionsConfig,
    ) -> Result<PluginId> {
        let existing = self
            .plugins
            .plugins()
            .iter()
            .find(|p| p.name == name)
            .map(|p| p.id);
        let mask = CapabilityFlags::ALL;
        if let Some(id) = existing {
            self.plugins.reload(
                id,
                source,
                mask,
                !permissions.filesystem.is_empty(),
                permissions.network,
                permissions.process,
            )
        } else {
            self.plugins.load(
                name,
                source,
                mask,
                !permissions.filesystem.is_empty(),
                permissions.network,
                permissions.process,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_project::{ProjectManifest, ScriptsConfig};

    #[test]
    fn bootstrap_without_scripts_or_plugins() {
        let root = std::env::temp_dir().join("starman-ext-empty");
        let _ = std::fs::create_dir_all(&root);
        let manifest = ProjectManifest::new("t", "scenes/main.scene.ron");
        let host = ExtensibilityHost::bootstrap(&root, &manifest).unwrap();
        assert!(host.lua.is_none());
        assert!(host.plugins.plugins().is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn bootstrap_with_missing_lua_entry_still_creates_runtime() {
        let root = std::env::temp_dir().join("starman-ext-lua");
        let scripts = root.join("scripts");
        let _ = std::fs::create_dir_all(&scripts);
        let mut manifest = ProjectManifest::new("t", "scenes/main.scene.ron");
        manifest.scripts = ScriptsConfig {
            entry: Some("scripts/main.lua".into()),
            roots: vec!["scripts/".into()],
        };
        let host = ExtensibilityHost::bootstrap(&root, &manifest).unwrap();
        assert!(host.lua.is_some());
        let _ = std::fs::remove_dir_all(&root);
    }
}
