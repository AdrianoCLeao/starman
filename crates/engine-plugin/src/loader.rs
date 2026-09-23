//! Dynamic `cdylib` plugin loader with ABI negotiate, shadow-copy, and panic fence.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use engine_core::{EngineError, Result};
use libloading::Library;
use starman_plugin_sdk::{
    AbiVersionFn, CapabilityFlags, NegotiateFn, OnLoadFn, OnUnloadFn, RestoreFn, SnapshotFn,
    StarmanHostV1, StarmanPluginCtx, StarmanPluginInfo, StarmanStr, SYM_ABI_VERSION, SYM_NEGOTIATE,
    SYM_ON_LOAD, SYM_ON_UNLOAD, SYM_RESTORE, SYM_SNAPSHOT, STARMAN_PLUGIN_ABI_VERSION,
};

use crate::host::{HostBus, SharedHostBus};
use crate::shadow_copy::shadow_copy_library;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PluginId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PluginState {
    Loaded,
    Faulted,
    Disabled,
}

pub struct LoadedPlugin {
    pub id: PluginId,
    pub name: String,
    pub version: String,
    pub state: PluginState,
    pub granted: CapabilityFlags,
    pub generation: u64,
    pub loaded_path: PathBuf,
    library: Library,
    on_unload: Option<OnUnloadFn>,
    snapshot: Option<SnapshotFn>,
    restore: Option<RestoreFn>,
    ctx: StarmanPluginCtx,
}

/// Host that loads and manages native plugins.
pub struct PluginHost {
    pub bus: SharedHostBus,
    cache_root: PathBuf,
    next_id: u64,
    next_generation: u64,
    plugins: Vec<LoadedPlugin>,
}

impl PluginHost {
    pub fn new(bus: SharedHostBus, cache_root: impl Into<PathBuf>) -> Self {
        Self {
            bus,
            cache_root: cache_root.into(),
            next_id: 1,
            next_generation: 1,
            plugins: Vec::new(),
        }
    }

    pub fn plugins(&self) -> &[LoadedPlugin] {
        &self.plugins
    }

    pub fn bus(&self) -> SharedHostBus {
        Arc::clone(&self.bus)
    }

    /// Loads a plugin library from `source_path` after shadow-copy.
    pub fn load(
        &mut self,
        name: &str,
        source_path: &Path,
        requested_mask: CapabilityFlags,
        project_permissions_fs: bool,
        project_permissions_network: bool,
        project_permissions_process: bool,
    ) -> Result<PluginId> {
        let generation = self.next_generation;
        self.next_generation += 1;
        let copied = shadow_copy_library(source_path, &self.cache_root, name, generation)?;

        let library = unsafe { Library::new(&copied) }.map_err(|error| EngineError::AssetLoad {
            path: copied.display().to_string(),
            reason: format!("failed to load plugin library: {error}"),
        })?;

        let abi_version: AbiVersionFn = unsafe {
            *library
                .get(SYM_ABI_VERSION)
                .map_err(|error| EngineError::AssetLoad {
                    path: copied.display().to_string(),
                    reason: format!("missing starman_plugin_abi_version: {error}"),
                })?
        };
        let version = unsafe { abi_version() };
        if version != STARMAN_PLUGIN_ABI_VERSION {
            return Err(EngineError::AssetLoad {
                path: copied.display().to_string(),
                reason: format!(
                    "plugin ABI version {version} is unsupported; host expects {STARMAN_PLUGIN_ABI_VERSION}"
                ),
            });
        }

        let negotiate: NegotiateFn = unsafe {
            *library
                .get(SYM_NEGOTIATE)
                .map_err(|error| EngineError::AssetLoad {
                    path: copied.display().to_string(),
                    reason: format!("missing starman_plugin_negotiate: {error}"),
                })?
        };

        let host_vtable = build_host_vtable();
        let mut info = StarmanPluginInfo {
            name: StarmanStr {
                ptr: std::ptr::null(),
                len: 0,
            },
            version: StarmanStr {
                ptr: std::ptr::null(),
                len: 0,
            },
            requested_capabilities: 0,
        };

        let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            negotiate(&host_vtable, &mut info)
        }))
        .unwrap_or(false);

        if !ok {
            return Err(EngineError::AssetLoad {
                path: copied.display().to_string(),
                reason: "plugin negotiate failed or panicked".to_owned(),
            });
        }

        let plugin_name = unsafe { info.name.as_str() }
            .unwrap_or(name)
            .to_owned();
        let plugin_version = unsafe { info.version.as_str() }
            .unwrap_or("0.0.0")
            .to_owned();
        let requested = CapabilityFlags::from_bits_truncate(info.requested_capabilities);

        let mut granted = requested & requested_mask;
        if !project_permissions_fs {
            granted.remove(CapabilityFlags::FILESYSTEM);
        }
        if !project_permissions_network {
            granted.remove(CapabilityFlags::NETWORK);
        }
        if !project_permissions_process {
            granted.remove(CapabilityFlags::PROCESS);
        }

        let id = PluginId(self.next_id);
        self.next_id += 1;

        let bus_ptr = Arc::as_ptr(&self.bus) as *mut std::ffi::c_void;
        let mut ctx = StarmanPluginCtx {
            host: bus_ptr,
            user_data: std::ptr::null_mut(),
            plugin_id: id.0,
            granted_capabilities: granted.bits(),
        };

        let on_load: OnLoadFn = unsafe {
            *library
                .get(SYM_ON_LOAD)
                .map_err(|error| EngineError::AssetLoad {
                    path: copied.display().to_string(),
                    reason: format!("missing starman_plugin_on_load: {error}"),
                })?
        };
        let on_unload: Option<OnUnloadFn> = unsafe { library.get(SYM_ON_UNLOAD).ok().map(|s| *s) };
        let snapshot: Option<SnapshotFn> = unsafe { library.get(SYM_SNAPSHOT).ok().map(|s| *s) };
        let restore: Option<RestoreFn> = unsafe { library.get(SYM_RESTORE).ok().map(|s| *s) };

        let load_ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            on_load(&mut ctx);
        }))
        .is_ok();

        let state = if load_ok {
            PluginState::Loaded
        } else {
            log::error!(
                target: "engine::plugin",
                "plugin '{plugin_name}' panicked during on_load"
            );
            PluginState::Faulted
        };

        self.plugins.push(LoadedPlugin {
            id,
            name: plugin_name,
            version: plugin_version,
            state,
            granted,
            generation,
            loaded_path: copied,
            library,
            on_unload,
            snapshot,
            restore,
            ctx,
        });

        Ok(id)
    }

    pub fn unload(&mut self, id: PluginId) -> Result<()> {
        let index = self
            .plugins
            .iter()
            .position(|p| p.id == id)
            .ok_or_else(|| EngineError::Config(format!("unknown plugin id {}", id.0)))?;
        let mut plugin = self.plugins.remove(index);

        if let Some(on_unload) = plugin.on_unload {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
                on_unload(&mut plugin.ctx);
            }));
        }

        if let Ok(mut bus) = self.bus.lock() {
            bus.registrations.clear_owner(id);
            bus.system_callbacks.retain(|(owner, _, _, _)| *owner != id);
        }

        // Library drops here, unloading the module.
        drop(plugin.library);
        Ok(())
    }

    pub fn snapshot(&mut self, id: PluginId) -> Result<Vec<u8>> {
        let plugin = self
            .plugins
            .iter_mut()
            .find(|p| p.id == id)
            .ok_or_else(|| EngineError::Config(format!("unknown plugin id {}", id.0)))?;
        if !plugin.granted.contains(CapabilityFlags::STATE) {
            return Ok(Vec::new());
        }
        let Some(snapshot) = plugin.snapshot else {
            return Ok(Vec::new());
        };
        let mut buf = vec![0u8; 64 * 1024];
        let len = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            snapshot(&mut plugin.ctx, buf.as_mut_ptr(), buf.len())
        }))
        .unwrap_or(0);
        buf.truncate(len.min(buf.len()));
        Ok(buf)
    }

    pub fn restore(&mut self, id: PluginId, data: &[u8]) -> Result<bool> {
        let plugin = self
            .plugins
            .iter_mut()
            .find(|p| p.id == id)
            .ok_or_else(|| EngineError::Config(format!("unknown plugin id {}", id.0)))?;
        let Some(restore) = plugin.restore else {
            return Ok(true);
        };
        let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            restore(&mut plugin.ctx, data.as_ptr(), data.len())
        }))
        .unwrap_or(false);
        if !ok {
            plugin.state = PluginState::Faulted;
        }
        Ok(ok)
    }

    /// Reload: snapshot → unload → load → restore.
    pub fn reload(
        &mut self,
        id: PluginId,
        source_path: &Path,
        requested_mask: CapabilityFlags,
        project_permissions_fs: bool,
        project_permissions_network: bool,
        project_permissions_process: bool,
    ) -> Result<PluginId> {
        let name = self
            .plugins
            .iter()
            .find(|p| p.id == id)
            .map(|p| p.name.clone())
            .ok_or_else(|| EngineError::Config(format!("unknown plugin id {}", id.0)))?;
        let snapshot = self.snapshot(id)?;
        self.unload(id)?;
        let new_id = self.load(
            &name,
            source_path,
            requested_mask,
            project_permissions_fs,
            project_permissions_network,
            project_permissions_process,
        )?;
        if !snapshot.is_empty() {
            let _ = self.restore(new_id, &snapshot)?;
        }
        Ok(new_id)
    }
}

fn build_host_vtable() -> StarmanHostV1 {
    StarmanHostV1 {
        abi_version: STARMAN_PLUGIN_ABI_VERSION,
        log: Some(host_log),
        spawn_entity: Some(host_spawn),
        despawn_entity: Some(host_despawn),
        set_component_field: Some(host_set_field),
        get_component_field: Some(host_get_field),
        register_system: Some(host_register_system),
    }
}

unsafe extern "C" fn host_log(host: *mut std::ffi::c_void, level: u32, message: StarmanStr) {
    let Some(message) = message.as_str() else {
        return;
    };
    if host.is_null() {
        return;
    }
    let bus = &*(host as *const Mutex<HostBus>);
    if let Ok(mut bus) = bus.lock() {
        match level {
            1 => log::error!(target: "engine::plugin", "{message}"),
            2 => log::warn!(target: "engine::plugin", "{message}"),
            3 => log::info!(target: "engine::plugin", "{message}"),
            _ => log::debug!(target: "engine::plugin", "{message}"),
        }
        bus.logs.push((level, message.to_owned()));
    }
}

unsafe extern "C" fn host_spawn(host: *mut std::ffi::c_void) -> starman_plugin_sdk::EntityHandle {
    // FFI spawn without world is a no-op handle; real spawn goes through HostBus from Lua/host.
    let _ = host;
    starman_plugin_sdk::EntityHandle(0)
}

unsafe extern "C" fn host_despawn(host: *mut std::ffi::c_void, _entity: starman_plugin_sdk::EntityHandle) -> bool {
    let _ = host;
    false
}

unsafe extern "C" fn host_set_field(
    host: *mut std::ffi::c_void,
    _entity: starman_plugin_sdk::EntityHandle,
    _component: StarmanStr,
    _field_path: StarmanStr,
    _json_value: StarmanStr,
) -> bool {
    let _ = host;
    false
}

unsafe extern "C" fn host_get_field(
    host: *mut std::ffi::c_void,
    _entity: starman_plugin_sdk::EntityHandle,
    _component: StarmanStr,
    _field_path: StarmanStr,
    _out_buf: *mut u8,
    _out_cap: usize,
    _out_len: *mut usize,
) -> bool {
    let _ = host;
    false
}

unsafe extern "C" fn host_register_system(
    host: *mut std::ffi::c_void,
    schedule: StarmanStr,
    name: StarmanStr,
    _callback: unsafe extern "C" fn(*mut std::ffi::c_void, f32),
    _user_data: *mut std::ffi::c_void,
) -> bool {
    if host.is_null() {
        return false;
    }
    let Some(schedule) = schedule.as_str() else {
        return false;
    };
    let Some(name) = name.as_str() else {
        return false;
    };
    let Some(schedule) = crate::host::ScheduleName::parse(schedule) else {
        return false;
    };
    let bus = &*(host as *const Mutex<HostBus>);
    if let Ok(mut bus) = bus.lock() {
        let owner = PluginId(0);
        bus.registrations
            .record_system(owner, schedule.as_str(), name);
        return true;
    }
    false
}

/// Discover the platform library file for a plugin directory.
pub fn find_plugin_library(plugin_dir: &Path, name: &str) -> Option<PathBuf> {
    let file_names = [
        format!("{name}.dll"),
        format!("lib{name}.so"),
        format!("lib{name}.dylib"),
        format!("{name}.so"),
        format!("{name}.dylib"),
        "plugin.dll".to_owned(),
        "libplugin.so".to_owned(),
        "libplugin.dylib".to_owned(),
    ];

    let search_dirs = [
        plugin_dir.to_path_buf(),
        plugin_dir.join("target/debug"),
        plugin_dir.join("target/release"),
        // When the plugin crate lives in the workspace, cargo emits into the
        // workspace target/ next to the repo root (two levels up from
        // examples/plugins/<name>).
        plugin_dir.join("../../target/debug"),
        plugin_dir.join("../../target/release"),
        plugin_dir.join("../../../target/debug"),
        plugin_dir.join("../../../target/release"),
    ];

    for dir in &search_dirs {
        for file in &file_names {
            let candidate = dir.join(file);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }

    // Scan the plugin directory for any dynamic library.
    if let Ok(entries) = std::fs::read_dir(plugin_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let ext = path.extension().and_then(OsStr::to_str);
            if matches!(ext, Some("dll" | "so" | "dylib")) {
                return Some(path);
            }
        }
    }
    None
}
