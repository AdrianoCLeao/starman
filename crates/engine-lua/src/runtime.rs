//! Lua 5.4 runtime bound to the shared host command/query bus.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use bevy_ecs::world::World;
use engine_core::{EngineError, Result};
use engine_plugin::{
    HostBus, HostCommand, HostQuery, PermissionGuard, PluginId, ScheduleName, SharedHostBus,
};
use engine_reflect::{ComponentRegistry, ReflectTypeRegistry};
use mlua::{Function, Lua, LuaOptions, StdLib, Table, Value};
use serde_json::Value as JsonValue;
use thiserror::Error;

use crate::debugger::LuaDebugger;
use crate::persist::{self, PERSIST_GLOBAL};

#[derive(Debug, Error)]
pub enum LuaError {
    #[error("{0}")]
    Message(String),
    #[error(transparent)]
    Engine(#[from] EngineError),
}

pub struct LuaRuntimeConfig {
    pub scripts_root: PathBuf,
    pub entry: Option<PathBuf>,
    pub persist_path: PathBuf,
    pub coroutine_budget: u32,
}

impl Default for LuaRuntimeConfig {
    fn default() -> Self {
        Self {
            scripts_root: PathBuf::from("scripts"),
            entry: None,
            persist_path: PathBuf::from(".starman/lua-persist.json"),
            coroutine_budget: 8,
        }
    }
}

/// Owner id reserved for Lua-registered systems.
pub const LUA_OWNER: PluginId = PluginId(0x4C554100); // "LUA\0"

pub struct LuaRuntime {
    lua: Lua,
    bus: SharedHostBus,
    config: LuaRuntimeConfig,
    pub debugger: LuaDebugger,
    /// Pending coroutines (thread handles as registry keys).
    coroutines: Vec<mlua::Thread>,
    last_error: Option<String>,
    loaded: bool,
}

impl LuaRuntime {
    pub fn new(bus: SharedHostBus, config: LuaRuntimeConfig) -> Result<Self> {
        let lua = Lua::new_with(StdLib::ALL_SAFE, LuaOptions::default())
            .map_err(|error| EngineError::Config(format!("failed to create Lua state: {error}")))?;

        // Sandbox: remove dangerous libs already excluded by ALL_SAFE;
        // also restrict package.path to scripts root.
        {
            let package: Table = lua
                .globals()
                .get("package")
                .map_err(|error| EngineError::Config(error.to_string()))?;
            let search = format!(
                "{}/?.lua;{}/?/init.lua",
                config.scripts_root.display(),
                config.scripts_root.display()
            );
            package
                .set("path", search)
                .map_err(|error| EngineError::Config(error.to_string()))?;
            package
                .set("cpath", "")
                .map_err(|error| EngineError::Config(error.to_string()))?;
        }

        Ok(Self {
            lua,
            bus,
            config,
            debugger: LuaDebugger::default(),
            coroutines: Vec::new(),
            last_error: None,
            loaded: false,
        })
    }

    pub fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    pub fn is_loaded(&self) -> bool {
        self.loaded
    }

    pub fn bus(&self) -> SharedHostBus {
        Arc::clone(&self.bus)
    }

    /// Installs `starman.*` bindings that forward to the host bus.
    ///
    /// Callers must pass world/registry pointers that remain valid for the
    /// duration of Lua entrypoints (tick/load). The runtime stores them as
    /// raw pointers refreshed each [`Self::with_world`] call.
    pub fn install_bindings(&self) -> Result<()> {
        let starman = self
            .lua
            .create_table()
            .map_err(|error| EngineError::Config(error.to_string()))?;

        let bus = Arc::clone(&self.bus);
        let log_fn = self
            .lua
            .create_function(move |_, (level, message): (u32, String)| {
                if let Ok(mut bus) = bus.lock() {
                    bus.logs.push((level, message.clone()));
                    match level {
                        1 => log::error!(target: "engine::lua", "{message}"),
                        2 => log::warn!(target: "engine::lua", "{message}"),
                        3 => log::info!(target: "engine::lua", "{message}"),
                        _ => log::debug!(target: "engine::lua", "{message}"),
                    }
                }
                Ok(())
            })
            .map_err(|error| EngineError::Config(error.to_string()))?;
        starman
            .set("log", log_fn)
            .map_err(|error| EngineError::Config(error.to_string()))?;

        // Placeholders filled by with_world via registry.
        self.lua.set_app_data(WorldPtrs::default());
        install_world_fns(&self.lua, &starman, Arc::clone(&self.bus))?;

        let persist = self
            .lua
            .create_table()
            .map_err(|error| EngineError::Config(error.to_string()))?;
        let set_persist = self
            .lua
            .create_function(|lua, (key, value): (String, Value)| {
                let globals = lua.globals();
                let table: Table = match globals.get(PERSIST_GLOBAL)? {
                    Value::Table(t) => t,
                    _ => {
                        let t = lua.create_table()?;
                        globals.set(PERSIST_GLOBAL, t.clone())?;
                        t
                    }
                };
                table.set(key, value)?;
                Ok(())
            })
            .map_err(|error| EngineError::Config(error.to_string()))?;
        let get_persist = self
            .lua
            .create_function(|lua, key: String| {
                let globals = lua.globals();
                let table: Value = globals.get(PERSIST_GLOBAL)?;
                match table {
                    Value::Table(t) => t.get::<Value>(key),
                    _ => Ok(Value::Nil),
                }
            })
            .map_err(|error| EngineError::Config(error.to_string()))?;
        persist
            .set("set", set_persist)
            .map_err(|error| EngineError::Config(error.to_string()))?;
        persist
            .set("get", get_persist)
            .map_err(|error| EngineError::Config(error.to_string()))?;
        starman
            .set("persist", persist)
            .map_err(|error| EngineError::Config(error.to_string()))?;

        self.lua
            .globals()
            .set("starman", starman)
            .map_err(|error| EngineError::Config(error.to_string()))?;

        // Empty persist table by default.
        let empty = self
            .lua
            .create_table()
            .map_err(|error| EngineError::Config(error.to_string()))?;
        self.lua
            .globals()
            .set(PERSIST_GLOBAL, empty)
            .map_err(|error| EngineError::Config(error.to_string()))?;

        Ok(())
    }

    /// Binds world pointers for the duration of `f`.
    pub fn with_world<R>(
        &mut self,
        world: &mut World,
        components: &ComponentRegistry,
        types: &ReflectTypeRegistry,
        f: impl FnOnce(&mut Self) -> Result<R>,
    ) -> Result<R> {
        {
            let mut ptrs = self
                .lua
                .app_data_mut::<WorldPtrs>()
                .ok_or_else(|| EngineError::Config("lua world ptrs missing".into()))?;
            ptrs.world = world as *mut World;
            ptrs.components = components as *const ComponentRegistry;
            ptrs.types = types as *const ReflectTypeRegistry;
        }
        let result = f(self);
        if let Some(mut ptrs) = self.lua.app_data_mut::<WorldPtrs>() {
            ptrs.clear();
        }
        result
    }

    /// Loads (or reloads) the entry script, preserving `STARMAN_PERSIST`.
    pub fn load_entry(&mut self) -> Result<()> {
        let Some(entry) = self.config.entry.clone() else {
            self.loaded = false;
            return Ok(());
        };
        if !entry.is_file() {
            return Err(EngineError::AssetLoad {
                path: entry.display().to_string(),
                reason: "lua entry script does not exist".to_owned(),
            });
        }

        let persist =
            persist::capture_persist(&self.lua).unwrap_or(JsonValue::Object(Default::default()));
        let source = std::fs::read_to_string(&entry).map_err(|error| EngineError::AssetLoad {
            path: entry.display().to_string(),
            reason: error.to_string(),
        })?;

        // Restore declared persist *before* executing the chunk so scripts that
        // do `STARMAN_PERSIST = STARMAN_PERSIST or {}` keep prior state, then
        // mutate it. An empty first-load table is a no-op restore.
        let _ = persist::restore_persist(&self.lua, &persist);

        let load_result = self
            .lua
            .load(&source)
            .set_name(entry.to_string_lossy())
            .exec();
        match load_result {
            Ok(()) => {
                self.loaded = true;
                self.last_error = None;
                if let Ok(current) = persist::capture_persist(&self.lua) {
                    let _ = persist::save_persist(&self.config.persist_path, &current);
                }
                Ok(())
            }
            Err(error) => {
                let message = format!("lua load error: {error}");
                self.last_error = Some(message.clone());
                self.loaded = false;
                log::error!(target: "engine::lua", "{message}");
                // Do not fail the host hard — error isolation.
                Ok(())
            }
        }
    }

    /// Hot-reload entry while keeping persist.
    pub fn hot_reload(&mut self) -> Result<()> {
        self.load_entry()
    }

    /// Resume coroutines and call optional global `on_update(dt)`.
    pub fn tick(&mut self, dt: f32) -> Result<()> {
        if self.debugger.paused {
            return Ok(());
        }
        if !self.loaded {
            return Ok(());
        }

        // Coroutine budget.
        let budget = self.config.coroutine_budget;
        let mut i = 0;
        while i < self.coroutines.len() && (i as u32) < budget {
            let status = self.coroutines[i].status();
            if matches!(status, mlua::ThreadStatus::Resumable) {
                match self.coroutines[i].resume::<Value>(()) {
                    Ok(_) => {}
                    Err(error) => {
                        let message = format!("lua coroutine error: {error}");
                        self.last_error = Some(message.clone());
                        log::error!(target: "engine::lua", "{message}");
                        self.coroutines.remove(i);
                        continue;
                    }
                }
            }
            if !matches!(self.coroutines[i].status(), mlua::ThreadStatus::Resumable) {
                self.coroutines.remove(i);
                continue;
            }
            i += 1;
        }

        let on_update: mlua::Result<Function> = self.lua.globals().get("on_update");
        if let Ok(on_update) = on_update {
            if let Err(error) = on_update.call::<()>(dt) {
                let message = format!("lua on_update error: {error}");
                self.last_error = Some(message.clone());
                log::error!(target: "engine::lua", "{message}");
            }
        }
        Ok(())
    }

    pub fn spawn_coroutine(&mut self, chunk: &str) -> Result<()> {
        let thread = self
            .lua
            .create_thread(
                self.lua
                    .load(chunk)
                    .into_function()
                    .map_err(|error| EngineError::Config(error.to_string()))?,
            )
            .map_err(|error| EngineError::Config(error.to_string()))?;
        self.coroutines.push(thread);
        Ok(())
    }
}

#[derive(Default)]
struct WorldPtrs {
    world: *mut World,
    components: *const ComponentRegistry,
    types: *const ReflectTypeRegistry,
}

impl WorldPtrs {
    fn clear(&mut self) {
        self.world = std::ptr::null_mut();
        self.components = std::ptr::null();
        self.types = std::ptr::null();
    }

    unsafe fn world_mut(&self) -> Result<&'static mut World> {
        if self.world.is_null() {
            return Err(EngineError::Config("lua world not bound".into()));
        }
        Ok(&mut *self.world)
    }

    unsafe fn components(&self) -> Result<&'static ComponentRegistry> {
        if self.components.is_null() {
            return Err(EngineError::Config("lua components not bound".into()));
        }
        Ok(&*self.components)
    }

    unsafe fn types(&self) -> Result<&'static ReflectTypeRegistry> {
        if self.types.is_null() {
            return Err(EngineError::Config("lua types not bound".into()));
        }
        Ok(&*self.types)
    }
}

// Safety: WorldPtrs is only used while the editor/runner holds exclusive world borrow.
unsafe impl Send for WorldPtrs {}
unsafe impl Sync for WorldPtrs {}

fn install_world_fns(lua: &Lua, starman: &Table, bus: SharedHostBus) -> Result<()> {
    let bus_spawn = Arc::clone(&bus);
    let spawn = lua
        .create_function(move |lua, name: Option<String>| {
            let ptrs = lua
                .app_data_ref::<WorldPtrs>()
                .ok_or_else(|| mlua::Error::external("world not bound"))?;
            let world = unsafe { ptrs.world_mut() }.map_err(mlua::Error::external)?;
            let components = unsafe { ptrs.components() }.map_err(mlua::Error::external)?;
            let mut bus = bus_spawn
                .lock()
                .map_err(|e| mlua::Error::external(e.to_string()))?;
            let value = bus
                .execute(world, components, HostCommand::Spawn { name })
                .map_err(|e| mlua::Error::external(e.to_string()))?;
            Ok(value.as_u64().unwrap_or(0))
        })
        .map_err(|error| EngineError::Config(error.to_string()))?;
    starman
        .set("spawn", spawn)
        .map_err(|error| EngineError::Config(error.to_string()))?;

    let bus_despawn = Arc::clone(&bus);
    let despawn = lua
        .create_function(move |lua, handle: u64| {
            let ptrs = lua
                .app_data_ref::<WorldPtrs>()
                .ok_or_else(|| mlua::Error::external("world not bound"))?;
            let world = unsafe { ptrs.world_mut() }.map_err(mlua::Error::external)?;
            let components = unsafe { ptrs.components() }.map_err(mlua::Error::external)?;
            let mut bus = bus_despawn
                .lock()
                .map_err(|e| mlua::Error::external(e.to_string()))?;
            bus.execute(
                world,
                components,
                HostCommand::Despawn {
                    entity: starman_plugin_sdk::EntityHandle(handle),
                },
            )
            .map_err(|e| mlua::Error::external(e.to_string()))?;
            Ok(())
        })
        .map_err(|error| EngineError::Config(error.to_string()))?;
    starman
        .set("despawn", despawn)
        .map_err(|error| EngineError::Config(error.to_string()))?;

    let bus_set = Arc::clone(&bus);
    let set_field = lua
        .create_function(
            move |lua, (handle, component, field_path, json): (u64, String, String, String)| {
                let ptrs = lua
                    .app_data_ref::<WorldPtrs>()
                    .ok_or_else(|| mlua::Error::external("world not bound"))?;
                let world = unsafe { ptrs.world_mut() }.map_err(mlua::Error::external)?;
                let components = unsafe { ptrs.components() }.map_err(mlua::Error::external)?;
                let value: JsonValue = serde_json::from_str(&json)
                    .map_err(|e| mlua::Error::external(e.to_string()))?;
                let mut bus = bus_set
                    .lock()
                    .map_err(|e| mlua::Error::external(e.to_string()))?;
                bus.execute(
                    world,
                    components,
                    HostCommand::SetField {
                        entity: starman_plugin_sdk::EntityHandle(handle),
                        component,
                        field_path,
                        value,
                    },
                )
                .map_err(|e| mlua::Error::external(e.to_string()))?;
                Ok(())
            },
        )
        .map_err(|error| EngineError::Config(error.to_string()))?;
    starman
        .set("set_field", set_field)
        .map_err(|error| EngineError::Config(error.to_string()))?;

    let bus_get = Arc::clone(&bus);
    let get_field = lua
        .create_function(
            move |lua, (handle, component, field_path): (u64, String, String)| {
                let ptrs = lua
                    .app_data_ref::<WorldPtrs>()
                    .ok_or_else(|| mlua::Error::external("world not bound"))?;
                let world = unsafe { ptrs.world_mut() }.map_err(mlua::Error::external)?;
                let components = unsafe { ptrs.components() }.map_err(mlua::Error::external)?;
                let types = unsafe { ptrs.types() }.map_err(mlua::Error::external)?;
                let mut bus = bus_get
                    .lock()
                    .map_err(|e| mlua::Error::external(e.to_string()))?;
                let value = bus
                    .query(
                        world,
                        components,
                        types,
                        HostQuery::GetField {
                            entity: starman_plugin_sdk::EntityHandle(handle),
                            component,
                            field_path,
                        },
                    )
                    .map_err(|e| mlua::Error::external(e.to_string()))?;
                Ok(value.to_string())
            },
        )
        .map_err(|error| EngineError::Config(error.to_string()))?;
    starman
        .set("get_field", get_field)
        .map_err(|error| EngineError::Config(error.to_string()))?;

    let bus_reg = Arc::clone(&bus);
    let register_system = lua
        .create_function(move |_, (schedule, name): (String, String)| {
            let Some(schedule) = ScheduleName::parse(&schedule) else {
                return Err(mlua::Error::external("unknown schedule"));
            };
            let mut bus = bus_reg
                .lock()
                .map_err(|e| mlua::Error::external(e.to_string()))?;
            bus.registrations
                .record_system(LUA_OWNER, schedule.as_str(), &name);
            bus.system_callbacks.push((LUA_OWNER, schedule, name, 0));
            Ok(())
        })
        .map_err(|error| EngineError::Config(error.to_string()))?;
    starman
        .set("register_system", register_system)
        .map_err(|error| EngineError::Config(error.to_string()))?;

    let entity_count = lua
        .create_function(move |lua, ()| {
            let ptrs = lua
                .app_data_ref::<WorldPtrs>()
                .ok_or_else(|| mlua::Error::external("world not bound"))?;
            let world = unsafe { ptrs.world_mut() }.map_err(mlua::Error::external)?;
            Ok(world.iter_entities().count() as u64)
        })
        .map_err(|error| EngineError::Config(error.to_string()))?;
    starman
        .set("entity_count", entity_count)
        .map_err(|error| EngineError::Config(error.to_string()))?;

    Ok(())
}

/// Convenience: build a runtime from project paths + permissions.
pub fn create_from_project(
    project_root: &Path,
    scripts_root: &Path,
    entry: Option<&Path>,
    permissions: PermissionGuard,
) -> Result<LuaRuntime> {
    let bus = Arc::new(Mutex::new(HostBus::new(permissions)));
    let config = LuaRuntimeConfig {
        scripts_root: scripts_root.to_path_buf(),
        entry: entry.map(|p| p.to_path_buf()),
        persist_path: project_root.join(".starman/lua-persist.json"),
        coroutine_budget: 8,
    };
    let runtime = LuaRuntime::new(bus, config)?;
    runtime.install_bindings()?;
    Ok(runtime)
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_plugin::ProjectPermissions;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn hot_reload_preserves_persist() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("starman-lua-{nanos}"));
        let scripts = root.join("scripts");
        std::fs::create_dir_all(&scripts).unwrap();
        let entry = scripts.join("main.lua");
        std::fs::write(
            &entry,
            r#"
STARMAN_PERSIST = STARMAN_PERSIST or { counter = 0 }
STARMAN_PERSIST.counter = (STARMAN_PERSIST.counter or 0) + 1
function on_update(dt) end
"#,
        )
        .unwrap();

        let guard = PermissionGuard::new(&root, ProjectPermissions::default());
        let mut rt = create_from_project(&root, &scripts, Some(&entry), guard).unwrap();
        let mut world = World::new();
        let components = ComponentRegistry::default();
        let types = ReflectTypeRegistry::default();
        rt.with_world(&mut world, &components, &types, |rt| rt.load_entry())
            .unwrap();
        rt.with_world(&mut world, &components, &types, |rt| rt.hot_reload())
            .unwrap();
        let persist = persist::capture_persist(&rt.lua).unwrap();
        assert_eq!(persist["counter"], 2);
        let _ = std::fs::remove_dir_all(&root);
    }
}
