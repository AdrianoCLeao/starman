//! In-process Lua 5.4 runtime with host-bus bindings (M3).

mod bootstrap;
mod debugger;
mod persist;
mod runtime;

pub use bootstrap::ExtensibilityHost;
pub use debugger::{DebuggerCommand, LuaDebugger};
pub use persist::{load_persist, save_persist, PERSIST_GLOBAL};
pub use runtime::{create_from_project, LuaError, LuaRuntime, LuaRuntimeConfig, LUA_OWNER};

pub fn module_name() -> &'static str {
    "engine-lua"
}
