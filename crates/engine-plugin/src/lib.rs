//! Dynamic plugin host, command/query bus, and permission gate (M3 / ADR 0008).

mod host;
mod loader;
mod permissions;
mod registry;
mod shadow_copy;

pub use host::{
    HostBus, HostCommand, HostError, HostQuery, HostValue, ScheduleName, SharedHostBus,
};
pub use loader::{
    find_plugin_library, LoadedPlugin, PluginHost, PluginId, PluginState,
};
pub use permissions::{PermissionGuard, ProjectPermissions};
pub use registry::DynamicRegistration;
pub use shadow_copy::shadow_copy_library;

pub use starman_plugin_sdk::{
    CapabilityFlags, EntityHandle, STARMAN_PLUGIN_ABI_VERSION,
};

pub fn module_name() -> &'static str {
    "engine-plugin"
}
