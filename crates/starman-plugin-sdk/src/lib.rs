//! Public C ABI surface for Starman native plugins (ADR 0008).
//!
//! Plugin authors link this crate (or the equivalent C header) and export the
//! required `extern "C"` symbols. The host never passes Rust-owned types
//! across this boundary.

#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_void};

/// Current host/plugin ABI version. Mismatches fail negotiate.
pub const STARMAN_PLUGIN_ABI_VERSION: u32 = 1;

/// Opaque entity handle (never an ECS `Entity` bit pattern assumed stable).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct EntityHandle(pub u64);

/// Opaque asset handle.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct AssetHandle(pub u64);

/// Opaque query handle.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct QueryHandle(pub u64);

bitflags::bitflags! {
    /// Capability flags negotiated at load time.
    #[repr(transparent)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct CapabilityFlags: u64 {
        const NONE = 0;
        const REGISTER_COMPONENT = 1 << 0;
        const REGISTER_SYSTEM = 1 << 1;
        const FILESYSTEM = 1 << 2;
        const NETWORK = 1 << 3;
        const PROCESS = 1 << 4;
        const EDITOR_UI = 1 << 5;
        const STATE = 1 << 6;
        const ALL = Self::REGISTER_COMPONENT.bits()
            | Self::REGISTER_SYSTEM.bits()
            | Self::FILESYSTEM.bits()
            | Self::NETWORK.bits()
            | Self::PROCESS.bits()
            | Self::EDITOR_UI.bits()
            | Self::STATE.bits();
    }
}

/// UTF-8 string view owned by the caller for the duration of the call.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct StarmanStr {
    pub ptr: *const c_char,
    pub len: usize,
}

impl StarmanStr {
    /// # Safety
    /// `s` must remain valid for the duration of the FFI call.
    pub unsafe fn from_str(s: &str) -> Self {
        Self {
            ptr: s.as_ptr().cast(),
            len: s.len(),
        }
    }

    /// # Safety
    /// Pointer/len must refer to valid UTF-8 for `len` bytes or be empty.
    pub unsafe fn as_str(&self) -> Option<&str> {
        if self.ptr.is_null() || self.len == 0 {
            return Some("");
        }
        let slice = std::slice::from_raw_parts(self.ptr.cast::<u8>(), self.len);
        std::str::from_utf8(slice).ok()
    }
}

/// Plugin identity filled during negotiate.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct StarmanPluginInfo {
    pub name: StarmanStr,
    pub version: StarmanStr,
    pub requested_capabilities: u64,
}

/// Per-plugin context: opaque host pointer + plugin-local user data.
#[repr(C)]
pub struct StarmanPluginCtx {
    pub host: *mut c_void,
    pub user_data: *mut c_void,
    pub plugin_id: u64,
    pub granted_capabilities: u64,
}

/// Host vtable passed to negotiate (versioned).
#[repr(C)]
pub struct StarmanHostV1 {
    pub abi_version: u32,
    pub log: Option<unsafe extern "C" fn(host: *mut c_void, level: u32, message: StarmanStr)>,
    pub spawn_entity: Option<unsafe extern "C" fn(host: *mut c_void) -> EntityHandle>,
    pub despawn_entity: Option<unsafe extern "C" fn(host: *mut c_void, entity: EntityHandle) -> bool>,
    pub set_component_field: Option<
        unsafe extern "C" fn(
            host: *mut c_void,
            entity: EntityHandle,
            component: StarmanStr,
            field_path: StarmanStr,
            json_value: StarmanStr,
        ) -> bool,
    >,
    pub get_component_field: Option<
        unsafe extern "C" fn(
            host: *mut c_void,
            entity: EntityHandle,
            component: StarmanStr,
            field_path: StarmanStr,
            out_buf: *mut u8,
            out_cap: usize,
            out_len: *mut usize,
        ) -> bool,
    >,
    pub register_system: Option<
        unsafe extern "C" fn(
            host: *mut c_void,
            schedule: StarmanStr,
            name: StarmanStr,
            callback: unsafe extern "C" fn(*mut c_void, f32),
            user_data: *mut c_void,
        ) -> bool,
    >,
}

/// Required: returns [`STARMAN_PLUGIN_ABI_VERSION`].
pub type AbiVersionFn = unsafe extern "C" fn() -> u32;
/// Required: fill info; return true if compatible with host.
pub type NegotiateFn =
    unsafe extern "C" fn(host: *const StarmanHostV1, info: *mut StarmanPluginInfo) -> bool;
pub type OnLoadFn = unsafe extern "C" fn(ctx: *mut StarmanPluginCtx);
pub type OnUnloadFn = unsafe extern "C" fn(ctx: *mut StarmanPluginCtx);
pub type SnapshotFn =
    unsafe extern "C" fn(ctx: *mut StarmanPluginCtx, out: *mut u8, cap: usize) -> usize;
pub type RestoreFn =
    unsafe extern "C" fn(ctx: *mut StarmanPluginCtx, data: *const u8, len: usize) -> bool;

pub const SYM_ABI_VERSION: &[u8] = b"starman_plugin_abi_version\0";
pub const SYM_NEGOTIATE: &[u8] = b"starman_plugin_negotiate\0";
pub const SYM_ON_LOAD: &[u8] = b"starman_plugin_on_load\0";
pub const SYM_ON_UNLOAD: &[u8] = b"starman_plugin_on_unload\0";
pub const SYM_SNAPSHOT: &[u8] = b"starman_plugin_snapshot\0";
pub const SYM_RESTORE: &[u8] = b"starman_plugin_restore\0";

/// Helper for plugin crates: export the standard ABI version symbol.
#[macro_export]
macro_rules! declare_abi_version {
    () => {
        #[no_mangle]
        pub extern "C" fn starman_plugin_abi_version() -> u32 {
            $crate::STARMAN_PLUGIN_ABI_VERSION
        }
    };
}
