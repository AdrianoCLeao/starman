//! Reference gameplay plugin for the M3 gate.

#![allow(clippy::missing_safety_doc)]

use std::os::raw::c_void;
use std::sync::atomic::{AtomicU32, Ordering};

use starman_plugin_sdk::{
    declare_abi_version, CapabilityFlags, StarmanHostV1, StarmanPluginCtx, StarmanPluginInfo,
    StarmanStr, STARMAN_PLUGIN_ABI_VERSION,
};

declare_abi_version!();

static TICKS: AtomicU32 = AtomicU32::new(0);

#[no_mangle]
pub unsafe extern "C" fn starman_plugin_negotiate(
    host: *const StarmanHostV1,
    info: *mut StarmanPluginInfo,
) -> bool {
    if host.is_null() || info.is_null() {
        return false;
    }
    let host = &*host;
    if host.abi_version != STARMAN_PLUGIN_ABI_VERSION {
        return false;
    }
    (*info).name = StarmanStr::from_str("example_gameplay");
    (*info).version = StarmanStr::from_str("0.1.0");
    (*info).requested_capabilities =
        (CapabilityFlags::REGISTER_SYSTEM | CapabilityFlags::STATE).bits();
    true
}

#[no_mangle]
pub unsafe extern "C" fn starman_plugin_on_load(ctx: *mut StarmanPluginCtx) {
    if ctx.is_null() {
        return;
    }
    let ctx = &mut *ctx;
    TICKS.store(0, Ordering::SeqCst);
    let _ = ctx;
}

#[no_mangle]
pub unsafe extern "C" fn starman_plugin_on_unload(_ctx: *mut StarmanPluginCtx) {}

#[no_mangle]
pub unsafe extern "C" fn starman_plugin_snapshot(
    _ctx: *mut StarmanPluginCtx,
    out: *mut u8,
    cap: usize,
) -> usize {
    if out.is_null() || cap < 4 {
        return 0;
    }
    let ticks = TICKS.load(Ordering::SeqCst).to_le_bytes();
    std::ptr::copy_nonoverlapping(ticks.as_ptr(), out, 4);
    4
}

#[no_mangle]
pub unsafe extern "C" fn starman_plugin_restore(
    _ctx: *mut StarmanPluginCtx,
    data: *const u8,
    len: usize,
) -> bool {
    if data.is_null() || len < 4 {
        return false;
    }
    let mut bytes = [0u8; 4];
    std::ptr::copy_nonoverlapping(data, bytes.as_mut_ptr(), 4);
    TICKS.store(u32::from_le_bytes(bytes), Ordering::SeqCst);
    true
}

/// Called by host-registered systems in integration tests.
#[no_mangle]
pub unsafe extern "C" fn example_gameplay_tick(_user: *mut c_void, _dt: f32) {
    TICKS.fetch_add(1, Ordering::SeqCst);
}
