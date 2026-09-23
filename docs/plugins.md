# Starman plugins (ADR 0008)

Native gameplay extensions are `cdylib` crates that export a versioned C ABI.
They never receive Rust types across the boundary — only POD, opaque
`uint64_t` handles, and UTF-8 string views.

## Authoring

1. Depend on `starman-plugin-sdk`.
2. Export the required symbols:

```rust
starman_plugin_sdk::declare_abi_version!();

#[no_mangle]
pub unsafe extern "C" fn starman_plugin_negotiate(
    host: *const StarmanHostV1,
    info: *mut StarmanPluginInfo,
) -> bool { /* ... */ }

#[no_mangle]
pub unsafe extern "C" fn starman_plugin_on_load(ctx: *mut StarmanPluginCtx) { /* ... */ }

#[no_mangle]
pub unsafe extern "C" fn starman_plugin_on_unload(ctx: *mut StarmanPluginCtx) { /* ... */ }
```

Optional: `starman_plugin_snapshot` / `starman_plugin_restore` when
`CapabilityFlags::STATE` is granted.

3. Declare the plugin in `project.ron`:

```ron
plugins: [
  ( name: "example_gameplay", version_req: Some("^0.1"), path: Some("plugins/example_gameplay"), ),
],
```

## Loading

The host shadow-copies the library into `.starman/plugin-cache/<name>-<gen>/`
before `dlopen`/`LoadLibrary`, so rebuilds can overwrite the original file.
Panics inside plugin callbacks are caught at the FFI border; the plugin is
marked `Faulted` and can be unloaded without taking down the editor.

## Permissions

Capability flags requested at negotiate are intersected with
`permissions` in the project manifest (deny-by-default for process/network).

See also [docs/lua.md](lua.md) and [ADR 0008](adr/0008-plugin-abi-v1.md).
