# ADR 0008: Plugin host ABI v1

Date: 2026-09-23

## Status

Accepted

## Context

ADR 0005 established that native plugins use a C ABI with opaque handles and
explicit serialized state. M3 needs a concrete, versioned surface so the host
can load `cdylib` plugins, negotiate capabilities, and reload safely.

## Decision

1. **ABI version** is a single `u32` exported as `starman_plugin_abi_version`.
   The host accepts only `STARMAN_PLUGIN_ABI_VERSION` (currently `1`).
2. **Lifecycle** symbols: `starman_plugin_negotiate`, `starman_plugin_on_load`,
   `starman_plugin_on_unload`, optional `starman_plugin_snapshot` /
   `starman_plugin_restore`.
3. **Types across the boundary** are C POD and `u64` opaque handles only.
   Strings are `(ptr, len)` copied by the host. No Rust types, no shared
   allocator ownership.
4. **Capability flags** are a `u64` bitset negotiated at load time and
   intersected with project permissions.
5. **Panic isolation**: every host→plugin callback is wrapped in
   `catch_unwind`; a panicking plugin is marked `Faulted` and may be unloaded.
6. **Shadow-copy**: before `dlopen`/`LoadLibrary`, the host copies the library
   into `.starman/plugin-cache/<name>-<generation>/` so the source file can be
   rebuilt while loaded.
7. **Lua and Rust plugins** share the same host command/query bus; only the
   binding layer differs.

## Consequences

- `starman-plugin-sdk` is the sole public surface for plugin authors.
- `engine-plugin` owns loading, ownership of dynamic registrations, and
  permission checks.
- Breaking ABI changes require bumping `STARMAN_PLUGIN_ABI_VERSION` and a
  migration note; old plugins fail negotiate with an actionable error.
