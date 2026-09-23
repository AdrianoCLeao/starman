# Lua scripting (M3)

Starman embeds **Lua 5.4** via `mlua` in-process during editor Play and in the
shared runner bootstrap. Scripts and native plugins share the same host
command/query bus (`engine-plugin`).

## Project layout

```ron
scripts: (
  entry: Some("scripts/main.lua"),
  roots: ["scripts/"],
),
permissions: (
  filesystem: ["assets/**", "scripts/**", "saves/**"],
  process: false,
  network: false,
),
```

`require` is limited to the declared script roots. Filesystem/network/process
access is deny-by-default and gated by `PermissionGuard`.

## Host API (Lua)

| Function | Role |
|----------|------|
| `starman.log(level, message)` | Diagnostic log |
| `starman.spawn(name?)` | Spawn entity → opaque handle |
| `starman.despawn(handle)` | Despawn |
| `starman.set_field(handle, component, path, json)` | Set reflect field |
| `starman.get_field(handle, component, path)` | Get field as JSON string |
| `starman.entity_count()` | World entity count |
| `starman.register_system(schedule, name)` | Record dynamic system |
| `starman.persist.set/get` | Hot-reload persist helpers |

Optional global `on_update(dt)` is called each frame while Play is running.

## Hot reload

Edit a `.lua` file under a watched scripts root during in-process Play. The
host serializes `STARMAN_PERSIST` (or the persist API table), reloads the
entry chunk, and restores persist. Errors are isolated with `pcall` —
script failures do not crash the editor.

## Debugger

Basic controls: breakpoints by file/line, step, continue
(`DebuggerCommand` / console commands). Locals inspection is available
through the runtime debugger state.

## Play modes

- **In-process (default when scripts are declared):** Lua ticks inside the
  editor world; hot reload works.
- **Standalone Preview:** spawns `game-runner` out-of-process (preview /
  M8 path); not the Lua hot-reload gate.
