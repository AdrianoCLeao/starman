# Project model

A Starman project is a directory with a versioned manifest (`project.ron`)
and a fixed layout that keeps authored content separate from anything the
engine generates. See [ADR 0001](adr/0001-project-model.md) for the
rationale.

## Layout

```
<project root>/
  project.ron              # the manifest
  assets/                  # authored, source-controlled content
  .starman/                # everything the engine generates; safe to delete
    cache/
      imported/             # content-addressed import cache
    diagnostics/            # local logs and crash reports
  build/                    # packaged output (development/release)
  .gitignore                # ignores .starman/ and build/ (written once, on create)
```

`.starman/` and `build/` are regenerated on demand — an empty project
missing them is not broken, just not yet touched (see `Project::validate`
below).

## The manifest (`project.ron`)

```ron
(
    version: 1,
    id: "5c0a9c2e-4b8a-4c2a-9c9a-9c9a9c9a9c9a",
    name: "My Game",
    entry_scene: "scenes/main.scene.ron",  // relative to assets/
    plugins: [],
    settings: {},
    targets: [],
)
```

`id` is a stable `ProjectId` (UUID v4, ADR 0002), assigned once at creation
and never reused. `version` gates the manifest format itself — there is
only one version today; an unrecognized one is a hard, actionable error,
never silently reinterpreted (ADR 0006).

## Working with a project

As a library (`engine-project`):

```rust
use engine_project::{CreateOptions, Project};

// Scaffold a new project. Safe to call again on an already-initialized
// path — it reopens instead of overwriting authored data.
let project = Project::create("my-game", CreateOptions {
    name: "My Game".into(),
    entry_scene: "scenes/main.scene.ron".into(),
})?;

// Open an existing one. Fails fast (missing manifest, unsupported
// version, missing assets/, missing entry scene) rather than starting in
// a half-valid state.
let project = Project::open("my-game")?;

// Lenient check: collects every problem instead of stopping at the
// first one. Missing generated directories (.starman/, build/) are
// warnings, not errors — they are created on demand.
let report = Project::validate("my-game")?;
```

From the command line (`starman-cli`, binary `starman`):

```
starman new <path> [--name <name>] [--entry-scene <relative path>]
starman validate <path>
starman import <path>
starman run <path>
starman test <path>
```

`test` validates the project and then headlessly loads its entry scene (no
window, no GPU device) — this is what proves a project actually opens, not
just that its files are well-formed.

## Reference project

`examples/reference-project/` is a real Starman project checked into this
repository — the same one the editor, `game-runner`, `sandbox`, and
`engine-smoke` open by default when run from the repo root with no
arguments. It exists so every capability in the roadmap gets exercised
against real, versioned content instead of only synthetic test fixtures
(ROADMAP.md's Definição de Pronto #5).
