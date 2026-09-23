# Prefabs and nested scenes

A **prefab** in Starman is just a scene asset (`.scene.ron`) that another
scene instances. There is no separate file extension. See
[ADR 0007](adr/0007-nested-scenes.md) and [scene-format.md](scene-format.md).

## Create an instance

1. Author a reusable scene (for example `scenes/props/crate.scene.ron`).
2. In a parent scene, add an entity with a `Transform` and an `instance`
   block pointing at that scene's `SourceAssetId` (and optionally
   `scene_path` for path fallback).
3. Load the parent — the runtime expands inherited children live. They are
   tagged as inherited and are **not** written back into the parent file.

## Overrides

Edit a field on an inherited entity in the parent context to create a
field override on the instance root:

- **Revert** — drop the local override and restore the template value.
- **Apply** — write the override into the source scene asset and clear it
  from the instance (affects every instance after resync).
- **Promote** — move a locally-added entity into the source scene.

Nesting depth is unlimited. Overrides stack from the leaf instance toward
the source asset.

## Open Prefab

In the hierarchy, instance roots are marked `[I]`. Use **Open Prefab** on
the context menu to edit the source scene in isolation. The breadcrumb at
the top of the hierarchy returns to the parent context.

## Clipboard and multi-select

- Ctrl/Cmd-click toggles multi-selection; Shift-click adds.
- Ctrl/Cmd+C / X / V copy, cut, and paste entity subtrees.
- Pasting onto an instance (or inherited child) creates a local-added
  entity owned by that instance.
- Duplicate assigns fresh `EntityId`s.

## Autosave and recovery

While a scene has unsaved changes the editor periodically writes an atomic
autosave under `.starman/autosave/`. If the editor exits uncleanly, the
next project open offers to recover or discard.

## CLI

`starman validate` checks the project layout and nested-scene cycles on
the entry scene. `starman import` / `starman test` exercise the reference
project's nested composition.
