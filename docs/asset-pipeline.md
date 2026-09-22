# Asset pipeline

Every source asset gets a stable identity, tracked import metadata, and a
content-addressed cache entry. See [ADR 0002](adr/0002-persistent-ids.md)
(identifiers) and [ADR 0003](adr/0003-assets-and-cache.md) (database and
cache) for the accepted design; this document is the practical, "how it
actually behaves" reference on top of those decisions.

## Identity

- **`SourceAssetId`** (UUID v4) — one per source file (a texture, a mesh
  file, a material, ...). Assigned once, on first import, and preserved
  across renames and content edits.
- **`SubAssetId`** (UUID v5, derived from the parent id + a stable key) —
  one per sub-resource *within* a file. Today only mesh files produce
  these: every mesh in a `.glb`/`.gltf` gets its own sub-asset, keyed
  `"mesh:<index>"` (in the file's own mesh order), even when the file has
  only one mesh. Textures and materials are always a single resource per
  file, so they never have sub-assets.
- **`EntityId`** (UUID v4) — see [scene-format.md](scene-format.md).

None of these are ever the same as a `bevy_ecs::Entity` or a runtime
`AssetId` (the process-local, ephemeral handle id `AssetServer` hands out)
— those exist only for the lifetime of one run and are never persisted.

## `.meta.ron`

Every imported source asset gets a sidecar next to it,
`<file>.meta.ron`:

```ron
(
    version: 1,
    id: "5c0a9c2e-4b8a-4c2a-9c9a-9c9a9c9a9c9a",
    importer: "mesh",
    dependencies: [],
    content_hash: "b3:...",
    sub_assets: [
        (id: "...", key: "mesh:0", label: Some("Body")),
        (id: "...", key: "mesh:1", label: Some("Wheel")),
    ],
)
```

`content_hash` (blake3) is what makes reimporting cheap and correct: a
file whose hash hasn't changed since the last import is left alone, even
if its mtime moved (a checkout, a save with identical bytes, ...).
`sub_assets` is empty for anything but a mesh file with `sub_assets`
worth recording.

## Cache

Imported output lives under `.starman/cache/imported/<hash prefix>/<hash>`,
addressed purely by content hash — two different files with identical
bytes collapse onto the same cache entry. For M1 the cached "output" is a
pass-through of the source bytes; per-type processing (texture decoding,
mesh baking into a GPU-ready format, ...) is layered on top of the same
addressing scheme in a later milestone, not a reason to change how assets
are identified.

## Dependency graph

An asset can declare (`AssetDatabase::set_dependencies`) which other
assets it depends on — e.g. a material depending on the textures it
references. `AssetDatabase::invalidate(id)` returns everything that
transitively depends on `id`, in preparation for cascade re-import when a
dependency changes (the file watcher, described below, already reports a
changed dependency's dependents alongside the change itself).

## Renaming and moving

`AssetDatabase::rename(old_relative_path, new_relative_path)` moves both
the source file and its `.meta.ron` sidecar together, so the id survives.
If a file is renamed *outside* the engine (a plain `mv`, a Finder rename)
its sidecar is left behind, dangling; the next `AssetDatabase::rescan`
re-associates the same id with the file wherever it resurfaces, by
matching content hash — best-effort recovery, not a guarantee (two
unrelated files with identical bytes would be ambiguous, though this is
vanishingly unlikely for real content).

## Typed references in scenes ("migration on save")

A scene component that references an asset (`MeshRenderer.mesh/texture/
material`, `Sprite.texture`) is persisted as a single string field that is
*either* a stable id (preferred, once known) *or* a plain relative path
(legacy, and the only option when no `AssetDatabase` is attached — e.g. a
handful of test fixtures). There is no separate tag field: the shape alone
disambiguates it, since a legitimate relative path never happens to also
parse as a UUID.

**This means a scene's reference only becomes rename-safe once it has been
saved at least once with a database attached.** Loading a legacy,
path-based reference (with a database attached) still works today and
teaches the `AssetServer` that reference's id as a side effect — but
nothing rewrites the file just from being read, even by `starman-cli
test`, which is deliberately read-only. The next time that scene is
*saved* (the normal editor authoring flow), the reference is written back
as an id. This mirrors how the v1→v2 scene migration works (see
[scene-format.md](scene-format.md)): a capability the format gains, not
something retroactively applied to every file that has ever existed.

`mesh` is one layer deeper than `texture`/`material`: it may resolve to
one specific sub-asset (a `SubAssetId`, for a mesh within a multi-mesh
file, or the sole mesh of a single-mesh file) or to a whole file merged (a
`SourceAssetId`, only possible for a multi-mesh file loaded as one
flattened blob). Both id kinds serialize as a bare UUID string with no
structural difference; resolving one tries it as a sub-asset id first,
then as a source id. A random v4 `SourceAssetId` colliding with a derived
v5 `SubAssetId` is astronomically unlikely (128 bits) — an accepted
trade-off for keeping the persisted format a single plain string instead
of a tagged one.

## File watching

`AssetWatcher` watches a project's `assets/` directory recursively.
Filesystem events are debounced (default 150 ms of quiet before a change
is processed, so a burst of saves collapses into one reimport) and handed
to a small worker pool that does the heavy part (read, hash, cache store)
off the polling thread; results are committed back — `.meta.ron` updates,
index/dependency-graph updates — atomically (temp file + rename), so a
reader never observes a half-written file. A newer event for the same
path supersedes an in-flight job for it rather than racing it.

`AssetServer::poll_hot_reload()` applies pending changes: textures,
meshes, and materials with a loaded handle get their payload reloaded (and
their revision bumped); a changed scene is only *reported*
(`HotReloadReport::scenes`) — nothing reloads it automatically, since a
scene may have unsaved, live editor state a blind reload would discard;
removed files are reported without discarding the last-known payload; a
reload that fails to parse keeps the previous payload and marks the asset
`Failed`, with an actionable log message.
