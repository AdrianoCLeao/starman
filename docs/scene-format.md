# Scene format

Scenes are RON files (`SceneFile`): a version number, a name, and a tree
of entities. See [ADR 0002](adr/0002-persistent-ids.md) (identity) and
[ADR 0006](adr/0006-compatibility-policy.md) (migration/compatibility
policy) for the accepted decisions behind this format.

## Version 2 (current)

```ron
(
    version: 2,
    name: "MyScene",
    entities: [
        (
            id: "5c0a9c2e-4b8a-4c2a-9c9a-9c9a9c9a9c9a",
            name: Some("Cube"),
            components: { /* ... */ },
            children: [ /* nested entities, same shape */ ],
        ),
    ],
)
```

Every entity carries a stable `EntityId` (UUID v4, `id`). It is never the
same as the `bevy_ecs::Entity` the entity happens to have in a given run
— that value is a runtime handle, regenerated every load, and is never
persisted (ADR 0002). An entity's `EntityId` is attached at load time as a
`PersistentId` component and read back at save time, so it survives a
load → edit → save round trip unchanged.

Component payloads inside `components` are whatever the reflection
registry (for built-in components) or a `SceneExternalComponents`
implementation (for `MeshRenderer`/`Sprite`, see
[asset-pipeline.md](asset-pipeline.md#typed-references-in-scenes-migration-on-save))
produces — versioning below is about the entity tree shape, not individual
component payload shapes.

## Migration from version 1

Version 1 (no `id` field on entities) is automatically migrated on load:
`SceneDeserializer::load_file` peeks the file's `version`, and for a `1`
it parses it under the legacy shape, assigns each entity a new `EntityId`
(v4, one per entity, in the file's own traversal order), backs up the
original file next to it (`<file>.v1.bak`), and rewrites the file in
place as version 2 before continuing to load it normally.

Two independent migrations of the *same* v1 content produce **different**
ids each time — they're UUID v4 (random), not v5 (derived) — because
migration is meant to run once per file; the v2 file it produces becomes
the source of truth for that entity's identity from then on. This is the
same trade-off documented for scenes' asset references
("migration on save"): a capability applies going forward from the
migration, not retroactively to every prior copy of the file.

An unrecognized version (neither 1 nor 2) is a hard, explicit error naming
the version found and expected — never silently reinterpreted as another
version.

## Compatibility fixture

Version 1 is a permanent compatibility fixture (ADR 0006): golden-file
tests (`crates/engine-assets/tests/scene_migration.rs`) migrate a minimal
synthetic v1 scene and assert the result is deterministic *within* one
migration run (stable ids relative to each other, unique per entity) and
that reloading the now-migrated v2 file is fully reproducible (same ids
every time, since they're now literally written in the file).
