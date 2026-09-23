# Scene format

Scenes are RON files (`SceneFile`): a version number, a name, and a tree
of entities. See [ADR 0002](adr/0002-persistent-ids.md) (identity),
[ADR 0006](adr/0006-compatibility-policy.md) (migration), and
[ADR 0007](adr/0007-nested-scenes.md) (nested instances) for the accepted
decisions behind this format. Prefab authoring is covered in
[prefabs.md](prefabs.md).

## Version 3 (current)

```ron
(
    version: 3,
    name: "MyScene",
    entities: [
        (
            id: "5c0a9c2e-4b8a-4c2a-9c9a-9c9a9c9a9c9a",
            name: Some("Cube"),
            components: { /* ... */ },
            children: [ /* nested entities, same shape */ ],
        ),
        (
            id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            name: Some("EnemyPack"),
            components: {
                "Transform": { /* instance root transform */ },
            },
            children: [],
            instance: Some((
                scene: "11111111-2222-3333-4444-555555555555",
                scene_path: Some("scenes/props/crate.scene.ron"),
                overrides: [
                    (
                        target: "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                        component: "Transform",
                        field_path: "translation",
                        value: { "x": 1.0, "y": 0.0, "z": 0.0 },
                    ),
                ],
                removed: [],
                added: [],
            )),
        ),
    ],
)
```

Every entity carries a stable `EntityId` (UUID v4, `id`). It is never the
same as the `bevy_ecs::Entity` the entity happens to have in a given run —
that value is a runtime handle, regenerated every load, and is never
persisted (ADR 0002). An entity's `EntityId` is attached at load time as a
`PersistentId` component and read back at save time, so it survives a
load → edit → save round trip unchanged.

When `instance` is set, the entity is an **instance root** of another
scene asset (ADR 0007). Inherited children are expanded at runtime and are
**not** stored under `children` in the parent file. Local-only entities
belong in `instance.added`; template entities hidden on this instance are
listed in `instance.removed`.

Component payloads inside `components` are whatever the reflection
registry (for built-in components) or a `SceneExternalComponents`
implementation (for `MeshRenderer`/`Sprite`, see
[asset-pipeline.md](asset-pipeline.md#typed-references-in-scenes-migration-on-save))
produces — versioning below is about the entity tree shape, not individual
component payload shapes.

Scene writes use an atomic temp+rename so readers never observe a partial
file.

## Migration from version 2

Version 2 (entity ids, no instances) is automatically migrated on load:
the file is rewritten as version 3 with `instance: None` on every entity,
and a backup `<file>.v2.bak` is preserved next to it.

## Migration from version 1

Version 1 (no `id` field on entities) is automatically migrated on load:
`SceneDeserializer::load_file` peeks the file's `version`, and for a `1`
it parses it under the legacy shape, assigns each entity a new `EntityId`
(v4, one per entity, in the file's own traversal order), backs up the
original file next to it (`<file>.v1.bak`), and rewrites the file in
place as the current version before continuing to load it normally.

Two independent migrations of the *same* v1 content produce **different**
ids each time — they're UUID v4 (random), not v5 (derived) — because
migration is meant to run once per file; the migrated file it produces
becomes the source of truth for that entity's identity from then on.

An unrecognized version is a hard, explicit error naming the version found
and expected — never silently reinterpreted as another version.

## Compatibility fixtures

Versions 1 and 2 are permanent compatibility fixtures (ADR 0006):
golden-file tests (`crates/engine-assets/tests/scene_migration.rs`) migrate
synthetic legacy scenes and assert the result is a valid current-version
document.
