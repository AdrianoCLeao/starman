# ADR 0007: Nested scenes and live instances

Date: 2026-09-23

## Status

Accepted

## Context

M1 established stable entity and asset identities and a flat scene document
(entity hierarchy only). Authoring reusable content requires nested scenes
(prefabs) with overrides, without inventing ephemeral ids or flattening away
the link to the source asset at runtime.

## Decision

1. **Prefab = scene.** Any `.scene.ron` document may be instanced. There is
   no separate prefab extension.
2. **Live instances.** A parent scene stores an instance-root entity with
   `SceneInstanceData` (`scene: SourceAssetId`, overrides, removed, added).
   At load time the runtime expands inherited entities into the world and
   keeps the `SceneInstance` component. Inherited entities are tagged
   `InheritedEntity` and are never written back into the parent document.
3. **Identity.** Native / instance-root / local-added entities keep
   `EntityId` in the owning document. Inherited entities are addressed as
   `(InstancePath, TemplateEntityId)` for overrides; they do not receive a
   new `EntityId` in the parent file.
4. **Nesting.** Unlimited depth. Cycle detection runs on the
   `SourceAssetId` dependency graph and fails with an actionable error.
5. **Overrides.** Field/component overrides stack from leaf instance toward
   the source asset. **Revert** drops local overrides; **Apply** writes them
   into the source scene and clears them; **Promote** moves a local-added
   entity into the source scene.
6. **Format.** Scene version 3. Versions 1 and 2 migrate forward with
   backups (ADR 0006). Scene writes are atomic (temp + rename).

## Consequences

- Serializer/deserializer live with the asset crate; composition
  (expand/resync, diff, apply/revert/promote, cycle checks) lives in
  `engine-scene`.
- Editor and runner must expand instances after a flat deserialize and must
  not materialize inherited entities on save.
- Hot-reload of a source scene resyncs live instances while preserving
  overrides stored on each instance root.
