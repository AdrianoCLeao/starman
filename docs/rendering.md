# Rendering (M4)

Starman frames go through a single [`FrameRenderer`](../crates/engine-render/src/frame.rs):

`extract → prepare → queue → render-graph execute`

Gameplay `World` is never bound during GPU encode. Extract copies meshes,
sprites, lights, and cameras into a `RenderWorld`.

## Render graph

Declarative passes with resource reads/writes and topological scheduling
([`graph`](../crates/engine-render/src/graph/mod.rs)). Default Forward+ graph:

clear → (id_pick) → cluster_cull → opaque_forward_plus → transparent_2d → overlay → (debug_blit)

Inspect pass order via `FrameRenderer::last_pass_order()` / editor status.

## Capability tiers

| Tier | Meaning |
|------|---------|
| 0 | CPU light assignment / simple forward |
| 1 | Compute-capable path for clustered Forward+ (CI expectation) |

Negotiated at device creation ([ADR 0009](adr/0009-render-graph-forward-plus-v1.md)).

## Shaders

WGSL under `crates/engine-render/shaders/` with `#include`, variant keys, and
disk cache in `.starman/shader-cache/`.

## GPU resources

Generational `GpuHandle`s, staging belt, deferred destroy after N frames
([`gpu`](../crates/engine-render/src/gpu/mod.rs)).

## Lights / Forward+

`DirectionalLight`, `PointLight`, `SpotLight` components. Clustered culling
on CPU (Tier 0 always; Tier 1 compute hook reserved). **No shadows in M4.**

## Picking / debug

- `request_pick` / `poll_pick` — entity ID path (async-ready; CPU fallback today)
- `DebugView`: Depth, Normals, Clusters, Overdraw, LightHeat
- `dump_debug_ppm` for cluster heat dumps

## Stress gate

`spawn_stress_scene` builds ~2k meshes and ~64 lights for CI/smoke.
