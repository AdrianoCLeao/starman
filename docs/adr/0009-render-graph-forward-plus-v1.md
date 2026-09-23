# ADR 0009: Render graph and Forward+ v1

Date: 2026-09-23

## Status

Accepted (supersedes the forward-looking parts of ADR 0004)

## Context

ADR 0004 committed to extract/prepare/queue, a render graph, Tier 1 backends,
and Forward+ as the first production lighting path. M0 only proved device
creation. M4 implements that architecture so M5 (PBR/shadows) can grow without
rewriting the frame loop.

## Decision

1. **Frame pipeline:** each frame runs `extract → prepare → queue → render`
   through a single [`FrameRenderer`]. Gameplay `World` is never bound during
   encode; extract copies into a `RenderWorld`.
2. **Render graph:** a host-owned declarative graph (Rust API) with named
   passes, declared resources, topological execution, and editor inspection.
3. **GPU resources:** generational `GpuHandle`s, staging-belt uploads, and
   deferred destruction after N frames.
4. **Shaders:** WGSL files with `#include`, variant keys, disk cache under
   `.starman/shader-cache/`, errors mapped to original source files.
5. **Capability tiers:** Tier 0 (no compute clusters) and Tier 1 (compute +
   storage buffers for Forward+). Device creation prefers Tier 1; CI expects
   Tier 1 on the supported matrix.
6. **Lighting (M4):** directional + point + spot with clustered light culling.
   No shadow maps (M5).
7. **Editor/runner:** one executor; the editor supplies a `TextureView` and
   camera; the runner owns a surface.
8. **Picking / debug:** GPU entity-ID pass with async readback; debug views
   (depth, normals, clusters, overdraw, light heat) and buffer dumps.

## Consequences

- `ViewportRenderModule` / `RenderState` encode paths collapse onto
  `FrameRenderer`.
- PBR materials, cascaded shadows, IBL, and post FX remain M5 but plug into
  the same graph and shader library.
- Validation layers are not a CI gate; the gate is stable frames on stress
  scenes across Windows, Linux, and macOS.
