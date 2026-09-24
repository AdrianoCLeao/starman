# Rendering (M5)

Starman frames go through a single [`FrameRenderer`](../crates/engine-render/src/frame.rs):

`extract → prepare → queue → render-graph execute`

Gameplay `World` is never bound during GPU encode. Scene color is **linear HDR**
(`RGBA16Float`); present/viewport receives **ACES tonemap + sRGB**.

See [ADR 0010](adr/0010-pbr-hdr-visual-v1.md).

## Render graph (default)

clear → shadow_csm → shadow_local → (id_pick) → cluster_cull → opaque_forward_plus →
skybox → transparent_2d → (ssao) → (bloom) → (taa) → tonemap_aces → overlay → (debug_blit)

Quality presets enable/disable post and shadow nodes.

## Color path

1. Opaque/transparent write HDR
2. Optional SSAO / bloom / TAA (quality-gated)
3. `tonemap_aces` (exposure × Narkowicz ACES → sRGB OETF) → surface

When TAA is on, MSAA is forced to 1×.

## Quality presets

| Preset | Shadows | Local | TAA | SSAO/Bloom | Max lights | Probes |
|--------|---------|-------|-----|------------|------------|--------|
| Low | 512 / 2 cascades | 0 | off | off | 32 | 1 |
| Medium | 1024 / 3 | 1 | on | on | 64 | 2 |
| High | 2048 / 4 | 2 | on | on | 128 | 4 |
| Ultra | 2048 / 4 | 2 | on | on | 128 | 4 |

Tier 0 collapses toward Low-equivalent (deterministic fallback).

## Materials / PBR

Metallic-roughness Cook-Torrance (`shaders/common/lighting.wgsl`). `MaterialData`
supports maps (paths), emissive, normal scale, occlusion, alpha modes. glTF
materials extract via `extract_gltf_materials`; `.ron` overrides for authoring.

## Lights / shadows / IBL

- Directional / point / spot uploaded to SSBO; FS evaluates PBR per light
- CSM + local shadow budgets (GPU depth encoding reserved; selection live)
- IBL/probe blend weights + skybox node (env bake cache under `.starman/`)

## LOD

`LodGroup` distance bands + quality `lod_bias`; mesh AABB from import.

## Editor

Status bar: tier, lights, pass graph, **quality combo**, debug view.
