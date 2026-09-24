# ADR 0010: PBR, HDR, shadows, and post (M5)

Date: 2026-09-23

## Status

Accepted (extends ADR 0009)

## Context

ADR 0009 delivered extract/prepare/queue, a render graph, and Forward+ scaffolding.
Shading remained Blinn-Phong, color wrote straight to sRGB, and shadows/IBL/post
were deferred. M5 needs a production visual baseline that M6+ can build on.

## Decision

1. **Color path:** scene color is linear HDR `RGBA16Float`. Exposure + ACES tonemap
   (Narkowicz fit) + sRGB OETF resolve to the present/viewport target.
2. **Materials:** metallic/roughness PBR (glTF-faithful Cook-Torrance). Maps:
   baseColor, metallicRoughness, normal, occlusion, emissive; alpha modes Opaque/
   Mask/Blend. glTF import produces Starman materials; `.ron` overrides win per field.
3. **Lighting:** Forward+ clusters uploaded to GPU; directional/point/spot in the BRDF.
   Tier 1 prefers compute cull; Tier 0 keeps CPU cull.
4. **Shadows:** 4-cascade CSM for the primary directional; up to 2 local casters
   (spot atlas / point cubemap) with PCF. Budgets follow `QualityPreset`.
5. **IBL:** equirect HDR env → irradiance + prefiltered specular + BRDF LUT; skybox
   pass. Up to 4 runtime reflection probes with distance blend.
6. **AA / post:** TAA is primary when enabled (MSAA forced to 1×). When TAA is off,
   MSAA 2×/4× applies. SSAO, bloom, and fog are graph passes gated by quality.
7. **Quality:** presets Low / Medium / High / Ultra control shadow res, cascade
   count, local slots, MSAA/TAA, post toggles, probe budget, max lights, LOD bias.
   Tier 0 forces a Low-equivalent subset.
8. **LOD:** import mesh AABB/sphere; authored `LodGroup` + `MSFT_lod` when present.

## Consequences

- Opaque/transparent scene passes target HDR; only tonemap/present writes sRGB.
- Editor and runner share the same graph and quality resolution.
- Validation layers remain non-gating; gate is stable frames + documented budgets.
