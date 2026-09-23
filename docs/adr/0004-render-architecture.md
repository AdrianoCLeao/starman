# ADR 0004: Render Architecture

Date: 2026-09-14

## Status

Superseded by [ADR 0009](0009-render-graph-forward-plus-v1.md) for graph,
Forward+, extract/prepare/queue, and capability tiers. Kept for historical
context of the M0 decision.

## Context

The engine needs a modern renderer without attempting every flagship rendering feature immediately.

## Decision

The renderer will evolve toward extract, prepare, queue, and render graph stages. Native Tier 1 backends are D3D12 on Windows x64, Vulkan on Linux x64, and Metal on macOS x64/arm64. The first production path is Forward+ with PBR.

## Consequences

M0 only proves backend creation and an offscreen pass through `engine-smoke`. Render graph, Forward+, and PBR are later milestones. See ADR 0009 for the M4 contract.
