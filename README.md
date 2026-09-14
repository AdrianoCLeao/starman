# Starman

Starman is a modular Rust game engine workspace focused on a custom runtime and an integrated visual editor.

## Project Overview

- Language: Rust (workspace, edition 2021)
- Main runtime stack: bevy_ecs, wgpu, rapier, kira
- Editor stack: eframe/egui + egui_dock
- Author: Adriano Leao

Core workspace members:

- engine-core
- engine-render
- engine-physics
- engine-audio
- engine-input
- engine-assets
- engine-reflect
- engine-reflect-derive
- engine-editor
- engine-diagnostics
- engine-smoke
- engine-benchmarks
- sandbox example app

## Prerequisites

- Rust stable toolchain
- Git
- Native platform linker toolchain

Windows linker requirements:

- Visual Studio Build Tools 2022 with Desktop development with C++
or
- Visual Studio Community with C++ workload and Windows SDK

Toolchain is pinned in [rust-toolchain.toml](rust-toolchain.toml).

## Quick Start

Build everything:

```bash
cargo build --workspace
```

Run the runtime sandbox:

```bash
cargo run -p sandbox
```

Run the editor:

```bash
cargo run -p engine-editor
```

## Daily Development Commands

Fast compile validation:

```bash
cargo check --workspace
```

Run all tests:

```bash
cargo test --workspace
```

Strict linting:

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

Formatting:

```bash
cargo fmt --all
cargo fmt --all -- --check
```

## Focused Commands

Editor-only checks:

```bash
cargo test -p engine-editor
cargo clippy -p engine-editor --all-targets -- -D warnings
```

Run the sandbox quickly after changes:

```bash
cargo run -p sandbox
```

Run the headless smoke test for the native backend:

```bash
cargo run -p engine-smoke -- --backend metal --report target/engine-smoke-report.json
cargo run -p engine-smoke -- --backend vulkan --report target/engine-smoke-report.json
cargo run -p engine-smoke -- --backend dx12 --fallback --report target/engine-smoke-report.json
```

Run Criterion benchmarks:

```bash
cargo bench -p engine-benchmarks --bench baseline
```

Run ignored benchmark/evidence tests when needed:

```bash
cargo test -p engine-core hierarchy_propagation_10k_entities_soft_benchmark -- --ignored --nocapture
cargo test -p engine-assets scene_serializer_serializes_ten_entities_under_fifty_ms -- --ignored --nocapture
```

## Diagnostics

Diagnostics are centralized in `engine-diagnostics`. Logs go to stderr by default, to JSONL files under `logs/<application>/`, and to an in-memory ring buffer used by the editor console. Set `STARMAN_LOG=trace|debug|info|warn|error` to control verbosity; `RUST_LOG` is also honored when `STARMAN_LOG` is not set.

The Play Mode runner emits JSONL diagnostics on stdout. The editor captures runner stdout/stderr and shows structured runner events in the Console tab. Local crash reports are written under `logs/<application>/crashes/` with `report.json` and recent diagnostics when available.

## Architecture Records

M0 architectural decisions live in `docs/adr/`. Start new decisions from `docs/adr/0000-template.md`.
