# ADR 0005: Plugin ABI

Date: 2026-09-14

## Status

Accepted

## Context

The engine will load projects and plugins dynamically, with Rust and Lua as scripting priorities.

## Decision

Native plugins use a C ABI. Public plugin boundaries do not expose Rust-owned types, references, panics, or allocator-sensitive ownership. Host objects cross the ABI as opaque handles. Hot reload preserves only explicit serialized plugin state.

## Consequences

The Rust plugin ABI spike in M0 validates version rejection and load/unload mechanics only. The real plugin SDK arrives later.
