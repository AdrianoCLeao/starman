# ADR 0006: Compatibility Policy

Date: 2026-09-14

## Status

Accepted

## Context

The project is still before a public baseline, but file formats and diagnostics should not drift accidentally.

## Decision

Rust APIs may break while the engine is pre-public. File formats are versioned from their first committed version. Scene version 1 is a permanent compatibility fixture. Migrations are forward-only and create a backup before modifying authored data.

## Consequences

M0 may correct APIs aggressively, but every persistent format change must come with a migration story and a fixture.
