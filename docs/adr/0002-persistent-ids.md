# ADR 0002: Persistent IDs

Date: 2026-09-14

## Status

Accepted

## Context

Scenes, prefabs, assets, and editor selections need stable identity across reloads, migrations, and source control merges.

## Decision

Projects, source assets, and scene entities use UUID v4 identifiers. Sub-assets use UUID v5 identifiers derived from the parent asset ID plus a stable importer key. Persisted files use lowercase hyphenated UUID strings. ECS `Entity` values are runtime handles and are never persisted as identity.

## Consequences

Scene version 1 remains supported as the baseline fixture. Migrations to stable IDs must be forward-only and preserve backups. Editor features may use `Entity` internally only while the world is loaded.
