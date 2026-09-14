# ADR 0003: Assets And Import Cache

Date: 2026-09-14

## Status

Accepted

## Context

Commercial projects need reproducible imports, stable handles, and fast editor startup without treating imported GPU/runtime data as hand-authored source.

## Decision

Source assets live in the project. Each source asset has versioned metadata containing its stable ID and import settings. Imported outputs are content-addressed cache entries with explicit dependency records.

## Consequences

The current asset server remains the M0 reference path loader. M1 must introduce `.meta` files, import records, and dependency tracking without changing source files silently.
