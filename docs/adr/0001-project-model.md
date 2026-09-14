# ADR 0001: Project Model

Date: 2026-09-14

## Status

Accepted

## Context

Starman is moving from a sample engine toward an editor-first engine that can load projects and plugins dynamically on Windows, Linux, and macOS.

## Decision

A Starman project is a directory with a versioned manifest, source assets, generated import cache, local diagnostics, and plugin declarations. The editor and runtime load a project from its manifest instead of assuming the repository `assets/` directory is the permanent product shape.

## Consequences

The existing `assets/` directory remains the reference project for M0. The full project manifest and import cache become M1 work, but all M0 diagnostics, smoke tests, and docs must avoid hardcoding repository-only assumptions where a project path is intended.
