//! Import pipeline: stable per-asset metadata, content hashing, a
//! content-addressed cache of imported output, and a dependency graph
//! between source assets (ADR 0003).

mod cache;
mod database;
mod dependency_graph;
mod hash;
mod meta;

pub use database::AssetDatabase;
pub use meta::AssetMeta;
