//! Import pipeline: stable per-asset metadata, content hashing, a
//! content-addressed cache of imported output, and a dependency graph
//! between source assets (ADR 0003).

mod atomic;
mod cache;
mod database;
mod dependency_graph;
mod hash;
mod meta;

pub(crate) use atomic::is_temp_file;
pub use atomic::write_atomic;
pub(crate) use cache::ImportedCache;
pub(crate) use database::ImportOutcome;
pub use database::{AssetDatabase, ImportSummary};
pub(crate) use hash::hash_bytes;
pub use meta::AssetMeta;
pub(crate) use meta::META_SUFFIX;
