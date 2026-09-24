//! Generic, typed asset storage and loading (ADR 0020).
//!
//! Any engine subsystem or project plugin can define an asset type by
//! implementing [`Asset`] and an [`AssetLoader`], then registering the
//! loader with [`Assets::register_loader`]. The rest is shared:
//!
//! * typed [`Handle<A>`]s with generational validity and load state;
//! * requests from ECS systems by [`AssetRef`] (stable UUID, with a path
//!   fallback) that never block — [`crate::AssetServer::update`] resolves
//!   them through the asset database and loads them on worker threads;
//! * blocking loads for tools, tests and deterministic headless runs;
//! * hot reload that re-runs the loader and bumps the payload revision;
//! * runtime-created assets ([`Assets::insert`]) for procedural content.
//!
//! [`Assets`] is a cheap, clonable, thread-safe view that lives in the
//! ECS world as a resource *and* inside the `AssetServer`, so systems read
//! payloads without borrowing the server.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use bevy_ecs::system::Resource;
use bevy_reflect::Reflect;
use engine_core::{EngineError, HardeningConfig, Result, SourceAssetId, SubAssetId};
use serde::{Deserialize, Serialize};

use crate::{next_asset_id, AssetId, AssetState, Handle};

/// A loadable asset payload type.
pub trait Asset: Send + Sync + 'static {
    /// Stable, human-readable type name (diagnostics, hot-reload reports).
    const TYPE_NAME: &'static str;
}

/// Everything a loader may consult while decoding one asset.
pub struct LoadContext<'a> {
    /// Absolute path of the source file on disk.
    pub disk_path: &'a Path,
    /// Path relative to the project's assets root (`/`-separated).
    pub relative_path: &'a str,
    /// Sub-asset key inside the file (`"anim:2"`), when addressing one.
    pub sub_key: Option<&'a str>,
    pub hardening: &'a HardeningConfig,
    dependencies: Vec<String>,
}

impl<'a> LoadContext<'a> {
    pub fn new(
        disk_path: &'a Path,
        relative_path: &'a str,
        sub_key: Option<&'a str>,
        hardening: &'a HardeningConfig,
    ) -> Self {
        Self {
            disk_path,
            relative_path,
            sub_key,
            hardening,
            dependencies: Vec::new(),
        }
    }

    /// Declares that this asset reads `relative_path` too (reload edges).
    pub fn add_dependency(&mut self, relative_path: impl Into<String>) {
        self.dependencies.push(relative_path.into());
    }

    pub fn dependencies(&self) -> &[String] {
        &self.dependencies
    }

    /// An `AssetLoad` error pointing at this asset.
    pub fn error(&self, reason: impl Into<String>) -> EngineError {
        EngineError::AssetLoad {
            path: match self.sub_key {
                Some(key) => format!("{}#{key}", self.relative_path),
                None => self.relative_path.to_owned(),
            },
            reason: reason.into(),
        }
    }
}

/// Decodes one asset type from source bytes.
pub trait AssetLoader: Send + Sync + 'static {
    type Asset: Asset;

    /// File-name suffixes this loader claims, without the leading dot and
    /// lowercase (`"anim.ron"`, `"ttf"`). The longest matching suffix wins.
    fn extensions(&self) -> &'static [&'static str];

    fn load(&self, bytes: &[u8], ctx: &mut LoadContext<'_>) -> Result<Self::Asset>;
}

/// A serializable, reflectable reference to an asset (ADR 0002 / 0020).
///
/// `id` is the stable [`SourceAssetId`] or [`SubAssetId`] (lowercase
/// hyphenated UUID) and is authoritative when present; `path` is the
/// assets-root-relative path (optionally `file#sub_key`) used as a fallback
/// and for readability. Components that reference assets store an
/// `AssetRef` field, so they serialize and inspect through reflection.
#[derive(Reflect, Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AssetRef {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub path: String,
}

impl AssetRef {
    pub fn from_path(path: impl Into<String>) -> Self {
        Self {
            id: String::new(),
            path: path.into(),
        }
    }

    pub fn from_id(id: impl fmt::Display) -> Self {
        Self {
            id: id.to_string(),
            path: String::new(),
        }
    }

    pub fn with_path(mut self, path: impl Into<String>) -> Self {
        self.path = path.into();
        self
    }

    pub fn is_empty(&self) -> bool {
        self.id.trim().is_empty() && self.path.trim().is_empty()
    }

    /// The key used to de-duplicate requests for the same asset.
    pub fn request_key(&self) -> String {
        if !self.id.trim().is_empty() {
            format!("id:{}", self.id.trim().to_ascii_lowercase())
        } else {
            format!("path:{}", self.path.trim())
        }
    }

    /// Splits `path` into `(file, sub_key)`.
    pub fn split_path(&self) -> (&str, Option<&str>) {
        split_sub_key(self.path.trim())
    }
}

impl fmt::Display for AssetRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.id.is_empty(), self.path.is_empty()) {
            (false, false) => write!(f, "{} ({})", self.path, self.id),
            (false, true) => f.write_str(&self.id),
            (true, false) => f.write_str(&self.path),
            (true, true) => f.write_str("<none>"),
        }
    }
}

/// Splits `"file.glb#anim:2"` into `("file.glb", Some("anim:2"))`.
pub fn split_sub_key(path: &str) -> (&str, Option<&str>) {
    match path.split_once('#') {
        Some((file, key)) if !key.is_empty() => (file, Some(key)),
        Some((file, _)) => (file, None),
        None => (path, None),
    }
}

/// Load progress of a typed handle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoadState {
    /// Requested, waiting for the server to resolve and dispatch it.
    Pending,
    /// Being decoded on a worker thread.
    Loading,
    Loaded,
    Failed(String),
}

impl LoadState {
    pub fn as_asset_state(&self) -> Option<AssetState> {
        match self {
            Self::Loaded => Some(AssetState::Loaded),
            Self::Failed(_) => Some(AssetState::Failed),
            _ => None,
        }
    }
}

/// Where a typed slot's source lives once resolved.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedSource {
    pub relative_path: String,
    pub disk_path: PathBuf,
    pub sub_key: Option<String>,
    pub source_id: Option<SourceAssetId>,
    pub sub_id: Option<SubAssetId>,
}

struct Slot<A> {
    generation: u32,
    request: Option<AssetRef>,
    source: Option<ResolvedSource>,
    state: LoadState,
    payload: Option<Arc<A>>,
    revision: u64,
    label: String,
}

struct TypedStore<A: Asset> {
    slots: HashMap<AssetId, Slot<A>>,
    by_key: HashMap<String, AssetId>,
}

impl<A: Asset> Default for TypedStore<A> {
    fn default() -> Self {
        Self {
            slots: HashMap::new(),
            by_key: HashMap::new(),
        }
    }
}

/// Type-erased operations the server needs on every typed store.
trait ErasedStore: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn type_name(&self) -> &'static str;
    fn pending(&self) -> Vec<(AssetId, AssetRef)>;
    fn resolve(&mut self, id: AssetId, source: Option<ResolvedSource>, error: Option<String>);
    fn begin_load(&mut self, id: AssetId) -> Option<ResolvedSource>;
    fn complete(
        &mut self,
        id: AssetId,
        result: std::result::Result<Box<dyn Any + Send>, String>,
        revision: u64,
    ) -> bool;
    fn slots_for_disk_path(&self, disk_path: &Path) -> Vec<AssetId>;
    fn stats(&self) -> StoreStats;
    fn list(&self) -> Vec<AssetSummary>;
}

impl<A: Asset> ErasedStore for TypedStore<A> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn type_name(&self) -> &'static str {
        A::TYPE_NAME
    }

    fn pending(&self) -> Vec<(AssetId, AssetRef)> {
        let mut out: Vec<_> = self
            .slots
            .iter()
            .filter(|(_, slot)| slot.state == LoadState::Pending)
            .filter_map(|(id, slot)| slot.request.clone().map(|request| (*id, request)))
            .collect();
        out.sort_by_key(|(id, _)| id.value());
        out
    }

    fn resolve(&mut self, id: AssetId, source: Option<ResolvedSource>, error: Option<String>) {
        if let Some(slot) = self.slots.get_mut(&id) {
            match (source, error) {
                (Some(source), _) => {
                    slot.label = match &source.sub_key {
                        Some(key) => format!("{}#{key}", source.relative_path),
                        None => source.relative_path.clone(),
                    };
                    slot.source = Some(source);
                }
                (None, Some(error)) => slot.state = LoadState::Failed(error),
                (None, None) => slot.state = LoadState::Failed("unresolved".to_owned()),
            }
        }
    }

    fn begin_load(&mut self, id: AssetId) -> Option<ResolvedSource> {
        let slot = self.slots.get_mut(&id)?;
        let source = slot.source.clone()?;
        if slot.payload.is_none() {
            slot.state = LoadState::Loading;
        }
        Some(source)
    }

    fn complete(
        &mut self,
        id: AssetId,
        result: std::result::Result<Box<dyn Any + Send>, String>,
        revision: u64,
    ) -> bool {
        let Some(slot) = self.slots.get_mut(&id) else {
            return false;
        };
        match result {
            Ok(payload) => match payload.downcast::<A>() {
                Ok(payload) => {
                    slot.payload = Some(Arc::new(*payload));
                    slot.state = LoadState::Loaded;
                    slot.revision = revision;
                    true
                }
                Err(_) => {
                    slot.state = LoadState::Failed("loader produced the wrong type".to_owned());
                    false
                }
            },
            Err(error) => {
                // Keep a previous payload on reload failure; only the state
                // reports the error.
                slot.state = LoadState::Failed(error);
                false
            }
        }
    }

    fn slots_for_disk_path(&self, disk_path: &Path) -> Vec<AssetId> {
        self.slots
            .iter()
            .filter(|(_, slot)| {
                slot.source
                    .as_ref()
                    .is_some_and(|source| source.disk_path == disk_path)
            })
            .map(|(id, _)| *id)
            .collect()
    }

    fn stats(&self) -> StoreStats {
        let mut stats = StoreStats {
            type_name: A::TYPE_NAME,
            ..StoreStats::default()
        };
        for slot in self.slots.values() {
            match slot.state {
                LoadState::Pending | LoadState::Loading => stats.in_flight += 1,
                LoadState::Loaded => stats.loaded += 1,
                LoadState::Failed(_) => stats.failed += 1,
            }
        }
        stats
    }

    fn list(&self) -> Vec<AssetSummary> {
        let mut out: Vec<_> = self
            .slots
            .iter()
            .map(|(id, slot)| AssetSummary {
                id: *id,
                type_name: A::TYPE_NAME,
                label: slot.label.clone(),
                state: slot.state.clone(),
                revision: slot.revision,
            })
            .collect();
        out.sort_by_key(|summary| summary.id.value());
        out
    }
}

/// Per-type load counters.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StoreStats {
    pub type_name: &'static str,
    pub loaded: usize,
    pub in_flight: usize,
    pub failed: usize,
}

/// One row of [`Assets::list`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssetSummary {
    pub id: AssetId,
    pub type_name: &'static str,
    pub label: String,
    pub state: LoadState,
    pub revision: u64,
}

type ErasedLoadFn =
    dyn Fn(&[u8], &mut LoadContext<'_>) -> Result<Box<dyn Any + Send>> + Send + Sync;

/// A registered loader, type-erased.
#[derive(Clone)]
pub(crate) struct LoaderEntry {
    pub asset_type: TypeId,
    pub type_name: &'static str,
    pub extensions: &'static [&'static str],
    pub load: Arc<ErasedLoadFn>,
}

#[derive(Default)]
struct AssetsInner {
    stores: HashMap<TypeId, Box<dyn ErasedStore>>,
    loaders: Vec<LoaderEntry>,
    next_revision: u64,
}

impl AssetsInner {
    fn store<A: Asset>(&self) -> Option<&TypedStore<A>> {
        self.stores
            .get(&TypeId::of::<A>())
            .and_then(|store| store.as_any().downcast_ref())
    }

    fn store_mut<A: Asset>(&mut self) -> &mut TypedStore<A> {
        self.stores
            .entry(TypeId::of::<A>())
            .or_insert_with(|| Box::<TypedStore<A>>::default())
            .as_any_mut()
            .downcast_mut()
            .expect("typed store type id matches")
    }

    fn allocate_revision(&mut self) -> u64 {
        self.next_revision += 1;
        self.next_revision
    }
}

/// Shared typed asset storage. Clone freely; all clones see the same data.
#[derive(Resource, Clone, Default)]
pub struct Assets {
    inner: Arc<RwLock<AssetsInner>>,
}

impl fmt::Debug for Assets {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Assets")
            .field("types", &self.stats().len())
            .finish()
    }
}

impl Assets {
    pub fn new() -> Self {
        Self::default()
    }

    fn read(&self) -> RwLockReadGuard<'_, AssetsInner> {
        self.inner
            .read()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    fn write(&self) -> RwLockWriteGuard<'_, AssetsInner> {
        self.inner
            .write()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    /// Registers `loader`. Registering a second loader for the same type
    /// replaces the first (a project plugin can override an engine loader).
    pub fn register_loader<L: AssetLoader>(&self, loader: L) {
        let extensions = loader.extensions();
        let loader = Arc::new(loader);
        let entry = LoaderEntry {
            asset_type: TypeId::of::<L::Asset>(),
            type_name: <L::Asset as Asset>::TYPE_NAME,
            extensions,
            load: Arc::new(move |bytes, ctx| {
                loader
                    .load(bytes, ctx)
                    .map(|asset| Box::new(asset) as Box<dyn Any + Send>)
            }),
        };
        let mut inner = self.write();
        inner
            .loaders
            .retain(|existing| existing.asset_type != entry.asset_type);
        inner.loaders.push(entry);
        inner.store_mut::<L::Asset>();
    }

    pub fn has_loader<A: Asset>(&self) -> bool {
        self.read()
            .loaders
            .iter()
            .any(|entry| entry.asset_type == TypeId::of::<A>())
    }

    pub(crate) fn loader_for_type(&self, asset_type: TypeId) -> Option<LoaderEntry> {
        self.read()
            .loaders
            .iter()
            .find(|entry| entry.asset_type == asset_type)
            .cloned()
    }

    /// The registered asset type name claiming `path`'s suffix, if any.
    pub fn type_name_for_path(&self, path: &str) -> Option<&'static str> {
        let lower = path.to_ascii_lowercase();
        self.read()
            .loaders
            .iter()
            .filter_map(|entry| {
                entry
                    .extensions
                    .iter()
                    .filter(|ext| lower.ends_with(&format!(".{ext}")))
                    .map(|ext| ext.len())
                    .max()
                    .map(|len| (len, entry.type_name))
            })
            .max_by_key(|(len, _)| *len)
            .map(|(_, name)| name)
    }

    /// Requests `asset` without blocking. Repeated requests for the same
    /// reference return the same handle. An empty reference yields a handle
    /// that is immediately `Failed`.
    pub fn request<A: Asset>(&self, asset: &AssetRef) -> Handle<A> {
        let key = asset.request_key();
        {
            let inner = self.read();
            if let Some(store) = inner.store::<A>() {
                if let Some(id) = store.by_key.get(&key) {
                    let generation = store.slots.get(id).map(|slot| slot.generation).unwrap_or(1);
                    return Handle::from_parts(*id, generation);
                }
            }
        }
        let mut inner = self.write();
        let store = inner.store_mut::<A>();
        if let Some(id) = store.by_key.get(&key) {
            let generation = store.slots.get(id).map(|slot| slot.generation).unwrap_or(1);
            return Handle::from_parts(*id, generation);
        }
        let id = next_asset_id();
        let state = if asset.is_empty() {
            LoadState::Failed("empty asset reference".to_owned())
        } else {
            LoadState::Pending
        };
        store.slots.insert(
            id,
            Slot {
                generation: 1,
                request: Some(asset.clone()),
                source: None,
                state,
                payload: None,
                revision: 0,
                label: asset.to_string(),
            },
        );
        store.by_key.insert(key, id);
        Handle::from_parts(id, 1)
    }

    /// Requests by assets-root-relative path (`file` or `file#sub_key`).
    pub fn request_path<A: Asset>(&self, relative_path: &str) -> Handle<A> {
        self.request(&AssetRef::from_path(relative_path))
    }

    /// Inserts a runtime-created asset under `key` (replacing any payload
    /// already stored under that key) and returns its handle.
    pub fn insert<A: Asset>(&self, key: &str, asset: A) -> Handle<A> {
        let mut inner = self.write();
        let revision = inner.allocate_revision();
        let store = inner.store_mut::<A>();
        let full_key = format!("mem:{key}");
        if let Some(id) = store.by_key.get(&full_key).copied() {
            let slot = store.slots.get_mut(&id).expect("indexed slot exists");
            slot.payload = Some(Arc::new(asset));
            slot.state = LoadState::Loaded;
            slot.revision = revision;
            return Handle::from_parts(id, slot.generation);
        }
        let id = next_asset_id();
        store.slots.insert(
            id,
            Slot {
                generation: 1,
                request: None,
                source: None,
                state: LoadState::Loaded,
                payload: Some(Arc::new(asset)),
                revision,
                label: key.to_owned(),
            },
        );
        store.by_key.insert(full_key, id);
        Handle::from_parts(id, 1)
    }

    /// Replaces the payload of an existing handle (editor documents apply
    /// edits this way so every user sees the new revision).
    pub fn replace<A: Asset>(&self, handle: Handle<A>, asset: A) -> bool {
        let mut inner = self.write();
        let revision = inner.allocate_revision();
        let store = inner.store_mut::<A>();
        match store.slots.get_mut(&handle.id()) {
            Some(slot) if slot.generation == handle.generation() => {
                slot.payload = Some(Arc::new(asset));
                slot.state = LoadState::Loaded;
                slot.revision = revision;
                true
            }
            _ => false,
        }
    }

    /// The payload, when loaded (also after a failed *re*load, which keeps
    /// the previous payload).
    pub fn get<A: Asset>(&self, handle: Handle<A>) -> Option<Arc<A>> {
        let inner = self.read();
        let slot = inner.store::<A>()?.slots.get(&handle.id())?;
        if slot.generation != handle.generation() {
            return None;
        }
        slot.payload.clone()
    }

    pub fn state<A: Asset>(&self, handle: Handle<A>) -> Option<LoadState> {
        let inner = self.read();
        let slot = inner.store::<A>()?.slots.get(&handle.id())?;
        (slot.generation == handle.generation()).then(|| slot.state.clone())
    }

    /// Monotonic revision of the payload (0 until first load).
    pub fn revision<A: Asset>(&self, handle: Handle<A>) -> u64 {
        let inner = self.read();
        inner
            .store::<A>()
            .and_then(|store| store.slots.get(&handle.id()))
            .filter(|slot| slot.generation == handle.generation())
            .map(|slot| slot.revision)
            .unwrap_or(0)
    }

    /// Where a handle was resolved to, once resolved.
    pub fn source<A: Asset>(&self, handle: Handle<A>) -> Option<ResolvedSource> {
        let inner = self.read();
        let slot = inner.store::<A>()?.slots.get(&handle.id())?;
        (slot.generation == handle.generation())
            .then(|| slot.source.clone())
            .flatten()
    }

    /// Every handle of type `A` with its load state.
    pub fn handles<A: Asset>(&self) -> Vec<(Handle<A>, LoadState)> {
        let inner = self.read();
        let Some(store) = inner.store::<A>() else {
            return Vec::new();
        };
        let mut out: Vec<_> = store
            .slots
            .iter()
            .map(|(id, slot)| (Handle::from_parts(*id, slot.generation), slot.state.clone()))
            .collect();
        out.sort_by_key(|(handle, _)| handle.id().value());
        out
    }

    /// Counts per registered type.
    pub fn stats(&self) -> Vec<StoreStats> {
        let inner = self.read();
        let mut out: Vec<_> = inner.stores.values().map(|store| store.stats()).collect();
        out.sort_by_key(|stats| stats.type_name);
        out
    }

    /// Every slot of every type (asset browser / diagnostics).
    pub fn list(&self) -> Vec<AssetSummary> {
        let inner = self.read();
        let mut out: Vec<_> = inner
            .stores
            .values()
            .flat_map(|store| store.list())
            .collect();
        out.sort_by_key(|summary| summary.id.value());
        out
    }

    /// Whether any request is still pending or loading.
    pub fn has_in_flight(&self) -> bool {
        self.stats().iter().any(|stats| stats.in_flight > 0)
    }

    // ---- server-side plumbing -------------------------------------------

    pub(crate) fn pending_requests(&self) -> Vec<(TypeId, AssetId, AssetRef)> {
        let inner = self.read();
        let mut out = Vec::new();
        for (type_id, store) in &inner.stores {
            for (id, request) in store.pending() {
                out.push((*type_id, id, request));
            }
        }
        out.sort_by_key(|(_, id, _)| id.value());
        out
    }

    pub(crate) fn resolve_request(
        &self,
        asset_type: TypeId,
        id: AssetId,
        source: std::result::Result<ResolvedSource, String>,
    ) {
        let mut inner = self.write();
        if let Some(store) = inner.stores.get_mut(&asset_type) {
            match source {
                Ok(source) => store.resolve(id, Some(source), None),
                Err(error) => store.resolve(id, None, Some(error)),
            }
        }
    }

    pub(crate) fn begin_load(&self, asset_type: TypeId, id: AssetId) -> Option<ResolvedSource> {
        let mut inner = self.write();
        inner.stores.get_mut(&asset_type)?.begin_load(id)
    }

    /// Stores a finished load; returns the type name when it succeeded.
    pub(crate) fn complete_load(
        &self,
        asset_type: TypeId,
        id: AssetId,
        result: std::result::Result<Box<dyn Any + Send>, String>,
    ) -> Option<&'static str> {
        let mut inner = self.write();
        let revision = inner.allocate_revision();
        let store = inner.stores.get_mut(&asset_type)?;
        let type_name = store.type_name();
        store.complete(id, result, revision).then_some(type_name)
    }

    pub(crate) fn slots_for_disk_path(&self, disk_path: &Path) -> Vec<(TypeId, AssetId)> {
        let inner = self.read();
        let mut out = Vec::new();
        for (type_id, store) in &inner.stores {
            for id in store.slots_for_disk_path(disk_path) {
                out.push((*type_id, id));
            }
        }
        out
    }
}

impl<A> Handle<A> {
    pub(crate) fn from_parts(id: AssetId, generation: u32) -> Self {
        Self {
            id,
            generation,
            marker: PhantomData,
        }
    }
}

#[cfg(test)]
#[path = "typed_tests.rs"]
mod tests;
