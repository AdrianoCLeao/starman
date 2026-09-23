use engine_core::{EngineError, HardeningConfig, Result, SourceAssetId, SubAssetId};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use loaders::{
    load_material_payload, load_mesh_payload_merged, load_mesh_payloads, load_texture_payload,
};
use pathing::{
    resolve_disk_path as resolve_asset_disk_path, to_asset_path, to_relative_asset_path,
};

pub mod import;
mod loaders;
mod pathing;
pub mod scene;
pub mod watch;

pub use import::{AssetDatabase, AssetMeta, ImportSummary};
pub use scene::{
    InheritedEntity, InstanceLocalEntity, LocalAddedEntity, LocalParent, OverrideEntry,
    SceneDeserializer, SceneEntityData, SceneExternalComponents, SceneFile, SceneInstance,
    SceneInstanceData, SceneSerializer, SceneValue, write_scene_ron,
};
pub use watch::{AssetChange, AssetWatcher, WatchConfig};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AssetPath(String);

impl AssetPath {
    pub fn new(path: impl Into<String>) -> Self {
        Self(path.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AssetId(u64);

impl AssetId {
    pub fn value(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Handle<T> {
    id: AssetId,
    generation: u32,
    marker: PhantomData<fn() -> T>,
}

impl<T> Handle<T> {
    pub fn id(self) -> AssetId {
        self.id
    }

    pub fn generation(self) -> u32 {
        self.generation
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AssetState {
    #[default]
    Loaded,
    Failed,
}

#[derive(Debug, Clone, Copy)]
struct HandleRecord {
    generation: u32,
    state: AssetState,
}

#[derive(Default)]
struct HandleRegistry {
    path_to_id: HashMap<AssetPath, AssetId>,
    id_to_path: HashMap<AssetId, AssetPath>,
    id_to_record: HashMap<AssetId, HandleRecord>,
    /// The stable [`SourceAssetId`] a handle was loaded from, when an
    /// [`AssetDatabase`] was attached at load time. Drives the typed
    /// (id-based) scene reference format — see `AssetServer::*_source_id`.
    id_to_source: HashMap<AssetId, SourceAssetId>,
    /// For meshes only: the specific [`SubAssetId`] a handle corresponds
    /// to, when unambiguous (see `AssetServer::load_mesh_handle`).
    id_to_sub_source: HashMap<AssetId, SubAssetId>,
}

impl HandleRegistry {
    fn get_or_create<T>(&mut self, path: AssetPath, id_source: &AtomicU64) -> Handle<T> {
        if let Some(id) = self.path_to_id.get(&path).copied() {
            let record = self.id_to_record.get(&id).copied().unwrap_or(HandleRecord {
                generation: 1,
                state: AssetState::Loaded,
            });

            return Handle {
                id,
                generation: record.generation,
                marker: PhantomData,
            };
        }

        let id = AssetId(id_source.fetch_add(1, Ordering::Relaxed));
        self.path_to_id.insert(path.clone(), id);
        self.id_to_path.insert(id, path);
        self.id_to_record.insert(
            id,
            HandleRecord {
                generation: 1,
                state: AssetState::Loaded,
            },
        );

        Handle {
            id,
            generation: 1,
            marker: PhantomData,
        }
    }

    fn state_for<T>(&self, handle: Handle<T>) -> Option<AssetState> {
        let record = self.id_to_record.get(&handle.id)?;
        if record.generation != handle.generation {
            return None;
        }
        Some(record.state)
    }

    fn path_for<T>(&self, handle: Handle<T>) -> Option<&AssetPath> {
        let record = self.id_to_record.get(&handle.id)?;
        if record.generation != handle.generation {
            return None;
        }

        self.id_to_path.get(&handle.id)
    }

    fn mark_failed<T>(&mut self, handle: Handle<T>) {
        if let Some(record) = self.id_to_record.get_mut(&handle.id) {
            record.state = AssetState::Failed;
        }
    }

    fn mark_loaded<T>(&mut self, handle: Handle<T>) {
        if let Some(record) = self.id_to_record.get_mut(&handle.id) {
            record.state = AssetState::Loaded;
        }
    }

    fn record_source_id(&mut self, id: AssetId, source_id: SourceAssetId) {
        self.id_to_source.insert(id, source_id);
    }

    fn source_id_for<T>(&self, handle: Handle<T>) -> Option<SourceAssetId> {
        let record = self.id_to_record.get(&handle.id)?;
        if record.generation != handle.generation {
            return None;
        }
        self.id_to_source.get(&handle.id).copied()
    }

    fn record_sub_source_id(&mut self, id: AssetId, sub_id: SubAssetId) {
        self.id_to_sub_source.insert(id, sub_id);
    }

    fn sub_source_id_for<T>(&self, handle: Handle<T>) -> Option<SubAssetId> {
        let record = self.id_to_record.get(&handle.id)?;
        if record.generation != handle.generation {
            return None;
        }
        self.id_to_sub_source.get(&handle.id).copied()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TextureAsset;

#[derive(Debug, Clone, Copy)]
pub struct MeshAsset;

#[derive(Debug, Clone, Copy)]
pub struct MaterialAsset;

pub type TextureHandle = Handle<TextureAsset>;
pub type MeshHandle = Handle<MeshAsset>;
pub type MaterialHandle = Handle<MaterialAsset>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextureData {
    pub width: u32,
    pub height: u32,
    pub pixels_rgba8: Vec<u8>,
    #[serde(default)]
    pub revision: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshData {
    pub name: String,
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialData {
    pub base_color_factor: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
}

pub struct AssetServer {
    root: AssetPath,
    hardening: HardeningConfig,
    next_id: AtomicU64,
    next_revision: u64,
    textures: HandleRegistry,
    meshes: HandleRegistry,
    materials: HandleRegistry,
    texture_payloads: HashMap<AssetId, TextureData>,
    mesh_payloads: HashMap<AssetId, MeshData>,
    material_payloads: HashMap<AssetId, MaterialData>,
    watcher: AssetWatcher,
    texture_files: HashMap<PathBuf, TextureHandle>,
    mesh_files: HashMap<PathBuf, MeshHandle>,
    material_files: HashMap<PathBuf, MaterialHandle>,
    mesh_revisions: HashMap<AssetId, u64>,
    material_revisions: HashMap<AssetId, u64>,
    database: Option<AssetDatabase>,
}

/// What a call to [`AssetServer::poll_hot_reload`] did. Paths are
/// normalized absolute disk paths.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct HotReloadReport {
    pub textures: Vec<PathBuf>,
    pub meshes: Vec<PathBuf>,
    pub materials: Vec<PathBuf>,
    /// Scenes changed on disk. Nothing is reloaded automatically; the
    /// consumer decides what to do.
    pub scenes: Vec<PathBuf>,
    /// Files that disappeared. Loaded payloads are kept.
    pub removed: Vec<PathBuf>,
    /// Assets whose reload failed; their previous payload is kept.
    pub failed: Vec<PathBuf>,
}

impl HotReloadReport {
    /// Number of assets whose payload was successfully reloaded.
    pub fn reloaded_count(&self) -> usize {
        self.textures.len() + self.meshes.len() + self.materials.len()
    }
}

impl AssetServer {
    pub fn new(root: impl Into<String>) -> Self {
        let root = AssetPath::new(root);
        let watcher = AssetWatcher::watch_root(Path::new(root.as_str()), WatchConfig::default());

        Self {
            root,
            hardening: HardeningConfig::default(),
            next_id: AtomicU64::new(1),
            next_revision: 1,
            textures: HandleRegistry::default(),
            meshes: HandleRegistry::default(),
            materials: HandleRegistry::default(),
            texture_payloads: HashMap::new(),
            mesh_payloads: HashMap::new(),
            material_payloads: HashMap::new(),
            watcher,
            texture_files: HashMap::new(),
            mesh_files: HashMap::new(),
            material_files: HashMap::new(),
            mesh_revisions: HashMap::new(),
            material_revisions: HashMap::new(),
            database: None,
        }
    }

    pub fn root(&self) -> &AssetPath {
        &self.root
    }

    pub fn configure_hardening(&mut self, hardening: HardeningConfig) {
        self.hardening = hardening;
    }

    /// Attaches an [`AssetDatabase`], enabling id-based loads
    /// (`load_*_handle_by_id`) that keep resolving correctly even after a
    /// source asset has been renamed through [`AssetDatabase::rename`].
    pub fn attach_database(&mut self, database: AssetDatabase) {
        self.watcher.set_cache(database.cache());
        self.database = Some(database);
    }

    /// Replaces the file watcher, e.g. to feed it a synthetic event source
    /// in tests or to change its debounce window. If a database is already
    /// attached the new watcher is wired to its import cache.
    pub fn replace_watcher(&mut self, mut watcher: AssetWatcher) {
        if let Some(database) = &self.database {
            watcher.set_cache(database.cache());
        }
        self.watcher = watcher;
    }

    pub fn database(&self) -> Option<&AssetDatabase> {
        self.database.as_ref()
    }

    pub fn database_mut(&mut self) -> Option<&mut AssetDatabase> {
        self.database.as_mut()
    }

    pub fn resolve_path(&self, relative_path: &str) -> Result<AssetPath> {
        let disk_path = self.resolve_disk_path(relative_path)?;
        Ok(to_asset_path(&disk_path))
    }

    fn resolve_disk_path(&self, relative_path: &str) -> Result<PathBuf> {
        resolve_asset_disk_path(&self.root, relative_path)
    }

    /// Best-effort: if a database is attached, ensures `relative_path` is
    /// imported (idempotent — a no-op re-hash when nothing changed) and
    /// returns its stable id. `None` when no database is attached, or the
    /// import itself fails (the caller already has its own error handling
    /// for the load; losing id tracking is not itself fatal).
    fn ensure_source_id(&mut self, relative_path: &str) -> Option<SourceAssetId> {
        self.database.as_mut()?.ensure_imported(relative_path).ok()
    }

    fn resolve_relative_path_for_id(&self, id: SourceAssetId) -> Result<String> {
        let database = self
            .database
            .as_ref()
            .ok_or_else(|| EngineError::AssetLoad {
                path: id.to_string(),
                reason: "no asset database attached to resolve this id".to_owned(),
            })?;

        database
            .resolve_relative_path(id)
            .map(str::to_owned)
            .ok_or_else(|| EngineError::AssetLoad {
                path: id.to_string(),
                reason: "asset database has no known path for this id".to_owned(),
            })
    }

    /// Loads a texture by its stable [`SourceAssetId`] instead
    /// of a raw path, resolving the current path through the attached
    /// [`AssetDatabase`] — the load keeps succeeding even if the source
    /// asset was renamed since the reference was recorded.
    pub fn load_texture_handle_by_id(&mut self, id: SourceAssetId) -> Result<TextureHandle> {
        let relative_path = self.resolve_relative_path_for_id(id)?;
        self.load_texture_handle(&relative_path)
    }

    pub fn load_mesh_handle_by_id(&mut self, id: SourceAssetId) -> Result<MeshHandle> {
        let relative_path = self.resolve_relative_path_for_id(id)?;
        self.load_mesh_handle(&relative_path)
    }

    pub fn load_material_handle_by_id(&mut self, id: SourceAssetId) -> Result<MaterialHandle> {
        let relative_path = self.resolve_relative_path_for_id(id)?;
        self.load_material_handle(&relative_path)
    }

    pub fn load_texture_handle(&mut self, relative_path: &str) -> Result<TextureHandle> {
        let disk_path = self.resolve_disk_path(relative_path)?;
        let path = to_asset_path(&disk_path);
        let handle = self.textures.get_or_create(path.clone(), &self.next_id);

        match load_texture_payload(&disk_path, &self.hardening) {
            Ok(mut payload) => {
                payload.revision = self.allocate_revision();
                self.texture_payloads.insert(handle.id(), payload);
                self.textures.mark_loaded(handle);
                self.texture_files.insert(disk_path.clone(), handle);
            }
            Err(error) => {
                self.textures.mark_failed(handle);
                log::warn!(
                    target: "engine::assets",
                    "skipping texture load {}: {}",
                    disk_path.display(),
                    error
                );
                return Err(error);
            }
        }

        if let Some(source_id) = self.ensure_source_id(relative_path) {
            self.textures.record_source_id(handle.id(), source_id);
        }

        log::trace!(
            target: "engine::assets",
            "Resolved texture handle {} for {}",
            handle.id().value(),
            path.as_str()
        );
        Ok(handle)
    }

    /// Loads the whole mesh file as one flattened, renderable blob — every
    /// mesh (and every primitive within each mesh) in the file merged into
    /// a single vertex/index buffer. This is the long-standing behavior of
    /// this function, unchanged; to address one specific mesh within a
    /// multi-mesh file, use [`Self::load_mesh_handle_by_sub_id`].
    pub fn load_mesh_handle(&mut self, relative_path: &str) -> Result<MeshHandle> {
        let disk_path = self.resolve_disk_path(relative_path)?;
        let path = to_asset_path(&disk_path);
        let handle = self.meshes.get_or_create(path.clone(), &self.next_id);

        match load_mesh_payload_merged(&disk_path, &self.hardening) {
            Ok(payload) => {
                let revision = self.allocate_revision();
                self.mesh_revisions.insert(handle.id(), revision);
                self.mesh_payloads.insert(handle.id(), payload);
                self.meshes.mark_loaded(handle);
                self.mesh_files.insert(disk_path.clone(), handle);
            }
            Err(error) => {
                self.meshes.mark_failed(handle);
                log::warn!(
                    target: "engine::assets",
                    "skipping mesh load {}: {}",
                    disk_path.display(),
                    error
                );
                return Err(error);
            }
        }

        if let Some(source_id) = self.ensure_source_id(relative_path) {
            self.meshes.record_source_id(handle.id(), source_id);

            // When the file has exactly one mesh, the merged handle *is*
            // that mesh: tag it with the sub-asset id too, so saving a
            // reference to it prefers the more specific (and more stable)
            // sub-asset id. A multi-mesh merge has no single sub-asset it
            // corresponds to, so it is left tagged with only the file's id.
            let sub_assets = self
                .database
                .as_ref()
                .map(|database| database.sub_assets_of(source_id).to_vec())
                .unwrap_or_default();
            if let [only] = sub_assets.as_slice() {
                self.meshes.record_sub_source_id(handle.id(), only.id);
            }
        }

        log::trace!(
            target: "engine::assets",
            "Resolved mesh handle {} for {}",
            handle.id().value(),
            path.as_str()
        );
        Ok(handle)
    }

    /// Loads exactly one mesh out of a (possibly multi-mesh) source file,
    /// addressed by its stable [`SubAssetId`], resolved through the
    /// attached [`AssetDatabase`]. Requires a database to be attached.
    pub fn load_mesh_handle_by_sub_id(&mut self, id: SubAssetId) -> Result<MeshHandle> {
        let (source_id, mesh_key) = {
            let database = self
                .database
                .as_ref()
                .ok_or_else(|| EngineError::AssetLoad {
                    path: id.to_string(),
                    reason: "no asset database attached to resolve this sub-asset id".to_owned(),
                })?;
            let (source_id, record) =
                database
                    .resolve_sub_asset(id)
                    .ok_or_else(|| EngineError::AssetLoad {
                        path: id.to_string(),
                        reason: "asset database has no known sub-asset for this id".to_owned(),
                    })?;
            (source_id, record.key.clone())
        };

        let mesh_index = mesh_index_from_key(&mesh_key).ok_or_else(|| EngineError::AssetLoad {
            path: mesh_key.clone(),
            reason: "sub-asset key is not a recognized mesh index".to_owned(),
        })?;

        let relative_path = self.resolve_relative_path_for_id(source_id)?;
        let disk_path = self.resolve_disk_path(&relative_path)?;
        let registry_path = mesh_sub_asset_registry_path(&disk_path, mesh_index);
        let handle = self.meshes.get_or_create(registry_path, &self.next_id);

        let payloads = match load_mesh_payloads(&disk_path, &self.hardening) {
            Ok(payloads) => payloads,
            Err(error) => {
                self.meshes.mark_failed(handle);
                return Err(error);
            }
        };
        let Some(payload) = payloads.into_iter().nth(mesh_index) else {
            self.meshes.mark_failed(handle);
            return Err(EngineError::AssetLoad {
                path: relative_path,
                reason: format!("mesh index {mesh_index} was not found in the file"),
            });
        };

        let revision = self.allocate_revision();
        self.mesh_revisions.insert(handle.id(), revision);
        self.mesh_payloads.insert(handle.id(), payload);
        self.meshes.mark_loaded(handle);
        self.meshes.record_source_id(handle.id(), source_id);
        self.meshes.record_sub_source_id(handle.id(), id);

        log::trace!(
            target: "engine::assets",
            "Resolved mesh handle {} for sub-asset {} ({})",
            handle.id().value(),
            id,
            relative_path
        );
        Ok(handle)
    }

    pub fn load_material_handle(&mut self, relative_path: &str) -> Result<MaterialHandle> {
        let disk_path = self.resolve_disk_path(relative_path)?;
        let path = to_asset_path(&disk_path);
        let handle = self.materials.get_or_create(path.clone(), &self.next_id);

        let (payload, state_ok) = if disk_path.is_file() {
            match load_material_payload(&disk_path) {
                Ok(payload) => (payload, true),
                Err(error) => {
                    log::warn!(
                        target: "engine::assets",
                        "material {} is invalid, using defaults until it is fixed: {}",
                        disk_path.display(),
                        error
                    );
                    (default_material(), false)
                }
            }
        } else {
            (default_material(), true)
        };

        let revision = self.allocate_revision();
        self.material_revisions.insert(handle.id(), revision);
        self.material_payloads.insert(handle.id(), payload);
        if state_ok {
            self.materials.mark_loaded(handle);
        } else {
            self.materials.mark_failed(handle);
        }
        self.material_files.insert(disk_path, handle);

        if let Some(source_id) = self.ensure_source_id(relative_path) {
            self.materials.record_source_id(handle.id(), source_id);
        }

        log::trace!(
            target: "engine::assets",
            "Resolved material handle {} for {}",
            handle.id().value(),
            path.as_str()
        );
        Ok(handle)
    }

    /// The stable [`SourceAssetId`] a handle was loaded from, if an
    /// [`AssetDatabase`] was attached at load time.
    pub fn texture_source_id(&self, handle: TextureHandle) -> Option<SourceAssetId> {
        self.textures.source_id_for(handle)
    }

    pub fn mesh_source_id(&self, handle: MeshHandle) -> Option<SourceAssetId> {
        self.meshes.source_id_for(handle)
    }

    pub fn material_source_id(&self, handle: MaterialHandle) -> Option<SourceAssetId> {
        self.materials.source_id_for(handle)
    }

    /// The specific [`SubAssetId`] a mesh handle corresponds to, when known
    /// and unambiguous (see [`Self::load_mesh_handle`] and
    /// [`Self::load_mesh_handle_by_sub_id`]).
    pub fn mesh_sub_source_id(&self, handle: MeshHandle) -> Option<SubAssetId> {
        self.meshes.sub_source_id_for(handle)
    }

    pub fn mesh_revision(&self, handle: MeshHandle) -> Option<u64> {
        self.mesh_revisions.get(&handle.id()).copied()
    }

    pub fn material_revision(&self, handle: MaterialHandle) -> Option<u64> {
        self.material_revisions.get(&handle.id()).copied()
    }

    pub fn texture_state(&self, handle: TextureHandle) -> Option<AssetState> {
        self.textures.state_for(handle)
    }

    pub fn mesh_state(&self, handle: MeshHandle) -> Option<AssetState> {
        self.meshes.state_for(handle)
    }

    pub fn material_state(&self, handle: MaterialHandle) -> Option<AssetState> {
        self.materials.state_for(handle)
    }

    pub fn texture_relative_path(&self, handle: TextureHandle) -> Option<String> {
        let path = self.textures.path_for(handle)?;
        to_relative_asset_path(&self.root, path)
    }

    pub fn mesh_relative_path(&self, handle: MeshHandle) -> Option<String> {
        let path = self.meshes.path_for(handle)?;
        to_relative_asset_path(&self.root, path)
    }

    pub fn material_relative_path(&self, handle: MaterialHandle) -> Option<String> {
        let path = self.materials.path_for(handle)?;
        to_relative_asset_path(&self.root, path)
    }

    pub fn mark_texture_failed(&mut self, handle: TextureHandle) {
        self.textures.mark_failed(handle);
    }

    pub fn texture_payload(&self, handle: TextureHandle) -> Option<&TextureData> {
        self.texture_payloads.get(&handle.id())
    }

    pub fn mesh_payload(&self, handle: MeshHandle) -> Option<&MeshData> {
        self.mesh_payloads.get(&handle.id())
    }

    pub fn material_payload(&self, handle: MaterialHandle) -> Option<&MaterialData> {
        self.material_payloads.get(&handle.id())
    }

    /// Applies pending file changes: reloads the payload (and bumps the
    /// revision) of every loaded texture, mesh and material whose source
    /// changed, and reports scene changes and removals without acting on
    /// them. A payload that fails to reload keeps its previous contents.
    pub fn poll_hot_reload(&mut self) -> HotReloadReport {
        let changes = self.watcher.poll(self.database.as_mut());
        let mut report = HotReloadReport::default();

        for change in changes {
            match change {
                AssetChange::Texture(path) => {
                    let Some(handle) = self.texture_files.get(&path).copied() else {
                        continue;
                    };
                    match load_texture_payload(&path, &self.hardening) {
                        Ok(mut payload) => {
                            payload.revision = self.allocate_revision();
                            self.texture_payloads.insert(handle.id(), payload);
                            self.textures.mark_loaded(handle);
                            report.textures.push(path);
                        }
                        Err(error) => {
                            self.textures.mark_failed(handle);
                            log::warn!(
                                target: "engine::assets",
                                "failed to hot-reload texture {}: {}",
                                path.display(),
                                error
                            );
                            report.failed.push(path);
                        }
                    }
                }
                AssetChange::Mesh(path) => {
                    let Some(handle) = self.mesh_files.get(&path).copied() else {
                        continue;
                    };
                    match load_mesh_payload_merged(&path, &self.hardening) {
                        Ok(payload) => {
                            let revision = self.allocate_revision();
                            self.mesh_revisions.insert(handle.id(), revision);
                            self.mesh_payloads.insert(handle.id(), payload);
                            self.meshes.mark_loaded(handle);
                            report.meshes.push(path);
                        }
                        Err(error) => {
                            self.meshes.mark_failed(handle);
                            log::warn!(
                                target: "engine::assets",
                                "failed to hot-reload mesh {}: {}",
                                path.display(),
                                error
                            );
                            report.failed.push(path);
                        }
                    }
                }
                AssetChange::Material(path) => {
                    let Some(handle) = self.material_files.get(&path).copied() else {
                        continue;
                    };
                    match load_material_payload(&path) {
                        Ok(payload) => {
                            let revision = self.allocate_revision();
                            self.material_revisions.insert(handle.id(), revision);
                            self.material_payloads.insert(handle.id(), payload);
                            self.materials.mark_loaded(handle);
                            report.materials.push(path);
                        }
                        Err(error) => {
                            self.materials.mark_failed(handle);
                            log::warn!(
                                target: "engine::assets",
                                "failed to hot-reload material {}: {}",
                                path.display(),
                                error
                            );
                            report.failed.push(path);
                        }
                    }
                }
                AssetChange::Scene(path) => report.scenes.push(path),
                AssetChange::Removed(path) => report.removed.push(path),
                AssetChange::Other(_) => {}
            }
        }

        report
    }

    /// Number of textures reloaded by [`Self::poll_hot_reload`].
    #[deprecated(note = "use `poll_hot_reload`, which also reloads meshes and materials")]
    pub fn poll_texture_hot_reload(&mut self) -> usize {
        self.poll_hot_reload().textures.len()
    }

    fn allocate_revision(&mut self) -> u64 {
        let revision = self.next_revision;
        self.next_revision = self.next_revision.saturating_add(1);
        revision
    }
}

fn default_material() -> MaterialData {
    MaterialData {
        base_color_factor: [1.0, 1.0, 1.0, 1.0],
        metallic: 0.0,
        roughness: 1.0,
    }
}

/// Parses a mesh sub-asset's importer key (`"mesh:<index>"`, see
/// `AssetDatabase::extract_mesh_sub_asset_keys`) back into its index.
fn mesh_index_from_key(key: &str) -> Option<usize> {
    key.strip_prefix("mesh:")?.parse::<usize>().ok()
}

/// A handle-registry key for one specific mesh within a file, distinct from
/// the plain-path key [`load_mesh_handle`] uses for the whole-file merge —
/// so the two can coexist as separate handles with separate payloads.
fn mesh_sub_asset_registry_path(disk_path: &Path, mesh_index: usize) -> AssetPath {
    let base = to_asset_path(disk_path);
    AssetPath::new(format!("{}#{mesh_index}", base.as_str()))
}

pub struct AssetModule {
    server: AssetServer,
}

impl AssetModule {
    pub fn new(root: impl Into<String>) -> Self {
        Self {
            server: AssetServer::new(root),
        }
    }

    pub fn load_stub(&self, relative_path: &str) -> Result<AssetPath> {
        let path = self.server.resolve_path(relative_path)?;
        log::trace!(target: "engine::assets", "Loading asset stub: {}", path.as_str());
        Ok(path)
    }

    pub fn configure_hardening(&mut self, hardening: HardeningConfig) {
        self.server.configure_hardening(hardening);
    }

    pub fn load_texture_handle(&mut self, relative_path: &str) -> Result<TextureHandle> {
        self.server.load_texture_handle(relative_path)
    }

    pub fn load_mesh_handle(&mut self, relative_path: &str) -> Result<MeshHandle> {
        self.server.load_mesh_handle(relative_path)
    }

    pub fn load_material_handle(&mut self, relative_path: &str) -> Result<MaterialHandle> {
        self.server.load_material_handle(relative_path)
    }

    pub fn asset_server(&self) -> &AssetServer {
        &self.server
    }

    pub fn asset_server_mut(&mut self) -> &mut AssetServer {
        &mut self.server
    }

    pub fn poll_hot_reload(&mut self) -> HotReloadReport {
        self.server.poll_hot_reload()
    }

    #[deprecated(note = "use `poll_hot_reload`, which also reloads meshes and materials")]
    pub fn poll_texture_hot_reload(&mut self) -> usize {
        self.server.poll_hot_reload().textures.len()
    }

    pub fn supported_formats() -> &'static [&'static str] {
        &["png", "jpeg", "gltf", "glb", "ron", "ogg", "wav", "mp3"]
    }
}

pub fn module_name() -> &'static str {
    "engine-assets"
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::mpsc;
    use std::time::{SystemTime, UNIX_EPOCH};

    use image::{Rgba, RgbaImage};
    use notify::{event::ModifyKind, Event, EventKind};

    use super::{AssetServer, AssetWatcher, WatchConfig};

    struct TempDirGuard {
        path: PathBuf,
    }

    impl TempDirGuard {
        fn new(prefix: &str) -> Self {
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time should be monotonic")
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "motley-engine-assets-{}-{}-{}",
                prefix,
                std::process::id(),
                timestamp
            ));

            std::fs::create_dir_all(&path).expect("temp directory should be created");
            Self { path }
        }
    }

    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    fn write_test_png(path: &Path, rgba: [u8; 4]) {
        let image = RgbaImage::from_pixel(1, 1, Rgba(rgba));
        image.save(path).expect("png should be saved");
    }

    #[test]
    fn poll_texture_hot_reload_reloads_tracked_texture_changes() {
        let guard = TempDirGuard::new("hot-reload-tracked");
        let texture_path = guard.path.join("tracked.png");
        write_test_png(&texture_path, [255, 0, 0, 255]);

        let mut server = AssetServer::new(guard.path.to_string_lossy().to_string());
        let handle = server
            .load_texture_handle("tracked.png")
            .expect("tracked texture should load");
        let initial_payload = server
            .texture_payload(handle)
            .expect("texture payload should exist")
            .clone();

        let (tx, rx) = mpsc::channel();
        server.replace_watcher(AssetWatcher::with_event_source(
            &guard.path,
            rx,
            WatchConfig {
                quiet_window: std::time::Duration::ZERO,
                worker_threads: 1,
            },
        ));

        write_test_png(&texture_path, [0, 255, 0, 255]);
        let event = Event::new(EventKind::Modify(ModifyKind::Any)).add_path(texture_path);
        tx.send(Ok(event)).expect("event should be sent");

        let reload_count = server.poll_hot_reload().textures.len();
        assert_eq!(reload_count, 1);

        let updated_payload = server
            .texture_payload(handle)
            .expect("updated payload should exist");
        assert!(updated_payload.revision > initial_payload.revision);
        assert_ne!(updated_payload.pixels_rgba8, initial_payload.pixels_rgba8);
    }

    #[test]
    fn poll_texture_hot_reload_ignores_untracked_files() {
        let guard = TempDirGuard::new("hot-reload-untracked");
        let tracked_texture_path = guard.path.join("tracked.png");
        let other_texture_path = guard.path.join("other.png");

        write_test_png(&tracked_texture_path, [255, 0, 0, 255]);
        write_test_png(&other_texture_path, [0, 0, 255, 255]);

        let mut server = AssetServer::new(guard.path.to_string_lossy().to_string());
        let handle = server
            .load_texture_handle("tracked.png")
            .expect("tracked texture should load");
        let initial_revision = server
            .texture_payload(handle)
            .expect("tracked payload should exist")
            .revision;

        let (tx, rx) = mpsc::channel();
        server.replace_watcher(AssetWatcher::with_event_source(
            &guard.path,
            rx,
            WatchConfig {
                quiet_window: std::time::Duration::ZERO,
                worker_threads: 1,
            },
        ));

        write_test_png(&other_texture_path, [0, 255, 0, 255]);
        let event = Event::new(EventKind::Modify(ModifyKind::Any)).add_path(other_texture_path);
        tx.send(Ok(event)).expect("event should be sent");

        let reload_count = server.poll_hot_reload().textures.len();
        assert_eq!(reload_count, 0);

        let final_revision = server
            .texture_payload(handle)
            .expect("tracked payload should still exist")
            .revision;
        assert_eq!(final_revision, initial_revision);
    }
}
