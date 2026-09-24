use engine_core::{EngineError, HardeningConfig, Result, SourceAssetId, SubAssetId};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

pub use loaders::{decode_gltf_meshes, extract_gltf_materials};
use loaders::{
    load_material_payload, load_mesh_payload_merged, load_mesh_payloads, load_texture_payload,
};
use pathing::{
    resolve_disk_path as resolve_asset_disk_path, to_asset_path, to_relative_asset_path,
};

mod builtin;
pub mod import;
mod loaders;
mod pathing;
pub mod scene;
mod server_typed;
pub mod typed;

pub use builtin::{
    AssetsPlugin, HdrImageData, HdrImageLoader, MaterialLoader, MeshLoader, TextureLoader,
};
pub mod watch;

pub use server_typed::AssetUpdateReport;
pub use typed::{
    split_sub_key, Asset, AssetLoader, AssetRef, AssetSummary, Assets, LoadContext, LoadState,
    ResolvedSource, StoreStats,
};

static NEXT_ASSET_ID: AtomicU64 = AtomicU64::new(1);

/// Allocates a process-unique [`AssetId`]. Legacy (texture/mesh/material)
/// and typed handles share this id space, so GPU caches keyed by id never
/// collide across asset kinds.
pub(crate) fn next_asset_id() -> AssetId {
    AssetId(NEXT_ASSET_ID.fetch_add(1, Ordering::Relaxed))
}

pub use import::{AssetDatabase, AssetMeta, ImportSummary};
pub use scene::{
    write_scene_ron, InheritedEntity, InstanceLocalEntity, LocalAddedEntity, LocalParent,
    OverrideEntry, SceneDeserializer, SceneEntityData, SceneExternalComponents, SceneFile,
    SceneInstance, SceneInstanceData, SceneSerializer, SceneValue,
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

    /// Builds an id from a raw value. Ids are process-local and never
    /// persisted; this exists for tests and tools that fabricate keys.
    pub fn from_raw(value: u64) -> Self {
        Self(value)
    }
}

pub struct Handle<T> {
    id: AssetId,
    generation: u32,
    marker: PhantomData<fn() -> T>,
}

// Manual impls: derives would require `T: Clone/Eq/...`, but a handle is
// plain data regardless of the asset type it points at.
impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Handle<T> {}

impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.generation == other.generation
    }
}

impl<T> Eq for Handle<T> {}

impl<T> std::hash::Hash for Handle<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self.generation.hash(state);
    }
}

impl<T> std::fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Handle({}#{})", self.id.0, self.generation)
    }
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

/// A contiguous index range drawn with one material.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubMesh {
    pub first_index: u32,
    pub index_count: u32,
    /// Material index inside the source file (glTF), if any.
    pub material: Option<usize>,
}

/// CPU-side mesh payload. Optional vertex streams are either empty or
/// exactly `vertices.len()` long.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshData {
    pub name: String,
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
    #[serde(default)]
    pub aabb_min: [f32; 3],
    #[serde(default)]
    pub aabb_max: [f32; 3],
    /// Per-vertex tangents (xyz, w = bitangent sign), glTF convention.
    #[serde(default)]
    pub tangents: Vec<[f32; 4]>,
    /// Up to four joint indices per vertex (into the bound skin's joints).
    #[serde(default)]
    pub joints: Vec<[u16; 4]>,
    /// Normalized weights matching `joints`.
    #[serde(default)]
    pub weights: Vec<[f32; 4]>,
    /// Skin index inside the source file this mesh is bound to.
    #[serde(default)]
    pub skin: Option<usize>,
    /// Material ranges; empty means one range covering every index.
    #[serde(default)]
    pub submeshes: Vec<SubMesh>,
}

impl MeshData {
    pub fn new(name: impl Into<String>, vertices: Vec<MeshVertex>, indices: Vec<u32>) -> Self {
        let mut mesh = Self {
            name: name.into(),
            vertices,
            indices,
            aabb_min: [0.0; 3],
            aabb_max: [0.0; 3],
            tangents: Vec::new(),
            joints: Vec::new(),
            weights: Vec::new(),
            skin: None,
            submeshes: Vec::new(),
        };
        mesh.recompute_bounds();
        mesh
    }

    /// Whether the mesh carries complete skinning streams.
    pub fn is_skinned(&self) -> bool {
        !self.vertices.is_empty()
            && self.joints.len() == self.vertices.len()
            && self.weights.len() == self.vertices.len()
    }

    pub fn has_tangents(&self) -> bool {
        !self.vertices.is_empty() && self.tangents.len() == self.vertices.len()
    }

    /// Generates per-vertex tangents from positions, normals and UVs
    /// (per-triangle accumulation, Gram-Schmidt orthogonalization and
    /// handedness in `w`). Degenerate UVs fall back to an arbitrary
    /// tangent perpendicular to the normal.
    pub fn generate_tangents(&mut self) {
        let count = self.vertices.len();
        let mut tan = vec![[0.0f32; 3]; count];
        let mut bitan = vec![[0.0f32; 3]; count];
        for triangle in self.indices.chunks_exact(3) {
            let [i0, i1, i2] = [
                triangle[0] as usize,
                triangle[1] as usize,
                triangle[2] as usize,
            ];
            if i0 >= count || i1 >= count || i2 >= count {
                continue;
            }
            let (v0, v1, v2) = (&self.vertices[i0], &self.vertices[i1], &self.vertices[i2]);
            let e1 = sub3(v1.position, v0.position);
            let e2 = sub3(v2.position, v0.position);
            let du1 = v1.uv[0] - v0.uv[0];
            let dv1 = v1.uv[1] - v0.uv[1];
            let du2 = v2.uv[0] - v0.uv[0];
            let dv2 = v2.uv[1] - v0.uv[1];
            let det = du1 * dv2 - du2 * dv1;
            if det.abs() < 1e-12 {
                continue;
            }
            let r = 1.0 / det;
            let t = scale3(sub3(scale3(e1, dv2), scale3(e2, dv1)), r);
            let b = scale3(sub3(scale3(e2, du1), scale3(e1, du2)), r);
            for index in [i0, i1, i2] {
                tan[index] = add3(tan[index], t);
                bitan[index] = add3(bitan[index], b);
            }
        }
        self.tangents = (0..count)
            .map(|index| {
                let n = normalize3(self.vertices[index].normal);
                let t = tan[index];
                // Gram-Schmidt.
                let mut ortho = sub3(t, scale3(n, dot3(n, t)));
                if dot3(ortho, ortho) < 1e-12 {
                    let axis = if n[0].abs() < 0.9 {
                        [1.0, 0.0, 0.0]
                    } else {
                        [0.0, 1.0, 0.0]
                    };
                    ortho = sub3(axis, scale3(n, dot3(n, axis)));
                }
                let ortho = normalize3(ortho);
                let handedness = if dot3(cross3(n, ortho), bitan[index]) < 0.0 {
                    -1.0
                } else {
                    1.0
                };
                [ortho[0], ortho[1], ortho[2], handedness]
            })
            .collect();
    }

    pub fn recompute_bounds(&mut self) {
        if self.vertices.is_empty() {
            self.aabb_min = [0.0; 3];
            self.aabb_max = [0.0; 3];
            return;
        }
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];
        for v in &self.vertices {
            for i in 0..3 {
                min[i] = min[i].min(v.position[i]);
                max[i] = max[i].max(v.position[i]);
            }
        }
        self.aabb_min = min;
        self.aabb_max = max;
    }

    pub fn bounding_sphere_radius(&self) -> f32 {
        let cx = (self.aabb_min[0] + self.aabb_max[0]) * 0.5;
        let cy = (self.aabb_min[1] + self.aabb_max[1]) * 0.5;
        let cz = (self.aabb_min[2] + self.aabb_max[2]) * 0.5;
        let mut r2 = 0.0f32;
        for v in &self.vertices {
            let dx = v.position[0] - cx;
            let dy = v.position[1] - cy;
            let dz = v.position[2] - cz;
            r2 = r2.max(dx * dx + dy * dy + dz * dz);
        }
        r2.sqrt()
    }
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale3(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize3(a: [f32; 3]) -> [f32; 3] {
    let len = dot3(a, a).sqrt();
    if len < 1e-12 {
        [0.0, 1.0, 0.0]
    } else {
        scale3(a, 1.0 / len)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialData {
    pub base_color_factor: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    #[serde(default)]
    pub emissive_factor: [f32; 3],
    #[serde(default = "default_normal_scale")]
    pub normal_scale: f32,
    #[serde(default = "default_occlusion_strength")]
    pub occlusion_strength: f32,
    #[serde(default = "default_alpha_mode")]
    pub alpha_mode: String,
    #[serde(default = "default_alpha_cutoff")]
    pub alpha_cutoff: f32,
    #[serde(default)]
    pub double_sided: bool,
    /// Project-relative texture paths (resolved to handles at prepare time).
    #[serde(default)]
    pub base_color_texture: Option<String>,
    #[serde(default)]
    pub metallic_roughness_texture: Option<String>,
    #[serde(default)]
    pub normal_texture: Option<String>,
    #[serde(default)]
    pub occlusion_texture: Option<String>,
    #[serde(default)]
    pub emissive_texture: Option<String>,
}

fn default_normal_scale() -> f32 {
    1.0
}
fn default_occlusion_strength() -> f32 {
    1.0
}
fn default_alpha_mode() -> String {
    "OPAQUE".to_owned()
}
fn default_alpha_cutoff() -> f32 {
    0.5
}

impl Default for MaterialData {
    fn default() -> Self {
        Self {
            base_color_factor: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0,
            roughness: 1.0,
            emissive_factor: [0.0; 3],
            normal_scale: 1.0,
            occlusion_strength: 1.0,
            alpha_mode: default_alpha_mode(),
            alpha_cutoff: default_alpha_cutoff(),
            double_sided: false,
            base_color_texture: None,
            metallic_roughness_texture: None,
            normal_texture: None,
            occlusion_texture: None,
            emissive_texture: None,
        }
    }
}

pub struct AssetServer {
    root: AssetPath,
    hardening: HardeningConfig,
    next_revision: u64,
    assets: Assets,
    workers: server_typed::LoadWorkers,
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
    /// Lua (or other) scripts changed on disk.
    pub scripts: Vec<PathBuf>,
    /// Native plugin libraries changed on disk.
    pub plugins: Vec<PathBuf>,
    /// Files that disappeared. Loaded payloads are kept.
    pub removed: Vec<PathBuf>,
    /// Assets whose reload failed; their previous payload is kept.
    pub failed: Vec<PathBuf>,
    /// Typed assets (see [`typed`]) whose load or reload finished.
    pub assets: AssetUpdateReport,
}

impl HotReloadReport {
    /// Number of assets whose payload was successfully reloaded.
    pub fn reloaded_count(&self) -> usize {
        self.textures.len() + self.meshes.len() + self.materials.len() + self.assets.loaded.len()
    }
}

impl AssetServer {
    pub fn new(root: impl Into<String>) -> Self {
        let root = AssetPath::new(root);
        let watcher = AssetWatcher::watch_root(Path::new(root.as_str()), WatchConfig::default());

        Self {
            root,
            hardening: HardeningConfig::default(),
            next_revision: 1,
            assets: Assets::new(),
            workers: server_typed::LoadWorkers::new(),
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

    /// The shared typed-asset store. Insert a clone into the ECS world
    /// (hosts do this before installing runtime plugins) so systems and
    /// plugins see the same storage this server fills.
    pub fn assets(&self) -> &Assets {
        &self.assets
    }

    /// Adopts an existing typed-asset store (e.g. one a runtime created and
    /// plugins already registered loaders into).
    pub fn set_assets(&mut self, assets: Assets) {
        self.assets = assets;
    }

    /// Resolves pending typed requests, dispatches their loads to worker
    /// threads and stores finished payloads. Never blocks; call once per
    /// frame.
    pub fn update(&mut self) -> AssetUpdateReport {
        server_typed::process(
            &self.assets,
            &mut self.workers,
            &self.root,
            self.database.as_mut(),
            &self.hardening,
            false,
        )
    }

    /// Like [`Self::update`] but decodes on the calling thread and returns
    /// only when no typed load is in flight. Deterministic headless runs,
    /// tools and tests use this.
    pub fn update_blocking(&mut self) -> AssetUpdateReport {
        let mut report = server_typed::process(
            &self.assets,
            &mut self.workers,
            &self.root,
            self.database.as_mut(),
            &self.hardening,
            true,
        );
        let finished = server_typed::collect_finished(&self.assets, &mut self.workers);
        report.loaded.extend(finished.loaded);
        report.failed.extend(finished.failed);
        report
    }

    /// Requests `relative_path` (`file` or `file#sub_key`) and loads it
    /// synchronously.
    pub fn load_blocking<A: Asset>(&mut self, relative_path: &str) -> Result<Handle<A>> {
        self.load_ref_blocking(&AssetRef::from_path(relative_path))
    }

    /// Requests `asset` and loads it synchronously.
    pub fn load_ref_blocking<A: Asset>(&mut self, asset: &AssetRef) -> Result<Handle<A>> {
        let handle = self.assets.request::<A>(asset);
        self.update_blocking();
        match self.assets.state(handle) {
            Some(LoadState::Loaded) => Ok(handle),
            Some(LoadState::Failed(reason)) => Err(EngineError::AssetLoad {
                path: asset.to_string(),
                reason,
            }),
            other => Err(EngineError::AssetLoad {
                path: asset.to_string(),
                reason: format!("load did not complete ({other:?})"),
            }),
        }
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
        let handle = self.textures.get_or_create(path.clone(), &NEXT_ASSET_ID);

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
        let handle = self.meshes.get_or_create(path.clone(), &NEXT_ASSET_ID);

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
        let handle = self.meshes.get_or_create(registry_path, &NEXT_ASSET_ID);

        let payloads = match load_mesh_payloads(&disk_path, &self.hardening) {
            Ok(payloads) => payloads,
            Err(error) => {
                self.meshes.mark_failed(handle);
                return Err(error);
            }
        };
        let Some(payload) = payloads
            .into_iter()
            .nth(mesh_index)
            .filter(|payload| !payload.vertices.is_empty())
        else {
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
        let handle = self.materials.get_or_create(path.clone(), &NEXT_ASSET_ID);

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

        for change in &changes {
            if matches!(
                change,
                AssetChange::Removed(_) | AssetChange::Script(_) | AssetChange::Plugin(_)
            ) {
                continue;
            }
            let disk_path = server_typed::normalized(change.path());
            server_typed::reload_disk_path(
                &self.assets,
                &mut self.workers,
                &self.hardening,
                &disk_path,
                false,
            );
        }
        report.assets = self.update();

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
                AssetChange::Script(path) => report.scripts.push(path),
                AssetChange::Plugin(path) => report.plugins.push(path),
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
    MaterialData::default()
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
