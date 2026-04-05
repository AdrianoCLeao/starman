use engine_core::{HardeningConfig, Result};
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    sync::mpsc::{Receiver, TryRecvError},
};

use hot_reload::{build_hot_reload_watcher, is_hot_reload_event};
use loaders::{load_mesh_payload, load_texture_payload};
use pathing::{
    normalize_disk_path, resolve_disk_path as resolve_asset_disk_path, to_asset_path,
    to_relative_asset_path,
};

mod hot_reload;
mod loaders;
mod pathing;
pub mod scene;

pub use scene::{
    SceneDeserializer, SceneEntityData, SceneExternalComponents, SceneFile, SceneSerializer,
    SceneValue,
};

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
    next_texture_revision: u64,
    textures: HandleRegistry,
    meshes: HandleRegistry,
    materials: HandleRegistry,
    texture_payloads: HashMap<AssetId, TextureData>,
    mesh_payloads: HashMap<AssetId, MeshData>,
    material_payloads: HashMap<AssetId, MaterialData>,
    hot_reload_watcher: Option<RecommendedWatcher>,
    hot_reload_rx: Option<Receiver<notify::Result<notify::Event>>>,
    tracked_texture_files: HashMap<PathBuf, TextureHandle>,
}

impl AssetServer {
    pub fn new(root: impl Into<String>) -> Self {
        let (hot_reload_watcher, hot_reload_rx) = build_hot_reload_watcher();

        Self {
            root: AssetPath::new(root),
            hardening: HardeningConfig::default(),
            next_id: AtomicU64::new(1),
            next_texture_revision: 1,
            textures: HandleRegistry::default(),
            meshes: HandleRegistry::default(),
            materials: HandleRegistry::default(),
            texture_payloads: HashMap::new(),
            mesh_payloads: HashMap::new(),
            material_payloads: HashMap::new(),
            hot_reload_watcher,
            hot_reload_rx,
            tracked_texture_files: HashMap::new(),
        }
    }

    pub fn root(&self) -> &AssetPath {
        &self.root
    }

    pub fn configure_hardening(&mut self, hardening: HardeningConfig) {
        self.hardening = hardening;
    }

    pub fn resolve_path(&self, relative_path: &str) -> Result<AssetPath> {
        let disk_path = self.resolve_disk_path(relative_path)?;
        Ok(to_asset_path(&disk_path))
    }

    fn resolve_disk_path(&self, relative_path: &str) -> Result<PathBuf> {
        resolve_asset_disk_path(&self.root, relative_path)
    }

    pub fn load_texture_handle(&mut self, relative_path: &str) -> Result<TextureHandle> {
        let disk_path = self.resolve_disk_path(relative_path)?;
        let path = to_asset_path(&disk_path);
        let handle = self.textures.get_or_create(path.clone(), &self.next_id);

        match load_texture_payload(&disk_path, &self.hardening) {
            Ok(mut payload) => {
                payload.revision = self.allocate_texture_revision();
                self.texture_payloads.insert(handle.id(), payload);
                self.textures.mark_loaded(handle);
                self.register_texture_watch(&disk_path, handle);
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

        log::trace!(
            target: "engine::assets",
            "Resolved texture handle {} for {}",
            handle.id().value(),
            path.as_str()
        );
        Ok(handle)
    }

    pub fn load_mesh_handle(&mut self, relative_path: &str) -> Result<MeshHandle> {
        let disk_path = self.resolve_disk_path(relative_path)?;
        let path = to_asset_path(&disk_path);
        let handle = self.meshes.get_or_create(path.clone(), &self.next_id);

        match load_mesh_payload(&disk_path, &self.hardening) {
            Ok(payload) => {
                self.mesh_payloads.insert(handle.id(), payload);
                self.meshes.mark_loaded(handle);
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

        log::trace!(
            target: "engine::assets",
            "Resolved mesh handle {} for {}",
            handle.id().value(),
            path.as_str()
        );
        Ok(handle)
    }

    pub fn load_material_handle(&mut self, relative_path: &str) -> Result<MaterialHandle> {
        let disk_path = self.resolve_disk_path(relative_path)?;
        let path = to_asset_path(&disk_path);
        let handle = self.materials.get_or_create(path.clone(), &self.next_id);
        self.material_payloads.insert(
            handle.id(),
            MaterialData {
                base_color_factor: [1.0, 1.0, 1.0, 1.0],
                metallic: 0.0,
                roughness: 1.0,
            },
        );
        self.materials.mark_loaded(handle);
        log::trace!(
            target: "engine::assets",
            "Resolved material handle {} for {}",
            handle.id().value(),
            path.as_str()
        );
        Ok(handle)
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

    pub fn poll_texture_hot_reload(&mut self) -> usize {
        let Some(rx) = self.hot_reload_rx.as_ref() else {
            return 0;
        };

        let mut changed_files = Vec::new();
        loop {
            match rx.try_recv() {
                Ok(Ok(event)) => {
                    if !is_hot_reload_event(&event.kind) {
                        continue;
                    }

                    for path in event.paths {
                        changed_files.push(normalize_disk_path(&path));
                    }
                }
                Ok(Err(error)) => {
                    log::warn!(target: "engine::assets", "hot-reload event error: {error}");
                }
                Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => break,
            }
        }

        let mut reload_count = 0;
        for file_path in changed_files {
            let Some(handle) = self.tracked_texture_files.get(&file_path).copied() else {
                continue;
            };

            match load_texture_payload(&file_path, &self.hardening) {
                Ok(mut payload) => {
                    payload.revision = self.allocate_texture_revision();
                    self.texture_payloads.insert(handle.id(), payload);
                    self.textures.mark_loaded(handle);
                    reload_count += 1;
                    log::info!(
                        target: "engine::assets",
                        "hot-reloaded texture {}",
                        file_path.display()
                    );
                }
                Err(error) => {
                    self.textures.mark_failed(handle);
                    log::warn!(
                        target: "engine::assets",
                        "failed to hot-reload texture {}: {}",
                        file_path.display(),
                        error
                    );
                }
            }
        }

        reload_count
    }

    fn allocate_texture_revision(&mut self) -> u64 {
        let revision = self.next_texture_revision;
        self.next_texture_revision = self.next_texture_revision.saturating_add(1);
        revision
    }

    fn register_texture_watch(&mut self, disk_path: &Path, handle: TextureHandle) {
        let normalized = normalize_disk_path(disk_path);
        self.tracked_texture_files
            .insert(normalized.clone(), handle);

        let Some(watcher) = self.hot_reload_watcher.as_mut() else {
            return;
        };

        if let Err(error) = watcher.watch(&normalized, RecursiveMode::NonRecursive) {
            log::warn!(
                target: "engine::assets",
                "failed to watch texture {} for hot-reload: {}",
                normalized.display(),
                error
            );
        }
    }
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

    pub fn poll_texture_hot_reload(&mut self) -> usize {
        self.server.poll_texture_hot_reload()
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

    use super::AssetServer;

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
        server.hot_reload_watcher = None;
        server.hot_reload_rx = Some(rx);

        write_test_png(&texture_path, [0, 255, 0, 255]);
        let event = Event::new(EventKind::Modify(ModifyKind::Any)).add_path(texture_path);
        tx.send(Ok(event)).expect("event should be sent");

        let reload_count = server.poll_texture_hot_reload();
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
        server.hot_reload_watcher = None;
        server.hot_reload_rx = Some(rx);

        write_test_png(&other_texture_path, [0, 255, 0, 255]);
        let event = Event::new(EventKind::Modify(ModifyKind::Any)).add_path(other_texture_path);
        tx.send(Ok(event)).expect("event should be sent");

        let reload_count = server.poll_texture_hot_reload();
        assert_eq!(reload_count, 0);

        let final_revision = server
            .texture_payload(handle)
            .expect("tracked payload should still exist")
            .revision;
        assert_eq!(final_revision, initial_revision);
    }
}
