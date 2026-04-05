use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::io::{self, ErrorKind};
use std::path::{Component, Path, PathBuf};
use std::sync::{
    mpsc::{self, Receiver, Sender, TryRecvError},
    Arc, Mutex,
};

use egui::TextureHandle;
use engine_math::glam::{Vec2, Vec3, Vec4};
use engine_math::Mat4;
use gltf::buffer::Data;
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};

use crate::config::{AssetBrowserConfig, AssetBrowserViewModeConfig};

const THUMBNAIL_EDGE_PX: u32 = 64;
const MAX_THUMBNAIL_CACHE_ITEMS: usize = 512;
const MAX_PENDING_THUMBNAIL_JOBS: usize = 64;
const THUMBNAIL_WORKER_COUNT: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AssetKind {
    Texture,
    Mesh,
    Audio,
    Scene,
    Other,
}

impl AssetKind {
    pub(crate) fn icon(self, is_directory: bool) -> &'static str {
        if is_directory {
            return "[DIR]";
        }

        match self {
            Self::Texture => "[IMG]",
            Self::Mesh => "[MESH]",
            Self::Audio => "[AUD]",
            Self::Scene => "[SCN]",
            Self::Other => "[FILE]",
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct AssetEntry {
    pub relative_path: String,
    pub name: String,
    pub is_directory: bool,
    pub kind: AssetKind,
    pub file_size: u64,
}

#[derive(Clone, Debug)]
struct ThumbnailImageData {
    width: usize,
    height: usize,
    rgba: Vec<u8>,
}

#[derive(Clone, Debug)]
struct ThumbnailResult {
    relative_path: String,
    image: Option<ThumbnailImageData>,
}

#[derive(Clone, Debug)]
struct ThumbnailJob {
    relative_path: String,
    disk_path: PathBuf,
    kind: AssetKind,
}

#[derive(Clone, Copy, Debug)]
struct ProjectedVertex {
    screen: Vec2,
    depth: f32,
}

pub(crate) struct AssetBrowserState {
    root_path: PathBuf,
    current_path: PathBuf,
    entries: Vec<AssetEntry>,
    selected_relative_path: Option<String>,
    filter: String,
    view_mode: AssetBrowserViewModeConfig,
    dirty: bool,
    _watcher: Option<RecommendedWatcher>,
    watcher_rx: Option<Receiver<notify::Result<Event>>>,
    thumbnail_job_tx: Sender<ThumbnailJob>,
    thumbnail_workers_available: bool,
    thumbnail_rx: Receiver<ThumbnailResult>,
    thumbnail_cache: HashMap<String, TextureHandle>,
    thumbnail_pending: HashSet<String>,
    thumbnail_failed: HashSet<String>,
    thumbnail_order: VecDeque<String>,
}

impl AssetBrowserState {
    pub(crate) fn new(root_path: PathBuf, config: &AssetBrowserConfig) -> Self {
        Self::new_with_watcher(root_path, config, true)
    }

    fn new_with_watcher(
        root_path: PathBuf,
        config: &AssetBrowserConfig,
        enable_watcher: bool,
    ) -> Self {
        let root_path = normalize_disk_path(&root_path);
        let (watcher, watcher_rx) = if enable_watcher {
            setup_recursive_watcher(&root_path)
        } else {
            (None, None)
        };
        let (thumbnail_job_tx, thumbnail_job_rx) = mpsc::channel();
        let (thumbnail_tx, thumbnail_rx) = mpsc::channel();
        let thumbnail_workers_available =
            spawn_thumbnail_workers(thumbnail_job_rx, thumbnail_tx, THUMBNAIL_WORKER_COUNT);

        let mut state = Self {
            current_path: root_path.clone(),
            root_path,
            entries: Vec::new(),
            selected_relative_path: None,
            filter: config.filter.clone(),
            view_mode: config.view_mode,
            dirty: true,
            _watcher: watcher,
            watcher_rx,
            thumbnail_job_tx,
            thumbnail_workers_available,
            thumbnail_rx,
            thumbnail_cache: HashMap::new(),
            thumbnail_pending: HashSet::new(),
            thumbnail_failed: HashSet::new(),
            thumbnail_order: VecDeque::new(),
        };

        state.set_current_relative_path(config.current_relative_path.as_deref());
        let _ = state.rescan_if_dirty();

        state
    }

    pub(crate) fn to_config(&self) -> AssetBrowserConfig {
        AssetBrowserConfig {
            current_relative_path: self.current_relative_path(),
            filter: self.filter.clone(),
            view_mode: self.view_mode,
        }
    }

    pub(crate) fn process_file_events(&mut self) -> usize {
        let mut relevant_events = 0;
        let mut touched_paths: Vec<PathBuf> = Vec::new();

        {
            let Some(watcher_rx) = self.watcher_rx.as_ref() else {
                return 0;
            };

            loop {
                match watcher_rx.try_recv() {
                    Ok(Ok(event)) => {
                        if is_relevant_fs_event(&event.kind) {
                            self.dirty = true;
                            relevant_events += 1;
                            touched_paths.extend(event.paths);
                        }
                    }
                    Ok(Err(error)) => {
                        log::warn!(
                            target: "engine::editor",
                            "asset browser watcher error: {}",
                            error
                        );
                    }
                    Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => break,
                }
            }
        }

        for path in touched_paths {
            self.invalidate_thumbnail_for_disk_path(&path);
        }

        relevant_events
    }

    pub(crate) fn process_thumbnail_results(&mut self, ctx: &egui::Context) -> usize {
        let mut loaded = 0;

        while let Ok(result) = self.thumbnail_rx.try_recv() {
            let relative_path = result.relative_path;
            self.thumbnail_pending.remove(&relative_path);

            if !self
                .resolve_relative_path(&relative_path)
                .is_some_and(|path| path.exists())
            {
                self.invalidate_thumbnail_for_relative_path(&relative_path);
                continue;
            }

            match result.image {
                Some(image) => {
                    let color_image = egui::ColorImage::from_rgba_unmultiplied(
                        [image.width, image.height],
                        &image.rgba,
                    );
                    let texture = ctx.load_texture(
                        format!("asset-thumbnail:{}", relative_path),
                        color_image,
                        egui::TextureOptions::LINEAR,
                    );

                    self.thumbnail_failed.remove(&relative_path);
                    self.thumbnail_cache.insert(relative_path.clone(), texture);
                    self.thumbnail_order.retain(|key| key != &relative_path);
                    self.thumbnail_order.push_back(relative_path);
                    loaded += 1;
                }
                None => {
                    self.thumbnail_failed.insert(relative_path);
                }
            }
        }

        self.enforce_thumbnail_cache_limit();
        loaded
    }

    pub(crate) fn request_thumbnail_for_entry(&mut self, entry: &AssetEntry) -> bool {
        if !should_render_thumbnail(entry) {
            return false;
        }

        if self.thumbnail_pending.len() >= MAX_PENDING_THUMBNAIL_JOBS {
            return false;
        }

        if self.thumbnail_cache.contains_key(&entry.relative_path)
            || self.thumbnail_pending.contains(&entry.relative_path)
            || self.thumbnail_failed.contains(&entry.relative_path)
        {
            return false;
        }

        if !self.thumbnail_workers_available {
            self.thumbnail_failed.insert(entry.relative_path.clone());
            return false;
        }

        let relative_path = entry.relative_path.clone();
        let Some(disk_path) = self.resolve_relative_path(&entry.relative_path) else {
            self.thumbnail_failed.insert(entry.relative_path.clone());
            return false;
        };

        self.thumbnail_pending.insert(relative_path);

        let send_result = self.thumbnail_job_tx.send(ThumbnailJob {
            relative_path: entry.relative_path.clone(),
            disk_path,
            kind: entry.kind,
        });

        if send_result.is_err() {
            self.thumbnail_pending.remove(&entry.relative_path);
            self.thumbnail_failed.insert(entry.relative_path.clone());
            return false;
        }

        true
    }

    pub(crate) fn thumbnail_texture_id(&self, relative_path: &str) -> Option<egui::TextureId> {
        self.thumbnail_cache
            .get(relative_path)
            .map(TextureHandle::id)
    }

    pub(crate) fn rescan_if_dirty(&mut self) -> std::io::Result<bool> {
        if !self.dirty {
            return Ok(false);
        }

        self.dirty = false;
        self.entries = self.collect_entries()?;
        self.validate_selected_path();
        Ok(true)
    }

    pub(crate) fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    pub(crate) fn current_relative_path(&self) -> Option<String> {
        if normalize_disk_path(&self.current_path) == normalize_disk_path(&self.root_path) {
            return None;
        }

        to_relative_slash_path(&self.root_path, &self.current_path)
    }

    pub(crate) fn set_current_relative_path(&mut self, relative: Option<&str>) {
        let Some(relative) = relative else {
            self.current_path = self.root_path.clone();
            self.dirty = true;
            return;
        };

        if relative.trim().is_empty() {
            self.current_path = self.root_path.clone();
            self.dirty = true;
            return;
        }

        let Some(candidate) = self.resolve_relative_directory(relative) else {
            self.current_path = self.root_path.clone();
            self.dirty = true;
            return;
        };

        if candidate.is_dir() {
            self.current_path = candidate;
            self.dirty = true;
        }
    }

    pub(crate) fn breadcrumbs(&self) -> Vec<(String, Option<String>)> {
        let mut breadcrumbs = Vec::new();
        breadcrumbs.push(("assets".to_owned(), None));

        let Ok(relative) = self.current_path.strip_prefix(&self.root_path) else {
            return breadcrumbs;
        };

        let mut running = PathBuf::new();
        for component in relative.components() {
            let Component::Normal(name) = component else {
                continue;
            };

            running.push(name);
            let label = name.to_string_lossy().to_string();
            let path = running.to_string_lossy().replace('\\', "/");
            breadcrumbs.push((label, Some(path)));
        }

        breadcrumbs
    }

    pub(crate) fn selected_relative_path(&self) -> Option<&str> {
        self.selected_relative_path.as_deref()
    }

    pub(crate) fn set_selected_relative_path(&mut self, relative_path: Option<String>) {
        self.selected_relative_path = relative_path;
        self.validate_selected_path();
    }

    pub(crate) fn selected_file_name(&self) -> Option<String> {
        self.selected_relative_path
            .as_deref()
            .and_then(|relative| Path::new(relative).file_name())
            .and_then(|name| name.to_str())
            .map(str::to_owned)
    }

    pub(crate) fn create_directory_in_current(
        &mut self,
        directory_name: &str,
    ) -> io::Result<String> {
        let validated_name = validate_asset_name(directory_name)?;
        let directory_path = self.current_path.join(validated_name);

        if directory_path.exists() {
            return Err(io::Error::new(
                ErrorKind::AlreadyExists,
                format!(
                    "cannot create folder '{}': target '{}' already exists",
                    validated_name,
                    directory_path.display()
                ),
            ));
        }

        fs::create_dir(&directory_path)?;

        let relative_path =
            to_relative_slash_path(&self.root_path, &directory_path).ok_or_else(|| {
                io::Error::new(
                    ErrorKind::InvalidInput,
                    format!(
                        "created directory '{}' escaped root",
                        directory_path.display()
                    ),
                )
            })?;

        self.selected_relative_path = Some(relative_path.clone());
        self.dirty = true;
        let _ = self.rescan_if_dirty()?;

        Ok(relative_path)
    }

    pub(crate) fn create_file_in_current(
        &mut self,
        file_name: &str,
        contents: &str,
    ) -> io::Result<String> {
        let validated_name = validate_asset_name(file_name)?;
        let file_path = self.current_path.join(validated_name);

        if file_path.exists() {
            return Err(io::Error::new(
                ErrorKind::AlreadyExists,
                format!(
                    "cannot create file '{}': target '{}' already exists",
                    validated_name,
                    file_path.display()
                ),
            ));
        }

        fs::write(&file_path, contents)?;

        let relative_path =
            to_relative_slash_path(&self.root_path, &file_path).ok_or_else(|| {
                io::Error::new(
                    ErrorKind::InvalidInput,
                    format!("created file '{}' escaped root", file_path.display()),
                )
            })?;

        self.selected_relative_path = Some(relative_path.clone());
        self.dirty = true;
        let _ = self.rescan_if_dirty()?;

        Ok(relative_path)
    }

    pub(crate) fn suggest_rename_selected_name(
        &self,
        new_name: &str,
    ) -> io::Result<Option<String>> {
        let Some(selected_relative_path) = self.selected_relative_path.clone() else {
            return Ok(None);
        };

        let selected_full_path = self
            .resolve_relative_path(&selected_relative_path)
            .ok_or_else(|| {
                io::Error::new(
                    ErrorKind::InvalidInput,
                    format!(
                        "selected asset path '{}' is invalid",
                        selected_relative_path
                    ),
                )
            })?;

        let parent_path = selected_full_path.parent().ok_or_else(|| {
            io::Error::new(
                ErrorKind::InvalidInput,
                format!(
                    "selected asset path '{}' has no parent directory",
                    selected_relative_path
                ),
            )
        })?;

        let validated_name = validate_asset_name(new_name)?;
        let suggested = suggest_available_name(parent_path, validated_name);
        if suggested == validated_name {
            return Ok(None);
        }

        Ok(Some(suggested))
    }

    pub(crate) fn suggest_move_selected_name(
        &self,
        target_directory_relative: Option<&str>,
    ) -> io::Result<Option<String>> {
        let Some(selected_relative_path) = self.selected_relative_path.clone() else {
            return Ok(None);
        };

        self.suggest_move_relative_name(&selected_relative_path, target_directory_relative)
    }

    pub(crate) fn suggest_move_relative_name(
        &self,
        source_relative_path: &str,
        target_directory_relative: Option<&str>,
    ) -> io::Result<Option<String>> {
        let source_path = self
            .resolve_relative_path(source_relative_path)
            .ok_or_else(|| {
                io::Error::new(
                    ErrorKind::InvalidInput,
                    format!("source asset path '{}' is invalid", source_relative_path),
                )
            })?;

        let target_directory = self.resolve_target_directory(target_directory_relative)?;

        let source_name = source_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                io::Error::new(
                    ErrorKind::InvalidInput,
                    format!(
                        "source asset path '{}' has no file name",
                        source_relative_path
                    ),
                )
            })?;

        let target_path = target_directory.join(source_name);
        if normalize_disk_path(&target_path) == normalize_disk_path(&source_path) {
            return Ok(None);
        }

        let suggested = suggest_available_name(&target_directory, source_name);
        if suggested == source_name {
            return Ok(None);
        }

        Ok(Some(suggested))
    }

    pub(crate) fn rename_selected(&mut self, new_name: &str) -> io::Result<Option<String>> {
        self.rename_selected_with_options(new_name, false)
    }

    pub(crate) fn rename_selected_with_options(
        &mut self,
        new_name: &str,
        overwrite_existing: bool,
    ) -> io::Result<Option<String>> {
        let Some(selected_relative_path) = self.selected_relative_path.clone() else {
            return Ok(None);
        };

        let selected_full_path = self
            .resolve_relative_path(&selected_relative_path)
            .ok_or_else(|| {
                io::Error::new(
                    ErrorKind::InvalidInput,
                    format!(
                        "selected asset path '{}' is invalid",
                        selected_relative_path
                    ),
                )
            })?;

        let parent_path = selected_full_path.parent().ok_or_else(|| {
            io::Error::new(
                ErrorKind::InvalidInput,
                format!(
                    "selected asset path '{}' has no parent directory",
                    selected_relative_path
                ),
            )
        })?;

        let validated_name = validate_asset_name(new_name)?;
        let renamed_path = parent_path.join(validated_name);

        if selected_full_path == renamed_path {
            return Ok(Some(selected_relative_path));
        }

        if renamed_path.exists() {
            if !overwrite_existing {
                return Err(io::Error::new(
                    ErrorKind::AlreadyExists,
                    format!(
                        "cannot rename '{}': target '{}' already exists",
                        selected_relative_path,
                        renamed_path.display()
                    ),
                ));
            }

            remove_path_for_overwrite(&renamed_path)?;
        }

        fs::rename(&selected_full_path, &renamed_path)?;

        let renamed_relative_path = to_relative_slash_path(&self.root_path, &renamed_path)
            .ok_or_else(|| {
                io::Error::new(
                    ErrorKind::InvalidInput,
                    format!(
                        "renamed asset path '{}' escaped root",
                        renamed_path.display()
                    ),
                )
            })?;

        self.invalidate_thumbnail_for_relative_path(&selected_relative_path);
        self.selected_relative_path = Some(renamed_relative_path.clone());
        self.dirty = true;
        let _ = self.rescan_if_dirty()?;

        Ok(Some(renamed_relative_path))
    }

    pub(crate) fn move_selected(
        &mut self,
        target_directory_relative: Option<&str>,
    ) -> io::Result<Option<String>> {
        self.move_selected_with_options(target_directory_relative, None, false)
    }

    pub(crate) fn move_selected_with_options(
        &mut self,
        target_directory_relative: Option<&str>,
        target_name: Option<&str>,
        overwrite_existing: bool,
    ) -> io::Result<Option<String>> {
        let Some(selected_relative_path) = self.selected_relative_path.clone() else {
            return Ok(None);
        };

        self.move_relative_path_with_options(
            &selected_relative_path,
            target_directory_relative,
            target_name,
            overwrite_existing,
        )
    }

    pub(crate) fn move_relative_path(
        &mut self,
        source_relative_path: &str,
        target_directory_relative: Option<&str>,
    ) -> io::Result<Option<String>> {
        self.move_relative_path_with_options(
            source_relative_path,
            target_directory_relative,
            None,
            false,
        )
    }

    pub(crate) fn move_relative_path_with_options(
        &mut self,
        source_relative_path: &str,
        target_directory_relative: Option<&str>,
        target_name: Option<&str>,
        overwrite_existing: bool,
    ) -> io::Result<Option<String>> {
        let source_relative_path_owned = source_relative_path.to_owned();

        let selected_full_path = self
            .resolve_relative_path(source_relative_path)
            .ok_or_else(|| {
                io::Error::new(
                    ErrorKind::InvalidInput,
                    format!("selected asset path '{}' is invalid", source_relative_path),
                )
            })?;

        let target_directory = self.resolve_target_directory(target_directory_relative)?;

        if selected_full_path.is_dir() {
            let selected_normalized = normalize_disk_path(&selected_full_path);
            let target_normalized = normalize_disk_path(&target_directory);
            if target_normalized.starts_with(&selected_normalized) {
                return Err(io::Error::new(
                    ErrorKind::InvalidInput,
                    format!(
                        "cannot move directory '{}' inside itself",
                        source_relative_path
                    ),
                ));
            }
        }

        let target_file_name = if let Some(name) = target_name {
            validate_asset_name(name)?.to_owned()
        } else {
            selected_full_path
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| {
                    io::Error::new(
                        ErrorKind::InvalidInput,
                        format!(
                            "selected asset path '{}' has no file name",
                            source_relative_path
                        ),
                    )
                })?
                .to_owned()
        };

        let moved_path = target_directory.join(target_file_name);
        if moved_path == selected_full_path {
            return Ok(Some(source_relative_path_owned));
        }

        if moved_path.exists() {
            if !overwrite_existing {
                return Err(io::Error::new(
                    ErrorKind::AlreadyExists,
                    format!(
                        "cannot move '{}': target '{}' already exists",
                        source_relative_path,
                        moved_path.display()
                    ),
                ));
            }

            remove_path_for_overwrite(&moved_path)?;
        }

        fs::rename(&selected_full_path, &moved_path)?;

        let moved_relative_path =
            to_relative_slash_path(&self.root_path, &moved_path).ok_or_else(|| {
                io::Error::new(
                    ErrorKind::InvalidInput,
                    format!("moved asset path '{}' escaped root", moved_path.display()),
                )
            })?;

        self.invalidate_thumbnail_for_relative_path(source_relative_path);
        self.invalidate_thumbnail_for_relative_path(&moved_relative_path);
        self.selected_relative_path = Some(moved_relative_path.clone());
        self.dirty = true;
        let _ = self.rescan_if_dirty()?;

        Ok(Some(moved_relative_path))
    }

    pub(crate) fn delete_selected(&mut self) -> io::Result<bool> {
        let Some(selected_relative_path) = self.selected_relative_path.clone() else {
            return Ok(false);
        };

        let selected_full_path = self
            .resolve_relative_path(&selected_relative_path)
            .ok_or_else(|| {
                io::Error::new(
                    ErrorKind::InvalidInput,
                    format!(
                        "selected asset path '{}' is invalid",
                        selected_relative_path
                    ),
                )
            })?;

        if selected_full_path.is_dir() {
            fs::remove_dir_all(&selected_full_path)?;
        } else {
            fs::remove_file(&selected_full_path)?;
        }

        self.invalidate_thumbnail_for_relative_path(&selected_relative_path);
        self.selected_relative_path = None;
        self.dirty = true;
        let _ = self.rescan_if_dirty()?;

        Ok(true)
    }

    fn resolve_target_directory(
        &self,
        target_directory_relative: Option<&str>,
    ) -> io::Result<PathBuf> {
        if let Some(relative) = target_directory_relative {
            if relative.trim().is_empty() {
                return Ok(self.root_path.clone());
            }

            return self.resolve_relative_directory(relative).ok_or_else(|| {
                io::Error::new(
                    ErrorKind::NotFound,
                    format!("target directory '{}' does not exist", relative),
                )
            });
        }

        Ok(self.root_path.clone())
    }

    pub(crate) fn filter_mut(&mut self) -> &mut String {
        &mut self.filter
    }

    pub(crate) fn view_mode(&self) -> AssetBrowserViewModeConfig {
        self.view_mode
    }

    pub(crate) fn set_view_mode(&mut self, view_mode: AssetBrowserViewModeConfig) {
        self.view_mode = view_mode;
    }

    pub(crate) fn filtered_entries(&self) -> Vec<AssetEntry> {
        if self.filter.trim().is_empty() {
            return self.entries.clone();
        }

        let needle = self.filter.to_ascii_lowercase();
        self.entries
            .iter()
            .filter(|entry| entry.name.to_ascii_lowercase().contains(&needle))
            .cloned()
            .collect()
    }

    pub(crate) fn validate_selected_path(&mut self) {
        let Some(relative_path) = self.selected_relative_path.clone() else {
            return;
        };

        let Some(full_path) = self.resolve_relative_path(&relative_path) else {
            self.selected_relative_path = None;
            return;
        };

        if !full_path.exists() {
            self.selected_relative_path = None;
        }
    }

    fn invalidate_thumbnail_for_disk_path(&mut self, disk_path: &Path) {
        let absolute_path = if disk_path.is_absolute() {
            disk_path.to_path_buf()
        } else {
            self.root_path.join(disk_path)
        };

        if let Some(relative_path) = to_relative_slash_path(&self.root_path, &absolute_path) {
            self.invalidate_thumbnail_for_relative_path(&relative_path);
        }
    }

    fn invalidate_thumbnail_for_relative_path(&mut self, relative_path: &str) {
        let prefix = format!("{}/", relative_path);

        self.thumbnail_cache
            .retain(|path, _| path != relative_path && !path.starts_with(&prefix));
        self.thumbnail_pending
            .retain(|path| path != relative_path && !path.starts_with(&prefix));
        self.thumbnail_failed
            .retain(|path| path != relative_path && !path.starts_with(&prefix));
        self.thumbnail_order
            .retain(|path| path != relative_path && !path.starts_with(&prefix));
    }

    fn enforce_thumbnail_cache_limit(&mut self) {
        while self.thumbnail_cache.len() > MAX_THUMBNAIL_CACHE_ITEMS {
            if let Some(oldest_key) = self.thumbnail_order.pop_front() {
                self.thumbnail_cache.remove(&oldest_key);
                continue;
            }

            if let Some(arbitrary_key) = self.thumbnail_cache.keys().next().cloned() {
                self.thumbnail_cache.remove(&arbitrary_key);
            } else {
                break;
            }
        }
    }

    fn collect_entries(&self) -> std::io::Result<Vec<AssetEntry>> {
        if !self.current_path.exists() {
            return Ok(Vec::new());
        }

        let mut entries = Vec::new();

        for dir_entry in fs::read_dir(&self.current_path)? {
            let Ok(dir_entry) = dir_entry else {
                continue;
            };

            let path = dir_entry.path();
            let metadata = match dir_entry.metadata() {
                Ok(metadata) => metadata,
                Err(_) => continue,
            };

            let is_directory = metadata.is_dir();
            let Some(relative_path) = to_relative_slash_path(&self.root_path, &path) else {
                continue;
            };

            let name = dir_entry.file_name().to_string_lossy().to_string();
            let kind = classify_asset_kind(&name, is_directory);
            let file_size = if is_directory { 0 } else { metadata.len() };

            entries.push(AssetEntry {
                relative_path,
                name,
                is_directory,
                kind,
                file_size,
            });
        }

        entries.sort_by(order_entries);
        Ok(entries)
    }

    fn resolve_relative_directory(&self, relative: &str) -> Option<PathBuf> {
        let resolved = self.resolve_relative_path(relative)?;
        if resolved.is_dir() {
            return Some(resolved);
        }

        None
    }

    fn resolve_relative_path(&self, relative: &str) -> Option<PathBuf> {
        let relative_path = Path::new(relative);
        if relative_path.is_absolute() {
            return None;
        }

        if relative_path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        }) {
            return None;
        }

        let candidate = normalize_disk_path(&self.root_path.join(relative_path));
        let normalized_root = normalize_disk_path(&self.root_path);

        if candidate.starts_with(&normalized_root) {
            return Some(candidate);
        }

        None
    }
}

fn setup_recursive_watcher(
    root_path: &Path,
) -> (
    Option<RecommendedWatcher>,
    Option<Receiver<notify::Result<Event>>>,
) {
    let (tx, rx) = mpsc::channel();
    let mut watcher = match notify::recommended_watcher(move |event| {
        let _ = tx.send(event);
    }) {
        Ok(watcher) => watcher,
        Err(error) => {
            log::warn!(
                target: "engine::editor",
                "asset browser watcher unavailable: {}",
                error
            );
            return (None, None);
        }
    };

    if let Err(error) = watcher.watch(root_path, RecursiveMode::Recursive) {
        log::warn!(
            target: "engine::editor",
            "asset browser could not watch '{}': {}",
            root_path.display(),
            error
        );
        return (None, None);
    }

    (Some(watcher), Some(rx))
}

fn spawn_thumbnail_workers(
    thumbnail_job_rx: Receiver<ThumbnailJob>,
    thumbnail_tx: Sender<ThumbnailResult>,
    worker_count: usize,
) -> bool {
    let shared_rx = Arc::new(Mutex::new(thumbnail_job_rx));
    let mut spawned_workers = 0;

    for worker_index in 0..worker_count {
        let worker_rx = Arc::clone(&shared_rx);
        let worker_tx = thumbnail_tx.clone();
        let worker_name = format!("asset-thumbnail-{}", worker_index + 1);

        let spawn_result = std::thread::Builder::new()
            .name(worker_name)
            .spawn(move || loop {
                let job = {
                    let Ok(rx_guard) = worker_rx.lock() else {
                        return;
                    };

                    match rx_guard.recv() {
                        Ok(job) => job,
                        Err(_) => return,
                    }
                };

                let image = generate_thumbnail_image(&job.disk_path, job.kind);
                let _ = worker_tx.send(ThumbnailResult {
                    relative_path: job.relative_path,
                    image,
                });
            });

        if spawn_result.is_ok() {
            spawned_workers += 1;
            continue;
        }

        log::warn!(
            target: "engine::editor",
            "asset browser could not spawn thumbnail worker {}",
            worker_index + 1
        );
    }

    if spawned_workers == 0 {
        log::warn!(
            target: "engine::editor",
            "asset browser thumbnail workers unavailable"
        );
    }

    spawned_workers > 0
}

fn should_render_thumbnail(entry: &AssetEntry) -> bool {
    !entry.is_directory && matches!(entry.kind, AssetKind::Texture | AssetKind::Mesh)
}

fn generate_thumbnail_image(path: &Path, kind: AssetKind) -> Option<ThumbnailImageData> {
    match kind {
        AssetKind::Texture => generate_texture_thumbnail(path),
        AssetKind::Mesh => generate_mesh_thumbnail(path),
        AssetKind::Audio | AssetKind::Scene | AssetKind::Other => None,
    }
}

fn generate_texture_thumbnail(path: &Path) -> Option<ThumbnailImageData> {
    let decoded = image::open(path).ok()?;
    let thumbnail = decoded
        .thumbnail(THUMBNAIL_EDGE_PX, THUMBNAIL_EDGE_PX)
        .to_rgba8();
    let (width, height) = thumbnail.dimensions();
    Some(ThumbnailImageData {
        width: width as usize,
        height: height as usize,
        rgba: thumbnail.into_raw(),
    })
}

fn generate_mesh_thumbnail(path: &Path) -> Option<ThumbnailImageData> {
    let (document, buffers, _) = gltf::import(path).ok()?;
    let (positions, indices) = load_first_mesh_primitive(&document, &buffers)?;
    rasterize_mesh_thumbnail(&positions, &indices, THUMBNAIL_EDGE_PX)
}

fn load_first_mesh_primitive(
    document: &gltf::Document,
    buffers: &[Data],
) -> Option<(Vec<[f32; 3]>, Vec<u32>)> {
    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            let reader = primitive
                .reader(|buffer| buffers.get(buffer.index()).map(|data| data.0.as_slice()));

            let Some(position_reader) = reader.read_positions() else {
                continue;
            };

            let positions: Vec<[f32; 3]> = position_reader.collect();
            if positions.len() < 3 {
                continue;
            }

            let indices: Vec<u32> = match reader.read_indices() {
                Some(index_reader) => index_reader.into_u32().collect(),
                None => (0..positions.len() as u32).collect(),
            };

            if indices.len() >= 3 {
                return Some((positions, indices));
            }
        }
    }

    None
}

fn rasterize_mesh_thumbnail(
    positions: &[[f32; 3]],
    indices: &[u32],
    edge_px: u32,
) -> Option<ThumbnailImageData> {
    if edge_px < 2 || positions.len() < 3 || indices.len() < 3 {
        return None;
    }

    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for position in positions {
        let value = Vec3::from_array(*position);
        min = min.min(value);
        max = max.max(value);
    }

    let center = (min + max) * 0.5;
    let mut radius = ((max - min).max_element() * 0.5).max(0.001);
    if !radius.is_finite() {
        radius = 1.0;
    }

    let camera_distance = radius * 2.6;
    let camera_position =
        center + Vec3::new(camera_distance, camera_distance * 0.85, camera_distance);
    let view = Mat4::look_at_rh(camera_position, center, Vec3::Y);
    let near = (radius * 0.05).max(0.01);
    let far = (camera_distance + radius * 3.0).max(10.0);
    let projection = Mat4::perspective_rh(42.0_f32.to_radians(), 1.0, near, far);
    let mvp = projection * view;

    let mut projected: Vec<Option<ProjectedVertex>> = Vec::with_capacity(positions.len());
    for position in positions {
        let clip = mvp * Vec4::new(position[0], position[1], position[2], 1.0);
        if !clip.is_finite() || clip.w.abs() <= f32::EPSILON {
            projected.push(None);
            continue;
        }

        let ndc = clip.truncate() / clip.w;
        if !ndc.is_finite() {
            projected.push(None);
            continue;
        }

        projected.push(Some(ProjectedVertex {
            screen: Vec2::new(
                (ndc.x * 0.5 + 0.5) * (edge_px as f32 - 1.0),
                (1.0 - (ndc.y * 0.5 + 0.5)) * (edge_px as f32 - 1.0),
            ),
            depth: ndc.z,
        }));
    }

    let edge = edge_px as usize;
    let pixel_count = edge * edge;
    let mut rgba = vec![0_u8; pixel_count * 4];
    fill_thumbnail_background(&mut rgba, edge);
    let mut depth_buffer = vec![f32::INFINITY; pixel_count];
    let light_direction = Vec3::new(0.4, 0.7, 0.5).normalize_or_zero();

    let mut drew_pixels = false;

    for triangle in indices.chunks_exact(3) {
        let [i0, i1, i2] = [
            triangle[0] as usize,
            triangle[1] as usize,
            triangle[2] as usize,
        ];

        if i0 >= positions.len() || i1 >= positions.len() || i2 >= positions.len() {
            continue;
        }

        let (Some(v0), Some(v1), Some(v2)) = (projected[i0], projected[i1], projected[i2]) else {
            continue;
        };

        let area = signed_area(v0.screen, v1.screen, v2.screen);
        if area.abs() < 1.0e-6 {
            continue;
        }

        let world0 = Vec3::from_array(positions[i0]);
        let world1 = Vec3::from_array(positions[i1]);
        let world2 = Vec3::from_array(positions[i2]);
        let normal = (world1 - world0).cross(world2 - world0);
        if normal.length_squared() < 1.0e-8 {
            continue;
        }

        let shade =
            (normal.normalize_or_zero().dot(light_direction).abs() * 0.7 + 0.2).clamp(0.0, 1.0);
        let triangle_color = [
            (70.0 + 130.0 * shade) as u8,
            (88.0 + 140.0 * shade) as u8,
            (104.0 + 140.0 * shade) as u8,
        ];

        let min_x = v0
            .screen
            .x
            .min(v1.screen.x)
            .min(v2.screen.x)
            .floor()
            .max(0.0) as i32;
        let max_x = v0
            .screen
            .x
            .max(v1.screen.x)
            .max(v2.screen.x)
            .ceil()
            .min((edge_px - 1) as f32) as i32;
        let min_y = v0
            .screen
            .y
            .min(v1.screen.y)
            .min(v2.screen.y)
            .floor()
            .max(0.0) as i32;
        let max_y = v0
            .screen
            .y
            .max(v1.screen.y)
            .max(v2.screen.y)
            .ceil()
            .min((edge_px - 1) as f32) as i32;

        if min_x > max_x || min_y > max_y {
            continue;
        }

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let sample = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);

                let w0 = signed_area(v1.screen, v2.screen, sample);
                let w1 = signed_area(v2.screen, v0.screen, sample);
                let w2 = signed_area(v0.screen, v1.screen, sample);

                if !is_inside_triangle(w0, w1, w2) {
                    continue;
                }

                let b0 = w0 / area;
                let b1 = w1 / area;
                let b2 = w2 / area;
                let depth = b0 * v0.depth + b1 * v1.depth + b2 * v2.depth;

                let index = y as usize * edge + x as usize;
                if depth >= depth_buffer[index] {
                    continue;
                }

                depth_buffer[index] = depth;
                let base = index * 4;
                rgba[base] = triangle_color[0];
                rgba[base + 1] = triangle_color[1];
                rgba[base + 2] = triangle_color[2];
                rgba[base + 3] = 255;
                drew_pixels = true;
            }
        }
    }

    if !drew_pixels {
        return None;
    }

    Some(ThumbnailImageData {
        width: edge,
        height: edge,
        rgba,
    })
}

fn fill_thumbnail_background(rgba: &mut [u8], edge_px: usize) {
    if edge_px == 0 {
        return;
    }

    for y in 0..edge_px {
        let t = y as f32 / (edge_px - 1).max(1) as f32;
        let r = (20.0 + 8.0 * t) as u8;
        let g = (24.0 + 10.0 * t) as u8;
        let b = (31.0 + 14.0 * t) as u8;

        for x in 0..edge_px {
            let index = (y * edge_px + x) * 4;
            rgba[index] = r;
            rgba[index + 1] = g;
            rgba[index + 2] = b;
            rgba[index + 3] = 255;
        }
    }
}

fn signed_area(a: Vec2, b: Vec2, c: Vec2) -> f32 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

fn is_inside_triangle(w0: f32, w1: f32, w2: f32) -> bool {
    (w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0) || (w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0)
}

fn classify_asset_kind(file_name: &str, is_directory: bool) -> AssetKind {
    if is_directory {
        return AssetKind::Other;
    }

    let lower_name = file_name.to_ascii_lowercase();
    if lower_name.ends_with(".scene.ron") {
        return AssetKind::Scene;
    }

    let extension = Path::new(file_name)
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();

    match extension.as_str() {
        "png" | "jpg" | "jpeg" | "webp" => AssetKind::Texture,
        "glb" | "gltf" => AssetKind::Mesh,
        "ogg" | "wav" | "mp3" => AssetKind::Audio,
        "ron" => AssetKind::Scene,
        _ => AssetKind::Other,
    }
}

fn is_relevant_fs_event(kind: &EventKind) -> bool {
    matches!(
        kind,
        EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_)
    )
}

fn normalize_disk_path(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

fn to_relative_slash_path(root_path: &Path, path: &Path) -> Option<String> {
    let relative = path.strip_prefix(root_path).ok()?;
    Some(relative.to_string_lossy().replace('\\', "/"))
}

fn order_entries(lhs: &AssetEntry, rhs: &AssetEntry) -> Ordering {
    match (lhs.is_directory, rhs.is_directory) {
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        _ => {
            let left = lhs.name.to_ascii_lowercase();
            let right = rhs.name.to_ascii_lowercase();
            left.cmp(&right).then_with(|| lhs.name.cmp(&rhs.name))
        }
    }
}

fn validate_asset_name(raw_name: &str) -> io::Result<&str> {
    let candidate = raw_name.trim();
    if candidate.is_empty() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "asset name cannot be empty",
        ));
    }

    if candidate == "." || candidate == ".." {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "asset name cannot be '.' or '..'",
        ));
    }

    let components: Vec<Component<'_>> = Path::new(candidate).components().collect();
    if components.len() != 1 || !matches!(components.first(), Some(Component::Normal(_))) {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            format!("asset name '{}' is not a valid file name", candidate),
        ));
    }

    Ok(candidate)
}

fn remove_path_for_overwrite(path: &Path) -> io::Result<()> {
    if !path.exists() {
        return Ok(());
    }

    if path.is_dir() {
        fs::remove_dir_all(path)?;
    } else {
        fs::remove_file(path)?;
    }

    Ok(())
}

fn suggest_available_name(parent_directory: &Path, desired_name: &str) -> String {
    let desired_path = parent_directory.join(desired_name);
    if !desired_path.exists() {
        return desired_name.to_owned();
    }

    let desired_file = Path::new(desired_name);
    let stem = desired_file
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or(desired_name);
    let extension = desired_file.extension().and_then(|value| value.to_str());

    let mut index = 1_u32;
    loop {
        let candidate = if let Some(ext) = extension {
            format!("{} ({}).{}", stem, index, ext)
        } else {
            format!("{} ({})", stem, index)
        };

        if !parent_directory.join(&candidate).exists() {
            return candidate;
        }

        index = index.saturating_add(1);
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::mpsc;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{
        classify_asset_kind, rasterize_mesh_thumbnail, AssetBrowserState, AssetEntry, AssetKind,
    };
    use crate::config::{AssetBrowserConfig, AssetBrowserViewModeConfig};

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
                "motley-engine-editor-{}-{}-{}",
                prefix,
                std::process::id(),
                timestamp
            ));

            fs::create_dir_all(&path).expect("temp directory should be created");
            Self { path }
        }
    }

    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn make_state(root_path: PathBuf, config: &AssetBrowserConfig) -> AssetBrowserState {
        AssetBrowserState::new_with_watcher(root_path, config, false)
    }

    #[test]
    fn classify_asset_kind_detects_scene_suffix() {
        assert_eq!(
            classify_asset_kind("demo.scene.ron", false),
            AssetKind::Scene
        );
        assert_eq!(classify_asset_kind("plain.ron", false), AssetKind::Scene);
        assert_eq!(classify_asset_kind("folder", true), AssetKind::Other);
    }

    #[test]
    fn filtered_entries_are_sorted_directories_first() {
        let guard = TempDirGuard::new("scan-order");

        fs::create_dir_all(guard.path.join("z_folder")).expect("folder should be created");
        fs::create_dir_all(guard.path.join("a_folder")).expect("folder should be created");
        fs::write(guard.path.join("m_file.txt"), "x").expect("file should be created");
        fs::write(guard.path.join("a_file.txt"), "x").expect("file should be created");

        let mut state = make_state(guard.path.clone(), &AssetBrowserConfig::default());
        let _ = state.rescan_if_dirty().expect("scan should succeed");

        let names: Vec<String> = state
            .filtered_entries()
            .into_iter()
            .map(|entry| entry.name)
            .collect();

        assert_eq!(
            names,
            vec![
                "a_folder".to_owned(),
                "z_folder".to_owned(),
                "a_file.txt".to_owned(),
                "m_file.txt".to_owned()
            ]
        );
    }

    #[test]
    fn filter_is_case_insensitive() {
        let guard = TempDirGuard::new("filter");

        fs::write(guard.path.join("Tree.png"), "x").expect("file should be created");
        fs::write(guard.path.join("rock.png"), "x").expect("file should be created");

        let mut config = AssetBrowserConfig {
            filter: "tRe".to_owned(),
            ..AssetBrowserConfig::default()
        };
        config.view_mode = AssetBrowserViewModeConfig::Grid;

        let mut state = make_state(guard.path.clone(), &config);
        let _ = state.rescan_if_dirty().expect("scan should succeed");

        let names: Vec<String> = state
            .filtered_entries()
            .into_iter()
            .map(|entry| entry.name)
            .collect();

        assert_eq!(names, vec!["Tree.png".to_owned()]);
    }

    #[test]
    fn invalid_relative_path_in_config_falls_back_to_root() {
        let guard = TempDirGuard::new("config-path");

        let config = AssetBrowserConfig {
            current_relative_path: Some("../outside".to_owned()),
            ..AssetBrowserConfig::default()
        };

        let state = make_state(guard.path.clone(), &config);
        assert!(state.current_relative_path().is_none());
    }

    #[test]
    fn request_thumbnail_skips_non_previewable_assets() {
        let guard = TempDirGuard::new("thumb-skip");
        let mut state = make_state(guard.path.clone(), &AssetBrowserConfig::default());

        let entry = AssetEntry {
            relative_path: "audio/click.ogg".to_owned(),
            name: "click.ogg".to_owned(),
            is_directory: false,
            kind: AssetKind::Audio,
            file_size: 0,
        };

        assert!(!state.request_thumbnail_for_entry(&entry));
    }

    #[test]
    fn request_thumbnail_rejects_path_escape() {
        let guard = TempDirGuard::new("thumb-path-escape");
        let mut state = make_state(guard.path.clone(), &AssetBrowserConfig::default());

        let entry = AssetEntry {
            relative_path: "../outside.png".to_owned(),
            name: "outside.png".to_owned(),
            is_directory: false,
            kind: AssetKind::Texture,
            file_size: 0,
        };

        assert!(!state.request_thumbnail_for_entry(&entry));
        assert!(state.thumbnail_failed.contains("../outside.png"));
        assert!(!state.thumbnail_pending.contains("../outside.png"));
    }

    #[test]
    fn process_file_events_marks_dirty_for_relevant_events() {
        let guard = TempDirGuard::new("watcher-relevant");
        fs::write(guard.path.join("sample.png"), "x").expect("file should be created");

        let mut state = make_state(guard.path.clone(), &AssetBrowserConfig::default());
        let _ = state.rescan_if_dirty().expect("scan should succeed");
        state.dirty = false;

        let (tx, rx) = mpsc::channel();
        state.watcher_rx = Some(rx);

        let event = notify::Event::new(notify::EventKind::Modify(notify::event::ModifyKind::Any))
            .add_path(guard.path.join("sample.png"));
        tx.send(Ok(event)).expect("event should be sent");

        let event_count = state.process_file_events();
        assert_eq!(event_count, 1);
        assert!(state.dirty);
    }

    #[test]
    fn process_file_events_ignores_irrelevant_events() {
        let guard = TempDirGuard::new("watcher-irrelevant");
        fs::write(guard.path.join("sample.png"), "x").expect("file should be created");

        let mut state = make_state(guard.path.clone(), &AssetBrowserConfig::default());
        let _ = state.rescan_if_dirty().expect("scan should succeed");
        state.dirty = false;

        let (tx, rx) = mpsc::channel();
        state.watcher_rx = Some(rx);

        let event =
            notify::Event::new(notify::EventKind::Any).add_path(guard.path.join("sample.png"));
        tx.send(Ok(event)).expect("event should be sent");

        let event_count = state.process_file_events();
        assert_eq!(event_count, 0);
        assert!(!state.dirty);
    }

    #[test]
    fn rasterize_mesh_thumbnail_returns_image() {
        let positions = vec![[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]];
        let indices = vec![0, 1, 2];

        let image =
            rasterize_mesh_thumbnail(&positions, &indices, 64).expect("thumbnail should render");

        assert_eq!(image.width, 64);
        assert_eq!(image.height, 64);

        let brightest_pixel = image
            .rgba
            .chunks_exact(4)
            .map(|pixel| pixel[0] as u16 + pixel[1] as u16 + pixel[2] as u16)
            .max()
            .unwrap_or(0);

        assert!(brightest_pixel > 260);
    }

    #[test]
    fn rename_selected_updates_selection_and_disk_entry() {
        let guard = TempDirGuard::new("rename-selected");
        fs::write(guard.path.join("before.txt"), "payload").expect("file should be created");

        let mut state = make_state(guard.path.clone(), &AssetBrowserConfig::default());
        let _ = state.rescan_if_dirty().expect("scan should succeed");
        state.set_selected_relative_path(Some("before.txt".to_owned()));

        let renamed = state
            .rename_selected("after.txt")
            .expect("rename should succeed");

        assert_eq!(renamed.as_deref(), Some("after.txt"));
        assert!(guard.path.join("after.txt").exists());
        assert!(!guard.path.join("before.txt").exists());
        assert_eq!(state.selected_relative_path(), Some("after.txt"));
    }

    #[test]
    fn move_selected_moves_file_to_target_directory() {
        let guard = TempDirGuard::new("move-selected");
        fs::create_dir_all(guard.path.join("target")).expect("target dir should be created");
        fs::write(guard.path.join("move_me.txt"), "payload").expect("file should be created");

        let mut state = make_state(guard.path.clone(), &AssetBrowserConfig::default());
        let _ = state.rescan_if_dirty().expect("scan should succeed");
        state.set_selected_relative_path(Some("move_me.txt".to_owned()));

        let moved = state
            .move_selected(Some("target"))
            .expect("move should succeed");

        assert_eq!(moved.as_deref(), Some("target/move_me.txt"));
        assert!(guard.path.join("target/move_me.txt").exists());
        assert!(!guard.path.join("move_me.txt").exists());
        assert_eq!(state.selected_relative_path(), Some("target/move_me.txt"));
    }

    #[test]
    fn delete_selected_removes_file_and_clears_selection() {
        let guard = TempDirGuard::new("delete-selected");
        fs::write(guard.path.join("delete_me.txt"), "payload").expect("file should be created");

        let mut state = make_state(guard.path.clone(), &AssetBrowserConfig::default());
        let _ = state.rescan_if_dirty().expect("scan should succeed");
        state.set_selected_relative_path(Some("delete_me.txt".to_owned()));

        let deleted = state.delete_selected().expect("delete should succeed");
        assert!(deleted);
        assert!(!guard.path.join("delete_me.txt").exists());
        assert!(state.selected_relative_path().is_none());
    }

    #[test]
    fn create_directory_in_current_creates_folder_and_selects_it() {
        let guard = TempDirGuard::new("create-directory");
        let mut state = make_state(guard.path.clone(), &AssetBrowserConfig::default());

        let created = state
            .create_directory_in_current("new_folder")
            .expect("folder should be created");

        assert_eq!(created, "new_folder");
        assert!(guard.path.join("new_folder").is_dir());
        assert_eq!(state.selected_relative_path(), Some("new_folder"));
    }

    #[test]
    fn create_file_in_current_creates_file_and_selects_it() {
        let guard = TempDirGuard::new("create-file");
        let mut state = make_state(guard.path.clone(), &AssetBrowserConfig::default());

        let created = state
            .create_file_in_current("new_asset.ron", "(\n)\n")
            .expect("file should be created");

        assert_eq!(created, "new_asset.ron");
        assert!(guard.path.join("new_asset.ron").is_file());
        assert_eq!(state.selected_relative_path(), Some("new_asset.ron"));
    }

    #[test]
    fn move_relative_path_with_overwrite_replaces_existing_target() {
        let guard = TempDirGuard::new("move-overwrite");
        fs::create_dir_all(guard.path.join("target")).expect("target dir should be created");
        fs::write(guard.path.join("source.txt"), "source").expect("source should be created");
        fs::write(guard.path.join("target/source.txt"), "target")
            .expect("target should be created");

        let mut state = make_state(guard.path.clone(), &AssetBrowserConfig::default());
        let moved = state
            .move_relative_path_with_options("source.txt", Some("target"), None, true)
            .expect("move with overwrite should succeed");

        assert_eq!(moved.as_deref(), Some("target/source.txt"));
        let payload = fs::read_to_string(guard.path.join("target/source.txt"))
            .expect("moved payload should exist");
        assert_eq!(payload, "source");
    }

    #[test]
    fn move_relative_path_moves_directory_to_target_directory() {
        let guard = TempDirGuard::new("move-directory");
        fs::create_dir_all(guard.path.join("source/nested"))
            .expect("source nested directory should be created");
        fs::create_dir_all(guard.path.join("target")).expect("target dir should be created");
        fs::write(guard.path.join("source/nested/data.txt"), "payload")
            .expect("source payload should be created");

        let mut state = make_state(guard.path.clone(), &AssetBrowserConfig::default());
        let moved = state
            .move_relative_path("source", Some("target"))
            .expect("directory move should succeed");

        assert_eq!(moved.as_deref(), Some("target/source"));
        assert!(guard.path.join("target/source/nested/data.txt").is_file());
        assert!(!guard.path.join("source").exists());
        assert_eq!(state.selected_relative_path(), Some("target/source"));
    }

    #[test]
    fn move_relative_path_rejects_directory_move_into_itself() {
        let guard = TempDirGuard::new("move-directory-inside-itself");
        fs::create_dir_all(guard.path.join("parent/child"))
            .expect("parent and child directories should be created");

        let mut state = make_state(guard.path.clone(), &AssetBrowserConfig::default());
        let error = state
            .move_relative_path("parent", Some("parent/child"))
            .expect_err("move should fail when target is inside source");

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert!(guard.path.join("parent/child").exists());
    }

    #[test]
    fn rename_selected_with_overwrite_replaces_existing_target() {
        let guard = TempDirGuard::new("rename-overwrite");
        fs::write(guard.path.join("source.txt"), "source payload")
            .expect("source should be created");
        fs::write(guard.path.join("target.txt"), "target payload")
            .expect("target should be created");

        let mut state = make_state(guard.path.clone(), &AssetBrowserConfig::default());
        state.set_selected_relative_path(Some("source.txt".to_owned()));

        let renamed = state
            .rename_selected_with_options("target.txt", true)
            .expect("rename with overwrite should succeed");

        assert_eq!(renamed.as_deref(), Some("target.txt"));
        assert!(!guard.path.join("source.txt").exists());
        let payload =
            fs::read_to_string(guard.path.join("target.txt")).expect("target payload should exist");
        assert_eq!(payload, "source payload");
    }

    #[test]
    fn suggest_move_relative_name_returns_incremented_candidate() {
        let guard = TempDirGuard::new("suggest-move-relative");
        fs::create_dir_all(guard.path.join("target")).expect("target dir should be created");
        fs::write(guard.path.join("item.txt"), "payload").expect("source should exist");
        fs::write(guard.path.join("target/item.txt"), "payload")
            .expect("first conflict should exist");
        fs::write(guard.path.join("target/item (1).txt"), "payload")
            .expect("second conflict should exist");

        let state = make_state(guard.path.clone(), &AssetBrowserConfig::default());
        let suggested = state
            .suggest_move_relative_name("item.txt", Some("target"))
            .expect("suggestion should succeed");

        assert_eq!(suggested.as_deref(), Some("item (2).txt"));
    }

    #[test]
    fn suggest_rename_selected_name_returns_incremented_candidate() {
        let guard = TempDirGuard::new("suggest-rename");
        fs::write(guard.path.join("item.txt"), "payload").expect("source file should exist");
        fs::write(guard.path.join("renamed.txt"), "payload").expect("conflict file should exist");

        let mut state = make_state(guard.path.clone(), &AssetBrowserConfig::default());
        state.set_selected_relative_path(Some("item.txt".to_owned()));

        let suggested = state
            .suggest_rename_selected_name("renamed.txt")
            .expect("suggestion should succeed");
        assert_eq!(suggested.as_deref(), Some("renamed (1).txt"));
    }
}
