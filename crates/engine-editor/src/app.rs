use std::collections::{HashMap, HashSet};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::time::Instant;

use bevy_ecs::entity::Entity;
use bevy_ecs::query::With;
use bevy_ecs::system::RunSystemOnce;
use bevy_ecs::world::World;
use eframe::egui;
use egui_dock::{DockArea, DockState, TabViewer};
use engine_assets::{
    AssetServer, MaterialHandle, MeshData, MeshHandle, SceneDeserializer, SceneSerializer,
    TextureHandle,
};
use engine_core::{
    create_world, register_core_reflection_types, Camera2d, Camera3d, Children, EditorEntityBundle,
    EntityName, GlobalTransform, Parent, PrimaryCamera, RenderLayer2D, RenderLayer3D, Result,
    SpatialBundle, Transform, Visible,
};
use engine_math::glam::{Affine3A, Quat, Vec3};
use engine_math::Mat4;
use engine_physics::{
    raycast, register_physics_reflection_types, ColliderEntityMap3D, ColliderShape3D,
    PhysicsWorld3D,
};
use engine_reflect::{ComponentRegistry, ReflectMetadataRegistry, ReflectTypeRegistry};
use engine_render::{MeshRenderable3d, RenderSceneAdapter, SpriteRenderable2d};

use crate::asset_browser::{AssetBrowserState, AssetKind};
use crate::commands::{
    CommandHistory, DeleteEntityCommand, DuplicateEntityCommand, EditorCommand,
    RenameEntityCommand, ReparentEntityCommand, SetComponentCommand, SpawnEntityCommand,
};
use crate::config::{
    AssetBrowserViewModeConfig, EditorConfig, GizmoAxisLockConfig, GizmoModeConfig,
    GizmoOrientationConfig, GizmoSnapConfig, GizmoToolConfig, ViewportCameraConfig,
    ViewportOverlayConfig,
};
use crate::inspector::InspectorPanel;
use crate::layout::{create_default_layout, Tab};
use crate::selection::Selection;
use crate::viewport::{EditorCamera, ViewportRenderer};

const DEFAULT_DROP_TEXTURE_PATH: &str = "textures/placeholder.png";
const DEFAULT_DROP_MATERIAL_PATH: &str = "materials/default.ron";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum LogLevel {
    Trace,
    Info,
    Warn,
    Error,
}

#[derive(Clone, Debug)]
struct LogEntry {
    level: LogLevel,
    message: String,
    timestamp: f64,
    module: String,
}

enum SceneTreeAction {
    AddRootEntity,
    AddChildEntity(Entity),
    Reparent {
        entity: Entity,
        new_parent: Option<Entity>,
    },
    BeginRename(Entity),
    CommitRename,
    CancelRename,
    Duplicate(Entity),
    Delete(Entity),
}

#[derive(Clone, Debug)]
struct AssetDragPayload {
    relative_path: String,
    kind: AssetKind,
}

impl AssetDragPayload {
    fn from_entry(relative_path: &str, kind: AssetKind) -> Self {
        Self {
            relative_path: relative_path.to_owned(),
            kind,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NewAssetTemplate {
    Scene,
    Material,
    Text,
}

impl NewAssetTemplate {
    fn label(self) -> &'static str {
        match self {
            Self::Scene => "Scene (.scene.ron)",
            Self::Material => "Material (.ron)",
            Self::Text => "Text (.txt)",
        }
    }

    fn default_file_name(self) -> &'static str {
        match self {
            Self::Scene => "new.scene.ron",
            Self::Material => "new_material.ron",
            Self::Text => "notes.txt",
        }
    }

    fn contents(self, file_name: &str) -> String {
        match self {
            Self::Scene => {
                let scene_name = Path::new(file_name)
                    .file_stem()
                    .and_then(|value| value.to_str())
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .unwrap_or("new.scene");

                format!(
                    "(\n    version: 1,\n    name: \"{}\",\n    entities: [],\n)\n",
                    scene_name
                )
            }
            Self::Material => {
                "(\n  base_color_factor: [1.0, 1.0, 1.0, 1.0],\n  metallic: 0.0,\n  roughness: 1.0,\n)\n"
                    .to_owned()
            }
            Self::Text => String::new(),
        }
    }
}

#[derive(Clone, Debug)]
enum AssetConflictAction {
    RenameSelected {
        desired_name: String,
    },
    MoveSelected {
        target_directory: Option<String>,
    },
    MoveRelativePath {
        source_relative_path: String,
        target_directory: Option<String>,
    },
}

#[derive(Clone, Debug)]
struct AssetConflictState {
    action: AssetConflictAction,
    source_label: String,
    target_label: String,
    suggested_name: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TextureDropTarget {
    Sprite2d,
    Mesh3d,
}

struct ReplaceRenderableTextureCommand {
    entity: Entity,
    target: TextureDropTarget,
    old_texture: TextureHandle,
    new_texture: TextureHandle,
    desc: String,
}

impl ReplaceRenderableTextureCommand {
    fn new(
        entity: Entity,
        target: TextureDropTarget,
        old_texture: TextureHandle,
        new_texture: TextureHandle,
    ) -> Self {
        let target_name = match target {
            TextureDropTarget::Sprite2d => "SpriteRenderable2d",
            TextureDropTarget::Mesh3d => "MeshRenderable3d",
        };

        Self {
            entity,
            target,
            old_texture,
            new_texture,
            desc: format!("Replace {} texture", target_name),
        }
    }

    fn apply_texture(&self, world: &mut World, texture: TextureHandle) {
        let Ok(mut entity_ref) = world.get_entity_mut(self.entity) else {
            return;
        };

        match self.target {
            TextureDropTarget::Sprite2d => {
                if let Some(mut sprite) = entity_ref.get_mut::<SpriteRenderable2d>() {
                    sprite.texture = texture;
                }
            }
            TextureDropTarget::Mesh3d => {
                if let Some(mut mesh) = entity_ref.get_mut::<MeshRenderable3d>() {
                    mesh.texture = texture;
                }
            }
        }
    }
}

impl EditorCommand for ReplaceRenderableTextureCommand {
    fn execute(&mut self, world: &mut World) {
        self.apply_texture(world, self.new_texture);
    }

    fn undo(&mut self, world: &mut World) {
        self.apply_texture(world, self.old_texture);
    }

    fn description(&self) -> &str {
        &self.desc
    }

    fn selection_hint(&self) -> Option<Entity> {
        Some(self.entity)
    }
}

struct SpawnMeshAssetDropCommand {
    entity_name: String,
    transform: Transform,
    mesh: MeshHandle,
    texture: TextureHandle,
    material: MaterialHandle,
    spawned_entity: Option<Entity>,
    desc: String,
}

impl SpawnMeshAssetDropCommand {
    fn new(
        entity_name: String,
        transform: Transform,
        mesh: MeshHandle,
        texture: TextureHandle,
        material: MaterialHandle,
    ) -> Self {
        Self {
            entity_name,
            transform,
            mesh,
            texture,
            material,
            spawned_entity: None,
            desc: "Spawn mesh from asset drop".to_owned(),
        }
    }
}

impl EditorCommand for SpawnMeshAssetDropCommand {
    fn execute(&mut self, world: &mut World) {
        let entity = world.spawn(EditorEntityBundle::default()).id();

        if let Ok(mut entity_ref) = world.get_entity_mut(entity) {
            entity_ref.insert((
                EntityName::new(self.entity_name.clone()),
                self.transform.clone(),
                MeshRenderable3d::new(self.mesh, self.texture, self.material),
                Visible,
                RenderLayer3D,
            ));

            let _ = entity_ref.remove::<RenderLayer2D>();
        }

        self.spawned_entity = Some(entity);
    }

    fn undo(&mut self, world: &mut World) {
        let Some(entity) = self.spawned_entity.take() else {
            return;
        };

        let _ = world.despawn(entity);
    }

    fn description(&self) -> &str {
        &self.desc
    }

    fn selection_hint(&self) -> Option<Entity> {
        self.spawned_entity
    }
}

struct AssetBrowserFileOpsState {
    create_folder_open: bool,
    create_folder_input: String,
    create_asset_open: bool,
    create_asset_input: String,
    create_asset_template: NewAssetTemplate,
    rename_open: bool,
    rename_input: String,
    move_open: bool,
    move_input: String,
    delete_open: bool,
    conflict_open: bool,
    conflict_state: Option<AssetConflictState>,
    feedback: Option<(LogLevel, String)>,
}

impl Default for AssetBrowserFileOpsState {
    fn default() -> Self {
        Self {
            create_folder_open: false,
            create_folder_input: "new_folder".to_owned(),
            create_asset_open: false,
            create_asset_input: NewAssetTemplate::Scene.default_file_name().to_owned(),
            create_asset_template: NewAssetTemplate::Scene,
            rename_open: false,
            rename_input: String::new(),
            move_open: false,
            move_input: String::new(),
            delete_open: false,
            conflict_open: false,
            conflict_state: None,
            feedback: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GizmoMode {
    Translate,
    Rotate,
    Scale,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GizmoOrientation {
    Local,
    Global,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GizmoDragIntent {
    AxisX,
    AxisY,
    Uniform,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GizmoAxisConstraint {
    AxisX,
    AxisY,
    AxisZ,
}

#[derive(Clone, Debug)]
struct GizmoDragInteraction {
    entity: Entity,
    drag_start_pointer: egui::Pos2,
    initial_local_transform: Transform,
    latest_local_transform: Transform,
    drag_intent: Option<GizmoDragIntent>,
    axis_constraint: Option<GizmoAxisConstraint>,
}

#[derive(Clone, Debug)]
struct GizmoState {
    mode: GizmoMode,
    orientation: GizmoOrientation,
    manual_axis_constraint: Option<GizmoAxisConstraint>,
    snapping_enabled: bool,
    translate_snap: f32,
    rotate_snap_degrees: f32,
    scale_snap: f32,
    active_drag: Option<GizmoDragInteraction>,
}

impl Default for GizmoState {
    fn default() -> Self {
        Self {
            mode: GizmoMode::Translate,
            orientation: GizmoOrientation::Local,
            manual_axis_constraint: None,
            snapping_enabled: false,
            translate_snap: 0.5,
            rotate_snap_degrees: 15.0,
            scale_snap: 0.1,
            active_drag: None,
        }
    }
}

impl GizmoState {
    fn from_configs(snap_config: &GizmoSnapConfig, tool_config: &GizmoToolConfig) -> Self {
        let mode = match tool_config.mode {
            GizmoModeConfig::Translate => GizmoMode::Translate,
            GizmoModeConfig::Rotate => GizmoMode::Rotate,
            GizmoModeConfig::Scale => GizmoMode::Scale,
        };

        let orientation = match tool_config.orientation {
            GizmoOrientationConfig::Local => GizmoOrientation::Local,
            GizmoOrientationConfig::Global => GizmoOrientation::Global,
        };

        let manual_axis_constraint = match tool_config.axis_lock {
            GizmoAxisLockConfig::Free => None,
            GizmoAxisLockConfig::AxisX => Some(GizmoAxisConstraint::AxisX),
            GizmoAxisLockConfig::AxisY => Some(GizmoAxisConstraint::AxisY),
            GizmoAxisLockConfig::AxisZ => Some(GizmoAxisConstraint::AxisZ),
        };

        Self {
            mode,
            orientation,
            manual_axis_constraint,
            snapping_enabled: snap_config.enabled,
            translate_snap: snap_config.translate_step.max(0.001),
            rotate_snap_degrees: snap_config.rotate_step_degrees.max(0.1),
            scale_snap: snap_config.scale_step.max(0.001),
            ..Self::default()
        }
    }
}

#[derive(Clone, Debug)]
struct ViewportOverlayState {
    show_grid: bool,
    show_collider_wireframes: bool,
    show_entity_labels: bool,
    show_fps: bool,
}

impl Default for ViewportOverlayState {
    fn default() -> Self {
        Self {
            show_grid: true,
            show_collider_wireframes: false,
            show_entity_labels: false,
            show_fps: true,
        }
    }
}

impl ViewportOverlayState {
    fn from_config(config: &ViewportOverlayConfig) -> Self {
        Self {
            show_grid: config.show_grid,
            show_collider_wireframes: config.show_collider_wireframes,
            show_entity_labels: config.show_entity_labels,
            show_fps: config.show_fps,
        }
    }
}

struct ConsolePanel {
    entries: Vec<LogEntry>,
    filter: LogLevel,
    auto_scroll: bool,
}

impl Default for ConsolePanel {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            filter: LogLevel::Trace,
            auto_scroll: true,
        }
    }
}

impl ConsolePanel {
    fn push(
        &mut self,
        level: LogLevel,
        message: impl Into<String>,
        module: impl Into<String>,
        timestamp: f64,
    ) {
        self.entries.push(LogEntry {
            level,
            message: message.into(),
            module: module.into(),
            timestamp,
        });

        const MAX_ENTRIES: usize = 2_000;
        if self.entries.len() > MAX_ENTRIES {
            let overflow = self.entries.len() - MAX_ENTRIES;
            self.entries.drain(0..overflow);
        }
    }

    fn show(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Filter:");
            ui.selectable_value(&mut self.filter, LogLevel::Trace, "All");
            ui.selectable_value(&mut self.filter, LogLevel::Info, "Info+");
            ui.selectable_value(&mut self.filter, LogLevel::Warn, "Warn+");
            ui.selectable_value(&mut self.filter, LogLevel::Error, "Error");
            ui.separator();
            if ui.button("Clear").clicked() {
                self.entries.clear();
            }
            ui.checkbox(&mut self.auto_scroll, "Auto-scroll");
        });

        ui.separator();

        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .stick_to_bottom(self.auto_scroll)
            .show(ui, |ui| {
                for entry in self
                    .entries
                    .iter()
                    .filter(|entry| entry.level >= self.filter)
                {
                    let color = match entry.level {
                        LogLevel::Error => egui::Color32::RED,
                        LogLevel::Warn => egui::Color32::YELLOW,
                        LogLevel::Info => egui::Color32::WHITE,
                        LogLevel::Trace => egui::Color32::GRAY,
                    };

                    ui.colored_label(
                        color,
                        format!(
                            "[{:.2}] [{}] {}",
                            entry.timestamp, entry.module, entry.message
                        ),
                    );
                }
            });
    }
}

pub struct EditorApp {
    pub dock_state: DockState<Tab>,
    pub world: World,
    pub asset_server: AssetServer,
    pub file_path: Option<PathBuf>,
    pub unsaved_changes: bool,
    pub show_about: bool,
    config: EditorConfig,
    selection: Selection,
    command_history: CommandHistory,
    asset_browser: AssetBrowserState,
    asset_browser_ops: AssetBrowserFileOpsState,
    scene_filter_query: String,
    renaming_entity: Option<Entity>,
    rename_buffer: String,
    rename_focus_pending: bool,
    dragging_entity: Option<Entity>,
    wgpu_render_state: Option<eframe::egui_wgpu::RenderState>,
    viewport_renderer: ViewportRenderer,
    editor_camera: EditorCamera,
    gizmo_state: GizmoState,
    viewport_overlay: ViewportOverlayState,
    started_at: Instant,
    console: ConsolePanel,
}

impl EditorApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let config = EditorConfig::load();
        let editor_camera = Self::editor_camera_from_config(&config);
        let viewport_overlay = ViewportOverlayState::from_config(&config.viewport_overlay);
        let gizmo_state = GizmoState::from_configs(&config.gizmo_snap, &config.gizmo_tool);
        let asset_server = AssetServer::new("assets");
        let asset_browser = AssetBrowserState::new(
            PathBuf::from(asset_server.root().as_str()),
            &config.asset_browser,
        );
        let mut world = build_editor_world();
        world.spawn(EditorEntityBundle::default());

        let mut console = ConsolePanel::default();
        console.push(LogLevel::Info, "Editor initialized", "engine::editor", 0.0);

        let mut app = Self {
            dock_state: config.dock_state.clone(),
            world,
            asset_server,
            file_path: None,
            unsaved_changes: false,
            show_about: false,
            config,
            selection: Selection::default(),
            command_history: CommandHistory::new(100),
            asset_browser,
            asset_browser_ops: AssetBrowserFileOpsState::default(),
            scene_filter_query: String::new(),
            renaming_entity: None,
            rename_buffer: String::new(),
            rename_focus_pending: false,
            dragging_entity: None,
            wgpu_render_state: cc.wgpu_render_state.clone(),
            viewport_renderer: ViewportRenderer::new(),
            editor_camera,
            gizmo_state,
            viewport_overlay,
            started_at: Instant::now(),
            console,
        };

        if let Some(last_scene) = app.config.last_opened_scene.clone() {
            if last_scene.exists() {
                if let Err(error) = app.load_scene(&last_scene) {
                    app.log_message(
                        LogLevel::Warn,
                        format!(
                            "Failed to restore last scene {}: {}",
                            last_scene.display(),
                            error
                        ),
                    );
                } else {
                    app.file_path = Some(last_scene.clone());
                    app.log_message(
                        LogLevel::Info,
                        format!("Restored last scene {}", last_scene.display()),
                    );
                }
            }
        }

        app.bootstrap_viewport_scene();

        app
    }

    fn now_seconds(&self) -> f64 {
        self.started_at.elapsed().as_secs_f64()
    }

    fn editor_camera_from_config(config: &EditorConfig) -> EditorCamera {
        let mut camera = EditorCamera::default();
        let saved = &config.viewport_camera;
        camera.restore_orbit(
            Vec3::from_array(saved.target),
            saved.distance,
            saved.yaw,
            saved.pitch,
        );
        camera
    }

    fn apply_selection_hint(&mut self, entity: Option<Entity>) {
        if let Some(entity) = entity {
            if self.world.get_entity(entity).is_ok() {
                self.selection.select_single(entity);
                return;
            }
        }

        self.selection.deselect();
    }

    fn validate_selection_state(&mut self) {
        if let Some(entity) = self.selection.primary() {
            if self.world.get_entity(entity).is_err() {
                self.selection.deselect();
            }
        }

        if let Some(active_drag) = self.gizmo_state.active_drag.as_ref() {
            if self.world.get_entity(active_drag.entity).is_err() {
                self.gizmo_state.active_drag = None;
            }
        }

        if let Some(entity) = self.renaming_entity {
            if self.world.get_entity(entity).is_err() {
                self.cancel_rename_entity();
            }
        }

        self.asset_browser.validate_selected_path();
    }

    fn log_message(&mut self, level: LogLevel, message: impl Into<String>) {
        let timestamp = self.now_seconds();
        self.console
            .push(level, message.into(), "engine::editor", timestamp);
    }

    fn sync_config_from_runtime(&mut self) {
        self.config.dock_state = self.dock_state.clone();
        self.config.last_opened_scene = self.file_path.clone();
        self.config.viewport_camera = ViewportCameraConfig {
            target: self.editor_camera.target.to_array(),
            distance: self.editor_camera.distance,
            yaw: self.editor_camera.yaw,
            pitch: self.editor_camera.pitch,
        };
        self.config.viewport_overlay = ViewportOverlayConfig {
            show_grid: self.viewport_overlay.show_grid,
            show_collider_wireframes: self.viewport_overlay.show_collider_wireframes,
            show_entity_labels: self.viewport_overlay.show_entity_labels,
            show_fps: self.viewport_overlay.show_fps,
        };
        self.config.gizmo_snap = GizmoSnapConfig {
            enabled: self.gizmo_state.snapping_enabled,
            translate_step: self.gizmo_state.translate_snap,
            rotate_step_degrees: self.gizmo_state.rotate_snap_degrees,
            scale_step: self.gizmo_state.scale_snap,
        };
        self.config.gizmo_tool = GizmoToolConfig {
            mode: match self.gizmo_state.mode {
                GizmoMode::Translate => GizmoModeConfig::Translate,
                GizmoMode::Rotate => GizmoModeConfig::Rotate,
                GizmoMode::Scale => GizmoModeConfig::Scale,
            },
            orientation: match self.gizmo_state.orientation {
                GizmoOrientation::Local => GizmoOrientationConfig::Local,
                GizmoOrientation::Global => GizmoOrientationConfig::Global,
            },
            axis_lock: match self.gizmo_state.manual_axis_constraint {
                None => GizmoAxisLockConfig::Free,
                Some(GizmoAxisConstraint::AxisX) => GizmoAxisLockConfig::AxisX,
                Some(GizmoAxisConstraint::AxisY) => GizmoAxisLockConfig::AxisY,
                Some(GizmoAxisConstraint::AxisZ) => GizmoAxisLockConfig::AxisZ,
            },
        };
        self.config.asset_browser = self.asset_browser.to_config();
    }

    fn persist_config(&mut self) {
        self.sync_config_from_runtime();
        if let Err(error) = self.config.save() {
            self.log_message(
                LogLevel::Warn,
                format!("Failed to persist editor config: {}", error),
            );
        }
    }

    fn bootstrap_viewport_scene(&mut self) {
        self.ensure_viewport_world_defaults();
        self.ensure_primary_camera();
        self.ensure_preview_mesh();
        self.apply_editor_camera_to_primary_camera(1280, 720);
        self.sync_global_transforms_for_viewport();
    }

    fn apply_editor_camera_to_primary_camera(&mut self, viewport_width: u32, viewport_height: u32) {
        let position = self.editor_camera.eye_position();
        let rotation = self.editor_camera.rotation();

        let mut query = self
            .world
            .query_filtered::<(&mut Transform, &mut Camera3d), With<PrimaryCamera>>();

        if let Some((mut transform, mut camera)) = query.iter_mut(&mut self.world).next() {
            camera.fov_y_radians = self.editor_camera.fov_y_radians;
            camera.near = self.editor_camera.near;
            camera.far = self.editor_camera.far;
            camera.set_aspect_ratio_from_viewport(viewport_width, viewport_height);

            transform.translation = position;
            transform.rotation = rotation;
            transform.scale = Vec3::ONE;
        }
    }

    fn focus_editor_camera_on_selection(&mut self) -> bool {
        let Some(entity) = self.selection.primary() else {
            return false;
        };

        let focus_position = self
            .world
            .get::<GlobalTransform>(entity)
            .map(GlobalTransform::translation)
            .or_else(|| {
                self.world
                    .get::<Transform>(entity)
                    .map(|transform| transform.translation)
            });

        let Some(position) = focus_position else {
            return false;
        };

        self.editor_camera.focus_on(position);
        let name = self.entity_name_for_scene_tree(entity);
        self.log_message(LogLevel::Info, format!("Focused camera on {}", name));
        true
    }

    fn ensure_viewport_world_defaults(&mut self) {
        let entities: Vec<Entity> = self
            .world
            .iter_entities()
            .map(|entity| entity.id())
            .collect();

        for entity in entities {
            let has_transform = self.world.get::<Transform>(entity).is_some();
            let has_global = self.world.get::<GlobalTransform>(entity).is_some();
            let has_camera3d = self.world.get::<Camera3d>(entity).is_some();
            let has_camera2d = self.world.get::<Camera2d>(entity).is_some();
            let has_mesh = self.world.get::<MeshRenderable3d>(entity).is_some();
            let has_sprite = self.world.get::<SpriteRenderable2d>(entity).is_some();
            let has_visible = self.world.get::<Visible>(entity).is_some();
            let has_layer3d = self.world.get::<RenderLayer3D>(entity).is_some();
            let has_layer2d = self.world.get::<RenderLayer2D>(entity).is_some();

            if let Ok(mut entity_ref) = self.world.get_entity_mut(entity) {
                if has_transform && !has_global {
                    entity_ref.insert(GlobalTransform::default());
                }

                if (has_camera3d || has_camera2d || has_mesh || has_sprite) && !has_visible {
                    entity_ref.insert(Visible);
                }

                if has_mesh && !has_layer3d {
                    entity_ref.insert(RenderLayer3D);
                }

                if has_sprite && !has_layer2d {
                    entity_ref.insert(RenderLayer2D);
                }
            }
        }
    }

    fn ensure_primary_camera(&mut self) {
        let has_primary_camera = self.world.iter_entities().any(|entity| {
            let id = entity.id();
            self.world.get::<Camera3d>(id).is_some()
                && self.world.get::<PrimaryCamera>(id).is_some()
        });

        if has_primary_camera {
            return;
        }

        self.world.spawn((
            EntityName::new("Editor Camera"),
            SpatialBundle {
                transform: Transform::from_xyz(0.0, 4.0, 10.0),
                ..SpatialBundle::default()
            },
            Camera3d::default(),
            PrimaryCamera,
            Visible,
        ));
        self.log_message(
            LogLevel::Info,
            "Inserted default editor camera for viewport rendering",
        );
    }

    fn ensure_preview_mesh(&mut self) {
        let has_mesh = self
            .world
            .iter_entities()
            .any(|entity| self.world.get::<MeshRenderable3d>(entity.id()).is_some());

        if has_mesh {
            return;
        }

        let mesh = match self.asset_server.load_mesh_handle("meshes/cube.glb") {
            Ok(handle) => handle,
            Err(error) => {
                self.log_message(
                    LogLevel::Warn,
                    format!("Preview mesh load failed (meshes/cube.glb): {}", error),
                );
                return;
            }
        };

        let texture = match self
            .asset_server
            .load_texture_handle("textures/placeholder.png")
        {
            Ok(handle) => handle,
            Err(error) => {
                self.log_message(
                    LogLevel::Warn,
                    format!(
                        "Preview texture load failed (textures/placeholder.png): {}",
                        error
                    ),
                );
                return;
            }
        };

        let material = match self
            .asset_server
            .load_material_handle("materials/default.ron")
        {
            Ok(handle) => handle,
            Err(error) => {
                self.log_message(
                    LogLevel::Warn,
                    format!(
                        "Preview material load failed (materials/default.ron): {}",
                        error
                    ),
                );
                return;
            }
        };

        self.world.spawn((
            EntityName::new("Preview Cube"),
            SpatialBundle {
                transform: Transform::from_xyz(0.0, 0.5, 0.0),
                ..SpatialBundle::default()
            },
            MeshRenderable3d::new(mesh, texture, material),
            Visible,
            RenderLayer3D,
        ));

        self.log_message(
            LogLevel::Info,
            "Inserted default preview cube for viewport validation",
        );
    }

    fn sync_global_transforms_for_viewport(&mut self) {
        if let Err(error) = self
            .world
            .run_system_once(engine_core::propagate_transforms)
        {
            self.log_message(
                LogLevel::Warn,
                format!(
                    "Transform propagation failed before viewport render: {}",
                    error
                ),
            );
        }
    }

    fn mesh_local_bounds(&self, mesh_renderable: &MeshRenderable3d) -> (Vec3, Vec3) {
        self.asset_server
            .mesh_payload(mesh_renderable.mesh)
            .and_then(mesh_local_bounds)
            .unwrap_or((Vec3::splat(-0.5), Vec3::splat(0.5)))
    }

    fn draw_gizmo_toolbar(
        &mut self,
        ctx: &egui::Context,
        viewport_rect: egui::Rect,
        viewport_has_focus: bool,
    ) {
        if viewport_has_focus {
            if ctx.input(|input| input.key_pressed(egui::Key::W)) {
                self.gizmo_state.mode = GizmoMode::Translate;
            }
            if ctx.input(|input| input.key_pressed(egui::Key::E)) {
                self.gizmo_state.mode = GizmoMode::Rotate;
            }
            if ctx.input(|input| input.key_pressed(egui::Key::R)) {
                self.gizmo_state.mode = GizmoMode::Scale;
            }
            if ctx.input(|input| input.key_pressed(egui::Key::C)) {
                self.gizmo_state.mode = cycle_gizmo_mode(self.gizmo_state.mode);
            }
            if ctx.input(|input| input.key_pressed(egui::Key::Q)) {
                self.gizmo_state.orientation =
                    toggle_gizmo_orientation(self.gizmo_state.orientation);
            }
            if ctx.input(|input| input.key_pressed(egui::Key::X)) {
                self.gizmo_state.snapping_enabled = !self.gizmo_state.snapping_enabled;
            }
            if ctx.input(|input| input.key_pressed(egui::Key::H)) {
                self.gizmo_state.manual_axis_constraint = Some(GizmoAxisConstraint::AxisX);
            }
            if ctx.input(|input| input.key_pressed(egui::Key::V)) {
                self.gizmo_state.manual_axis_constraint = Some(GizmoAxisConstraint::AxisY);
            }
            if ctx.input(|input| input.key_pressed(egui::Key::Z)) {
                self.gizmo_state.manual_axis_constraint = Some(GizmoAxisConstraint::AxisZ);
            }
            if ctx.input(|input| input.key_pressed(egui::Key::N)) {
                self.gizmo_state.manual_axis_constraint = None;
            }
            if ctx.input(|input| input.key_pressed(egui::Key::G)) {
                self.viewport_overlay.show_grid = !self.viewport_overlay.show_grid;
            }
            if ctx.input(|input| input.key_pressed(egui::Key::B)) {
                self.viewport_overlay.show_collider_wireframes =
                    !self.viewport_overlay.show_collider_wireframes;
            }
            if ctx.input(|input| input.key_pressed(egui::Key::L)) {
                self.viewport_overlay.show_entity_labels =
                    !self.viewport_overlay.show_entity_labels;
            }
        }

        let toolbar_position = viewport_rect.left_top() + egui::vec2(12.0, 12.0);
        egui::Area::new(egui::Id::new("viewport_gizmo_toolbar"))
            .fixed_pos(toolbar_position)
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                egui::Frame::window(ui.style())
                    .inner_margin(egui::Margin::same(6))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            if ui
                                .selectable_label(
                                    self.gizmo_state.mode == GizmoMode::Translate,
                                    "Move (W)",
                                )
                                .clicked()
                            {
                                self.gizmo_state.mode = GizmoMode::Translate;
                            }

                            if ui
                                .selectable_label(
                                    self.gizmo_state.mode == GizmoMode::Rotate,
                                    "Rotate (E)",
                                )
                                .clicked()
                            {
                                self.gizmo_state.mode = GizmoMode::Rotate;
                            }

                            if ui
                                .selectable_label(
                                    self.gizmo_state.mode == GizmoMode::Scale,
                                    "Scale (R)",
                                )
                                .clicked()
                            {
                                self.gizmo_state.mode = GizmoMode::Scale;
                            }

                            ui.separator();

                            if ui
                                .selectable_label(
                                    self.gizmo_state.orientation == GizmoOrientation::Local,
                                    "Local",
                                )
                                .clicked()
                            {
                                self.gizmo_state.orientation = GizmoOrientation::Local;
                            }

                            if ui
                                .selectable_label(
                                    self.gizmo_state.orientation == GizmoOrientation::Global,
                                    "Global",
                                )
                                .clicked()
                            {
                                self.gizmo_state.orientation = GizmoOrientation::Global;
                            }

                            if self.gizmo_state.active_drag.is_some() {
                                ui.separator();
                                ui.label("Dragging");
                            }

                            ui.separator();
                            if ui
                                .selectable_label(
                                    self.gizmo_state.manual_axis_constraint.is_none(),
                                    "Free (N)",
                                )
                                .clicked()
                            {
                                self.gizmo_state.manual_axis_constraint = None;
                            }

                            if ui
                                .selectable_label(
                                    self.gizmo_state.manual_axis_constraint
                                        == Some(GizmoAxisConstraint::AxisX),
                                    "Lock X (H)",
                                )
                                .clicked()
                            {
                                self.gizmo_state.manual_axis_constraint =
                                    Some(GizmoAxisConstraint::AxisX);
                            }

                            if ui
                                .selectable_label(
                                    self.gizmo_state.manual_axis_constraint
                                        == Some(GizmoAxisConstraint::AxisY),
                                    "Lock Y (V)",
                                )
                                .clicked()
                            {
                                self.gizmo_state.manual_axis_constraint =
                                    Some(GizmoAxisConstraint::AxisY);
                            }

                            if ui
                                .selectable_label(
                                    self.gizmo_state.manual_axis_constraint
                                        == Some(GizmoAxisConstraint::AxisZ),
                                    "Lock Z (Z)",
                                )
                                .clicked()
                            {
                                self.gizmo_state.manual_axis_constraint =
                                    Some(GizmoAxisConstraint::AxisZ);
                            }

                            ui.separator();
                            if ui
                                .selectable_label(self.gizmo_state.snapping_enabled, "Snap (X)")
                                .clicked()
                            {
                                self.gizmo_state.snapping_enabled =
                                    !self.gizmo_state.snapping_enabled;
                            }

                            ui.menu_button("Snap Values", |ui| {
                                ui.label("Translate");
                                ui.add(
                                    egui::DragValue::new(&mut self.gizmo_state.translate_snap)
                                        .speed(0.05)
                                        .range(0.001..=1000.0),
                                );

                                ui.label("Rotate (deg)");
                                ui.add(
                                    egui::DragValue::new(&mut self.gizmo_state.rotate_snap_degrees)
                                        .speed(0.5)
                                        .range(0.1..=180.0),
                                );

                                ui.label("Scale");
                                ui.add(
                                    egui::DragValue::new(&mut self.gizmo_state.scale_snap)
                                        .speed(0.01)
                                        .range(0.001..=10.0),
                                );
                            });

                            ui.menu_button("Overlay", |ui| {
                                ui.checkbox(&mut self.viewport_overlay.show_grid, "Grid (G)");
                                ui.checkbox(
                                    &mut self.viewport_overlay.show_collider_wireframes,
                                    "Collider Wireframes (B)",
                                );
                                ui.checkbox(
                                    &mut self.viewport_overlay.show_entity_labels,
                                    "Entity Labels (L)",
                                );
                                ui.checkbox(&mut self.viewport_overlay.show_fps, "FPS");
                            });
                        });
                    });
            });
    }

    fn cancel_active_gizmo_drag(&mut self, emit_log: bool) -> bool {
        let Some(active_drag) = self.gizmo_state.active_drag.take() else {
            return false;
        };

        if let Ok(mut entity_ref) = self.world.get_entity_mut(active_drag.entity) {
            entity_ref.insert(active_drag.initial_local_transform);
            self.selection.select_single(active_drag.entity);
            self.sync_global_transforms_for_viewport();

            if emit_log {
                self.log_message(LogLevel::Info, "Canceled active gizmo interaction");
            }

            return true;
        }

        false
    }

    fn commit_active_gizmo_drag(&mut self) -> bool {
        let Some(active_drag) = self.gizmo_state.active_drag.take() else {
            return false;
        };

        if self.world.get_entity(active_drag.entity).is_err() {
            return false;
        }

        if transforms_approximately_equal(
            &active_drag.initial_local_transform,
            &active_drag.latest_local_transform,
        ) {
            return false;
        }

        let _ = self.command_history.execute(
            Box::new(SetComponentCommand {
                entity: active_drag.entity,
                component_name: "Transform".to_owned(),
                old_value: Box::new(active_drag.initial_local_transform),
                new_value: Box::new(active_drag.latest_local_transform),
                desc: "Transform gizmo interaction".to_owned(),
            }),
            &mut self.world,
        );

        self.selection.select_single(active_drag.entity);
        self.unsaved_changes = true;
        self.log_message(LogLevel::Info, "Committed gizmo transform interaction");
        true
    }

    fn begin_gizmo_drag(&mut self, entity: Entity, pointer_position: egui::Pos2) {
        let initial_local_transform = self
            .world
            .get::<Transform>(entity)
            .cloned()
            .unwrap_or_default();

        self.gizmo_state.active_drag = Some(GizmoDragInteraction {
            entity,
            drag_start_pointer: pointer_position,
            initial_local_transform: initial_local_transform.clone(),
            latest_local_transform: initial_local_transform,
            drag_intent: None,
            axis_constraint: None,
        });
    }

    fn apply_gizmo_drag_preview(
        &mut self,
        viewport_rect: egui::Rect,
        pointer_position: egui::Pos2,
        axis_lock_enabled: bool,
    ) {
        let mode = self.gizmo_state.mode;
        let orientation = self.gizmo_state.orientation;
        let manual_axis_constraint = self.gizmo_state.manual_axis_constraint;
        let snapping_enabled = self.gizmo_state.snapping_enabled;
        let translate_snap = self.gizmo_state.translate_snap.max(0.001);
        let rotate_snap_degrees = self.gizmo_state.rotate_snap_degrees.max(0.1);
        let scale_snap = self.gizmo_state.scale_snap.max(0.001);
        let camera_distance = self.editor_camera.distance;
        let camera_fov_y_radians = self.editor_camera.fov_y_radians;

        let Some(active_drag) = self.gizmo_state.active_drag.as_mut() else {
            return;
        };

        let drag_delta = pointer_position - active_drag.drag_start_pointer;
        let initial = active_drag.initial_local_transform.clone();
        let normalized_x = drag_delta.x / viewport_rect.width().max(1.0);
        let normalized_y = drag_delta.y / viewport_rect.height().max(1.0);
        let drag_intent = compute_gizmo_drag_intent(mode, normalized_x, normalized_y);
        let axis_constraint = compute_gizmo_axis_constraint(
            mode,
            manual_axis_constraint,
            axis_lock_enabled,
            drag_intent,
        );
        active_drag.drag_intent = Some(drag_intent);
        active_drag.axis_constraint = axis_constraint;

        let new_local_transform = compute_gizmo_drag_transform(
            &initial,
            mode,
            orientation,
            axis_constraint,
            snapping_enabled,
            normalized_x,
            normalized_y,
            camera_distance,
            camera_fov_y_radians,
            translate_snap,
            rotate_snap_degrees,
            scale_snap,
        );

        active_drag.latest_local_transform = new_local_transform.clone();

        if let Ok(mut entity_ref) = self.world.get_entity_mut(active_drag.entity) {
            entity_ref.insert(new_local_transform);
            self.sync_global_transforms_for_viewport();
        }
    }

    fn draw_transform_gizmo(&mut self, ui: &mut egui::Ui, viewport_response: &egui::Response) {
        self.draw_gizmo_toolbar(
            ui.ctx(),
            viewport_response.rect,
            viewport_response.has_focus(),
        );

        let Some(entity) = self.selection.primary() else {
            if self.gizmo_state.active_drag.is_some()
                && !ui.input(|input| input.pointer.primary_down())
            {
                let _ = self.commit_active_gizmo_drag();
            }
            return;
        };

        if viewport_response.drag_started_by(egui::PointerButton::Primary)
            && self.gizmo_state.active_drag.is_none()
        {
            if let Some(pointer_position) = viewport_response.interact_pointer_pos() {
                self.begin_gizmo_drag(entity, pointer_position);
            }
        }

        if viewport_response.dragged_by(egui::PointerButton::Primary) {
            if self.gizmo_state.active_drag.is_none() {
                if let Some(pointer_position) = viewport_response.interact_pointer_pos() {
                    self.begin_gizmo_drag(entity, pointer_position);
                }
            }

            if let Some(pointer_position) = viewport_response.interact_pointer_pos() {
                let axis_lock_enabled = ui.input(|input| input.modifiers.shift);
                self.apply_gizmo_drag_preview(
                    viewport_response.rect,
                    pointer_position,
                    axis_lock_enabled,
                );
                ui.ctx().request_repaint();
            }
        }

        let center = viewport_response.rect.left_top() + egui::vec2(16.0, 56.0);
        let gizmo_color = match self.gizmo_state.mode {
            GizmoMode::Translate => egui::Color32::from_rgb(110, 190, 255),
            GizmoMode::Rotate => egui::Color32::from_rgb(255, 190, 80),
            GizmoMode::Scale => egui::Color32::from_rgb(130, 220, 130),
        };
        ui.painter()
            .circle_stroke(center, 8.0, egui::Stroke::new(2.0, gizmo_color));

        let active_drag_intent = self
            .gizmo_state
            .active_drag
            .as_ref()
            .and_then(|drag| drag.drag_intent);
        let active_axis_constraint = self
            .gizmo_state
            .active_drag
            .as_ref()
            .and_then(|drag| drag.axis_constraint);
        draw_gizmo_mode_guides(
            ui.painter(),
            center,
            self.gizmo_state.mode,
            active_drag_intent,
            active_axis_constraint,
        );

        let gizmo_state_text = format!(
            "{} | {} | Lock {} | Snap {}",
            gizmo_mode_label(self.gizmo_state.mode),
            gizmo_orientation_label(self.gizmo_state.orientation),
            gizmo_manual_axis_lock_label(self.gizmo_state.manual_axis_constraint),
            if self.gizmo_state.snapping_enabled {
                "On"
            } else {
                "Off"
            }
        );
        ui.painter().text(
            center + egui::vec2(14.0, 0.0),
            egui::Align2::LEFT_CENTER,
            gizmo_state_text,
            egui::FontId::monospace(11.0),
            egui::Color32::from_gray(220),
        );

        if let Some(intent) = active_drag_intent {
            let drag_intent_text = format!(
                "Drag: {}",
                gizmo_drag_intent_label(self.gizmo_state.mode, intent)
            );
            ui.painter().text(
                center + egui::vec2(14.0, 12.0),
                egui::Align2::LEFT_TOP,
                drag_intent_text,
                egui::FontId::monospace(10.5),
                egui::Color32::from_gray(200),
            );
        }

        if let Some(axis_constraint) = active_axis_constraint {
            let axis_source = if self.gizmo_state.manual_axis_constraint.is_some() {
                "Manual"
            } else {
                "Auto"
            };
            let axis_lock_text = format!(
                "Axis Lock: {} ({})",
                gizmo_axis_constraint_label(axis_constraint),
                axis_source
            );
            ui.painter().text(
                center + egui::vec2(14.0, 24.0),
                egui::Align2::LEFT_TOP,
                axis_lock_text,
                egui::FontId::monospace(10.5),
                egui::Color32::from_gray(190),
            );
        }

        if self.gizmo_state.active_drag.is_some()
            && !ui.input(|input| input.pointer.primary_down())
            && self.commit_active_gizmo_drag()
        {
            self.sync_global_transforms_for_viewport();
        }
    }

    fn draw_viewport_overlay(&mut self, ui: &egui::Ui, viewport_rect: egui::Rect) {
        let view_proj = self
            .editor_camera
            .projection_matrix(viewport_rect.width(), viewport_rect.height())
            * self.editor_camera.view_matrix();

        if self.viewport_overlay.show_grid {
            self.draw_viewport_grid_overlay(ui, viewport_rect, view_proj);
        }

        if self.viewport_overlay.show_collider_wireframes {
            self.draw_collider_wireframes_overlay(ui, viewport_rect, view_proj);
        }

        if self.viewport_overlay.show_entity_labels {
            self.draw_entity_labels_overlay(ui, viewport_rect, view_proj);
        }

        if self.viewport_overlay.show_fps {
            let predicted_dt = ui.ctx().input(|input| input.predicted_dt);
            let fps = if predicted_dt > f32::EPSILON {
                1.0 / predicted_dt
            } else {
                0.0
            };
            ui.painter().text(
                viewport_rect.right_top() + egui::vec2(-8.0, 8.0),
                egui::Align2::RIGHT_TOP,
                format!("{fps:.0} FPS"),
                egui::FontId::monospace(12.0),
                egui::Color32::WHITE,
            );
        }
    }

    fn draw_viewport_grid_overlay(
        &self,
        ui: &egui::Ui,
        viewport_rect: egui::Rect,
        view_proj: Mat4,
    ) {
        const HALF_EXTENT: i32 = 20;
        let painter = ui.painter();

        for i in -HALF_EXTENT..=HALF_EXTENT {
            let major = i % 5 == 0;
            let stroke = if i == 0 {
                egui::Stroke::new(1.5, egui::Color32::from_rgb(220, 80, 80))
            } else if major {
                egui::Stroke::new(1.0, egui::Color32::from_gray(130))
            } else {
                egui::Stroke::new(1.0, egui::Color32::from_gray(80))
            };

            let from = Vec3::new(i as f32, 0.0, -HALF_EXTENT as f32);
            let to = Vec3::new(i as f32, 0.0, HALF_EXTENT as f32);
            if let (Some(a), Some(b)) = (
                project_world_to_viewport(view_proj, viewport_rect, from),
                project_world_to_viewport(view_proj, viewport_rect, to),
            ) {
                painter.line_segment([a, b], stroke);
            }
        }

        for i in -HALF_EXTENT..=HALF_EXTENT {
            let major = i % 5 == 0;
            let stroke = if i == 0 {
                egui::Stroke::new(1.5, egui::Color32::from_rgb(90, 140, 255))
            } else if major {
                egui::Stroke::new(1.0, egui::Color32::from_gray(130))
            } else {
                egui::Stroke::new(1.0, egui::Color32::from_gray(80))
            };

            let from = Vec3::new(-HALF_EXTENT as f32, 0.0, i as f32);
            let to = Vec3::new(HALF_EXTENT as f32, 0.0, i as f32);
            if let (Some(a), Some(b)) = (
                project_world_to_viewport(view_proj, viewport_rect, from),
                project_world_to_viewport(view_proj, viewport_rect, to),
            ) {
                painter.line_segment([a, b], stroke);
            }
        }
    }

    fn draw_collider_wireframes_overlay(
        &mut self,
        ui: &egui::Ui,
        viewport_rect: egui::Rect,
        view_proj: Mat4,
    ) {
        let mut query = self
            .world
            .query_filtered::<(&ColliderShape3D, &GlobalTransform), With<Visible>>();

        for (shape, global_transform) in query.iter(&self.world) {
            match shape {
                ColliderShape3D::Box { half_extents } => {
                    self.draw_world_space_wire_box(
                        ui,
                        viewport_rect,
                        view_proj,
                        global_transform.0,
                        *half_extents,
                        egui::Color32::from_rgb(140, 220, 140),
                    );
                }
                ColliderShape3D::Sphere { radius } => {
                    self.draw_world_space_ring(
                        ui,
                        viewport_rect,
                        view_proj,
                        global_transform.0,
                        Vec3::ZERO,
                        Vec3::X,
                        Vec3::Y,
                        *radius,
                        egui::Color32::from_rgb(140, 220, 140),
                    );
                    self.draw_world_space_ring(
                        ui,
                        viewport_rect,
                        view_proj,
                        global_transform.0,
                        Vec3::ZERO,
                        Vec3::X,
                        Vec3::Z,
                        *radius,
                        egui::Color32::from_rgb(140, 220, 140),
                    );
                    self.draw_world_space_ring(
                        ui,
                        viewport_rect,
                        view_proj,
                        global_transform.0,
                        Vec3::ZERO,
                        Vec3::Y,
                        Vec3::Z,
                        *radius,
                        egui::Color32::from_rgb(140, 220, 140),
                    );
                }
                ColliderShape3D::Capsule {
                    half_height,
                    radius,
                } => {
                    let top = Vec3::Y * *half_height;
                    let bottom = Vec3::NEG_Y * *half_height;
                    self.draw_world_space_ring(
                        ui,
                        viewport_rect,
                        view_proj,
                        global_transform.0,
                        top,
                        Vec3::X,
                        Vec3::Z,
                        *radius,
                        egui::Color32::from_rgb(140, 220, 140),
                    );
                    self.draw_world_space_ring(
                        ui,
                        viewport_rect,
                        view_proj,
                        global_transform.0,
                        bottom,
                        Vec3::X,
                        Vec3::Z,
                        *radius,
                        egui::Color32::from_rgb(140, 220, 140),
                    );

                    let spine_points = [
                        (
                            Vec3::new(*radius, *half_height, 0.0),
                            Vec3::new(*radius, -*half_height, 0.0),
                        ),
                        (
                            Vec3::new(-*radius, *half_height, 0.0),
                            Vec3::new(-*radius, -*half_height, 0.0),
                        ),
                        (
                            Vec3::new(0.0, *half_height, *radius),
                            Vec3::new(0.0, -*half_height, *radius),
                        ),
                        (
                            Vec3::new(0.0, *half_height, -*radius),
                            Vec3::new(0.0, -*half_height, -*radius),
                        ),
                    ];

                    for (from_local, to_local) in spine_points {
                        let from_world = global_transform.0.transform_point3(from_local);
                        let to_world = global_transform.0.transform_point3(to_local);
                        if let (Some(a), Some(b)) = (
                            project_world_to_viewport(view_proj, viewport_rect, from_world),
                            project_world_to_viewport(view_proj, viewport_rect, to_world),
                        ) {
                            ui.painter().line_segment(
                                [a, b],
                                egui::Stroke::new(1.0, egui::Color32::from_rgb(140, 220, 140)),
                            );
                        }
                    }
                }
                ColliderShape3D::Trimesh => {}
            }
        }
    }

    fn draw_entity_labels_overlay(
        &mut self,
        ui: &egui::Ui,
        viewport_rect: egui::Rect,
        view_proj: Mat4,
    ) {
        let mut query = self
            .world
            .query_filtered::<(Entity, &GlobalTransform, Option<&EntityName>), With<Visible>>();

        for (entity, global_transform, entity_name) in query.iter(&self.world) {
            let world_position = global_transform.translation() + Vec3::Y * 0.25;
            let Some(screen_position) =
                project_world_to_viewport(view_proj, viewport_rect, world_position)
            else {
                continue;
            };

            let label = entity_name
                .map(|name| name.0.clone())
                .unwrap_or_else(|| format!("Entity {:?}", entity));

            let color = if self.selection.primary() == Some(entity) {
                egui::Color32::from_rgb(255, 215, 120)
            } else {
                egui::Color32::from_rgb(220, 220, 220)
            };

            ui.painter().text(
                screen_position,
                egui::Align2::CENTER_BOTTOM,
                label,
                egui::FontId::monospace(11.0),
                color,
            );
        }
    }

    fn draw_world_space_wire_box(
        &self,
        ui: &egui::Ui,
        viewport_rect: egui::Rect,
        view_proj: Mat4,
        transform: Affine3A,
        half_extents: Vec3,
        color: egui::Color32,
    ) {
        let corners = [
            Vec3::new(-half_extents.x, -half_extents.y, -half_extents.z),
            Vec3::new(-half_extents.x, -half_extents.y, half_extents.z),
            Vec3::new(-half_extents.x, half_extents.y, -half_extents.z),
            Vec3::new(-half_extents.x, half_extents.y, half_extents.z),
            Vec3::new(half_extents.x, -half_extents.y, -half_extents.z),
            Vec3::new(half_extents.x, -half_extents.y, half_extents.z),
            Vec3::new(half_extents.x, half_extents.y, -half_extents.z),
            Vec3::new(half_extents.x, half_extents.y, half_extents.z),
        ];

        let mut projected = [None; 8];
        for (index, corner) in corners.into_iter().enumerate() {
            let world = transform.transform_point3(corner);
            projected[index] = project_world_to_viewport(view_proj, viewport_rect, world);
        }

        const EDGES: [(usize, usize); 12] = [
            (0, 1),
            (0, 2),
            (0, 4),
            (1, 3),
            (1, 5),
            (2, 3),
            (2, 6),
            (3, 7),
            (4, 5),
            (4, 6),
            (5, 7),
            (6, 7),
        ];

        for (from, to) in EDGES {
            if let (Some(a), Some(b)) = (projected[from], projected[to]) {
                ui.painter()
                    .line_segment([a, b], egui::Stroke::new(1.0, color));
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_world_space_ring(
        &self,
        ui: &egui::Ui,
        viewport_rect: egui::Rect,
        view_proj: Mat4,
        transform: Affine3A,
        center_local: Vec3,
        axis_a_local: Vec3,
        axis_b_local: Vec3,
        radius: f32,
        color: egui::Color32,
    ) {
        const SEGMENTS: usize = 24;
        let mut previous = None;
        let mut first = None;

        for segment in 0..=SEGMENTS {
            let t = (segment as f32 / SEGMENTS as f32) * std::f32::consts::TAU;
            let local = center_local
                + axis_a_local * (t.cos() * radius)
                + axis_b_local * (t.sin() * radius);
            let world = transform.transform_point3(local);
            let projected = project_world_to_viewport(view_proj, viewport_rect, world);

            if let Some(current) = projected {
                if first.is_none() {
                    first = Some(current);
                }
                if let Some(prev) = previous {
                    ui.painter()
                        .line_segment([prev, current], egui::Stroke::new(1.0, color));
                }
                previous = Some(current);
            } else {
                previous = None;
            }
        }

        if let (Some(a), Some(b)) = (previous, first) {
            ui.painter()
                .line_segment([a, b], egui::Stroke::new(1.0, color));
        }
    }

    fn try_pick_viewport_entity(&self, response: &egui::Response) -> Option<Entity> {
        let pointer_position = response.interact_pointer_pos()?;
        let (ray_origin, ray_direction) = viewport_pick_ray(
            pointer_position,
            response.rect,
            self.editor_camera.eye_position(),
            self.editor_camera.rotation(),
            self.editor_camera.fov_y_radians,
        )?;

        if let (Some(physics_world), Some(collider_entity_map)) = (
            self.world.get_resource::<PhysicsWorld3D>(),
            self.world.get_resource::<ColliderEntityMap3D>(),
        ) {
            if let Some(hit) = raycast(
                ray_origin,
                ray_direction,
                self.editor_camera.far,
                physics_world,
                collider_entity_map.as_map(),
            ) {
                return Some(hit.entity);
            }
        }

        let mut best_hit: Option<(Entity, f32)> = None;
        for entity_ref in self.world.iter_entities() {
            let entity = entity_ref.id();

            if self.world.get::<Visible>(entity).is_none() {
                continue;
            }

            let Some(global_transform) = self.world.get::<GlobalTransform>(entity) else {
                continue;
            };

            if let Some(mesh_renderable) = self.world.get::<MeshRenderable3d>(entity) {
                let (local_min, local_max) = self.mesh_local_bounds(mesh_renderable);
                let (world_min, world_max) =
                    transform_aabb(global_transform.0, local_min, local_max);

                if let Some(distance) = ray_intersects_aabb(
                    ray_origin,
                    ray_direction,
                    world_min,
                    world_max,
                    self.editor_camera.far,
                ) {
                    let replace = best_hit
                        .map(|(_, best_distance)| distance < best_distance)
                        .unwrap_or(true);
                    if replace {
                        best_hit = Some((entity, distance));
                    }
                }
                continue;
            }

            if let Some(sprite_renderable) = self.world.get::<SpriteRenderable2d>(entity) {
                let half_width = sprite_renderable.size[0].max(0.001) * 0.5;
                let half_height = sprite_renderable.size[1].max(0.001) * 0.5;
                let (world_min, world_max) = transform_aabb(
                    global_transform.0,
                    Vec3::new(-half_width, -half_height, -0.05),
                    Vec3::new(half_width, half_height, 0.05),
                );

                if let Some(distance) = ray_intersects_aabb(
                    ray_origin,
                    ray_direction,
                    world_min,
                    world_max,
                    self.editor_camera.far,
                ) {
                    let replace = best_hit
                        .map(|(_, best_distance)| distance < best_distance)
                        .unwrap_or(true);
                    if replace {
                        best_hit = Some((entity, distance));
                    }
                }
            }
        }

        best_hit.map(|(entity, _)| entity)
    }

    fn handle_viewport_camera_input(
        &mut self,
        ui: &egui::Ui,
        response: &egui::Response,
        viewport_size: egui::Vec2,
    ) {
        if response.clicked() {
            response.request_focus();
        }

        if response.clicked_by(egui::PointerButton::Primary)
            && self.gizmo_state.active_drag.is_none()
        {
            if let Some(entity) = self.try_pick_viewport_entity(response) {
                self.selection.select_single(entity);
            } else {
                self.selection.deselect();
            }
        }

        let pointer_delta = ui.input(|input| input.pointer.delta());
        let scroll_delta_y =
            ui.input(|input| input.raw_scroll_delta.y + input.smooth_scroll_delta.y);

        let mut camera_changed = false;

        if response.dragged_by(egui::PointerButton::Secondary) {
            self.editor_camera.orbit(pointer_delta);
            camera_changed = true;
        } else if response.dragged_by(egui::PointerButton::Middle) {
            self.editor_camera.pan(pointer_delta, viewport_size);
            camera_changed = true;
        }

        if response.hovered() && scroll_delta_y.abs() > f32::EPSILON {
            self.editor_camera.zoom(scroll_delta_y);
            camera_changed = true;
        }

        let focus_pressed =
            response.has_focus() && ui.input(|input| input.key_pressed(egui::Key::F));
        if focus_pressed {
            camera_changed = self.focus_editor_camera_on_selection() || camera_changed;
        }

        if camera_changed {
            self.apply_editor_camera_to_primary_camera(
                viewport_size.x.max(1.0) as u32,
                viewport_size.y.max(1.0) as u32,
            );
            self.sync_global_transforms_for_viewport();
            ui.ctx().request_repaint();
        }

        if response.dragged() {
            ui.ctx().request_repaint();
        }
    }

    fn draw_menu_bar(&mut self, ctx: &egui::Context) {
        let recent_files = self.config.recent_files.clone();
        let mut open_recent: Option<PathBuf> = None;

        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui: &mut egui::Ui| {
                ui.menu_button("File", |ui: &mut egui::Ui| {
                    if ui.button("New Scene          Ctrl+N").clicked() {
                        if let Err(error) = self.cmd_new_scene() {
                            self.log_message(
                                LogLevel::Error,
                                format!("New scene failed: {}", error),
                            );
                        }
                    }

                    if ui.button("Open Scene...      Ctrl+O").clicked() {
                        if let Err(error) = self.cmd_open_scene() {
                            self.log_message(
                                LogLevel::Error,
                                format!("Open scene failed: {}", error),
                            );
                        }
                    }

                    ui.menu_button("Open Recent", |ui| {
                        if recent_files.is_empty() {
                            ui.add_enabled(false, egui::Button::new("No recent files"));
                        } else {
                            for path in &recent_files {
                                if ui.button(path.display().to_string()).clicked() {
                                    open_recent = Some(path.clone());
                                    ui.close_menu();
                                }
                            }
                        }
                    });

                    ui.separator();

                    if ui
                        .add_enabled(
                            self.file_path.is_some(),
                            egui::Button::new("Save            Ctrl+S"),
                        )
                        .clicked()
                    {
                        if let Err(error) = self.cmd_save() {
                            self.log_message(LogLevel::Error, format!("Save failed: {}", error));
                        }
                    }

                    if ui.button("Save As...      Ctrl+Shift+S").clicked() {
                        if let Err(error) = self.cmd_save_as() {
                            self.log_message(LogLevel::Error, format!("Save as failed: {}", error));
                        }
                    }

                    ui.separator();

                    if ui.button("Exit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });

                ui.menu_button("Edit", |ui: &mut egui::Ui| {
                    if ui
                        .add_enabled(
                            self.command_history.can_undo(),
                            egui::Button::new("Undo    Ctrl+Z"),
                        )
                        .clicked()
                    {
                        let _ = self.cancel_active_gizmo_drag(false);
                        let selection_hint = self.command_history.undo(&mut self.world);
                        self.apply_selection_hint(selection_hint);
                        self.unsaved_changes = true;
                    }

                    if ui
                        .add_enabled(
                            self.command_history.can_redo(),
                            egui::Button::new("Redo    Ctrl+Y"),
                        )
                        .clicked()
                    {
                        let _ = self.cancel_active_gizmo_drag(false);
                        let selection_hint = self.command_history.redo(&mut self.world);
                        self.apply_selection_hint(selection_hint);
                        self.unsaved_changes = true;
                    }

                    ui.separator();
                    if ui
                        .add_enabled(
                            self.selection.has_selection(),
                            egui::Button::new("Delete    Del"),
                        )
                        .clicked()
                    {
                        self.cmd_delete_selected();
                    }

                    if ui
                        .add_enabled(
                            self.selection.has_selection(),
                            egui::Button::new("Duplicate    Ctrl+D"),
                        )
                        .clicked()
                    {
                        self.cmd_duplicate_selected();
                    }
                });

                ui.menu_button("Help", |ui: &mut egui::Ui| {
                    if ui.button("About").clicked() {
                        self.show_about = true;
                    }
                });
            });
        });

        if let Some(path) = open_recent {
            if let Err(error) = self.open_scene_from_path(&path) {
                self.log_message(LogLevel::Error, format!("Open recent failed: {}", error));
            }
        }
    }

    fn handle_keyboard_shortcuts(&mut self, ctx: &egui::Context) {
        let mut new_scene = false;
        let mut open_scene = false;
        let mut save_scene = false;
        let mut save_as_scene = false;
        let mut undo = false;
        let mut redo = false;
        let mut delete_selected = false;
        let mut duplicate_selected = false;
        let mut begin_rename = false;
        let mut clear_selection = false;

        ctx.input_mut(|input| {
            new_scene = input.consume_shortcut(&egui::KeyboardShortcut::new(
                egui::Modifiers::CTRL,
                egui::Key::N,
            ));
            open_scene = input.consume_shortcut(&egui::KeyboardShortcut::new(
                egui::Modifiers::CTRL,
                egui::Key::O,
            ));
            save_scene = input.consume_shortcut(&egui::KeyboardShortcut::new(
                egui::Modifiers::CTRL,
                egui::Key::S,
            ));
            save_as_scene = input.consume_shortcut(&egui::KeyboardShortcut::new(
                egui::Modifiers {
                    alt: false,
                    ctrl: true,
                    shift: true,
                    mac_cmd: false,
                    command: false,
                },
                egui::Key::S,
            ));

            undo = input.consume_shortcut(&egui::KeyboardShortcut::new(
                egui::Modifiers::CTRL,
                egui::Key::Z,
            ));

            redo = input.consume_shortcut(&egui::KeyboardShortcut::new(
                egui::Modifiers::CTRL,
                egui::Key::Y,
            ));

            duplicate_selected = input.consume_shortcut(&egui::KeyboardShortcut::new(
                egui::Modifiers::CTRL,
                egui::Key::D,
            ));

            delete_selected = input.key_pressed(egui::Key::Delete);
            begin_rename = input.key_pressed(egui::Key::F2);

            clear_selection = input.key_pressed(egui::Key::Escape);
        });

        if new_scene {
            if let Err(error) = self.cmd_new_scene() {
                self.log_message(LogLevel::Error, format!("New scene failed: {}", error));
            }
        }

        if open_scene {
            if let Err(error) = self.cmd_open_scene() {
                self.log_message(LogLevel::Error, format!("Open scene failed: {}", error));
            }
        }

        if save_as_scene {
            if let Err(error) = self.cmd_save_as() {
                self.log_message(LogLevel::Error, format!("Save as failed: {}", error));
            }
        } else if save_scene {
            if let Err(error) = self.cmd_save() {
                self.log_message(LogLevel::Error, format!("Save failed: {}", error));
            }
        }

        if undo {
            let _ = self.cancel_active_gizmo_drag(false);
            let selection_hint = self.command_history.undo(&mut self.world);
            self.apply_selection_hint(selection_hint);
            self.unsaved_changes = true;
        }

        if redo {
            let _ = self.cancel_active_gizmo_drag(false);
            let selection_hint = self.command_history.redo(&mut self.world);
            self.apply_selection_hint(selection_hint);
            self.unsaved_changes = true;
        }

        if duplicate_selected {
            self.cmd_duplicate_selected();
        }

        if delete_selected {
            self.cmd_delete_selected();
        }

        if begin_rename {
            if let Some(entity) = self.selection.primary() {
                self.begin_rename_entity(entity);
            }
        }

        if clear_selection {
            if self.cancel_active_gizmo_drag(true) {
                return;
            }

            if self.renaming_entity.is_some() {
                self.cancel_rename_entity();
            } else {
                self.selection.deselect();
            }
        }
    }

    fn cmd_delete_selected(&mut self) {
        let Some(entity) = self.selection.primary() else {
            return;
        };

        self.cmd_delete_entity(entity);
    }

    fn cmd_delete_entity(&mut self, entity: Entity) {
        let selection_hint = self
            .command_history
            .execute(Box::new(DeleteEntityCommand::new(entity)), &mut self.world);

        self.apply_selection_hint(selection_hint);
        self.unsaved_changes = true;
        self.log_message(LogLevel::Info, "Deleted selected entity");
    }

    fn cmd_duplicate_selected(&mut self) {
        let Some(entity) = self.selection.primary() else {
            return;
        };

        self.cmd_duplicate_entity(entity);
    }

    fn cmd_duplicate_entity(&mut self, entity: Entity) {
        let selection_hint = self.command_history.execute(
            Box::new(DuplicateEntityCommand::new(entity)),
            &mut self.world,
        );

        self.apply_selection_hint(selection_hint);
        self.unsaved_changes = true;
        self.log_message(LogLevel::Info, "Duplicated selected entity");
    }

    fn cmd_add_root_entity(&mut self) {
        let selection_hint = self
            .command_history
            .execute(Box::new(SpawnEntityCommand::new_root()), &mut self.world);

        self.apply_selection_hint(selection_hint);
        self.unsaved_changes = true;
        self.log_message(LogLevel::Info, "Added root entity");
    }

    fn cmd_add_child_entity(&mut self, parent: Entity) {
        if self.world.get_entity(parent).is_err() {
            self.log_message(
                LogLevel::Warn,
                format!(
                    "Cannot add child entity: parent {:?} does not exist",
                    parent
                ),
            );
            return;
        }

        let selection_hint = self.command_history.execute(
            Box::new(SpawnEntityCommand::new_child(parent)),
            &mut self.world,
        );

        self.apply_selection_hint(selection_hint);
        self.unsaved_changes = true;
        self.log_message(LogLevel::Info, "Added child entity");
    }

    fn cmd_reparent_entity(&mut self, entity: Entity, new_parent: Option<Entity>) {
        if self.world.get_entity(entity).is_err() {
            self.log_message(
                LogLevel::Warn,
                format!("Cannot reparent entity {:?}: entity does not exist", entity),
            );
            return;
        }

        if let Some(parent) = new_parent {
            if parent == entity {
                self.log_message(
                    LogLevel::Warn,
                    format!(
                        "Cannot reparent entity {:?}: entity cannot be parent of itself",
                        entity
                    ),
                );
                return;
            }

            if self.world.get_entity(parent).is_err() {
                self.log_message(
                    LogLevel::Warn,
                    format!(
                        "Cannot reparent entity {:?}: parent {:?} does not exist",
                        entity, parent
                    ),
                );
                return;
            }

            if self.scene_tree_is_descendant(parent, entity) {
                self.log_message(
                    LogLevel::Warn,
                    format!(
                        "Cannot reparent entity {:?}: target parent {:?} is in its descendant chain",
                        entity,
                        parent
                    ),
                );
                return;
            }
        }

        if self.world.get::<Parent>(entity).map(|parent| parent.0) == new_parent {
            return;
        }

        let selection_hint = self.command_history.execute(
            Box::new(ReparentEntityCommand::new(entity, new_parent)),
            &mut self.world,
        );

        self.apply_selection_hint(selection_hint);
        self.unsaved_changes = true;

        if let Some(parent) = new_parent {
            self.log_message(
                LogLevel::Info,
                format!("Reparented entity {:?} under {:?}", entity, parent),
            );
        } else {
            self.log_message(LogLevel::Info, format!("Moved entity {:?} to root", entity));
        }
    }

    fn scene_tree_is_descendant(&self, candidate: Entity, ancestor: Entity) -> bool {
        is_scene_tree_descendant(&self.world, candidate, ancestor)
    }

    fn begin_rename_entity(&mut self, entity: Entity) {
        let current_name = self
            .world
            .get::<EntityName>(entity)
            .map(|value| value.0.clone())
            .unwrap_or_else(|| "Entity".to_owned());

        self.renaming_entity = Some(entity);
        self.rename_buffer = current_name;
        self.rename_focus_pending = true;
    }

    fn commit_rename_entity(&mut self) {
        let Some(entity) = self.renaming_entity else {
            return;
        };

        let new_name = self.rename_buffer.trim().to_owned();
        if new_name.is_empty() {
            self.renaming_entity = None;
            self.rename_buffer.clear();
            self.rename_focus_pending = false;
            return;
        }

        let old_name = self
            .world
            .get::<EntityName>(entity)
            .map(|value| value.0.clone())
            .unwrap_or_else(|| "Entity".to_owned());

        if old_name == new_name {
            self.renaming_entity = None;
            self.rename_buffer.clear();
            self.rename_focus_pending = false;
            return;
        }

        let selection_hint = self.command_history.execute(
            Box::new(RenameEntityCommand::new(entity, old_name, new_name)),
            &mut self.world,
        );

        self.apply_selection_hint(selection_hint);
        self.unsaved_changes = true;
        self.renaming_entity = None;
        self.rename_buffer.clear();
        self.rename_focus_pending = false;
        self.log_message(LogLevel::Info, "Renamed entity");
    }

    fn cancel_rename_entity(&mut self) {
        self.renaming_entity = None;
        self.rename_buffer.clear();
        self.rename_focus_pending = false;
    }

    fn set_asset_browser_feedback(&mut self, level: LogLevel, message: impl Into<String>) {
        let message = message.into();
        self.asset_browser_ops
            .feedback
            .replace((level, message.clone()));
        self.log_message(level, message);
    }

    fn begin_asset_create_folder_dialog(&mut self) {
        self.asset_browser_ops.create_folder_input = "new_folder".to_owned();
        self.asset_browser_ops.create_folder_open = true;
    }

    fn begin_asset_create_asset_dialog(&mut self) {
        let template = self.asset_browser_ops.create_asset_template;
        self.asset_browser_ops.create_asset_input = template.default_file_name().to_owned();
        self.asset_browser_ops.create_asset_open = true;
    }

    fn commit_asset_create_folder(&mut self) {
        let folder_name = self.asset_browser_ops.create_folder_input.trim().to_owned();
        if folder_name.is_empty() {
            self.set_asset_browser_feedback(LogLevel::Warn, "Create folder aborted: empty name");
            return;
        }

        match self.asset_browser.create_directory_in_current(&folder_name) {
            Ok(relative_path) => {
                self.set_asset_browser_feedback(
                    LogLevel::Info,
                    format!("Created folder '{}'", relative_path),
                );
            }
            Err(error) => {
                self.set_asset_browser_feedback(
                    LogLevel::Warn,
                    format!("Create folder failed: {}", error),
                );
            }
        }
    }

    fn commit_asset_create_asset(&mut self) {
        let file_name = self.asset_browser_ops.create_asset_input.trim().to_owned();
        if file_name.is_empty() {
            self.set_asset_browser_feedback(LogLevel::Warn, "Create asset aborted: empty name");
            return;
        }

        let template = self.asset_browser_ops.create_asset_template;
        let contents = template.contents(&file_name);
        match self
            .asset_browser
            .create_file_in_current(&file_name, &contents)
        {
            Ok(relative_path) => {
                self.set_asset_browser_feedback(
                    LogLevel::Info,
                    format!("Created asset '{}'", relative_path),
                );
            }
            Err(error) => {
                self.set_asset_browser_feedback(
                    LogLevel::Warn,
                    format!("Create asset failed: {}", error),
                );
            }
        }
    }

    fn open_asset_conflict_dialog(
        &mut self,
        action: AssetConflictAction,
        source_label: String,
        target_label: String,
        suggested_name: Option<String>,
    ) {
        self.asset_browser_ops.conflict_state = Some(AssetConflictState {
            action,
            source_label,
            target_label,
            suggested_name,
        });
        self.asset_browser_ops.conflict_open = true;
    }

    fn apply_asset_conflict_with_overwrite(&mut self) {
        let Some(conflict_state) = self.asset_browser_ops.conflict_state.take() else {
            self.asset_browser_ops.conflict_open = false;
            return;
        };
        self.asset_browser_ops.conflict_open = false;

        let result = match conflict_state.action {
            AssetConflictAction::RenameSelected { desired_name } => self
                .asset_browser
                .rename_selected_with_options(&desired_name, true),
            AssetConflictAction::MoveSelected { target_directory } => self
                .asset_browser
                .move_selected_with_options(target_directory.as_deref(), None, true),
            AssetConflictAction::MoveRelativePath {
                source_relative_path,
                target_directory,
            } => self.asset_browser.move_relative_path_with_options(
                &source_relative_path,
                target_directory.as_deref(),
                None,
                true,
            ),
        };

        match result {
            Ok(Some(relative_path)) => {
                self.set_asset_browser_feedback(
                    LogLevel::Info,
                    format!(
                        "Asset operation completed with overwrite: '{}'",
                        relative_path
                    ),
                );
            }
            Ok(None) => {
                self.set_asset_browser_feedback(
                    LogLevel::Warn,
                    "Asset operation failed: no selected source",
                );
            }
            Err(error) => {
                self.set_asset_browser_feedback(
                    LogLevel::Warn,
                    format!("Asset overwrite failed: {}", error),
                );
            }
        }
    }

    fn apply_asset_conflict_with_suggested_name(&mut self) {
        let Some(conflict_state) = self.asset_browser_ops.conflict_state.take() else {
            self.asset_browser_ops.conflict_open = false;
            return;
        };
        self.asset_browser_ops.conflict_open = false;

        let Some(suggested_name) = conflict_state.suggested_name else {
            self.set_asset_browser_feedback(
                LogLevel::Warn,
                "No suggested name available for conflict resolution",
            );
            return;
        };

        let result = match conflict_state.action {
            AssetConflictAction::RenameSelected { .. } => self
                .asset_browser
                .rename_selected_with_options(&suggested_name, false),
            AssetConflictAction::MoveSelected { target_directory } => {
                self.asset_browser.move_selected_with_options(
                    target_directory.as_deref(),
                    Some(&suggested_name),
                    false,
                )
            }
            AssetConflictAction::MoveRelativePath {
                source_relative_path,
                target_directory,
            } => self.asset_browser.move_relative_path_with_options(
                &source_relative_path,
                target_directory.as_deref(),
                Some(&suggested_name),
                false,
            ),
        };

        match result {
            Ok(Some(relative_path)) => {
                self.set_asset_browser_feedback(
                    LogLevel::Info,
                    format!(
                        "Asset operation completed with suggested name '{}'",
                        relative_path
                    ),
                );
            }
            Ok(None) => {
                self.set_asset_browser_feedback(
                    LogLevel::Warn,
                    "Asset operation failed: no selected source",
                );
            }
            Err(error) => {
                self.set_asset_browser_feedback(
                    LogLevel::Warn,
                    format!("Asset rename suggestion failed: {}", error),
                );
            }
        }
    }

    fn begin_asset_rename_dialog(&mut self) {
        let Some(file_name) = self.asset_browser.selected_file_name() else {
            self.set_asset_browser_feedback(
                LogLevel::Warn,
                "Select an asset before opening rename",
            );
            return;
        };

        self.asset_browser_ops.rename_input = file_name;
        self.asset_browser_ops.rename_open = true;
    }

    fn begin_asset_move_dialog(&mut self) {
        if self.asset_browser.selected_relative_path().is_none() {
            self.set_asset_browser_feedback(LogLevel::Warn, "Select an asset before opening move");
            return;
        }

        self.asset_browser_ops.move_input = self
            .asset_browser
            .current_relative_path()
            .unwrap_or_default();
        self.asset_browser_ops.move_open = true;
    }

    fn begin_asset_delete_dialog(&mut self) {
        if self.asset_browser.selected_relative_path().is_none() {
            self.set_asset_browser_feedback(
                LogLevel::Warn,
                "Select an asset before opening delete",
            );
            return;
        }

        self.asset_browser_ops.delete_open = true;
    }

    fn commit_asset_rename(&mut self) {
        let new_name = self.asset_browser_ops.rename_input.trim().to_owned();
        if new_name.is_empty() {
            self.set_asset_browser_feedback(LogLevel::Warn, "Rename aborted: empty name");
            return;
        }

        match self.asset_browser.rename_selected(&new_name) {
            Ok(Some(relative_path)) => {
                self.set_asset_browser_feedback(
                    LogLevel::Info,
                    format!("Renamed asset to '{}'", relative_path),
                );
            }
            Ok(None) => {
                self.set_asset_browser_feedback(LogLevel::Warn, "Rename failed: no selected asset");
            }
            Err(error) => {
                if error.kind() == ErrorKind::AlreadyExists {
                    let source_label = self
                        .asset_browser
                        .selected_relative_path()
                        .unwrap_or("<unknown>")
                        .to_owned();
                    let desired_name = new_name.clone();
                    let suggested_name = self
                        .asset_browser
                        .suggest_rename_selected_name(&new_name)
                        .ok()
                        .flatten();

                    self.open_asset_conflict_dialog(
                        AssetConflictAction::RenameSelected {
                            desired_name: desired_name.clone(),
                        },
                        source_label,
                        desired_name,
                        suggested_name,
                    );
                } else {
                    self.set_asset_browser_feedback(
                        LogLevel::Warn,
                        format!("Rename failed: {}", error),
                    );
                }
            }
        }
    }

    fn commit_asset_move(&mut self) {
        let move_target = self.asset_browser_ops.move_input.trim().to_owned();
        let move_target_ref = if move_target.is_empty() {
            None
        } else {
            Some(move_target.as_str())
        };

        match self.asset_browser.move_selected(move_target_ref) {
            Ok(Some(relative_path)) => {
                self.set_asset_browser_feedback(
                    LogLevel::Info,
                    format!("Moved asset to '{}'", relative_path),
                );
            }
            Ok(None) => {
                self.set_asset_browser_feedback(LogLevel::Warn, "Move failed: no selected asset");
            }
            Err(error) => {
                if error.kind() == ErrorKind::AlreadyExists {
                    let source_label = self
                        .asset_browser
                        .selected_relative_path()
                        .unwrap_or("<unknown>")
                        .to_owned();
                    let target_directory = if move_target.is_empty() {
                        None
                    } else {
                        Some(move_target.clone())
                    };
                    let target_label = target_directory
                        .clone()
                        .unwrap_or_else(|| "assets root".to_owned());
                    let suggested_name = self
                        .asset_browser
                        .suggest_move_selected_name(target_directory.as_deref())
                        .ok()
                        .flatten();

                    self.open_asset_conflict_dialog(
                        AssetConflictAction::MoveSelected { target_directory },
                        source_label,
                        target_label,
                        suggested_name,
                    );
                } else {
                    self.set_asset_browser_feedback(
                        LogLevel::Warn,
                        format!("Move failed: {}", error),
                    );
                }
            }
        }
    }

    fn commit_asset_drop_move(
        &mut self,
        source_relative_path: &str,
        target_directory_relative: Option<&str>,
    ) {
        match self
            .asset_browser
            .move_relative_path(source_relative_path, target_directory_relative)
        {
            Ok(Some(relative_path)) => {
                if relative_path == source_relative_path {
                    self.set_asset_browser_feedback(
                        LogLevel::Info,
                        "Asset is already in the target folder",
                    );
                } else {
                    self.set_asset_browser_feedback(
                        LogLevel::Info,
                        format!("Moved asset to '{}'", relative_path),
                    );
                }
            }
            Ok(None) => {
                self.set_asset_browser_feedback(
                    LogLevel::Warn,
                    "Move failed: invalid source asset",
                );
            }
            Err(error) => {
                if error.kind() == ErrorKind::AlreadyExists {
                    let target_directory = target_directory_relative.map(str::to_owned);
                    let target_label = target_directory
                        .clone()
                        .unwrap_or_else(|| "assets root".to_owned());
                    let suggested_name = self
                        .asset_browser
                        .suggest_move_relative_name(source_relative_path, target_directory_relative)
                        .ok()
                        .flatten();

                    self.open_asset_conflict_dialog(
                        AssetConflictAction::MoveRelativePath {
                            source_relative_path: source_relative_path.to_owned(),
                            target_directory,
                        },
                        source_relative_path.to_owned(),
                        target_label,
                        suggested_name,
                    );
                } else {
                    self.set_asset_browser_feedback(
                        LogLevel::Warn,
                        format!("Move failed: {}", error),
                    );
                }
            }
        }
    }

    fn commit_asset_delete(&mut self) {
        let deleted_relative_path = self
            .asset_browser
            .selected_relative_path()
            .map(str::to_owned);

        match self.asset_browser.delete_selected() {
            Ok(true) => {
                let deleted_label = deleted_relative_path.unwrap_or_else(|| "<unknown>".to_owned());
                self.set_asset_browser_feedback(
                    LogLevel::Info,
                    format!("Deleted asset '{}'", deleted_label),
                );
            }
            Ok(false) => {
                self.set_asset_browser_feedback(LogLevel::Warn, "Delete failed: no selected asset");
            }
            Err(error) => {
                self.set_asset_browser_feedback(
                    LogLevel::Warn,
                    format!("Delete failed: {}", error),
                );
            }
        }
    }

    fn draw_asset_browser_operation_modals(&mut self, ctx: &egui::Context) {
        if self.asset_browser_ops.create_folder_open {
            let mut open = self.asset_browser_ops.create_folder_open;
            let mut commit = false;
            let mut cancel = false;

            egui::Window::new("Create Folder")
                .open(&mut open)
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    if let Some(current_relative) = self.asset_browser.current_relative_path() {
                        ui.label(format!("Current folder: {}", current_relative));
                    } else {
                        ui.label("Current folder: assets root");
                    }

                    ui.separator();
                    ui.label("Folder name:");
                    let response =
                        ui.text_edit_singleline(&mut self.asset_browser_ops.create_folder_input);

                    if response.lost_focus()
                        && ui.input(|input| input.key_pressed(egui::Key::Enter))
                    {
                        commit = true;
                    }

                    ui.horizontal(|ui| {
                        if ui.button("Cancel").clicked() {
                            cancel = true;
                        }

                        if ui
                            .add_enabled(
                                !self.asset_browser_ops.create_folder_input.trim().is_empty(),
                                egui::Button::new("Create"),
                            )
                            .clicked()
                        {
                            commit = true;
                        }
                    });
                });

            if cancel {
                open = false;
            }

            if commit {
                open = false;
                self.commit_asset_create_folder();
            }

            self.asset_browser_ops.create_folder_open = open;
        }

        if self.asset_browser_ops.create_asset_open {
            let mut open = self.asset_browser_ops.create_asset_open;
            let mut commit = false;
            let mut cancel = false;

            egui::Window::new("Create Asset")
                .open(&mut open)
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    if let Some(current_relative) = self.asset_browser.current_relative_path() {
                        ui.label(format!("Current folder: {}", current_relative));
                    } else {
                        ui.label("Current folder: assets root");
                    }

                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label("Template:");
                        egui::ComboBox::from_id_salt("asset_template_combo")
                            .selected_text(self.asset_browser_ops.create_asset_template.label())
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut self.asset_browser_ops.create_asset_template,
                                    NewAssetTemplate::Scene,
                                    NewAssetTemplate::Scene.label(),
                                );
                                ui.selectable_value(
                                    &mut self.asset_browser_ops.create_asset_template,
                                    NewAssetTemplate::Material,
                                    NewAssetTemplate::Material.label(),
                                );
                                ui.selectable_value(
                                    &mut self.asset_browser_ops.create_asset_template,
                                    NewAssetTemplate::Text,
                                    NewAssetTemplate::Text.label(),
                                );
                            });
                    });

                    ui.label("File name:");
                    let response =
                        ui.text_edit_singleline(&mut self.asset_browser_ops.create_asset_input);
                    if response.lost_focus()
                        && ui.input(|input| input.key_pressed(egui::Key::Enter))
                    {
                        commit = true;
                    }

                    if ui.button("Use template default name").clicked() {
                        self.asset_browser_ops.create_asset_input = self
                            .asset_browser_ops
                            .create_asset_template
                            .default_file_name()
                            .to_owned();
                    }

                    ui.horizontal(|ui| {
                        if ui.button("Cancel").clicked() {
                            cancel = true;
                        }

                        if ui
                            .add_enabled(
                                !self.asset_browser_ops.create_asset_input.trim().is_empty(),
                                egui::Button::new("Create"),
                            )
                            .clicked()
                        {
                            commit = true;
                        }
                    });
                });

            if cancel {
                open = false;
            }

            if commit {
                open = false;
                self.commit_asset_create_asset();
            }

            self.asset_browser_ops.create_asset_open = open;
        }

        if self.asset_browser_ops.rename_open {
            let mut open = self.asset_browser_ops.rename_open;
            let mut commit = false;
            let mut cancel = false;

            egui::Window::new("Rename Asset")
                .open(&mut open)
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    if let Some(selected_path) = self.asset_browser.selected_relative_path() {
                        ui.label(format!("Selected: {}", selected_path));
                    }

                    ui.separator();
                    ui.label("New name:");
                    let response =
                        ui.text_edit_singleline(&mut self.asset_browser_ops.rename_input);
                    if response.lost_focus()
                        && ui.input(|input| input.key_pressed(egui::Key::Enter))
                    {
                        commit = true;
                    }

                    ui.horizontal(|ui| {
                        if ui.button("Cancel").clicked() {
                            cancel = true;
                        }

                        if ui
                            .add_enabled(
                                !self.asset_browser_ops.rename_input.trim().is_empty(),
                                egui::Button::new("Rename"),
                            )
                            .clicked()
                        {
                            commit = true;
                        }
                    });
                });

            if cancel {
                open = false;
            }

            if commit {
                open = false;
                self.commit_asset_rename();
            }

            self.asset_browser_ops.rename_open = open;
        }

        if self.asset_browser_ops.move_open {
            let mut open = self.asset_browser_ops.move_open;
            let mut commit = false;
            let mut cancel = false;

            egui::Window::new("Move Asset")
                .open(&mut open)
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    if let Some(selected_path) = self.asset_browser.selected_relative_path() {
                        ui.label(format!("Selected: {}", selected_path));
                    }

                    ui.separator();
                    ui.label("Target directory (relative to assets root):");
                    ui.label("Leave empty to move to assets root");
                    let response = ui.text_edit_singleline(&mut self.asset_browser_ops.move_input);
                    if response.lost_focus()
                        && ui.input(|input| input.key_pressed(egui::Key::Enter))
                    {
                        commit = true;
                    }

                    ui.horizontal(|ui| {
                        if ui.button("Cancel").clicked() {
                            cancel = true;
                        }

                        if ui.button("Move").clicked() {
                            commit = true;
                        }
                    });
                });

            if cancel {
                open = false;
            }

            if commit {
                open = false;
                self.commit_asset_move();
            }

            self.asset_browser_ops.move_open = open;
        }

        if self.asset_browser_ops.delete_open {
            let mut open = self.asset_browser_ops.delete_open;
            let mut confirm = false;
            let mut cancel = false;

            egui::Window::new("Delete Asset")
                .open(&mut open)
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    if let Some(selected_path) = self.asset_browser.selected_relative_path() {
                        ui.label(format!("Delete '{}' ?", selected_path));
                    } else {
                        ui.label("Delete selected asset?");
                    }

                    ui.colored_label(
                        egui::Color32::YELLOW,
                        "This operation removes files from disk and cannot be undone by editor history.",
                    );

                    ui.horizontal(|ui| {
                        if ui.button("Cancel").clicked() {
                            cancel = true;
                        }

                        if ui
                            .add(egui::Button::new("Delete").fill(egui::Color32::from_rgb(120, 35, 35)))
                            .clicked()
                        {
                            confirm = true;
                        }
                    });
                });

            if cancel {
                open = false;
            }

            if confirm {
                open = false;
                self.commit_asset_delete();
            }

            self.asset_browser_ops.delete_open = open;
        }

        if self.asset_browser_ops.conflict_open {
            let mut open = self.asset_browser_ops.conflict_open;
            let mut cancel = false;
            let mut overwrite = false;
            let mut use_suggested_name = false;

            egui::Window::new("Asset Conflict")
                .open(&mut open)
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    let (source_label, target_label, suggested_name) = if let Some(conflict_state) =
                        self.asset_browser_ops.conflict_state.as_ref()
                    {
                        (
                            conflict_state.source_label.clone(),
                            conflict_state.target_label.clone(),
                            conflict_state.suggested_name.clone(),
                        )
                    } else {
                        ("<unknown>".to_owned(), "<unknown>".to_owned(), None)
                    };

                    ui.label(format!(
                        "'{}' conflicts with an existing target in '{}'.",
                        source_label, target_label
                    ));
                    ui.label("Choose how to continue:");

                    ui.horizontal(|ui| {
                        if ui.button("Cancel").clicked() {
                            cancel = true;
                        }

                        if ui
                            .add(egui::Button::new("Overwrite existing target"))
                            .clicked()
                        {
                            overwrite = true;
                        }

                        if ui
                            .add_enabled(
                                suggested_name.is_some(),
                                egui::Button::new("Use suggested name"),
                            )
                            .clicked()
                        {
                            use_suggested_name = true;
                        }
                    });

                    if let Some(suggested) = suggested_name {
                        ui.separator();
                        ui.label(format!("Suggested name: {}", suggested));
                    }
                });

            if cancel {
                open = false;
                self.asset_browser_ops.conflict_state = None;
            }

            if overwrite {
                open = false;
                self.apply_asset_conflict_with_overwrite();
            }

            if use_suggested_name {
                open = false;
                self.apply_asset_conflict_with_suggested_name();
            }

            if !open {
                self.asset_browser_ops.conflict_state = None;
            }

            self.asset_browser_ops.conflict_open = open;
        }
    }

    fn draw_status_bar(&self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("status_bar")
            .exact_height(22.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let entity_count = self.world.entities().len();
                    ui.label(format!("Entities: {}", entity_count));
                    ui.separator();

                    let fps = 1.0 / ctx.input(|i| i.predicted_dt.max(0.0001));
                    ui.label(format!("FPS: {:.0}", fps));
                    ui.separator();

                    if let Some(path) = &self.file_path {
                        let name = path
                            .file_name()
                            .and_then(|name| name.to_str())
                            .unwrap_or("untitled");

                        if self.unsaved_changes {
                            ui.label(format!("* {}", name));
                        } else {
                            ui.label(name);
                        }
                    } else {
                        ui.label("untitled");
                    }
                });
            });
    }

    fn draw_modals(&mut self, ctx: &egui::Context) {
        if self.show_about {
            let mut open = true;
            egui::Window::new("About Starman Editor")
                .open(&mut open)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label("Starman Editor (EP-09 foundation)");
                    ui.label("Rust + egui + wgpu");
                });

            self.show_about = open;
        }

        self.draw_asset_browser_operation_modals(ctx);
    }

    fn cmd_new_scene(&mut self) -> Result<()> {
        self.world.clear_entities();
        self.world.spawn(EditorEntityBundle::default());
        self.bootstrap_viewport_scene();
        self.file_path = None;
        self.selection.deselect();
        self.cancel_rename_entity();
        self.scene_filter_query.clear();
        self.dragging_entity = None;
        self.gizmo_state.active_drag = None;
        self.command_history.clear();
        self.unsaved_changes = false;
        self.persist_config();
        self.log_message(LogLevel::Info, "Created new scene");
        Ok(())
    }

    fn cmd_open_scene(&mut self) -> Result<()> {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Scene RON", &["ron"])
            .pick_file()
        else {
            return Ok(());
        };

        self.open_scene_from_path(&path)
    }

    fn open_scene_from_path(&mut self, path: &Path) -> Result<()> {
        self.load_scene(path)?;
        self.file_path = Some(path.to_path_buf());
        self.config.touch_recent_file(path.to_path_buf());
        self.unsaved_changes = false;
        self.persist_config();
        self.log_message(LogLevel::Info, format!("Opened scene {}", path.display()));
        Ok(())
    }

    fn cmd_save(&mut self) -> Result<()> {
        if let Some(path) = self.file_path.clone() {
            self.save_scene(&path)?;
            self.config.touch_recent_file(path.clone());
            self.unsaved_changes = false;
            self.persist_config();
            self.log_message(LogLevel::Info, format!("Saved scene {}", path.display()));
            return Ok(());
        }

        self.cmd_save_as()
    }

    fn cmd_save_as(&mut self) -> Result<()> {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Scene RON", &["ron"])
            .set_file_name("untitled.scene.ron")
            .save_file()
        else {
            return Ok(());
        };

        self.save_scene(&path)?;
        self.file_path = Some(path.clone());
        self.config.touch_recent_file(path.clone());
        self.unsaved_changes = false;
        self.persist_config();
        self.log_message(LogLevel::Info, format!("Saved scene {}", path.display()));

        Ok(())
    }

    fn save_scene(&mut self, path: &Path) -> Result<()> {
        let scene_name = path
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or("Untitled");

        let render_scene_adapter = RenderSceneAdapter;

        self.with_scene_context(
            |world, component_registry, type_registry, metadata_registry, asset_server| {
                let serializer = SceneSerializer::new(world, component_registry, type_registry)
                    .with_metadata_registry(metadata_registry)
                    .with_asset_server(asset_server)
                    .with_external_components(&render_scene_adapter);
                serializer.save_file(path, scene_name)?;
                Ok(())
            },
        )
    }

    fn load_scene(&mut self, path: &Path) -> Result<()> {
        let render_scene_adapter = RenderSceneAdapter;

        self.with_scene_context(
            |world, component_registry, type_registry, _metadata_registry, asset_server| {
                world.clear_entities();
                let mut deserializer =
                    SceneDeserializer::new(world, component_registry, type_registry, asset_server)
                        .with_external_components(&render_scene_adapter);
                let _ = deserializer.load_file(path)?;
                Ok(())
            },
        )?;

        self.bootstrap_viewport_scene();
        self.selection.deselect();
        self.cancel_rename_entity();
        self.scene_filter_query.clear();
        self.dragging_entity = None;
        self.gizmo_state.active_drag = None;
        self.command_history.clear();

        Ok(())
    }

    fn with_scene_context<R>(
        &mut self,
        f: impl FnOnce(
            &mut World,
            &ComponentRegistry,
            &ReflectTypeRegistry,
            &ReflectMetadataRegistry,
            &mut AssetServer,
        ) -> Result<R>,
    ) -> Result<R> {
        let type_registry = self
            .world
            .remove_resource::<ReflectTypeRegistry>()
            .unwrap_or_default();
        let component_registry = self
            .world
            .remove_resource::<ComponentRegistry>()
            .unwrap_or_default();
        let metadata_registry = self
            .world
            .remove_resource::<ReflectMetadataRegistry>()
            .unwrap_or_default();

        let result = f(
            &mut self.world,
            &component_registry,
            &type_registry,
            &metadata_registry,
            &mut self.asset_server,
        );

        self.world.insert_resource(type_registry);
        self.world.insert_resource(component_registry);
        self.world.insert_resource(metadata_registry);

        result
    }

    fn show_viewport_panel(&mut self, ui: &mut egui::Ui) {
        let Some(render_state) = self.wgpu_render_state.clone() else {
            ui.vertical_centered_justified(|ui| {
                ui.label("WGPU renderer state unavailable in this runtime.");
            });
            return;
        };

        let available = ui.available_size();
        let width = available.x.max(1.0) as u32;
        let height = available.y.max(1.0) as u32;

        self.ensure_viewport_world_defaults();
        self.apply_editor_camera_to_primary_camera(width, height);
        self.sync_global_transforms_for_viewport();

        self.viewport_renderer
            .ensure_size(&render_state, width, height);
        self.viewport_renderer
            .render(&render_state, &mut self.world, &self.asset_server);

        let mut viewport_response = None;
        if let Some(texture_id) = self.viewport_renderer.texture_id() {
            let sized_texture =
                egui::load::SizedTexture::new(texture_id, egui::vec2(width as f32, height as f32));
            let response =
                ui.add(egui::Image::new(sized_texture).sense(egui::Sense::click_and_drag()));
            viewport_response = Some(response);
        }

        if let Some(response) = viewport_response.as_ref() {
            self.handle_viewport_asset_drop(ui, response);
            self.draw_transform_gizmo(ui, response);
            self.handle_viewport_camera_input(
                ui,
                response,
                egui::vec2(width as f32, height as f32),
            );
            self.draw_viewport_overlay(ui, response.rect);
        }

        if let Some(error) = self.viewport_renderer.last_error() {
            ui.colored_label(
                egui::Color32::RED,
                format!("Viewport render error: {}", error),
            );
        } else {
            ui.label(
                "Viewport controls: RMB orbit | MMB pan | Wheel zoom | F focus | W/E/R tool | C cycle tool | Q orientation | H/V/Z/N manual axis lock | X snap | Hold Shift auto axis lock | G/B/L overlay",
            );
        }
    }

    fn handle_viewport_asset_drop(&mut self, ui: &egui::Ui, response: &egui::Response) {
        if let Some(payload) = response.dnd_hover_payload::<AssetDragPayload>() {
            if payload.kind == AssetKind::Mesh {
                let highlight_color = ui.visuals().selection.bg_fill.gamma_multiply(0.25);
                ui.painter()
                    .rect_filled(response.rect, 6.0, highlight_color);
                ui.painter().text(
                    response.rect.center_top() + egui::vec2(0.0, 12.0),
                    egui::Align2::CENTER_TOP,
                    "Drop mesh to spawn in viewport",
                    egui::FontId::proportional(13.0),
                    ui.visuals().strong_text_color(),
                );
            }
        }

        let Some(payload) = response.dnd_release_payload::<AssetDragPayload>() else {
            return;
        };

        if payload.kind != AssetKind::Mesh {
            self.log_message(
                LogLevel::Info,
                "Viewport accepts mesh assets only (drop textures on Inspector)",
            );
            return;
        }

        let pointer_position = response
            .interact_pointer_pos()
            .or_else(|| ui.ctx().input(|input| input.pointer.hover_pos()))
            .unwrap_or(response.rect.center());

        let mut drop_position = viewport_drop_position(
            pointer_position,
            response.rect,
            self.editor_camera.eye_position(),
            self.editor_camera.rotation(),
            self.editor_camera.fov_y_radians,
            self.editor_camera.target,
        );
        drop_position.y += 0.5;

        self.spawn_mesh_asset_from_drop(&payload.relative_path, drop_position);
    }

    fn spawn_mesh_asset_from_drop(&mut self, mesh_relative_path: &str, drop_position: Vec3) {
        let mesh = match self.asset_server.load_mesh_handle(mesh_relative_path) {
            Ok(handle) => handle,
            Err(error) => {
                self.log_message(
                    LogLevel::Warn,
                    format!(
                        "Failed to load mesh '{}' from drop: {}",
                        mesh_relative_path, error
                    ),
                );
                return;
            }
        };

        let texture = match self
            .asset_server
            .load_texture_handle(DEFAULT_DROP_TEXTURE_PATH)
        {
            Ok(handle) => handle,
            Err(error) => {
                self.log_message(
                    LogLevel::Warn,
                    format!(
                        "Failed to load default drop texture '{}': {}",
                        DEFAULT_DROP_TEXTURE_PATH, error
                    ),
                );
                return;
            }
        };

        let material = match self
            .asset_server
            .load_material_handle(DEFAULT_DROP_MATERIAL_PATH)
        {
            Ok(handle) => handle,
            Err(error) => {
                self.log_message(
                    LogLevel::Warn,
                    format!(
                        "Failed to load default drop material '{}': {}",
                        DEFAULT_DROP_MATERIAL_PATH, error
                    ),
                );
                return;
            }
        };

        let entity_name = asset_drop_entity_name(mesh_relative_path);
        let transform = Transform {
            translation: drop_position,
            ..Transform::default()
        };

        let selection_hint = self.command_history.execute(
            Box::new(SpawnMeshAssetDropCommand::new(
                entity_name.clone(),
                transform,
                mesh,
                texture,
                material,
            )),
            &mut self.world,
        );

        self.apply_selection_hint(selection_hint);
        self.unsaved_changes = true;
        self.sync_global_transforms_for_viewport();
        self.log_message(
            LogLevel::Info,
            format!("Spawned mesh '{}' from asset drop", entity_name),
        );
    }

    fn handle_inspector_asset_drop(&mut self, ui: &egui::Ui, panel_response: &egui::Response) {
        let selected_entity = self.selection.primary();
        let selected_supports_texture = selected_entity.is_some_and(|entity| {
            self.world.get::<SpriteRenderable2d>(entity).is_some()
                || self.world.get::<MeshRenderable3d>(entity).is_some()
        });

        if let Some(payload) = panel_response.dnd_hover_payload::<AssetDragPayload>() {
            if payload.kind == AssetKind::Texture && selected_supports_texture {
                let highlight_color = ui.visuals().selection.bg_fill.gamma_multiply(0.20);
                ui.painter()
                    .rect_filled(panel_response.rect, 6.0, highlight_color);
                ui.painter().text(
                    panel_response.rect.right_top() + egui::vec2(-8.0, 8.0),
                    egui::Align2::RIGHT_TOP,
                    "Drop texture to apply",
                    egui::FontId::proportional(12.0),
                    ui.visuals().strong_text_color(),
                );
            }
        }

        let Some(payload) = panel_response.dnd_release_payload::<AssetDragPayload>() else {
            return;
        };

        if payload.kind != AssetKind::Texture {
            self.log_message(
                LogLevel::Info,
                "Inspector texture drop supports texture assets only",
            );
            return;
        }

        self.apply_texture_asset_to_selected_entity(&payload.relative_path);
    }

    fn apply_texture_asset_to_selected_entity(&mut self, texture_relative_path: &str) {
        let Some(entity) = self.selection.primary() else {
            self.log_message(
                LogLevel::Warn,
                "No selected entity to receive dropped texture",
            );
            return;
        };

        if self.world.get_entity(entity).is_err() {
            self.log_message(
                LogLevel::Warn,
                "Selected entity no longer exists for texture drop",
            );
            return;
        }

        let new_texture = match self.asset_server.load_texture_handle(texture_relative_path) {
            Ok(handle) => handle,
            Err(error) => {
                self.log_message(
                    LogLevel::Warn,
                    format!(
                        "Failed to load dropped texture '{}': {}",
                        texture_relative_path, error
                    ),
                );
                return;
            }
        };

        let sprite_old = self
            .world
            .get::<SpriteRenderable2d>(entity)
            .map(|value| value.texture);
        let mesh_old = self
            .world
            .get::<MeshRenderable3d>(entity)
            .map(|value| value.texture);

        let mut applied = 0;

        if let Some(old_texture) = sprite_old {
            if !texture_handle_eq(old_texture, new_texture) {
                let _ = self.command_history.execute(
                    Box::new(ReplaceRenderableTextureCommand::new(
                        entity,
                        TextureDropTarget::Sprite2d,
                        old_texture,
                        new_texture,
                    )),
                    &mut self.world,
                );
                applied += 1;
            }
        }

        if let Some(old_texture) = mesh_old {
            if !texture_handle_eq(old_texture, new_texture) {
                let _ = self.command_history.execute(
                    Box::new(ReplaceRenderableTextureCommand::new(
                        entity,
                        TextureDropTarget::Mesh3d,
                        old_texture,
                        new_texture,
                    )),
                    &mut self.world,
                );
                applied += 1;
            }
        }

        if applied == 0 {
            if sprite_old.is_none() && mesh_old.is_none() {
                self.log_message(
                    LogLevel::Warn,
                    "Selected entity does not have SpriteRenderable2d or MeshRenderable3d",
                );
            } else {
                self.log_message(
                    LogLevel::Info,
                    "Dropped texture is already applied to selected render component(s)",
                );
            }
            return;
        }

        self.selection.select_single(entity);
        self.unsaved_changes = true;
        self.log_message(
            LogLevel::Info,
            format!(
                "Applied texture '{}' to {} render component(s)",
                texture_relative_path, applied
            ),
        );
    }

    fn show_scene_tree_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Scene Hierarchy");
        ui.separator();

        let mut pending_action = None;

        ui.horizontal(|ui| {
            if ui.button("+").on_hover_text("Add Entity").clicked() {
                pending_action = Some(SceneTreeAction::AddRootEntity);
            }

            ui.separator();
            ui.label("Search:");
            let _ = ui.add(
                egui::TextEdit::singleline(&mut self.scene_filter_query)
                    .desired_width(200.0)
                    .hint_text("Filter entities..."),
            );
        });

        ui.separator();

        let roots = collect_scene_tree_roots(&self.world);

        let filter_query = self.scene_filter_query.trim().to_ascii_lowercase();
        let visibility = if filter_query.is_empty() {
            None
        } else {
            Some(self.build_scene_tree_visibility(&roots, &filter_query))
        };

        let mut root_drop_hovered = false;

        egui::ScrollArea::vertical().show(ui, |ui| {
            let mut visited = HashSet::new();
            for entity in roots.iter().copied() {
                self.draw_entity_node(
                    ui,
                    entity,
                    0,
                    visibility.as_ref(),
                    &mut visited,
                    &mut pending_action,
                );
            }

            let available = ui.available_size_before_wrap();
            let drop_area = ui.allocate_response(
                egui::vec2(available.x.max(0.0), available.y.max(24.0)),
                egui::Sense::hover(),
            );
            root_drop_hovered = drop_area.hovered();

            if self.dragging_entity.is_some() && root_drop_hovered {
                let color = ui.visuals().selection.bg_fill.gamma_multiply(0.20);
                ui.painter().rect_filled(drop_area.rect, 4.0, color);
                ui.painter().text(
                    drop_area.rect.center(),
                    egui::Align2::CENTER_CENTER,
                    "Drop here to move to root",
                    egui::FontId::default(),
                    ui.visuals().strong_text_color(),
                );
            }
        });

        let pointer_released = ui.input(|input| input.pointer.any_released());
        if pointer_released {
            if let Some(dragging_entity) = self.dragging_entity {
                let node_drop_already_selected =
                    matches!(pending_action, Some(SceneTreeAction::Reparent { .. }));
                if !node_drop_already_selected && root_drop_hovered {
                    pending_action = Some(SceneTreeAction::Reparent {
                        entity: dragging_entity,
                        new_parent: None,
                    });
                }

                self.dragging_entity = None;
            }
        }

        if let Some(action) = pending_action {
            match action {
                SceneTreeAction::AddRootEntity => {
                    self.cmd_add_root_entity();
                }
                SceneTreeAction::AddChildEntity(parent) => {
                    self.cmd_add_child_entity(parent);
                }
                SceneTreeAction::Reparent { entity, new_parent } => {
                    self.cmd_reparent_entity(entity, new_parent);
                }
                SceneTreeAction::BeginRename(entity) => {
                    self.begin_rename_entity(entity);
                }
                SceneTreeAction::CommitRename => {
                    self.commit_rename_entity();
                }
                SceneTreeAction::CancelRename => {
                    self.cancel_rename_entity();
                }
                SceneTreeAction::Duplicate(entity) => {
                    self.selection.select_single(entity);
                    self.cmd_duplicate_entity(entity);
                }
                SceneTreeAction::Delete(entity) => {
                    self.selection.select_single(entity);
                    self.cmd_delete_entity(entity);
                }
            }
        }
    }

    fn draw_entity_node(
        &mut self,
        ui: &mut egui::Ui,
        entity: Entity,
        depth: usize,
        visibility: Option<&HashMap<Entity, bool>>,
        visited: &mut HashSet<Entity>,
        pending_action: &mut Option<SceneTreeAction>,
    ) {
        if let Some(visibility) = visibility {
            if !visibility.get(&entity).copied().unwrap_or(false) {
                return;
            }
        }

        let indent = depth as f32 * 14.0;

        let name = self.entity_name_for_scene_tree(entity);

        if !visited.insert(entity) {
            ui.horizontal(|ui| {
                ui.add_space(indent);
                ui.colored_label(egui::Color32::YELLOW, format!("{} (cycle)", name));
            });
            return;
        }

        let is_selected = self.selection.primary() == Some(entity);
        ui.horizontal(|ui| {
            ui.add_space(indent);

            if self.renaming_entity == Some(entity) {
                let response = ui
                    .add(egui::TextEdit::singleline(&mut self.rename_buffer).desired_width(180.0));

                if self.rename_focus_pending {
                    response.request_focus();
                    self.rename_focus_pending = false;
                }

                if response.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter)) {
                    *pending_action = Some(SceneTreeAction::CommitRename);
                }

                if ui.input(|input| input.key_pressed(egui::Key::Escape)) {
                    *pending_action = Some(SceneTreeAction::CancelRename);
                }

                return;
            }

            let response = ui.selectable_label(is_selected, name);

            if response.clicked() {
                self.selection.select_single(entity);
            }

            if response.drag_started() {
                self.dragging_entity = Some(entity);
            }

            if let Some(dragging_entity) = self.dragging_entity {
                let pointer_released = ui.input(|input| input.pointer.any_released());
                let is_drop_target = dragging_entity != entity && response.hovered();

                if is_drop_target {
                    let color = ui.visuals().selection.bg_fill.gamma_multiply(0.20);
                    ui.painter()
                        .rect_filled(response.rect.expand(1.0), 2.0, color);
                }

                if pointer_released && is_drop_target {
                    *pending_action = Some(SceneTreeAction::Reparent {
                        entity: dragging_entity,
                        new_parent: Some(entity),
                    });
                }
            }

            if response.double_clicked() {
                *pending_action = Some(SceneTreeAction::BeginRename(entity));
            }

            response.context_menu(|ui| {
                if ui.button("Add Child Entity").clicked() {
                    *pending_action = Some(SceneTreeAction::AddChildEntity(entity));
                    ui.close_menu();
                }

                ui.separator();

                if ui.button("Rename").clicked() {
                    *pending_action = Some(SceneTreeAction::BeginRename(entity));
                    ui.close_menu();
                }

                if ui.button("Duplicate").clicked() {
                    *pending_action = Some(SceneTreeAction::Duplicate(entity));
                    ui.close_menu();
                }

                if ui.button("Delete").clicked() {
                    *pending_action = Some(SceneTreeAction::Delete(entity));
                    ui.close_menu();
                }
            });
        });

        if self.renaming_entity == Some(entity) {
            visited.remove(&entity);
            return;
        }

        let children = self
            .world
            .get::<Children>(entity)
            .map(|children| children.0.clone())
            .unwrap_or_default();

        for child in children {
            self.draw_entity_node(ui, child, depth + 1, visibility, visited, pending_action);
        }

        visited.remove(&entity);
    }

    fn build_scene_tree_visibility(&self, roots: &[Entity], query: &str) -> HashMap<Entity, bool> {
        build_scene_tree_visibility_map(&self.world, roots, query)
    }

    fn entity_name_for_scene_tree(&self, entity: Entity) -> String {
        scene_tree_entity_name(&self.world, entity)
    }

    fn show_inspector_panel(&mut self, ui: &mut egui::Ui) {
        let panel_response = ui
            .scope(|ui| {
                ui.heading("Inspector");
                ui.separator();
                if InspectorPanel::show(
                    ui,
                    &mut self.world,
                    &self.selection,
                    &mut self.command_history,
                ) {
                    self.unsaved_changes = true;
                }
            })
            .response;

        self.handle_inspector_asset_drop(ui, &panel_response);
    }

    fn show_asset_browser_panel(&mut self, ui: &mut egui::Ui) {
        if let Err(error) = self.asset_browser.rescan_if_dirty() {
            ui.colored_label(
                egui::Color32::YELLOW,
                format!("Asset scan failed: {}", error),
            );
        }

        ui.heading("Assets");
        ui.separator();

        let mut move_entry_via_drop: Option<(String, Option<String>)> = None;
        let mut navigate_to: Option<Option<String>> = None;
        ui.horizontal_wrapped(|ui| {
            for (index, (label, relative)) in self.asset_browser.breadcrumbs().iter().enumerate() {
                if index > 0 {
                    ui.label("/");
                }

                let response = ui.small_button(label);
                if response.clicked() {
                    navigate_to = Some(relative.clone());
                }

                if response.dnd_hover_payload::<AssetDragPayload>().is_some() {
                    ui.painter().rect_filled(
                        response.rect.expand(2.0),
                        4.0,
                        ui.visuals().selection.bg_fill.gamma_multiply(0.20),
                    );
                }

                if let Some(payload) = response.dnd_release_payload::<AssetDragPayload>() {
                    if relative.as_deref() != Some(payload.relative_path.as_str()) {
                        move_entry_via_drop =
                            Some((payload.relative_path.clone(), relative.clone()));
                    }
                }
            }
        });

        if let Some(relative) = navigate_to {
            self.asset_browser
                .set_current_relative_path(relative.as_deref());
            self.asset_browser.set_selected_relative_path(None);
        }

        ui.separator();

        ui.horizontal(|ui| {
            ui.label("Filter:");
            let _ = ui.add(
                egui::TextEdit::singleline(self.asset_browser.filter_mut())
                    .hint_text("Search assets...")
                    .desired_width(220.0),
            );
            ui.separator();

            let view_mode = self.asset_browser.view_mode();
            if ui
                .selectable_label(view_mode == AssetBrowserViewModeConfig::Grid, "Grid")
                .clicked()
            {
                self.asset_browser
                    .set_view_mode(AssetBrowserViewModeConfig::Grid);
            }

            if ui
                .selectable_label(view_mode == AssetBrowserViewModeConfig::List, "List")
                .clicked()
            {
                self.asset_browser
                    .set_view_mode(AssetBrowserViewModeConfig::List);
            }

            ui.separator();

            if ui.button("Refresh").clicked() {
                self.asset_browser.mark_dirty();
            }

            if ui.button("New Folder").clicked() {
                self.begin_asset_create_folder_dialog();
            }

            if ui.button("New Asset").clicked() {
                self.begin_asset_create_asset_dialog();
            }
        });

        ui.separator();

        let selected_relative = self
            .asset_browser
            .selected_relative_path()
            .map(str::to_owned);

        ui.horizontal_wrapped(|ui| {
            if let Some(selected) = selected_relative.as_deref() {
                ui.label(format!("Selected: {}", selected));
            } else {
                ui.label("Selected: none");
            }

            ui.separator();

            if ui
                .add_enabled(selected_relative.is_some(), egui::Button::new("Rename"))
                .clicked()
            {
                self.begin_asset_rename_dialog();
            }

            if ui
                .add_enabled(selected_relative.is_some(), egui::Button::new("Move"))
                .clicked()
            {
                self.begin_asset_move_dialog();
            }

            if ui
                .add_enabled(selected_relative.is_some(), egui::Button::new("Delete"))
                .clicked()
            {
                self.begin_asset_delete_dialog();
            }
        });

        if let Some((level, message)) = self.asset_browser_ops.feedback.as_ref() {
            let color = match level {
                LogLevel::Trace | LogLevel::Info => egui::Color32::from_rgb(140, 210, 140),
                LogLevel::Warn => egui::Color32::from_rgb(235, 190, 120),
                LogLevel::Error => egui::Color32::from_rgb(230, 120, 120),
            };
            ui.colored_label(color, message);
        }

        ui.separator();

        let current_drop_target = self.asset_browser.current_relative_path();
        let drop_target_label = current_drop_target.as_deref().unwrap_or("assets root");
        let drop_zone = ui.add_sized(
            [ui.available_width(), 26.0],
            egui::Label::new(format!(
                "Drop here to move into current folder ({})",
                drop_target_label
            ))
            .sense(egui::Sense::hover()),
        );

        if drop_zone.dnd_hover_payload::<AssetDragPayload>().is_some() {
            ui.painter().rect_filled(
                drop_zone.rect.expand(2.0),
                4.0,
                ui.visuals().selection.bg_fill.gamma_multiply(0.20),
            );
        }

        if move_entry_via_drop.is_none() {
            if let Some(payload) = drop_zone.dnd_release_payload::<AssetDragPayload>() {
                if current_drop_target.as_deref() != Some(payload.relative_path.as_str()) {
                    move_entry_via_drop =
                        Some((payload.relative_path.clone(), current_drop_target.clone()));
                }
            }
        }

        ui.separator();

        let entries = self.asset_browser.filtered_entries();
        if entries.is_empty() {
            ui.label("No assets found for this folder/filter.");
            return;
        }

        let mut thumbnail_budget = 10;
        for entry in &entries {
            if thumbnail_budget == 0 {
                break;
            }

            if self.asset_browser.request_thumbnail_for_entry(entry) {
                thumbnail_budget -= 1;
            }
        }

        let mut select_entry: Option<Option<String>> = None;
        let mut open_directory: Option<String> = None;

        match self.asset_browser.view_mode() {
            AssetBrowserViewModeConfig::List => {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    for entry in &entries {
                        let is_selected = selected_relative
                            .as_deref()
                            .is_some_and(|selected| selected == entry.relative_path);

                        ui.horizontal(|ui| {
                            if let Some(texture_id) = self
                                .asset_browser
                                .thumbnail_texture_id(&entry.relative_path)
                            {
                                ui.image((texture_id, egui::vec2(20.0, 20.0)));
                            } else {
                                ui.add_sized(
                                    [20.0, 20.0],
                                    egui::Label::new(entry.kind.icon(entry.is_directory)),
                                );
                            }

                            let response = ui.selectable_label(is_selected, &entry.name);
                            response.dnd_set_drag_payload(AssetDragPayload::from_entry(
                                &entry.relative_path,
                                entry.kind,
                            ));

                            if entry.is_directory {
                                if response.dnd_hover_payload::<AssetDragPayload>().is_some() {
                                    ui.painter().rect_filled(
                                        response.rect.expand(2.0),
                                        4.0,
                                        ui.visuals().selection.bg_fill.gamma_multiply(0.20),
                                    );
                                }

                                if let Some(payload) =
                                    response.dnd_release_payload::<AssetDragPayload>()
                                {
                                    if payload.relative_path != entry.relative_path {
                                        move_entry_via_drop = Some((
                                            payload.relative_path.clone(),
                                            Some(entry.relative_path.clone()),
                                        ));
                                    }
                                }
                            }

                            if response.clicked() {
                                select_entry = Some(Some(entry.relative_path.clone()));
                            }

                            if response.double_clicked() && entry.is_directory {
                                open_directory = Some(entry.relative_path.clone());
                            }

                            if !entry.is_directory {
                                ui.separator();
                                ui.label(format_file_size(entry.file_size));
                            }
                        });
                    }
                });
            }
            AssetBrowserViewModeConfig::Grid => {
                let columns = (ui.available_width() / 140.0).floor().max(1.0) as usize;

                egui::ScrollArea::vertical().show(ui, |ui| {
                    egui::Grid::new("asset_browser_grid")
                        .num_columns(columns)
                        .spacing([8.0, 8.0])
                        .show(ui, |ui| {
                            for (index, entry) in entries.iter().enumerate() {
                                let is_selected = selected_relative
                                    .as_deref()
                                    .is_some_and(|selected| selected == entry.relative_path);

                                ui.vertical(|ui| {
                                    if let Some(texture_id) = self
                                        .asset_browser
                                        .thumbnail_texture_id(&entry.relative_path)
                                    {
                                        ui.image((texture_id, egui::vec2(64.0, 64.0)));
                                    } else {
                                        ui.allocate_ui_with_layout(
                                            egui::vec2(64.0, 64.0),
                                            egui::Layout::centered_and_justified(
                                                egui::Direction::LeftToRight,
                                            ),
                                            |ui| {
                                                ui.label(entry.kind.icon(entry.is_directory));
                                            },
                                        );
                                    }

                                    let response = ui.selectable_label(
                                        is_selected,
                                        truncate_asset_label(&entry.name, 16),
                                    );
                                    response.dnd_set_drag_payload(AssetDragPayload::from_entry(
                                        &entry.relative_path,
                                        entry.kind,
                                    ));

                                    if entry.is_directory {
                                        if response
                                            .dnd_hover_payload::<AssetDragPayload>()
                                            .is_some()
                                        {
                                            ui.painter().rect_filled(
                                                response.rect.expand(2.0),
                                                4.0,
                                                ui.visuals().selection.bg_fill.gamma_multiply(0.20),
                                            );
                                        }

                                        if let Some(payload) =
                                            response.dnd_release_payload::<AssetDragPayload>()
                                        {
                                            if payload.relative_path != entry.relative_path {
                                                move_entry_via_drop = Some((
                                                    payload.relative_path.clone(),
                                                    Some(entry.relative_path.clone()),
                                                ));
                                            }
                                        }
                                    }

                                    if response.clicked() {
                                        select_entry = Some(Some(entry.relative_path.clone()));
                                    }

                                    if response.double_clicked() && entry.is_directory {
                                        open_directory = Some(entry.relative_path.clone());
                                    }
                                });

                                if (index + 1) % columns == 0 {
                                    ui.end_row();
                                }
                            }
                        });
                });
            }
        }

        if let Some((source_relative_path, target_directory)) = move_entry_via_drop {
            self.commit_asset_drop_move(&source_relative_path, target_directory.as_deref());
        }

        if let Some(relative) = open_directory {
            self.asset_browser
                .set_current_relative_path(Some(relative.as_str()));
            self.asset_browser.set_selected_relative_path(None);
        } else if let Some(selected) = select_entry {
            self.asset_browser.set_selected_relative_path(selected);
        }
    }

    fn show_console_panel(&mut self, ui: &mut egui::Ui) {
        self.console.show(ui);
    }
}

impl eframe::App for EditorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let fs_events = self.asset_browser.process_file_events();
        if fs_events > 0 {
            self.log_message(
                LogLevel::Info,
                format!("Detected {} asset file change event(s)", fs_events),
            );
        }

        if let Err(error) = self.asset_browser.rescan_if_dirty() {
            self.log_message(
                LogLevel::Warn,
                format!("Asset Browser rescan failed: {}", error),
            );
        }

        let thumbnails_loaded = self.asset_browser.process_thumbnail_results(ctx);
        if thumbnails_loaded > 0 {
            ctx.request_repaint();
        }

        let hot_reload_count = self.asset_server.poll_texture_hot_reload();
        if hot_reload_count > 0 {
            self.log_message(
                LogLevel::Info,
                format!("Hot-reloaded {} texture asset(s)", hot_reload_count),
            );
        }

        self.handle_keyboard_shortcuts(ctx);
        self.validate_selection_state();
        self.draw_menu_bar(ctx);
        self.draw_status_bar(ctx);

        let mut dock_state = std::mem::replace(&mut self.dock_state, create_default_layout());
        {
            let mut tab_viewer = EditorTabViewer { app: self };
            DockArea::new(&mut dock_state)
                .style(egui_dock::Style::from_egui(ctx.style().as_ref()))
                .show(ctx, &mut tab_viewer);
        }
        self.dock_state = dock_state;

        self.draw_modals(ctx);
    }
}

impl Drop for EditorApp {
    fn drop(&mut self) {
        if let Some(render_state) = self.wgpu_render_state.as_ref() {
            self.viewport_renderer.free(render_state);
        }

        self.sync_config_from_runtime();
        if let Err(error) = self.config.save() {
            log::warn!(
                target: "engine::editor",
                "Failed to persist editor config on shutdown: {}",
                error
            );
        }
    }
}

struct EditorTabViewer<'a> {
    app: &'a mut EditorApp,
}

impl TabViewer for EditorTabViewer<'_> {
    type Tab = Tab;

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        tab.title().into()
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        match tab {
            Tab::Viewport => self.app.show_viewport_panel(ui),
            Tab::SceneTree => self.app.show_scene_tree_panel(ui),
            Tab::Inspector => self.app.show_inspector_panel(ui),
            Tab::AssetBrowser => self.app.show_asset_browser_panel(ui),
            Tab::Console => self.app.show_console_panel(ui),
        }
    }
}

pub(crate) fn collect_scene_tree_roots(world: &World) -> Vec<Entity> {
    world
        .iter_entities()
        .filter_map(|entity_ref| {
            let entity = entity_ref.id();
            match world.get::<Parent>(entity) {
                Some(parent) if world.get_entity(parent.0).is_ok() => None,
                _ => Some(entity),
            }
        })
        .collect()
}

pub(crate) fn build_scene_tree_visibility_map(
    world: &World,
    roots: &[Entity],
    query: &str,
) -> HashMap<Entity, bool> {
    let mut visibility = HashMap::new();
    let mut stack = HashSet::new();

    for root in roots {
        let _ = compute_scene_tree_visibility_map(world, *root, query, &mut visibility, &mut stack);
    }

    visibility
}

fn compute_scene_tree_visibility_map(
    world: &World,
    entity: Entity,
    query: &str,
    visibility: &mut HashMap<Entity, bool>,
    stack: &mut HashSet<Entity>,
) -> bool {
    if let Some(existing) = visibility.get(&entity) {
        return *existing;
    }

    if !stack.insert(entity) {
        visibility.insert(entity, false);
        return false;
    }

    let mut is_visible = scene_tree_entity_name(world, entity)
        .to_ascii_lowercase()
        .contains(query);

    let children = world
        .get::<Children>(entity)
        .map(|children| children.0.clone())
        .unwrap_or_default();

    for child in children {
        if compute_scene_tree_visibility_map(world, child, query, visibility, stack) {
            is_visible = true;
        }
    }

    stack.remove(&entity);
    visibility.insert(entity, is_visible);
    is_visible
}

pub(crate) fn scene_tree_entity_name(world: &World, entity: Entity) -> String {
    world
        .get::<EntityName>(entity)
        .map(|value| value.0.clone())
        .unwrap_or_else(|| format!("Entity {:?}", entity))
}

pub(crate) fn is_scene_tree_descendant(world: &World, candidate: Entity, ancestor: Entity) -> bool {
    let mut visited = HashSet::new();
    let mut current = Some(candidate);

    while let Some(entity) = current {
        if !visited.insert(entity) {
            break;
        }

        if entity == ancestor {
            return true;
        }

        current = world.get::<Parent>(entity).map(|parent| parent.0);
    }

    false
}

fn truncate_asset_label(label: &str, max_chars: usize) -> String {
    if label.chars().count() <= max_chars {
        return label.to_owned();
    }

    let mut truncated = label
        .chars()
        .take(max_chars.saturating_sub(3))
        .collect::<String>();
    truncated.push_str("...");
    truncated
}

fn format_file_size(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;

    if bytes < 1024 {
        return format!("{} B", bytes);
    }

    let size = bytes as f64;
    if size < MIB {
        return format!("{:.1} KiB", size / KIB);
    }

    if size < GIB {
        return format!("{:.1} MiB", size / MIB);
    }

    format!("{:.1} GiB", size / GIB)
}

fn mesh_local_bounds(mesh_data: &MeshData) -> Option<(Vec3, Vec3)> {
    if mesh_data.vertices.is_empty() {
        return None;
    }

    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);

    for vertex in &mesh_data.vertices {
        let position = Vec3::from_array(vertex.position);
        min = min.min(position);
        max = max.max(position);
    }

    Some((min, max))
}

fn transform_aabb(transform: Affine3A, local_min: Vec3, local_max: Vec3) -> (Vec3, Vec3) {
    let corners = [
        Vec3::new(local_min.x, local_min.y, local_min.z),
        Vec3::new(local_min.x, local_min.y, local_max.z),
        Vec3::new(local_min.x, local_max.y, local_min.z),
        Vec3::new(local_min.x, local_max.y, local_max.z),
        Vec3::new(local_max.x, local_min.y, local_min.z),
        Vec3::new(local_max.x, local_min.y, local_max.z),
        Vec3::new(local_max.x, local_max.y, local_min.z),
        Vec3::new(local_max.x, local_max.y, local_max.z),
    ];

    let mut world_min = Vec3::splat(f32::INFINITY);
    let mut world_max = Vec3::splat(f32::NEG_INFINITY);

    for corner in corners {
        let point = transform.transform_point3(corner);
        world_min = world_min.min(point);
        world_max = world_max.max(point);
    }

    (world_min, world_max)
}

fn texture_handle_eq(lhs: TextureHandle, rhs: TextureHandle) -> bool {
    lhs.id() == rhs.id() && lhs.generation() == rhs.generation()
}

pub(crate) fn asset_drop_entity_name(relative_path: &str) -> String {
    let stem = Path::new(relative_path)
        .file_stem()
        .and_then(|value| value.to_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("Dropped Mesh");

    let normalized = stem.replace(['_', '-'], " ");
    normalized.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub(crate) fn viewport_drop_position(
    pointer_position: egui::Pos2,
    viewport_rect: egui::Rect,
    camera_position: Vec3,
    camera_rotation: Quat,
    fov_y_radians: f32,
    fallback_position: Vec3,
) -> Vec3 {
    let Some((ray_origin, ray_direction)) = viewport_pick_ray(
        pointer_position,
        viewport_rect,
        camera_position,
        camera_rotation,
        fov_y_radians,
    ) else {
        return fallback_position;
    };

    if ray_direction.y.abs() <= f32::EPSILON {
        return fallback_position;
    }

    let t = -ray_origin.y / ray_direction.y;
    if !t.is_finite() || t <= 0.0 {
        return fallback_position;
    }

    let hit_position = ray_origin + ray_direction * t;
    if !hit_position.is_finite() {
        return fallback_position;
    }

    hit_position
}

pub(crate) fn viewport_pick_ray(
    pointer_position: egui::Pos2,
    viewport_rect: egui::Rect,
    camera_position: Vec3,
    camera_rotation: Quat,
    fov_y_radians: f32,
) -> Option<(Vec3, Vec3)> {
    let viewport_width = viewport_rect.width();
    let viewport_height = viewport_rect.height();
    if viewport_width <= 0.0 || viewport_height <= 0.0 {
        return None;
    }

    let local_x = pointer_position.x - viewport_rect.min.x;
    let local_y = pointer_position.y - viewport_rect.min.y;

    let ndc_x = (local_x / viewport_width) * 2.0 - 1.0;
    let ndc_y = 1.0 - (local_y / viewport_height) * 2.0;
    let aspect_ratio = viewport_width / viewport_height;
    let tan_half_fov = (fov_y_radians * 0.5).tan();

    let direction_camera = Vec3::new(
        ndc_x * aspect_ratio * tan_half_fov,
        ndc_y * tan_half_fov,
        -1.0,
    )
    .normalize_or_zero();
    if direction_camera.length_squared() <= f32::EPSILON {
        return None;
    }

    let direction_world = (camera_rotation * direction_camera).normalize_or_zero();
    if direction_world.length_squared() <= f32::EPSILON {
        return None;
    }

    Some((camera_position, direction_world))
}

pub(crate) fn ray_intersects_aabb(
    origin: Vec3,
    direction: Vec3,
    min: Vec3,
    max: Vec3,
    max_distance: f32,
) -> Option<f32> {
    let max_distance = max_distance.max(0.0);
    if max_distance <= 0.0 {
        return None;
    }

    let mut t_near: f32 = 0.0;
    let mut t_far: f32 = max_distance;

    let axes = [
        (origin.x, direction.x, min.x, max.x),
        (origin.y, direction.y, min.y, max.y),
        (origin.z, direction.z, min.z, max.z),
    ];

    for (origin_axis, direction_axis, min_axis, max_axis) in axes {
        if direction_axis.abs() <= f32::EPSILON {
            if origin_axis < min_axis || origin_axis > max_axis {
                return None;
            }
            continue;
        }

        let inv_dir = 1.0 / direction_axis;
        let mut t1 = (min_axis - origin_axis) * inv_dir;
        let mut t2 = (max_axis - origin_axis) * inv_dir;

        if t1 > t2 {
            std::mem::swap(&mut t1, &mut t2);
        }

        t_near = t_near.max(t1);
        t_far = t_far.min(t2);

        if t_far < t_near {
            return None;
        }
    }

    if t_far < 0.0 {
        return None;
    }

    Some(t_near.max(0.0))
}

fn transforms_approximately_equal(lhs: &Transform, rhs: &Transform) -> bool {
    let translation_close = (lhs.translation - rhs.translation).length_squared() <= 1e-10;
    let scale_close = (lhs.scale - rhs.scale).length_squared() <= 1e-10;
    let rotation_dot = lhs.rotation.dot(rhs.rotation).abs();
    let rotation_close = (1.0 - rotation_dot) <= 1e-6;

    translation_close && scale_close && rotation_close
}

pub(crate) fn snap_scalar(value: f32, step: f32) -> f32 {
    if step <= f32::EPSILON {
        return value;
    }
    (value / step).round() * step
}

pub(crate) fn snap_vec3(value: Vec3, step: f32) -> Vec3 {
    Vec3::new(
        snap_scalar(value.x, step),
        snap_scalar(value.y, step),
        snap_scalar(value.z, step),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_gizmo_drag_transform(
    initial: &Transform,
    mode: GizmoMode,
    orientation: GizmoOrientation,
    axis_constraint: Option<GizmoAxisConstraint>,
    snapping_enabled: bool,
    normalized_x: f32,
    normalized_y: f32,
    camera_distance: f32,
    camera_fov_y_radians: f32,
    translate_snap: f32,
    rotate_snap_degrees: f32,
    scale_snap: f32,
) -> Transform {
    match mode {
        GizmoMode::Translate => {
            let distance = camera_distance.max(0.25);
            let vertical_span = 2.0 * distance * (camera_fov_y_radians * 0.5).tan();
            let movement_scale = vertical_span * 2.0;
            let (axis_x, axis_y, axis_z) = match orientation {
                GizmoOrientation::Global => (Vec3::X, Vec3::Y, Vec3::Z),
                GizmoOrientation::Local => (
                    initial.rotation * Vec3::X,
                    initial.rotation * Vec3::Y,
                    initial.rotation * Vec3::Z,
                ),
            };
            let delta_x = normalized_x * movement_scale;
            let delta_y = -normalized_y * movement_scale;

            let mut translation = match axis_constraint {
                Some(GizmoAxisConstraint::AxisX) => initial.translation + axis_x * delta_x,
                Some(GizmoAxisConstraint::AxisY) => initial.translation + axis_y * delta_y,
                Some(GizmoAxisConstraint::AxisZ) => initial.translation + axis_z * delta_y,
                None => initial.translation + axis_x * delta_x + axis_y * delta_y,
            };

            if snapping_enabled {
                translation = snap_vec3(translation, translate_snap.max(0.001));
            }

            Transform {
                translation,
                rotation: initial.rotation,
                scale: initial.scale,
            }
        }
        GizmoMode::Rotate => {
            let mut yaw = -normalized_x * std::f32::consts::TAU;
            let mut pitch = -normalized_y * std::f32::consts::TAU;
            let mut roll = -normalized_x * std::f32::consts::TAU;

            match axis_constraint {
                Some(GizmoAxisConstraint::AxisX) => {
                    yaw = 0.0;
                    roll = 0.0;
                }
                Some(GizmoAxisConstraint::AxisY) => {
                    pitch = 0.0;
                    roll = 0.0;
                }
                Some(GizmoAxisConstraint::AxisZ) => {
                    yaw = 0.0;
                    pitch = 0.0;
                }
                None => {
                    roll = 0.0;
                }
            }

            if snapping_enabled {
                let rotate_snap_radians = rotate_snap_degrees.max(0.1).to_radians();
                yaw = snap_scalar(yaw, rotate_snap_radians);
                pitch = snap_scalar(pitch, rotate_snap_radians);
                roll = snap_scalar(roll, rotate_snap_radians);
            }

            let rotation_delta = Quat::from_rotation_y(yaw)
                * Quat::from_rotation_x(pitch)
                * Quat::from_rotation_z(roll);

            let rotation = match orientation {
                GizmoOrientation::Global => (rotation_delta * initial.rotation).normalize(),
                GizmoOrientation::Local => (initial.rotation * rotation_delta).normalize(),
            };

            Transform {
                translation: initial.translation,
                rotation,
                scale: initial.scale,
            }
        }
        GizmoMode::Scale => {
            let snap_step = scale_snap.max(0.001);
            let mut uniform_factor = (1.0 + normalized_x - normalized_y).max(0.05);
            let mut x_factor = (1.0 + normalized_x).max(0.05);
            let mut y_factor = (1.0 - normalized_y).max(0.05);
            let mut z_factor = (1.0 + normalized_x).max(0.05);

            if snapping_enabled {
                uniform_factor = snap_scalar(uniform_factor, snap_step).max(0.05);
                x_factor = snap_scalar(x_factor, snap_step).max(0.05);
                y_factor = snap_scalar(y_factor, snap_step).max(0.05);
                z_factor = snap_scalar(z_factor, snap_step).max(0.05);
            }

            let scale_factors = match axis_constraint {
                Some(GizmoAxisConstraint::AxisX) => Vec3::new(x_factor, 1.0, 1.0),
                Some(GizmoAxisConstraint::AxisY) => Vec3::new(1.0, y_factor, 1.0),
                Some(GizmoAxisConstraint::AxisZ) => Vec3::new(1.0, 1.0, z_factor),
                None => Vec3::splat(uniform_factor),
            };
            let scale = (initial.scale * scale_factors).max(Vec3::splat(0.001));

            Transform {
                translation: initial.translation,
                rotation: initial.rotation,
                scale,
            }
        }
    }
}

pub(crate) fn compute_gizmo_axis_constraint(
    mode: GizmoMode,
    manual_axis_constraint: Option<GizmoAxisConstraint>,
    axis_lock_enabled: bool,
    drag_intent: GizmoDragIntent,
) -> Option<GizmoAxisConstraint> {
    if let Some(manual_axis_constraint) = manual_axis_constraint {
        return Some(manual_axis_constraint);
    }

    if !axis_lock_enabled {
        return None;
    }

    match mode {
        GizmoMode::Translate | GizmoMode::Rotate => match drag_intent {
            GizmoDragIntent::AxisX => Some(GizmoAxisConstraint::AxisX),
            GizmoDragIntent::AxisY => Some(GizmoAxisConstraint::AxisY),
            GizmoDragIntent::Uniform => None,
        },
        GizmoMode::Scale => None,
    }
}

#[cfg(test)]
pub(crate) fn apply_axis_constraint(
    axis_constraint: Option<GizmoAxisConstraint>,
    normalized_x: f32,
    normalized_y: f32,
) -> (f32, f32) {
    match axis_constraint {
        Some(GizmoAxisConstraint::AxisX) => (normalized_x, 0.0),
        Some(GizmoAxisConstraint::AxisY) => (0.0, normalized_y),
        Some(GizmoAxisConstraint::AxisZ) => (0.0, 0.0),
        None => (normalized_x, normalized_y),
    }
}

pub(crate) fn compute_gizmo_drag_intent(
    mode: GizmoMode,
    normalized_x: f32,
    normalized_y: f32,
) -> GizmoDragIntent {
    match mode {
        GizmoMode::Translate | GizmoMode::Rotate => {
            if normalized_x.abs() >= normalized_y.abs() {
                GizmoDragIntent::AxisX
            } else {
                GizmoDragIntent::AxisY
            }
        }
        GizmoMode::Scale => GizmoDragIntent::Uniform,
    }
}

pub(crate) fn gizmo_drag_intent_label(mode: GizmoMode, intent: GizmoDragIntent) -> &'static str {
    match (mode, intent) {
        (GizmoMode::Translate, GizmoDragIntent::AxisX) => "Move X",
        (GizmoMode::Translate, GizmoDragIntent::AxisY) => "Move Y",
        (GizmoMode::Translate, GizmoDragIntent::Uniform) => "Move",
        (GizmoMode::Rotate, GizmoDragIntent::AxisX) => "Yaw",
        (GizmoMode::Rotate, GizmoDragIntent::AxisY) => "Pitch",
        (GizmoMode::Rotate, GizmoDragIntent::Uniform) => "Rotate",
        (GizmoMode::Scale, GizmoDragIntent::Uniform) => "Uniform",
        (GizmoMode::Scale, GizmoDragIntent::AxisX) => "Scale X",
        (GizmoMode::Scale, GizmoDragIntent::AxisY) => "Scale Y",
    }
}

pub(crate) fn gizmo_axis_constraint_label(axis_constraint: GizmoAxisConstraint) -> &'static str {
    match axis_constraint {
        GizmoAxisConstraint::AxisX => "X",
        GizmoAxisConstraint::AxisY => "Y",
        GizmoAxisConstraint::AxisZ => "Z",
    }
}

pub(crate) fn gizmo_manual_axis_lock_label(
    manual_axis_constraint: Option<GizmoAxisConstraint>,
) -> &'static str {
    match manual_axis_constraint {
        Some(GizmoAxisConstraint::AxisX) => "X",
        Some(GizmoAxisConstraint::AxisY) => "Y",
        Some(GizmoAxisConstraint::AxisZ) => "Z",
        None => "Free",
    }
}

fn draw_gizmo_mode_guides(
    painter: &egui::Painter,
    center: egui::Pos2,
    mode: GizmoMode,
    active_intent: Option<GizmoDragIntent>,
    active_axis_constraint: Option<GizmoAxisConstraint>,
) {
    let axis_highlighted =
        |axis: GizmoAxisConstraint, intent: GizmoDragIntent| match active_axis_constraint {
            Some(active) => active == axis,
            None => active_intent == Some(intent),
        };

    let highlighted = |base: egui::Color32, axis: GizmoAxisConstraint, intent: GizmoDragIntent| {
        if axis_highlighted(axis, intent) {
            base
        } else {
            base.gamma_multiply(0.55)
        }
    };

    match mode {
        GizmoMode::Translate => {
            let x_color = highlighted(
                egui::Color32::from_rgb(255, 110, 110),
                GizmoAxisConstraint::AxisX,
                GizmoDragIntent::AxisX,
            );
            let y_color = highlighted(
                egui::Color32::from_rgb(120, 220, 120),
                GizmoAxisConstraint::AxisY,
                GizmoDragIntent::AxisY,
            );
            let z_base = egui::Color32::from_rgb(125, 170, 245);
            let z_color = if active_axis_constraint == Some(GizmoAxisConstraint::AxisZ) {
                z_base
            } else {
                z_base.gamma_multiply(0.55)
            };

            painter.line_segment(
                [
                    center + egui::vec2(-14.0, 0.0),
                    center + egui::vec2(14.0, 0.0),
                ],
                egui::Stroke::new(1.75, x_color),
            );
            painter.line_segment(
                [
                    center + egui::vec2(0.0, -14.0),
                    center + egui::vec2(0.0, 14.0),
                ],
                egui::Stroke::new(1.75, y_color),
            );
            painter.line_segment(
                [
                    center + egui::vec2(-10.0, 10.0),
                    center + egui::vec2(10.0, -10.0),
                ],
                egui::Stroke::new(1.5, z_color),
            );
        }
        GizmoMode::Rotate => {
            let yaw_color = highlighted(
                egui::Color32::from_rgb(255, 205, 120),
                GizmoAxisConstraint::AxisX,
                GizmoDragIntent::AxisX,
            );
            let pitch_color = highlighted(
                egui::Color32::from_rgb(140, 200, 255),
                GizmoAxisConstraint::AxisY,
                GizmoDragIntent::AxisY,
            );
            let roll_base = egui::Color32::from_rgb(245, 170, 170);
            let roll_color = if active_axis_constraint == Some(GizmoAxisConstraint::AxisZ) {
                roll_base
            } else {
                roll_base.gamma_multiply(0.55)
            };

            painter.circle_stroke(center, 12.0, egui::Stroke::new(1.25, yaw_color));
            painter.circle_stroke(center, 16.0, egui::Stroke::new(1.0, pitch_color));
            painter.circle_stroke(center, 20.0, egui::Stroke::new(1.0, roll_color));
        }
        GizmoMode::Scale => {
            let uniform_active =
                active_axis_constraint.is_none() && active_intent == Some(GizmoDragIntent::Uniform);
            let x_color = if uniform_active {
                egui::Color32::from_rgb(130, 220, 130)
            } else {
                highlighted(
                    egui::Color32::from_rgb(170, 235, 170),
                    GizmoAxisConstraint::AxisX,
                    GizmoDragIntent::AxisX,
                )
            };
            let y_color = if uniform_active {
                egui::Color32::from_rgb(130, 220, 130)
            } else {
                highlighted(
                    egui::Color32::from_rgb(130, 205, 245),
                    GizmoAxisConstraint::AxisY,
                    GizmoDragIntent::AxisY,
                )
            };
            let z_base = egui::Color32::from_rgb(245, 180, 130);
            let z_color =
                if uniform_active || active_axis_constraint == Some(GizmoAxisConstraint::AxisZ) {
                    z_base
                } else {
                    z_base.gamma_multiply(0.55)
                };

            painter.line_segment(
                [
                    center + egui::vec2(-12.0, 0.0),
                    center + egui::vec2(12.0, 0.0),
                ],
                egui::Stroke::new(1.75, x_color),
            );
            painter.line_segment(
                [
                    center + egui::vec2(0.0, -12.0),
                    center + egui::vec2(0.0, 12.0),
                ],
                egui::Stroke::new(1.75, y_color),
            );
            painter.line_segment(
                [
                    center + egui::vec2(-9.0, 9.0),
                    center + egui::vec2(9.0, -9.0),
                ],
                egui::Stroke::new(1.75, z_color),
            );
        }
    }
}

pub(crate) fn cycle_gizmo_mode(mode: GizmoMode) -> GizmoMode {
    match mode {
        GizmoMode::Translate => GizmoMode::Rotate,
        GizmoMode::Rotate => GizmoMode::Scale,
        GizmoMode::Scale => GizmoMode::Translate,
    }
}

pub(crate) fn toggle_gizmo_orientation(orientation: GizmoOrientation) -> GizmoOrientation {
    match orientation {
        GizmoOrientation::Local => GizmoOrientation::Global,
        GizmoOrientation::Global => GizmoOrientation::Local,
    }
}

pub(crate) fn gizmo_mode_label(mode: GizmoMode) -> &'static str {
    match mode {
        GizmoMode::Translate => "Translate",
        GizmoMode::Rotate => "Rotate",
        GizmoMode::Scale => "Scale",
    }
}

pub(crate) fn gizmo_orientation_label(orientation: GizmoOrientation) -> &'static str {
    match orientation {
        GizmoOrientation::Local => "Local",
        GizmoOrientation::Global => "Global",
    }
}

fn project_world_to_viewport(
    view_proj: Mat4,
    viewport_rect: egui::Rect,
    point: Vec3,
) -> Option<egui::Pos2> {
    let clip = view_proj * point.extend(1.0);
    if clip.w <= f32::EPSILON {
        return None;
    }

    let ndc = clip.truncate() / clip.w;
    if ndc.z < -1.0 || ndc.z > 1.0 {
        return None;
    }

    let screen_x = viewport_rect.left() + (ndc.x * 0.5 + 0.5) * viewport_rect.width();
    let screen_y = viewport_rect.top() + (1.0 - (ndc.y * 0.5 + 0.5)) * viewport_rect.height();

    Some(egui::pos2(screen_x, screen_y))
}

fn build_editor_world() -> World {
    let mut world = create_world();

    let mut type_registry = ReflectTypeRegistry::default();
    let mut component_registry = ComponentRegistry::default();
    let mut metadata_registry = ReflectMetadataRegistry::default();

    register_core_reflection_types(
        &mut type_registry,
        &mut component_registry,
        &mut metadata_registry,
    );
    register_physics_reflection_types(
        &mut type_registry,
        &mut component_registry,
        &mut metadata_registry,
    );

    world.insert_resource(type_registry);
    world.insert_resource(component_registry);
    world.insert_resource(metadata_registry);

    world
}
