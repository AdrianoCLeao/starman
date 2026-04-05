use std::fs;
use std::path::PathBuf;

use egui_dock::DockState;
use engine_core::{EngineError, Result};
use serde::{Deserialize, Serialize};

use crate::layout::{create_default_layout, Tab};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ViewportCameraConfig {
    #[serde(default)]
    pub target: [f32; 3],
    #[serde(default = "ViewportCameraConfig::default_distance")]
    pub distance: f32,
    #[serde(default)]
    pub yaw: f32,
    #[serde(default = "ViewportCameraConfig::default_pitch")]
    pub pitch: f32,
}

impl ViewportCameraConfig {
    const fn default_distance() -> f32 {
        12.0
    }

    const fn default_pitch() -> f32 {
        -20.0_f32.to_radians()
    }
}

impl Default for ViewportCameraConfig {
    fn default() -> Self {
        Self {
            target: [0.0, 0.0, 0.0],
            distance: Self::default_distance(),
            yaw: 0.0,
            pitch: Self::default_pitch(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ViewportOverlayConfig {
    #[serde(default = "ViewportOverlayConfig::default_show_grid")]
    pub show_grid: bool,
    #[serde(default)]
    pub show_collider_wireframes: bool,
    #[serde(default)]
    pub show_entity_labels: bool,
    #[serde(default = "ViewportOverlayConfig::default_show_fps")]
    pub show_fps: bool,
}

impl ViewportOverlayConfig {
    const fn default_show_grid() -> bool {
        true
    }

    const fn default_show_fps() -> bool {
        true
    }
}

impl Default for ViewportOverlayConfig {
    fn default() -> Self {
        Self {
            show_grid: Self::default_show_grid(),
            show_collider_wireframes: false,
            show_entity_labels: false,
            show_fps: Self::default_show_fps(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GizmoSnapConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "GizmoSnapConfig::default_translate_step")]
    pub translate_step: f32,
    #[serde(default = "GizmoSnapConfig::default_rotate_step_degrees")]
    pub rotate_step_degrees: f32,
    #[serde(default = "GizmoSnapConfig::default_scale_step")]
    pub scale_step: f32,
}

impl GizmoSnapConfig {
    const fn default_translate_step() -> f32 {
        0.5
    }

    const fn default_rotate_step_degrees() -> f32 {
        15.0
    }

    const fn default_scale_step() -> f32 {
        0.1
    }
}

impl Default for GizmoSnapConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            translate_step: Self::default_translate_step(),
            rotate_step_degrees: Self::default_rotate_step_degrees(),
            scale_step: Self::default_scale_step(),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GizmoModeConfig {
    Translate,
    Rotate,
    Scale,
}

impl Default for GizmoModeConfig {
    fn default() -> Self {
        Self::Translate
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GizmoOrientationConfig {
    Local,
    Global,
}

impl Default for GizmoOrientationConfig {
    fn default() -> Self {
        Self::Local
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GizmoAxisLockConfig {
    Free,
    AxisX,
    AxisY,
    AxisZ,
}

impl Default for GizmoAxisLockConfig {
    fn default() -> Self {
        Self::Free
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GizmoToolConfig {
    #[serde(default)]
    pub mode: GizmoModeConfig,
    #[serde(default)]
    pub orientation: GizmoOrientationConfig,
    #[serde(default)]
    pub axis_lock: GizmoAxisLockConfig,
}

impl Default for GizmoToolConfig {
    fn default() -> Self {
        Self {
            mode: GizmoModeConfig::default(),
            orientation: GizmoOrientationConfig::default(),
            axis_lock: GizmoAxisLockConfig::default(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EditorConfig {
    #[serde(default = "create_default_layout")]
    pub dock_state: DockState<Tab>,
    #[serde(default)]
    pub recent_files: Vec<PathBuf>,
    pub last_opened_scene: Option<PathBuf>,
    #[serde(default)]
    pub viewport_camera: ViewportCameraConfig,
    #[serde(default)]
    pub viewport_overlay: ViewportOverlayConfig,
    #[serde(default)]
    pub gizmo_snap: GizmoSnapConfig,
    #[serde(default)]
    pub gizmo_tool: GizmoToolConfig,
}

impl Default for EditorConfig {
    fn default() -> Self {
        Self {
            dock_state: create_default_layout(),
            recent_files: Vec::new(),
            last_opened_scene: None,
            viewport_camera: ViewportCameraConfig::default(),
            viewport_overlay: ViewportOverlayConfig::default(),
            gizmo_snap: GizmoSnapConfig::default(),
            gizmo_tool: GizmoToolConfig::default(),
        }
    }
}

impl EditorConfig {
    pub fn config_path() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("starman")
            .join("engine-editor")
            .join("config.ron")
    }

    pub fn load() -> Self {
        let path = Self::config_path();
        let Ok(contents) = fs::read_to_string(&path) else {
            return Self::default();
        };

        match ron::from_str::<Self>(&contents) {
            Ok(config) => config,
            Err(error) => {
                log::warn!(
                    target: "engine::editor",
                    "Failed to parse editor config at {}: {}",
                    path.display(),
                    error
                );
                Self::default()
            }
        }
    }

    pub fn save(&self) -> Result<()> {
        let path = Self::config_path();

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|error| {
                EngineError::Config(format!(
                    "failed to create config directory '{}': {}",
                    parent.display(),
                    error
                ))
            })?;
        }

        let payload =
            ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::new()).map_err(|error| {
                EngineError::Config(format!("failed to serialize config: {}", error))
            })?;

        fs::write(&path, payload).map_err(|error| {
            EngineError::Config(format!(
                "failed to write config '{}': {}",
                path.display(),
                error
            ))
        })
    }

    pub fn touch_recent_file(&mut self, path: PathBuf) {
        self.recent_files.retain(|entry| entry != &path);
        self.recent_files.insert(0, path.clone());
        self.recent_files.truncate(10);
        self.last_opened_scene = Some(path);
    }
}
