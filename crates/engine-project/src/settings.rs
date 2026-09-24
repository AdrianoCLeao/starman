//! Typed game settings stored in the project manifest (manifest v2,
//! ADR 0020). Each section is owned by one subsystem; every field has a
//! serde default so a section can be omitted entirely.

use engine_assets::AssetRef;
use serde::{Deserialize, Serialize};

use crate::validate::ValidationReport;

/// Maximum number of named physics collision layers (bit width of the
/// Rapier interaction groups).
pub const MAX_PHYSICS_LAYERS: usize = 32;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
pub struct GameSettings {
    #[serde(default)]
    pub physics: PhysicsSettings,
    #[serde(default)]
    pub input: InputSettings,
    #[serde(default)]
    pub audio: AudioSettings,
    #[serde(default)]
    pub localization: LocalizationSettings,
    #[serde(default)]
    pub ui: UiSettings,
    #[serde(default)]
    pub save: SaveSettings,
    #[serde(default)]
    pub rendering: RenderingSettings,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct PhysicsSettings {
    #[serde(default = "default_gravity")]
    pub gravity: [f32; 3],
    /// Layer names by bit index. Layer 0 is the default layer.
    #[serde(default = "default_layers")]
    pub layers: Vec<String>,
    /// `collision_matrix[i]` is the bitmask of layers layer `i` collides
    /// with. Missing rows collide with everything.
    #[serde(default)]
    pub collision_matrix: Vec<u32>,
}

fn default_gravity() -> [f32; 3] {
    [0.0, -9.81, 0.0]
}

fn default_layers() -> Vec<String> {
    vec!["default".to_owned()]
}

impl Default for PhysicsSettings {
    fn default() -> Self {
        Self {
            gravity: default_gravity(),
            layers: default_layers(),
            collision_matrix: Vec::new(),
        }
    }
}

impl PhysicsSettings {
    /// Bit index of a named layer.
    pub fn layer_index(&self, name: &str) -> Option<usize> {
        self.layers.iter().position(|layer| layer == name)
    }

    /// Collision mask for `layer` (all layers when unspecified).
    pub fn mask_for(&self, layer: usize) -> u32 {
        self.collision_matrix
            .get(layer)
            .copied()
            .unwrap_or(u32::MAX)
    }

    /// Whether layers `a` and `b` interact (the matrix must agree both ways).
    pub fn collides(&self, a: usize, b: usize) -> bool {
        a < 32 && b < 32 && self.mask_for(a) & (1 << b) != 0 && self.mask_for(b) & (1 << a) != 0
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct InputSettings {
    /// The project's `InputActions` asset (`*.input.ron`).
    #[serde(default)]
    pub actions: Option<AssetRef>,
    #[serde(default = "default_max_players")]
    pub max_local_players: u8,
}

fn default_max_players() -> u8 {
    1
}

impl Default for InputSettings {
    fn default() -> Self {
        Self {
            actions: None,
            max_local_players: default_max_players(),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
pub struct AudioSettings {
    /// The project's `AudioMixer` asset (`*.mixer.ron`).
    #[serde(default)]
    pub mixer: Option<AssetRef>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct LocalizationSettings {
    #[serde(default = "default_locale")]
    pub default_locale: String,
    #[serde(default = "default_supported_locales")]
    pub supported: Vec<String>,
    /// Directory (relative to `assets/`) holding `<locale>/*.ftl`.
    #[serde(default = "default_localization_root")]
    pub root: String,
}

fn default_locale() -> String {
    "en".to_owned()
}

fn default_supported_locales() -> Vec<String> {
    vec![default_locale()]
}

fn default_localization_root() -> String {
    "localization".to_owned()
}

impl Default for LocalizationSettings {
    fn default() -> Self {
        Self {
            default_locale: default_locale(),
            supported: default_supported_locales(),
            root: default_localization_root(),
        }
    }
}

/// How game UI scales from its reference resolution to the window.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum UiScaleMode {
    /// Scale by window height / reference height.
    #[default]
    MatchHeight,
    MatchWidth,
    /// Scale by the smaller of the two ratios (UI always fits).
    Fit,
    /// No scaling beyond the OS DPI factor.
    ConstantPixelSize,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct UiSettings {
    #[serde(default = "default_reference_resolution")]
    pub reference_resolution: [u32; 2],
    #[serde(default)]
    pub scale_mode: UiScaleMode,
    /// Default font asset (`.ttf`/`.otf`).
    #[serde(default)]
    pub default_font: Option<AssetRef>,
}

fn default_reference_resolution() -> [u32; 2] {
    [1920, 1080]
}

impl Default for UiSettings {
    fn default() -> Self {
        Self {
            reference_resolution: default_reference_resolution(),
            scale_mode: UiScaleMode::default(),
            default_font: None,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct SaveSettings {
    #[serde(default = "default_slots")]
    pub slots: u32,
    #[serde(default = "default_true")]
    pub autosave: bool,
    #[serde(default = "default_autosave_interval")]
    pub autosave_interval_seconds: f32,
}

fn default_slots() -> u32 {
    3
}

fn default_true() -> bool {
    true
}

fn default_autosave_interval() -> f32 {
    300.0
}

impl Default for SaveSettings {
    fn default() -> Self {
        Self {
            slots: default_slots(),
            autosave: true,
            autosave_interval_seconds: default_autosave_interval(),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct RenderingSettings {
    /// `Low`, `Medium`, `High` or `Ultra`.
    #[serde(default = "default_quality")]
    pub default_quality: String,
}

fn default_quality() -> String {
    "High".to_owned()
}

impl Default for RenderingSettings {
    fn default() -> Self {
        Self {
            default_quality: default_quality(),
        }
    }
}

impl GameSettings {
    /// Validates value ranges and cross-field consistency. Asset
    /// references are checked by the project against its assets directory.
    pub fn validate(&self, report: &mut ValidationReport) {
        let physics = &self.physics;
        if physics.layers.is_empty() {
            report.push_error("game.physics.layers must declare at least the default layer");
        }
        if physics.layers.len() > MAX_PHYSICS_LAYERS {
            report.push_error(format!(
                "game.physics.layers declares {} layers; the maximum is {MAX_PHYSICS_LAYERS}",
                physics.layers.len()
            ));
        }
        let mut seen = std::collections::HashSet::new();
        for layer in &physics.layers {
            if layer.trim().is_empty() {
                report.push_error("game.physics.layers contains an empty layer name");
            } else if !seen.insert(layer.as_str()) {
                report.push_error(format!("game.physics.layers repeats the layer '{layer}'"));
            }
        }
        if physics.collision_matrix.len() > physics.layers.len() {
            report.push_warning(format!(
                "game.physics.collision_matrix has {} rows for {} layers; extra rows are ignored",
                physics.collision_matrix.len(),
                physics.layers.len()
            ));
        }
        if physics.gravity.iter().any(|value| !value.is_finite()) {
            report.push_error("game.physics.gravity must be finite");
        }

        if self.input.max_local_players == 0 || self.input.max_local_players > 8 {
            report.push_error("game.input.max_local_players must be between 1 and 8");
        }

        let localization = &self.localization;
        if localization.supported.is_empty() {
            report.push_error("game.localization.supported must list at least one locale");
        }
        if !localization
            .supported
            .contains(&localization.default_locale)
        {
            report.push_error(format!(
                "game.localization.default_locale '{}' is not in supported locales {:?}",
                localization.default_locale, localization.supported
            ));
        }
        for locale in &localization.supported {
            let valid = !locale.is_empty()
                && locale
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
            if !valid {
                report.push_error(format!(
                    "game.localization locale '{locale}' is not a valid language tag"
                ));
            }
        }

        if self.ui.reference_resolution.contains(&0) {
            report.push_error("game.ui.reference_resolution must be non-zero");
        }

        if self.save.slots == 0 || self.save.slots > 100 {
            report.push_error("game.save.slots must be between 1 and 100");
        }
        if !(self.save.autosave_interval_seconds.is_finite()
            && self.save.autosave_interval_seconds >= 10.0)
        {
            report.push_error("game.save.autosave_interval_seconds must be at least 10");
        }

        if !matches!(
            self.rendering.default_quality.as_str(),
            "Low" | "Medium" | "High" | "Ultra"
        ) {
            report.push_error(format!(
                "game.rendering.default_quality '{}' must be Low, Medium, High or Ultra",
                self.rendering.default_quality
            ));
        }
    }

    /// Every asset reference declared by the settings, with its field name.
    pub fn asset_refs(&self) -> Vec<(&'static str, &AssetRef)> {
        let mut out = Vec::new();
        if let Some(asset) = &self.input.actions {
            out.push(("game.input.actions", asset));
        }
        if let Some(asset) = &self.audio.mixer {
            out.push(("game.audio.mixer", asset));
        }
        if let Some(asset) = &self.ui.default_font {
            out.push(("game.ui.default_font", asset));
        }
        out
    }
}
