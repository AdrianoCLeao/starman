//! The shared gameplay runtime assembly (ADR 0012).
//!
//! Every host that simulates a game — the windowed runner, editor
//! play-mode, `starman test`, `engine-smoke` and the benchmarks — builds
//! its [`GameRuntime`] through [`build_runtime`], so the installed
//! subsystems, their ordering and their reflected types are identical
//! everywhere.

pub use engine_core::{GameRuntime, RuntimePlugin};

use engine_assets::Assets;
use engine_core::DEFAULT_FIXED_TIMESTEP_SECONDS;
use engine_project::GameSettings;

/// Options for [`build_runtime`].
#[derive(Clone, Debug)]
pub struct RuntimeOptions {
    pub fixed_timestep_seconds: f64,
    /// The typed-asset store of the host's `AssetServer`. Inserted before
    /// plugins install so their loaders register into the store the server
    /// actually fills. `None` creates a fresh, server-less store.
    pub assets: Option<Assets>,
    /// Project game settings to apply after the plugins are installed.
    pub game_settings: Option<GameSettings>,
    /// The project's `assets/` directory (localization folders, tools).
    pub assets_root: Option<std::path::PathBuf>,
}

impl Default for RuntimeOptions {
    fn default() -> Self {
        Self {
            fixed_timestep_seconds: DEFAULT_FIXED_TIMESTEP_SECONDS,
            assets: None,
            game_settings: None,
            assets_root: None,
        }
    }
}

impl RuntimeOptions {
    pub fn with_assets(mut self, assets: Assets) -> Self {
        self.assets = Some(assets);
        self
    }

    pub fn with_game_settings(mut self, settings: GameSettings) -> Self {
        self.game_settings = Some(settings);
        self
    }

    pub fn with_assets_root(mut self, root: impl Into<std::path::PathBuf>) -> Self {
        self.assets_root = Some(root.into());
        self
    }
}

/// The engine's standard plugin set, in installation order.
pub fn default_plugins() -> Vec<Box<dyn RuntimePlugin>> {
    vec![
        Box::new(engine_assets::AssetsPlugin),
        Box::new(engine_input::InputPlugin),
        Box::new(engine_physics::PhysicsPlugin),
        Box::new(engine_animation::AnimationPlugin),
        Box::new(engine_vfx::VfxPlugin),
        Box::new(engine_audio::AudioPlugin),
        Box::new(engine_localization::LocalizationPlugin),
        Box::new(engine_ui::UiPlugin),
        Box::new(engine_render::RenderPlugin),
    ]
}

/// Installs [`default_plugins`] into `runtime`.
pub fn install_default_plugins(runtime: &mut GameRuntime) {
    runtime.add_plugins(default_plugins());
}

/// Builds a runtime with every standard subsystem installed.
pub fn build_runtime(options: &RuntimeOptions) -> GameRuntime {
    let mut runtime = GameRuntime::with_fixed_timestep(options.fixed_timestep_seconds);
    if let Some(assets) = &options.assets {
        runtime.insert_resource(assets.clone());
    }
    if let Some(root) = &options.assets_root {
        runtime.insert_resource(engine_assets::AssetsRoot(root.clone()));
    }
    install_default_plugins(&mut runtime);
    if let Some(settings) = &options.game_settings {
        apply_game_settings(&mut runtime, settings);
    }
    runtime
}

/// Pushes the project's typed game settings into the subsystems' runtime
/// resources. Safe to call again after the settings change (editor).
pub fn apply_game_settings(runtime: &mut GameRuntime, settings: &GameSettings) {
    apply_input_settings(runtime, &settings.input);
    apply_physics_settings(runtime, &settings.physics);
    apply_audio_settings(runtime, &settings.audio);
    apply_ui_settings(runtime, &settings.ui);
    apply_localization_settings(runtime, &settings.localization);
    runtime.insert_resource(ProjectGameSettings(settings.clone()));
}

fn apply_audio_settings(runtime: &mut GameRuntime, audio: &engine_project::AudioSettings) {
    let Some(mut source) = runtime
        .world
        .get_resource_mut::<engine_audio::AudioMixerSource>()
    else {
        return;
    };
    match &audio.mixer {
        Some(mixer) => source.set_asset(mixer.clone()),
        None if source.asset().is_some() => source.clear(),
        None => {}
    }
}

fn apply_ui_settings(runtime: &mut GameRuntime, ui: &engine_project::UiSettings) {
    let Some(mut current) = runtime.world.get_resource_mut::<engine_ui::UiSettings>() else {
        return;
    };
    let [w, h] = ui.reference_resolution;
    let next = engine_ui::UiSettings {
        reference_resolution: engine_math::Vec2::new(w as f32, h as f32),
        scale_mode: match ui.scale_mode {
            engine_project::UiScaleMode::MatchHeight => engine_ui::ScaleMode::MatchHeight,
            engine_project::UiScaleMode::MatchWidth => engine_ui::ScaleMode::MatchWidth,
            engine_project::UiScaleMode::Fit => engine_ui::ScaleMode::Fit,
            engine_project::UiScaleMode::ConstantPixelSize => {
                engine_ui::ScaleMode::ConstantPixelSize
            }
        },
        user_scale: current.user_scale,
        fonts: ui.default_font.iter().cloned().collect(),
    };
    if *current != next {
        *current = next;
    }
}

/// Loads `assets/<root>/<locale>/*.ftl` when the assets directory is known
/// (hosts insert [`engine_assets::AssetsRoot`]); keeps the active locale.
fn apply_localization_settings(
    runtime: &mut GameRuntime,
    settings: &engine_project::LocalizationSettings,
) {
    let Some(root) = runtime
        .world
        .get_resource::<engine_assets::AssetsRoot>()
        .cloned()
    else {
        return;
    };
    let previous = runtime
        .world
        .get_resource::<engine_localization::Localization>()
        .map(|l| l.current().to_owned());
    let mut localization = engine_localization::Localization::load(
        root.0.join(&settings.root),
        &settings.default_locale,
        &settings.supported,
    );
    for issue in localization.issues() {
        log::warn!(target: "engine::l10n", "{issue}");
    }
    if let Some(locale) = previous {
        localization.set_locale(&locale);
    }
    runtime.world.insert_resource(localization);
}

fn apply_physics_settings(runtime: &mut GameRuntime, physics: &engine_project::PhysicsSettings) {
    let world = &mut runtime.world;
    if let Some(mut physics_world) = world.get_resource_mut::<engine_physics::PhysicsWorld3D>() {
        let [x, y, z] = physics.gravity;
        physics_world.gravity = engine_physics::pose::to_vector(engine_math::Vec3::new(x, y, z));
    }
    let layers = engine_physics::PhysicsLayers::new(&physics.layers, &physics.collision_matrix);
    let unchanged = world
        .get_resource::<engine_physics::PhysicsLayers>()
        .is_some_and(|current| *current == layers);
    if !unchanged {
        world.insert_resource(layers);
    }
}

fn apply_input_settings(runtime: &mut GameRuntime, input: &engine_project::InputSettings) {
    let world = &mut runtime.world;
    if let Some(mut source) = world.get_resource_mut::<engine_input::InputActionsSource>() {
        match &input.actions {
            Some(actions) => source.set_asset(actions.clone()),
            None => source.clear(),
        }
    }
    if let Some(mut players) = world.get_resource_mut::<engine_input::LocalPlayers>() {
        if players.max_players != input.max_local_players.clamp(1, 8) {
            *players = engine_input::LocalPlayers::new(input.max_local_players);
        }
    }
}

/// The project's game settings as last applied, for systems and tools that
/// need the raw values.
#[derive(bevy_ecs::system::Resource, Clone, Debug, Default)]
pub struct ProjectGameSettings(pub GameSettings);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_runtime_installs_every_standard_plugin() {
        let runtime = build_runtime(&RuntimeOptions::default());
        for plugin in default_plugins() {
            assert!(runtime.has_plugin(plugin.name()), "{}", plugin.name());
        }
    }

    #[test]
    fn game_settings_configure_input() {
        let mut settings = GameSettings::default();
        settings.input.actions = Some(engine_assets::AssetRef::from_path("input/game.input.ron"));
        settings.input.max_local_players = 2;
        let runtime = build_runtime(&RuntimeOptions::default().with_game_settings(settings));
        let source = runtime.world.resource::<engine_input::InputActionsSource>();
        assert_eq!(source.asset().unwrap().path, "input/game.input.ron");
        assert_eq!(
            runtime
                .world
                .resource::<engine_input::LocalPlayers>()
                .max_players,
            2
        );
    }

    #[test]
    fn game_settings_select_the_audio_mixer() {
        let mut settings = GameSettings::default();
        settings.audio.mixer = Some(engine_assets::AssetRef::from_path("audio/game.mixer.ron"));
        let runtime = build_runtime(&RuntimeOptions::default().with_game_settings(settings));
        let source = runtime.world.resource::<engine_audio::AudioMixerSource>();
        assert_eq!(source.asset().unwrap().path, "audio/game.mixer.ron");
        assert_eq!(
            runtime
                .world
                .resource::<engine_audio::AudioEngine>()
                .backend_name(),
            "null",
            "hosts opt into devices"
        );
    }

    #[test]
    fn default_runtime_steps_headless() {
        let mut runtime = build_runtime(&RuntimeOptions::default());
        for _ in 0..10 {
            runtime.step(1.0 / 60.0);
        }
        assert_eq!(runtime.frame_count(), 10);
    }
}
