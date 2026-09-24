pub mod camera;
pub mod error;
pub mod hardening;
pub mod hierarchy;
pub mod id;
pub mod reflect;
pub mod runtime;
pub mod schedule;
pub mod tag;
pub mod time;
pub mod transform;
pub mod window;

pub use camera::{sync_camera_aspect_from_window, Camera2d, Camera3d, PrimaryCamera, WindowSize};
pub use error::{EngineError, Result};
pub use hardening::HardeningConfig;
pub use hierarchy::{despawn_recursive, set_parent, HierarchyCommandsExt};
pub use id::{EntityId, PersistentId, ProjectId, SourceAssetId, SubAssetId};
pub use reflect::register_core_reflection_types;
pub use runtime::{
    FrameHooks, GameClock, GameRuntime, NoHooks, RuntimeFrame, RuntimePlugin,
    MAX_FIXED_STEPS_PER_FRAME,
};
pub use schedule::{
    EngineSchedules, FixedSet, FixedUpdate, PreRender, PreRenderSet, ScheduleKind, Startup, Update,
    UpdateSet,
};
pub use tag::{Hidden, PhysicsControlled, RenderLayer2D, RenderLayer3D, Visible};
pub use time::{
    FixedStepIterator, Time, TimeConfig, DEFAULT_FIXED_TIMESTEP_SECONDS,
    DEFAULT_FPS_AVERAGE_WINDOW_SAMPLES, DEFAULT_MAX_FRAME_TIME_SECONDS,
};
pub use transform::{
    propagate_transforms, Children, EditorEntityBundle, EntityName, GlobalTransform, Parent,
    SpatialBundle, Transform,
};
pub use window::{run_windowed, WindowConfig, WindowLoop};
pub use winit::event::WindowEvent;
pub use winit::window::Window;

use bevy_ecs::{schedule::IntoSystemConfigs, system::Resource, world::World};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub app_name: String,
    pub window: WindowConfig,
    pub time: TimeConfig,
}

impl Default for EngineConfig {
    fn default() -> Self {
        let app_name = engine_name().to_owned();

        Self {
            app_name: app_name.clone(),
            window: WindowConfig::default().with_title(app_name),
            time: TimeConfig::default(),
        }
    }
}

impl EngineConfig {
    pub fn with_app_name(app_name: impl Into<String>) -> Self {
        let app_name = app_name.into();

        Self {
            app_name: app_name.clone(),
            window: WindowConfig::default().with_title(app_name),
            time: TimeConfig::default(),
        }
    }

    pub fn with_window_config(mut self, window: WindowConfig) -> Self {
        self.window = window;
        self
    }

    pub fn with_time_config(mut self, time: TimeConfig) -> Self {
        self.time = time;
        self
    }

    fn validate(&self) -> Result<()> {
        if self.app_name.trim().is_empty() {
            return Err(EngineError::Config("app_name cannot be empty".to_owned()));
        }

        if self.window.width == 0 || self.window.height == 0 {
            return Err(EngineError::Config(
                "window width and height must be greater than zero".to_owned(),
            ));
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FrameStats {
    pub fps_rolling: f32,
    pub fps_instant: f32,
    pub jitter_seconds: f32,
    pub delta_seconds: f32,
    pub elapsed_seconds: f64,
    pub frame_count: u64,
}

#[derive(Resource, Debug, Clone, Copy)]
pub struct FrameTime {
    /// Scaled game delta (0 while [`GameClock::paused`]).
    pub delta_seconds: f32,
    /// Unscaled wall-clock delta (UI, menus, audio fades).
    pub real_delta_seconds: f32,
    pub fixed_delta_seconds: f32,
    pub alpha: f32,
    pub elapsed_seconds: f64,
    pub real_elapsed_seconds: f64,
    pub frame_count: u64,
}

impl Default for FrameTime {
    fn default() -> Self {
        Self {
            delta_seconds: 0.0,
            real_delta_seconds: 0.0,
            fixed_delta_seconds: DEFAULT_FIXED_TIMESTEP_SECONDS as f32,
            alpha: 0.0,
            elapsed_seconds: 0.0,
            real_elapsed_seconds: 0.0,
            frame_count: 0,
        }
    }
}

pub trait EngineModules {
    /// Called once after the OS window has been created and before the first redraw.
    fn window_created(
        &mut self,
        _window: Arc<Window>,
        _window_config: &WindowConfig,
    ) -> Result<()> {
        Ok(())
    }

    /// Called for each window event dispatched by the window loop.
    fn window_event(&mut self, _event: &WindowEvent) -> Result<()> {
        Ok(())
    }

    /// Called once per rendered frame before fixed-step simulation.
    fn flush_input(&mut self, _world: &mut World) -> Result<()>;

    /// Called zero or more times per rendered frame using a fixed timestep.
    fn fixed_update(&mut self, _fixed_dt_seconds: f32) -> Result<()> {
        Ok(())
    }

    /// Called once per rendered frame with variable delta time.
    fn update(&mut self, _delta_seconds: f32) -> Result<()> {
        Ok(())
    }

    /// Called once per rendered frame after update with interpolation alpha [0, 1).
    fn render(&mut self, _world: &mut World, _alpha: f32) -> Result<()> {
        Ok(())
    }

    /// Called when the OS reports a window resize event for the active window.
    fn resized(&mut self, _width: u32, _height: u32) -> Result<()> {
        Ok(())
    }
}

pub trait Plugin<M: EngineModules> {
    fn build(&self, engine: &mut Engine<M>);
}

pub struct Engine<M: EngineModules> {
    pub runtime: GameRuntime,
    pub time: Time,
    pub modules: M,
    pub config: EngineConfig,
}

impl<M: EngineModules> std::ops::Deref for Engine<M> {
    type Target = GameRuntime;

    fn deref(&self) -> &GameRuntime {
        &self.runtime
    }
}

impl<M: EngineModules> std::ops::DerefMut for Engine<M> {
    fn deref_mut(&mut self) -> &mut GameRuntime {
        &mut self.runtime
    }
}

struct ModuleHooks<'a, M: EngineModules>(&'a mut M);

impl<M: EngineModules> FrameHooks for ModuleHooks<'_, M> {
    fn begin_frame(&mut self, world: &mut World) -> Result<()> {
        self.0.flush_input(world)
    }

    fn after_fixed(&mut self, _world: &mut World, fixed_dt: f32) -> Result<()> {
        self.0.fixed_update(fixed_dt)
    }

    fn after_update(&mut self, _world: &mut World, dt: f32) -> Result<()> {
        self.0.update(dt)
    }

    fn render(&mut self, world: &mut World, alpha: f32) -> Result<()> {
        self.0.render(world, alpha)
    }
}

impl<M: EngineModules> Engine<M> {
    pub fn new(config: EngineConfig, modules: M) -> Result<Self> {
        config.validate()?;

        let window_size = WindowSize::new(config.window.width, config.window.height);
        let mut runtime = GameRuntime::with_fixed_timestep(config.time.fixed_timestep_seconds);
        runtime.insert_resource(window_size);

        Ok(Self {
            runtime,
            time: Time::with_config(config.time),
            modules,
            config,
        })
    }

    pub fn add_plugin<P: Plugin<M>>(&mut self, plugin: P) -> &mut Self {
        plugin.build(self);
        self
    }

    /// Installs a [`RuntimePlugin`] into the underlying runtime.
    pub fn add_runtime_plugin<P: RuntimePlugin>(&mut self, plugin: P) -> &mut Self {
        self.runtime.add_plugin(plugin);
        self
    }

    pub fn insert_resource<R: Resource>(&mut self, resource: R) -> &mut Self {
        self.runtime.insert_resource(resource);
        self
    }

    pub fn add_startup_systems<Marker>(
        &mut self,
        systems: impl IntoSystemConfigs<Marker>,
    ) -> &mut Self {
        self.runtime.add_systems(ScheduleKind::Startup, systems);
        self
    }

    pub fn add_fixed_update_systems<Marker>(
        &mut self,
        systems: impl IntoSystemConfigs<Marker>,
    ) -> &mut Self {
        self.runtime.add_systems(ScheduleKind::FixedUpdate, systems);
        self
    }

    pub fn add_update_systems<Marker>(
        &mut self,
        systems: impl IntoSystemConfigs<Marker>,
    ) -> &mut Self {
        self.runtime.add_systems(ScheduleKind::Update, systems);
        self
    }

    pub fn add_pre_render_systems<Marker>(
        &mut self,
        systems: impl IntoSystemConfigs<Marker>,
    ) -> &mut Self {
        self.runtime.add_systems(ScheduleKind::PreRender, systems);
        self
    }

    pub fn tick(&mut self) -> Result<FrameStats> {
        self.time.advance();
        self.run_frame()
    }

    pub fn tick_with_frame_time(&mut self, frame_time_seconds: f64) -> Result<FrameStats> {
        self.time.advance_by(frame_time_seconds);
        self.run_frame()
    }

    pub fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        self.runtime.insert_resource(WindowSize::new(width, height));
        self.modules.resized(width, height)
    }

    pub fn frame_stats(&self) -> FrameStats {
        FrameStats {
            fps_rolling: self.time.fps(),
            fps_instant: self.time.instant_fps(),
            jitter_seconds: self.time.jitter_seconds(),
            delta_seconds: self.time.delta_seconds(),
            elapsed_seconds: self.time.elapsed_seconds(),
            frame_count: self.time.frame_count(),
        }
    }

    pub fn window_title(&self) -> String {
        let vsync_status = if self.config.window.vsync {
            "VSync On"
        } else {
            "VSync Off"
        };

        format!(
            "{} | {:.1} FPS | {}",
            self.config.app_name,
            self.time.fps(),
            vsync_status
        )
    }

    pub fn run(self) -> Result<()>
    where
        M: 'static,
    {
        let window_config = self.config.window.clone();
        run_windowed(window_config, self)
    }

    fn run_frame(&mut self) -> Result<FrameStats> {
        let delta = self.time.delta_seconds();
        let mut hooks = ModuleHooks(&mut self.modules);
        self.runtime.run_frame(delta, &mut hooks)?;
        Ok(self.frame_stats())
    }
}

impl<M: EngineModules> WindowLoop for Engine<M> {
    fn window_created(&mut self, window: Arc<Window>, window_config: &WindowConfig) -> Result<()> {
        self.modules.window_created(window, window_config)
    }

    fn window_event(&mut self, event: &WindowEvent) -> Result<()> {
        self.modules.window_event(event)
    }

    fn tick(&mut self) -> Result<()> {
        Engine::tick(self).map(|_| ())
    }

    fn resized(&mut self, width: u32, height: u32) -> Result<()> {
        self.resize(width, height)
    }

    fn title(&self) -> String {
        self.window_title()
    }
}

pub fn init_logging() {
    let _ = engine_diagnostics::initialize(engine_diagnostics::DiagnosticsConfig::for_application(
        engine_name(),
    ));
}

pub fn engine_name() -> &'static str {
    "Starman"
}

pub fn create_world() -> World {
    World::new()
}

#[cfg(test)]
mod tests;
