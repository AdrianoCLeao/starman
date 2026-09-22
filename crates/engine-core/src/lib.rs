pub mod camera;
pub mod error;
pub mod hardening;
pub mod id;
pub mod reflect;
pub mod schedule;
pub mod tag;
pub mod time;
pub mod transform;
pub mod window;

pub use camera::{sync_camera_aspect_from_window, Camera2d, Camera3d, PrimaryCamera, WindowSize};
pub use error::{EngineError, Result};
pub use hardening::HardeningConfig;
pub use id::{EntityId, PersistentId, ProjectId, SourceAssetId, SubAssetId};
pub use reflect::register_core_reflection_types;
pub use schedule::{EngineSchedules, FixedUpdate, PreRender, Startup, Update};
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
    pub delta_seconds: f32,
    pub fixed_delta_seconds: f32,
    pub alpha: f32,
    pub elapsed_seconds: f64,
    pub frame_count: u64,
}

impl Default for FrameTime {
    fn default() -> Self {
        Self {
            delta_seconds: 0.0,
            fixed_delta_seconds: DEFAULT_FIXED_TIMESTEP_SECONDS as f32,
            alpha: 0.0,
            elapsed_seconds: 0.0,
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
    pub world: World,
    pub time: Time,
    pub modules: M,
    pub config: EngineConfig,
    schedules: EngineSchedules,
    startup_completed: bool,
}

impl<M: EngineModules> Engine<M> {
    pub fn new(config: EngineConfig, modules: M) -> Result<Self> {
        config.validate()?;

        let window_size = WindowSize::new(config.window.width, config.window.height);

        let mut engine = Self {
            world: create_world(),
            time: Time::with_config(config.time),
            modules,
            config,
            schedules: EngineSchedules::new(),
            startup_completed: false,
        };

        let mut reflect_type_registry = engine_reflect::ReflectTypeRegistry::default();
        let mut component_registry = engine_reflect::ComponentRegistry::default();
        let mut metadata_registry = engine_reflect::ReflectMetadataRegistry::default();
        register_core_reflection_types(
            &mut reflect_type_registry,
            &mut component_registry,
            &mut metadata_registry,
        );

        engine
            .insert_resource(reflect_type_registry)
            .insert_resource(component_registry)
            .insert_resource(metadata_registry)
            .insert_resource(window_size)
            .insert_resource(FrameTime::default())
            .insert_resource(HardeningConfig::default())
            .add_pre_render_systems(propagate_transforms)
            .add_pre_render_systems(sync_camera_aspect_from_window);

        Ok(engine)
    }

    pub fn add_plugin<P: Plugin<M>>(&mut self, plugin: P) -> &mut Self {
        plugin.build(self);
        self
    }

    pub fn insert_resource<R: Resource>(&mut self, resource: R) -> &mut Self {
        self.world.insert_resource(resource);
        self
    }

    pub fn add_startup_systems<Marker>(
        &mut self,
        systems: impl IntoSystemConfigs<Marker>,
    ) -> &mut Self {
        self.schedules.startup.add_systems(systems);
        self
    }

    pub fn add_fixed_update_systems<Marker>(
        &mut self,
        systems: impl IntoSystemConfigs<Marker>,
    ) -> &mut Self {
        self.schedules.fixed_update.add_systems(systems);
        self
    }

    pub fn add_update_systems<Marker>(
        &mut self,
        systems: impl IntoSystemConfigs<Marker>,
    ) -> &mut Self {
        self.schedules.update.add_systems(systems);
        self
    }

    pub fn add_pre_render_systems<Marker>(
        &mut self,
        systems: impl IntoSystemConfigs<Marker>,
    ) -> &mut Self {
        self.schedules.pre_render.add_systems(systems);
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
        self.insert_resource(WindowSize::new(width, height));
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
        self.world.insert_resource(FrameTime {
            delta_seconds: self.time.delta_seconds(),
            fixed_delta_seconds: self.time.fixed_delta_seconds(),
            alpha: self.time.alpha(),
            elapsed_seconds: self.time.elapsed_seconds(),
            frame_count: self.time.frame_count(),
        });

        if !self.startup_completed {
            self.schedules.startup.run(&mut self.world);
            self.startup_completed = true;
        }

        self.modules.flush_input(&mut self.world)?;

        for fixed_dt in self.time.fixed_steps() {
            self.schedules.fixed_update.run(&mut self.world);
            self.modules.fixed_update(fixed_dt)?;
        }

        self.schedules.update.run(&mut self.world);
        self.modules.update(self.time.delta_seconds())?;

        self.schedules.pre_render.run(&mut self.world);
        let alpha = self.time.alpha();
        self.modules.render(&mut self.world, alpha)?;

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
