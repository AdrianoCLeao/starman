//! The shared gameplay runtime (ADR 0012).
//!
//! [`GameRuntime`] owns the ECS [`World`], the engine schedules and the
//! set of installed [`RuntimePlugin`]s. The windowed runner, editor
//! play-mode, `starman test`, the smoke test and benchmarks all build the
//! *same* runtime from the same plugin list, so a subsystem behaves
//! identically everywhere it runs.
//!
//! A frame is driven by [`GameRuntime::run_frame`], which advances the
//! game clock (with time scale and pause), runs the fixed-step schedule
//! zero or more times, then the update and pre-render schedules, calling
//! [`FrameHooks`] around them for non-ECS modules (renderer, audio device,
//! OS input pump, script hosts).

use std::collections::HashSet;

use bevy_ecs::event::{Event, Events};
use bevy_ecs::schedule::IntoSystemConfigs;
use bevy_ecs::system::Resource;
use bevy_ecs::world::World;
use bevy_reflect::GetTypeRegistration;
use engine_reflect::{
    with_reflection_registries, ComponentRegistry, ReflectMetadataRegistry, ReflectRegistration,
    ReflectTypeRegistry,
};

use crate::schedule::{EngineSchedules, PreRenderSet, ScheduleKind};
use crate::{
    propagate_transforms, register_core_reflection_types, sync_camera_aspect_from_window,
    FrameTime, HardeningConfig, Result, WindowSize, DEFAULT_FIXED_TIMESTEP_SECONDS,
};

/// Maximum fixed steps simulated in one frame; excess time is dropped so a
/// long hitch cannot spiral into ever longer frames.
pub const MAX_FIXED_STEPS_PER_FRAME: u32 = 8;

/// A self-contained engine subsystem that installs resources, systems,
/// events and reflected types into a [`GameRuntime`].
pub trait RuntimePlugin: Send + Sync + 'static {
    /// Unique, stable name (used for de-duplication and diagnostics).
    fn name(&self) -> &'static str;

    /// Installs the plugin. Called exactly once per runtime.
    fn build(&self, runtime: &mut GameRuntime);
}

/// Game-time control: time scale and pause (e.g. a pause menu).
///
/// Paused/scaled time affects `FrameTime::delta_seconds` and the fixed
/// schedule; `FrameTime::real_delta_seconds` stays unscaled so UI, audio
/// fades and menus keep running.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct GameClock {
    pub scale: f32,
    pub paused: bool,
}

impl Default for GameClock {
    fn default() -> Self {
        Self {
            scale: 1.0,
            paused: false,
        }
    }
}

impl GameClock {
    pub fn effective_scale(&self) -> f32 {
        if self.paused {
            0.0
        } else {
            self.scale.max(0.0)
        }
    }
}

/// Callbacks into non-ECS modules around the schedules of one frame.
pub trait FrameHooks {
    /// Before any schedule runs (OS input pump, control channels, …).
    fn begin_frame(&mut self, _world: &mut World) -> Result<()> {
        Ok(())
    }

    /// After each fixed step.
    fn after_fixed(&mut self, _world: &mut World, _fixed_dt: f32) -> Result<()> {
        Ok(())
    }

    /// After the update schedule.
    fn after_update(&mut self, _world: &mut World, _dt: f32) -> Result<()> {
        Ok(())
    }

    /// After the pre-render schedule, with the fixed-step interpolation alpha.
    fn render(&mut self, _world: &mut World, _alpha: f32) -> Result<()> {
        Ok(())
    }
}

/// Hooks that do nothing: headless runs, tests, benchmarks.
pub struct NoHooks;

impl FrameHooks for NoHooks {}

/// Per-frame summary returned by [`GameRuntime::run_frame`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RuntimeFrame {
    pub fixed_steps: u32,
    pub delta_seconds: f32,
    pub real_delta_seconds: f32,
    pub alpha: f32,
}

type EventUpdater = fn(&mut World);

pub struct GameRuntime {
    pub world: World,
    schedules: EngineSchedules,
    plugins: Vec<&'static str>,
    plugin_names: HashSet<&'static str>,
    event_updaters: Vec<EventUpdater>,
    registered_events: HashSet<std::any::TypeId>,
    startup_completed: bool,
    fixed_timestep_seconds: f64,
    accumulator_seconds: f64,
    elapsed_seconds: f64,
    real_elapsed_seconds: f64,
    frame_count: u64,
}

impl Default for GameRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl GameRuntime {
    pub fn new() -> Self {
        Self::with_fixed_timestep(DEFAULT_FIXED_TIMESTEP_SECONDS)
    }

    pub fn with_fixed_timestep(fixed_timestep_seconds: f64) -> Self {
        let fixed_timestep_seconds = if fixed_timestep_seconds > 0.0 {
            fixed_timestep_seconds
        } else {
            DEFAULT_FIXED_TIMESTEP_SECONDS
        };
        let mut runtime = Self {
            world: World::new(),
            schedules: EngineSchedules::new(),
            plugins: Vec::new(),
            plugin_names: HashSet::new(),
            event_updaters: Vec::new(),
            registered_events: HashSet::new(),
            startup_completed: false,
            fixed_timestep_seconds,
            accumulator_seconds: 0.0,
            elapsed_seconds: 0.0,
            real_elapsed_seconds: 0.0,
            frame_count: 0,
        };

        let mut type_registry = ReflectTypeRegistry::default();
        let mut component_registry = ComponentRegistry::default();
        let mut metadata_registry = ReflectMetadataRegistry::default();
        register_core_reflection_types(
            &mut type_registry,
            &mut component_registry,
            &mut metadata_registry,
        );

        runtime
            .insert_resource(type_registry)
            .insert_resource(component_registry)
            .insert_resource(metadata_registry)
            .insert_resource(WindowSize::default())
            .insert_resource(FrameTime {
                fixed_delta_seconds: fixed_timestep_seconds as f32,
                ..FrameTime::default()
            })
            .insert_resource(HardeningConfig::default())
            .insert_resource(GameClock::default())
            .add_systems(
                ScheduleKind::PreRender,
                (propagate_transforms, sync_camera_aspect_from_window)
                    .in_set(PreRenderSet::TransformPropagate),
            );
        runtime
    }

    /// Installs `plugin` unless a plugin with the same name already is.
    pub fn add_plugin<P: RuntimePlugin>(&mut self, plugin: P) -> &mut Self {
        self.add_boxed_plugin(Box::new(plugin))
    }

    pub fn add_boxed_plugin(&mut self, plugin: Box<dyn RuntimePlugin>) -> &mut Self {
        let name = plugin.name();
        if !self.plugin_names.insert(name) {
            log::debug!(target: "engine::runtime", "plugin '{name}' already installed; skipping");
            return self;
        }
        plugin.build(self);
        self.plugins.push(name);
        log::debug!(target: "engine::runtime", "installed runtime plugin '{name}'");
        self
    }

    pub fn add_plugins(&mut self, plugins: Vec<Box<dyn RuntimePlugin>>) -> &mut Self {
        for plugin in plugins {
            self.add_boxed_plugin(plugin);
        }
        self
    }

    pub fn has_plugin(&self, name: &str) -> bool {
        self.plugin_names.contains(name)
    }

    /// Installed plugin names in installation order.
    pub fn plugins(&self) -> &[&'static str] {
        &self.plugins
    }

    pub fn insert_resource<R: Resource>(&mut self, resource: R) -> &mut Self {
        self.world.insert_resource(resource);
        self
    }

    pub fn init_resource<R: Resource + Default>(&mut self) -> &mut Self {
        if !self.world.contains_resource::<R>() {
            self.world.insert_resource(R::default());
        }
        self
    }

    pub fn add_systems<Marker>(
        &mut self,
        schedule: ScheduleKind,
        systems: impl IntoSystemConfigs<Marker>,
    ) -> &mut Self {
        self.schedules.get_mut(schedule).add_systems(systems);
        self
    }

    pub fn schedules_mut(&mut self) -> &mut EngineSchedules {
        &mut self.schedules
    }

    /// Registers a double-buffered event channel. Buffers swap once per
    /// frame, so events sent during fixed steps are readable in the same
    /// frame's update and the next frame.
    pub fn add_event<E: Event>(&mut self) -> &mut Self {
        if self.registered_events.insert(std::any::TypeId::of::<E>()) {
            self.world.init_resource::<Events<E>>();
            self.event_updaters.push(update_events::<E>);
        }
        self
    }

    /// Registers a reflected component (serialization, inspector, scripts).
    pub fn register_component<T: ReflectRegistration>(&mut self) -> &mut Self {
        with_reflection_registries(&mut self.world, |types, components, metadata| {
            T::register_reflect(types, components, metadata);
        });
        self
    }

    /// Registers a reflected non-component type (field types, resources).
    pub fn register_type<T: GetTypeRegistration>(&mut self) -> &mut Self {
        with_reflection_registries(&mut self.world, |types, _, _| types.register::<T>());
        self
    }

    pub fn fixed_timestep_seconds(&self) -> f32 {
        self.fixed_timestep_seconds as f32
    }

    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    pub fn elapsed_seconds(&self) -> f64 {
        self.elapsed_seconds
    }

    /// Runs the startup schedule once (idempotent).
    pub fn run_startup(&mut self) {
        if !self.startup_completed {
            self.schedules.startup.run(&mut self.world);
            self.startup_completed = true;
        }
    }

    /// Runs one frame of `real_delta_seconds` (already clamped by the
    /// caller's clock) without external hooks.
    pub fn step(&mut self, real_delta_seconds: f32) -> RuntimeFrame {
        self.run_frame(real_delta_seconds, &mut NoHooks)
            .expect("NoHooks never fails")
    }

    /// Runs one full frame.
    pub fn run_frame(
        &mut self,
        real_delta_seconds: f32,
        hooks: &mut dyn FrameHooks,
    ) -> Result<RuntimeFrame> {
        let real_delta = real_delta_seconds.max(0.0);
        let clock = self
            .world
            .get_resource::<GameClock>()
            .copied()
            .unwrap_or_default();
        let delta = real_delta * clock.effective_scale();

        self.frame_count = self.frame_count.saturating_add(1);
        self.elapsed_seconds += delta as f64;
        self.real_elapsed_seconds += real_delta as f64;
        self.accumulator_seconds += delta as f64;

        for updater in &self.event_updaters {
            updater(&mut self.world);
        }

        self.publish_frame_time(delta, real_delta);
        self.run_startup();
        hooks.begin_frame(&mut self.world)?;

        let mut fixed_steps = 0;
        while self.accumulator_seconds >= self.fixed_timestep_seconds {
            if fixed_steps >= MAX_FIXED_STEPS_PER_FRAME {
                log::debug!(
                    target: "engine::runtime",
                    "dropping {:.3}s of simulation after {MAX_FIXED_STEPS_PER_FRAME} fixed steps",
                    self.accumulator_seconds
                );
                self.accumulator_seconds = 0.0;
                break;
            }
            self.accumulator_seconds -= self.fixed_timestep_seconds;
            self.schedules.fixed_update.run(&mut self.world);
            hooks.after_fixed(&mut self.world, self.fixed_timestep_seconds as f32)?;
            fixed_steps += 1;
        }

        self.schedules.update.run(&mut self.world);
        hooks.after_update(&mut self.world, delta)?;

        let alpha = (self.accumulator_seconds / self.fixed_timestep_seconds).clamp(0.0, 1.0) as f32;
        if let Some(mut frame_time) = self.world.get_resource_mut::<FrameTime>() {
            frame_time.alpha = alpha;
        }
        self.schedules.pre_render.run(&mut self.world);
        hooks.render(&mut self.world, alpha)?;

        Ok(RuntimeFrame {
            fixed_steps,
            delta_seconds: delta,
            real_delta_seconds: real_delta,
            alpha,
        })
    }

    fn publish_frame_time(&mut self, delta: f32, real_delta: f32) {
        let fixed = self.fixed_timestep_seconds as f32;
        let alpha = (self.accumulator_seconds / self.fixed_timestep_seconds).clamp(0.0, 1.0) as f32;
        self.world.insert_resource(FrameTime {
            delta_seconds: delta,
            real_delta_seconds: real_delta,
            fixed_delta_seconds: fixed,
            alpha,
            elapsed_seconds: self.elapsed_seconds,
            real_elapsed_seconds: self.real_elapsed_seconds,
            frame_count: self.frame_count,
        });
    }
}

fn update_events<E: Event>(world: &mut World) {
    if let Some(mut events) = world.get_resource_mut::<Events<E>>() {
        events.update();
    }
}

#[cfg(test)]
#[path = "runtime_tests.rs"]
mod tests;
