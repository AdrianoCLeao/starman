//! Shared runtime bootstrap: assembling an [`Engine`] wired with rendering,
//! audio, input and asset modules, loading a scene into it, and (only for
//! [`run_scene_windowed`]) running the windowed event loop.
//!
//! This exists so `examples/game-runner` and `starman-cli run`/`test` don't
//! duplicate the same bootstrap logic. [`prepare_scene_world`] is entirely
//! headless — it never touches a GPU device or opens a window (device
//! creation happens lazily in `RenderModule::initialize_with_window`, only
//! ever called from `window_created`) — so it is safe to call from
//! automated tests, including on headless CI runners.

use std::path::Path;
use std::sync::mpsc::Receiver;
use std::sync::Arc;

use bevy_ecs::prelude::Res;
use bevy_ecs::schedule::IntoSystemConfigs;
use bevy_ecs::system::Resource;
use bevy_ecs::world::World;
use engine_assets::{AssetDatabase, AssetModule, AssetServer, SceneDeserializer};
use engine_audio::AudioModule;
use engine_core::{
    Engine, EngineConfig, EngineError, EngineModules, HardeningConfig, Result, Window,
    WindowConfig, WindowEvent,
};
use engine_input::{InputModule, InputState};
use engine_physics::{
    physics_fixed_update_systems_3d, register_physics_reflection_types, ColliderEntityMap3D,
    PhysicsEntityHandles3D, PhysicsStepConfig3D, PhysicsWorld3D,
};
use engine_reflect::{
    with_reflection_registries, ComponentRegistry, ReflectMetadataRegistry, ReflectTypeRegistry,
};
use engine_render::{RenderModule, RenderSceneAdapter};

/// The window error message [`run_scene_windowed`] treats as a clean exit
/// (a `Stop` control command was received), rather than a failure.
pub const STOP_REQUESTED_WINDOW_ERROR: &str = "runner stop requested";

/// A command sent over a [`RunnerOptions::control_rx`] channel, typically
/// fed by a caller reading `stdin` (see `game-runner`'s play-mode protocol).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunnerControlCommand {
    Pause,
    Resume,
    Stop,
}

#[derive(Resource, Clone, Copy, Debug, Default)]
struct RunnerPlaybackState {
    paused: bool,
}

fn runner_is_playing(state: Res<RunnerPlaybackState>) -> bool {
    !state.paused
}

/// Options controlling how a scene is bootstrapped and (for
/// [`run_scene_windowed`]) presented.
pub struct RunnerOptions {
    pub app_name: String,
    pub window: WindowConfig,
    /// Optional remote-control channel (pause/resume/stop), e.g. fed by a
    /// caller reading commands from `stdin`. `None` means the run cannot be
    /// paused/stopped except by closing the window or returning an error.
    pub control_rx: Option<Receiver<RunnerControlCommand>>,
    /// An asset database to attach to the [`AssetServer`], so hot-reload
    /// during the run keeps `.meta`/cache entries current. `None` runs
    /// without one, reloading payloads only (matches `game-runner`'s
    /// current behavior).
    pub database: Option<AssetDatabase>,
}

impl RunnerOptions {
    pub fn new(app_name: impl Into<String>) -> Self {
        Self {
            app_name: app_name.into(),
            window: WindowConfig::default(),
            control_rx: None,
            database: None,
        }
    }

    pub fn with_window(mut self, window: WindowConfig) -> Self {
        self.window = window;
        self
    }

    pub fn with_control_rx(mut self, control_rx: Receiver<RunnerControlCommand>) -> Self {
        self.control_rx = Some(control_rx);
        self
    }

    pub fn with_database(mut self, database: AssetDatabase) -> Self {
        self.database = Some(database);
        self
    }
}

pub struct RunnerModules {
    renderer: RenderModule,
    audio: AudioModule,
    input: InputModule,
    assets: AssetModule,
    control_rx: Option<Receiver<RunnerControlCommand>>,
    stop_requested: bool,
}

impl RunnerModules {
    fn new(
        assets_root: impl Into<String>,
        control_rx: Option<Receiver<RunnerControlCommand>>,
        database: Option<AssetDatabase>,
    ) -> Result<Self> {
        let mut assets = AssetModule::new(assets_root);
        let _ = assets.load_stub("textures/placeholder.png")?;
        if let Some(database) = database {
            assets.asset_server_mut().attach_database(database);
        }

        Ok(Self {
            renderer: RenderModule::new(),
            audio: AudioModule::new(),
            input: InputModule::new(),
            assets,
            control_rx,
            stop_requested: false,
        })
    }

    fn process_control_messages(&mut self, world: &mut World) {
        let Some(control_rx) = &self.control_rx else {
            return;
        };

        while let Ok(command) = control_rx.try_recv() {
            match command {
                RunnerControlCommand::Pause => {
                    let mut changed = false;
                    if let Some(mut playback) = world.get_resource_mut::<RunnerPlaybackState>() {
                        if !playback.paused {
                            playback.paused = true;
                            changed = true;
                        }
                    }

                    if changed {
                        log::info!(target: "engine::runner", "Runner paused via control protocol");
                    }
                }
                RunnerControlCommand::Resume => {
                    let mut changed = false;
                    if let Some(mut playback) = world.get_resource_mut::<RunnerPlaybackState>() {
                        if playback.paused {
                            playback.paused = false;
                            changed = true;
                        }
                    }

                    if changed {
                        log::info!(target: "engine::runner", "Runner resumed via control protocol");
                    }
                }
                RunnerControlCommand::Stop => {
                    self.stop_requested = true;
                    log::info!(target: "engine::runner", "Runner stop requested via control protocol");
                }
            }
        }
    }

    /// The asset server backing this runner's loaded scene.
    pub fn asset_server(&self) -> &AssetServer {
        self.assets.asset_server()
    }

    pub fn asset_server_mut(&mut self) -> &mut AssetServer {
        self.assets.asset_server_mut()
    }
}

impl EngineModules for RunnerModules {
    fn window_created(&mut self, window: Arc<Window>, window_config: &WindowConfig) -> Result<()> {
        self.renderer
            .initialize_with_window(window, window_config.vsync)
    }

    fn window_event(&mut self, event: &WindowEvent) -> Result<()> {
        self.input.handle_window_event(event)
    }

    fn flush_input(&mut self, world: &mut World) -> Result<()> {
        if let Some(mut input_state) = world.get_resource_mut::<InputState>() {
            self.input.pump(&mut input_state)?;
        }

        self.process_control_messages(world);
        Ok(())
    }

    fn fixed_update(&mut self, _fixed_dt_seconds: f32) -> Result<()> {
        Ok(())
    }

    fn update(&mut self, _delta_seconds: f32) -> Result<()> {
        if self.stop_requested {
            return Err(EngineError::Window(STOP_REQUESTED_WINDOW_ERROR.to_owned()));
        }

        let reload = self.assets.poll_hot_reload();
        if reload.reloaded_count() > 0 {
            log::info!(
                target: "engine::runner",
                "Hot-reloaded {} texture(s), {} mesh(es), {} material(s)",
                reload.textures.len(),
                reload.meshes.len(),
                reload.materials.len()
            );
        }

        self.audio.update()
    }

    fn render(&mut self, world: &mut World, _alpha: f32) -> Result<()> {
        self.renderer.tick(world, self.assets.asset_server())
    }

    fn resized(&mut self, width: u32, height: u32) -> Result<()> {
        self.renderer.resize(width, height);
        Ok(())
    }
}

fn with_scene_context<R>(
    world: &mut World,
    asset_server: &mut AssetServer,
    f: impl FnOnce(
        &mut World,
        &ComponentRegistry,
        &ReflectTypeRegistry,
        &ReflectMetadataRegistry,
        &mut AssetServer,
    ) -> Result<R>,
) -> Result<R> {
    let type_registry = world
        .remove_resource::<ReflectTypeRegistry>()
        .unwrap_or_default();
    let component_registry = world
        .remove_resource::<ComponentRegistry>()
        .unwrap_or_default();
    let metadata_registry = world
        .remove_resource::<ReflectMetadataRegistry>()
        .unwrap_or_default();

    let result = f(
        world,
        &component_registry,
        &type_registry,
        &metadata_registry,
        asset_server,
    );

    world.insert_resource(type_registry);
    world.insert_resource(component_registry);
    world.insert_resource(metadata_registry);

    result
}

/// Clears `world` and loads `scene_path` into it, returning the number of
/// root entities spawned.
pub fn load_scene_into_world(
    world: &mut World,
    asset_server: &mut AssetServer,
    scene_path: &Path,
) -> Result<usize> {
    let render_scene_adapter = RenderSceneAdapter;

    with_scene_context(
        world,
        asset_server,
        |world, component_registry, type_registry, _metadata_registry, asset_server| {
            world.clear_entities();
            let mut deserializer =
                SceneDeserializer::new(world, component_registry, type_registry, asset_server)
                    .with_external_components(&render_scene_adapter);
            let roots = deserializer.load_file(scene_path)?;
            Ok(roots.len())
        },
    )
}

fn configure_runner_world(engine: &mut Engine<RunnerModules>) {
    let fixed_dt_seconds = engine.time.fixed_delta_seconds();
    let hardening = engine
        .world
        .get_resource::<HardeningConfig>()
        .copied()
        .unwrap_or_default();

    engine.modules.input.configure_hardening(hardening);
    engine.modules.assets.configure_hardening(hardening);

    engine
        .insert_resource(PhysicsWorld3D::with_timestep(fixed_dt_seconds))
        .insert_resource(PhysicsStepConfig3D::new(fixed_dt_seconds))
        .insert_resource(ColliderEntityMap3D::default())
        .insert_resource(PhysicsEntityHandles3D::default())
        .insert_resource(InputState::default())
        .insert_resource(RunnerPlaybackState::default())
        .add_fixed_update_systems(physics_fixed_update_systems_3d().run_if(runner_is_playing));

    with_reflection_registries(
        &mut engine.world,
        |type_registry, component_registry, metadata_registry| {
            register_physics_reflection_types(type_registry, component_registry, metadata_registry);
        },
    );
}

/// An assembled, scene-loaded [`Engine`], not yet running. Entirely
/// headless: no window has been created and no GPU device exists yet.
pub struct PreparedRun {
    pub engine: Engine<RunnerModules>,
    /// Root entities spawned when the scene was loaded.
    pub root_entity_count: usize,
}

impl PreparedRun {
    /// Total entity count in the loaded world (roots and their children).
    pub fn entity_count(&self) -> usize {
        self.engine.world.iter_entities().count()
    }
}

/// Assembles an [`Engine`] with rendering/audio/input/asset modules,
/// registers reflection types, and loads `scene_path` into it. Never
/// creates a window or a GPU device — safe to call from headless tests.
pub fn prepare_scene_world(
    assets_root: &str,
    scene_path: &Path,
    options: RunnerOptions,
) -> Result<PreparedRun> {
    if !scene_path.exists() {
        return Err(EngineError::AssetLoad {
            path: scene_path.display().to_string(),
            reason: "scene file does not exist".to_owned(),
        });
    }

    let modules = RunnerModules::new(assets_root, options.control_rx, options.database)?;

    let config = EngineConfig::with_app_name(options.app_name).with_window_config(options.window);

    let mut engine = Engine::new(config, modules)?;
    configure_runner_world(&mut engine);

    let root_entity_count = load_scene_into_world(
        &mut engine.world,
        engine.modules.assets.asset_server_mut(),
        scene_path,
    )?;
    log::info!(
        target: "engine::runner",
        "Loaded scene '{}' with {} root entities",
        scene_path.display(),
        root_entity_count
    );

    Ok(PreparedRun {
        engine,
        root_entity_count,
    })
}

/// Prepares (see [`prepare_scene_world`]) and then runs `scene_path`
/// windowed: opens a window, creates a GPU device, and blocks in the event
/// loop until the window closes or a `Stop` control command is received.
pub fn run_scene_windowed(
    assets_root: &str,
    scene_path: &Path,
    options: RunnerOptions,
) -> Result<()> {
    let prepared = prepare_scene_world(assets_root, scene_path, options)?;

    match prepared.engine.run() {
        Ok(()) => Ok(()),
        Err(EngineError::Window(message)) if message == STOP_REQUESTED_WINDOW_ERROR => Ok(()),
        Err(error) => Err(error),
    }
}
