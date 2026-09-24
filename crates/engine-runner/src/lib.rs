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

use bevy_ecs::world::World;
use engine_assets::{AssetDatabase, AssetModule, AssetServer, SceneDeserializer};
use engine_audio::AudioModule;
use engine_core::{
    Engine, EngineConfig, EngineError, EngineModules, FrameTime, GameClock, HardeningConfig,
    Result, Window, WindowConfig, WindowEvent,
};
use engine_input::{InputModule, InputState};
use engine_lua::ExtensibilityHost;
use engine_project::Project;
use engine_reflect::{ComponentRegistry, ReflectMetadataRegistry, ReflectTypeRegistry};
use engine_render::{RenderModule, RenderSceneAdapter};
use engine_scene::expand_all_instances;

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
    /// Optional project root for plugins/Lua bootstrap (M3).
    pub project_root: Option<std::path::PathBuf>,
}

impl RunnerOptions {
    pub fn new(app_name: impl Into<String>) -> Self {
        Self {
            app_name: app_name.into(),
            window: WindowConfig::default(),
            control_rx: None,
            database: None,
            project_root: None,
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

    pub fn with_project_root(mut self, project_root: impl Into<std::path::PathBuf>) -> Self {
        self.project_root = Some(project_root.into());
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
    extensibility: Option<ExtensibilityHost>,
    game_settings: Option<engine_project::GameSettings>,
}

impl RunnerModules {
    fn new(
        assets_root: impl Into<String>,
        control_rx: Option<Receiver<RunnerControlCommand>>,
        database: Option<AssetDatabase>,
        project_root: Option<std::path::PathBuf>,
    ) -> Result<Self> {
        let mut assets = AssetModule::new(assets_root);
        let _ = assets.load_stub("textures/placeholder.png")?;
        if let Some(database) = database {
            assets.asset_server_mut().attach_database(database);
        }

        let mut game_settings = None;
        let extensibility = if let Some(root) = project_root {
            match Project::open(&root) {
                Ok(project) => {
                    game_settings = Some(project.manifest.game.clone());
                    let mut host =
                        ExtensibilityHost::bootstrap(project.paths.root(), &project.manifest)?;
                    if let Err(error) = host.load_lua_entry() {
                        log::warn!(
                            target: "engine::runner",
                            "failed to load Lua entry: {error}"
                        );
                    }
                    Some(host)
                }
                Err(error) => {
                    log::warn!(
                        target: "engine::runner",
                        "project open for extensibility failed: {error}"
                    );
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            renderer: RenderModule::new(),
            audio: AudioModule::new(),
            input: InputModule::new(),
            assets,
            control_rx,
            stop_requested: false,
            extensibility,
            game_settings,
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
                    if let Some(mut clock) = world.get_resource_mut::<GameClock>() {
                        if !clock.paused {
                            clock.paused = true;
                            changed = true;
                        }
                    }

                    if changed {
                        log::info!(target: "engine::runner", "Runner paused via control protocol");
                    }
                }
                RunnerControlCommand::Resume => {
                    let mut changed = false;
                    if let Some(mut clock) = world.get_resource_mut::<GameClock>() {
                        if clock.paused {
                            clock.paused = false;
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

    fn tick_lua(&mut self, world: &mut World, dt: f32) {
        let Some(host) = self.extensibility.as_mut() else {
            return;
        };
        let Some(lua) = host.lua.as_mut() else {
            return;
        };
        let components_ptr = world
            .get_resource::<ComponentRegistry>()
            .map(|r| r as *const ComponentRegistry);
        let types_ptr = world
            .get_resource::<ReflectTypeRegistry>()
            .map(|r| r as *const ReflectTypeRegistry);
        let (Some(components), Some(types)) = (components_ptr, types_ptr) else {
            return;
        };
        // Safety: exclusive world borrow for the duration of the Lua tick.
        let result = unsafe {
            let components = &*components;
            let types = &*types;
            lua.with_world(world, components, types, |lua| lua.tick(dt))
        };
        if let Err(error) = result {
            log::error!(target: "engine::runner", "Lua tick error (isolated): {error}");
        }
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

        let delta = world
            .get_resource::<FrameTime>()
            .map(|time| time.delta_seconds)
            .unwrap_or(0.0);
        if delta > 0.0 {
            self.tick_lua(world, delta);
        }
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
        if !reload.scripts.is_empty() {
            if let Some(host) = self.extensibility.as_mut() {
                if let Err(error) = host.hot_reload_lua() {
                    log::warn!(target: "engine::runner", "Lua hot reload failed: {error}");
                } else {
                    log::info!(target: "engine::runner", "Lua scripts hot-reloaded");
                }
            }
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
            let expanded = expand_all_instances(
                world,
                component_registry,
                type_registry,
                asset_server,
                Some(&render_scene_adapter as &dyn engine_assets::SceneExternalComponents),
            )?;
            if expanded > 0 {
                log::info!(
                    target: "engine::runner",
                    "Expanded {expanded} nested scene instance(s)"
                );
            }
            Ok(roots.len())
        },
    )
}

fn configure_runner_world(engine: &mut Engine<RunnerModules>) {
    let hardening = engine
        .runtime
        .world
        .get_resource::<HardeningConfig>()
        .copied()
        .unwrap_or_default();

    engine.modules.input.configure_hardening(hardening);
    engine.modules.assets.configure_hardening(hardening);

    let assets = engine.modules.assets.asset_server().assets().clone();
    engine.runtime.insert_resource(assets);
    engine_runtime::install_default_plugins(&mut engine.runtime);
    if let Some(settings) = engine.modules.game_settings.clone() {
        engine_runtime::apply_game_settings(&mut engine.runtime, &settings);
    }
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
        self.engine.runtime.world.iter_entities().count()
    }

    /// Simulates one frame of `delta_seconds` without a window or GPU:
    /// input/control/script pumping and every runtime schedule run, the
    /// renderer is skipped.
    pub fn step_headless(&mut self, delta_seconds: f32) -> Result<engine_core::RuntimeFrame> {
        struct HeadlessHooks<'a>(&'a mut RunnerModules);

        impl engine_core::FrameHooks for HeadlessHooks<'_> {
            fn begin_frame(&mut self, world: &mut World) -> Result<()> {
                // Deterministic headless runs finish every requested load
                // before the frame's systems observe it.
                self.0.assets.asset_server_mut().update_blocking();
                self.0.flush_input(world)
            }
        }

        let engine = &mut self.engine;
        engine
            .runtime
            .run_frame(delta_seconds, &mut HeadlessHooks(&mut engine.modules))
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

    let modules = RunnerModules::new(
        assets_root,
        options.control_rx,
        options.database,
        options.project_root,
    )?;

    let config = EngineConfig::with_app_name(options.app_name).with_window_config(options.window);

    let mut engine = Engine::new(config, modules)?;
    configure_runner_world(&mut engine);

    let root_entity_count = load_scene_into_world(
        &mut engine.runtime.world,
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
