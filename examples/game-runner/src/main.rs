use std::io::{self, BufRead};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;

use bevy_ecs::prelude::Res;
use bevy_ecs::schedule::IntoSystemConfigs;
use bevy_ecs::system::Resource;
use bevy_ecs::world::World;
use engine_assets::{AssetModule, AssetServer, SceneDeserializer};
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

const STOP_REQUESTED_WINDOW_ERROR: &str = "runner stop requested";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RunnerControlCommand {
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

struct RunnerModules {
    renderer: RenderModule,
    audio: AudioModule,
    input: InputModule,
    assets: AssetModule,
    control_rx: Receiver<RunnerControlCommand>,
    stop_requested: bool,
}

impl RunnerModules {
    fn new(
        assets_root: impl Into<String>,
        control_rx: Receiver<RunnerControlCommand>,
    ) -> Result<Self> {
        let assets = AssetModule::new(assets_root);
        let _ = assets.load_stub("textures/placeholder.png")?;

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
        while let Ok(command) = self.control_rx.try_recv() {
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

        let reload_count = self.assets.poll_texture_hot_reload();
        if reload_count > 0 {
            log::info!(
                target: "engine::runner",
                "Hot-reloaded {} texture asset(s)",
                reload_count
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

fn parse_control_command(line: &str) -> Option<RunnerControlCommand> {
    match line.trim().to_ascii_lowercase().as_str() {
        "pause" => Some(RunnerControlCommand::Pause),
        "resume" => Some(RunnerControlCommand::Resume),
        "stop" => Some(RunnerControlCommand::Stop),
        _ => None,
    }
}

fn spawn_control_channel() -> Receiver<RunnerControlCommand> {
    let (tx, rx) = mpsc::channel::<RunnerControlCommand>();

    if let Err(error) = std::thread::Builder::new()
        .name("runner-control-stdin".to_owned())
        .spawn(move || {
            let stdin = io::stdin();
            let mut reader = io::BufReader::new(stdin.lock());
            let mut line = String::new();

            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => {
                        if let Some(command) = parse_control_command(&line) {
                            if tx.send(command).is_err() {
                                break;
                            }
                        }
                    }
                    Err(read_error) => {
                        log::warn!(
                            target: "engine::runner",
                            "Control channel read error: {}",
                            read_error
                        );
                        break;
                    }
                }
            }
        })
    {
        log::warn!(
            target: "engine::runner",
            "Failed to spawn control channel thread: {}",
            error
        );
    }

    rx
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

fn load_scene_into_world(
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

fn parse_args() -> Result<(PathBuf, String)> {
    let mut args = std::env::args_os();
    let _binary = args.next();

    let Some(scene_path) = args.next() else {
        return Err(EngineError::Config(
            "usage: game-runner <scene_path> [assets_root]".to_owned(),
        ));
    };

    let assets_root = args
        .next()
        .map(|value| PathBuf::from(value).to_string_lossy().into_owned())
        .unwrap_or_else(|| "assets".to_owned());

    Ok((PathBuf::from(scene_path), assets_root))
}

fn run() -> Result<()> {
    let (scene_path, assets_root) = parse_args()?;

    if !scene_path.exists() {
        return Err(EngineError::AssetLoad {
            path: scene_path.display().to_string(),
            reason: "scene file does not exist".to_owned(),
        });
    }

    let control_rx = spawn_control_channel();
    let modules = RunnerModules::new(assets_root, control_rx)?;

    let scene_label = scene_path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("scene");

    let window_title = format!("Motley Play Mode - {}", scene_label);

    let config = EngineConfig::with_app_name(window_title.clone()).with_window_config(
        WindowConfig::default()
            .with_title(window_title)
            .with_size(1280, 720)
            .with_resizable(true)
            .with_vsync(true),
    );

    let mut engine = Engine::new(config, modules)?;
    configure_runner_world(&mut engine);

    let root_count = load_scene_into_world(
        &mut engine.world,
        engine.modules.assets.asset_server_mut(),
        &scene_path,
    )?;
    log::info!(
        target: "engine::runner",
        "Loaded scene '{}' with {} root entities",
        scene_path.display(),
        root_count
    );

    match engine.run() {
        Ok(()) => Ok(()),
        Err(EngineError::Window(message)) if message == STOP_REQUESTED_WINDOW_ERROR => Ok(()),
        Err(error) => Err(error),
    }
}

fn main() {
    engine_core::init_logging();

    if let Err(error) = run() {
        log::error!(target: "engine::runner", "Startup failed: {}", error);
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_control_command, RunnerControlCommand};

    #[test]
    fn parse_control_command_is_case_insensitive_and_trimmed() {
        assert_eq!(
            parse_control_command(" pause\n"),
            Some(RunnerControlCommand::Pause)
        );
        assert_eq!(
            parse_control_command("RESUME"),
            Some(RunnerControlCommand::Resume)
        );
        assert_eq!(
            parse_control_command(" stop  "),
            Some(RunnerControlCommand::Stop)
        );
    }

    #[test]
    fn parse_control_command_rejects_unknown_values() {
        assert_eq!(parse_control_command(""), None);
        assert_eq!(parse_control_command("play"), None);
        assert_eq!(parse_control_command("foobar"), None);
    }
}
