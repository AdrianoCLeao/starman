use bevy_ecs::prelude::{Commands, Query, Res, ResMut, Resource, With};
use bevy_ecs::world::World;
use engine_assets::{AssetDatabase, AssetModule, MaterialHandle, MeshHandle, TextureHandle};
use engine_core::{
    self, Camera2d, Camera3d, Children, Engine, EngineConfig, EngineModules, FrameTime,
    HardeningConfig, Parent, Plugin, PrimaryCamera, RenderLayer3D, Result, SpatialBundle,
    Transform, Visible, WindowConfig, WindowEvent,
};
use engine_input::{InputModule, InputState};
use engine_math::{glam::EulerRot, Quat, Vec2, Vec3};
use engine_physics::{ColliderShape3D, RigidBody3DBundle, RigidBodyType};
use engine_render::{MeshRenderable3d, RenderModule};
use gilrs::{Axis, Button};
use std::sync::{Arc, Once};
use winit::window::Window;
use winit::{event::MouseButton, keyboard::KeyCode};

static ECS_UPDATE_LOG_ONCE: Once = Once::new();

#[derive(Resource, Clone, Copy)]
struct SandboxRenderAssets {
    mesh: MeshHandle,
    texture: TextureHandle,
    material: MaterialHandle,
}

fn ecs_startup_smoke() {
    log::info!(target: "engine::sandbox", "ECS Startup schedule executed");
}

fn ecs_spawn_smoke_entities(mut commands: Commands) {
    let parent = commands
        .spawn((
            SpatialBundle {
                transform: Transform::from_xyz(0.0, 1.0, 0.0),
                ..SpatialBundle::default()
            },
            Children::default(),
            Visible,
            RenderLayer3D,
        ))
        .id();

    let child = commands
        .spawn((
            SpatialBundle {
                transform: Transform::from_xyz(1.0, 0.0, 0.0),
                ..SpatialBundle::default()
            },
            Parent(parent),
            Visible,
            RenderLayer3D,
        ))
        .id();

    commands.entity(parent).insert(Children(vec![child]));

    commands.spawn((
        SpatialBundle {
            transform: Transform::from_xyz(0.0, 5.0, 10.0),
            ..SpatialBundle::default()
        },
        Camera3d::default(),
        PrimaryCamera,
        Visible,
    ));

    commands.spawn((
        SpatialBundle::default(),
        Camera2d::default(),
        PrimaryCamera,
        Visible,
    ));

    log::info!(
        target: "engine::sandbox",
        "ECS smoke entities spawned (hierarchy + 3D/2D cameras)"
    );
}

fn ecs_spawn_renderable_entity(
    mut commands: Commands,
    render_assets: Option<Res<SandboxRenderAssets>>,
) {
    let Some(render_assets) = render_assets else {
        log::warn!(
            target: "engine::sandbox",
            "Skipping 3D renderable spawn because demo assets are unavailable"
        );
        return;
    };

    let render_assets = *render_assets;
    commands.spawn((
        SpatialBundle {
            transform: Transform::from_xyz(0.0, 0.0, -3.0),
            ..SpatialBundle::default()
        },
        MeshRenderable3d::new(
            render_assets.mesh,
            render_assets.texture,
            render_assets.material,
        ),
        Visible,
        RenderLayer3D,
    ));

    log::info!(
        target: "engine::sandbox",
        "Spawned 3D mesh entity for EP-04 render validation"
    );
}

fn ecs_spawn_physics_scene(
    mut commands: Commands,
    render_assets: Option<Res<SandboxRenderAssets>>,
) {
    let render_assets = render_assets.map(|assets| *assets);

    let mut floor_transform = Transform::from_xyz(0.0, -0.5, 0.0);
    floor_transform.scale = Vec3::new(40.0, 1.0, 40.0);

    let mut floor = commands.spawn((
        RigidBody3DBundle {
            body_type: RigidBodyType::Static,
            shape: ColliderShape3D::Box {
                half_extents: Vec3::new(20.0, 0.5, 20.0),
            },
            transform: floor_transform,
            ..RigidBody3DBundle::default()
        },
        Visible,
        RenderLayer3D,
    ));

    if let Some(render_assets) = render_assets {
        floor.insert(MeshRenderable3d::new(
            render_assets.mesh,
            render_assets.texture,
            render_assets.material,
        ));
    }

    for index in 0..5 {
        let x = index as f32 * 1.6 - 3.2;
        let y = 5.0 + (index as f32 * 0.75);

        let mut cube = commands.spawn((
            RigidBody3DBundle {
                body_type: RigidBodyType::Dynamic,
                shape: ColliderShape3D::Box {
                    half_extents: Vec3::splat(0.5),
                },
                transform: Transform::from_xyz(x, y, 0.0),
                ..RigidBody3DBundle::default()
            },
            Visible,
            RenderLayer3D,
        ));

        if let Some(render_assets) = render_assets {
            cube.insert(MeshRenderable3d::new(
                render_assets.mesh,
                render_assets.texture,
                render_assets.material,
            ));
        }
    }

    log::info!(
        target: "engine::sandbox",
        "Spawned EP-05 physics validation scene (static floor + dynamic cubes)"
    );
}

fn ecs_update_smoke(
    primary_cameras: Query<(), With<PrimaryCamera>>,
    parented_entities: Query<(), With<Parent>>,
) {
    ECS_UPDATE_LOG_ONCE.call_once(|| {
        let camera_count = primary_cameras.iter().count();
        let parented_count = parented_entities.iter().count();

        log::info!(
            target: "engine::sandbox",
            "ECS Update schedule executed (primary_cameras={}, parented_entities={})",
            camera_count,
            parented_count
        );
    });
}

#[derive(Resource, Default)]
struct CameraLookState {
    yaw: f32,
    pitch: f32,
    initialized: bool,
}

fn keyboard_movement_intent(input: &InputState) -> Vec3 {
    let mut movement = Vec3::ZERO;

    if input.key_held(KeyCode::KeyW) {
        movement.z += 1.0;
    }
    if input.key_held(KeyCode::KeyS) {
        movement.z -= 1.0;
    }
    if input.key_held(KeyCode::KeyA) {
        movement.x -= 1.0;
    }
    if input.key_held(KeyCode::KeyD) {
        movement.x += 1.0;
    }
    if input.key_held(KeyCode::Space) {
        movement.y += 1.0;
    }
    if input.key_held(KeyCode::ShiftLeft) || input.key_held(KeyCode::ShiftRight) {
        movement.y -= 1.0;
    }

    movement
}

fn ecs_camera_controller(
    input: Res<InputState>,
    frame_time: Res<FrameTime>,
    mut look_state: ResMut<CameraLookState>,
    mut cameras: Query<&mut Transform, (With<Camera3d>, With<PrimaryCamera>)>,
) {
    let Ok(mut transform) = cameras.get_single_mut() else {
        return;
    };

    if !look_state.initialized {
        let (yaw, pitch, _) = transform.rotation.to_euler(EulerRot::YXZ);
        look_state.yaw = yaw;
        look_state.pitch = pitch;
        look_state.initialized = true;
    }

    let mut movement = keyboard_movement_intent(&input);

    let mut mouse_look_delta = Vec2::ZERO;
    if input.mouse_held(MouseButton::Right) {
        mouse_look_delta = input.mouse_delta;
    }

    let mut gamepad_look_delta = Vec2::ZERO;
    if let Some(gamepad) = input.first_connected_gamepad() {
        movement.x += input.gamepad_axis(gamepad, Axis::LeftStickX);
        movement.z += -input.gamepad_axis(gamepad, Axis::LeftStickY);

        if input.gamepad_button_held(gamepad, Button::South)
            || input.gamepad_button_held(gamepad, Button::RightTrigger)
            || input.gamepad_button_held(gamepad, Button::RightTrigger2)
        {
            movement.y += 1.0;
        }

        if input.gamepad_button_held(gamepad, Button::East)
            || input.gamepad_button_held(gamepad, Button::LeftTrigger)
            || input.gamepad_button_held(gamepad, Button::LeftTrigger2)
        {
            movement.y -= 1.0;
        }

        gamepad_look_delta.x = input.gamepad_axis(gamepad, Axis::RightStickX);
        gamepad_look_delta.y = input.gamepad_axis(gamepad, Axis::RightStickY);
    }

    const LOOK_SENSITIVITY_DEGREES: f32 = 0.12;
    let mouse_sensitivity_radians = LOOK_SENSITIVITY_DEGREES.to_radians();
    look_state.yaw -= mouse_look_delta.x * mouse_sensitivity_radians;
    look_state.pitch -= mouse_look_delta.y * mouse_sensitivity_radians;

    let gamepad_look_speed_radians = 2.5;
    look_state.yaw -= gamepad_look_delta.x * gamepad_look_speed_radians * frame_time.delta_seconds;
    look_state.pitch -=
        gamepad_look_delta.y * gamepad_look_speed_radians * frame_time.delta_seconds;

    let pitch_limit = 89.0_f32.to_radians();
    look_state.pitch = look_state.pitch.clamp(-pitch_limit, pitch_limit);
    transform.rotation = Quat::from_euler(EulerRot::YXZ, look_state.yaw, look_state.pitch, 0.0);

    if movement.length_squared() > 0.0 {
        movement = movement.normalize();
        let forward = transform.rotation * Vec3::new(0.0, 0.0, -1.0);
        let right = transform.rotation * Vec3::new(1.0, 0.0, 0.0);
        let world_movement = right * movement.x + forward * movement.z + Vec3::Y * movement.y;

        transform.translation += world_movement * 5.0 * frame_time.delta_seconds;
    }
}

/// Best-effort: if run from a directory that is (or is inside) a Starman
/// project — the normal case, run from the repo root against
/// `examples/reference-project` — returns it, so the sandbox can use its
/// `assets/` directory and attach an `AssetDatabase`. `None` otherwise,
/// which is not an error: the sandbox still works standalone from any CWD,
/// falling back to the reference project's path directly.
fn open_project() -> Option<engine_project::Project> {
    engine_project::Project::open(".").ok()
}

fn resolve_assets_root(project: Option<&engine_project::Project>) -> std::path::PathBuf {
    project
        .map(|project| project.paths.assets_dir())
        .unwrap_or_else(|| std::path::PathBuf::from("examples/reference-project/assets"))
}

struct SandboxModules {
    renderer: RenderModule,
    input: InputModule,
    assets: AssetModule,
}

impl SandboxModules {
    fn new() -> Result<Self> {
        let project = open_project();
        let assets_root = resolve_assets_root(project.as_ref());
        let mut assets = AssetModule::new(assets_root.to_string_lossy().to_string());
        if let Some(project) = &project {
            if let Ok(database) =
                AssetDatabase::open(project.paths.assets_dir(), project.paths.imported_dir())
            {
                assets.asset_server_mut().attach_database(database);
            }
        }
        let _asset_path = assets.load_stub("textures/placeholder.png")?;

        Ok(Self {
            renderer: RenderModule::new(),
            input: InputModule::new(),
            assets,
        })
    }
}

impl EngineModules for SandboxModules {
    fn window_created(&mut self, window: Arc<Window>, window_config: &WindowConfig) -> Result<()> {
        self.renderer
            .initialize_with_window(window, window_config.vsync)
    }

    fn window_event(&mut self, event: &WindowEvent) -> Result<()> {
        self.input.handle_window_event(event)
    }

    fn flush_input(&mut self, world: &mut World) -> Result<()> {
        let Some(mut input_state) = world.get_resource_mut::<InputState>() else {
            return Ok(());
        };

        self.input.pump(&mut input_state)
    }

    fn fixed_update(&mut self, _fixed_dt_seconds: f32) -> Result<()> {
        // Physics simulation is driven by ECS FixedUpdate systems.
        Ok(())
    }

    fn update(&mut self, _delta_seconds: f32) -> Result<()> {
        let reload = self.assets.poll_hot_reload();
        if reload.reloaded_count() > 0 {
            log::info!(
                target: "engine::sandbox",
                "Hot-reloaded {} texture(s), {} mesh(es), {} material(s)",
                reload.textures.len(),
                reload.meshes.len(),
                reload.materials.len()
            );
        }
        Ok(())
    }

    fn render(&mut self, world: &mut World, _alpha: f32) -> Result<()> {
        self.renderer.tick(world, self.assets.asset_server())
    }

    fn resized(&mut self, width: u32, height: u32) -> Result<()> {
        self.renderer.resize(width, height);
        Ok(())
    }
}

struct SandboxBootstrapPlugin;

impl Plugin<SandboxModules> for SandboxBootstrapPlugin {
    fn build(&self, engine: &mut Engine<SandboxModules>) {
        let _identity = engine_math::identity();
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
        engine
            .runtime
            .world
            .resource_mut::<engine_audio::AudioEngine>()
            .set_backend(engine_audio::open_device_backend());
        engine.insert_resource(CameraLookState::default());

        engine.runtime.world.spawn(engine_audio::AudioSource {
            clip: engine_assets::AssetRef::from_path("audio/ambient.wav"),
            bus: "music".to_owned(),
            looping: true,
            autoplay: true,
            spatial: false,
            fade_in: 1.0,
            ..Default::default()
        });

        let render_assets = (|| -> Result<SandboxRenderAssets> {
            let texture = engine
                .modules
                .assets
                .load_texture_handle("textures/placeholder.png")?;
            let mesh = engine.modules.assets.load_mesh_handle("meshes/cube.glb")?;
            let material = engine
                .modules
                .assets
                .load_material_handle("materials/default.ron")?;
            Ok(SandboxRenderAssets {
                mesh,
                texture,
                material,
            })
        })();

        match render_assets {
            Ok(render_assets) => {
                engine.insert_resource(render_assets);
                log::info!(
                    target: "engine::sandbox",
                    "Sandbox render assets loaded (mesh/texture/material handles)"
                );
            }
            Err(error) => {
                log::warn!(
                    target: "engine::sandbox",
                    "Sandbox render assets unavailable: {}",
                    error
                );
            }
        }

        engine
            .add_startup_systems(ecs_startup_smoke)
            .add_startup_systems(ecs_spawn_smoke_entities)
            .add_startup_systems(ecs_spawn_renderable_entity)
            .add_startup_systems(ecs_spawn_physics_scene)
            .add_update_systems(ecs_camera_controller)
            .add_update_systems(ecs_update_smoke);

        log::info!(target: "engine::sandbox", "Engine: {}", engine_core::engine_name());
        log::info!(
            target: "engine::sandbox",
            "Modules: {}, {}, {}, {}, {}, {}",
            engine_math::module_name(),
            engine_render::module_name(),
            engine_physics::module_name(),
            engine_audio::module_name(),
            engine_input::module_name(),
            engine_assets::module_name()
        );
        log::info!(
            target: "engine::sandbox",
            "Backends: render={}, physics2d/3d={:?}, audio={}, input={:?}",
            engine.modules.renderer.backend_type_name(),
            engine_physics::dimensions_supported(),
            engine
                .runtime
                .world
                .resource::<engine_audio::AudioEngine>()
                .backend_name(),
            engine.modules.input.backend_type_names(),
        );
        log::info!(target: "engine::sandbox", "Sandbox bootstrap complete");
    }
}

fn run() -> Result<()> {
    let modules = SandboxModules::new()?;

    let config = EngineConfig::with_app_name("Starman Sandbox").with_window_config(
        WindowConfig::default()
            .with_title("Starman Sandbox")
            .with_size(1280, 720)
            .with_resizable(true)
            .with_vsync(true),
    );

    let mut engine = Engine::new(config, modules)?;
    engine.add_plugin(SandboxBootstrapPlugin);

    engine.run()
}

fn main() {
    engine_core::init_logging();

    if let Err(error) = run() {
        log::error!(target: "engine::sandbox", "Startup failed: {}", error);
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::keyboard_movement_intent;
    use engine_input::InputState;
    use winit::{event::ElementState, keyboard::KeyCode};

    #[test]
    fn keyboard_forward_and_backward_mapping_is_not_inverted() {
        let mut input = InputState::default();

        input.begin_frame();
        input.process_key_input(KeyCode::KeyW, ElementState::Pressed, false);
        let forward_movement = keyboard_movement_intent(&input);
        assert!(forward_movement.z > 0.0);

        input.begin_frame();
        input.process_key_input(KeyCode::KeyW, ElementState::Released, false);
        input.process_key_input(KeyCode::KeyS, ElementState::Pressed, false);
        let backward_movement = keyboard_movement_intent(&input);
        assert!(backward_movement.z < 0.0);
    }
}
