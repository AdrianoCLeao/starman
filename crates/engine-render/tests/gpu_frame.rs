//! End-to-end GPU frame tests. They run on any available adapter
//! (including software rasterizers such as lavapipe/WARP) and are skipped
//! with a message when the machine has none.

use std::path::{Path, PathBuf};

use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;
use engine_assets::AssetServer;
use engine_core::{
    Camera3d, DebugDraw, GlobalTransform, PrimaryCamera, RenderLayer3D, Transform, Visible,
};
use engine_math::{Quat, Vec3};
use engine_render::{
    CameraRenderSettings, DebugView, DirectionalLight, Environment, FrameRenderer,
    FrameRendererConfig, MeshRenderable3d, PointLight, QualityPreset, ReflectionProbe, SpotLight,
};

fn device() -> Option<(
    wgpu::Device,
    wgpu::Queue,
    engine_render::NegotiatedCapabilities,
)> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .or_else(|| {
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: true,
        }))
    })?;
    let (device, queue, caps) = engine_render_test_device(&adapter)?;
    Some((device, queue, caps))
}

fn engine_render_test_device(
    adapter: &wgpu::Adapter,
) -> Option<(
    wgpu::Device,
    wgpu::Queue,
    engine_render::NegotiatedCapabilities,
)> {
    let tier = engine_render::negotiate_tier(adapter);
    let info = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("gpu-frame-test"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: Default::default(),
        },
        None,
    ))
    .ok()?;
    Some((
        device,
        queue,
        engine_render::NegotiatedCapabilities {
            tier,
            adapter_name: info.name,
            backend: format!("{:?}", info.backend),
            features: adapter.features(),
            limits: adapter.limits(),
        },
    ))
}

struct Scene {
    _dir: TempDir,
    server: AssetServer,
    world: World,
    center: Entity,
}

struct TempDir(PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn reference_assets() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/reference-project/assets")
}

fn write(path: &Path, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

fn build_scene(tag: &str) -> Scene {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!("starman-gpu-{tag}-{nanos}"));
    std::fs::create_dir_all(root.join("meshes")).unwrap();
    std::fs::copy(
        reference_assets().join("meshes/cube.glb"),
        root.join("meshes/cube.glb"),
    )
    .unwrap();
    std::fs::create_dir_all(root.join("textures")).unwrap();
    std::fs::copy(
        reference_assets().join("textures/placeholder.png"),
        root.join("textures/placeholder.png"),
    )
    .unwrap();
    write(
        &root.join("materials/opaque.ron"),
        "(base_color_factor: [0.8, 0.6, 0.4, 1.0], metallic: 0.0, roughness: 0.6)",
    );
    write(
        &root.join("materials/metal.ron"),
        "(base_color_factor: [0.9, 0.9, 0.9, 1.0], metallic: 1.0, roughness: 0.2)",
    );
    write(
        &root.join("materials/glass.ron"),
        "(base_color_factor: [0.3, 0.6, 1.0, 0.4], metallic: 0.0, roughness: 0.1, alpha_mode: \"BLEND\")",
    );
    write(
        &root.join("materials/cutout.ron"),
        "(base_color_factor: [0.2, 0.8, 0.2, 0.3], metallic: 0.0, roughness: 0.8, alpha_mode: \"MASK\", alpha_cutoff: 0.5, base_color_texture: Some(\"textures/placeholder.png\"))",
    );

    let mut server = AssetServer::new(root.to_string_lossy().to_string());
    let assets = server.assets().clone();
    assets.register_loader(engine_assets::TextureLoader);
    let mesh = server.load_mesh_handle("meshes/cube.glb").unwrap();
    let texture = server
        .load_texture_handle("textures/placeholder.png")
        .unwrap();
    let opaque = server.load_material_handle("materials/opaque.ron").unwrap();
    let metal = server.load_material_handle("materials/metal.ron").unwrap();
    let glass = server.load_material_handle("materials/glass.ron").unwrap();
    let cutout = server.load_material_handle("materials/cutout.ron").unwrap();

    let mut world = World::new();
    let spawn_mesh = |world: &mut World, position: Vec3, scale: Vec3, material| {
        let transform = Transform::from_translation(position).with_scale(scale);
        world
            .spawn((
                GlobalTransform(transform.to_affine()),
                transform,
                MeshRenderable3d::new(mesh, texture, material),
                Visible,
                RenderLayer3D,
            ))
            .id()
    };
    // Ground, a grid of cubes, a transparent and an alpha-tested cube.
    spawn_mesh(
        &mut world,
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(20.0, 0.2, 20.0),
        opaque,
    );
    for x in -2..=2 {
        for z in -2..=2 {
            let material = if (x + z) % 2 == 0 { opaque } else { metal };
            spawn_mesh(
                &mut world,
                Vec3::new(x as f32 * 2.0, 0.0, z as f32 * 2.0 - 4.0),
                Vec3::ONE,
                material,
            );
        }
    }
    let center = spawn_mesh(
        &mut world,
        Vec3::new(0.0, 1.5, 2.0),
        Vec3::splat(1.2),
        opaque,
    );
    spawn_mesh(&mut world, Vec3::new(2.0, 0.5, 2.0), Vec3::ONE, glass);
    spawn_mesh(&mut world, Vec3::new(-2.0, 0.5, 2.0), Vec3::ONE, cutout);

    let camera = Transform::from_xyz(0.0, 2.0, 8.0).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y);
    world.spawn((
        GlobalTransform(camera.to_affine()),
        camera,
        Camera3d::default(),
        PrimaryCamera,
        CameraRenderSettings {
            fog_enabled: true,
            ..Default::default()
        },
    ));
    let sun = Transform::IDENTITY;
    world.spawn((
        GlobalTransform(sun.to_affine()),
        sun,
        DirectionalLight::default(),
    ));
    let lamp = Transform::from_xyz(1.0, 2.0, 1.0);
    world.spawn((
        GlobalTransform(lamp.to_affine()),
        lamp,
        PointLight {
            color: [1.0, 0.6, 0.3],
            intensity: 20.0,
            range: 6.0,
            cast_shadows: true,
        },
    ));
    let spot = Transform::from_xyz(-2.0, 3.0, 0.0).with_rotation(Quat::IDENTITY);
    world.spawn((
        GlobalTransform(spot.to_affine()),
        spot,
        SpotLight {
            intensity: 30.0,
            cast_shadows: true,
            ..Default::default()
        },
    ));
    world.spawn((
        GlobalTransform::default(),
        Transform::IDENTITY,
        Environment::default(),
    ));
    let probe = Transform::from_xyz(0.0, 1.0, 0.0);
    world.spawn((
        GlobalTransform(probe.to_affine()),
        probe,
        ReflectionProbe::default(),
    ));
    let mut debug = DebugDraw::default();
    debug.aabb(Vec3::splat(-1.0), Vec3::splat(1.0), [1.0, 0.0, 0.0, 1.0]);
    world.insert_resource(debug);

    Scene {
        _dir: TempDir(root),
        server,
        world,
        center,
    }
}

fn renderer(preset: QualityPreset) -> Option<FrameRenderer> {
    let (device, queue, caps) = device()?;
    Some(
        FrameRenderer::try_new(
            device,
            queue,
            caps,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            160,
            120,
            FrameRendererConfig {
                quality: preset,
                shader_cache: std::env::temp_dir().join("starman-gpu-test-shader-cache"),
                ..Default::default()
            },
        )
        .expect("renderer initializes"),
    )
}

fn render_frames(frame: &mut FrameRenderer, scene: &mut Scene, frames: usize) -> Vec<u8> {
    let mut image = Vec::new();
    for _ in 0..frames {
        scene.server.update_blocking();
        frame.device.push_error_scope(wgpu::ErrorFilter::Validation);
        image = frame
            .render_to_image(&mut scene.world, &scene.server)
            .expect("frame renders");
        let error = pollster::block_on(frame.device.pop_error_scope());
        assert!(error.is_none(), "GPU validation error: {error:?}");
    }
    image
}

fn distinct_colors(image: &[u8]) -> usize {
    let mut colors: Vec<[u8; 3]> = image
        .chunks_exact(4)
        .map(|p| [p[0] / 16, p[1] / 16, p[2] / 16])
        .collect();
    colors.sort();
    colors.dedup();
    colors.len()
}

#[test]
fn every_quality_preset_renders_the_lit_scene_without_validation_errors() {
    for preset in QualityPreset::all() {
        let Some(mut frame) = renderer(preset) else {
            eprintln!("skipping GPU test: no adapter");
            return;
        };
        let mut scene = build_scene(preset.as_str());
        let image = render_frames(&mut frame, &mut scene, 3);
        assert_eq!(image.len(), 160 * 120 * 4);
        if let Ok(dir) = std::env::var("STARMAN_GPU_TEST_DUMP") {
            let path = Path::new(&dir).join(format!("frame-{}.png", preset.as_str()));
            image::save_buffer(&path, &image, 160, 120, image::ColorType::Rgba8).unwrap();
        }
        assert!(distinct_colors(&image) > 20, "{preset:?}: image looks flat");
        let stats = frame.stats().clone();
        assert!(stats.visible_meshes >= 25, "{preset:?}: {stats:?}");
        assert!(stats.draw_calls > 0);
        assert_eq!(stats.transparent_meshes, 1, "{preset:?}: {stats:?}");
        assert_eq!(stats.environment_bakes, 1, "environment is baked once");
        if frame.quality().cascade_count > 0 {
            assert!(stats.cascades > 0, "{preset:?}: {stats:?}");
            assert!(stats.shadow_casters > 0);
        }
        if frame.quality().local_shadow_slots > 0 {
            assert!(stats.local_shadow_slots >= 1, "{preset:?}: {stats:?}");
        }
        if frame.quality().max_probes > 0 {
            assert_eq!(stats.probe_bakes, 1, "{preset:?}: probe baked once");
            assert_eq!(stats.probes_active, 1);
        }
        assert_eq!(stats.debug_lines, 0, "debug lines are consumed");
    }
}

#[test]
fn debug_views_render() {
    let Some(mut frame) = renderer(QualityPreset::High) else {
        eprintln!("skipping GPU test: no adapter");
        return;
    };
    let mut scene = build_scene("debug-views");
    for view in DebugView::all() {
        frame.set_debug_view(view);
        let image = render_frames(&mut frame, &mut scene, 1);
        // A static scene has zero motion everywhere, so the velocity view
        // is legitimately uniform.
        if view != DebugView::Velocity {
            assert!(
                distinct_colors(&image) > 1,
                "{view:?} produced a flat image"
            );
        }
    }
}

#[test]
fn gpu_picking_finds_the_entity_under_the_cursor() {
    let Some(mut frame) = renderer(QualityPreset::Medium) else {
        eprintln!("skipping GPU test: no adapter");
        return;
    };
    let mut scene = build_scene("picking");
    render_frames(&mut frame, &mut scene, 1);
    // Project the center cube to find its pixel.
    let camera = Transform::from_xyz(0.0, 2.0, 8.0).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y);
    let view = camera.to_affine().inverse();
    let proj =
        engine_math::Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 160.0 / 120.0, 0.1, 1000.0);
    let clip = proj * engine_math::Mat4::from(view) * Vec3::new(0.0, 1.5, 2.0).extend(1.0);
    let ndc = clip.truncate() / clip.w;
    let x = ((ndc.x * 0.5 + 0.5) * 160.0) as u32;
    let y = ((0.5 - ndc.y * 0.5) * 120.0) as u32;
    frame.request_pick(x, y);
    let mut result = None;
    for _ in 0..10 {
        render_frames(&mut frame, &mut scene, 1);
        let _ = frame.device.poll(wgpu::Maintain::Wait);
        if let Some(pick) = frame.poll_pick() {
            result = Some(pick);
            break;
        }
    }
    let result = result.expect("pick completes within a few frames");
    assert_eq!(result.entity, Some(scene.center), "{result:?}");
}

/// Writes reference renders for visual inspection:
/// `STARMAN_GPU_TEST_DUMP=dir cargo test -p engine-render --test gpu_frame -- --ignored`.
#[test]
#[ignore = "manual visual inspection helper"]
fn dump_reference_renders() {
    let Ok(dir) = std::env::var("STARMAN_GPU_TEST_DUMP") else {
        return;
    };
    let Some((device, queue, caps)) = device() else {
        return;
    };
    let mut frame = FrameRenderer::try_new(
        device,
        queue,
        caps,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        480,
        320,
        FrameRendererConfig {
            quality: QualityPreset::High,
            ..Default::default()
        },
    )
    .unwrap();
    let mut scene = build_scene("reference");
    // Look from the front-right so top, front and side faces show.
    for (name, eye) in [
        ("front", Vec3::new(4.0, 4.0, 9.0)),
        ("low", Vec3::new(-6.0, 1.0, 6.0)),
    ] {
        let camera = Transform::from_translation(eye).looking_at(Vec3::new(0.0, 0.5, 0.0), Vec3::Y);
        let mut query = scene
            .world
            .query::<(&mut Transform, &mut GlobalTransform, &Camera3d)>();
        for (mut transform, mut global, _) in query.iter_mut(&mut scene.world) {
            *transform = camera.clone();
            global.0 = camera.to_affine();
        }
        let image = render_frames(&mut frame, &mut scene, 4);
        let path = Path::new(&dir).join(format!("reference-{name}.png"));
        image::save_buffer(&path, &image, 480, 320, image::ColorType::Rgba8).unwrap();
    }
}
