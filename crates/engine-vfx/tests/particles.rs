//! Particle effects through the runtime (CPU backend, headless) and the
//! renderer (GPU compute backend and CPU upload, under validation).

use engine_assets::{AssetRef, AssetServer, Assets};
use engine_core::{Camera3d, GameRuntime, GlobalTransform, PrimaryCamera, Transform};
use engine_math::{Curve, Vec3, Vec4};
use engine_physics::{ColliderShape3D, PhysicsPlugin, RigidBodyType};
use engine_render::{FrameRenderer, FrameRendererConfig, QualityPreset};
use engine_vfx::*;

const DT: f32 = 1.0 / 30.0;

fn fountain(backend: SimulationBackend) -> ParticleEffect {
    let mut emitter = EmitterDef::new("fountain", 2000);
    emitter.backend = backend;
    emitter.spawn.rate = 300.0;
    emitter.spawn.bursts.push(Burst {
        time: 0.0,
        count: 50,
        cycles: 1,
        interval: 0.0,
    });
    emitter.shape = Shape::Cone {
        angle: 0.3,
        radius: 0.1,
    };
    emitter.init.lifetime = Range::new(1.0, 1.5);
    emitter.init.speed = Range::new(3.0, 4.0);
    emitter.init.size = Range::new(0.2, 0.3);
    emitter.init.color = Vec4::new(2.0, 1.0, 0.4, 1.0);
    emitter.modules = vec![
        Module::Gravity(1.0),
        Module::Drag(0.2),
        Module::CurlNoise {
            strength: 0.5,
            frequency: 1.0,
            scroll_speed: 0.2,
        },
        Module::SizeOverLife(Curve::linear([(0.0, 1.0), (1.0, 0.2)])),
        Module::ColorOverLife(Curve::linear([
            (0.0, Vec4::ONE),
            (1.0, Vec4::new(1.0, 1.0, 1.0, 0.0)),
        ])),
    ];
    emitter.render.blend = BlendMode::Alpha;
    emitter.render.sort = true;
    emitter.render.soft_distance = 0.3;
    ParticleEffect {
        emitters: vec![emitter],
        bounds: Some((Vec3::splat(-5.0), Vec3::splat(5.0))),
        ..ParticleEffect::template()
    }
}

fn install(runtime: &mut GameRuntime, assets: &Assets, path: &str, effect: ParticleEffect) {
    let handle = assets.request::<ParticleEffect>(&AssetRef::from_path(path));
    assert!(assets.replace(handle, effect));
    let _ = runtime;
}

fn runtime() -> (GameRuntime, Assets) {
    let assets = Assets::new();
    let mut runtime = GameRuntime::new();
    runtime.insert_resource(assets.clone());
    runtime.add_plugin(PhysicsPlugin);
    runtime.add_plugin(VfxPlugin);
    (runtime, assets)
}

fn emitter(path: &str) -> ParticleEmitter {
    ParticleEmitter {
        effect: AssetRef::from_path(path),
        ..Default::default()
    }
}

#[test]
fn cpu_backend_spawns_simulates_and_obeys_commands() {
    let (mut runtime, assets) = runtime();
    install(
        &mut runtime,
        &assets,
        "fx/fountain.vfx.ron",
        fountain(SimulationBackend::Auto),
    );
    let entity = runtime
        .world
        .spawn((
            emitter("fx/fountain.vfx.ron"),
            Transform::default(),
            GlobalTransform::default(),
        ))
        .id();
    for _ in 0..16 {
        runtime.step(DT);
    }
    let state = runtime.world.get::<ParticleEffectState>(entity).unwrap();
    assert_eq!(
        state.emitters[0].backend,
        ActiveBackend::Cpu,
        "headless: no compute"
    );
    let alive = state.alive();
    // 50 burst + ~0.5 s at 300/s, nothing has died yet.
    assert!((190..=210).contains(&alive), "{alive}");
    let highest = state.emitters[0]
        .cpu
        .particles
        .iter()
        .map(|p| p.position.y)
        .fold(f32::MIN, f32::max);
    assert!(highest > 0.5, "particles fly up: {highest}");

    runtime.world.send_event(ParticleCommand::Burst {
        entity,
        emitter: None,
        count: 100,
    });
    runtime.step(DT);
    let after = runtime
        .world
        .get::<ParticleEffectState>(entity)
        .unwrap()
        .alive();
    assert!(after >= alive + 100, "{alive} -> {after}");

    runtime
        .world
        .get_mut::<ParticleEmitter>(entity)
        .unwrap()
        .emitting = false;
    for _ in 0..60 {
        runtime.step(DT);
    }
    assert_eq!(
        runtime
            .world
            .get::<ParticleEffectState>(entity)
            .unwrap()
            .alive(),
        0,
        "died out"
    );

    runtime.world.send_event(ParticleCommand::Restart(entity));
    runtime
        .world
        .get_mut::<ParticleEmitter>(entity)
        .unwrap()
        .emitting = true;
    runtime.step(DT);
    let state = runtime.world.get::<ParticleEffectState>(entity).unwrap();
    assert!(state.alive() >= 50, "restart replays the burst");
}

#[test]
fn cpu_particles_are_deterministic_per_seed() {
    let run = |seed: u32| {
        let (mut runtime, assets) = runtime();
        install(
            &mut runtime,
            &assets,
            "fx/f.vfx.ron",
            fountain(SimulationBackend::Cpu),
        );
        let entity = runtime
            .world
            .spawn((
                ParticleEmitter {
                    seed,
                    ..emitter("fx/f.vfx.ron")
                },
                Transform::default(),
                GlobalTransform::default(),
            ))
            .id();
        for _ in 0..10 {
            runtime.step(DT);
        }
        runtime
            .world
            .get::<ParticleEffectState>(entity)
            .unwrap()
            .emitters[0]
            .cpu
            .particles
            .clone()
    };
    assert_eq!(run(1), run(1));
    assert_ne!(run(1), run(2));
}

#[test]
fn one_shot_effects_with_sub_emitters_finish_and_despawn() {
    let (mut runtime, assets) = runtime();
    let mut rocket = EmitterDef::new("rocket", 8);
    rocket.spawn.bursts.push(Burst {
        time: 0.0,
        count: 2,
        cycles: 1,
        interval: 0.0,
    });
    rocket.init.lifetime = Range::constant(0.3);
    rocket.sub_emitters.push(SubEmitter {
        trigger: SubEmitterTrigger::Death,
        emitter: "sparks".into(),
        count: 20,
        inherit_velocity: 0.5,
    });
    let mut sparks = EmitterDef::new("sparks", 100);
    sparks.sub_emitter_only = true;
    sparks.shape = Shape::Sphere {
        radius: 0.1,
        surface: true,
    };
    sparks.init.lifetime = Range::constant(0.3);
    let effect = ParticleEffect {
        emitters: vec![rocket, sparks],
        duration: 0.1,
        looping: false,
        ..ParticleEffect::template()
    };
    effect.validate().unwrap();
    install(&mut runtime, &assets, "fx/rocket.vfx.ron", effect);
    let entity = runtime
        .world
        .spawn((
            ParticleEmitter {
                despawn_when_finished: true,
                ..emitter("fx/rocket.vfx.ron")
            },
            Transform::default(),
            GlobalTransform::default(),
        ))
        .id();
    let mut max_sparks = 0;
    for _ in 0..40 {
        runtime.step(DT);
        if let Some(state) = runtime.world.get::<ParticleEffectState>(entity) {
            max_sparks = max_sparks.max(state.emitters.get(1).map_or(0, |e| e.alive));
        }
    }
    assert_eq!(max_sparks, 40, "two rockets burst into twenty sparks each");
    assert!(
        runtime.world.get_entity(entity).is_err(),
        "despawned when finished"
    );
}

#[test]
fn cpu_collision_uses_physics_colliders() {
    let (mut runtime, assets) = runtime();
    let mut rain = EmitterDef::new("rain", 64);
    rain.backend = SimulationBackend::Cpu;
    rain.spawn.bursts.push(Burst {
        time: 0.0,
        count: 32,
        cycles: 1,
        interval: 0.0,
    });
    rain.shape = Shape::Box {
        half_extents: Vec3::new(1.0, 0.0, 1.0),
    };
    rain.init.speed = Range::constant(0.0);
    rain.init.lifetime = Range::constant(5.0);
    rain.modules = vec![
        Module::Gravity(1.0),
        Module::Collision {
            bounce: 0.0,
            friction: 1.0,
            kill: false,
        },
    ];
    install(
        &mut runtime,
        &assets,
        "fx/rain.vfx.ron",
        ParticleEffect {
            emitters: vec![rain],
            ..ParticleEffect::template()
        },
    );
    runtime.world.spawn((
        RigidBodyType::Static,
        ColliderShape3D::Box {
            half_extents: Vec3::new(5.0, 0.5, 5.0),
        },
        Transform::from_xyz(0.0, -0.5, 0.0),
    ));
    let transform = Transform::from_xyz(0.0, 2.0, 0.0);
    let entity = runtime
        .world
        .spawn((
            emitter("fx/rain.vfx.ron"),
            GlobalTransform(transform.to_affine()),
            transform,
        ))
        .id();
    for _ in 0..60 {
        runtime.step(DT);
    }
    let state = runtime.world.get::<ParticleEffectState>(entity).unwrap();
    assert_eq!(state.alive(), 32);
    for particle in &state.emitters[0].cpu.particles {
        assert!(
            particle.position.y > -0.05 && particle.position.y < 0.1,
            "resting on the floor: {particle:?}"
        );
    }
}

fn gpu() -> Option<FrameRenderer> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))?;
    let tier = engine_render::negotiate_tier(&adapter);
    let info = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("vfx-test"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: Default::default(),
        },
        None,
    ))
    .ok()?;
    let caps = engine_render::NegotiatedCapabilities {
        tier,
        adapter_name: info.name,
        backend: format!("{:?}", info.backend),
        features: adapter.features(),
        limits: adapter.limits(),
    };
    Some(
        FrameRenderer::try_new(
            device,
            queue,
            caps,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            160,
            120,
            FrameRendererConfig {
                quality: QualityPreset::Medium,
                shader_cache: std::env::temp_dir().join("starman-vfx-test-shader-cache"),
                ..Default::default()
            },
        )
        .expect("renderer"),
    )
}

#[test]
fn gpu_and_cpu_emitters_render_without_validation_errors() {
    let Some(mut frame) = gpu() else {
        eprintln!("skipping GPU test: no adapter");
        return;
    };
    let root = std::env::temp_dir().join(format!(
        "starman-vfx-gpu-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let mut server = AssetServer::new(root.to_string_lossy().to_string());
    let assets = server.assets().clone();
    let reference = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/reference-project/assets");
    std::fs::copy(reference.join("meshes/cube.glb"), root.join("cube.glb")).unwrap();
    std::fs::copy(
        reference.join("textures/placeholder.png"),
        root.join("smoke.png"),
    )
    .unwrap();
    let mut runtime = GameRuntime::new();
    runtime.insert_resource(assets.clone());
    runtime.add_plugin(engine_assets::AssetsPlugin);
    runtime.add_plugin(PhysicsPlugin);
    runtime.add_plugin(VfxPlugin);
    let mut debris = fountain(SimulationBackend::Gpu);
    debris.emitters[0].render.mode = RenderMode::Mesh {
        mesh: AssetRef::from_path("cube.glb"),
    };
    debris.emitters[0].render.sort = false;
    debris.emitters[0].init.size = Range::new(0.05, 0.1);
    let mut smoke = fountain(SimulationBackend::Cpu);
    smoke.emitters[0].render.mode = RenderMode::Stretched { length_scale: 0.1 };
    smoke.emitters[0].render.texture = Some(AssetRef::from_path("smoke.png"));
    smoke.emitters[0].render.flipbook = Some(Flipbook {
        columns: 2,
        rows: 2,
        fps: 0.0,
    });
    smoke.emitters[0].render.blend = BlendMode::Premultiplied;
    smoke.emitters[0].modules.push(Module::Collision {
        bounce: 0.3,
        friction: 0.1,
        kill: false,
    });
    install(&mut runtime, &assets, "fx/debris.vfx.ron", debris);
    install(&mut runtime, &assets, "fx/smoke.vfx.ron", smoke);
    let mut additive = fountain(SimulationBackend::Gpu);
    additive.emitters[0].render.blend = BlendMode::Additive;
    install(
        &mut runtime,
        &assets,
        "fx/gpu.vfx.ron",
        fountain(SimulationBackend::Gpu),
    );
    install(&mut runtime, &assets, "fx/add.vfx.ron", additive);
    install(
        &mut runtime,
        &assets,
        "fx/cpu.vfx.ron",
        fountain(SimulationBackend::Cpu),
    );

    let camera = Transform::from_xyz(0.0, 1.5, 6.0).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y);
    runtime.world.spawn((
        GlobalTransform(camera.to_affine()),
        camera,
        Camera3d::default(),
        PrimaryCamera,
    ));
    let mut spawn = |path: &str, x: f32| {
        let t = Transform::from_xyz(x, 0.0, 0.0);
        runtime
            .world
            .spawn((emitter(path), GlobalTransform(t.to_affine()), t))
            .id()
    };
    let gpu_entity = spawn("fx/gpu.vfx.ron", -1.5);
    spawn("fx/add.vfx.ron", 0.0);
    spawn("fx/cpu.vfx.ron", 1.5);
    spawn("fx/debris.vfx.ron", -1.5);
    spawn("fx/smoke.vfx.ron", 1.5);

    let mut image = Vec::new();
    let mut empty = None;
    for i in 0..12 {
        runtime.step(DT);
        server.update_blocking();
        if std::env::var("VFX_DEBUG").is_err() {
            frame.device.push_error_scope(wgpu::ErrorFilter::Validation);
        }
        image = frame
            .render_to_image(&mut runtime.world, &server)
            .expect("frame renders");
        if std::env::var("VFX_DEBUG").is_err() {
            let error = pollster::block_on(frame.device.pop_error_scope());
            assert!(error.is_none(), "GPU validation error: {error:?}");
        }
        if i == 0 {
            empty = Some(image.clone());
        }
    }
    if let Ok(path) = std::env::var("STARMAN_VFX_DUMP") {
        image::save_buffer(&path, &image, 160, 120, image::ColorType::Rgba8).unwrap();
    }
    let state = runtime
        .world
        .get::<ParticleEffectState>(gpu_entity)
        .unwrap();
    assert_eq!(
        state.emitters[0].backend,
        ActiveBackend::Gpu,
        "compute available after the first frame"
    );
    let stats = frame.stats().clone();
    let stat = |name: &str| {
        stats
            .extensions
            .iter()
            .find(|(n, _)| n.ends_with(name))
            .map(|(_, v)| *v)
            .unwrap_or(-1.0)
    };
    assert_eq!(stat("particle_emitters"), 5.0, "{:?}", stats.extensions);
    assert!(stat("particles_cpu") > 50.0, "{:?}", stats.extensions);
    assert!(stat("particle_gpu_steps") >= 2.0, "{:?}", stats.extensions);
    assert_eq!(stat("particle_draws"), 5.0, "{:?}", stats.extensions);
    // Each third of the frame (left: GPU alpha + GPU meshes, middle: GPU
    // additive, right: CPU textured flipbook) differs clearly from the
    // particle-free first frame.
    let empty = empty.unwrap();
    for third in 0..3 {
        let changed = empty
            .chunks_exact(4)
            .zip(image.chunks_exact(4))
            .enumerate()
            .filter(|(i, (a, b))| {
                (i % 160) / 54 == third && (0..3).any(|c| (a[c] as i32 - b[c] as i32).abs() > 30)
            })
            .count();
        assert!(changed > 20, "fountain {third} is visible: {changed}");
    }
    let _ = std::fs::remove_dir_all(&root);
}
