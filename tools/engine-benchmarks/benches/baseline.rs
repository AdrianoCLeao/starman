use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use bevy_ecs::system::RunSystemOnce;
use bevy_ecs::world::World;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use engine_assets::{AssetDatabase, AssetServer, SceneDeserializer};
use engine_core::{
    create_world, propagate_transforms, register_core_reflection_types, Children, EntityName,
    Parent, SpatialBundle,
};
use engine_reflect::{ComponentRegistry, ReflectMetadataRegistry, ReflectTypeRegistry};
use engine_render::RenderSceneAdapter;

fn spawn_entities(world: &mut World, count: usize) {
    for index in 0..count {
        world.spawn((
            EntityName::new(format!("Entity {index}")),
            SpatialBundle::default(),
        ));
    }
}

fn spawn_transform_chain(world: &mut World, depth: usize) {
    let root = world
        .spawn((
            EntityName::new("Root"),
            SpatialBundle::default(),
            Children::default(),
        ))
        .id();
    let mut parent = root;

    for index in 0..depth {
        let child = world
            .spawn((
                EntityName::new(format!("Child {index}")),
                SpatialBundle {
                    transform: engine_core::Transform::from_xyz(0.0, 1.0, 0.0),
                    ..SpatialBundle::default()
                },
                Parent(parent),
                Children::default(),
            ))
            .id();

        world
            .get_mut::<Children>(parent)
            .expect("parent should have children")
            .0
            .push(child);
        parent = child;
    }
}

fn bench_world_spawn(c: &mut Criterion) {
    c.bench_function("world_spawn_1000", |b| {
        b.iter(|| {
            let mut world = engine_core::create_world();
            spawn_entities(&mut world, 1000);
            world.iter_entities().count()
        });
    });
}

fn bench_transform_propagation(c: &mut Criterion) {
    c.bench_function("transform_propagation_chain_1000", |b| {
        b.iter_batched(
            || {
                let mut world = engine_core::create_world();
                spawn_transform_chain(&mut world, 1000);
                world
            },
            |mut world| {
                let _ = world.run_system_once(propagate_transforms);
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

fn bench_asset_lookup(c: &mut Criterion) {
    c.bench_function("asset_texture_load_cached_placeholder", |b| {
        let mut assets = AssetServer::new(reference_assets_root().to_string_lossy().to_string());
        let _ = assets.load_texture_handle("textures/placeholder.png");

        b.iter(|| assets.load_texture_handle("textures/placeholder.png"));
    });
}

// -- Reference-project-backed benchmarks ---------------------------------
//
// These run against a scratch *copy* of `examples/reference-project/assets`
// (never the checked-in directory itself — importing writes `.meta.ron`
// sidecars next to every source file as a side effect, which must not land
// in the repo).

fn reference_assets_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("examples")
        .join("reference-project")
        .join("assets")
}

fn scratch_dir(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be valid")
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{nanos}"))
}

fn copy_dir_recursive(source: &Path, destination: &Path) {
    std::fs::create_dir_all(destination).expect("destination dir should be created");
    for entry in std::fs::read_dir(source).expect("source dir should be readable") {
        let entry = entry.expect("entry should be readable");
        let path = entry.path();
        let dest_path = destination.join(entry.file_name());
        if path.is_dir() {
            copy_dir_recursive(&path, &dest_path);
        } else {
            std::fs::copy(&path, &dest_path).expect("file should copy");
        }
    }
}

/// Quantifies the "tempo de import incremental" indicator from the
/// ROADMAP: a cold `import_all` over the reference project's real assets
/// vs. a warm one where nothing changed.
fn bench_import_incremental(c: &mut Criterion) {
    let mut group = c.benchmark_group("asset_import");

    group.bench_function("cold_import_all", |b| {
        b.iter_batched(
            || {
                let root = scratch_dir("starman-bench-import-cold");
                let assets_root = root.join("assets");
                copy_dir_recursive(&reference_assets_root(), &assets_root);
                (root, assets_root)
            },
            |(root, assets_root)| {
                let mut database =
                    AssetDatabase::open(&assets_root, root.join(".starman/cache/imported"))
                        .expect("database should open");
                let summary = database.import_all().expect("import should succeed");
                assert!(summary.is_success());
                let _ = std::fs::remove_dir_all(&root);
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("warm_import_all_unchanged", |b| {
        let root = scratch_dir("starman-bench-import-warm");
        let assets_root = root.join("assets");
        copy_dir_recursive(&reference_assets_root(), &assets_root);
        let mut database = AssetDatabase::open(&assets_root, root.join(".starman/cache/imported"))
            .expect("database should open");
        database
            .import_all()
            .expect("warm-up import should succeed");

        b.iter(|| {
            database.rescan().expect("rescan should succeed");
            let summary = database.import_all().expect("import should succeed");
            debug_assert!(summary.imported.is_empty());
        });

        let _ = std::fs::remove_dir_all(&root);
    });

    group.finish();
}

fn build_registries() -> (
    ReflectTypeRegistry,
    ComponentRegistry,
    ReflectMetadataRegistry,
) {
    let mut type_registry = ReflectTypeRegistry::default();
    let mut component_registry = ComponentRegistry::default();
    let mut metadata_registry = ReflectMetadataRegistry::default();
    register_core_reflection_types(
        &mut type_registry,
        &mut component_registry,
        &mut metadata_registry,
    );
    (type_registry, component_registry, metadata_registry)
}

/// Quantifies the cost `AssetServer::ensure_source_id` (Fatia 6.1) adds to
/// loading the reference scene: with a database attached, every asset
/// reference the scene resolves also gets hashed and looked up in the
/// asset database, on top of the plain path-based load.
fn bench_scene_load_with_database(c: &mut Criterion) {
    let mut group = c.benchmark_group("scene_load");
    let (type_registry, component_registry, _) = build_registries();
    let adapter = RenderSceneAdapter;
    let scene_path = reference_assets_root().join("scenes/level.scene.ron");

    group.bench_function("without_database", |b| {
        b.iter_batched(
            || {
                let world = create_world();
                let assets =
                    AssetServer::new(reference_assets_root().to_string_lossy().to_string());
                (world, assets)
            },
            |(mut world, mut assets)| {
                let mut deserializer = SceneDeserializer::new(
                    &mut world,
                    &component_registry,
                    &type_registry,
                    &mut assets,
                )
                .with_external_components(&adapter);
                let _ = deserializer.load_file(&scene_path);
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("with_database", |b| {
        b.iter_batched(
            || {
                let root = scratch_dir("starman-bench-scene-load-db");
                let assets_root = root.join("assets");
                copy_dir_recursive(&reference_assets_root(), &assets_root);
                let database =
                    AssetDatabase::open(&assets_root, root.join(".starman/cache/imported"))
                        .expect("database should open");
                let world = create_world();
                let mut assets = AssetServer::new(assets_root.to_string_lossy().to_string());
                assets.attach_database(database);
                (root, world, assets)
            },
            |(root, mut world, mut assets)| {
                let scene_path = root.join("assets/scenes/level.scene.ron");
                let mut deserializer = SceneDeserializer::new(
                    &mut world,
                    &component_registry,
                    &type_registry,
                    &mut assets,
                )
                .with_external_components(&adapter);
                let _ = deserializer.load_file(&scene_path);
                let _ = std::fs::remove_dir_all(&root);
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_host_set_get_field(c: &mut Criterion) {
    c.bench_function("host_set_get_field_1000", |b| {
        b.iter_batched(
            || {
                let mut world = create_world();
                let mut components = ComponentRegistry::default();
                let mut types = ReflectTypeRegistry::default();
                let mut meta = ReflectMetadataRegistry::default();
                register_core_reflection_types(&mut types, &mut components, &mut meta);
                let guard = engine_plugin::PermissionGuard::new(
                    "/tmp/bench",
                    engine_plugin::ProjectPermissions::default(),
                );
                let mut bus = engine_plugin::HostBus::new(guard);
                let handle = bus
                    .execute(
                        &mut world,
                        &components,
                        engine_plugin::HostCommand::Spawn { name: None },
                    )
                    .unwrap();
                let id = handle.as_u64().unwrap();
                (world, components, types, bus, id)
            },
            |(mut world, components, types, mut bus, id)| {
                for i in 0..1000 {
                    let y = (i as f32) * 0.001;
                    bus.execute(
                        &mut world,
                        &components,
                        engine_plugin::HostCommand::SetField {
                            entity: engine_plugin::EntityHandle(id),
                            component: "Transform".into(),
                            field_path: "translation.y".into(),
                            value: serde_json::json!(y),
                        },
                    )
                    .unwrap();
                    let _ = bus
                        .query(
                            &world,
                            &components,
                            &types,
                            engine_plugin::HostQuery::GetField {
                                entity: engine_plugin::EntityHandle(id),
                                component: "Transform".into(),
                                field_path: "translation.y".into(),
                            },
                        )
                        .unwrap();
                }
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(
    baseline,
    bench_world_spawn,
    bench_transform_propagation,
    bench_asset_lookup,
    bench_import_incremental,
    bench_scene_load_with_database,
    bench_host_set_get_field,
    bench_frustum_and_clusters
);
criterion_main!(baseline);

fn bench_frustum_and_clusters(c: &mut Criterion) {
    use engine_math::{Mat4, Vec3};
    use engine_render::{cull_lights_cpu, Aabb, CapabilityTier, Frustum, GpuLight};

    c.bench_function("frustum_aabb_2000", |b| {
        let frustum = Frustum::from_view_proj(Mat4::perspective_rh(1.0, 1.6, 0.1, 500.0));
        let aabbs: Vec<Aabb> = (0..2000)
            .map(|i| {
                let x = (i % 50) as f32 * 2.0;
                let z = (i / 50) as f32 * 2.0;
                Aabb::from_center_extents(Vec3::new(x, 0.0, z), Vec3::splat(0.5))
            })
            .collect();
        b.iter(|| {
            aabbs
                .iter()
                .filter(|a| frustum.intersects_aabb(**a))
                .count()
        });
    });

    c.bench_function("cluster_cull_64_lights", |b| {
        let lights: Vec<GpuLight> = (0..64)
            .map(|i| GpuLight {
                position_range: [i as f32, 1.0, i as f32, 8.0],
                color_intensity: [1.0, 1.0, 1.0, 1.0],
                direction_cone: [0.0, -1.0, 0.0, 0.0],
                light_type: if i == 0 {
                    GpuLight::TYPE_DIRECTIONAL
                } else {
                    GpuLight::TYPE_POINT
                },
                _pad: [0; 3],
            })
            .collect();
        b.iter(|| cull_lights_cpu(&lights, [0.0; 3], CapabilityTier::Tier1).cluster_count());
    });
}
