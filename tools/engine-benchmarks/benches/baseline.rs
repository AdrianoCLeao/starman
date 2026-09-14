use bevy_ecs::system::RunSystemOnce;
use bevy_ecs::world::World;
use criterion::{criterion_group, criterion_main, Criterion};
use engine_core::{propagate_transforms, Children, EntityName, Parent, SpatialBundle};

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
        let mut assets = engine_assets::AssetServer::new("assets");
        let _ = assets.load_texture_handle("textures/placeholder.png");

        b.iter(|| assets.load_texture_handle("textures/placeholder.png"));
    });
}

criterion_group!(
    baseline,
    bench_world_spawn,
    bench_transform_propagation,
    bench_asset_lookup
);
criterion_main!(baseline);
