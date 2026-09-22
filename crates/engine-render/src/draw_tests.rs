use bevy_ecs::prelude::World;
use engine_assets::AssetServer;
use engine_core::{GlobalTransform, RenderLayer2D, RenderLayer3D, Visible};
use engine_math::{glam::Affine3A, Vec3};
use std::path::PathBuf;

use crate::{
    draw::{
        build_contiguous_ranges_by_key, build_draw_batches_2d, collect_draw_items_2d,
        collect_draw_items_3d, DrawItem2d,
    },
    MeshRenderable3d, SpriteRenderable2d,
};

fn assets_root() -> String {
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    root.push("..");
    root.push("..");
    root.push("examples");
    root.push("reference-project");
    root.push("assets");
    root.to_string_lossy().into_owned()
}

#[test]
fn collect_draw_items_3d_respects_hard_limit() {
    let mut asset_server = AssetServer::new(assets_root());
    let mesh = asset_server
        .load_mesh_handle("meshes/cube.glb")
        .expect("load test mesh");
    let texture = asset_server
        .load_texture_handle("textures/placeholder.png")
        .expect("load test texture");
    let material = asset_server
        .load_material_handle("materials/default.ron")
        .expect("load test material");

    let mut world = World::new();
    world.spawn((
        GlobalTransform(Affine3A::from_translation(Vec3::new(1.0, 0.0, 0.0))),
        MeshRenderable3d::new(mesh, texture, material),
        Visible,
        RenderLayer3D,
    ));
    world.spawn((
        GlobalTransform(Affine3A::from_translation(Vec3::new(2.0, 0.0, 0.0))),
        MeshRenderable3d::new(mesh, texture, material),
        Visible,
        RenderLayer3D,
    ));
    world.spawn((
        GlobalTransform(Affine3A::from_translation(Vec3::new(3.0, 0.0, 0.0))),
        MeshRenderable3d::new(mesh, texture, material),
        Visible,
        RenderLayer3D,
    ));

    let draw_items = collect_draw_items_3d(&mut world, 2);

    assert_eq!(draw_items.len(), 2);
}

#[test]
fn collect_draw_items_2d_respects_hard_limit_and_sorts_by_depth() {
    let mut asset_server = AssetServer::new(assets_root());
    let texture = asset_server
        .load_texture_handle("textures/placeholder.png")
        .expect("load test texture");

    let mut world = World::new();
    world.spawn((
        GlobalTransform(Affine3A::from_translation(Vec3::new(0.0, 0.0, 2.0))),
        SpriteRenderable2d::new(texture),
        Visible,
        RenderLayer2D,
    ));
    world.spawn((
        GlobalTransform(Affine3A::from_translation(Vec3::new(0.0, 0.0, -1.0))),
        SpriteRenderable2d::new(texture),
        Visible,
        RenderLayer2D,
    ));
    world.spawn((
        GlobalTransform(Affine3A::from_translation(Vec3::new(0.0, 0.0, 1.0))),
        SpriteRenderable2d::new(texture),
        Visible,
        RenderLayer2D,
    ));

    let draw_items = collect_draw_items_2d(&mut world, 2);

    assert_eq!(draw_items.len(), 2);
    assert_eq!(draw_items[0].sort_z, -1.0);
    assert_eq!(draw_items[1].sort_z, 2.0);
}

#[test]
fn contiguous_ranges_split_only_on_key_changes() {
    let ranges = build_contiguous_ranges_by_key([5_u64, 5, 9, 9, 1, 2, 2]);

    assert_eq!(ranges, vec![(0, 2), (2, 4), (4, 5), (5, 7)]);
}

#[test]
fn contiguous_ranges_is_empty_for_empty_input() {
    let ranges = build_contiguous_ranges_by_key(std::iter::empty::<u64>());

    assert!(ranges.is_empty());
}

#[test]
fn build_draw_batches_2d_creates_single_batch_for_same_texture_sequence() {
    let mut asset_server = AssetServer::new(assets_root());
    let texture = asset_server
        .load_texture_handle("textures/placeholder.png")
        .expect("load test texture");

    let draw_items = vec![
        DrawItem2d {
            texture,
            model: [[0.0; 4]; 4],
            color: [1.0, 1.0, 1.0, 1.0],
            uv_rect: [0.0, 0.0, 1.0, 1.0],
            sort_z: 0.0,
        },
        DrawItem2d {
            texture,
            model: [[0.0; 4]; 4],
            color: [1.0, 1.0, 1.0, 1.0],
            uv_rect: [0.0, 0.0, 1.0, 1.0],
            sort_z: 1.0,
        },
        DrawItem2d {
            texture,
            model: [[0.0; 4]; 4],
            color: [1.0, 1.0, 1.0, 1.0],
            uv_rect: [0.0, 0.0, 1.0, 1.0],
            sort_z: 2.0,
        },
    ];

    let batches = build_draw_batches_2d(draw_items.as_slice());

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].start, 0);
    assert_eq!(batches[0].end, 3);
    assert_eq!(batches[0].texture.id(), texture.id());
}
