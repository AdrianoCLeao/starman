#![allow(dead_code)] // Legacy world collectors kept for unit tests / migration.

use bevy_ecs::{prelude::World, query::With};
use engine_assets::{MaterialHandle, MeshHandle, TextureHandle};
use engine_core::{GlobalTransform, RenderLayer2D, RenderLayer3D, Visible};
use engine_math::{Mat4, Vec3};
use std::cmp::Ordering;

use crate::{MeshRenderable3d, SpriteRenderable2d};

#[derive(Clone, Copy)]
pub(crate) struct DrawItem3d {
    pub(crate) mesh: MeshHandle,
    pub(crate) texture: TextureHandle,
    pub(crate) material: MaterialHandle,
    pub(crate) model: [[f32; 4]; 4],
    pub(crate) normal: [[f32; 4]; 4],
}

#[derive(Clone, Copy)]
pub(crate) struct DrawItem2d {
    pub(crate) texture: TextureHandle,
    pub(crate) model: [[f32; 4]; 4],
    pub(crate) color: [f32; 4],
    pub(crate) uv_rect: [f32; 4],
    pub(crate) sort_z: f32,
}

#[derive(Clone, Copy)]
pub(crate) struct DrawBatch2d {
    pub(crate) texture: TextureHandle,
    pub(crate) start: usize,
    pub(crate) end: usize,
}

pub(crate) fn build_contiguous_ranges_by_key<I, K>(keys: I) -> Vec<(usize, usize)>
where
    I: IntoIterator<Item = K>,
    K: PartialEq + Copy,
{
    let mut ranges = Vec::new();
    let mut iter = keys.into_iter();

    let Some(mut previous) = iter.next() else {
        return ranges;
    };

    let mut start = 0usize;
    let mut index = 1usize;

    for key in iter {
        if key != previous {
            ranges.push((start, index));
            start = index;
            previous = key;
        }
        index += 1;
    }

    ranges.push((start, index));
    ranges
}

pub(crate) fn build_draw_batches_2d(draw_items: &[DrawItem2d]) -> Vec<DrawBatch2d> {
    build_contiguous_ranges_by_key(draw_items.iter().map(|item| item.texture.id().value()))
        .into_iter()
        .map(|(start, end)| DrawBatch2d {
            texture: draw_items[start].texture,
            start,
            end,
        })
        .collect()
}

pub(crate) fn collect_draw_items_3d(world: &mut World, max_draw_items: usize) -> Vec<DrawItem3d> {
    let mut draw_items = Vec::new();
    let mut query = world.query_filtered::<
        (&GlobalTransform, &MeshRenderable3d),
        (With<Visible>, With<RenderLayer3D>),
    >();

    for (global_transform, mesh_renderable) in query.iter(world) {
        if draw_items.len() >= max_draw_items {
            log::warn!(
                target: "engine::render",
                "3D draw item limit reached ({}); skipping remaining visible 3D items this frame",
                max_draw_items
            );
            break;
        }

        let model = Mat4::from(global_transform.0);
        let normal = model.inverse().transpose();

        draw_items.push(DrawItem3d {
            mesh: mesh_renderable.mesh,
            texture: mesh_renderable.texture,
            material: mesh_renderable.material,
            model: model.to_cols_array_2d(),
            normal: normal.to_cols_array_2d(),
        });
    }

    draw_items
}

pub(crate) fn collect_draw_items_2d(world: &mut World, max_draw_items: usize) -> Vec<DrawItem2d> {
    let mut draw_items = Vec::new();
    let mut query = world.query_filtered::<
        (&GlobalTransform, &SpriteRenderable2d),
        (With<Visible>, With<RenderLayer2D>),
    >();

    for (global_transform, sprite) in query.iter(world) {
        if draw_items.len() >= max_draw_items {
            log::warn!(
                target: "engine::render",
                "2D draw item limit reached ({}); skipping remaining visible 2D items this frame",
                max_draw_items
            );
            break;
        }

        let model = Mat4::from(global_transform.0)
            * Mat4::from_scale(Vec3::new(sprite.size[0], sprite.size[1], 1.0));
        let translation = global_transform.translation();

        draw_items.push(DrawItem2d {
            texture: sprite.texture,
            model: model.to_cols_array_2d(),
            color: sprite.color,
            uv_rect: [
                sprite.uv_min[0],
                sprite.uv_min[1],
                sprite.uv_max[0],
                sprite.uv_max[1],
            ],
            sort_z: translation.z,
        });
    }

    draw_items.sort_by(|left, right| {
        left.sort_z
            .partial_cmp(&right.sort_z)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.texture.id().value().cmp(&right.texture.id().value()))
    });

    draw_items
}
