use bevy_ecs::{prelude::World, query::With};
use engine_core::{Camera2d, GlobalTransform, PrimaryCamera};
use engine_math::Mat4;

use crate::Camera2dUniform;

pub(crate) fn extract_camera_uniform_2d(
    world: &mut World,
    viewport_width: u32,
    viewport_height: u32,
) -> Camera2dUniform {
    let mut query = world.query_filtered::<(&Camera2d, &GlobalTransform), With<PrimaryCamera>>();

    let view_proj = if let Some((camera, global_transform)) = query.iter(world).next() {
        let view = Mat4::from(global_transform.0.inverse());
        camera.projection_matrix() * view
    } else {
        let width = viewport_width.max(1) as f32;
        let height = viewport_height.max(1) as f32;
        Mat4::orthographic_rh(
            -width * 0.5,
            width * 0.5,
            -height * 0.5,
            height * 0.5,
            -1.0,
            1.0,
        )
    };

    Camera2dUniform {
        view_proj: view_proj.to_cols_array_2d(),
    }
}
