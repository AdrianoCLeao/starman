use bevy_ecs::prelude::World;
use engine_core::{Camera2d, GlobalTransform, PrimaryCamera};
use engine_math::{glam::Affine3A, Mat4, Vec3};

use crate::camera_uniforms::extract_camera_uniform_2d;

fn assert_matrix_approx_eq(actual: [[f32; 4]; 4], expected: [[f32; 4]; 4], epsilon: f32) {
    for row in 0..4 {
        for col in 0..4 {
            let delta = (actual[row][col] - expected[row][col]).abs();
            assert!(
                delta <= epsilon,
                "matrix mismatch at [{row}][{col}] delta={delta} actual={} expected={}",
                actual[row][col],
                expected[row][col]
            );
        }
    }
}

#[test]
fn extract_camera_uniform_2d_falls_back_to_viewport_if_no_primary_camera() {
    let mut world = World::new();

    let uniform = extract_camera_uniform_2d(&mut world, 640, 480);
    let expected = Mat4::orthographic_rh(-320.0, 320.0, -240.0, 240.0, -1.0, 1.0);

    assert_matrix_approx_eq(uniform.view_proj, expected.to_cols_array_2d(), 1e-6);
}

#[test]
fn extract_camera_uniform_2d_uses_primary_camera_projection_and_view() {
    let mut world = World::new();
    let camera = Camera2d {
        viewport_width: 200.0,
        viewport_height: 100.0,
        near: -2.0,
        far: 2.0,
    };
    let transform = GlobalTransform(Affine3A::from_translation(Vec3::new(10.0, -20.0, 0.0)));

    world.spawn((camera.clone(), transform.clone(), PrimaryCamera));

    let uniform = extract_camera_uniform_2d(&mut world, 1, 1);
    let expected = camera.projection_matrix() * Mat4::from(transform.0.inverse());

    assert_matrix_approx_eq(uniform.view_proj, expected.to_cols_array_2d(), 1e-5);
}
