use eframe::egui;
use engine_math::glam::Vec3;

use crate::viewport::EditorCamera;

#[test]
fn editor_camera_orbit_clamps_pitch_to_prevent_flip() {
    let mut camera = EditorCamera::default();

    camera.orbit(egui::vec2(0.0, -100_000.0));
    assert!(camera.pitch <= 89.0_f32.to_radians() + 1e-4);

    camera.orbit(egui::vec2(0.0, 200_000.0));
    assert!(camera.pitch >= -89.0_f32.to_radians() - 1e-4);
}

#[test]
fn editor_camera_zoom_clamps_distance_bounds() {
    let mut camera = EditorCamera::default();

    camera.zoom(1_000_000.0);
    assert!((camera.distance - 0.25).abs() <= 1e-6);

    camera.zoom(-1_000_000.0);
    assert!((camera.distance - 10_000.0).abs() <= 1e-3);
}

#[test]
fn editor_camera_focus_updates_target() {
    let mut camera = EditorCamera::default();
    let new_target = Vec3::new(5.0, 2.0, -3.0);

    camera.focus_on(new_target);

    assert!((camera.target - new_target).length() <= 1e-6);
}

#[test]
fn editor_camera_pan_moves_target_position() {
    let mut camera = EditorCamera::default();
    let initial_target = camera.target;

    camera.pan(egui::vec2(24.0, -16.0), egui::vec2(1280.0, 720.0));

    assert!((camera.target - initial_target).length() > 0.0);
}
