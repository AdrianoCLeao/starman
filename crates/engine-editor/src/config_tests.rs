use crate::config::{
    EditorConfig, GizmoAxisLockConfig, GizmoModeConfig, GizmoOrientationConfig, GizmoSnapConfig,
    GizmoToolConfig, ViewportOverlayConfig,
};

#[test]
fn editor_config_default_has_expected_overlay_and_snap_settings() {
    let config = EditorConfig::default();

    assert!(config.viewport_overlay.show_grid);
    assert!(!config.viewport_overlay.show_collider_wireframes);
    assert!(!config.viewport_overlay.show_entity_labels);
    assert!(config.viewport_overlay.show_fps);

    assert!(!config.gizmo_snap.enabled);
    assert!((config.gizmo_snap.translate_step - 0.5).abs() <= f32::EPSILON);
    assert!((config.gizmo_snap.rotate_step_degrees - 15.0).abs() <= f32::EPSILON);
    assert!((config.gizmo_snap.scale_step - 0.1).abs() <= f32::EPSILON);

    assert_eq!(config.gizmo_tool.mode, GizmoModeConfig::Translate);
    assert_eq!(config.gizmo_tool.orientation, GizmoOrientationConfig::Local);
    assert_eq!(config.gizmo_tool.axis_lock, GizmoAxisLockConfig::Free);
}

#[test]
fn editor_config_roundtrip_preserves_overlay_and_snap_settings() {
    let config = EditorConfig {
        viewport_overlay: ViewportOverlayConfig {
            show_grid: false,
            show_collider_wireframes: true,
            show_entity_labels: true,
            show_fps: false,
        },
        gizmo_snap: GizmoSnapConfig {
            enabled: true,
            translate_step: 1.25,
            rotate_step_degrees: 30.0,
            scale_step: 0.25,
        },
        gizmo_tool: GizmoToolConfig {
            mode: GizmoModeConfig::Rotate,
            orientation: GizmoOrientationConfig::Global,
            axis_lock: GizmoAxisLockConfig::AxisZ,
        },
        ..EditorConfig::default()
    };

    let serialized = ron::ser::to_string_pretty(&config, ron::ser::PrettyConfig::new())
        .expect("config serialization should succeed");
    let parsed: EditorConfig =
        ron::from_str(&serialized).expect("config deserialization should succeed");

    assert!(!parsed.viewport_overlay.show_grid);
    assert!(parsed.viewport_overlay.show_collider_wireframes);
    assert!(parsed.viewport_overlay.show_entity_labels);
    assert!(!parsed.viewport_overlay.show_fps);

    assert!(parsed.gizmo_snap.enabled);
    assert!((parsed.gizmo_snap.translate_step - 1.25).abs() <= f32::EPSILON);
    assert!((parsed.gizmo_snap.rotate_step_degrees - 30.0).abs() <= f32::EPSILON);
    assert!((parsed.gizmo_snap.scale_step - 0.25).abs() <= f32::EPSILON);

    assert_eq!(parsed.gizmo_tool.mode, GizmoModeConfig::Rotate);
    assert_eq!(
        parsed.gizmo_tool.orientation,
        GizmoOrientationConfig::Global
    );
    assert_eq!(parsed.gizmo_tool.axis_lock, GizmoAxisLockConfig::AxisZ);
}

#[test]
fn editor_config_deserializes_legacy_payload_with_new_defaults() {
    let legacy_payload = r#"(
        recent_files: [],
        last_opened_scene: None,
        viewport_camera: (
            target: (0.0, 0.0, 0.0),
            distance: 8.0,
            yaw: 0.1,
            pitch: -0.2,
        ),
    )"#;

    let parsed: EditorConfig =
        ron::from_str(legacy_payload).expect("legacy payload should deserialize with defaults");

    assert!(parsed.viewport_overlay.show_grid);
    assert!(!parsed.viewport_overlay.show_collider_wireframes);
    assert!(!parsed.viewport_overlay.show_entity_labels);
    assert!(parsed.viewport_overlay.show_fps);

    assert!(!parsed.gizmo_snap.enabled);
    assert!((parsed.gizmo_snap.translate_step - 0.5).abs() <= f32::EPSILON);
    assert!((parsed.gizmo_snap.rotate_step_degrees - 15.0).abs() <= f32::EPSILON);
    assert!((parsed.gizmo_snap.scale_step - 0.1).abs() <= f32::EPSILON);

    assert_eq!(parsed.gizmo_tool.mode, GizmoModeConfig::Translate);
    assert_eq!(parsed.gizmo_tool.orientation, GizmoOrientationConfig::Local);
    assert_eq!(parsed.gizmo_tool.axis_lock, GizmoAxisLockConfig::Free);
}
