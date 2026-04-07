use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;
use eframe::egui;
use engine_core::{Children, EntityName, Parent, Transform};
use engine_math::glam::{Quat, Vec3};

use crate::app::{
    apply_axis_constraint, asset_drop_entity_name, build_play_mode_snapshot_path,
    build_scene_tree_visibility_map, collect_scene_tree_roots, compute_gizmo_axis_constraint,
    compute_gizmo_drag_intent, compute_gizmo_drag_transform, cycle_gizmo_mode,
    gizmo_axis_constraint_label, gizmo_drag_intent_label, gizmo_manual_axis_lock_label,
    gizmo_mode_label, gizmo_orientation_label, is_scene_tree_descendant, ray_intersects_aabb,
    runner_executable_candidates, select_runner_executable_path, snap_scalar, snap_vec3,
    toggle_gizmo_orientation, viewport_drop_position, viewport_pick_ray, GizmoAxisConstraint,
    GizmoDragIntent, GizmoMode, GizmoOrientation,
};

fn quat_is_close(lhs: Quat, rhs: Quat, epsilon: f32) -> bool {
    (1.0 - lhs.dot(rhs).abs()) <= epsilon
}

fn attach_child(world: &mut World, parent: Entity, child: Entity) {
    if let Ok(mut child_ref) = world.get_entity_mut(child) {
        child_ref.insert(Parent(parent));
    }

    if let Ok(mut parent_ref) = world.get_entity_mut(parent) {
        if let Some(mut children) = parent_ref.get_mut::<Children>() {
            if !children.0.contains(&child) {
                children.0.push(child);
            }
        } else {
            parent_ref.insert(Children(vec![child]));
        }
    }
}

fn create_temp_test_dir(prefix: &str) -> PathBuf {
    let timestamp_nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_nanos())
        .unwrap_or_default();
    let path = std::env::temp_dir().join(format!(
        "motley-editor-tests-{}-{}-{}",
        prefix,
        std::process::id(),
        timestamp_nanos
    ));

    std::fs::create_dir_all(&path).expect("failed to create temporary test directory");
    path
}

#[test]
fn scene_tree_visibility_keeps_matching_ancestors_visible() {
    let mut world = World::new();

    let root = world
        .spawn((EntityName::new("Level Root"), Children::default()))
        .id();
    let branch = world
        .spawn((EntityName::new("Gameplay"), Children::default()))
        .id();
    let camera = world.spawn((EntityName::new("Main Camera"),)).id();
    let unrelated = world.spawn((EntityName::new("Audio Bus"),)).id();

    attach_child(&mut world, root, branch);
    attach_child(&mut world, branch, camera);

    let roots = collect_scene_tree_roots(&world);
    let visibility = build_scene_tree_visibility_map(&world, &roots, "camera");

    assert!(visibility.get(&root).copied().unwrap_or(false));
    assert!(visibility.get(&branch).copied().unwrap_or(false));
    assert!(visibility.get(&camera).copied().unwrap_or(false));
    assert!(!visibility.get(&unrelated).copied().unwrap_or(false));
}

#[test]
fn collect_scene_tree_roots_treats_entities_with_missing_parent_as_roots() {
    let mut world = World::new();

    let parent = world
        .spawn((EntityName::new("Parent"), Children::default()))
        .id();
    let child = world.spawn((EntityName::new("Child"),)).id();
    attach_child(&mut world, parent, child);

    let removed_parent = world.spawn((EntityName::new("Removed Parent"),)).id();
    let orphan = world
        .spawn((EntityName::new("Orphan"), Parent(removed_parent)))
        .id();
    let independent_root = world.spawn((EntityName::new("Independent"),)).id();

    let _ = world.despawn(removed_parent);

    let roots = collect_scene_tree_roots(&world);

    assert!(roots.contains(&parent));
    assert!(roots.contains(&orphan));
    assert!(roots.contains(&independent_root));
    assert!(!roots.contains(&child));
}

#[test]
fn scene_tree_descendant_check_uses_parent_chain() {
    let mut world = World::new();

    let root = world.spawn((EntityName::new("Root"),)).id();
    let child = world.spawn((EntityName::new("Child"), Parent(root))).id();
    let grandchild = world
        .spawn((EntityName::new("Grandchild"), Parent(child)))
        .id();
    let unrelated = world.spawn((EntityName::new("Unrelated"),)).id();

    assert!(is_scene_tree_descendant(&world, grandchild, root));
    assert!(is_scene_tree_descendant(&world, child, root));
    assert!(!is_scene_tree_descendant(&world, root, grandchild));
    assert!(!is_scene_tree_descendant(&world, unrelated, root));
}

#[test]
fn scene_tree_visibility_map_handles_child_cycles_without_recursing_forever() {
    let mut world = World::new();

    let node_a = world
        .spawn((EntityName::new("Node A"), Children::default()))
        .id();
    let node_b = world
        .spawn((EntityName::new("Node B"), Children::default()))
        .id();

    attach_child(&mut world, node_a, node_b);
    attach_child(&mut world, node_b, node_a);

    let visibility = build_scene_tree_visibility_map(&world, &[node_a], "not-found");

    assert!(!visibility.get(&node_a).copied().unwrap_or(false));
    assert!(!visibility.get(&node_b).copied().unwrap_or(false));
}

#[test]
fn viewport_pick_ray_points_forward_at_viewport_center() {
    let rect = egui::Rect::from_min_size(egui::pos2(10.0, 20.0), egui::vec2(200.0, 100.0));
    let camera_position = Vec3::new(1.0, 2.0, 3.0);

    let (origin, direction) = viewport_pick_ray(
        rect.center(),
        rect,
        camera_position,
        Quat::IDENTITY,
        std::f32::consts::FRAC_PI_4,
    )
    .expect("expected a valid pick ray");

    assert!((origin - camera_position).length() <= 1e-6);
    assert!((direction - Vec3::NEG_Z).length() <= 1e-6);
}

#[test]
fn viewport_drop_position_projects_to_ground_plane_when_camera_is_pitched_down() {
    let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(320.0, 180.0));
    let fallback = Vec3::new(5.0, 5.0, 5.0);
    let pointer = egui::pos2(rect.center().x, rect.max.y - 2.0);

    let drop_position = viewport_drop_position(
        pointer,
        rect,
        Vec3::new(0.0, 3.0, 6.0),
        Quat::IDENTITY,
        std::f32::consts::FRAC_PI_4,
        fallback,
    );

    assert!(drop_position.y.abs() <= 1e-4);
    assert!((drop_position - fallback).length() > 1.0);
}

#[test]
fn asset_drop_entity_name_uses_file_stem_and_normalizes_separators() {
    assert_eq!(
        asset_drop_entity_name("meshes/robot-knight.glb"),
        "robot knight"
    );
    assert_eq!(asset_drop_entity_name("meshes/My_Crate.gltf"), "My Crate");
    assert_eq!(asset_drop_entity_name(""), "Dropped Mesh");
}

#[test]
fn ray_intersects_aabb_returns_hit_distance_and_none_for_miss() {
    let hit_distance = ray_intersects_aabb(
        Vec3::new(0.0, 0.0, 5.0),
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::splat(-1.0),
        Vec3::splat(1.0),
        100.0,
    )
    .expect("ray should hit axis-aligned box");

    assert!((hit_distance - 4.0).abs() <= 1e-6);

    let miss = ray_intersects_aabb(
        Vec3::new(0.0, 0.0, 5.0),
        Vec3::X,
        Vec3::splat(-1.0),
        Vec3::splat(1.0),
        100.0,
    );

    assert!(miss.is_none());
}

#[test]
fn snap_scalar_rounds_to_nearest_step() {
    assert!((snap_scalar(1.24, 0.5) - 1.0).abs() <= 1e-6);
    assert!((snap_scalar(1.26, 0.5) - 1.5).abs() <= 1e-6);
    assert!((snap_scalar(-1.24, 0.5) - -1.0).abs() <= 1e-6);
    assert!((snap_scalar(-1.26, 0.5) - -1.5).abs() <= 1e-6);
}

#[test]
fn snap_scalar_with_non_positive_step_returns_original_value() {
    let value = std::f32::consts::PI;
    assert!((snap_scalar(value, 0.0) - value).abs() <= f32::EPSILON);
    assert!((snap_scalar(value, -2.0) - value).abs() <= f32::EPSILON);
}

#[test]
fn snap_vec3_snaps_each_component_independently() {
    let value = Vec3::new(1.24, -0.74, 0.26);
    let snapped = snap_vec3(value, 0.5);
    let expected = Vec3::new(1.0, -0.5, 0.5);

    assert!((snapped - expected).length() <= 1e-6);

    let unsnapped = snap_vec3(value, 0.0);
    assert!((unsnapped - value).length() <= 1e-6);
}

#[test]
fn cycle_gizmo_mode_cycles_all_modes_in_order() {
    assert_eq!(cycle_gizmo_mode(GizmoMode::Translate), GizmoMode::Rotate);
    assert_eq!(cycle_gizmo_mode(GizmoMode::Rotate), GizmoMode::Scale);
    assert_eq!(cycle_gizmo_mode(GizmoMode::Scale), GizmoMode::Translate);
}

#[test]
fn toggle_gizmo_orientation_flips_between_local_and_global() {
    assert_eq!(
        toggle_gizmo_orientation(GizmoOrientation::Local),
        GizmoOrientation::Global
    );
    assert_eq!(
        toggle_gizmo_orientation(GizmoOrientation::Global),
        GizmoOrientation::Local
    );
}

#[test]
fn gizmo_state_labels_match_expected_user_facing_names() {
    assert_eq!(gizmo_mode_label(GizmoMode::Translate), "Translate");
    assert_eq!(gizmo_mode_label(GizmoMode::Rotate), "Rotate");
    assert_eq!(gizmo_mode_label(GizmoMode::Scale), "Scale");

    assert_eq!(gizmo_orientation_label(GizmoOrientation::Local), "Local");
    assert_eq!(gizmo_orientation_label(GizmoOrientation::Global), "Global");
}

#[test]
fn compute_gizmo_drag_transform_translate_global_snaps_to_grid() {
    let initial = Transform {
        translation: Vec3::new(0.12, 0.13, 0.0),
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    let transformed = compute_gizmo_drag_transform(
        &initial,
        GizmoMode::Translate,
        GizmoOrientation::Global,
        None,
        true,
        0.3,
        -0.1,
        1.0,
        std::f32::consts::FRAC_PI_2,
        0.5,
        15.0,
        0.1,
    );

    assert!((transformed.translation - Vec3::new(1.5, 0.5, 0.0)).length() <= 1e-6);
    assert!(quat_is_close(transformed.rotation, initial.rotation, 1e-6));
    assert!((transformed.scale - initial.scale).length() <= 1e-6);
}

#[test]
fn compute_gizmo_drag_transform_translate_local_uses_local_axes() {
    let initial = Transform {
        translation: Vec3::ZERO,
        rotation: Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
        scale: Vec3::ONE,
    };

    let transformed = compute_gizmo_drag_transform(
        &initial,
        GizmoMode::Translate,
        GizmoOrientation::Local,
        None,
        false,
        0.25,
        0.0,
        1.0,
        std::f32::consts::FRAC_PI_2,
        0.5,
        15.0,
        0.1,
    );

    assert!((transformed.translation - Vec3::new(0.0, 1.0, 0.0)).length() <= 1e-5);
}

#[test]
fn compute_gizmo_drag_transform_rotate_respects_snap_step() {
    let initial = Transform {
        translation: Vec3::new(4.0, 5.0, 6.0),
        rotation: Quat::IDENTITY,
        scale: Vec3::splat(2.0),
    };

    let transformed = compute_gizmo_drag_transform(
        &initial,
        GizmoMode::Rotate,
        GizmoOrientation::Local,
        None,
        true,
        0.26,
        0.0,
        10.0,
        std::f32::consts::FRAC_PI_4,
        0.5,
        90.0,
        0.1,
    );

    let expected = Quat::from_rotation_y(-std::f32::consts::FRAC_PI_2);
    assert!(quat_is_close(transformed.rotation, expected, 1e-5));
    assert!((transformed.translation - initial.translation).length() <= 1e-6);
    assert!((transformed.scale - initial.scale).length() <= 1e-6);
}

#[test]
fn compute_gizmo_drag_transform_scale_clamps_to_positive_minimum() {
    let initial = Transform {
        translation: Vec3::new(-1.0, 2.0, 3.0),
        rotation: Quat::from_rotation_x(0.5),
        scale: Vec3::splat(0.01),
    };

    let transformed = compute_gizmo_drag_transform(
        &initial,
        GizmoMode::Scale,
        GizmoOrientation::Global,
        None,
        true,
        -10.0,
        10.0,
        1.0,
        std::f32::consts::FRAC_PI_2,
        0.5,
        15.0,
        0.5,
    );

    assert!((transformed.scale - Vec3::splat(0.001)).length() <= 1e-6);
    assert!((transformed.translation - initial.translation).length() <= 1e-6);
    assert!(quat_is_close(transformed.rotation, initial.rotation, 1e-6));
}

#[test]
fn compute_gizmo_drag_transform_rotate_local_and_global_differ_when_pre_rotated() {
    let initial = Transform {
        translation: Vec3::new(1.0, 2.0, 3.0),
        rotation: Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
        scale: Vec3::splat(1.5),
    };

    let local = compute_gizmo_drag_transform(
        &initial,
        GizmoMode::Rotate,
        GizmoOrientation::Local,
        None,
        false,
        0.2,
        -0.1,
        5.0,
        std::f32::consts::FRAC_PI_4,
        0.5,
        15.0,
        0.1,
    );

    let global = compute_gizmo_drag_transform(
        &initial,
        GizmoMode::Rotate,
        GizmoOrientation::Global,
        None,
        false,
        0.2,
        -0.1,
        5.0,
        std::f32::consts::FRAC_PI_4,
        0.5,
        15.0,
        0.1,
    );

    assert!(!quat_is_close(local.rotation, global.rotation, 1e-5));
    assert!((local.translation - initial.translation).length() <= 1e-6);
    assert!((global.translation - initial.translation).length() <= 1e-6);
    assert!((local.scale - initial.scale).length() <= 1e-6);
    assert!((global.scale - initial.scale).length() <= 1e-6);
}

#[test]
fn compute_gizmo_drag_intent_translate_prefers_dominant_axis() {
    assert_eq!(
        compute_gizmo_drag_intent(GizmoMode::Translate, 0.35, 0.1),
        GizmoDragIntent::AxisX
    );
    assert_eq!(
        compute_gizmo_drag_intent(GizmoMode::Translate, 0.1, -0.35),
        GizmoDragIntent::AxisY
    );
}

#[test]
fn compute_gizmo_drag_intent_rotate_prefers_dominant_axis() {
    assert_eq!(
        compute_gizmo_drag_intent(GizmoMode::Rotate, -0.4, 0.2),
        GizmoDragIntent::AxisX
    );
    assert_eq!(
        compute_gizmo_drag_intent(GizmoMode::Rotate, 0.05, -0.4),
        GizmoDragIntent::AxisY
    );
}

#[test]
fn compute_gizmo_drag_intent_scale_is_uniform() {
    assert_eq!(
        compute_gizmo_drag_intent(GizmoMode::Scale, 1.0, 0.0),
        GizmoDragIntent::Uniform
    );
    assert_eq!(
        compute_gizmo_drag_intent(GizmoMode::Scale, 0.0, 1.0),
        GizmoDragIntent::Uniform
    );
}

#[test]
fn gizmo_drag_intent_label_is_mode_specific() {
    assert_eq!(
        gizmo_drag_intent_label(GizmoMode::Translate, GizmoDragIntent::AxisX),
        "Move X"
    );
    assert_eq!(
        gizmo_drag_intent_label(GizmoMode::Translate, GizmoDragIntent::AxisY),
        "Move Y"
    );
    assert_eq!(
        gizmo_drag_intent_label(GizmoMode::Rotate, GizmoDragIntent::AxisX),
        "Yaw"
    );
    assert_eq!(
        gizmo_drag_intent_label(GizmoMode::Rotate, GizmoDragIntent::AxisY),
        "Pitch"
    );
    assert_eq!(
        gizmo_drag_intent_label(GizmoMode::Scale, GizmoDragIntent::Uniform),
        "Uniform"
    );
    assert_eq!(
        gizmo_drag_intent_label(GizmoMode::Scale, GizmoDragIntent::AxisX),
        "Scale X"
    );
    assert_eq!(
        gizmo_drag_intent_label(GizmoMode::Scale, GizmoDragIntent::AxisY),
        "Scale Y"
    );
}

#[test]
fn compute_gizmo_axis_constraint_respects_mode_and_lock_flag() {
    assert_eq!(
        compute_gizmo_axis_constraint(GizmoMode::Translate, None, false, GizmoDragIntent::AxisX),
        None
    );
    assert_eq!(
        compute_gizmo_axis_constraint(GizmoMode::Translate, None, true, GizmoDragIntent::AxisX),
        Some(GizmoAxisConstraint::AxisX)
    );
    assert_eq!(
        compute_gizmo_axis_constraint(GizmoMode::Rotate, None, true, GizmoDragIntent::AxisY),
        Some(GizmoAxisConstraint::AxisY)
    );
    assert_eq!(
        compute_gizmo_axis_constraint(GizmoMode::Scale, None, true, GizmoDragIntent::Uniform),
        None
    );
}

#[test]
fn compute_gizmo_axis_constraint_manual_lock_has_priority_over_shift() {
    assert_eq!(
        compute_gizmo_axis_constraint(
            GizmoMode::Translate,
            Some(GizmoAxisConstraint::AxisY),
            false,
            GizmoDragIntent::AxisX,
        ),
        Some(GizmoAxisConstraint::AxisY)
    );

    assert_eq!(
        compute_gizmo_axis_constraint(
            GizmoMode::Rotate,
            Some(GizmoAxisConstraint::AxisX),
            true,
            GizmoDragIntent::AxisY,
        ),
        Some(GizmoAxisConstraint::AxisX)
    );

    assert_eq!(
        compute_gizmo_axis_constraint(
            GizmoMode::Scale,
            Some(GizmoAxisConstraint::AxisX),
            true,
            GizmoDragIntent::Uniform,
        ),
        Some(GizmoAxisConstraint::AxisX)
    );

    assert_eq!(
        compute_gizmo_axis_constraint(
            GizmoMode::Rotate,
            Some(GizmoAxisConstraint::AxisZ),
            true,
            GizmoDragIntent::AxisY,
        ),
        Some(GizmoAxisConstraint::AxisZ)
    );
}

#[test]
fn apply_axis_constraint_zeroes_non_selected_component() {
    assert_eq!(
        apply_axis_constraint(Some(GizmoAxisConstraint::AxisX), 0.3, -0.2),
        (0.3, 0.0)
    );
    assert_eq!(
        apply_axis_constraint(Some(GizmoAxisConstraint::AxisY), 0.3, -0.2),
        (0.0, -0.2)
    );
    assert_eq!(
        apply_axis_constraint(Some(GizmoAxisConstraint::AxisZ), 0.3, -0.2),
        (0.0, 0.0)
    );
    assert_eq!(apply_axis_constraint(None, 0.3, -0.2), (0.3, -0.2));
}

#[test]
fn compute_gizmo_drag_transform_translate_axis_lock_x_ignores_vertical_delta() {
    let initial = Transform {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    let transformed = compute_gizmo_drag_transform(
        &initial,
        GizmoMode::Translate,
        GizmoOrientation::Global,
        Some(GizmoAxisConstraint::AxisX),
        false,
        0.25,
        0.25,
        1.0,
        std::f32::consts::FRAC_PI_2,
        0.5,
        15.0,
        0.1,
    );

    assert!((transformed.translation - Vec3::new(1.0, 0.0, 0.0)).length() <= 1e-6);
}

#[test]
fn compute_gizmo_drag_transform_translate_axis_lock_z_only_moves_depth() {
    let initial = Transform {
        translation: Vec3::new(1.0, 2.0, 3.0),
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    let transformed = compute_gizmo_drag_transform(
        &initial,
        GizmoMode::Translate,
        GizmoOrientation::Global,
        Some(GizmoAxisConstraint::AxisZ),
        false,
        0.25,
        -0.25,
        1.0,
        std::f32::consts::FRAC_PI_2,
        0.5,
        15.0,
        0.1,
    );

    assert!((transformed.translation - Vec3::new(1.0, 2.0, 4.0)).length() <= 1e-6);
}

#[test]
fn compute_gizmo_drag_transform_rotate_axis_lock_x_ignores_yaw() {
    let initial = Transform {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    let transformed = compute_gizmo_drag_transform(
        &initial,
        GizmoMode::Rotate,
        GizmoOrientation::Global,
        Some(GizmoAxisConstraint::AxisX),
        false,
        0.3,
        -0.2,
        1.0,
        std::f32::consts::FRAC_PI_2,
        0.5,
        15.0,
        0.1,
    );

    let expected = Quat::from_rotation_x(0.2 * std::f32::consts::TAU);
    assert!(quat_is_close(transformed.rotation, expected, 1e-5));
}

#[test]
fn compute_gizmo_drag_transform_rotate_axis_lock_y_ignores_pitch() {
    let initial = Transform {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    let transformed = compute_gizmo_drag_transform(
        &initial,
        GizmoMode::Rotate,
        GizmoOrientation::Global,
        Some(GizmoAxisConstraint::AxisY),
        false,
        0.3,
        -0.2,
        1.0,
        std::f32::consts::FRAC_PI_2,
        0.5,
        15.0,
        0.1,
    );

    let expected = Quat::from_rotation_y(-0.3 * std::f32::consts::TAU);
    assert!(quat_is_close(transformed.rotation, expected, 1e-5));
}

#[test]
fn compute_gizmo_drag_transform_rotate_axis_lock_z_rolls_only() {
    let initial = Transform {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    let transformed = compute_gizmo_drag_transform(
        &initial,
        GizmoMode::Rotate,
        GizmoOrientation::Global,
        Some(GizmoAxisConstraint::AxisZ),
        false,
        0.25,
        -0.2,
        1.0,
        std::f32::consts::FRAC_PI_2,
        0.5,
        15.0,
        0.1,
    );

    let expected = Quat::from_rotation_z(-0.25 * std::f32::consts::TAU);
    assert!(quat_is_close(transformed.rotation, expected, 1e-5));
}

#[test]
fn compute_gizmo_drag_transform_scale_axis_lock_x_only_scales_x() {
    let initial = Transform {
        translation: Vec3::new(1.0, 2.0, 3.0),
        rotation: Quat::from_rotation_y(0.2),
        scale: Vec3::new(2.0, 3.0, 4.0),
    };

    let transformed = compute_gizmo_drag_transform(
        &initial,
        GizmoMode::Scale,
        GizmoOrientation::Global,
        Some(GizmoAxisConstraint::AxisX),
        false,
        0.2,
        -0.4,
        1.0,
        std::f32::consts::FRAC_PI_2,
        0.5,
        15.0,
        0.1,
    );

    let expected_scale = Vec3::new(2.4, 3.0, 4.0);
    assert!((transformed.scale - expected_scale).length() <= 1e-5);
    assert!((transformed.translation - initial.translation).length() <= 1e-6);
    assert!(quat_is_close(transformed.rotation, initial.rotation, 1e-6));
}

#[test]
fn compute_gizmo_drag_transform_scale_axis_lock_y_only_scales_y() {
    let initial = Transform {
        translation: Vec3::new(-2.0, 1.0, 0.0),
        rotation: Quat::from_rotation_x(0.3),
        scale: Vec3::new(2.0, 3.0, 4.0),
    };

    let transformed = compute_gizmo_drag_transform(
        &initial,
        GizmoMode::Scale,
        GizmoOrientation::Global,
        Some(GizmoAxisConstraint::AxisY),
        false,
        0.2,
        -0.4,
        1.0,
        std::f32::consts::FRAC_PI_2,
        0.5,
        15.0,
        0.1,
    );

    let expected_scale = Vec3::new(2.0, 4.2, 4.0);
    assert!((transformed.scale - expected_scale).length() <= 1e-5);
    assert!((transformed.translation - initial.translation).length() <= 1e-6);
    assert!(quat_is_close(transformed.rotation, initial.rotation, 1e-6));
}

#[test]
fn compute_gizmo_drag_transform_scale_axis_lock_z_only_scales_z() {
    let initial = Transform {
        translation: Vec3::new(0.0, -1.0, 2.0),
        rotation: Quat::from_rotation_z(0.15),
        scale: Vec3::new(2.0, 3.0, 4.0),
    };

    let transformed = compute_gizmo_drag_transform(
        &initial,
        GizmoMode::Scale,
        GizmoOrientation::Global,
        Some(GizmoAxisConstraint::AxisZ),
        false,
        0.25,
        -0.4,
        1.0,
        std::f32::consts::FRAC_PI_2,
        0.5,
        15.0,
        0.1,
    );

    let expected_scale = Vec3::new(2.0, 3.0, 5.0);
    assert!((transformed.scale - expected_scale).length() <= 1e-5);
    assert!((transformed.translation - initial.translation).length() <= 1e-6);
    assert!(quat_is_close(transformed.rotation, initial.rotation, 1e-6));
}

#[test]
fn gizmo_axis_constraint_label_maps_to_axis_names() {
    assert_eq!(gizmo_axis_constraint_label(GizmoAxisConstraint::AxisX), "X");
    assert_eq!(gizmo_axis_constraint_label(GizmoAxisConstraint::AxisY), "Y");
    assert_eq!(gizmo_axis_constraint_label(GizmoAxisConstraint::AxisZ), "Z");
}

#[test]
fn gizmo_manual_axis_lock_label_maps_to_user_facing_names() {
    assert_eq!(gizmo_manual_axis_lock_label(None), "Free");
    assert_eq!(
        gizmo_manual_axis_lock_label(Some(GizmoAxisConstraint::AxisX)),
        "X"
    );
    assert_eq!(
        gizmo_manual_axis_lock_label(Some(GizmoAxisConstraint::AxisY)),
        "Y"
    );
    assert_eq!(
        gizmo_manual_axis_lock_label(Some(GizmoAxisConstraint::AxisZ)),
        "Z"
    );
}

#[test]
fn build_play_mode_snapshot_path_changes_with_sequence_suffix() {
    let temp_root = PathBuf::from("C:/tmp");
    let process_id = 1234;
    let timestamp_nanos = 987_654_321_u128;

    let first = build_play_mode_snapshot_path(&temp_root, process_id, timestamp_nanos, 0);
    let second = build_play_mode_snapshot_path(&temp_root, process_id, timestamp_nanos, 1);

    assert_ne!(first, second);
    assert!(first
        .to_string_lossy()
        .contains("scene-1234-987654321-0.scene.ron"));
    assert!(second
        .to_string_lossy()
        .contains("scene-1234-987654321-1.scene.ron"));
}

#[test]
fn runner_executable_candidates_include_hyphen_and_underscore_names() {
    let base = PathBuf::from("C:/runner");
    let candidates = runner_executable_candidates(&base);

    let first = candidates[0].to_string_lossy();
    let second = candidates[1].to_string_lossy();

    assert!(first.contains(&format!("game-runner{}", std::env::consts::EXE_SUFFIX)));
    assert!(second.contains(&format!("game_runner{}", std::env::consts::EXE_SUFFIX)));
}

#[test]
fn select_runner_executable_path_prefers_hyphenated_binary_then_fallback() {
    let temp_dir = create_temp_test_dir("runner-selection");
    let candidates = runner_executable_candidates(&temp_dir);
    let hyphen = candidates[0].clone();
    let underscore = candidates[1].clone();

    std::fs::write(&underscore, b"binary").expect("failed to create underscore runner file");
    let selected_without_hyphen =
        select_runner_executable_path(&temp_dir).expect("expected underscore fallback candidate");
    assert_eq!(selected_without_hyphen, underscore);

    std::fs::write(&hyphen, b"binary").expect("failed to create hyphen runner file");
    let selected_with_hyphen =
        select_runner_executable_path(&temp_dir).expect("expected hyphenated candidate");
    assert_eq!(selected_with_hyphen, hyphen);

    let _ = std::fs::remove_dir_all(temp_dir);
}
