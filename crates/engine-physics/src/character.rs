//! Kinematic character controller on top of Rapier's
//! `KinematicCharacterController`: slopes, steps, snap-to-ground, pushing
//! dynamic bodies, gravity/jumping and root-motion input.
//!
//! The entity needs `RigidBodyType::Kinematic` and a collider (a capsule
//! is recommended). Gameplay writes [`CharacterInput`] (any time during
//! the frame); every fixed step consumes it and publishes
//! [`CharacterState`].

use bevy_ecs::prelude::*;
use bevy_reflect::Reflect;
use engine_core::{Parent, Transform};
use engine_math::{Quat, Vec3};
use rapier3d::control::{
    CharacterAutostep, CharacterCollision, CharacterLength, KinematicCharacterController,
};
use rapier3d::prelude::{QueryFilter, UnitVector};

use crate::components::{CollisionLayer, PhysicsPose, RigidBodyHandle3D, RigidBodyType};
use crate::layers::PhysicsLayers;
use crate::mapping::PhysicsEntityHandles3D;
use crate::pose::{
    from_isometry, from_vector, parent_affine, to_isometry, to_vector, world_to_local,
};
use crate::systems::PhysicsStepConfig3D;
use crate::world3d::PhysicsWorld3D;

#[derive(Component, Clone, Copy, Debug, PartialEq, Reflect, engine_reflect::RegisterReflect)]
pub struct CharacterController {
    #[engine_reflect(degrees)]
    pub max_slope_climb_angle: f32,
    #[engine_reflect(degrees)]
    pub min_slope_slide_angle: f32,
    /// Autostep height (0 disables stepping).
    #[engine_reflect(range(min = 0.0, max = 2.0))]
    pub step_height: f32,
    #[engine_reflect(range(min = 0.0, max = 2.0))]
    pub step_min_width: f32,
    /// Distance within which the character sticks to the ground when
    /// walking down slopes/steps (0 disables).
    #[engine_reflect(range(min = 0.0, max = 2.0))]
    pub snap_to_ground: f32,
    /// Skin width kept between the character and obstacles.
    #[engine_reflect(range(min = 0.001, max = 0.5))]
    pub offset: f32,
    #[engine_reflect(range(min = -10.0, max = 10.0))]
    pub gravity_scale: f32,
    #[engine_reflect(range(min = 0.0, max = 200.0))]
    pub max_fall_speed: f32,
    /// Seconds after leaving the ground during which a jump still works.
    #[engine_reflect(range(min = 0.0, max = 1.0))]
    pub coyote_time: f32,
    pub push_dynamic_bodies: bool,
    /// Mass used when pushing dynamic bodies.
    #[engine_reflect(range(min = 0.1, max = 10000.0))]
    pub mass: f32,
}

impl Default for CharacterController {
    fn default() -> Self {
        Self {
            max_slope_climb_angle: 45f32.to_radians(),
            min_slope_slide_angle: 50f32.to_radians(),
            step_height: 0.3,
            step_min_width: 0.15,
            snap_to_ground: 0.25,
            offset: 0.02,
            gravity_scale: 1.0,
            max_fall_speed: 50.0,
            coyote_time: 0.12,
            push_dynamic_bodies: true,
            mass: 80.0,
        }
    }
}

/// What gameplay wants the character to do. `velocity` persists;
/// `jump_speed`, `root_motion` and `root_rotation` are consumed by the
/// next fixed step.
#[derive(Component, Clone, Copy, Debug, PartialEq, Reflect, engine_reflect::RegisterReflect)]
pub struct CharacterInput {
    /// Desired world-space velocity. The component along "up" is added to
    /// the gravity-driven vertical speed (ladders, swimming).
    pub velocity: Vec3,
    /// Jump with this upward speed (when grounded or within coyote time).
    pub jump_speed: f32,
    /// World-space translation from animation root motion.
    pub root_motion: Vec3,
    /// Rotation delta from animation root motion.
    pub root_rotation: Quat,
}

impl Default for CharacterInput {
    fn default() -> Self {
        Self {
            velocity: Vec3::ZERO,
            jump_speed: 0.0,
            root_motion: Vec3::ZERO,
            root_rotation: Quat::IDENTITY,
        }
    }
}

/// Output of the last fixed step.
#[derive(
    Component, Clone, Copy, Debug, Default, PartialEq, Reflect, engine_reflect::RegisterReflect,
)]
pub struct CharacterState {
    pub grounded: bool,
    pub sliding: bool,
    /// Effective world velocity of the last step.
    pub velocity: Vec3,
    /// Gravity-driven speed along "up".
    pub vertical_speed: f32,
    pub seconds_since_grounded: f32,
    /// A jump started during the last step.
    pub jumped: bool,
    /// Landed during the last step.
    pub landed: bool,
}

/// Adds the input/state components to new controllers.
pub fn ensure_character_components(
    mut commands: Commands,
    missing: Query<
        (Entity, Has<CharacterInput>, Has<CharacterState>),
        (
            With<CharacterController>,
            Or<(Without<CharacterInput>, Without<CharacterState>)>,
        ),
    >,
) {
    for (entity, has_input, has_state) in &missing {
        let mut entity = commands.entity(entity);
        if !has_input {
            entity.insert(CharacterInput::default());
        }
        if !has_state {
            entity.insert(CharacterState::default());
        }
    }
}

fn rapier_controller(settings: &CharacterController, up: Vec3) -> KinematicCharacterController {
    KinematicCharacterController {
        up: UnitVector::new_normalize(to_vector(up)),
        offset: CharacterLength::Absolute(settings.offset.max(0.001)),
        slide: true,
        autostep: (settings.step_height > 0.0).then_some(CharacterAutostep {
            max_height: CharacterLength::Absolute(settings.step_height),
            min_width: CharacterLength::Absolute(settings.step_min_width.max(0.0)),
            include_dynamic_bodies: false,
        }),
        max_slope_climb_angle: settings.max_slope_climb_angle,
        min_slope_slide_angle: settings.min_slope_slide_angle,
        snap_to_ground: (settings.snap_to_ground > 0.0)
            .then_some(CharacterLength::Absolute(settings.snap_to_ground)),
        normal_nudge_factor: 1.0e-4,
    }
}

type CharacterData = (
    Entity,
    &'static CharacterController,
    &'static mut CharacterInput,
    &'static mut CharacterState,
    &'static RigidBodyHandle3D,
    &'static RigidBodyType,
    &'static mut Transform,
    Option<&'static mut PhysicsPose>,
    Option<&'static CollisionLayer>,
);

/// Moves every character controller by one fixed step.
#[allow(clippy::too_many_arguments)]
pub fn move_character_controllers(
    mut set: ParamSet<(
        Query<(&'static Transform, Option<&'static Parent>)>,
        Query<CharacterData>,
    )>,
    parents: Query<&Parent>,
    mut physics: Option<ResMut<PhysicsWorld3D>>,
    entity_handles: Option<Res<PhysicsEntityHandles3D>>,
    layers: Option<Res<PhysicsLayers>>,
    step: Option<Res<PhysicsStepConfig3D>>,
) {
    let (Some(physics), Some(entity_handles)) = (physics.as_deref_mut(), entity_handles) else {
        return;
    };
    let dt = step.map_or(1.0 / 60.0, |step| step.fixed_dt_seconds);
    let gravity = from_vector(&physics.gravity);
    let up = if gravity.length_squared() > 1e-8 {
        -gravity.normalize()
    } else {
        Vec3::Y
    };
    let gravity_magnitude = gravity.length();

    let with_parents: Vec<Entity> = set
        .p1()
        .iter()
        .map(|item| item.0)
        .filter(|entity| parents.contains(*entity))
        .collect();
    let parent_spaces: std::collections::HashMap<Entity, engine_math::Affine3A> = {
        let transforms = set.p0();
        with_parents
            .into_iter()
            .map(|entity| (entity, parent_affine(entity, &transforms)))
            .collect()
    };

    let PhysicsWorld3D {
        rigid_body_set,
        collider_set,
        query_pipeline,
        ..
    } = physics;

    for (
        entity,
        settings,
        mut input,
        mut state,
        body_handle,
        body_type,
        mut transform,
        pose,
        layer,
    ) in &mut set.p1()
    {
        if *body_type != RigidBodyType::Kinematic {
            continue;
        }
        let Some(collider_handle) = entity_handles.collider(entity).or_else(|| {
            rigid_body_set
                .get(body_handle.0)
                .and_then(|body| body.colliders().first().copied())
        }) else {
            continue;
        };
        let Some(body) = rigid_body_set.get(body_handle.0) else {
            continue;
        };
        let Some(collider) = collider_set.get(collider_handle) else {
            continue;
        };
        let body_pose = *body.next_position();
        let collider_offset = collider
            .position_wrt_parent()
            .copied()
            .unwrap_or_else(rapier3d::na::Isometry3::identity);
        let character_pos = body_pose * collider_offset;
        let shape = collider.shape();

        // Vertical speed: jump, ground contact, gravity.
        let was_grounded = state.grounded;
        let mut vertical = state.vertical_speed;
        let can_jump = was_grounded || state.seconds_since_grounded <= settings.coyote_time;
        state.jumped = false;
        if input.jump_speed > 0.0 && can_jump {
            vertical = input.jump_speed;
            state.jumped = true;
            state.seconds_since_grounded = settings.coyote_time + 1.0;
        } else if was_grounded && vertical < 0.0 {
            vertical = 0.0;
        }
        vertical -= gravity_magnitude * settings.gravity_scale * dt;
        vertical = vertical.max(-settings.max_fall_speed.abs());

        let planar = input.velocity - up * input.velocity.dot(up);
        let extra_up = input.velocity.dot(up);
        let desired = (planar + up * (vertical + extra_up)) * dt + input.root_motion;

        let mut filter = QueryFilter::new()
            .exclude_rigid_body(body_handle.0)
            .exclude_sensors();
        if let Some(layers) = layers.as_deref() {
            filter = filter.groups(layers.groups(layer.map_or(0, |layer| layer.layer)));
        }

        let controller = rapier_controller(settings, up);
        let mut collisions: Vec<CharacterCollision> = Vec::new();
        let movement = controller.move_shape(
            dt,
            rigid_body_set,
            collider_set,
            query_pipeline,
            shape,
            &character_pos,
            to_vector(desired),
            filter,
            |collision| collisions.push(collision),
        );
        let moved = from_vector(&movement.translation);

        if settings.push_dynamic_bodies && !collisions.is_empty() {
            let shape = collider_set
                .get(collider_handle)
                .map(|collider| collider.shared_shape().clone());
            if let Some(shape) = shape {
                controller.solve_character_collision_impulses(
                    dt,
                    rigid_body_set,
                    collider_set,
                    query_pipeline,
                    &*shape,
                    settings.mass.max(0.1),
                    &collisions,
                    filter,
                );
            }
        }

        // Head bump: moving up but blocked.
        let desired_up = desired.dot(up);
        if desired_up > 1e-5 && moved.dot(up) < desired_up * 0.5 && vertical > 0.0 {
            vertical = 0.0;
        }
        if movement.grounded && vertical < 0.0 {
            vertical = 0.0;
        }

        let (translation, rotation) = from_isometry(&body_pose);
        let new_world = (
            translation + moved,
            (input.root_rotation * rotation).normalize(),
        );
        if let Some(body) = rigid_body_set.get_mut(body_handle.0) {
            body.set_next_kinematic_position(to_isometry(new_world));
        }

        state.landed = movement.grounded && !was_grounded;
        state.grounded = movement.grounded;
        state.sliding = movement.is_sliding_down_slope;
        state.velocity = if dt > 0.0 { moved / dt } else { Vec3::ZERO };
        state.vertical_speed = vertical;
        state.seconds_since_grounded = if movement.grounded {
            0.0
        } else {
            state.seconds_since_grounded + dt
        };

        input.jump_speed = 0.0;
        input.root_motion = Vec3::ZERO;
        input.root_rotation = Quat::IDENTITY;

        let local = match parent_spaces.get(&entity) {
            Some(parent) => world_to_local(parent, new_world),
            None => new_world,
        };
        transform.translation = local.0;
        transform.rotation = local.1;
        if let Some(mut pose) = pose {
            pose.previous = pose.current;
            pose.current = new_world;
            pose.last_written = local;
            pose.interpolate = true;
        }
    }
}
