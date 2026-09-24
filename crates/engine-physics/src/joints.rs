//! Joints between bodies, referencing the other body by its persistent
//! entity id so they survive scene save/load.

use std::collections::HashMap;

use bevy_ecs::prelude::*;
use bevy_reflect::Reflect;
use engine_core::PersistentId;
use engine_math::{Vec2, Vec3};
use rapier3d::prelude::{
    FixedJointBuilder, GenericJoint, JointAxis, Point, PrismaticJointBuilder, Real,
    RevoluteJointBuilder, RopeJointBuilder, SphericalJointBuilder, SpringJointBuilder, UnitVector,
};

use crate::components::{JointHandle3D, RigidBodyHandle3D};
use crate::mapping::PhysicsEntityHandles3D;
use crate::pose::to_vector;
use crate::world3d::PhysicsWorld3D;

#[derive(Clone, Copy, Debug, Default, PartialEq, Reflect)]
pub enum JointKind {
    #[default]
    /// Welds the two bodies together.
    Fixed,
    /// Hinge around `axis` (doors, wheels).
    Revolute,
    /// Slider along `axis` (pistons, lifts).
    Prismatic,
    /// Ball-and-socket.
    Spherical,
    /// Keeps the anchors at most `max_distance` apart.
    Rope { max_distance: f32 },
    /// Pulls the anchors towards `rest_length`.
    Spring {
        rest_length: f32,
        stiffness: f32,
        damping: f32,
    },
}

/// Drives a revolute/prismatic joint towards a position and/or velocity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Reflect)]
pub struct JointMotor {
    pub enabled: bool,
    pub target_position: f32,
    pub target_velocity: f32,
    pub stiffness: f32,
    pub damping: f32,
    /// 0 = unlimited.
    pub max_force: f32,
}

/// A joint from this entity's body to `target`'s body (or to the world
/// when `target` is empty).
#[derive(Component, Clone, Debug, PartialEq, Reflect, engine_reflect::RegisterReflect)]
pub struct PhysicsJoint {
    pub kind: JointKind,
    /// `EntityId` (UUID) of the other body's entity; empty = world.
    pub target: String,
    /// Anchor in this body's local space.
    pub local_anchor: Vec3,
    /// Anchor in the target body's local space (world space for world).
    pub target_anchor: Vec3,
    /// Hinge/slide axis in this body's local space.
    pub axis: Vec3,
    pub limits_enabled: bool,
    /// Radians (revolute) or meters (prismatic).
    pub limits: Vec2,
    pub motor: JointMotor,
    /// Whether the jointed bodies still collide with each other.
    pub contacts_enabled: bool,
}

impl Default for PhysicsJoint {
    fn default() -> Self {
        Self {
            kind: JointKind::Fixed,
            target: String::new(),
            local_anchor: Vec3::ZERO,
            target_anchor: Vec3::ZERO,
            axis: Vec3::Y,
            limits_enabled: false,
            limits: Vec2::new(-1.0, 1.0),
            motor: JointMotor::default(),
            contacts_enabled: false,
        }
    }
}

fn point(v: Vec3) -> Point<Real> {
    Point::new(v.x, v.y, v.z)
}

fn axis(v: Vec3) -> UnitVector<Real> {
    let v = if v.length_squared() > 1e-8 {
        v.normalize()
    } else {
        Vec3::Y
    };
    UnitVector::new_normalize(to_vector(v))
}

fn with_motor(mut joint: GenericJoint, joint_axis: JointAxis, motor: JointMotor) -> GenericJoint {
    if motor.enabled {
        joint.set_motor(
            joint_axis,
            motor.target_position,
            motor.target_velocity,
            motor.stiffness.max(0.0),
            motor.damping.max(0.0),
        );
        if motor.max_force > 0.0 {
            joint.set_motor_max_force(joint_axis, motor.max_force);
        }
    }
    joint
}

/// Builds the Rapier joint description for `joint`.
pub fn build_joint(joint: &PhysicsJoint) -> GenericJoint {
    let a1 = point(joint.local_anchor);
    let a2 = point(joint.target_anchor);
    let limits = [
        joint.limits.x.min(joint.limits.y),
        joint.limits.x.max(joint.limits.y),
    ];
    let motor = joint.motor;
    match joint.kind {
        JointKind::Fixed => FixedJointBuilder::new()
            .local_anchor1(a1)
            .local_anchor2(a2)
            .contacts_enabled(joint.contacts_enabled)
            .build()
            .into(),
        JointKind::Revolute => {
            let mut builder = RevoluteJointBuilder::new(axis(joint.axis))
                .local_anchor1(a1)
                .local_anchor2(a2)
                .contacts_enabled(joint.contacts_enabled);
            if joint.limits_enabled {
                builder = builder.limits(limits);
            }
            with_motor(builder.build().into(), JointAxis::AngX, motor)
        }
        JointKind::Prismatic => {
            let mut builder = PrismaticJointBuilder::new(axis(joint.axis))
                .local_anchor1(a1)
                .local_anchor2(a2)
                .contacts_enabled(joint.contacts_enabled);
            if joint.limits_enabled {
                builder = builder.limits(limits);
            }
            with_motor(builder.build().into(), JointAxis::LinX, motor)
        }
        JointKind::Spherical => SphericalJointBuilder::new()
            .local_anchor1(a1)
            .local_anchor2(a2)
            .contacts_enabled(joint.contacts_enabled)
            .build()
            .into(),
        JointKind::Rope { max_distance } => RopeJointBuilder::new(max_distance.max(0.0))
            .local_anchor1(a1)
            .local_anchor2(a2)
            .contacts_enabled(joint.contacts_enabled)
            .build()
            .into(),
        JointKind::Spring {
            rest_length,
            stiffness,
            damping,
        } => SpringJointBuilder::new(rest_length.max(0.0), stiffness.max(0.0), damping.max(0.0))
            .local_anchor1(a1)
            .local_anchor2(a2)
            .contacts_enabled(joint.contacts_enabled)
            .build()
            .into(),
    }
}

/// Creates, updates and removes Rapier joints for [`PhysicsJoint`]s.
pub fn sync_joints(
    mut commands: Commands,
    joints: Query<(
        Entity,
        Ref<PhysicsJoint>,
        Option<&JointHandle3D>,
        Option<&RigidBodyHandle3D>,
    )>,
    orphaned: Query<(Entity, &JointHandle3D), Without<PhysicsJoint>>,
    ids: Query<(Entity, &PersistentId)>,
    mut physics: Option<ResMut<PhysicsWorld3D>>,
    entity_handles: Option<Res<PhysicsEntityHandles3D>>,
) {
    let (Some(physics), Some(entity_handles)) = (physics.as_deref_mut(), entity_handles) else {
        return;
    };

    for (entity, handle) in &orphaned {
        physics.impulse_joint_set.remove(handle.handle, true);
        commands.entity(entity).remove::<JointHandle3D>();
    }

    let mut by_id: Option<HashMap<String, Entity>> = None;
    for (entity, joint, existing, body) in &joints {
        if let Some(existing) = existing {
            if !physics.impulse_joint_set.contains(existing.handle) {
                // A body was removed; recreate once both exist again.
                commands.entity(entity).remove::<JointHandle3D>();
                continue;
            }
            if existing.target != joint.target {
                physics.impulse_joint_set.remove(existing.handle, true);
                commands.entity(entity).remove::<JointHandle3D>();
                continue;
            }
            if joint.is_changed() {
                if let Some(rapier_joint) = physics.impulse_joint_set.get_mut(existing.handle) {
                    rapier_joint.data = build_joint(&joint);
                }
            }
            continue;
        }

        let Some(body) = body else {
            continue;
        };
        let target_body = if joint.target.is_empty() {
            physics.world_anchor()
        } else {
            let by_id = by_id.get_or_insert_with(|| {
                ids.iter()
                    .map(|(entity, id)| (id.0.to_string(), entity))
                    .collect()
            });
            let Some(target_body) = by_id
                .get(&joint.target)
                .and_then(|target| entity_handles.body(*target))
            else {
                continue; // target not spawned or not simulated yet
            };
            target_body
        };
        let handle =
            physics
                .impulse_joint_set
                .insert(body.0, target_body, build_joint(&joint), true);
        commands.entity(entity).insert(JointHandle3D {
            handle,
            target: joint.target.clone(),
        });
    }
}
