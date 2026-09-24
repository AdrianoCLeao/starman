//! Reflected physics components (authored in scenes, edited in the
//! inspector) and the runtime handle components the systems attach.

use bevy_ecs::prelude::{Bundle, Component};
use bevy_reflect::Reflect;
use engine_assets::AssetRef;
use engine_core::{GlobalTransform, PhysicsControlled, Transform};
use engine_math::{Quat, Vec3};
use rapier3d::prelude::{ColliderHandle, ImpulseJointHandle, RigidBodyHandle};

/// The Rapier body backing an entity (runtime only).
#[derive(Component, Clone, Copy, Debug)]
pub struct RigidBodyHandle3D(pub RigidBodyHandle);

/// The Rapier collider backing an entity (runtime only).
#[derive(Component, Clone, Copy, Debug)]
pub struct ColliderHandle3D(pub ColliderHandle);

/// The Rapier joint backing a [`crate::PhysicsJoint`] (runtime only).
#[derive(Component, Clone, Debug)]
pub struct JointHandle3D {
    pub handle: ImpulseJointHandle,
    /// Target the joint was created with; a change recreates the joint.
    pub target: String,
}

/// Marks an entity whose physics setup was rejected (e.g. a dynamic
/// trimesh), so it is reported once instead of every step. Removed when
/// the offending components change.
#[derive(Component, Clone, Debug)]
pub struct PhysicsRejected(pub String);

#[derive(
    Component, Clone, Copy, Debug, Default, PartialEq, Eq, Reflect, engine_reflect::RegisterReflect,
)]
pub enum RigidBodyType {
    #[default]
    Dynamic,
    /// Moved by gameplay through its `Transform` (or a character controller).
    Kinematic,
    Static,
}

/// Collision geometry. An entity with a shape but no [`RigidBodyType`]
/// becomes an extra collider of the nearest ancestor body, or a static
/// collider when no ancestor has a body (level geometry, trigger volumes).
#[derive(Component, Clone, Debug, PartialEq, Reflect, engine_reflect::RegisterReflect)]
pub enum ColliderShape3D {
    Box {
        half_extents: Vec3,
    },
    Sphere {
        radius: f32,
    },
    Capsule {
        half_height: f32,
        radius: f32,
    },
    Cylinder {
        half_height: f32,
        radius: f32,
    },
    Cone {
        half_height: f32,
        radius: f32,
    },
    /// Legacy placeholder: a 1×1 horizontal quad (static bodies only).
    Trimesh,
    /// Geometry of a mesh asset: exact triangles (static/kinematic only)
    /// or its convex hull (any body type).
    Mesh {
        mesh: AssetRef,
        convex: bool,
    },
}

impl Default for ColliderShape3D {
    fn default() -> Self {
        Self::Box {
            half_extents: Vec3::splat(0.5),
        }
    }
}

impl ColliderShape3D {
    /// Concave shapes cannot be simulated on dynamic bodies.
    pub fn is_concave(&self) -> bool {
        matches!(self, Self::Trimesh | Self::Mesh { convex: false, .. })
    }
}

#[derive(Component, Clone, Copy, Debug, Reflect, engine_reflect::RegisterReflect)]
pub struct PhysicsMaterial {
    #[engine_reflect(range(min = 0.0, max = 1.0))]
    pub restitution: f32,
    #[engine_reflect(range(min = 0.0, max = 1.0))]
    pub friction: f32,
    #[engine_reflect(range(min = 0.001, max = 10000.0))]
    pub density: f32,
}

impl Default for PhysicsMaterial {
    fn default() -> Self {
        Self {
            restitution: 0.3,
            friction: 0.7,
            density: 1.0,
        }
    }
}

/// Turns the entity's collider into a trigger volume: no contact
/// response, `TriggerEntered`/`TriggerExited` events for overlaps with
/// any body type (including kinematic character controllers).
#[derive(Component, Clone, Copy, Debug, Default, Reflect, engine_reflect::RegisterReflect)]
pub struct Sensor;

/// Collision layer by index into the project's physics layers
/// (`game.physics.layers`); the project collision matrix decides which
/// layers interact.
#[derive(
    Component, Clone, Copy, Debug, Default, PartialEq, Eq, Reflect, engine_reflect::RegisterReflect,
)]
pub struct CollisionLayer {
    #[engine_reflect(range(min = 0.0, max = 31.0))]
    pub layer: u8,
}

/// Per-body dynamics settings.
#[derive(Component, Clone, Copy, Debug, PartialEq, Reflect, engine_reflect::RegisterReflect)]
pub struct RigidBodySettings {
    #[engine_reflect(range(min = -10.0, max = 10.0))]
    pub gravity_scale: f32,
    #[engine_reflect(range(min = 0.0, max = 100.0))]
    pub linear_damping: f32,
    #[engine_reflect(range(min = 0.0, max = 100.0))]
    pub angular_damping: f32,
    /// Continuous collision detection for fast bodies.
    pub ccd: bool,
    pub can_sleep: bool,
    pub lock_translation_x: bool,
    pub lock_translation_y: bool,
    pub lock_translation_z: bool,
    pub lock_rotation_x: bool,
    pub lock_rotation_y: bool,
    pub lock_rotation_z: bool,
    /// Smooth rendering between fixed steps.
    pub interpolate: bool,
}

impl Default for RigidBodySettings {
    fn default() -> Self {
        Self {
            gravity_scale: 1.0,
            linear_damping: 0.0,
            angular_damping: 0.05,
            ccd: false,
            can_sleep: true,
            lock_translation_x: false,
            lock_translation_y: false,
            lock_translation_z: false,
            lock_rotation_x: false,
            lock_rotation_y: false,
            lock_rotation_z: false,
            interpolate: true,
        }
    }
}

/// Linear/angular velocity of a dynamic body (world space). Written back
/// after every step; setting it from gameplay overrides the body's
/// velocity.
#[derive(
    Component, Clone, Copy, Debug, Default, PartialEq, Reflect, engine_reflect::RegisterReflect,
)]
pub struct Velocity {
    pub linear: Vec3,
    pub angular: Vec3,
}

/// A continuous force/torque applied every step while present.
#[derive(
    Component, Clone, Copy, Debug, Default, PartialEq, Reflect, engine_reflect::RegisterReflect,
)]
pub struct ExternalForce {
    pub force: Vec3,
    pub torque: Vec3,
}

/// A one-shot impulse, applied at the next step and then zeroed.
#[derive(
    Component, Clone, Copy, Debug, Default, PartialEq, Reflect, engine_reflect::RegisterReflect,
)]
pub struct ExternalImpulse {
    pub impulse: Vec3,
    pub torque_impulse: Vec3,
}

/// Simulation poses of a body for render interpolation and teleport
/// detection (runtime only).
#[derive(Component, Clone, Copy, Debug)]
pub struct PhysicsPose {
    pub previous: (Vec3, Quat),
    pub current: (Vec3, Quat),
    /// The local transform the physics systems last wrote; a different
    /// `Transform` means gameplay moved (teleported) the entity.
    pub last_written: (Vec3, Quat),
    pub interpolate: bool,
}

impl PhysicsPose {
    pub fn new(world: (Vec3, Quat), local: (Vec3, Quat), interpolate: bool) -> Self {
        Self {
            previous: world,
            current: world,
            last_written: local,
            interpolate,
        }
    }

    /// Interpolated world pose at `alpha` between the last two steps.
    pub fn interpolated(&self, alpha: f32) -> (Vec3, Quat) {
        (
            self.previous.0.lerp(self.current.0, alpha),
            self.previous.1.slerp(self.current.1, alpha),
        )
    }
}

#[derive(Bundle)]
pub struct RigidBody3DBundle {
    pub body_type: RigidBodyType,
    pub shape: ColliderShape3D,
    pub material: PhysicsMaterial,
    pub transform: Transform,
    pub global_transform: GlobalTransform,
    pub physics_controlled: PhysicsControlled,
}

impl Default for RigidBody3DBundle {
    fn default() -> Self {
        Self {
            body_type: RigidBodyType::Dynamic,
            shape: ColliderShape3D::default(),
            material: PhysicsMaterial::default(),
            transform: Transform::default(),
            global_transform: GlobalTransform::default(),
            physics_controlled: PhysicsControlled,
        }
    }
}

#[cfg(test)]
#[path = "components_tests.rs"]
mod tests;
