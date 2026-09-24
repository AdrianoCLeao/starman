use engine_reflect::{
    ComponentRegistry, ReflectMetadataRegistry, ReflectRegistration, ReflectTypeRegistry,
};

use crate::character::{CharacterController, CharacterInput, CharacterState};
use crate::components::{
    CollisionLayer, ExternalForce, ExternalImpulse, RigidBodySettings, Sensor, Velocity,
};
use crate::joints::{JointKind, JointMotor, PhysicsJoint};
use crate::{ColliderShape3D, PhysicsMaterial, RigidBodyType};

pub fn register_physics_reflection_types(
    types: &mut ReflectTypeRegistry,
    components: &mut ComponentRegistry,
    metadata: &mut ReflectMetadataRegistry,
) {
    types.register::<engine_assets::AssetRef>();
    types.register::<JointKind>();
    types.register::<JointMotor>();
    RigidBodyType::register_reflect(types, components, metadata);
    ColliderShape3D::register_reflect(types, components, metadata);
    PhysicsMaterial::register_reflect(types, components, metadata);
    Sensor::register_reflect(types, components, metadata);
    CollisionLayer::register_reflect(types, components, metadata);
    RigidBodySettings::register_reflect(types, components, metadata);
    Velocity::register_reflect(types, components, metadata);
    ExternalForce::register_reflect(types, components, metadata);
    ExternalImpulse::register_reflect(types, components, metadata);
    PhysicsJoint::register_reflect(types, components, metadata);
    CharacterController::register_reflect(types, components, metadata);
    CharacterInput::register_reflect(types, components, metadata);
    CharacterState::register_reflect(types, components, metadata);
}

#[cfg(test)]
#[path = "reflect_tests.rs"]
mod tests;
