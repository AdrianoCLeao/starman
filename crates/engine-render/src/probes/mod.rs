//! Reflection probe component + capture state.

use bevy_ecs::prelude::Component;
use engine_math::Vec3;

#[derive(Component, Clone, Copy, Debug)]
pub struct ReflectionProbe {
    pub radius: f32,
    pub priority: i32,
    pub intensity: f32,
    pub resolution: u32,
    pub dirty: bool,
}

impl Default for ReflectionProbe {
    fn default() -> Self {
        Self {
            radius: 10.0,
            priority: 0,
            intensity: 1.0,
            resolution: 128,
            dirty: true,
        }
    }
}

impl ReflectionProbe {
    pub fn to_data(self, position: Vec3) -> crate::ibl::ReflectionProbeData {
        crate::ibl::ReflectionProbeData {
            position,
            radius: self.radius,
            priority: self.priority,
            intensity: self.intensity,
        }
    }
}
