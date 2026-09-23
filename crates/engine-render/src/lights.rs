//! Light components for Forward+ (M4 — no shadows).

use bevy_ecs::prelude::Component;

#[derive(Component, Clone, Copy, Debug)]
pub struct DirectionalLight {
    pub direction: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            direction: [-0.35, -1.0, -0.2],
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
        }
    }
}

#[derive(Component, Clone, Copy, Debug)]
pub struct PointLight {
    pub color: [f32; 3],
    pub intensity: f32,
    pub range: f32,
}

impl Default for PointLight {
    fn default() -> Self {
        Self {
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
            range: 8.0,
        }
    }
}

#[derive(Component, Clone, Copy, Debug)]
pub struct SpotLight {
    pub direction: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
    pub range: f32,
    pub inner_cone_radians: f32,
    pub outer_cone_radians: f32,
}

impl Default for SpotLight {
    fn default() -> Self {
        Self {
            direction: [0.0, -1.0, 0.0],
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
            range: 10.0,
            inner_cone_radians: 0.2,
            outer_cone_radians: 0.4,
        }
    }
}
