//! Procedural stress scene for M4 gate: thousands of meshes + dozens of lights.

use crate::{DirectionalLight, MeshRenderable3d, PointLight, SpotLight};
use bevy_ecs::prelude::World;
use engine_assets::{MaterialHandle, MeshHandle, TextureHandle};
use engine_core::{
    Camera3d, EntityName, GlobalTransform, PersistentId, PrimaryCamera, RenderLayer3D,
    SpatialBundle, Transform, Visible,
};

pub struct StressSceneConfig {
    pub mesh_count: usize,
    pub point_lights: usize,
    pub spot_lights: usize,
    pub grid_spacing: f32,
}

impl Default for StressSceneConfig {
    fn default() -> Self {
        Self {
            mesh_count: 2000,
            point_lights: 48,
            spot_lights: 16,
            grid_spacing: 2.0,
        }
    }
}

fn spatial_at(x: f32, y: f32, z: f32) -> SpatialBundle {
    let transform = Transform::from_xyz(x, y, z);
    SpatialBundle {
        global_transform: GlobalTransform(transform.to_affine()),
        transform,
    }
}

/// Spawn a camera, directional light, a grid of mesh instances, and many local lights.
pub fn spawn_stress_scene(
    world: &mut World,
    mesh: MeshHandle,
    texture: TextureHandle,
    material: MaterialHandle,
    config: StressSceneConfig,
) {
    world.spawn((
        EntityName::new("stress-camera"),
        spatial_at(0.0, 40.0, 80.0),
        Camera3d::default(),
        PrimaryCamera,
        PersistentId(engine_core::EntityId::new_v4()),
    ));

    world.spawn((
        EntityName::new("sun"),
        SpatialBundle::default(),
        DirectionalLight::default(),
        PersistentId(engine_core::EntityId::new_v4()),
    ));

    let side = (config.mesh_count as f32).sqrt().ceil() as i32;
    let mut spawned = 0usize;
    for z in 0..side {
        for x in 0..side {
            if spawned >= config.mesh_count {
                break;
            }
            let xf = (x as f32 - side as f32 * 0.5) * config.grid_spacing;
            let zf = (z as f32 - side as f32 * 0.5) * config.grid_spacing;
            world.spawn((
                EntityName::new(format!("mesh-{spawned}")),
                spatial_at(xf, 0.0, zf),
                MeshRenderable3d::new(mesh, texture, material),
                Visible,
                RenderLayer3D,
                PersistentId(engine_core::EntityId::new_v4()),
            ));
            spawned += 1;
        }
    }

    for i in 0..config.point_lights {
        let angle = (i as f32 / config.point_lights.max(1) as f32) * std::f32::consts::TAU;
        let radius = 20.0;
        world.spawn((
            EntityName::new(format!("point-{i}")),
            spatial_at(angle.cos() * radius, 3.0, angle.sin() * radius),
            PointLight {
                cast_shadows: false,
                color: [1.0, 0.85, 0.7],
                intensity: 2.0,
                range: 12.0,
            },
            PersistentId(engine_core::EntityId::new_v4()),
        ));
    }

    for i in 0..config.spot_lights {
        let angle = (i as f32 / config.spot_lights.max(1) as f32) * std::f32::consts::TAU;
        world.spawn((
            EntityName::new(format!("spot-{i}")),
            spatial_at(angle.cos() * 8.0, 6.0, angle.sin() * 8.0),
            SpotLight {
                cast_shadows: false,
                direction: [0.0, -1.0, 0.0],
                color: [0.6, 0.8, 1.0],
                intensity: 3.0,
                range: 15.0,
                inner_cone_radians: 0.15,
                outer_cone_radians: 0.35,
            },
            PersistentId(engine_core::EntityId::new_v4()),
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_matches_gate_budget() {
        let config = StressSceneConfig::default();
        assert!(config.mesh_count >= 1000 && config.mesh_count <= 5000);
        assert!(config.point_lights + config.spot_lights >= 50);
        assert!(config.point_lights + config.spot_lights <= 100);
    }
}
