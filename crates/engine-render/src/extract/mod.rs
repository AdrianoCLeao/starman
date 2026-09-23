#![allow(dead_code)]

//! Extract gameplay world into a render-only [`RenderWorld`].

use bevy_ecs::entity::Entity;
use bevy_ecs::prelude::World;
use bevy_ecs::query::With;
use engine_assets::{MaterialHandle, MeshHandle, TextureHandle};
use engine_core::{
    GlobalTransform, HardeningConfig, PersistentId, RenderLayer2D, RenderLayer3D, Visible,
};
use engine_math::{Mat4, Vec3};

use crate::camera_uniforms::{extract_camera_uniform_2d, extract_camera_uniform_3d};
use crate::lights::{DirectionalLight, PointLight, SpotLight};
use crate::{Camera2dUniform, Camera3dUniform, MeshRenderable3d, SpriteRenderable2d};

#[derive(Clone, Copy, Debug, Default)]
pub struct Aabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl Aabb {
    pub fn from_center_extents(center: Vec3, extents: Vec3) -> Self {
        Self {
            min: (center - extents).to_array(),
            max: (center + extents).to_array(),
        }
    }

    pub fn center(self) -> Vec3 {
        Vec3::from_array([
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ])
    }
}

#[derive(Clone, Debug)]
pub struct ExtractedMesh {
    pub entity: Entity,
    pub pick_id: u32,
    pub mesh: MeshHandle,
    pub texture: TextureHandle,
    pub material: MaterialHandle,
    pub model: [[f32; 4]; 4],
    pub normal: [[f32; 4]; 4],
    pub aabb: Aabb,
    pub world_position: [f32; 3],
}

#[derive(Clone, Debug)]
pub struct ExtractedSprite {
    pub entity: Entity,
    pub pick_id: u32,
    pub texture: TextureHandle,
    pub model: [[f32; 4]; 4],
    pub color: [f32; 4],
    pub uv_rect: [f32; 4],
    pub sort_z: f32,
}

#[derive(Clone, Copy, Debug)]
pub enum ExtractedLightKind {
    Directional {
        direction: [f32; 3],
        color: [f32; 3],
        intensity: f32,
    },
    Point {
        position: [f32; 3],
        color: [f32; 3],
        intensity: f32,
        range: f32,
    },
    Spot {
        position: [f32; 3],
        direction: [f32; 3],
        color: [f32; 3],
        intensity: f32,
        range: f32,
        inner_cone: f32,
        outer_cone: f32,
    },
}

#[derive(Clone, Debug)]
pub struct ExtractedLight {
    pub entity: Entity,
    pub kind: ExtractedLightKind,
}

#[derive(Default)]
pub struct RenderWorld {
    pub meshes: Vec<ExtractedMesh>,
    pub sprites: Vec<ExtractedSprite>,
    pub lights: Vec<ExtractedLight>,
    pub(crate) camera_3d: Option<Camera3dUniform>,
    pub(crate) camera_2d: Option<Camera2dUniform>,
    pub view_proj: Option<[[f32; 4]; 4]>,
    pub camera_position: [f32; 3],
    pub selected: Vec<Entity>,
}

/// Copy render-relevant state out of the gameplay world.
pub fn extract_render_world(
    world: &mut World,
    width: u32,
    height: u32,
    selected: &[Entity],
) -> RenderWorld {
    let hardening = world
        .get_resource::<HardeningConfig>()
        .copied()
        .unwrap_or_default();
    let max_3d = hardening.max_draw_items_3d.max(1);
    let max_2d = hardening.max_draw_items_2d.max(1);

    let camera_3d = extract_camera_uniform_3d(world);
    let camera_2d = Some(extract_camera_uniform_2d(world, width, height));
    let (view_proj, camera_position) = camera_3d
        .map(|c| {
            (
                Some(c.view_proj),
                [
                    c.camera_position[0],
                    c.camera_position[1],
                    c.camera_position[2],
                ],
            )
        })
        .unwrap_or((None, [0.0, 0.0, 0.0]));

    let mut meshes = Vec::new();
    {
        let mut query = world.query_filtered::<(
            Entity,
            Option<&PersistentId>,
            &GlobalTransform,
            &MeshRenderable3d,
        ), (With<Visible>, With<RenderLayer3D>)>();
        for (entity, persistent, global, mesh) in query.iter(world) {
            if meshes.len() >= max_3d {
                break;
            }
            let model = Mat4::from(global.0);
            let normal = model.inverse().transpose();
            let translation = model.w_axis.truncate();
            let pick_id = persistent
                .map(|p| {
                    let bytes = p.0.as_uuid().as_u128().to_le_bytes();
                    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                })
                .unwrap_or(entity.to_bits() as u32);
            meshes.push(ExtractedMesh {
                entity,
                pick_id,
                mesh: mesh.mesh,
                texture: mesh.texture,
                material: mesh.material,
                model: model.to_cols_array_2d(),
                normal: normal.to_cols_array_2d(),
                aabb: Aabb::from_center_extents(translation, Vec3::splat(0.5)),
                world_position: translation.to_array(),
            });
        }
    }

    let mut sprites = Vec::new();
    {
        let mut query = world.query_filtered::<(
            Entity,
            Option<&PersistentId>,
            &GlobalTransform,
            &SpriteRenderable2d,
        ), (With<Visible>, With<RenderLayer2D>)>();
        for (entity, persistent, global, sprite) in query.iter(world) {
            if sprites.len() >= max_2d {
                break;
            }
            let model = Mat4::from(global.0)
                * Mat4::from_scale(Vec3::new(sprite.size[0], sprite.size[1], 1.0));
            let pick_id = persistent
                .map(|p| {
                    let bytes = p.0.as_uuid().as_u128().to_le_bytes();
                    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                })
                .unwrap_or(entity.to_bits() as u32);
            sprites.push(ExtractedSprite {
                entity,
                pick_id,
                texture: sprite.texture,
                model: model.to_cols_array_2d(),
                color: sprite.color,
                uv_rect: [
                    sprite.uv_min[0],
                    sprite.uv_min[1],
                    sprite.uv_max[0],
                    sprite.uv_max[1],
                ],
                sort_z: model.w_axis.z,
            });
        }
        sprites.sort_by(|a, b| {
            a.sort_z
                .partial_cmp(&b.sort_z)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    let mut lights = Vec::new();
    {
        let mut query = world.query::<(Entity, &GlobalTransform, &DirectionalLight)>();
        for (entity, global, light) in query.iter(world) {
            let direction = (Mat4::from(global.0)
                .transform_vector3(Vec3::from_array(light.direction)))
            .normalize_or_zero()
            .to_array();
            lights.push(ExtractedLight {
                entity,
                kind: ExtractedLightKind::Directional {
                    direction,
                    color: light.color,
                    intensity: light.intensity,
                },
            });
        }
    }
    {
        let mut query = world.query::<(Entity, &GlobalTransform, &PointLight)>();
        for (entity, global, light) in query.iter(world) {
            let position = Mat4::from(global.0).w_axis.truncate().to_array();
            lights.push(ExtractedLight {
                entity,
                kind: ExtractedLightKind::Point {
                    position,
                    color: light.color,
                    intensity: light.intensity,
                    range: light.range,
                },
            });
        }
    }
    {
        let mut query = world.query::<(Entity, &GlobalTransform, &SpotLight)>();
        for (entity, global, light) in query.iter(world) {
            let model = Mat4::from(global.0);
            lights.push(ExtractedLight {
                entity,
                kind: ExtractedLightKind::Spot {
                    position: model.w_axis.truncate().to_array(),
                    direction: model
                        .transform_vector3(Vec3::from_array(light.direction))
                        .normalize_or_zero()
                        .to_array(),
                    color: light.color,
                    intensity: light.intensity,
                    range: light.range,
                    inner_cone: light.inner_cone_radians,
                    outer_cone: light.outer_cone_radians,
                },
            });
        }
    }

    RenderWorld {
        meshes,
        sprites,
        lights,
        camera_3d,
        camera_2d,
        view_proj,
        camera_position,
        selected: selected.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_assets::AssetServer;
    use engine_core::{
        Camera3d, GlobalTransform, PrimaryCamera, RenderLayer3D, Transform, Visible,
    };
    use engine_math::glam::Affine3A;
    use std::path::PathBuf;

    fn assets_root() -> String {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/reference-project/assets")
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn extract_counts_meshes_and_lights() {
        let mut asset_server = AssetServer::new(assets_root());
        let mesh = asset_server
            .load_mesh_handle("meshes/cube.glb")
            .expect("mesh");
        let texture = asset_server
            .load_texture_handle("textures/placeholder.png")
            .expect("texture");
        let material = asset_server
            .load_material_handle("materials/default.ron")
            .expect("material");

        let mut world = bevy_ecs::world::World::new();
        world.spawn((
            Camera3d::default(),
            PrimaryCamera,
            GlobalTransform(Affine3A::IDENTITY),
            Transform::IDENTITY,
        ));
        for i in 0..3 {
            world.spawn((
                MeshRenderable3d::new(mesh, texture, material),
                Visible,
                RenderLayer3D,
                GlobalTransform(Transform::from_xyz(i as f32, 0.0, 0.0).to_affine()),
                Transform::from_xyz(i as f32, 0.0, 0.0),
            ));
        }
        world.spawn((
            PointLight::default(),
            GlobalTransform(Affine3A::IDENTITY),
            Transform::IDENTITY,
        ));
        world.spawn((
            DirectionalLight::default(),
            GlobalTransform(Affine3A::IDENTITY),
            Transform::IDENTITY,
        ));

        let extracted = extract_render_world(&mut world, 128, 128, &[]);
        assert_eq!(extracted.meshes.len(), 3);
        assert_eq!(extracted.lights.len(), 2);
        assert!(extracted.camera_3d.is_some());
    }
}
