//! Extract: copy everything the renderer needs out of the gameplay world
//! into a render-only [`RenderWorld`]. After this step the renderer never
//! touches the ECS world during the frame (ADR 0009).

use bevy_ecs::entity::Entity;
use bevy_ecs::prelude::World;
use bevy_ecs::query::With;
use engine_assets::{MaterialHandle, MeshHandle, TextureHandle};
use engine_core::{
    Camera3d, DebugDraw, DebugLine, GlobalTransform, HardeningConfig, PersistentId, PrimaryCamera,
    RenderLayer2D, RenderLayer3D, SkinPalette, Visible,
};
use engine_math::{Mat4, Vec3};

use crate::camera_uniforms::extract_camera_uniform_2d;
use crate::components::{
    CameraRenderSettings, DirectionalLight, Environment, NotShadowCaster, NotShadowReceiver,
    PointLight, ReflectionProbe, SpotLight,
};
use crate::{Camera2dUniform, MeshRenderable3d, SpriteRenderable2d};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
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

    pub fn from_min_max(min: Vec3, max: Vec3) -> Self {
        Self {
            min: min.to_array(),
            max: max.to_array(),
        }
    }

    pub fn center(self) -> Vec3 {
        (Vec3::from_array(self.min) + Vec3::from_array(self.max)) * 0.5
    }

    pub fn half_extents(self) -> Vec3 {
        (Vec3::from_array(self.max) - Vec3::from_array(self.min)) * 0.5
    }

    /// World-space AABB of this local box under `transform` (Arvo).
    pub fn transformed(self, transform: Mat4) -> Self {
        let center = transform.transform_point3(self.center());
        let half = self.half_extents();
        let extents = Vec3::new(
            transform.x_axis.x.abs() * half.x
                + transform.y_axis.x.abs() * half.y
                + transform.z_axis.x.abs() * half.z,
            transform.x_axis.y.abs() * half.x
                + transform.y_axis.y.abs() * half.y
                + transform.z_axis.y.abs() * half.z,
            transform.x_axis.z.abs() * half.x
                + transform.y_axis.z.abs() * half.y
                + transform.z_axis.z.abs() * half.z,
        );
        Self::from_center_extents(center, extents)
    }

    pub fn union(self, other: Self) -> Self {
        Self {
            min: [
                self.min[0].min(other.min[0]),
                self.min[1].min(other.min[1]),
                self.min[2].min(other.min[2]),
            ],
            max: [
                self.max[0].max(other.max[0]),
                self.max[1].max(other.max[1]),
                self.max[2].max(other.max[2]),
            ],
        }
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
    /// Conservative placeholder until the mesh's own bounds are known in
    /// prepare (unit box around the origin).
    pub aabb: Aabb,
    pub world_position: [f32; 3],
    /// Range of this instance's joint matrices in [`RenderWorld::palettes`].
    pub palette: Option<(u32, u32)>,
    /// Model-space posed bounds reported by the animation system.
    pub skin_bounds: Option<Aabb>,
    pub cast_shadows: bool,
    pub receive_shadows: bool,
    pub selected: bool,
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
    pub cast_shadows: bool,
}

#[derive(Clone, Debug)]
pub struct ExtractedCamera {
    pub entity: Entity,
    /// Camera-to-world.
    pub world: Mat4,
    pub fov_y_radians: f32,
    pub near: f32,
    pub far: f32,
    pub settings: CameraRenderSettings,
}

impl ExtractedCamera {
    pub fn view(&self) -> Mat4 {
        self.world.inverse()
    }

    pub fn position(&self) -> Vec3 {
        self.world.w_axis.truncate()
    }

    pub fn projection(&self, width: u32, height: u32) -> Mat4 {
        let aspect = width.max(1) as f32 / height.max(1) as f32;
        Mat4::perspective_rh(self.fov_y_radians, aspect, self.near, self.far)
    }
}

#[derive(Clone, Debug)]
pub struct ExtractedProbe {
    pub entity: Entity,
    pub position: [f32; 3],
    pub probe: ReflectionProbe,
    pub persistent: Option<PersistentId>,
}

#[derive(Default)]
pub struct RenderWorld {
    pub meshes: Vec<ExtractedMesh>,
    pub sprites: Vec<ExtractedSprite>,
    pub lights: Vec<ExtractedLight>,
    pub camera: Option<ExtractedCamera>,
    pub(crate) camera_2d: Option<Camera2dUniform>,
    pub environment: Option<Environment>,
    pub probes: Vec<ExtractedProbe>,
    /// Joint matrices of every skinned instance, concatenated.
    pub palettes: Vec<[[f32; 4]; 4]>,
    pub debug_lines: Vec<DebugLine>,
    /// View-projection of the primary camera for the extract viewport
    /// (kept for tools and tests).
    pub view_proj: Option<[[f32; 4]; 4]>,
    pub camera_position: [f32; 3],
    pub selected: Vec<Entity>,
}

fn pick_id_for(entity: Entity, persistent: Option<&PersistentId>) -> u32 {
    let id = persistent
        .map(|p| {
            let bytes = p.0.as_uuid().as_u128().to_le_bytes();
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
        })
        .unwrap_or(entity.to_bits() as u32);
    // 0 is reserved for "nothing" in the pick buffer.
    id.max(1)
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

    let camera = {
        let mut query = world.query_filtered::<(
            Entity,
            &Camera3d,
            &GlobalTransform,
            Option<&CameraRenderSettings>,
        ), With<PrimaryCamera>>();
        query
            .iter(world)
            .next()
            .map(|(entity, camera, global, settings)| ExtractedCamera {
                entity,
                world: Mat4::from(global.0),
                fov_y_radians: camera.fov_y_radians,
                near: camera.near.max(1e-3),
                far: camera.far.max(camera.near + 1e-2),
                settings: settings.cloned().unwrap_or_default(),
            })
    };
    let camera_2d = Some(extract_camera_uniform_2d(world, width, height));
    let (view_proj, camera_position) = camera
        .as_ref()
        .map(|c| {
            (
                Some((c.projection(width, height) * c.view()).to_cols_array_2d()),
                c.position().to_array(),
            )
        })
        .unwrap_or((None, [0.0; 3]));

    let mut meshes = Vec::new();
    let mut palettes = Vec::new();
    {
        let mut query = world.query_filtered::<(
            Entity,
            Option<&PersistentId>,
            &GlobalTransform,
            &MeshRenderable3d,
            Option<&SkinPalette>,
            Option<&NotShadowCaster>,
            Option<&NotShadowReceiver>,
        ), (With<Visible>, With<RenderLayer3D>)>();
        for (entity, persistent, global, mesh, skin, no_cast, no_receive) in query.iter(world) {
            if meshes.len() >= max_3d {
                break;
            }
            let model = Mat4::from(global.0);
            let normal = model.inverse().transpose();
            let translation = model.w_axis.truncate();
            let palette = skin.filter(|s| !s.joint_matrices.is_empty()).map(|skin| {
                let offset = palettes.len() as u32;
                palettes.extend(skin.joint_matrices.iter().map(|m| m.to_cols_array_2d()));
                (offset, skin.joint_matrices.len() as u32)
            });
            meshes.push(ExtractedMesh {
                entity,
                pick_id: pick_id_for(entity, persistent),
                mesh: mesh.mesh,
                texture: mesh.texture,
                material: mesh.material,
                model: model.to_cols_array_2d(),
                normal: normal.to_cols_array_2d(),
                aabb: Aabb::from_center_extents(translation, Vec3::splat(0.5)),
                world_position: translation.to_array(),
                palette,
                skin_bounds: skin
                    .and_then(|s| s.bounds)
                    .map(|(min, max)| Aabb::from_min_max(min, max)),
                cast_shadows: no_cast.is_none(),
                receive_shadows: no_receive.is_none(),
                selected: selected.contains(&entity),
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
            sprites.push(ExtractedSprite {
                entity,
                pick_id: pick_id_for(entity, persistent),
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
            let direction = Mat4::from(global.0)
                .transform_vector3(Vec3::from_array(light.direction))
                .normalize_or_zero()
                .to_array();
            lights.push(ExtractedLight {
                entity,
                kind: ExtractedLightKind::Directional {
                    direction,
                    color: light.color,
                    intensity: light.intensity,
                },
                cast_shadows: light.cast_shadows,
            });
        }
        // Stable order: the shadowed sun first.
        lights.sort_by_key(|light| !light.cast_shadows);
    }
    {
        let mut query = world.query::<(Entity, &GlobalTransform, &PointLight)>();
        for (entity, global, light) in query.iter(world) {
            lights.push(ExtractedLight {
                entity,
                kind: ExtractedLightKind::Point {
                    position: Mat4::from(global.0).w_axis.truncate().to_array(),
                    color: light.color,
                    intensity: light.intensity,
                    range: light.range,
                },
                cast_shadows: light.cast_shadows,
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
                cast_shadows: light.cast_shadows,
            });
        }
    }

    let environment = {
        let mut query = world.query::<&Environment>();
        query.iter(world).next().cloned()
    };

    let probes = {
        let mut query = world.query::<(
            Entity,
            &GlobalTransform,
            &ReflectionProbe,
            Option<&PersistentId>,
        )>();
        let mut probes: Vec<ExtractedProbe> = query
            .iter(world)
            .map(|(entity, global, probe, persistent)| ExtractedProbe {
                entity,
                position: global.translation().to_array(),
                probe: *probe,
                persistent: persistent.copied(),
            })
            .collect();
        probes.sort_by_key(|probe| std::cmp::Reverse(probe.probe.priority));
        probes
    };

    let debug_lines = world
        .get_resource_mut::<DebugDraw>()
        .map(|mut draw| draw.take_lines())
        .unwrap_or_default();

    RenderWorld {
        meshes,
        sprites,
        lights,
        camera,
        camera_2d,
        environment,
        probes,
        palettes,
        debug_lines,
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
    fn extract_counts_meshes_lights_and_palettes() {
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

        let mut world = World::new();
        world.spawn((
            Camera3d::default(),
            PrimaryCamera,
            GlobalTransform(Affine3A::IDENTITY),
            Transform::IDENTITY,
        ));
        for i in 0..3 {
            let mut entity = world.spawn((
                MeshRenderable3d::new(mesh, texture, material),
                Visible,
                RenderLayer3D,
                GlobalTransform(Transform::from_xyz(i as f32, 0.0, 0.0).to_affine()),
                Transform::from_xyz(i as f32, 0.0, 0.0),
            ));
            if i == 1 {
                entity.insert((
                    SkinPalette {
                        joint_matrices: vec![Mat4::IDENTITY; 4],
                        bounds: None,
                    },
                    NotShadowCaster,
                ));
            }
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
        let mut debug = DebugDraw::default();
        debug.cross(Vec3::ZERO, 1.0, [1.0; 4]);
        world.insert_resource(debug);

        let extracted = extract_render_world(&mut world, 128, 128, &[]);
        assert_eq!(extracted.meshes.len(), 3);
        assert_eq!(extracted.lights.len(), 2);
        assert!(extracted.camera.is_some());
        assert_eq!(extracted.palettes.len(), 4);
        assert_eq!(
            extracted
                .meshes
                .iter()
                .filter(|m| m.palette.is_some())
                .count(),
            1
        );
        assert_eq!(
            extracted.meshes.iter().filter(|m| !m.cast_shadows).count(),
            1
        );
        assert_eq!(extracted.debug_lines.len(), 3);
        assert!(world.resource::<DebugDraw>().lines().is_empty());
    }

    #[test]
    fn transformed_aabb_is_conservative() {
        let local = Aabb::from_center_extents(Vec3::ZERO, Vec3::ONE);
        let rotated = local.transformed(Mat4::from_rotation_y(std::f32::consts::FRAC_PI_4));
        let half = rotated.half_extents();
        assert!((half.x - 2f32.sqrt()).abs() < 1e-4);
        assert!((half.y - 1.0).abs() < 1e-4);
    }
}
