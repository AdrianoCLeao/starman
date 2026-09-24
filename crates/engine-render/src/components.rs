//! Authoring components owned by the renderer. All of them are reflected,
//! so they serialize in scenes, show in the inspector and are scriptable.

use bevy_ecs::prelude::Component;
use bevy_reflect::Reflect;
use engine_assets::AssetRef;

/// Sun-like light at infinity. The first directional light with
/// `cast_shadows` drives the cascaded shadow maps.
#[derive(Component, Clone, Copy, Debug, Reflect, engine_reflect::RegisterReflect)]
pub struct DirectionalLight {
    /// Direction the light travels in local space (rotated by the entity).
    pub direction: [f32; 3],
    #[engine_reflect(color)]
    pub color: [f32; 3],
    /// Illuminance multiplier.
    #[engine_reflect(range(min = 0.0, max = 100.0))]
    pub intensity: f32,
    pub cast_shadows: bool,
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            direction: [-0.35, -1.0, -0.2],
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
            cast_shadows: true,
        }
    }
}

#[derive(Component, Clone, Copy, Debug, Reflect, engine_reflect::RegisterReflect)]
pub struct PointLight {
    #[engine_reflect(color)]
    pub color: [f32; 3],
    /// Radiance at 1 m before the range window.
    #[engine_reflect(range(min = 0.0, max = 10000.0))]
    pub intensity: f32,
    #[engine_reflect(range(min = 0.01, max = 1000.0))]
    pub range: f32,
    pub cast_shadows: bool,
}

impl Default for PointLight {
    fn default() -> Self {
        Self {
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
            range: 8.0,
            cast_shadows: false,
        }
    }
}

#[derive(Component, Clone, Copy, Debug, Reflect, engine_reflect::RegisterReflect)]
pub struct SpotLight {
    pub direction: [f32; 3],
    #[engine_reflect(color)]
    pub color: [f32; 3],
    #[engine_reflect(range(min = 0.0, max = 10000.0))]
    pub intensity: f32,
    #[engine_reflect(range(min = 0.01, max = 1000.0))]
    pub range: f32,
    #[engine_reflect(degrees)]
    pub inner_cone_radians: f32,
    #[engine_reflect(degrees)]
    pub outer_cone_radians: f32,
    pub cast_shadows: bool,
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
            cast_shadows: false,
        }
    }
}

/// Where the sky and image-based lighting come from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Reflect)]
pub enum SkyMode {
    /// Analytic gradient sky with a sun disk (follows the first
    /// directional light when `sun_follows_light`).
    #[default]
    Procedural,
    /// Equirectangular HDR image (`.hdr`).
    Hdr,
}

/// Scene environment: sky, image-based lighting and ambient. One per
/// scene; the first one found is used.
#[derive(Component, Clone, Debug, Reflect, engine_reflect::RegisterReflect)]
pub struct Environment {
    pub mode: SkyMode,
    /// HDR panorama used when `mode` is `Hdr`.
    pub hdr: AssetRef,
    #[engine_reflect(color)]
    pub zenith_color: [f32; 3],
    #[engine_reflect(color)]
    pub horizon_color: [f32; 3],
    #[engine_reflect(color)]
    pub ground_color: [f32; 3],
    #[engine_reflect(range(min = 0.1, max = 16.0))]
    pub horizon_sharpness: f32,
    pub sun_follows_light: bool,
    #[engine_reflect(range(min = 0.0, max = 100.0))]
    pub sky_intensity: f32,
    /// Multiplier on diffuse and specular image-based lighting.
    #[engine_reflect(range(min = 0.0, max = 16.0))]
    pub ibl_intensity: f32,
    /// Yaw of the HDR panorama.
    #[engine_reflect(degrees)]
    pub rotation_radians: f32,
    /// Ambient tint applied to image-based lighting.
    #[engine_reflect(color)]
    pub ambient_tint: [f32; 3],
}

impl Default for Environment {
    fn default() -> Self {
        Self {
            mode: SkyMode::Procedural,
            hdr: AssetRef::default(),
            zenith_color: [0.18, 0.32, 0.62],
            horizon_color: [0.62, 0.72, 0.85],
            ground_color: [0.22, 0.2, 0.18],
            horizon_sharpness: 3.0,
            sun_follows_light: true,
            sky_intensity: 1.0,
            ibl_intensity: 1.0,
            rotation_radians: 0.0,
            ambient_tint: [1.0, 1.0, 1.0],
        }
    }
}

/// Per-camera image settings (exposure, fog, bloom, SSAO, TAA opt-out).
#[derive(Component, Clone, Debug, Reflect, engine_reflect::RegisterReflect)]
pub struct CameraRenderSettings {
    #[engine_reflect(range(min = 0.01, max = 64.0))]
    pub exposure: f32,
    pub fog_enabled: bool,
    #[engine_reflect(color)]
    pub fog_color: [f32; 3],
    #[engine_reflect(range(min = 0.0, max = 1.0))]
    pub fog_density: f32,
    #[engine_reflect(range(min = 0.0, max = 10.0))]
    pub fog_height_falloff: f32,
    pub fog_base_height: f32,
    #[engine_reflect(range(min = 0.0, max = 10000.0))]
    pub fog_start_distance: f32,
    #[engine_reflect(range(min = 0.0, max = 1.0))]
    pub fog_max_opacity: f32,
    #[engine_reflect(range(min = 0.0, max = 4.0))]
    pub bloom_intensity: f32,
    #[engine_reflect(range(min = 0.0, max = 16.0))]
    pub bloom_threshold: f32,
    #[engine_reflect(range(min = 0.05, max = 8.0))]
    pub ssao_radius: f32,
    #[engine_reflect(range(min = 0.0, max = 1.0))]
    pub ssao_intensity: f32,
}

impl Default for CameraRenderSettings {
    fn default() -> Self {
        Self {
            exposure: 1.0,
            fog_enabled: false,
            fog_color: [0.55, 0.62, 0.72],
            fog_density: 0.02,
            fog_height_falloff: 0.2,
            fog_base_height: 0.0,
            fog_start_distance: 5.0,
            fog_max_opacity: 0.9,
            bloom_intensity: 0.04,
            bloom_threshold: 1.0,
            ssao_radius: 0.5,
            ssao_intensity: 1.0,
        }
    }
}

/// Box-projected reflection probe. Set `bake` to request a (re)bake; the
/// renderer clears it once the cubemap is captured.
#[derive(Component, Clone, Copy, Debug, Reflect, engine_reflect::RegisterReflect)]
pub struct ReflectionProbe {
    /// Half extents of the influence/projection box (local axes = world).
    pub half_extents: [f32; 3],
    /// Distance over which the probe fades out at its box edges.
    #[engine_reflect(range(min = 0.01, max = 100.0))]
    pub blend_distance: f32,
    pub priority: i32,
    #[engine_reflect(range(min = 0.0, max = 16.0))]
    pub intensity: f32,
    pub bake: bool,
}

impl Default for ReflectionProbe {
    fn default() -> Self {
        Self {
            half_extents: [5.0, 3.0, 5.0],
            blend_distance: 1.0,
            priority: 0,
            intensity: 1.0,
            bake: true,
        }
    }
}

/// Opt-out markers for shadowing.
#[derive(Component, Clone, Copy, Debug, Default, Reflect, engine_reflect::RegisterReflect)]
pub struct NotShadowCaster;

#[derive(Component, Clone, Copy, Debug, Default, Reflect, engine_reflect::RegisterReflect)]
pub struct NotShadowReceiver;

/// Registers every renderer component with the reflection registries.
pub fn register_render_reflection_types(
    types: &mut engine_reflect::ReflectTypeRegistry,
    components: &mut engine_reflect::ComponentRegistry,
    metadata: &mut engine_reflect::ReflectMetadataRegistry,
) {
    use engine_reflect::ReflectRegistration;
    types.register::<SkyMode>();
    types.register::<AssetRef>();
    DirectionalLight::register_reflect(types, components, metadata);
    PointLight::register_reflect(types, components, metadata);
    SpotLight::register_reflect(types, components, metadata);
    Environment::register_reflect(types, components, metadata);
    CameraRenderSettings::register_reflect(types, components, metadata);
    ReflectionProbe::register_reflect(types, components, metadata);
    NotShadowCaster::register_reflect(types, components, metadata);
    NotShadowReceiver::register_reflect(types, components, metadata);
}
