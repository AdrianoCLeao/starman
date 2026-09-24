use bevy_ecs::prelude::{Component, World};
use bytemuck::{Pod, Zeroable};
use engine_assets::{AssetServer, MaterialHandle, MeshHandle, TextureHandle};
use engine_core::{EngineError, GameRuntime, Result, RuntimePlugin};
use std::sync::Arc;
use winit::window::Window;

use surface::{acquire_frame, choose_present_mode};

mod camera_uniforms;
#[cfg(test)]
mod camera_uniforms_tests;
mod capabilities;
pub mod components;
mod culling;
mod debug;
mod draw;
#[cfg(test)]
mod draw_tests;
pub mod extension;
mod extract;
mod forward_plus;
mod frame;
pub mod gpu;
mod graph;
pub mod layouts;
mod lod;
pub mod passes;
mod picking;
mod quality;
mod scene_adapter;
#[cfg(test)]
mod scene_adapter_tests;
pub mod shader;
mod shadows;
mod stress;
mod surface;
#[cfg(test)]
mod surface_tests;
mod taa;
mod texture_upload;
#[cfg(test)]
mod texture_upload_tests;
pub mod uniforms;

pub use capabilities::{negotiate_tier, CapabilityTier, NegotiatedCapabilities};
pub use components::{
    register_render_reflection_types, CameraRenderSettings, DirectionalLight, Environment,
    NotShadowCaster, NotShadowReceiver, PointLight, ReflectionProbe, SkyMode, SpotLight,
};
pub use culling::{build_batches_3d, frustum_cull_meshes, Frustum};
pub use debug::DebugView;
pub use extension::{
    EncodeContext, ExtensionNode, ExtractInfo, NodeTarget, PrepareContext, RenderExtension,
    RenderExtensions,
};
pub use extract::{extract_render_world, Aabb, ExtractedMesh, RenderWorld};
pub use forward_plus::{
    assign_clusters, build_gpu_lights, cull_lights_cpu, pack_cluster_buffers, ClusterGrid,
    ClusterLayout, GpuLight,
};
pub use frame::{
    create_frame_renderer_from_adapter, create_frame_renderer_from_device, read_texture_rgba8,
    FrameRenderer, FrameRendererConfig, RenderStats,
};
pub use graph::{PassId, RenderGraph};
pub use lod::LodGroup;
pub use picking::PickResult;
pub use quality::{QualityPreset, QualitySettings};
pub use scene_adapter::RenderSceneAdapter;
pub use shadows::{cascade_split_depths, fit_cascades, select_local_shadow_casters};
pub use stress::{spawn_stress_scene, StressSceneConfig};

/// Registers renderer components and the render-extension registry in a
/// runtime (reflection for lights, environment, camera settings, probes).
#[derive(Default)]
pub struct RenderPlugin;

impl RuntimePlugin for RenderPlugin {
    fn name(&self) -> &'static str {
        "engine::render"
    }

    fn build(&self, runtime: &mut GameRuntime) {
        runtime.init_resource::<RenderExtensions>();
        engine_reflect::with_reflection_registries(
            &mut runtime.world,
            |types, components, metadata| {
                register_render_reflection_types(types, components, metadata)
            },
        );
    }
}

#[derive(Component, Clone, Copy, Debug)]
pub struct MeshRenderable3d {
    pub mesh: MeshHandle,
    pub texture: TextureHandle,
    pub material: MaterialHandle,
}

impl MeshRenderable3d {
    pub fn new(mesh: MeshHandle, texture: TextureHandle, material: MaterialHandle) -> Self {
        Self {
            mesh,
            texture,
            material,
        }
    }
}

#[derive(Component, Clone, Copy, Debug)]
pub struct SpriteRenderable2d {
    pub texture: TextureHandle,
    pub size: [f32; 2],
    pub color: [f32; 4],
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
}

impl SpriteRenderable2d {
    pub fn new(texture: TextureHandle) -> Self {
        Self {
            texture,
            size: [1.0, 1.0],
            color: [1.0, 1.0, 1.0, 1.0],
            uv_min: [0.0, 0.0],
            uv_max: [1.0, 1.0],
        }
    }

    pub fn with_size(mut self, width: f32, height: f32) -> Self {
        self.size = [width.max(0.001), height.max(0.001)];
        self
    }

    pub fn with_color(mut self, rgba: [f32; 4]) -> Self {
        self.color = rgba;
        self
    }

    pub fn with_uv_rect(mut self, uv_min: [f32; 2], uv_max: [f32; 2]) -> Self {
        self.uv_min = uv_min;
        self.uv_max = uv_max;
        self
    }
}

pub struct RenderModule {
    state: Option<RenderState>,
    clear_color: wgpu::Color,
}

impl Default for RenderModule {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderModule {
    pub fn new() -> Self {
        Self {
            state: None,
            clear_color: wgpu::Color {
                r: 0.06,
                g: 0.08,
                b: 0.12,
                a: 1.0,
            },
        }
    }

    pub fn initialize_with_window(&mut self, window: Arc<Window>, vsync: bool) -> Result<()> {
        if self.state.is_some() {
            return Ok(());
        }

        let state = RenderState::new(window, vsync)?;
        log::info!(
            target: "engine::render",
            "Render backend initialized: {} ({:?})",
            state.adapter_info.name,
            state.adapter_info.backend
        );

        self.state = Some(state);
        Ok(())
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if let Some(state) = self.state.as_mut() {
            state.resize(width, height);
        }
    }

    pub fn tick(&mut self, world: &mut World, assets: &AssetServer) -> Result<()> {
        let Some(state) = self.state.as_mut() else {
            log::trace!(target: "engine::render", "Render tick skipped (backend not initialized)");
            return Ok(());
        };

        state.render(world, assets, self.clear_color)
    }

    pub fn backend_type_name(&self) -> &'static str {
        std::any::type_name::<wgpu::Backends>()
    }
}

pub struct ViewportRenderModule {
    frame: FrameRenderer,
}

impl ViewportRenderModule {
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        target_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let caps = NegotiatedCapabilities {
            tier: CapabilityTier::Tier1,
            adapter_name: "egui-viewport".to_owned(),
            backend: "shared".to_owned(),
            features: device.features(),
            limits: device.limits(),
        };
        Self {
            frame: FrameRenderer::new(
                device,
                queue,
                caps,
                target_format,
                width,
                height,
                FrameRendererConfig::default(),
            ),
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.frame.resize(width, height);
    }

    pub fn frame_renderer(&self) -> &FrameRenderer {
        &self.frame
    }

    pub fn frame_renderer_mut(&mut self) -> &mut FrameRenderer {
        &mut self.frame
    }

    pub fn render(
        &mut self,
        world: &mut World,
        assets: &AssetServer,
        target_view: &wgpu::TextureView,
    ) -> Result<()> {
        self.frame.render_to_view(world, assets, target_view)
    }
}

struct RenderState {
    _instance: wgpu::Instance,
    _window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    adapter_info: wgpu::AdapterInfo,
    config: wgpu::SurfaceConfiguration,
    frame: FrameRenderer,
    clear_color: wgpu::Color,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct Camera2dUniform {
    pub(crate) view_proj: [[f32; 4]; 4],
}

impl RenderState {
    fn new(window: Arc<Window>, vsync: bool) -> Result<Self> {
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance
            .create_surface(window.clone())
            .map_err(|error| EngineError::Render(format!("failed to create surface: {error}")))?;

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| EngineError::Render("failed to request a compatible adapter".to_owned()))?;

        let adapter_info = adapter.get_info();
        let (device, queue, caps) = crate::capabilities::request_device(&adapter)
            .map_err(|error| EngineError::Render(format!("failed to request device: {error}")))?;

        log::info!(
            target: "engine::render",
            "Negotiated {} on {} ({:?})",
            caps.tier.as_str(),
            adapter_info.name,
            adapter_info.backend
        );

        let capabilities = surface.get_capabilities(&adapter);
        let mut config = surface
            .get_default_config(&adapter, width, height)
            .ok_or_else(|| {
                EngineError::Render("surface does not expose a default configuration".to_owned())
            })?;

        config.format = capabilities
            .formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
            .unwrap_or(config.format);
        config.present_mode = choose_present_mode(vsync, &capabilities.present_modes);
        config.alpha_mode = capabilities
            .alpha_modes
            .first()
            .copied()
            .unwrap_or(wgpu::CompositeAlphaMode::Auto);

        surface.configure(&device, &config);
        let frame = FrameRenderer::new(
            device,
            queue,
            caps,
            config.format,
            width,
            height,
            FrameRendererConfig::default(),
        );

        Ok(Self {
            _instance: instance,
            _window: window,
            surface,
            adapter_info,
            config,
            frame,
            clear_color: wgpu::Color {
                r: 0.06,
                g: 0.08,
                b: 0.12,
                a: 1.0,
            },
        })
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.frame.device, &self.config);
        self.frame.resize(width, height);
    }

    fn render(
        &mut self,
        world: &mut World,
        assets: &AssetServer,
        clear_color: wgpu::Color,
    ) -> Result<()> {
        self.clear_color = clear_color;
        self.frame.clear_color = clear_color;
        let Some(frame) = acquire_frame(&self.surface, &self.frame.device, &self.config)? else {
            return Ok(());
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.frame.render_to_view(world, assets, &view)?;
        frame.present();
        Ok(())
    }
}

pub fn module_name() -> &'static str {
    "engine-render"
}
