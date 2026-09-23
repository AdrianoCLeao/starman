use bevy_ecs::prelude::{Component, World};
use bytemuck::{Pod, Zeroable};
use engine_assets::{AssetServer, MaterialHandle, MeshHandle, TextureHandle};
use engine_core::{EngineError, Result};
use std::{mem::size_of, sync::Arc};
use winit::window::Window;

use surface::{acquire_frame, choose_present_mode};

mod camera_uniforms;
#[cfg(test)]
mod camera_uniforms_tests;
mod capabilities;
mod culling;
mod debug;
mod draw;
#[cfg(test)]
mod draw_tests;
mod extract;
mod forward_plus;
mod frame;
mod gpu;
mod gpu_resources;
#[cfg(test)]
mod gpu_resources_tests;
mod graph;
mod lights;
mod picking;
mod pipelines;
mod scene_adapter;
#[cfg(test)]
mod scene_adapter_tests;
mod shader;
mod stress;
mod surface;
#[cfg(test)]
mod surface_tests;
mod texture_upload;
#[cfg(test)]
mod texture_upload_tests;

pub use capabilities::{negotiate_tier, CapabilityTier, NegotiatedCapabilities};
pub use culling::{build_batches_3d, frustum_cull_meshes, Frustum};
pub use debug::DebugView;
pub use extract::{extract_render_world, Aabb, ExtractedMesh, RenderWorld};
pub use forward_plus::{
    build_gpu_lights, cull_lights_cpu, pack_cluster_buffers, ClusterGrid, GpuLight,
};
pub use frame::{
    create_frame_renderer_from_adapter, create_frame_renderer_from_device, FrameRenderer,
    FrameRendererConfig,
};
pub use graph::{PassId, RenderGraph};
pub use lights::{DirectionalLight, PointLight, SpotLight};
pub use picking::PickResult;
pub use scene_adapter::RenderSceneAdapter;
pub use stress::{spawn_stress_scene, StressSceneConfig};

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

pub(crate) struct DepthTarget {
    pub(crate) _texture: wgpu::Texture,
    pub(crate) view: wgpu::TextureView,
}

pub(crate) struct Pipeline3d {
    pub(crate) pipeline: wgpu::RenderPipeline,
    pub(crate) camera_buffer: wgpu::Buffer,
    pub(crate) camera_bind_group: wgpu::BindGroup,
    pub(crate) model_layout: wgpu::BindGroupLayout,
    pub(crate) material_layout: wgpu::BindGroupLayout,
}

pub(crate) struct Pipeline2d {
    pub(crate) pipeline: wgpu::RenderPipeline,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    pub(crate) sprite_layout: wgpu::BindGroupLayout,
    pub(crate) quad_vertex_buffer: wgpu::Buffer,
    pub(crate) quad_index_buffer: wgpu::Buffer,
    pub(crate) quad_index_count: u32,
}

pub(crate) struct GpuMesh {
    pub(crate) vertex_buffer: wgpu::Buffer,
    pub(crate) index_buffer: wgpu::Buffer,
    pub(crate) index_count: u32,
}

pub(crate) struct GpuTexture {
    pub(crate) texture: wgpu::Texture,
    pub(crate) view: wgpu::TextureView,
    pub(crate) sampler: wgpu::Sampler,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) revision: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct Camera3dUniform {
    pub(crate) view_proj: [[f32; 4]; 4],
    pub(crate) camera_position: [f32; 4],
    pub(crate) light_direction: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct Camera2dUniform {
    pub(crate) view_proj: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct ModelUniform {
    pub(crate) model: [[f32; 4]; 4],
    pub(crate) normal: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct MaterialUniform {
    pub(crate) base_color: [f32; 4],
    pub(crate) metallic_roughness: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuVertex {
    pub(crate) position: [f32; 3],
    pub(crate) normal: [f32; 3],
    pub(crate) uv: [f32; 2],
}

impl GpuVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2];

    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<GpuVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct SpriteQuadVertex {
    position: [f32; 2],
    uv: [f32; 2],
}

impl SpriteQuadVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2];

    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<SpriteQuadVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct SpriteInstance {
    pub(crate) model: [[f32; 4]; 4],
    pub(crate) color: [f32; 4],
    pub(crate) uv_rect: [f32; 4],
}

impl SpriteInstance {
    const ATTRIBUTES: [wgpu::VertexAttribute; 6] = wgpu::vertex_attr_array![
        2 => Float32x4,
        3 => Float32x4,
        4 => Float32x4,
        5 => Float32x4,
        6 => Float32x4,
        7 => Float32x4
    ];

    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<SpriteInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRIBUTES,
        }
    }
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
