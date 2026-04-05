use bevy_ecs::prelude::{Component, World};
use bytemuck::{Pod, Zeroable};
use engine_assets::{
    AssetId, AssetServer, MaterialData, MaterialHandle, MeshHandle, TextureHandle,
};
use engine_core::{EngineError, HardeningConfig, Result};
use std::{collections::HashMap, mem::size_of, sync::Arc};
use wgpu::util::DeviceExt;
use winit::window::Window;

use camera_uniforms::{extract_camera_uniform_2d, extract_camera_uniform_3d};
use draw::{
    build_draw_batches_2d, collect_draw_items_2d, collect_draw_items_3d, DrawItem2d, DrawItem3d,
};
use pipelines::{create_pipeline_2d, create_pipeline_3d};
use surface::{acquire_frame, choose_present_mode, create_depth_target};

mod camera_uniforms;
#[cfg(test)]
mod camera_uniforms_tests;
mod draw;
#[cfg(test)]
mod draw_tests;
mod gpu_resources;
#[cfg(test)]
mod gpu_resources_tests;
mod pipelines;
mod scene_adapter;
#[cfg(test)]
mod scene_adapter_tests;
mod surface;
#[cfg(test)]
mod surface_tests;
mod texture_upload;
#[cfg(test)]
mod texture_upload_tests;

pub use scene_adapter::RenderSceneAdapter;

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
    device: wgpu::Device,
    queue: wgpu::Queue,
    width: u32,
    height: u32,
    clear_color: wgpu::Color,
    depth_target: DepthTarget,
    pipeline_3d: Pipeline3d,
    pipeline_2d: Pipeline2d,
    gpu_meshes: HashMap<AssetId, GpuMesh>,
    gpu_textures: HashMap<AssetId, GpuTexture>,
}

impl ViewportRenderModule {
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        target_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let width = width.max(1);
        let height = height.max(1);

        Self {
            depth_target: create_depth_target(&device, width, height),
            pipeline_3d: create_pipeline_3d(&device, target_format),
            pipeline_2d: create_pipeline_2d(&device, target_format),
            device,
            queue,
            width,
            height,
            clear_color: wgpu::Color {
                r: 0.06,
                g: 0.08,
                b: 0.12,
                a: 1.0,
            },
            gpu_meshes: HashMap::new(),
            gpu_textures: HashMap::new(),
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        if self.width == width && self.height == height {
            return;
        }

        self.width = width;
        self.height = height;
        self.depth_target = create_depth_target(&self.device, width, height);
    }

    pub fn render(
        &mut self,
        world: &mut World,
        assets: &AssetServer,
        target_view: &wgpu::TextureView,
    ) -> Result<()> {
        let hardening = world
            .get_resource::<HardeningConfig>()
            .copied()
            .unwrap_or_default();

        if let Some(camera_uniform) = extract_camera_uniform_3d(world) {
            self.queue.write_buffer(
                &self.pipeline_3d.camera_buffer,
                0,
                bytemuck::bytes_of(&camera_uniform),
            );
        }

        let camera_2d_uniform = extract_camera_uniform_2d(world, self.width, self.height);
        self.queue.write_buffer(
            &self.pipeline_2d.camera_buffer,
            0,
            bytemuck::bytes_of(&camera_2d_uniform),
        );

        let draw_items_3d = collect_draw_items_3d(world, hardening.max_draw_items_3d.max(1));
        for draw_item in &draw_items_3d {
            self.ensure_gpu_mesh(draw_item.mesh, assets)?;
            self.ensure_gpu_texture(draw_item.texture, assets)?;
        }

        let draw_items_2d = collect_draw_items_2d(world, hardening.max_draw_items_2d.max(1));
        for draw_item in &draw_items_2d {
            self.ensure_gpu_texture(draw_item.texture, assets)?;
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("engine-render-viewport-encoder"),
            });

        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("engine-render-viewport-clear-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_target.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });
        }

        self.encode_3d_pass(&mut encoder, target_view, assets, &draw_items_3d);
        self.encode_2d_pass(&mut encoder, target_view, &draw_items_2d);

        self.queue.submit(std::iter::once(encoder.finish()));

        Ok(())
    }

    fn encode_3d_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        assets: &AssetServer,
        draw_items: &[DrawItem3d],
    ) {
        let fallback_material = MaterialData {
            base_color_factor: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0,
            roughness: 1.0,
        };

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("engine-render-viewport-3d-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_target.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        render_pass.set_pipeline(&self.pipeline_3d.pipeline);
        render_pass.set_bind_group(0, &self.pipeline_3d.camera_bind_group, &[]);

        let mut frame_model_buffers = Vec::with_capacity(draw_items.len());
        let mut frame_material_buffers = Vec::with_capacity(draw_items.len());
        let mut frame_model_bind_groups = Vec::with_capacity(draw_items.len());
        let mut frame_material_bind_groups = Vec::with_capacity(draw_items.len());

        for draw_item in draw_items {
            let Some(gpu_mesh) = self.gpu_meshes.get(&draw_item.mesh.id()) else {
                continue;
            };
            let Some(gpu_texture) = self.gpu_textures.get(&draw_item.texture.id()) else {
                continue;
            };

            let material = assets
                .material_payload(draw_item.material)
                .unwrap_or(&fallback_material);

            let model_uniform = ModelUniform {
                model: draw_item.model,
                normal: draw_item.normal,
            };
            let model_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("engine-render-model-uniform"),
                    contents: bytemuck::bytes_of(&model_uniform),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            frame_model_buffers.push(model_buffer);
            let model_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("engine-render-model-bind-group"),
                layout: &self.pipeline_3d.model_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame_model_buffers
                        .last()
                        .expect("model buffer inserted")
                        .as_entire_binding(),
                }],
            });
            frame_model_bind_groups.push(model_bind_group);

            let material_uniform = MaterialUniform {
                base_color: material.base_color_factor,
                metallic_roughness: [material.metallic, material.roughness, 0.0, 0.0],
            };
            let material_buffer =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("engine-render-material-uniform"),
                        contents: bytemuck::bytes_of(&material_uniform),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
            frame_material_buffers.push(material_buffer);
            let material_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("engine-render-material-bind-group"),
                layout: &self.pipeline_3d.material_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: frame_material_buffers
                            .last()
                            .expect("material buffer inserted")
                            .as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&gpu_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&gpu_texture.sampler),
                    },
                ],
            });
            frame_material_bind_groups.push(material_bind_group);

            let model_bind_group = frame_model_bind_groups
                .last()
                .expect("model bind group inserted");
            let material_bind_group = frame_material_bind_groups
                .last()
                .expect("material bind group inserted");

            render_pass.set_bind_group(1, model_bind_group, &[]);
            render_pass.set_bind_group(2, material_bind_group, &[]);
            render_pass.set_vertex_buffer(0, gpu_mesh.vertex_buffer.slice(..));
            render_pass
                .set_index_buffer(gpu_mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..gpu_mesh.index_count, 0, 0..1);
        }
    }

    fn encode_2d_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        draw_items: &[DrawItem2d],
    ) {
        if draw_items.is_empty() {
            return;
        }

        let instances: Vec<SpriteInstance> = draw_items
            .iter()
            .map(|draw_item| SpriteInstance {
                model: draw_item.model,
                color: draw_item.color,
                uv_rect: draw_item.uv_rect,
            })
            .collect();

        let instance_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("engine-render-sprite-instance-buffer"),
                contents: bytemuck::cast_slice(instances.as_slice()),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("engine-render-viewport-2d-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        render_pass.set_pipeline(&self.pipeline_2d.pipeline);
        render_pass.set_bind_group(0, &self.pipeline_2d.camera_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.pipeline_2d.quad_vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, instance_buffer.slice(..));
        render_pass.set_index_buffer(
            self.pipeline_2d.quad_index_buffer.slice(..),
            wgpu::IndexFormat::Uint16,
        );

        let draw_batches = build_draw_batches_2d(draw_items);
        let mut frame_sprite_bind_groups = Vec::with_capacity(draw_batches.len());

        for batch in draw_batches {
            let Some(gpu_texture) = self.gpu_textures.get(&batch.texture.id()) else {
                continue;
            };

            let sprite_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("engine-render-sprite-bind-group"),
                layout: &self.pipeline_2d.sprite_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&gpu_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&gpu_texture.sampler),
                    },
                ],
            });
            frame_sprite_bind_groups.push(sprite_bind_group);

            let bind_group = frame_sprite_bind_groups
                .last()
                .expect("sprite bind group inserted");
            render_pass.set_bind_group(1, bind_group, &[]);
            render_pass.draw_indexed(
                0..self.pipeline_2d.quad_index_count,
                0,
                batch.start as u32..batch.end as u32,
            );
        }
    }

    fn ensure_gpu_mesh(&mut self, handle: MeshHandle, assets: &AssetServer) -> Result<()> {
        gpu_resources::ensure_gpu_mesh(&self.device, &mut self.gpu_meshes, handle, assets)
    }

    fn ensure_gpu_texture(&mut self, handle: TextureHandle, assets: &AssetServer) -> Result<()> {
        gpu_resources::ensure_gpu_texture(
            &self.device,
            &self.queue,
            &mut self.gpu_textures,
            handle,
            assets,
        )
    }
}

struct RenderState {
    _instance: wgpu::Instance,
    _window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    adapter_info: wgpu::AdapterInfo,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    depth_target: DepthTarget,
    pipeline_3d: Pipeline3d,
    pipeline_2d: Pipeline2d,
    gpu_meshes: HashMap<AssetId, GpuMesh>,
    gpu_textures: HashMap<AssetId, GpuTexture>,
}

struct DepthTarget {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

struct Pipeline3d {
    pipeline: wgpu::RenderPipeline,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    model_layout: wgpu::BindGroupLayout,
    material_layout: wgpu::BindGroupLayout,
}

struct Pipeline2d {
    pipeline: wgpu::RenderPipeline,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    sprite_layout: wgpu::BindGroupLayout,
    quad_vertex_buffer: wgpu::Buffer,
    quad_index_buffer: wgpu::Buffer,
    quad_index_count: u32,
}

struct GpuMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
}

struct GpuTexture {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    width: u32,
    height: u32,
    revision: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Camera3dUniform {
    view_proj: [[f32; 4]; 4],
    camera_position: [f32; 4],
    light_direction: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Camera2dUniform {
    view_proj: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ModelUniform {
    model: [[f32; 4]; 4],
    normal: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MaterialUniform {
    base_color: [f32; 4],
    metallic_roughness: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuVertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
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
struct SpriteQuadVertex {
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
struct SpriteInstance {
    model: [[f32; 4]; 4],
    color: [f32; 4],
    uv_rect: [f32; 4],
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

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("engine-render-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|error| EngineError::Render(format!("failed to request device: {error}")))?;

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
        let depth_target = create_depth_target(&device, width, height);
        let pipeline_3d = create_pipeline_3d(&device, config.format);
        let pipeline_2d = create_pipeline_2d(&device, config.format);

        Ok(Self {
            _instance: instance,
            _window: window,
            surface,
            adapter_info,
            device,
            queue,
            config,
            depth_target,
            pipeline_3d,
            pipeline_2d,
            gpu_meshes: HashMap::new(),
            gpu_textures: HashMap::new(),
        })
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.depth_target = create_depth_target(&self.device, width, height);
    }

    fn render(
        &mut self,
        world: &mut World,
        assets: &AssetServer,
        clear_color: wgpu::Color,
    ) -> Result<()> {
        let hardening = world
            .get_resource::<HardeningConfig>()
            .copied()
            .unwrap_or_default();

        if let Some(camera_uniform) = extract_camera_uniform_3d(world) {
            self.queue.write_buffer(
                &self.pipeline_3d.camera_buffer,
                0,
                bytemuck::bytes_of(&camera_uniform),
            );
        }

        let camera_2d_uniform =
            extract_camera_uniform_2d(world, self.config.width, self.config.height);
        self.queue.write_buffer(
            &self.pipeline_2d.camera_buffer,
            0,
            bytemuck::bytes_of(&camera_2d_uniform),
        );

        let draw_items_3d = collect_draw_items_3d(world, hardening.max_draw_items_3d.max(1));
        for draw_item in &draw_items_3d {
            self.ensure_gpu_mesh(draw_item.mesh, assets)?;
            self.ensure_gpu_texture(draw_item.texture, assets)?;
        }

        let draw_items_2d = collect_draw_items_2d(world, hardening.max_draw_items_2d.max(1));
        for draw_item in &draw_items_2d {
            self.ensure_gpu_texture(draw_item.texture, assets)?;
        }

        let Some(frame) = acquire_frame(&self.surface, &self.device, &self.config)? else {
            return Ok(());
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("engine-render-main-encoder"),
            });

        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("engine-render-clear-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_target.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });
        }

        self.encode_3d_pass(&mut encoder, &view, assets, &draw_items_3d);
        self.encode_2d_pass(&mut encoder, &view, &draw_items_2d);

        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
        Ok(())
    }

    fn encode_3d_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        assets: &AssetServer,
        draw_items: &[DrawItem3d],
    ) {
        let fallback_material = MaterialData {
            base_color_factor: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0,
            roughness: 1.0,
        };

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("engine-render-3d-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_target.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        render_pass.set_pipeline(&self.pipeline_3d.pipeline);
        render_pass.set_bind_group(0, &self.pipeline_3d.camera_bind_group, &[]);

        let mut frame_model_buffers = Vec::with_capacity(draw_items.len());
        let mut frame_material_buffers = Vec::with_capacity(draw_items.len());
        let mut frame_model_bind_groups = Vec::with_capacity(draw_items.len());
        let mut frame_material_bind_groups = Vec::with_capacity(draw_items.len());

        for draw_item in draw_items {
            let Some(gpu_mesh) = self.gpu_meshes.get(&draw_item.mesh.id()) else {
                continue;
            };
            let Some(gpu_texture) = self.gpu_textures.get(&draw_item.texture.id()) else {
                continue;
            };

            let material = assets
                .material_payload(draw_item.material)
                .unwrap_or(&fallback_material);

            let model_uniform = ModelUniform {
                model: draw_item.model,
                normal: draw_item.normal,
            };
            let model_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("engine-render-model-uniform"),
                    contents: bytemuck::bytes_of(&model_uniform),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            frame_model_buffers.push(model_buffer);
            let model_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("engine-render-model-bind-group"),
                layout: &self.pipeline_3d.model_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame_model_buffers
                        .last()
                        .expect("model buffer inserted")
                        .as_entire_binding(),
                }],
            });
            frame_model_bind_groups.push(model_bind_group);

            let material_uniform = MaterialUniform {
                base_color: material.base_color_factor,
                metallic_roughness: [material.metallic, material.roughness, 0.0, 0.0],
            };
            let material_buffer =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("engine-render-material-uniform"),
                        contents: bytemuck::bytes_of(&material_uniform),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
            frame_material_buffers.push(material_buffer);
            let material_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("engine-render-material-bind-group"),
                layout: &self.pipeline_3d.material_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: frame_material_buffers
                            .last()
                            .expect("material buffer inserted")
                            .as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&gpu_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&gpu_texture.sampler),
                    },
                ],
            });
            frame_material_bind_groups.push(material_bind_group);

            let model_bind_group = frame_model_bind_groups
                .last()
                .expect("model bind group inserted");
            let material_bind_group = frame_material_bind_groups
                .last()
                .expect("material bind group inserted");

            render_pass.set_bind_group(1, model_bind_group, &[]);
            render_pass.set_bind_group(2, material_bind_group, &[]);
            render_pass.set_vertex_buffer(0, gpu_mesh.vertex_buffer.slice(..));
            render_pass
                .set_index_buffer(gpu_mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..gpu_mesh.index_count, 0, 0..1);
        }
    }

    fn encode_2d_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        draw_items: &[DrawItem2d],
    ) {
        if draw_items.is_empty() {
            return;
        }

        let instances: Vec<SpriteInstance> = draw_items
            .iter()
            .map(|draw_item| SpriteInstance {
                model: draw_item.model,
                color: draw_item.color,
                uv_rect: draw_item.uv_rect,
            })
            .collect();

        let instance_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("engine-render-sprite-instance-buffer"),
                contents: bytemuck::cast_slice(instances.as_slice()),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("engine-render-2d-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        render_pass.set_pipeline(&self.pipeline_2d.pipeline);
        render_pass.set_bind_group(0, &self.pipeline_2d.camera_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.pipeline_2d.quad_vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, instance_buffer.slice(..));
        render_pass.set_index_buffer(
            self.pipeline_2d.quad_index_buffer.slice(..),
            wgpu::IndexFormat::Uint16,
        );

        let draw_batches = build_draw_batches_2d(draw_items);
        let mut frame_sprite_bind_groups = Vec::with_capacity(draw_batches.len());

        for batch in draw_batches {
            let Some(gpu_texture) = self.gpu_textures.get(&batch.texture.id()) else {
                continue;
            };

            let sprite_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("engine-render-sprite-bind-group"),
                layout: &self.pipeline_2d.sprite_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&gpu_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&gpu_texture.sampler),
                    },
                ],
            });
            frame_sprite_bind_groups.push(sprite_bind_group);

            let bind_group = frame_sprite_bind_groups
                .last()
                .expect("sprite bind group inserted");
            render_pass.set_bind_group(1, bind_group, &[]);
            render_pass.draw_indexed(
                0..self.pipeline_2d.quad_index_count,
                0,
                batch.start as u32..batch.end as u32,
            );
        }
    }

    fn ensure_gpu_mesh(&mut self, handle: MeshHandle, assets: &AssetServer) -> Result<()> {
        gpu_resources::ensure_gpu_mesh(&self.device, &mut self.gpu_meshes, handle, assets)
    }

    fn ensure_gpu_texture(&mut self, handle: TextureHandle, assets: &AssetServer) -> Result<()> {
        gpu_resources::ensure_gpu_texture(
            &self.device,
            &self.queue,
            &mut self.gpu_textures,
            handle,
            assets,
        )
    }
}

const MESH3D_SHADER: &str = r#"
struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_position: vec4<f32>,
    light_direction: vec4<f32>,
};

struct ModelUniform {
    model: mat4x4<f32>,
    normal: mat4x4<f32>,
};

struct MaterialUniform {
    base_color: vec4<f32>,
    metallic_roughness: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> model: ModelUniform;
@group(2) @binding(0) var<uniform> material: MaterialUniform;
@group(2) @binding(1) var t_albedo: texture_2d<f32>;
@group(2) @binding(2) var s_albedo: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    let world_pos = model.model * vec4<f32>(input.position, 1.0);
    output.clip_position = camera.view_proj * world_pos;
    output.world_position = world_pos.xyz;
    output.world_normal = normalize((model.normal * vec4<f32>(input.normal, 0.0)).xyz);
    output.uv = input.uv;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let albedo_sample = textureSample(t_albedo, s_albedo, input.uv);
    let base_color = albedo_sample * material.base_color;
    let metallic = clamp(material.metallic_roughness.x, 0.0, 1.0);
    let roughness = clamp(material.metallic_roughness.y, 0.04, 1.0);

    let n = normalize(input.world_normal);
    let l = normalize(-camera.light_direction.xyz);
    let v = normalize(camera.camera_position.xyz - input.world_position);
    let h = normalize(l + v);

    let ndotl = max(dot(n, l), 0.0);
    let ndoth = max(dot(n, h), 0.0);
    let spec_power = mix(256.0, 4.0, roughness);
    let specular_strength = pow(ndoth, spec_power);

    let diffuse = base_color.rgb * ndotl * (1.0 - metallic);
    let specular = vec3<f32>(specular_strength) * mix(0.04, 1.0, metallic);
    let ambient = base_color.rgb * 0.08;

    let color = ambient + diffuse + specular;
    return vec4<f32>(color, base_color.a);
}
"#;

const SPRITE2D_SHADER: &str = r#"
struct CameraUniform {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var t_sprite: texture_2d<f32>;
@group(1) @binding(1) var s_sprite: sampler;

struct VertexInput {
    @location(0) local_position: vec2<f32>,
    @location(1) quad_uv: vec2<f32>,
    @location(2) model_0: vec4<f32>,
    @location(3) model_1: vec4<f32>,
    @location(4) model_2: vec4<f32>,
    @location(5) model_3: vec4<f32>,
    @location(6) color: vec4<f32>,
    @location(7) uv_rect: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    let model = mat4x4<f32>(input.model_0, input.model_1, input.model_2, input.model_3);
    let world_position = model * vec4<f32>(input.local_position, 0.0, 1.0);

    output.clip_position = camera.view_proj * world_position;
    output.uv = mix(input.uv_rect.xy, input.uv_rect.zw, input.quad_uv);
    output.color = input.color;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let texel = textureSample(t_sprite, s_sprite, input.uv);
    return texel * input.color;
}
"#;

pub fn module_name() -> &'static str {
    "engine-render"
}
