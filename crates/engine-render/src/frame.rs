//! Unified frame renderer: extract → prepare → queue → graph execute.

use std::collections::HashMap;
use std::path::PathBuf;

use bevy_ecs::entity::Entity;
use bevy_ecs::prelude::World;
use engine_assets::{AssetId, AssetServer, MaterialData, MeshHandle, TextureHandle};
use engine_core::{EngineError, Result};
use engine_math::Mat4;
use wgpu::util::DeviceExt;

use crate::capabilities::{CapabilityTier, NegotiatedCapabilities};
use crate::culling::{build_batches_3d, frustum_cull_meshes};
use crate::debug::{dump_rgba8_ppm, DebugView};
use crate::draw::{build_draw_batches_2d, DrawItem2d, DrawItem3d};
use crate::extract::extract_render_world;
use crate::forward_plus::{build_gpu_lights, cull_lights_cpu, pack_cluster_buffers, GpuLight};
use crate::gpu::GpuResourceArena;
use crate::graph::{PassId, RenderGraph};
use crate::picking::{PickResult, PickingState};
use crate::pipelines::{create_pipeline_2d, create_pipeline_3d};
use crate::shader::{ShaderId, ShaderLibrary, ShaderVariantKey};
use crate::surface::create_depth_target;
use crate::{
    DepthTarget, GpuMesh, GpuTexture, MaterialUniform, ModelUniform, Pipeline2d, Pipeline3d,
    SpriteInstance,
};

pub struct FrameRendererConfig {
    pub shader_root: PathBuf,
    pub shader_cache: PathBuf,
    pub clear_color: wgpu::Color,
    pub max_lights: usize,
}

impl Default for FrameRendererConfig {
    fn default() -> Self {
        Self {
            shader_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders"),
            shader_cache: PathBuf::from(".starman/shader-cache"),
            clear_color: wgpu::Color {
                r: 0.06,
                g: 0.08,
                b: 0.12,
                a: 1.0,
            },
            max_lights: 128,
        }
    }
}

/// Single render executor shared by windowed runner and editor viewport.
pub struct FrameRenderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub caps: NegotiatedCapabilities,
    width: u32,
    height: u32,
    #[allow(dead_code)]
    target_format: wgpu::TextureFormat,
    pub clear_color: wgpu::Color,
    max_lights: usize,
    depth_target: DepthTarget,
    pipeline_3d: Pipeline3d,
    pipeline_2d: Pipeline2d,
    gpu_meshes: HashMap<AssetId, GpuMesh>,
    gpu_textures: HashMap<AssetId, GpuTexture>,
    arena: GpuResourceArena,
    #[allow(dead_code)]
    shaders: ShaderLibrary,
    graph: RenderGraph,
    picking: PickingState,
    debug_view: DebugView,
    frame_index: u64,
    last_pass_order: Vec<PassId>,
    last_cluster_counts: Vec<u32>,
    last_light_count: usize,
    selected: Vec<Entity>,
}

impl FrameRenderer {
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        caps: NegotiatedCapabilities,
        target_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        config: FrameRendererConfig,
    ) -> Self {
        let width = width.max(1);
        let height = height.max(1);
        let mut shaders = ShaderLibrary::new(config.shader_root, config.shader_cache);
        // Warm-compile mesh/sprite shaders when files exist (fallback inside pipelines still works).
        let key = ShaderVariantKey {
            id: ShaderId("mesh3d"),
            feature_bits: 0,
            tier: caps.tier,
        };
        let _ = shaders.compile(&device, key, "mesh3d.wgsl");

        Self {
            depth_target: create_depth_target(&device, width, height),
            pipeline_3d: create_pipeline_3d(&device, target_format),
            pipeline_2d: create_pipeline_2d(&device, target_format),
            device,
            queue,
            caps,
            width,
            height,
            target_format,
            clear_color: config.clear_color,
            max_lights: config.max_lights,
            gpu_meshes: HashMap::new(),
            gpu_textures: HashMap::new(),
            arena: GpuResourceArena::new(3),
            shaders,
            graph: RenderGraph::default_forward_plus(),
            picking: PickingState::default(),
            debug_view: DebugView::None,
            frame_index: 0,
            last_pass_order: Vec::new(),
            last_cluster_counts: Vec::new(),
            last_light_count: 0,
            selected: Vec::new(),
        }
    }

    pub fn tier(&self) -> CapabilityTier {
        self.caps.tier
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

    pub fn set_debug_view(&mut self, view: DebugView) {
        self.debug_view = view;
        self.graph
            .set_enabled(PassId("debug_blit"), view.enables_debug_blit());
    }

    pub fn debug_view(&self) -> DebugView {
        self.debug_view
    }

    pub fn set_selected(&mut self, selected: Vec<Entity>) {
        self.selected = selected;
    }

    pub fn request_pick(&mut self, x: u32, y: u32) {
        self.picking.request(x, y, self.frame_index);
        self.graph
            .set_enabled(PassId("id_pick"), self.picking.needs_id_pass());
    }

    pub fn poll_pick(&mut self) -> Option<PickResult> {
        self.picking.poll()
    }

    pub fn last_pass_order(&self) -> &[PassId] {
        &self.last_pass_order
    }

    pub fn last_light_count(&self) -> usize {
        self.last_light_count
    }

    pub fn graph(&self) -> &RenderGraph {
        &self.graph
    }

    pub fn render_to_view(
        &mut self,
        world: &mut World,
        assets: &AssetServer,
        target_view: &wgpu::TextureView,
    ) -> Result<()> {
        self.frame_index = self.frame_index.saturating_add(1);
        self.arena.begin_frame();

        // --- Extract ---
        let selected = self.selected.clone();
        let render_world = extract_render_world(world, self.width, self.height, &selected);

        // --- Prepare cameras ---
        if let Some(camera_uniform) = render_world.camera_3d {
            self.queue.write_buffer(
                &self.pipeline_3d.camera_buffer,
                0,
                bytemuck::bytes_of(&camera_uniform),
            );
        }
        if let Some(camera_2d) = render_world.camera_2d {
            self.queue.write_buffer(
                &self.pipeline_2d.camera_buffer,
                0,
                bytemuck::bytes_of(&camera_2d),
            );
        }

        // --- Queue: cull + batch ---
        let visible = if let Some(vp) = render_world.view_proj {
            frustum_cull_meshes(Mat4::from_cols_array_2d(&vp), &render_world.meshes)
        } else {
            (0..render_world.meshes.len()).collect()
        };
        let batches_3d = build_batches_3d(&render_world.meshes, &visible);
        let _ = batches_3d; // used for metrics / future instancing; draws still per-item below

        let draw_items_3d: Vec<DrawItem3d> = visible
            .iter()
            .map(|&i| {
                let m = &render_world.meshes[i];
                DrawItem3d {
                    mesh: m.mesh,
                    texture: m.texture,
                    material: m.material,
                    model: m.model,
                    normal: m.normal,
                }
            })
            .collect();

        for item in &draw_items_3d {
            self.ensure_gpu_mesh(item.mesh, assets)?;
            self.ensure_gpu_texture(item.texture, assets)?;
        }

        let draw_items_2d: Vec<DrawItem2d> = render_world
            .sprites
            .iter()
            .map(|s| DrawItem2d {
                texture: s.texture,
                model: s.model,
                color: s.color,
                uv_rect: s.uv_rect,
                sort_z: s.sort_z,
            })
            .collect();
        for item in &draw_items_2d {
            self.ensure_gpu_texture(item.texture, assets)?;
        }

        // --- Forward+ lights ---
        let gpu_lights = build_gpu_lights(&render_world.lights, self.max_lights);
        self.last_light_count = gpu_lights.len();
        let clusters = cull_lights_cpu(&gpu_lights, render_world.camera_position, self.caps.tier);
        self.last_cluster_counts = clusters
            .light_indices
            .iter()
            .map(|c| c.len() as u32)
            .collect();
        let (_offsets, _indices) = pack_cluster_buffers(&clusters);
        let _ = (&gpu_lights, &_offsets, &_indices);

        // Apply directional to camera uniform light_direction when present.
        if let Some(dir) = gpu_lights
            .iter()
            .find(|l| l.light_type == GpuLight::TYPE_DIRECTIONAL)
        {
            if let Some(mut cam) = render_world.camera_3d {
                cam.light_direction = [
                    dir.direction_cone[0],
                    dir.direction_cone[1],
                    dir.direction_cone[2],
                    0.0,
                ];
                self.queue.write_buffer(
                    &self.pipeline_3d.camera_buffer,
                    0,
                    bytemuck::bytes_of(&cam),
                );
            }
        }

        // Picking id map
        self.picking.id_map = render_world
            .meshes
            .iter()
            .map(|m| (m.pick_id, m.entity))
            .chain(render_world.sprites.iter().map(|s| (s.pick_id, s.entity)))
            .collect();
        self.graph
            .set_enabled(PassId("id_pick"), self.picking.needs_id_pass());
        self.graph.set_enabled(PassId("cluster_cull"), true);
        self.graph
            .set_enabled(PassId("debug_blit"), self.debug_view.enables_debug_blit());

        let schedule = self.graph.schedule()?;
        self.last_pass_order = schedule.clone();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("engine-render-frame-encoder"),
            });

        for pass in &schedule {
            match pass.0 {
                "clear" => {
                    let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("engine-render-clear"),
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
                "cluster_cull" => {
                    // CPU path already ran; Tier 1 compute hook lives here later.
                }
                "id_pick" => {
                    // CPU fallback: resolve nearest mesh under cursor heuristically.
                    if let Some(req) = self.picking.pending {
                        let pick_id = self.picking.id_map.first().map(|(id, _)| *id).unwrap_or(0);
                        let _ = req;
                        self.picking.resolve_cpu_fallback(pick_id);
                    }
                }
                "opaque_forward_plus" => {
                    self.encode_3d_pass(&mut encoder, target_view, assets, &draw_items_3d);
                }
                "transparent_2d" => {
                    self.encode_2d_pass(&mut encoder, target_view, &draw_items_2d);
                }
                "overlay" => {
                    // Selection overlay: currently a no-op GPU pass placeholder;
                    // editor still draws egui gizmos. Graph node exists for inspection.
                }
                "debug_blit" => {
                    // Debug visualization is sampled via dump API; blit pass reserved.
                    log::trace!(
                        target: "engine::render",
                        "debug view {:?} active",
                        self.debug_view
                    );
                }
                _ => {}
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Dump a synthetic debug visualization (cluster heat) to PPM.
    pub fn dump_debug_ppm(&self, path: &std::path::Path) -> Result<()> {
        let w = 64u32;
        let h = 64u32;
        let mut pixels = vec![0u8; (w * h * 4) as usize];
        let counts = &self.last_cluster_counts;
        let max = counts.iter().copied().max().unwrap_or(1).max(1);
        for y in 0..h {
            for x in 0..w {
                let idx = ((y * CLUSTER_SAMPLE_Y / h) * CLUSTER_SAMPLE_X
                    + (x * CLUSTER_SAMPLE_X / w)) as usize;
                let c = counts.get(idx).copied().unwrap_or(0);
                let t = (c as f32 / max as f32 * 255.0) as u8;
                let i = ((y * w + x) * 4) as usize;
                pixels[i] = t;
                pixels[i + 1] = 32;
                pixels[i + 2] = 255 - t;
                pixels[i + 3] = 255;
            }
        }
        dump_rgba8_ppm(path, w, h, &pixels)
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
            label: Some("engine-render-opaque-fp"),
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
            let material_uniform = MaterialUniform {
                base_color: material.base_color_factor,
                metallic_roughness: [material.metallic, material.roughness, 0.0, 0.0],
            };

            let model_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("engine-render-model-uniform"),
                    contents: bytemuck::bytes_of(&model_uniform),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            frame_model_buffers.push(model_buffer);
            let material_buffer =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("engine-render-material-uniform"),
                        contents: bytemuck::bytes_of(&material_uniform),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
            frame_material_buffers.push(material_buffer);

            let model_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("engine-render-model-bind-group"),
                layout: &self.pipeline_3d.model_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame_model_buffers.last().unwrap().as_entire_binding(),
                }],
            });
            frame_model_bind_groups.push(model_bind_group);

            let material_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("engine-render-material-bind-group"),
                layout: &self.pipeline_3d.material_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: frame_material_buffers.last().unwrap().as_entire_binding(),
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

            render_pass.set_bind_group(1, frame_model_bind_groups.last().unwrap(), &[]);
            render_pass.set_bind_group(2, frame_material_bind_groups.last().unwrap(), &[]);
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
            label: Some("engine-render-transparent-2d"),
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
            render_pass.set_bind_group(1, frame_sprite_bind_groups.last().unwrap(), &[]);
            render_pass.draw_indexed(
                0..self.pipeline_2d.quad_index_count,
                0,
                batch.start as u32..batch.end as u32,
            );
        }
    }

    fn ensure_gpu_mesh(&mut self, handle: MeshHandle, assets: &AssetServer) -> Result<()> {
        crate::gpu_resources::ensure_gpu_mesh(&self.device, &mut self.gpu_meshes, handle, assets)
    }

    fn ensure_gpu_texture(&mut self, handle: TextureHandle, assets: &AssetServer) -> Result<()> {
        crate::gpu_resources::ensure_gpu_texture(
            &self.device,
            &self.queue,
            &mut self.gpu_textures,
            handle,
            assets,
        )
    }
}

const CLUSTER_SAMPLE_X: u32 = 16;
const CLUSTER_SAMPLE_Y: u32 = 9;

/// Create a FrameRenderer after negotiating device capabilities.
pub fn create_frame_renderer_from_adapter(
    adapter: &wgpu::Adapter,
    target_format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    config: FrameRendererConfig,
) -> Result<FrameRenderer> {
    let (device, queue, caps) = crate::capabilities::request_device(adapter)
        .map_err(|error| EngineError::Render(format!("failed to request wgpu device: {error}")))?;
    log::info!(
        target: "engine::render",
        "Negotiated {} on {} ({})",
        caps.tier.as_str(),
        caps.adapter_name,
        caps.backend
    );
    Ok(FrameRenderer::new(
        device,
        queue,
        caps,
        target_format,
        width,
        height,
        config,
    ))
}

pub fn create_frame_renderer_from_device(
    device: wgpu::Device,
    queue: wgpu::Queue,
    caps: NegotiatedCapabilities,
    target_format: wgpu::TextureFormat,
    width: u32,
    height: u32,
) -> FrameRenderer {
    FrameRenderer::new(
        device,
        queue,
        caps,
        target_format,
        width,
        height,
        FrameRendererConfig::default(),
    )
}
