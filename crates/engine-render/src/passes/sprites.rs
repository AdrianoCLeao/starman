//! World-space 2D sprites (alpha blended into HDR, sorted by Z, batched by
//! texture, instanced).

use std::mem::size_of;

use bytemuck::{Pod, Zeroable};
use engine_assets::AssetServer;
use engine_core::Result;
use wgpu::util::DeviceExt;

use crate::draw::{build_draw_batches_2d, DrawItem2d};
use crate::gpu::assets::GpuAssetCache;
use crate::gpu::{GrowableBuffer, StagingBelt};
use crate::layouts::{sampler_entry, texture_entry, SharedLayouts, HDR_FORMAT};
use crate::shader::ShaderLibrary;
use crate::Camera2dUniform;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SpriteQuadVertex {
    position: [f32; 2],
    uv: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct SpriteInstance {
    pub(crate) model: [[f32; 4]; 4],
    pub(crate) color: [f32; 4],
    pub(crate) uv_rect: [f32; 4],
}

const QUAD_ATTRIBUTES: [wgpu::VertexAttribute; 2] =
    wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2];
const INSTANCE_ATTRIBUTES: [wgpu::VertexAttribute; 6] = wgpu::vertex_attr_array![
    2 => Float32x4,
    3 => Float32x4,
    4 => Float32x4,
    5 => Float32x4,
    6 => Float32x4,
    7 => Float32x4
];

pub struct SpritePass {
    pipeline: wgpu::RenderPipeline,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    texture_layout: wgpu::BindGroupLayout,
    quad_vertices: wgpu::Buffer,
    quad_indices: wgpu::Buffer,
    instances: GrowableBuffer,
    items: Vec<DrawItem2d>,
}

impl SpritePass {
    pub fn new(device: &wgpu::Device, shaders: &mut ShaderLibrary) -> Result<Self> {
        let module = shaders.module(device, "sprite2d.wgsl", &[])?;
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine-render-camera2d-layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine-render-sprite-layout"),
            entries: &[
                texture_entry(
                    0,
                    wgpu::TextureViewDimension::D2,
                    wgpu::TextureSampleType::Float { filterable: true },
                ),
                sampler_entry(1, wgpu::SamplerBindingType::Filtering),
            ],
        });
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine-render-camera2d-uniform"),
            size: size_of::<Camera2dUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("engine-render-camera2d-bg"),
            layout: &camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });
        let quad = [
            SpriteQuadVertex {
                position: [-0.5, -0.5],
                uv: [0.0, 1.0],
            },
            SpriteQuadVertex {
                position: [0.5, -0.5],
                uv: [1.0, 1.0],
            },
            SpriteQuadVertex {
                position: [0.5, 0.5],
                uv: [1.0, 0.0],
            },
            SpriteQuadVertex {
                position: [-0.5, 0.5],
                uv: [0.0, 0.0],
            },
        ];
        let quad_vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("engine-render-sprite-quad"),
            contents: bytemuck::cast_slice(&quad),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let quad_indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("engine-render-sprite-quad-indices"),
            contents: bytemuck::cast_slice(&[0u16, 1, 2, 0, 2, 3]),
            usage: wgpu::BufferUsages::INDEX,
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("engine-render-sprite-pl"),
            bind_group_layouts: &[&camera_layout, &texture_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("engine-render-sprite2d"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: size_of::<SpriteQuadVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &QUAD_ATTRIBUTES,
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: size_of::<SpriteInstance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &INSTANCE_ATTRIBUTES,
                    },
                ],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        Ok(Self {
            pipeline,
            camera_buffer,
            camera_bind_group,
            texture_layout,
            quad_vertices,
            quad_indices,
            instances: GrowableBuffer::new(
                device,
                "engine-render-sprite-instances",
                wgpu::BufferUsages::VERTEX,
                size_of::<SpriteInstance>() as u64 * 64,
            ),
            items: Vec::new(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        belt: &mut StagingBelt,
        cache: &mut GpuAssetCache,
        server: &AssetServer,
        camera: Option<Camera2dUniform>,
        items: Vec<DrawItem2d>,
    ) {
        self.items = items;
        if self.items.is_empty() {
            return;
        }
        if let Some(camera) = camera {
            belt.write_buffer(
                device,
                encoder,
                &self.camera_buffer,
                0,
                bytemuck::bytes_of(&camera),
            );
        }
        for item in &self.items {
            cache.prepare_texture(device, queue, item.texture, server, true);
        }
        let instances: Vec<SpriteInstance> = self
            .items
            .iter()
            .map(|item| SpriteInstance {
                model: item.model,
                color: item.color,
                uv_rect: item.uv_rect,
            })
            .collect();
        self.instances
            .upload(device, encoder, belt, bytemuck::cast_slice(&instances));
    }

    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        layouts: &SharedLayouts,
        cache: &GpuAssetCache,
        target: &wgpu::TextureView,
    ) {
        if self.items.is_empty() {
            return;
        }
        let batches = build_draw_batches_2d(&self.items);
        let bind_groups: Vec<Option<wgpu::BindGroup>> = batches
            .iter()
            .map(|batch| {
                cache.texture(batch.texture.id(), true).map(|texture| {
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("engine-render-sprite-bg"),
                        layout: &self.texture_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&texture.view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(
                                    &layouts.linear_repeat_aniso,
                                ),
                            },
                        ],
                    })
                })
            })
            .collect();
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("engine-render-sprites-2d"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
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
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.camera_bind_group, &[]);
        pass.set_vertex_buffer(0, self.quad_vertices.slice(..));
        pass.set_vertex_buffer(1, self.instances.buffer().slice(..));
        pass.set_index_buffer(self.quad_indices.slice(..), wgpu::IndexFormat::Uint16);
        for (batch, bind_group) in batches.iter().zip(&bind_groups) {
            let Some(bind_group) = bind_group else {
                continue;
            };
            pass.set_bind_group(1, bind_group, &[]);
            pass.draw_indexed(0..6, 0, batch.start as u32..batch.end as u32);
        }
    }
}
