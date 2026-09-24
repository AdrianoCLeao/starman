//! The UI render extension: one `ui` node on the final (display-encoded)
//! target, after every 3D pass and debug overlay.

use std::sync::Arc;

use bevy_ecs::prelude::World;
use bytemuck::{Pod, Zeroable};
use engine_assets::{Handle, TextureData};
use engine_core::Result;
use engine_render::extension::{
    EncodeContext, ExtensionNode, ExtractInfo, NodeTarget, PrepareContext, RenderExtension,
};
use engine_render::gpu::assets::TextureRole;

use crate::draw::{AtlasImage, QuadTexture, UiDrawList, UiQuad};
use crate::instance::Rect;

const SHADER: &str = include_str!("../shaders/ui.wgsl");
pub const NODE: &str = "ui";

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct QuadGpu {
    rect: [f32; 4],
    uv: [f32; 4],
    color: [f32; 4],
    border_color: [f32; 4],
    params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScreenUniform {
    size: [f32; 2],
    atlas_size: [f32; 2],
    flags: [u32; 4],
}

struct Batch {
    texture: QuadTexture,
    scissor: [u32; 4],
    start: u32,
    count: u32,
}

#[derive(Default)]
pub struct UiRenderer {
    quads: Vec<UiQuad>,
    atlas: Option<AtlasImage>,
    uploaded_atlas: u64,
    atlas_texture: Option<(wgpu::Texture, wgpu::TextureView, u32)>,
    module: Option<Arc<wgpu::ShaderModule>>,
    layouts: Option<(wgpu::BindGroupLayout, wgpu::BindGroupLayout)>,
    pipelines: Vec<(wgpu::TextureFormat, wgpu::RenderPipeline)>,
    instances: Option<(wgpu::Buffer, u64)>,
    screen: Option<wgpu::Buffer>,
    batches: Vec<Batch>,
    draws: usize,
}

fn scissor(clip: &Rect, width: u32, height: u32) -> Option<[u32; 4]> {
    let full = Rect {
        x: 0.0,
        y: 0.0,
        width: width as f32,
        height: height as f32,
    };
    let r = full.intersect(clip);
    let (x0, y0) = (r.x.floor().max(0.0) as u32, r.y.floor().max(0.0) as u32);
    let (x1, y1) = (
        (r.x + r.width).ceil().min(width as f32) as u32,
        (r.y + r.height).ceil().min(height as f32) as u32,
    );
    (x1 > x0 && y1 > y0).then_some([x0, y0, x1 - x0, y1 - y0])
}

impl UiRenderer {
    fn pipeline(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
    ) -> Option<&wgpu::RenderPipeline> {
        if !self.pipelines.iter().any(|(f, _)| *f == format) {
            let (module, (screen_layout, texture_layout)) =
                (self.module.as_ref()?, self.layouts.as_ref()?);
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("ui-pipeline-layout"),
                bind_group_layouts: &[screen_layout, texture_layout],
                push_constant_ranges: &[],
            });
            const ATTRIBUTES: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
                0 => Float32x4,
                1 => Float32x4,
                2 => Float32x4,
                3 => Float32x4,
                4 => Float32x4
            ];
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("ui"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some("vs_main"),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: size_of::<QuadGpu>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &ATTRIBUTES,
                    }],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview: None,
                cache: None,
            });
            self.pipelines.push((format, pipeline));
        }
        self.pipelines
            .iter()
            .find(|(f, _)| *f == format)
            .map(|(_, p)| p)
    }
}

impl RenderExtension for UiRenderer {
    fn name(&self) -> &'static str {
        "engine::ui"
    }

    fn nodes(&self) -> Vec<ExtensionNode> {
        vec![ExtensionNode {
            name: NODE,
            target: NodeTarget::Output,
            after: vec!["tonemap", "debug_view", "debug_lines"],
            before: vec![],
        }]
    }

    fn extract(&mut self, world: &mut World, _info: &ExtractInfo) {
        match world.get_resource::<UiDrawList>() {
            Some(list) => {
                self.quads.clone_from(&list.quads);
                self.atlas = list.atlas.clone();
            }
            None => self.quads.clear(),
        }
    }

    fn prepare(&mut self, ctx: &mut PrepareContext<'_>) -> Result<()> {
        self.batches.clear();
        if self.quads.is_empty() {
            return Ok(());
        }
        let device = ctx.device;
        if self.module.is_none() {
            ctx.shaders.add_source("ui/ui.wgsl", SHADER);
            self.module = Some(ctx.shaders.module(device, "ui/ui.wgsl", &[])?);
            let screen_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ui-screen-layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
            let texture_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("ui-texture-layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });
            self.layouts = Some((screen_layout, texture_layout));
        }

        // Glyph atlas.
        if let Some(atlas) = &self.atlas {
            let recreate = self
                .atlas_texture
                .as_ref()
                .is_none_or(|(_, _, size)| *size != atlas.size);
            if recreate {
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("ui-glyph-atlas"),
                    size: wgpu::Extent3d {
                        width: atlas.size,
                        height: atlas.size,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::R8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                let view = texture.create_view(&Default::default());
                self.atlas_texture = Some((texture, view, atlas.size));
                self.uploaded_atlas = 0;
            }
            if self.uploaded_atlas != atlas.revision {
                if let Some((texture, _, _)) = &self.atlas_texture {
                    ctx.queue.write_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        &atlas.pixels,
                        wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(atlas.size),
                            rows_per_image: Some(atlas.size),
                        },
                        wgpu::Extent3d {
                            width: atlas.size,
                            height: atlas.size,
                            depth_or_array_layers: 1,
                        },
                    );
                }
                self.uploaded_atlas = atlas.revision;
            }
        }

        // Images.
        let assets = ctx.server.assets();
        let mut images: Vec<Handle<TextureData>> = Vec::new();
        for quad in &self.quads {
            if let QuadTexture::Image(handle) = quad.texture {
                if !images.contains(&handle) {
                    images.push(handle);
                }
            }
        }
        for handle in images {
            ctx.cache
                .prepare_typed_texture(device, ctx.queue, handle, assets, true);
        }

        // Instances and batches.
        let (width, height) = (ctx.width, ctx.height);
        let mut gpu = Vec::with_capacity(self.quads.len());
        for quad in &self.quads {
            let Some(rect) = scissor(&quad.clip, width, height) else {
                continue;
            };
            let mode = match quad.texture {
                QuadTexture::None => 0.0,
                QuadTexture::Image(_) => 1.0,
                QuadTexture::Glyphs => 2.0,
            };
            let index = gpu.len() as u32;
            gpu.push(QuadGpu {
                rect: quad.rect,
                uv: quad.uv,
                color: quad.color,
                border_color: quad.border_color,
                params: [quad.corner_radius, quad.border_width, mode, 0.0],
            });
            match self.batches.last_mut() {
                Some(batch) if batch.texture == quad.texture && batch.scissor == rect => {
                    batch.count += 1
                }
                _ => self.batches.push(Batch {
                    texture: quad.texture,
                    scissor: rect,
                    start: index,
                    count: 1,
                }),
            }
        }
        let bytes = (gpu.len() * size_of::<QuadGpu>()) as u64;
        if self
            .instances
            .as_ref()
            .is_none_or(|(_, capacity)| *capacity < bytes)
        {
            let capacity = bytes.next_power_of_two().max(4096);
            self.instances = Some((
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("ui-quads"),
                    size: capacity,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                capacity,
            ));
        }
        if let Some((buffer, _)) = &self.instances {
            ctx.queue
                .write_buffer(buffer, 0, bytemuck::cast_slice(&gpu));
        }
        let screen = self.screen.get_or_insert_with(|| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ui-screen"),
                size: size_of::<ScreenUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        let atlas_size = self
            .atlas_texture
            .as_ref()
            .map_or(1.0, |(_, _, s)| *s as f32);
        let uniform = ScreenUniform {
            size: [width as f32, height as f32],
            atlas_size: [atlas_size, atlas_size],
            flags: [u32::from(ctx.target_format.is_srgb()), 0, 0, 0],
        };
        ctx.queue
            .write_buffer(screen, 0, bytemuck::bytes_of(&uniform));
        Ok(())
    }

    fn encode(&mut self, node: &'static str, ctx: &mut EncodeContext<'_>) -> Result<()> {
        self.draws = 0;
        if node != NODE || self.batches.is_empty() {
            return Ok(());
        }
        let device = ctx.device;
        let format = ctx.color_format;
        if self.pipeline(device, format).is_none() {
            return Ok(());
        }
        let (Some((instances, _)), Some(screen), Some((screen_layout, texture_layout))) = (
            self.instances.as_ref(),
            self.screen.as_ref(),
            self.layouts.as_ref(),
        ) else {
            return Ok(());
        };
        let screen_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ui-screen-group"),
            layout: screen_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: screen.as_entire_binding(),
            }],
        });
        let sampler = &ctx.layouts.linear_clamp;
        let white = &ctx.cache.fallback_for(TextureRole::Color).view;
        let texture_group = |view: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ui-texture-group"),
                layout: texture_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            })
        };
        let solid_group = texture_group(white);
        let atlas_group = self
            .atlas_texture
            .as_ref()
            .map(|(_, view, _)| texture_group(view));
        let mut image_groups: Vec<(QuadTexture, wgpu::BindGroup)> = Vec::new();
        for batch in &self.batches {
            if let QuadTexture::Image(handle) = batch.texture {
                if !image_groups.iter().any(|(t, _)| *t == batch.texture) {
                    let view = ctx
                        .cache
                        .texture(handle.id(), true)
                        .map(|t| &t.view)
                        .unwrap_or(white);
                    image_groups.push((batch.texture, texture_group(view)));
                }
            }
        }
        let pipeline = self
            .pipelines
            .iter()
            .find(|(f, _)| *f == format)
            .map(|(_, p)| p)
            .expect("created above");
        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ui"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.color,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &screen_group, &[]);
        pass.set_vertex_buffer(0, instances.slice(..));
        for batch in &self.batches {
            let group = match batch.texture {
                QuadTexture::None => &solid_group,
                QuadTexture::Glyphs => match &atlas_group {
                    Some(group) => group,
                    None => continue,
                },
                QuadTexture::Image(_) => {
                    &image_groups
                        .iter()
                        .find(|(t, _)| *t == batch.texture)
                        .expect("prepared")
                        .1
                }
            };
            let [x, y, w, h] = batch.scissor;
            pass.set_scissor_rect(x, y, w, h);
            pass.set_bind_group(1, group, &[]);
            pass.draw(0..6, batch.start..batch.start + batch.count);
            self.draws += 1;
        }
        Ok(())
    }

    fn stats(&self) -> Vec<(&'static str, f64)> {
        vec![
            ("ui_quads", self.quads.len() as f64),
            ("ui_batches", self.batches.len() as f64),
            ("ui_draws", self.draws as f64),
        ]
    }
}
