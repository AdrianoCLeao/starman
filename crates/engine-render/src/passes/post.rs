//! Screen-space passes: SSAO (+ depth-aware blur), dual-filter bloom,
//! temporal anti-aliasing, tonemapping and debug views.

use engine_core::Result;
use wgpu::util::DeviceExt;

use crate::layouts::{sampler_entry, texture_entry, SharedLayouts, AO_FORMAT, HDR_FORMAT};
use crate::shader::ShaderLibrary;

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn view_entry(binding: u32, size: u64) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: std::num::NonZeroU64::new(size),
        },
        count: None,
    }
}

const FLOAT: wgpu::TextureSampleType = wgpu::TextureSampleType::Float { filterable: true };
const UNFILTERABLE: wgpu::TextureSampleType = wgpu::TextureSampleType::Float { filterable: false };

fn pipeline(
    device: &wgpu::Device,
    label: &str,
    module: &wgpu::ShaderModule,
    layouts: &[&wgpu::BindGroupLayout],
    format: wgpu::TextureFormat,
    entry: &str,
    blend: Option<wgpu::BlendState>,
) -> wgpu::RenderPipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: layouts,
        push_constant_ranges: &[],
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(entry),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

pub fn fullscreen_pass(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    target: &wgpu::TextureView,
    load: wgpu::LoadOp<wgpu::Color>,
    pipeline: &wgpu::RenderPipeline,
    bind_groups: &[(&wgpu::BindGroup, &[u32])],
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            resolve_target: None,
            ops: wgpu::Operations {
                load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        occlusion_query_set: None,
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    for (index, (bind_group, offsets)) in bind_groups.iter().enumerate() {
        pass.set_bind_group(index as u32, *bind_group, offsets);
    }
    pass.draw(0..3, 0..1);
}

fn target_texture(
    device: &wgpu::Device,
    label: &str,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    mips: u32,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: mips.max(1),
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

/// SSAO settings as uploaded.
#[derive(Clone, Copy, Debug)]
pub struct SsaoParams {
    pub radius: f32,
    pub bias: f32,
    pub power: f32,
    pub samples: u32,
}

pub struct SsaoPass {
    layout: wgpu::BindGroupLayout,
    blur_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    blur_pipeline: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    raw: wgpu::TextureView,
    pub ao: wgpu::Texture,
    pub ao_view: wgpu::TextureView,
    size: (u32, u32),
}

impl SsaoPass {
    pub fn new(
        device: &wgpu::Device,
        shaders: &mut ShaderLibrary,
        layouts: &SharedLayouts,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine-render-ssao-layout"),
            entries: &[
                view_entry(0, layouts.view_uniform_size),
                texture_entry(
                    1,
                    wgpu::TextureViewDimension::D2,
                    wgpu::TextureSampleType::Depth,
                ),
                uniform_entry(2),
            ],
        });
        let blur_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine-render-ssao-blur-layout"),
            entries: &[
                texture_entry(0, wgpu::TextureViewDimension::D2, UNFILTERABLE),
                texture_entry(
                    1,
                    wgpu::TextureViewDimension::D2,
                    wgpu::TextureSampleType::Depth,
                ),
            ],
        });
        let module = shaders.module(device, "post/ssao.wgsl", &[])?;
        let blur_module = shaders.module(device, "post/ssao_blur.wgsl", &[])?;
        let pipeline = pipeline(
            device,
            "engine-render-ssao",
            &module,
            &[&layout],
            AO_FORMAT,
            "fs_main",
            None,
        );
        let blur_pipeline = pipeline_fn(device, &blur_module, &blur_layout);
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine-render-ssao-uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (raw, ao) = Self::targets(device, width, height);
        Ok(Self {
            layout,
            blur_layout,
            pipeline,
            blur_pipeline,
            uniform,
            raw: raw.create_view(&Default::default()),
            ao_view: ao.create_view(&Default::default()),
            ao,
            size: (width, height),
        })
    }

    fn targets(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::Texture) {
        (
            target_texture(
                device,
                "engine-render-ssao-raw",
                width,
                height,
                AO_FORMAT,
                1,
            ),
            target_texture(device, "engine-render-ssao", width, height, AO_FORMAT, 1),
        )
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if self.size == (width, height) {
            return;
        }
        let (raw, ao) = Self::targets(device, width, height);
        self.raw = raw.create_view(&Default::default());
        self.ao_view = ao.create_view(&Default::default());
        self.ao = ao;
        self.size = (width, height);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view_buffer: &wgpu::Buffer,
        view_size: u64,
        view_offset: u32,
        depth: &wgpu::TextureView,
        params: SsaoParams,
    ) {
        queue.write_buffer(
            &self.uniform,
            0,
            bytemuck::cast_slice(&[
                params.radius,
                params.bias,
                params.power,
                params.samples as f32,
            ]),
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("engine-render-ssao-bg"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: view_buffer,
                        offset: 0,
                        size: std::num::NonZeroU64::new(view_size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.uniform.as_entire_binding(),
                },
            ],
        });
        fullscreen_pass(
            encoder,
            "engine-render-ssao",
            &self.raw,
            wgpu::LoadOp::Clear(wgpu::Color::WHITE),
            &self.pipeline,
            &[(&bind_group, &[view_offset])],
        );
        let blur = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("engine-render-ssao-blur-bg"),
            layout: &self.blur_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.raw),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth),
                },
            ],
        });
        fullscreen_pass(
            encoder,
            "engine-render-ssao-blur",
            &self.ao_view,
            wgpu::LoadOp::Clear(wgpu::Color::WHITE),
            &self.blur_pipeline,
            &[(&blur, &[])],
        );
    }

    /// Fills the AO target with 1.0 (SSAO disabled).
    pub fn clear(&self, encoder: &mut wgpu::CommandEncoder) {
        let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("engine-render-ssao-clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.ao_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
    }
}

fn pipeline_fn(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    pipeline(
        device,
        "engine-render-ssao-blur",
        module,
        &[layout],
        AO_FORMAT,
        "fs_main",
        None,
    )
}

pub const BLOOM_MIPS: u32 = 6;

pub struct BloomPass {
    layout: wgpu::BindGroupLayout,
    down: wgpu::RenderPipeline,
    up: wgpu::RenderPipeline,
    chain: wgpu::Texture,
    views: Vec<wgpu::TextureView>,
    size: (u32, u32),
    pub output_view: wgpu::TextureView,
}

impl BloomPass {
    pub fn new(
        device: &wgpu::Device,
        shaders: &mut ShaderLibrary,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine-render-bloom-layout"),
            entries: &[
                texture_entry(0, wgpu::TextureViewDimension::D2, FLOAT),
                sampler_entry(1, wgpu::SamplerBindingType::Filtering),
                uniform_entry(2),
            ],
        });
        let module = shaders.module(device, "post/bloom.wgsl", &[])?;
        let down = pipeline(
            device,
            "engine-render-bloom-down",
            &module,
            &[&layout],
            HDR_FORMAT,
            "fs_downsample",
            None,
        );
        let additive = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::REPLACE,
        };
        let up = pipeline(
            device,
            "engine-render-bloom-up",
            &module,
            &[&layout],
            HDR_FORMAT,
            "fs_upsample",
            Some(additive),
        );
        let (chain, views) = Self::chain(device, width, height);
        Ok(Self {
            layout,
            down,
            up,
            output_view: views[0].clone(),
            chain,
            views,
            size: (width, height),
        })
    }

    fn chain(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, Vec<wgpu::TextureView>) {
        let (w, h) = ((width / 2).max(1), (height / 2).max(1));
        let mips = BLOOM_MIPS.min(crate::gpu::assets::mip_count(w, h));
        let chain = target_texture(device, "engine-render-bloom-chain", w, h, HDR_FORMAT, mips);
        let views = (0..mips)
            .map(|mip| {
                chain.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        (chain, views)
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if self.size == (width, height) {
            return;
        }
        let (chain, views) = Self::chain(device, width, height);
        self.output_view = views[0].clone();
        self.chain = chain;
        self.views = views;
        self.size = (width, height);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        layouts: &SharedLayouts,
        source: &wgpu::TextureView,
        threshold: f32,
    ) {
        let make = |view: &wgpu::TextureView, params: [f32; 8]| {
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("engine-render-bloom-uniform"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("engine-render-bloom-bg"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&layouts.linear_clamp),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffer.as_entire_binding(),
                    },
                ],
            })
        };
        let (w0, h0) = (self.size.0.max(1) as f32, self.size.1.max(1) as f32);
        for mip in 0..self.views.len() {
            let (source_view, texel, first) = if mip == 0 {
                (source, [1.0 / w0, 1.0 / h0], 1.0)
            } else {
                let scale = (1u32 << mip) as f32;
                (&self.views[mip - 1], [scale / w0, scale / h0], 0.0)
            };
            let bind_group = make(
                source_view,
                [threshold, 0.5, 1.0, first, texel[0], texel[1], 0.0, 0.0],
            );
            fullscreen_pass(
                encoder,
                "engine-render-bloom-down",
                &self.views[mip],
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                &self.down,
                &[(&bind_group, &[])],
            );
        }
        for mip in (1..self.views.len()).rev() {
            let scale = (1u32 << (mip + 1)) as f32;
            let bind_group = make(
                &self.views[mip],
                [threshold, 0.5, 1.0, 0.0, scale / w0, scale / h0, 0.0, 0.0],
            );
            fullscreen_pass(
                encoder,
                "engine-render-bloom-up",
                &self.views[mip - 1],
                wgpu::LoadOp::Load,
                &self.up,
                &[(&bind_group, &[])],
            );
        }
    }
}

pub struct TaaPass {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    history: [wgpu::Texture; 2],
    history_views: [wgpu::TextureView; 2],
    current: usize,
    history_valid: bool,
    size: (u32, u32),
}

impl TaaPass {
    pub fn new(
        device: &wgpu::Device,
        shaders: &mut ShaderLibrary,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine-render-taa-layout"),
            entries: &[
                texture_entry(0, wgpu::TextureViewDimension::D2, UNFILTERABLE),
                texture_entry(1, wgpu::TextureViewDimension::D2, FLOAT),
                texture_entry(2, wgpu::TextureViewDimension::D2, UNFILTERABLE),
                sampler_entry(3, wgpu::SamplerBindingType::Filtering),
                uniform_entry(4),
            ],
        });
        let module = shaders.module(device, "post/taa.wgsl", &[])?;
        let pipeline = pipeline(
            device,
            "engine-render-taa",
            &module,
            &[&layout],
            HDR_FORMAT,
            "fs_main",
            None,
        );
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine-render-taa-uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let history = Self::history(device, width, height);
        Ok(Self {
            layout,
            pipeline,
            uniform,
            history_views: [
                history[0].create_view(&Default::default()),
                history[1].create_view(&Default::default()),
            ],
            history,
            current: 0,
            history_valid: false,
            size: (width, height),
        })
    }

    fn history(device: &wgpu::Device, width: u32, height: u32) -> [wgpu::Texture; 2] {
        [
            target_texture(
                device,
                "engine-render-taa-history-a",
                width,
                height,
                HDR_FORMAT,
                1,
            ),
            target_texture(
                device,
                "engine-render-taa-history-b",
                width,
                height,
                HDR_FORMAT,
                1,
            ),
        ]
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if self.size == (width, height) {
            return;
        }
        self.history = Self::history(device, width, height);
        self.history_views = [
            self.history[0].create_view(&Default::default()),
            self.history[1].create_view(&Default::default()),
        ];
        self.history_valid = false;
        self.size = (width, height);
    }

    /// Drops accumulated history (camera cut, TAA toggled).
    pub fn reset(&mut self) {
        self.history_valid = false;
    }

    /// Resolves `current` into the next history texture and returns its
    /// view (the anti-aliased HDR image for the rest of the frame).
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        layouts: &SharedLayouts,
        current: &wgpu::TextureView,
        velocity: &wgpu::TextureView,
    ) -> wgpu::TextureView {
        let read = self.current;
        let write = 1 - self.current;
        queue.write_buffer(
            &self.uniform,
            0,
            bytemuck::cast_slice(&[
                0.9f32,
                if self.history_valid { 1.0 } else { 0.0 },
                1.0 / self.size.0.max(1) as f32,
                1.0 / self.size.1.max(1) as f32,
            ]),
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("engine-render-taa-bg"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(current),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.history_views[read]),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(velocity),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&layouts.linear_clamp),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.uniform.as_entire_binding(),
                },
            ],
        });
        fullscreen_pass(
            encoder,
            "engine-render-taa",
            &self.history_views[write],
            wgpu::LoadOp::Clear(wgpu::Color::BLACK),
            &self.pipeline,
            &[(&bind_group, &[])],
        );
        self.current = write;
        self.history_valid = true;
        self.history_views[write].clone()
    }
}

pub struct TonemapPass {
    layout: wgpu::BindGroupLayout,
    pipelines: std::collections::HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
    uniform: wgpu::Buffer,
    module_linear: std::sync::Arc<wgpu::ShaderModule>,
    module_encode: std::sync::Arc<wgpu::ShaderModule>,
    black: wgpu::TextureView,
}

/// Whether the shader must apply the sRGB transfer itself.
pub fn needs_manual_srgb(format: wgpu::TextureFormat) -> bool {
    !format.is_srgb()
}

impl TonemapPass {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shaders: &mut ShaderLibrary,
    ) -> Result<Self> {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine-render-tonemap-layout"),
            entries: &[
                texture_entry(0, wgpu::TextureViewDimension::D2, FLOAT),
                sampler_entry(1, wgpu::SamplerBindingType::Filtering),
                uniform_entry(2),
                texture_entry(3, wgpu::TextureViewDimension::D2, FLOAT),
            ],
        });
        let module_linear = shaders.module(device, "post/tonemap_aces.wgsl", &[])?;
        let module_encode =
            shaders.module(device, "post/tonemap_aces.wgsl", &["OUTPUT_SRGB_ENCODE"])?;
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine-render-tonemap-uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let black = device
            .create_texture_with_data(
                queue,
                &wgpu::TextureDescriptor {
                    label: Some("engine-render-black-hdr"),
                    size: wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: HDR_FORMAT,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
                wgpu::util::TextureDataOrder::LayerMajor,
                &[0u8; 8],
            )
            .create_view(&Default::default());
        Ok(Self {
            layout,
            pipelines: std::collections::HashMap::new(),
            uniform,
            module_linear,
            module_encode,
            black,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        layouts: &SharedLayouts,
        hdr: &wgpu::TextureView,
        bloom: Option<&wgpu::TextureView>,
        target: &wgpu::TextureView,
        target_format: wgpu::TextureFormat,
        exposure: f32,
        bloom_intensity: f32,
    ) {
        let layout = &self.layout;
        let module = if needs_manual_srgb(target_format) {
            &self.module_encode
        } else {
            &self.module_linear
        };
        let pipeline = self.pipelines.entry(target_format).or_insert_with(|| {
            pipeline(
                device,
                "engine-render-tonemap",
                module,
                &[layout],
                target_format,
                "fs_main",
                None,
            )
        });
        let dither = if target_format.is_srgb()
            || matches!(
                target_format,
                wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Bgra8Unorm
            ) {
            1.0
        } else {
            0.0
        };
        queue.write_buffer(
            &self.uniform,
            0,
            bytemuck::cast_slice(&[
                exposure,
                bloom_intensity,
                if bloom.is_some() { 1.0 } else { 0.0 },
                dither,
            ]),
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("engine-render-tonemap-bg"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(hdr),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&layouts.linear_clamp),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(bloom.unwrap_or(&self.black)),
                },
            ],
        });
        fullscreen_pass(
            encoder,
            "engine-render-tonemap",
            target,
            wgpu::LoadOp::Clear(wgpu::Color::BLACK),
            pipeline,
            &[(&bind_group, &[])],
        );
    }
}

pub struct DebugViewPass {
    layout: wgpu::BindGroupLayout,
    pipelines: std::collections::HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
    uniform: wgpu::Buffer,
    module_linear: std::sync::Arc<wgpu::ShaderModule>,
    module_encode: std::sync::Arc<wgpu::ShaderModule>,
}

impl DebugViewPass {
    pub fn new(
        device: &wgpu::Device,
        shaders: &mut ShaderLibrary,
        layouts: &SharedLayouts,
    ) -> Result<Self> {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine-render-debug-view-layout"),
            entries: &[
                view_entry(0, layouts.view_uniform_size),
                texture_entry(
                    1,
                    wgpu::TextureViewDimension::D2,
                    wgpu::TextureSampleType::Depth,
                ),
                texture_entry(2, wgpu::TextureViewDimension::D2, UNFILTERABLE),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                uniform_entry(4),
            ],
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine-render-debug-view-uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Self {
            layout,
            pipelines: std::collections::HashMap::new(),
            uniform,
            module_linear: shaders.module(device, "post/debug_view.wgsl", &[])?,
            module_encode: shaders.module(
                device,
                "post/debug_view.wgsl",
                &["OUTPUT_SRGB_ENCODE"],
            )?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view_buffer: &wgpu::Buffer,
        view_size: u64,
        view_offset: u32,
        depth: &wgpu::TextureView,
        aux: &wgpu::TextureView,
        clusters: &wgpu::Buffer,
        mode: u32,
        target: &wgpu::TextureView,
        target_format: wgpu::TextureFormat,
    ) {
        let layout = &self.layout;
        let module = if needs_manual_srgb(target_format) {
            &self.module_encode
        } else {
            &self.module_linear
        };
        let pipeline = self.pipelines.entry(target_format).or_insert_with(|| {
            pipeline(
                device,
                "engine-render-debug-view",
                module,
                &[layout],
                target_format,
                "fs_main",
                None,
            )
        });
        queue.write_buffer(&self.uniform, 0, bytemuck::cast_slice(&[mode, 0u32, 0, 0]));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("engine-render-debug-view-bg"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: view_buffer,
                        offset: 0,
                        size: std::num::NonZeroU64::new(view_size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(aux),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: clusters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.uniform.as_entire_binding(),
                },
            ],
        });
        fullscreen_pass(
            encoder,
            "engine-render-debug-view",
            target,
            wgpu::LoadOp::Clear(wgpu::Color::BLACK),
            pipeline,
            &[(&bind_group, &[view_offset])],
        );
    }
}
