//! Environment lighting: bakes the sky (procedural or HDR panorama) into a
//! cubemap with a full mip chain, convolves diffuse irradiance, prefilters
//! GGX specular per roughness mip, computes the split-sum BRDF LUT once,
//! and draws the skybox. Re-baked only when the environment changes.

use std::hash::{Hash, Hasher};

use engine_assets::{Assets, Handle, HdrImageData};
use engine_core::Result;
use engine_math::Vec3;
use wgpu::util::DeviceExt;

use crate::components::{Environment, SkyMode};
use crate::layouts::{sampler_entry, texture_entry, SharedLayouts, HDR_FORMAT};
use crate::shader::ShaderLibrary;
use crate::uniforms::SkyUniform;

pub const ENVIRONMENT_SIZE: u32 = 256;
pub const IRRADIANCE_SIZE: u32 = 32;
pub const PREFILTER_SIZE: u32 = 128;
pub const PREFILTER_MIPS: u32 = 6;
pub const BRDF_LUT_SIZE: u32 = 256;
const CUBE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

pub fn cube_texture(
    device: &wgpu::Device,
    label: &str,
    size: u32,
    mips: u32,
    layers: u32,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 6 * layers,
        },
        mip_level_count: mips,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: CUBE_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    })
}

pub fn cube_view(texture: &wgpu::Texture) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    })
}

/// A single-face, single-mip render view of a cube (array) texture.
pub fn face_view(texture: &wgpu::Texture, layer: u32, mip: u32) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2),
        base_array_layer: layer,
        array_layer_count: Some(1),
        base_mip_level: mip,
        mip_level_count: Some(1),
        ..Default::default()
    })
}

fn fullscreen_pipeline(
    device: &wgpu::Device,
    label: &str,
    module: &wgpu::ShaderModule,
    layout: &wgpu::BindGroupLayout,
    format: wgpu::TextureFormat,
    fragment_entry: &str,
) -> wgpu::RenderPipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(fragment_entry),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
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

fn uniform_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

fn run_fullscreen(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    target: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        occlusion_query_set: None,
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.draw(0..3, 0..1);
}

/// Cube prefiltering shared by the sky environment and reflection probes.
pub struct CubeFilter {
    downsample_layout: wgpu::BindGroupLayout,
    downsample: wgpu::RenderPipeline,
    irradiance_layout: wgpu::BindGroupLayout,
    irradiance: wgpu::RenderPipeline,
    prefilter_layout: wgpu::BindGroupLayout,
    prefilter: wgpu::RenderPipeline,
}

impl CubeFilter {
    pub fn new(device: &wgpu::Device, shaders: &mut ShaderLibrary) -> Result<Self> {
        let float = wgpu::TextureSampleType::Float { filterable: true };
        let cube_source_layout = |label| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &[
                    texture_entry(0, wgpu::TextureViewDimension::Cube, float),
                    sampler_entry(1, wgpu::SamplerBindingType::Filtering),
                    uniform_layout_entry(2),
                ],
            })
        };
        let downsample_layout = cube_source_layout("engine-render-cube-downsample-layout");
        let irradiance_layout = cube_source_layout("engine-render-irradiance-layout");
        let prefilter_layout = cube_source_layout("engine-render-prefilter-layout");
        let downsample_module = shaders.module(device, "ibl/cube_downsample.wgsl", &[])?;
        let irradiance_module = shaders.module(device, "ibl/irradiance.wgsl", &[])?;
        let prefilter_module = shaders.module(device, "ibl/prefilter.wgsl", &[])?;
        Ok(Self {
            downsample: fullscreen_pipeline(
                device,
                "engine-render-cube-downsample",
                &downsample_module,
                &downsample_layout,
                CUBE_FORMAT,
                "fs_main",
            ),
            irradiance: fullscreen_pipeline(
                device,
                "engine-render-irradiance",
                &irradiance_module,
                &irradiance_layout,
                CUBE_FORMAT,
                "fs_main",
            ),
            prefilter: fullscreen_pipeline(
                device,
                "engine-render-prefilter",
                &prefilter_module,
                &prefilter_layout,
                CUBE_FORMAT,
                "fs_main",
            ),
            downsample_layout,
            irradiance_layout,
            prefilter_layout,
        })
    }

    fn uniform(device: &wgpu::Device, data: [u32; 8]) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("engine-render-cube-filter-uniform"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    }

    /// Fills mips 1.. of `cube` (layer range `layer_base..+6`) from mip 0.
    #[allow(clippy::too_many_arguments)]
    pub fn build_mips(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        cube: &wgpu::Texture,
        layer_base: u32,
        mips: u32,
        sampler: &wgpu::Sampler,
    ) {
        for mip in 1..mips {
            // Each mip reads a view restricted to the previous level.
            let source = cube.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::Cube),
                base_array_layer: layer_base,
                array_layer_count: Some(6),
                base_mip_level: mip - 1,
                mip_level_count: Some(1),
                ..Default::default()
            });
            for face in 0..6 {
                let uniform = Self::uniform(device, [face, 0, 0, 0, 0, 0, 0, 0]);
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("engine-render-cube-downsample-bg"),
                    layout: &self.downsample_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&source),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: uniform.as_entire_binding(),
                        },
                    ],
                });
                let target = face_view(cube, layer_base + face, mip);
                run_fullscreen(
                    encoder,
                    "engine-render-cube-downsample",
                    &target,
                    &self.downsample,
                    &bind_group,
                );
            }
        }
    }

    pub fn irradiance(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::TextureView,
        target: &wgpu::Texture,
        sampler: &wgpu::Sampler,
    ) {
        for face in 0..6 {
            let uniform = Self::uniform(device, [face, 0, 0, 0, 0, 0, 0, 0]);
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("engine-render-irradiance-bg"),
                layout: &self.irradiance_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(source),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: uniform.as_entire_binding(),
                    },
                ],
            });
            let view = face_view(target, face, 0);
            run_fullscreen(
                encoder,
                "engine-render-irradiance",
                &view,
                &self.irradiance,
                &bind_group,
            );
        }
    }

    /// Prefilters `source` into `target` layers `layer_base..+6`, one
    /// roughness per mip (`roughness = mip / (mips - 1)`).
    #[allow(clippy::too_many_arguments)]
    pub fn prefilter(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::TextureView,
        source_size: u32,
        target: &wgpu::Texture,
        layer_base: u32,
        mips: u32,
        sampler: &wgpu::Sampler,
    ) {
        for mip in 0..mips {
            let roughness = mip as f32 / (mips - 1).max(1) as f32;
            let samples: u32 = if mip == 0 { 1 } else { 64 };
            for face in 0..6 {
                let mut data = [0u32; 8];
                data[0] = face;
                data[1] = samples;
                data[4] = roughness.to_bits();
                data[5] = (source_size as f32).to_bits();
                let uniform = Self::uniform(device, data);
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("engine-render-prefilter-bg"),
                    layout: &self.prefilter_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(source),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: uniform.as_entire_binding(),
                        },
                    ],
                });
                let view = face_view(target, layer_base + face, mip);
                run_fullscreen(
                    encoder,
                    "engine-render-prefilter",
                    &view,
                    &self.prefilter,
                    &bind_group,
                );
            }
        }
    }
}

/// Everything the lit passes sample for image-based lighting.
pub struct EnvironmentMaps {
    pub source: wgpu::Texture,
    pub source_view: wgpu::TextureView,
    pub irradiance: wgpu::Texture,
    pub irradiance_view: wgpu::TextureView,
    pub prefiltered: wgpu::Texture,
    pub prefiltered_view: wgpu::TextureView,
    pub brdf_lut: wgpu::Texture,
    pub brdf_lut_view: wgpu::TextureView,
}

pub struct EnvironmentPass {
    bake_layout: wgpu::BindGroupLayout,
    bake_pipeline: wgpu::RenderPipeline,
    skybox_layout: wgpu::BindGroupLayout,
    skybox_pipeline_layout: wgpu::PipelineLayout,
    skybox_pipelines: std::collections::HashMap<u32, wgpu::RenderPipeline>,
    skybox_bind_group: Option<wgpu::BindGroup>,
    pub filter: CubeFilter,
    pub maps: EnvironmentMaps,
    baked_hash: Option<u64>,
    hdr_request: Option<(String, Handle<HdrImageData>)>,
    hdr_texture: Option<(u64, wgpu::Texture)>,
    placeholder_equirect: wgpu::Texture,
    skybox_module: std::sync::Arc<wgpu::ShaderModule>,
    pub bake_count: u64,
}

impl EnvironmentPass {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shaders: &mut ShaderLibrary,
        layouts: &SharedLayouts,
    ) -> Result<Self> {
        let float = wgpu::TextureSampleType::Float { filterable: true };
        let bake_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine-render-environment-bake-layout"),
            entries: &[
                uniform_layout_entry(0),
                uniform_layout_entry(1),
                texture_entry(2, wgpu::TextureViewDimension::D2, float),
                sampler_entry(3, wgpu::SamplerBindingType::Filtering),
            ],
        });
        let bake_module = shaders.module(device, "sky/environment_bake.wgsl", &[])?;
        let bake_pipeline = fullscreen_pipeline(
            device,
            "engine-render-environment-bake",
            &bake_module,
            &bake_layout,
            CUBE_FORMAT,
            "fs_main",
        );
        let skybox_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine-render-skybox-layout"),
            entries: &[
                texture_entry(0, wgpu::TextureViewDimension::Cube, float),
                sampler_entry(1, wgpu::SamplerBindingType::Filtering),
            ],
        });
        let skybox_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("engine-render-skybox-pl"),
                bind_group_layouts: &[&layouts.view_basic, &skybox_layout],
                push_constant_ranges: &[],
            });
        let skybox_module = shaders.module(device, "sky/skybox.wgsl", &[])?;
        let filter = CubeFilter::new(device, shaders)?;

        let source = cube_texture(
            device,
            "engine-render-environment",
            ENVIRONMENT_SIZE,
            crate::gpu::assets::mip_count(ENVIRONMENT_SIZE, ENVIRONMENT_SIZE),
            1,
        );
        let irradiance = cube_texture(device, "engine-render-irradiance", IRRADIANCE_SIZE, 1, 1);
        let prefiltered = cube_texture(
            device,
            "engine-render-prefiltered",
            PREFILTER_SIZE,
            PREFILTER_MIPS,
            1,
        );
        let brdf_lut = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("engine-render-brdf-lut"),
            size: wgpu::Extent3d {
                width: BRDF_LUT_SIZE,
                height: BRDF_LUT_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        // BRDF LUT is view-independent: compute once.
        {
            let lut_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("engine-render-brdf-lut-layout"),
                entries: &[],
            });
            let lut_module = shaders.module(device, "ibl/brdf_lut.wgsl", &[])?;
            let lut_pipeline = fullscreen_pipeline(
                device,
                "engine-render-brdf-lut",
                &lut_module,
                &lut_layout,
                wgpu::TextureFormat::Rg16Float,
                "fs_main",
            );
            let empty = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("engine-render-brdf-lut-bg"),
                layout: &lut_layout,
                entries: &[],
            });
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("engine-render-brdf-lut-encoder"),
            });
            let view = brdf_lut.create_view(&wgpu::TextureViewDescriptor::default());
            run_fullscreen(
                &mut encoder,
                "engine-render-brdf-lut",
                &view,
                &lut_pipeline,
                &empty,
            );
            queue.submit(Some(encoder.finish()));
        }
        let placeholder_equirect = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("engine-render-equirect-placeholder"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            bytemuck::cast_slice(&[f32_to_f16(0.0); 4]),
        );

        let maps = EnvironmentMaps {
            source_view: cube_view(&source),
            irradiance_view: cube_view(&irradiance),
            prefiltered_view: cube_view(&prefiltered),
            brdf_lut_view: brdf_lut.create_view(&wgpu::TextureViewDescriptor::default()),
            source,
            irradiance,
            prefiltered,
            brdf_lut,
        };
        Ok(Self {
            bake_layout,
            bake_pipeline,
            skybox_layout,
            skybox_pipeline_layout,
            skybox_pipelines: std::collections::HashMap::new(),
            skybox_bind_group: None,
            filter,
            maps,
            baked_hash: None,
            hdr_request: None,
            hdr_texture: None,
            placeholder_equirect,
            skybox_module,
            bake_count: 0,
        })
    }

    /// Sky parameters for `environment` with the sun from `sun_direction`
    /// (direction light travels) when following the light.
    pub fn sky_uniform(
        environment: &Environment,
        sun: Option<(Vec3, [f32; 3], f32)>,
    ) -> SkyUniform {
        let (sun_dir, sun_color, sun_intensity) = sun
            .filter(|_| environment.sun_follows_light)
            .map(|(dir, color, intensity)| (-dir.normalize_or(Vec3::NEG_Y), color, intensity))
            .unwrap_or((Vec3::new(0.3, 0.8, 0.2).normalize(), [1.0, 0.95, 0.85], 1.0));
        SkyUniform {
            zenith: [
                environment.zenith_color[0],
                environment.zenith_color[1],
                environment.zenith_color[2],
                environment.sky_intensity,
            ],
            horizon: [
                environment.horizon_color[0],
                environment.horizon_color[1],
                environment.horizon_color[2],
                environment.horizon_sharpness,
            ],
            ground: [
                environment.ground_color[0],
                environment.ground_color[1],
                environment.ground_color[2],
                0.0,
            ],
            sun_direction: [sun_dir.x, sun_dir.y, sun_dir.z, 0.0093],
            sun_color: [
                sun_color[0] * sun_intensity,
                sun_color[1] * sun_intensity,
                sun_color[2] * sun_intensity,
                600.0,
            ],
            params: [
                match environment.mode {
                    SkyMode::Procedural => 0.0,
                    SkyMode::Hdr => 1.0,
                },
                environment.rotation_radians,
                environment.sky_intensity,
                0.0,
            ],
        }
    }

    /// Re-bakes when the environment (or its HDR image) changed. Returns
    /// whether a bake was encoded.
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        layouts: &SharedLayouts,
        assets: &Assets,
        environment: &Environment,
        sun: Option<(Vec3, [f32; 3], f32)>,
    ) -> bool {
        let sky = Self::sky_uniform(environment, sun);
        let mut hdr_revision = 0;
        if environment.mode == SkyMode::Hdr && !environment.hdr.is_empty() {
            let key = environment.hdr.request_key();
            let handle = match &self.hdr_request {
                Some((cached, handle)) if *cached == key => *handle,
                _ => {
                    let handle = assets.request::<HdrImageData>(&environment.hdr);
                    self.hdr_request = Some((key, handle));
                    handle
                }
            };
            hdr_revision = assets.revision(handle);
            if hdr_revision != 0
                && self.hdr_texture.as_ref().map(|(rev, _)| *rev) != Some(hdr_revision)
            {
                if let Some(image) = assets.get(handle) {
                    self.hdr_texture = Some((hdr_revision, upload_hdr(device, queue, &image)));
                }
            }
        }
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        bytemuck::bytes_of(&sky).hash(&mut hasher);
        hdr_revision.hash(&mut hasher);
        let hash = hasher.finish();
        if self.baked_hash == Some(hash) {
            return false;
        }
        self.baked_hash = Some(hash);

        let use_hdr = environment.mode == SkyMode::Hdr && self.hdr_texture.is_some();
        let mut sky = sky;
        if !use_hdr {
            sky.params[0] = 0.0;
        }
        let equirect = self
            .hdr_texture
            .as_ref()
            .filter(|_| use_hdr)
            .map(|(_, texture)| texture)
            .unwrap_or(&self.placeholder_equirect)
            .create_view(&wgpu::TextureViewDescriptor::default());
        let sky_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("engine-render-sky-uniform"),
            contents: bytemuck::bytes_of(&sky),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        for face in 0..6u32 {
            let face_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("engine-render-sky-face"),
                contents: bytemuck::cast_slice(&[face, 0, 0, 0]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("engine-render-sky-bake-bg"),
                layout: &self.bake_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: sky_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: face_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&equirect),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&layouts.linear_clamp),
                    },
                ],
            });
            let target = face_view(&self.maps.source, face, 0);
            run_fullscreen(
                encoder,
                "engine-render-sky-bake",
                &target,
                &self.bake_pipeline,
                &bind_group,
            );
        }
        let mips = self.maps.source.mip_level_count();
        self.filter.build_mips(
            device,
            encoder,
            &self.maps.source,
            0,
            mips,
            &layouts.linear_clamp,
        );
        self.filter.irradiance(
            device,
            encoder,
            &self.maps.source_view,
            &self.maps.irradiance,
            &layouts.linear_clamp,
        );
        self.filter.prefilter(
            device,
            encoder,
            &self.maps.source_view,
            ENVIRONMENT_SIZE,
            &self.maps.prefiltered,
            0,
            PREFILTER_MIPS,
            &layouts.linear_clamp,
        );
        self.skybox_bind_group = None;
        self.bake_count += 1;
        true
    }

    fn skybox_pipeline(&mut self, device: &wgpu::Device, samples: u32) -> &wgpu::RenderPipeline {
        let module = &self.skybox_module;
        let layout = &self.skybox_pipeline_layout;
        self.skybox_pipelines.entry(samples).or_insert_with(|| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("engine-render-skybox"),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: HDR_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: crate::layouts::DEPTH_FORMAT,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: samples,
                    ..Default::default()
                },
                multiview: None,
                cache: None,
            })
        })
    }

    /// Draws the skybox into an open pass (group 0 = basic view).
    pub fn prepare_skybox(&mut self, device: &wgpu::Device, layouts: &SharedLayouts, samples: u32) {
        if self.skybox_bind_group.is_none() {
            self.skybox_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("engine-render-skybox-bg"),
                layout: &self.skybox_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.maps.source_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&layouts.linear_clamp),
                    },
                ],
            }));
        }
        self.skybox_pipeline(device, samples);
    }

    pub fn draw_skybox(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        view_bind_group: &wgpu::BindGroup,
        view_offset: u32,
        samples: u32,
    ) {
        let (Some(pipeline), Some(bind_group)) =
            (self.skybox_pipelines.get(&samples), &self.skybox_bind_group)
        else {
            return;
        };
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, view_bind_group, &[view_offset]);
        pass.set_bind_group(1, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn upload_hdr(device: &wgpu::Device, queue: &wgpu::Queue, image: &HdrImageData) -> wgpu::Texture {
    let half: Vec<u16> = image.rgba.iter().map(|v| f32_to_f16(*v)).collect();
    device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("engine-render-environment-hdr"),
            size: wgpu::Extent3d {
                width: image.width.max(1),
                height: image.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        bytemuck::cast_slice(&half),
    )
}

/// IEEE 754 binary32 → binary16 (round to nearest even, saturating to
/// the largest finite half; NaN preserved).
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x007f_ffff;
    if exponent == 0xff {
        return sign | if mantissa != 0 { 0x7e00 } else { 0x7c00 };
    }
    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 0x1f {
        return sign | 0x7bff;
    }
    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        let m = mantissa | 0x0080_0000;
        let shift = (14 - half_exponent) as u32;
        let half_m = m >> shift;
        let round = (m >> (shift - 1)) & 1;
        return sign | (half_m + round) as u16;
    }
    let half = sign | ((half_exponent as u16) << 10) | ((mantissa >> 13) as u16);
    let round_bits = mantissa & 0x1fff;
    if round_bits > 0x1000 || (round_bits == 0x1000 && (half & 1) == 1) {
        half + 1
    } else {
        half
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_conversion_matches_reference_values() {
        assert_eq!(f32_to_f16(0.0), 0x0000);
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(-2.0), 0xc000);
        assert_eq!(f32_to_f16(0.5), 0x3800);
        assert_eq!(f32_to_f16(65504.0), 0x7bff);
        assert_eq!(f32_to_f16(1e10), 0x7bff);
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_f16(6.1035156e-5), 0x0400);
    }

    #[test]
    fn sky_follows_the_sun_light() {
        let environment = Environment::default();
        let sky =
            EnvironmentPass::sky_uniform(&environment, Some((Vec3::NEG_Y, [1.0, 0.0, 0.0], 3.0)));
        assert!((sky.sun_direction[1] - 1.0).abs() < 1e-5);
        assert_eq!(sky.sun_color[0], 3.0);
        let mut fixed = environment.clone();
        fixed.sun_follows_light = false;
        let sky = EnvironmentPass::sky_uniform(&fixed, Some((Vec3::NEG_Y, [1.0, 0.0, 0.0], 3.0)));
        assert!(sky.sun_direction[1] < 1.0);
    }
}

/// View matrix rendering cube face `face` from `position` so that the
/// image matches the hardware cube addressing (and `cube_direction` in
/// `sky_common.wgsl`). Cube faces are mirrored, so the basis is
/// left-handed: draw with culling disabled.
pub fn cube_face_view(face: u32, position: Vec3) -> engine_math::Mat4 {
    let (right, up, forward) = match face {
        0 => (Vec3::NEG_Z, Vec3::Y, Vec3::X),
        1 => (Vec3::Z, Vec3::Y, Vec3::NEG_X),
        2 => (Vec3::X, Vec3::NEG_Z, Vec3::Y),
        3 => (Vec3::X, Vec3::Z, Vec3::NEG_Y),
        4 => (Vec3::X, Vec3::Y, Vec3::Z),
        _ => (Vec3::NEG_X, Vec3::Y, Vec3::NEG_Z),
    };
    let back = -forward;
    engine_math::Mat4::from_cols(
        engine_math::Vec4::new(right.x, up.x, back.x, 0.0),
        engine_math::Vec4::new(right.y, up.y, back.y, 0.0),
        engine_math::Vec4::new(right.z, up.z, back.z, 0.0),
        engine_math::Vec4::new(
            -right.dot(position),
            -up.dot(position),
            -back.dot(position),
            1.0,
        ),
    )
}

#[cfg(test)]
mod cube_tests {
    use super::*;
    use engine_math::{Mat4, Vec4};

    /// Mirror of `cube_direction` in sky_common.wgsl.
    fn cube_direction(face: u32, u: f32, v: f32) -> Vec3 {
        let (sx, sy) = (u * 2.0 - 1.0, v * 2.0 - 1.0);
        match face {
            0 => Vec3::new(1.0, -sy, -sx),
            1 => Vec3::new(-1.0, -sy, sx),
            2 => Vec3::new(sx, 1.0, sy),
            3 => Vec3::new(sx, -1.0, -sy),
            4 => Vec3::new(sx, -sy, 1.0),
            _ => Vec3::new(-sx, -sy, -1.0),
        }
        .normalize()
    }

    #[test]
    fn face_views_project_texels_onto_their_cube_directions() {
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0);
        for face in 0..6 {
            let view = cube_face_view(face, Vec3::ZERO);
            for (u, v) in [(0.5, 0.5), (0.25, 0.75), (0.9, 0.1)] {
                let dir = cube_direction(face, u, v) * 10.0;
                let clip = proj * view * Vec4::new(dir.x, dir.y, dir.z, 1.0);
                let ndc = clip.truncate() / clip.w;
                let (pu, pv) = (ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
                assert!(
                    (pu - u).abs() < 1e-4 && (pv - v).abs() < 1e-4,
                    "face {face}: ({pu},{pv}) vs ({u},{v})"
                );
                assert!((0.0..=1.0).contains(&ndc.z));
            }
        }
    }
}
