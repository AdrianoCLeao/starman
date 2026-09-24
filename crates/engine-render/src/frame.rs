//! Frame orchestration (ADR 0009): extract → prepare → queue → execute the
//! render graph. One `FrameRenderer` drives the runner window, the editor
//! viewport and headless tests alike.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use bevy_ecs::entity::Entity;
use bevy_ecs::prelude::World;
use engine_assets::AssetServer;
use engine_core::{EngineError, Result};
use engine_math::{Mat4, Vec3};

use crate::capabilities::{CapabilityTier, NegotiatedCapabilities};
use crate::components::ReflectionProbe;
use crate::culling::Frustum;
use crate::debug::{dump_rgba8_ppm, DebugView};
use crate::draw::DrawItem2d;
use crate::extension::{
    EncodeContext, ExtensionNode, ExtractInfo, NodeTarget, PrepareContext, RenderExtension,
    RenderExtensions,
};
use crate::extract::{extract_render_world, Aabb, RenderWorld};
use crate::forward_plus::{
    assign_clusters, build_gpu_lights, ClusterLayout, GpuLight, DEFAULT_DEPTH_SLICES,
    DEFAULT_TILE_PX,
};
use crate::gpu::assets::GpuAssetCache;
use crate::gpu::{AlignedWriter, GrowableBuffer, StagingBelt, TransientKey, TransientPool};
use crate::graph::{PassId, PassNode, RenderGraph};
use crate::layouts::{SharedLayouts, DEPTH_FORMAT, HDR_FORMAT, PICK_FORMAT, VELOCITY_FORMAT};
use crate::passes::debug_lines::DebugLinesPass;
use crate::passes::environment::{cube_face_view, EnvironmentPass, PREFILTER_MIPS};
use crate::passes::mesh::{
    draw_lists, AlphaMode, DrawObject, IndirectionBuilder, InstanceBuffers, MaterialCache,
    MeshPass, MeshPipelineKey, MeshPipelineStore, MeshPipelines, PassDrawList,
};
use crate::passes::post::{BloomPass, DebugViewPass, SsaoParams, SsaoPass, TaaPass, TonemapPass};
use crate::passes::probes::{ProbeKey, ProbeResources, ProbeSlot, MAX_PROBES, PROBE_RESOLUTION};
use crate::passes::sprites::SpritePass;
use crate::picking::{PickResult, PickingState};
use crate::quality::{QualityPreset, QualitySettings};
use crate::shader::ShaderLibrary;
use crate::shadows::{
    aabb_in_clip_volume, atlas_tile, atlas_viewport, fit_cascades, point_face_projection,
    point_face_views, select_local_shadow_casters, spot_view_proj, LocalShadowKind, MAX_CASCADES,
    MAX_LOCAL_SHADOW_MATRICES,
};
use crate::taa::HaltonSequence;
use crate::uniforms::{
    cols, InstanceGpu, ProbeGpu, ProbeUniform, ShadowUniform, ViewUniform, FEATURE_FOG,
    FEATURE_IBL, FEATURE_PROBES, FEATURE_SHADOWS_CSM, FEATURE_SHADOWS_LOCAL, FEATURE_SSAO,
    INSTANCE_FLAG_RECEIVE_SHADOWS, INSTANCE_FLAG_SELECTED,
};

pub struct FrameRendererConfig {
    pub shader_root: PathBuf,
    pub shader_cache: PathBuf,
    pub clear_color: wgpu::Color,
    pub max_lights: usize,
    pub quality: QualityPreset,
}

impl Default for FrameRendererConfig {
    fn default() -> Self {
        Self {
            shader_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders"),
            shader_cache: PathBuf::from(".starman/shader-cache"),
            clear_color: wgpu::Color {
                r: 0.02,
                g: 0.03,
                b: 0.05,
                a: 1.0,
            },
            max_lights: 256,
            quality: QualityPreset::Medium,
        }
    }
}

/// Per-frame counters and CPU timings (editor status bar, smoke report,
/// performance budgets).
#[derive(Clone, Debug, Default)]
pub struct RenderStats {
    pub total_meshes: usize,
    pub visible_meshes: usize,
    pub transparent_meshes: usize,
    pub skinned_instances: usize,
    pub draw_calls: u32,
    pub shadow_casters: usize,
    pub lights: usize,
    pub local_shadow_slots: usize,
    pub max_cluster_lights: u32,
    pub cascades: u32,
    pub probes_active: usize,
    pub uploaded_bytes: u64,
    pub debug_lines: u32,
    pub pipeline_variants: usize,
    pub gpu_meshes: usize,
    pub gpu_textures: usize,
    pub environment_bakes: u64,
    pub probe_bakes: u64,
    pub cpu_extract_ms: f32,
    pub cpu_prepare_ms: f32,
    pub cpu_encode_ms: f32,
    pub extensions: Vec<(String, f64)>,
}

struct Targets {
    depth_view: wgpu::TextureView,
    velocity_view: wgpu::TextureView,
    hdr_view: wgpu::TextureView,
}

impl Targets {
    fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let make = |label, format, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let attach_sample =
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let depth = make("engine-render-depth", DEPTH_FORMAT, attach_sample);
        let velocity = make("engine-render-velocity", VELOCITY_FORMAT, attach_sample);
        let hdr = make("engine-render-hdr", HDR_FORMAT, attach_sample);
        Self {
            depth_view: depth.create_view(&Default::default()),
            velocity_view: velocity.create_view(&Default::default()),
            hdr_view: hdr.create_view(&Default::default()),
        }
    }
}

struct ShadowMaps {
    csm_resolution: u32,
    csm_view: wgpu::TextureView,
    csm_layers: Vec<wgpu::TextureView>,
    atlas_size: u32,
    atlas_view: wgpu::TextureView,
}

impl ShadowMaps {
    fn new(device: &wgpu::Device, csm_resolution: u32, atlas_size: u32) -> Self {
        let csm_resolution = csm_resolution.max(64);
        let csm = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("engine-render-csm"),
            size: wgpu::Extent3d {
                width: csm_resolution,
                height: csm_resolution,
                depth_or_array_layers: MAX_CASCADES as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let csm_view = csm.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let csm_layers = (0..MAX_CASCADES as u32)
            .map(|layer| {
                csm.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let atlas_size = atlas_size.max(256);
        let atlas = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("engine-render-shadow-atlas"),
            size: wgpu::Extent3d {
                width: atlas_size,
                height: atlas_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        Self {
            csm_resolution,
            csm_view,
            csm_layers,
            atlas_size,
            atlas_view: atlas.create_view(&Default::default()),
        }
    }
}

/// Result of preparing mesh instances for this frame.
#[derive(Default)]
struct PreparedDraws {
    main_opaque: Vec<PassDrawList>,
    main_transparent: Vec<PassDrawList>,
    prepass: Vec<PassDrawList>,
    cascades: Vec<Vec<PassDrawList>>,
    local_faces: Vec<Vec<PassDrawList>>,
    picking: Vec<PassDrawList>,
    overdraw: Vec<PassDrawList>,
    probe_capture: Vec<PassDrawList>,
    pipeline_keys: Vec<MeshPipelineKey>,
}

/// View uniform offsets for this frame.
#[derive(Default)]
struct ViewOffsets {
    main: u32,
    cascades: Vec<u32>,
    local: Vec<u32>,
    probe_faces: Vec<u32>,
}

/// A probe chosen to bake this frame.
struct ProbeBake {
    entity: Entity,
    slot: usize,
    data: ProbeSlot,
}

pub struct FrameRenderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub caps: NegotiatedCapabilities,
    width: u32,
    height: u32,
    target_format: wgpu::TextureFormat,
    pub clear_color: wgpu::Color,
    max_lights: usize,
    quality: QualitySettings,
    shaders: ShaderLibrary,
    layouts: SharedLayouts,
    cache: GpuAssetCache,
    materials: MaterialCache,
    mesh_pipelines: MeshPipelines,
    instances: InstanceBuffers,
    belt: StagingBelt,
    transient: TransientPool,
    views: GrowableBuffer,
    lights_buffer: GrowableBuffer,
    cluster_ranges: GrowableBuffer,
    cluster_indices: GrowableBuffer,
    shadow_uniform: wgpu::Buffer,
    probe_uniform: wgpu::Buffer,
    targets: Targets,
    shadow_maps: ShadowMaps,
    environment: EnvironmentPass,
    probes: ProbeResources,
    ssao: SsaoPass,
    bloom: BloomPass,
    taa: TaaPass,
    tonemap: TonemapPass,
    debug_view_pass: DebugViewPass,
    debug_lines: DebugLinesPass,
    sprites: SpritePass,
    graph: RenderGraph,
    extensions: Vec<Box<dyn RenderExtension>>,
    extension_nodes: Vec<(usize, ExtensionNode)>,
    picking: PickingState,
    debug_view: DebugView,
    frame_index: u64,
    last_pass_order: Vec<PassId>,
    last_cluster_counts: Vec<u32>,
    last_light_count: usize,
    selected: Vec<Entity>,
    halton: HaltonSequence,
    last_jitter: [f32; 2],
    prev_view_proj: Option<Mat4>,
    prev_camera: Option<Entity>,
    prev_models: HashMap<Entity, [[f32; 4]; 4]>,
    prev_palettes: HashMap<Entity, Vec<[[f32; 4]; 4]>>,
    pending_probe_clears: Vec<Entity>,
    stats: RenderStats,
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
        Self::try_new(device, queue, caps, target_format, width, height, config)
            .unwrap_or_else(|error| panic!("failed to initialize the renderer: {error}"))
    }

    pub fn try_new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        caps: NegotiatedCapabilities,
        target_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        config: FrameRendererConfig,
    ) -> Result<Self> {
        let width = width.max(1);
        let height = height.max(1);
        let quality = QualitySettings::from_preset(config.quality).apply_tier_fallback(caps.tier);
        let mut shaders = ShaderLibrary::new(config.shader_root, config.shader_cache);
        let layouts = SharedLayouts::new(&device);
        let mut graph = RenderGraph::default_forward_plus();
        apply_quality_to_graph(&mut graph, &quality);
        let environment = EnvironmentPass::new(&device, &queue, &mut shaders, &layouts)?;
        let ssao = SsaoPass::new(&device, &mut shaders, &layouts, width, height)?;
        let bloom = BloomPass::new(&device, &mut shaders, width, height)?;
        let taa = TaaPass::new(&device, &mut shaders, width, height)?;
        let tonemap = TonemapPass::new(&device, &queue, &mut shaders)?;
        let debug_view_pass = DebugViewPass::new(&device, &mut shaders, &layouts)?;
        let debug_lines = DebugLinesPass::new(&device, &mut shaders, &layouts)?;
        let sprites = SpritePass::new(&device, &mut shaders)?;
        let storage = wgpu::BufferUsages::STORAGE;
        let shadow_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine-render-shadow-uniform"),
            size: std::mem::size_of::<ShadowUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let probe_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine-render-probe-uniform"),
            size: std::mem::size_of::<ProbeUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Self {
            cache: GpuAssetCache::new(&device, &queue, 3),
            materials: MaterialCache::default(),
            mesh_pipelines: MeshPipelines::new(&device, &layouts, HDR_FORMAT),
            instances: InstanceBuffers::new(&device),
            belt: StagingBelt::new(1 << 20),
            transient: TransientPool::default(),
            views: GrowableBuffer::new(
                &device,
                "engine-render-views",
                wgpu::BufferUsages::UNIFORM,
                4096,
            ),
            lights_buffer: GrowableBuffer::new(&device, "engine-render-lights", storage, 4096),
            cluster_ranges: GrowableBuffer::new(
                &device,
                "engine-render-cluster-ranges",
                storage,
                4096,
            ),
            cluster_indices: GrowableBuffer::new(
                &device,
                "engine-render-cluster-indices",
                storage,
                4096,
            ),
            shadow_uniform,
            probe_uniform,
            targets: Targets::new(&device, width, height),
            shadow_maps: ShadowMaps::new(
                &device,
                quality.shadow_map_resolution,
                quality.shadow_map_resolution * 2,
            ),
            probes: ProbeResources::new(&device),
            environment,
            ssao,
            bloom,
            taa,
            tonemap,
            debug_view_pass,
            debug_lines,
            sprites,
            graph,
            extensions: Vec::new(),
            extension_nodes: Vec::new(),
            picking: PickingState::default(),
            debug_view: DebugView::None,
            frame_index: 0,
            last_pass_order: Vec::new(),
            last_cluster_counts: Vec::new(),
            last_light_count: 0,
            selected: Vec::new(),
            halton: HaltonSequence::new(),
            last_jitter: [0.0; 2],
            prev_view_proj: None,
            prev_camera: None,
            prev_models: HashMap::new(),
            prev_palettes: HashMap::new(),
            pending_probe_clears: Vec::new(),
            stats: RenderStats::default(),
            shaders,
            layouts,
            device,
            queue,
            caps,
            width,
            height,
            target_format,
            clear_color: config.clear_color,
            max_lights: config.max_lights,
            quality,
        })
    }

    pub fn tier(&self) -> CapabilityTier {
        self.caps.tier
    }

    pub fn quality(&self) -> &QualitySettings {
        &self.quality
    }

    pub fn stats(&self) -> &RenderStats {
        &self.stats
    }

    pub fn target_format(&self) -> wgpu::TextureFormat {
        self.target_format
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn set_quality_preset(&mut self, preset: QualityPreset) {
        let exposure = self.quality.exposure;
        self.quality = QualitySettings::from_preset(preset).apply_tier_fallback(self.caps.tier);
        self.quality.exposure = exposure;
        apply_quality_to_graph(&mut self.graph, &self.quality);
        if self.shadow_maps.csm_resolution != self.quality.shadow_map_resolution {
            self.shadow_maps = ShadowMaps::new(
                &self.device,
                self.quality.shadow_map_resolution,
                self.quality.shadow_map_resolution * 2,
            );
        }
        self.taa.reset();
    }

    /// Global exposure multiplier (camera settings multiply on top).
    pub fn set_exposure(&mut self, exposure: f32) {
        self.quality.exposure = exposure.max(0.01);
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
        self.targets = Targets::new(&self.device, width, height);
        self.ssao.resize(&self.device, width, height);
        self.bloom.resize(&self.device, width, height);
        self.taa.resize(&self.device, width, height);
    }

    pub fn set_debug_view(&mut self, view: DebugView) {
        self.debug_view = view;
        self.graph
            .set_enabled(PassId("debug_view"), view.enables_debug_blit());
        self.graph
            .set_enabled(PassId("overdraw"), view == DebugView::Overdraw);
    }

    pub fn debug_view(&self) -> DebugView {
        self.debug_view
    }

    pub fn set_selected(&mut self, selected: Vec<Entity>) {
        self.selected = selected;
    }

    /// Requests the entity under pixel `(x, y)`; the answer arrives through
    /// [`Self::poll_pick`] once the GPU readback completes.
    pub fn request_pick(&mut self, x: u32, y: u32) {
        self.picking.request(x, y, self.frame_index);
    }

    pub fn poll_pick(&mut self) -> Option<PickResult> {
        self.picking.update(&self.device);
        self.picking.poll()
    }

    pub fn last_pass_order(&self) -> &[PassId] {
        &self.last_pass_order
    }

    pub fn last_light_count(&self) -> usize {
        self.last_light_count
    }

    pub fn last_jitter(&self) -> [f32; 2] {
        self.last_jitter
    }

    pub fn graph(&self) -> &RenderGraph {
        &self.graph
    }

    pub fn shader_library(&mut self) -> &mut ShaderLibrary {
        &mut self.shaders
    }

    /// Installs an extension directly (hosts that do not use the world
    /// registry, tests).
    pub fn add_extension(&mut self, extension: Box<dyn RenderExtension>) {
        let index = self.extensions.len();
        for node in extension.nodes() {
            let pass = PassNode {
                id: PassId(node.name),
                reads: Vec::new(),
                writes: Vec::new(),
                after: node.after.iter().map(|name| PassId(name)).collect(),
                enabled: true,
            };
            let before: Vec<PassId> = node.before.iter().map(|name| PassId(name)).collect();
            self.graph.add_pass_before(pass, &before);
            self.extension_nodes.push((index, node));
        }
        self.extensions.push(extension);
    }

    pub fn extension_names(&self) -> Vec<&'static str> {
        self.extensions.iter().map(|ext| ext.name()).collect()
    }

    fn sync_extensions(&mut self, world: &World) {
        let Some(registry) = world.get_resource::<RenderExtensions>() else {
            return;
        };
        let existing = self.extension_names();
        for extension in registry.instantiate_missing(&existing) {
            log::info!(target: "engine::render", "render extension '{}' installed", extension.name());
            self.add_extension(extension);
        }
    }

    pub fn render_to_view(
        &mut self,
        world: &mut World,
        assets: &AssetServer,
        target_view: &wgpu::TextureView,
    ) -> Result<()> {
        let frame_start = Instant::now();
        self.frame_index = self.frame_index.saturating_add(1);
        self.cache.begin_frame();
        self.materials.begin_frame();
        self.transient.begin_frame();
        self.picking.update(&self.device);
        self.sync_extensions(world);

        for entity in self.pending_probe_clears.drain(..) {
            if let Some(mut probe) = world.get_mut::<ReflectionProbe>(entity) {
                probe.bake = false;
            }
        }

        // ---- extract ---------------------------------------------------
        let selected = self.selected.clone();
        let render_world = extract_render_world(world, self.width, self.height, &selected);
        let info = ExtractInfo {
            width: self.width,
            height: self.height,
            frame_index: self.frame_index,
        };
        for extension in &mut self.extensions {
            extension.extract(world, &info);
        }
        let extract_done = Instant::now();

        // ---- prepare ---------------------------------------------------
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("engine-render-frame"),
            });
        let mut stats = RenderStats {
            total_meshes: render_world.meshes.len(),
            debug_lines: render_world.debug_lines.len() as u32,
            ..RenderStats::default()
        };

        let camera = render_world.camera.clone();
        if let Some(camera) = &camera {
            if self.prev_camera != Some(camera.entity) {
                self.taa.reset();
                self.prev_view_proj = None;
                self.prev_camera = Some(camera.entity);
            }
        }
        let settings = camera
            .as_ref()
            .map(|c| c.settings.clone())
            .unwrap_or_default();
        let taa_active = self.quality.taa_enabled && camera.is_some();
        let samples = if camera.is_some() {
            self.quality.effective_msaa_samples().clamp(1, 4)
        } else {
            1
        };
        let samples = if samples > 1 { 4 } else { 1 };
        self.last_jitter = if taa_active {
            self.halton.next_jitter(self.width, self.height)
        } else {
            [0.0; 2]
        };

        // Lights and clusters.
        let (mut gpu_lights, directional_count) = build_gpu_lights(
            &render_world.lights,
            self.max_lights.min(self.quality.max_lights),
        );
        self.last_light_count = gpu_lights.len();
        stats.lights = gpu_lights.len();

        let environment = render_world.environment.clone().unwrap_or_default();
        let sun = render_world
            .lights
            .iter()
            .find_map(|light| match light.kind {
                crate::extract::ExtractedLightKind::Directional {
                    direction,
                    color,
                    intensity,
                } => Some((Vec3::from_array(direction), color, intensity)),
                _ => None,
            });
        if self.environment.update(
            &self.device,
            &self.queue,
            &mut encoder,
            &self.layouts,
            assets.assets(),
            &environment,
            sun,
        ) {
            // A new environment invalidates baked probes.
            for slot in &mut self.probes.slots {
                *slot = None;
            }
        }

        let (view_matrix, proj, jittered_proj) = match &camera {
            Some(camera) => {
                let proj = camera.projection(self.width, self.height);
                let mut jittered = proj;
                jittered.z_axis.x += self.last_jitter[0];
                jittered.z_axis.y += self.last_jitter[1];
                (camera.view(), proj, jittered)
            }
            None => (Mat4::IDENTITY, Mat4::IDENTITY, Mat4::IDENTITY),
        };
        let view_proj = jittered_proj * view_matrix;
        let unjittered_view_proj = proj * view_matrix;
        let (near, far) = camera
            .as_ref()
            .map(|c| (c.near, c.far))
            .unwrap_or((0.1, 1000.0));
        let cluster_layout = ClusterLayout::new(
            self.width,
            self.height,
            near,
            far,
            DEFAULT_TILE_PX,
            DEFAULT_DEPTH_SLICES,
        );

        // Shadows: choose casters and matrices.
        let mut shadow_uniform = ShadowUniform::default();
        let csm_enabled = camera.is_some()
            && self.quality.cascade_count > 0
            && render_world.lights.first().is_some_and(|light| {
                light.cast_shadows
                    && matches!(
                        light.kind,
                        crate::extract::ExtractedLightKind::Directional { .. }
                    )
            });
        let mut cascade_matrices: Vec<Mat4> = Vec::new();
        if let (true, Some(camera), Some((dir, _, _))) = (csm_enabled, &camera, sun) {
            let csm = fit_cascades(
                dir,
                camera.world,
                camera.fov_y_radians,
                self.width as f32 / self.height as f32,
                camera.near,
                self.quality.shadow_distance.min(camera.far),
                self.quality.cascade_count,
                self.shadow_maps.csm_resolution,
                self.quality.shadow_distance * 0.5,
            );
            for (i, cascade) in csm.cascades.iter().enumerate() {
                shadow_uniform.cascades[i] = cols(cascade.view_proj);
                shadow_uniform.cascade_splits[i] = cascade.far;
                cascade_matrices.push(cascade.view_proj);
            }
            shadow_uniform.csm_params = [
                1.0 / self.shadow_maps.csm_resolution as f32,
                0.0015,
                0.03,
                csm.cascades.len() as f32,
            ];
        }
        stats.cascades = cascade_matrices.len() as u32;

        // Local shadow slots.
        let local_candidates: Vec<(u32, f32, LocalShadowKind)> = render_world
            .lights
            .iter()
            .zip(0u32..)
            .filter(|(light, _)| light.cast_shadows)
            .filter_map(|(light, _)| {
                let index = gpu_lights.iter().position(|gpu| match light.kind {
                    crate::extract::ExtractedLightKind::Point { position, .. }
                    | crate::extract::ExtractedLightKind::Spot { position, .. } => {
                        gpu.position_range[..3] == position
                            && gpu.light_type() != GpuLight::TYPE_DIRECTIONAL
                    }
                    _ => false,
                })? as u32;
                let gpu = &gpu_lights[index as usize];
                let kind = if gpu.light_type() == GpuLight::TYPE_SPOT {
                    LocalShadowKind::Spot
                } else {
                    LocalShadowKind::Point
                };
                let distance = camera
                    .as_ref()
                    .map(|c| {
                        c.position()
                            .distance(Vec3::from_slice(&gpu.position_range[..3]))
                    })
                    .unwrap_or(0.0);
                let score = gpu.color_intensity[3] * gpu.position_range[3] / (1.0 + distance);
                Some((index, score, kind))
            })
            .collect();
        let local_budget = if camera.is_some() {
            self.quality.local_shadow_slots
        } else {
            0
        };
        let local_slots = select_local_shadow_casters(&local_candidates, local_budget);
        let mut local_matrices: Vec<Mat4> = Vec::new();
        for slot in &local_slots {
            let light = &mut gpu_lights[slot.light_index as usize];
            let needed = slot.kind.face_count();
            if local_matrices.len() + needed > MAX_LOCAL_SHADOW_MATRICES {
                continue;
            }
            let position = Vec3::from_slice(&light.position_range[..3]);
            let range = light.position_range[3];
            light.params[1] = local_matrices.len() as u32;
            match slot.kind {
                LocalShadowKind::Spot => local_matrices.push(spot_view_proj(
                    position,
                    Vec3::from_slice(&light.direction_cone[..3]),
                    light.direction_cone[3],
                    range,
                )),
                LocalShadowKind::Point => {
                    let proj = point_face_projection(range, 0.02);
                    for face in point_face_views(position) {
                        local_matrices.push(proj * face);
                    }
                }
            }
        }
        for (i, matrix) in local_matrices.iter().enumerate() {
            shadow_uniform.local_matrices[i] = cols(*matrix);
            shadow_uniform.local_tiles[i] = atlas_tile(i as u32);
        }
        shadow_uniform.local_params = [1.0 / self.shadow_maps.atlas_size as f32, 0.0008, 0.02, 1.0];
        stats.local_shadow_slots = local_slots.len();

        let clusters = assign_clusters(
            cluster_layout,
            &gpu_lights,
            view_matrix,
            proj,
            self.width,
            self.height,
        );
        stats.max_cluster_lights = clusters.max_lights_in_cluster();
        self.last_cluster_counts = clusters.ranges.iter().map(|range| range[1]).collect();
        // One extra cluster holding every local light (probe capture views).
        let mut ranges = clusters.ranges.clone();
        let mut indices = clusters.indices.clone();
        let all_locals: Vec<u32> = (directional_count..gpu_lights.len() as u32).collect();
        ranges.push([indices.len() as u32, all_locals.len() as u32]);
        indices.extend_from_slice(&all_locals);
        let all_cluster_index = (ranges.len() - 1) as u32;
        let lights_upload: Vec<GpuLight> = if gpu_lights.is_empty() {
            vec![GpuLight {
                position_range: [0.0; 4],
                color_intensity: [0.0; 4],
                direction_cone: [0.0; 4],
                params: [GpuLight::TYPE_POINT, GpuLight::NO_SHADOW, 0, 0],
            }]
        } else {
            gpu_lights.clone()
        };
        self.lights_buffer.upload(
            &self.device,
            &mut encoder,
            &mut self.belt,
            bytemuck::cast_slice(&lights_upload),
        );
        self.cluster_ranges.upload(
            &self.device,
            &mut encoder,
            &mut self.belt,
            bytemuck::cast_slice(&ranges),
        );
        self.cluster_indices.upload(
            &self.device,
            &mut encoder,
            &mut self.belt,
            bytemuck::cast_slice(&indices),
        );
        self.belt.write_buffer(
            &self.device,
            &mut encoder,
            &self.shadow_uniform,
            0,
            bytemuck::bytes_of(&shadow_uniform),
        );

        // Probes: slots, one bake per frame.
        let probe_bake = self.choose_probe_bake(&render_world, camera.is_some());
        let mut probe_uniform = ProbeUniform {
            ibl_params: [
                environment.ibl_intensity,
                PREFILTER_MIPS as f32,
                environment.sky_intensity,
                0.0,
            ],
            ..Default::default()
        };
        let mut probe_count = 0u32;
        for (slot, data) in self.probes.active() {
            if probe_count as usize >= self.quality.max_probes.min(MAX_PROBES as u32) as usize {
                break;
            }
            probe_uniform.probes[probe_count as usize] = ProbeGpu {
                position: [data.position[0], data.position[1], data.position[2], 0.0],
                extents_intensity: [
                    data.half_extents[0],
                    data.half_extents[1],
                    data.half_extents[2],
                    data.intensity,
                ],
                params: [slot as u32, data.blend_distance.to_bits(), 0, 0],
            };
            probe_count += 1;
        }
        stats.probes_active = probe_count as usize;
        self.belt.write_buffer(
            &self.device,
            &mut encoder,
            &self.probe_uniform,
            0,
            bytemuck::bytes_of(&probe_uniform),
        );

        // Meshes → instances and per-pass draw lists.
        let frustum = Frustum::from_view_proj(unjittered_view_proj);
        let draws = self.prepare_meshes(
            &render_world,
            assets,
            &frustum,
            view_matrix,
            &cascade_matrices,
            &local_matrices,
            samples,
            &mut encoder,
            &mut stats,
        )?;

        // View uniforms.
        let align = self.device.limits().min_uniform_buffer_offset_alignment as u64;
        let mut writer = AlignedWriter::new(align);
        let mut feature_bits = FEATURE_IBL;
        if !cascade_matrices.is_empty() {
            feature_bits |= FEATURE_SHADOWS_CSM;
        }
        if !local_matrices.is_empty() {
            feature_bits |= FEATURE_SHADOWS_LOCAL;
        }
        let ssao_active = self.quality.ssao_enabled && camera.is_some();
        if ssao_active {
            feature_bits |= FEATURE_SSAO;
        }
        if settings.fog_enabled && self.quality.fog_enabled {
            feature_bits |= FEATURE_FOG;
        }
        if probe_count > 0 {
            feature_bits |= FEATURE_PROBES;
        }
        let elapsed = frame_start.duration_since(*process_start()).as_secs_f32();
        let main_view = ViewUniform {
            view: cols(view_matrix),
            proj: cols(jittered_proj),
            view_proj: cols(view_proj),
            inv_view_proj: cols(view_proj.inverse()),
            prev_view_proj: cols(self.prev_view_proj.unwrap_or(unjittered_view_proj)),
            unjittered_view_proj: cols(unjittered_view_proj),
            camera_position: camera
                .as_ref()
                .map(|c| c.position().extend(elapsed).to_array())
                .unwrap_or([0.0, 0.0, 0.0, elapsed]),
            viewport: [
                self.width as f32,
                self.height as f32,
                1.0 / self.width as f32,
                1.0 / self.height as f32,
            ],
            jitter: [self.last_jitter[0], self.last_jitter[1], 0.0, 0.0],
            near_far: [
                near,
                far,
                self.quality.exposure * settings.exposure,
                self.frame_index as f32,
            ],
            fog_color_density: [
                settings.fog_color[0],
                settings.fog_color[1],
                settings.fog_color[2],
                settings.fog_density,
            ],
            fog_params: [
                settings.fog_height_falloff,
                settings.fog_base_height,
                settings.fog_start_distance,
                settings.fog_max_opacity,
            ],
            ambient: [
                environment.ambient_tint[0],
                environment.ambient_tint[1],
                environment.ambient_tint[2],
                settings.ssao_intensity,
            ],
            counts: [directional_count, 0, probe_count, feature_bits],
            cluster_dims: cluster_layout.dims(),
            cluster_params: cluster_layout.params(),
        };
        let mut offsets = ViewOffsets {
            main: writer.push(&main_view),
            ..Default::default()
        };
        for matrix in cascade_matrices.iter().chain(local_matrices.iter()) {
            let shadow_view = ViewUniform {
                view_proj: cols(*matrix),
                unjittered_view_proj: cols(*matrix),
                prev_view_proj: cols(*matrix),
                inv_view_proj: cols(matrix.inverse()),
                ..main_view
            };
            let offset = writer.push(&shadow_view);
            if offsets.cascades.len() < cascade_matrices.len() {
                offsets.cascades.push(offset);
            } else {
                offsets.local.push(offset);
            }
        }
        if let Some(bake) = &probe_bake {
            let position = Vec3::from_array(bake.data.position);
            let probe_proj =
                Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.05, far.min(500.0));
            for face in 0..6 {
                let face_view = cube_face_view(face, position);
                let face_vp = probe_proj * face_view;
                let probe_view = ViewUniform {
                    view: cols(face_view),
                    proj: cols(probe_proj),
                    view_proj: cols(face_vp),
                    inv_view_proj: cols(face_vp.inverse()),
                    prev_view_proj: cols(face_vp),
                    unjittered_view_proj: cols(face_vp),
                    camera_position: position.extend(elapsed).to_array(),
                    viewport: [
                        PROBE_RESOLUTION as f32,
                        PROBE_RESOLUTION as f32,
                        1.0 / PROBE_RESOLUTION as f32,
                        1.0 / PROBE_RESOLUTION as f32,
                    ],
                    jitter: [0.0; 4],
                    counts: [
                        directional_count,
                        all_cluster_index,
                        0,
                        feature_bits & !(FEATURE_SSAO | FEATURE_PROBES),
                    ],
                    cluster_dims: [1, 1, 1, u32::MAX],
                    cluster_params: [0.0, 0.0, near, far],
                    ..main_view
                };
                offsets.probe_faces.push(writer.push(&probe_view));
            }
        }
        self.views
            .upload(&self.device, &mut encoder, &mut self.belt, writer.bytes());

        // Extensions upload.
        let view_for_ext = camera.as_ref().map(|_| main_view);
        for extension in &mut self.extensions {
            let mut ctx = PrepareContext {
                device: &self.device,
                queue: &self.queue,
                encoder: &mut encoder,
                belt: &mut self.belt,
                shaders: &mut self.shaders,
                cache: &mut self.cache,
                server: assets,
                layouts: &self.layouts,
                view: view_for_ext.as_ref(),
                width: self.width,
                height: self.height,
                target_format: self.target_format,
                tier: self.caps.tier,
                quality: &self.quality,
                frame_index: self.frame_index,
            };
            if let Err(error) = extension.prepare(&mut ctx) {
                log::error!(target: "engine::render", "render extension '{}' prepare failed: {error}", extension.name());
            }
        }

        // 2D sprites.
        let sprite_items: Vec<DrawItem2d> = render_world
            .sprites
            .iter()
            .map(|sprite| DrawItem2d {
                texture: sprite.texture,
                model: sprite.model,
                color: sprite.color,
                uv_rect: sprite.uv_rect,
                sort_z: sprite.sort_z,
            })
            .collect();
        self.sprites.prepare(
            &self.device,
            &self.queue,
            &mut encoder,
            &mut self.belt,
            &mut self.cache,
            assets,
            render_world.camera_2d,
            sprite_items,
        );
        self.debug_lines.prepare(
            &self.device,
            &mut encoder,
            &mut self.belt,
            &render_world.debug_lines,
        );

        // Pipelines needed this frame.
        for key in &draws.pipeline_keys {
            self.mesh_pipelines
                .get(&self.device, &mut self.shaders, *key)?;
        }
        if camera.is_some() {
            self.environment
                .prepare_skybox(&self.device, &self.layouts, samples);
            self.environment
                .prepare_skybox(&self.device, &self.layouts, 1);
        }
        stats.pipeline_variants = self.mesh_pipelines.variant_count();
        let prepare_done = Instant::now();

        // ---- encode ----------------------------------------------------
        let view_basic_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("engine-render-view-basic-bg"),
            layout: &self.layouts.view_basic,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: self.views.buffer(),
                    offset: 0,
                    size: std::num::NonZeroU64::new(self.layouts.view_uniform_size),
                }),
            }],
        });
        let view_lit_bg = self.lit_bind_group();
        let instances_bg = self
            .instances
            .bind_group(&self.device, &self.layouts)
            .clone();
        let pipeline_store: MeshPipelineStore<'_> = draws
            .pipeline_keys
            .iter()
            .filter_map(|key| {
                self.mesh_pipelines
                    .lookup(key)
                    .map(|pipeline| (*key, pipeline))
            })
            .collect();

        let schedule = self.graph.schedule()?;
        self.last_pass_order = schedule.clone();
        let mut hdr_current = self.targets.hdr_view.clone();
        let mut draw_calls = 0u32;
        let mut msaa: Option<(wgpu::TextureView, wgpu::TextureView)> = None;
        let has_camera = camera.is_some();
        let bloom_enabled = self.quality.bloom_enabled && has_camera;

        for pass in &schedule {
            match pass.0 {
                "shadow_csm" if has_camera => {
                    for (cascade, lists) in draws.cascades.iter().enumerate() {
                        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("engine-render-shadow-cascade"),
                            color_attachments: &[],
                            depth_stencil_attachment: Some(
                                wgpu::RenderPassDepthStencilAttachment {
                                    view: &self.shadow_maps.csm_layers[cascade],
                                    depth_ops: Some(wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(1.0),
                                        store: wgpu::StoreOp::Store,
                                    }),
                                    stencil_ops: None,
                                },
                            ),
                            occlusion_query_set: None,
                            timestamp_writes: None,
                        });
                        rp.set_bind_group(0, &view_basic_bg, &[offsets.cascades[cascade]]);
                        draw_calls += draw_lists(
                            &mut rp,
                            lists,
                            &pipeline_store,
                            &instances_bg,
                            &self.materials,
                            &self.cache,
                        );
                    }
                }
                "shadow_local" if has_camera && !draws.local_faces.is_empty() => {
                    let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("engine-render-shadow-local"),
                        color_attachments: &[],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &self.shadow_maps.atlas_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        occlusion_query_set: None,
                        timestamp_writes: None,
                    });
                    for (index, lists) in draws.local_faces.iter().enumerate() {
                        let (x, y, size) =
                            atlas_viewport(index as u32, self.shadow_maps.atlas_size);
                        rp.set_viewport(x as f32, y as f32, size as f32, size as f32, 0.0, 1.0);
                        rp.set_scissor_rect(x, y, size, size);
                        rp.set_bind_group(0, &view_basic_bg, &[offsets.local[index]]);
                        draw_calls += draw_lists(
                            &mut rp,
                            lists,
                            &pipeline_store,
                            &instances_bg,
                            &self.materials,
                            &self.cache,
                        );
                    }
                }
                "probe_bake" => {
                    if let Some(bake) = &probe_bake {
                        draw_calls += self.encode_probe_bake(
                            &mut encoder,
                            bake,
                            &offsets,
                            &view_lit_bg,
                            &view_basic_bg,
                            &instances_bg,
                            &draws.probe_capture,
                            &pipeline_store,
                        );
                    }
                }
                "prepass" if has_camera => {
                    let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("engine-render-prepass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &self.targets.velocity_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &self.targets.depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        occlusion_query_set: None,
                        timestamp_writes: None,
                    });
                    rp.set_bind_group(0, &view_basic_bg, &[offsets.main]);
                    draw_calls += draw_lists(
                        &mut rp,
                        &draws.prepass,
                        &pipeline_store,
                        &instances_bg,
                        &self.materials,
                        &self.cache,
                    );
                }
                "prepass" => {
                    // No camera: still clear depth so overlays can bind it.
                    let _rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("engine-render-clear-depth"),
                        color_attachments: &[],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &self.targets.depth_view,
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
                "ssao" => {
                    if ssao_active {
                        self.ssao.encode(
                            &self.device,
                            &self.queue,
                            &mut encoder,
                            self.views.buffer(),
                            self.layouts.view_uniform_size,
                            offsets.main,
                            &self.targets.depth_view,
                            SsaoParams {
                                radius: settings.ssao_radius,
                                bias: 0.025,
                                power: 1.5,
                                samples: self.quality.ssao_samples,
                            },
                        );
                    } else {
                        self.ssao.clear(&mut encoder);
                    }
                }
                "opaque" => {
                    if !ssao_active && !self.graph.is_enabled(PassId("ssao")) {
                        self.ssao.clear(&mut encoder);
                    }
                    let clear = wgpu::Color {
                        r: self.clear_color.r,
                        g: self.clear_color.g,
                        b: self.clear_color.b,
                        a: 1.0,
                    };
                    if samples > 1 {
                        let key = |format, usage| TransientKey {
                            width: self.width,
                            height: self.height,
                            format,
                            mip_levels: 1,
                            layers: 1,
                            samples,
                            usage,
                        };
                        let color = self
                            .transient
                            .acquire(
                                &self.device,
                                "engine-render-msaa-color",
                                key(HDR_FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT),
                            )
                            .create_view(&Default::default());
                        let depth = self
                            .transient
                            .acquire(
                                &self.device,
                                "engine-render-msaa-depth",
                                key(DEPTH_FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT),
                            )
                            .create_view(&Default::default());
                        msaa = Some((color, depth));
                    }
                    let (color, depth, depth_load) = match &msaa {
                        Some((color, depth)) => (color, depth, wgpu::LoadOp::Clear(1.0)),
                        None => (
                            &self.targets.hdr_view,
                            &self.targets.depth_view,
                            wgpu::LoadOp::Load,
                        ),
                    };
                    let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("engine-render-opaque"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: color,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(clear),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: depth,
                            depth_ops: Some(wgpu::Operations {
                                load: depth_load,
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        occlusion_query_set: None,
                        timestamp_writes: None,
                    });
                    if has_camera {
                        rp.set_bind_group(0, &view_lit_bg, &[offsets.main]);
                        draw_calls += draw_lists(
                            &mut rp,
                            &draws.main_opaque,
                            &pipeline_store,
                            &instances_bg,
                            &self.materials,
                            &self.cache,
                        );
                    }
                }
                "skybox" if has_camera => {
                    let (color, depth) = match &msaa {
                        Some((color, depth)) => (color, depth),
                        None => (&self.targets.hdr_view, &self.targets.depth_view),
                    };
                    let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("engine-render-skybox"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: color,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: depth,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        occlusion_query_set: None,
                        timestamp_writes: None,
                    });
                    self.environment
                        .draw_skybox(&mut rp, &view_basic_bg, offsets.main, samples);
                }
                "transparent_3d" if has_camera => {
                    let (color, resolve, depth) = match &msaa {
                        Some((color, depth)) => (color, Some(&self.targets.hdr_view), depth),
                        None => (&self.targets.hdr_view, None, &self.targets.depth_view),
                    };
                    let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("engine-render-transparent"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: color,
                            resolve_target: resolve,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: depth,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        occlusion_query_set: None,
                        timestamp_writes: None,
                    });
                    rp.set_bind_group(0, &view_lit_bg, &[offsets.main]);
                    draw_calls += draw_lists(
                        &mut rp,
                        &draws.main_transparent,
                        &pipeline_store,
                        &instances_bg,
                        &self.materials,
                        &self.cache,
                    );
                }
                "sprites_2d" => {
                    self.sprites.encode(
                        &self.device,
                        &mut encoder,
                        &self.layouts,
                        &self.cache,
                        &hdr_current,
                    );
                }
                "taa" if taa_active => {
                    hdr_current = self.taa.encode(
                        &self.device,
                        &self.queue,
                        &mut encoder,
                        &self.layouts,
                        &hdr_current,
                        &self.targets.velocity_view,
                    );
                }
                "bloom" if bloom_enabled => {
                    self.bloom.encode(
                        &self.device,
                        &mut encoder,
                        &self.layouts,
                        &hdr_current,
                        settings.bloom_threshold,
                    );
                }
                "tonemap" => {
                    let bloom_view = bloom_enabled.then_some(&self.bloom.output_view);
                    self.tonemap.encode(
                        &self.device,
                        &self.queue,
                        &mut encoder,
                        &self.layouts,
                        &hdr_current,
                        bloom_view,
                        target_view,
                        self.target_format,
                        self.quality.exposure * settings.exposure,
                        if bloom_enabled {
                            settings.bloom_intensity
                        } else {
                            0.0
                        },
                    );
                }
                "overdraw" if has_camera => {
                    let overdraw = self
                        .transient
                        .acquire(
                            &self.device,
                            "engine-render-overdraw",
                            TransientKey {
                                width: self.width,
                                height: self.height,
                                format: HDR_FORMAT,
                                mip_levels: 1,
                                layers: 1,
                                samples: 1,
                                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                                    | wgpu::TextureUsages::TEXTURE_BINDING,
                            },
                        )
                        .create_view(&Default::default());
                    {
                        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("engine-render-overdraw"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &overdraw,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: Some(
                                wgpu::RenderPassDepthStencilAttachment {
                                    view: &self.targets.depth_view,
                                    depth_ops: None,
                                    stencil_ops: None,
                                },
                            ),
                            occlusion_query_set: None,
                            timestamp_writes: None,
                        });
                        rp.set_bind_group(0, &view_basic_bg, &[offsets.main]);
                        draw_calls += draw_lists(
                            &mut rp,
                            &draws.overdraw,
                            &pipeline_store,
                            &instances_bg,
                            &self.materials,
                            &self.cache,
                        );
                    }
                    self.debug_view_pass.encode(
                        &self.device,
                        &self.queue,
                        &mut encoder,
                        self.views.buffer(),
                        self.layouts.view_uniform_size,
                        offsets.main,
                        &self.targets.depth_view,
                        &overdraw,
                        self.cluster_ranges.buffer(),
                        4,
                        target_view,
                        self.target_format,
                    );
                }
                "debug_view" if has_camera && self.debug_view != DebugView::Overdraw => {
                    let (mode, aux) = match self.debug_view {
                        DebugView::Depth => (1, &self.targets.velocity_view),
                        DebugView::Normals => (2, &self.targets.velocity_view),
                        DebugView::Clusters => (3, &self.targets.velocity_view),
                        DebugView::LightHeat => (5, &self.targets.velocity_view),
                        DebugView::Ssao => (6, &self.ssao.ao_view),
                        DebugView::Velocity => (7, &self.targets.velocity_view),
                        DebugView::Overdraw | DebugView::None => (0, &self.targets.velocity_view),
                    };
                    if mode != 0 {
                        self.debug_view_pass.encode(
                            &self.device,
                            &self.queue,
                            &mut encoder,
                            self.views.buffer(),
                            self.layouts.view_uniform_size,
                            offsets.main,
                            &self.targets.depth_view,
                            aux,
                            self.cluster_ranges.buffer(),
                            mode,
                            target_view,
                            self.target_format,
                        );
                    }
                }
                "debug_lines" if has_camera => {
                    self.debug_lines.encode(
                        &self.device,
                        &mut encoder,
                        target_view,
                        self.target_format,
                        &self.targets.depth_view,
                        &view_basic_bg,
                        offsets.main,
                    );
                }
                "picking" if has_camera && self.picking.needs_id_pass() => {
                    let key = |format, usage| TransientKey {
                        width: self.width,
                        height: self.height,
                        format,
                        mip_levels: 1,
                        layers: 1,
                        samples: 1,
                        usage,
                    };
                    let ids = self.transient.acquire(
                        &self.device,
                        "engine-render-pick-ids",
                        key(
                            PICK_FORMAT,
                            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                        ),
                    );
                    let pick_depth = self
                        .transient
                        .acquire(
                            &self.device,
                            "engine-render-pick-depth",
                            key(DEPTH_FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT),
                        )
                        .create_view(&Default::default());
                    {
                        let ids_view = ids.create_view(&Default::default());
                        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("engine-render-picking"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &ids_view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: Some(
                                wgpu::RenderPassDepthStencilAttachment {
                                    view: &pick_depth,
                                    depth_ops: Some(wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(1.0),
                                        store: wgpu::StoreOp::Discard,
                                    }),
                                    stencil_ops: None,
                                },
                            ),
                            occlusion_query_set: None,
                            timestamp_writes: None,
                        });
                        rp.set_bind_group(0, &view_basic_bg, &[offsets.main]);
                        draw_calls += draw_lists(
                            &mut rp,
                            &draws.picking,
                            &pipeline_store,
                            &instances_bg,
                            &self.materials,
                            &self.cache,
                        );
                    }
                    self.picking.id_map = render_world
                        .meshes
                        .iter()
                        .map(|m| (m.pick_id, m.entity))
                        .chain(render_world.sprites.iter().map(|s| (s.pick_id, s.entity)))
                        .collect();
                    self.picking.encode_copy(&self.device, &mut encoder, &ids);
                }
                name => {
                    let Some((extension_index, node)) = self
                        .extension_nodes
                        .iter()
                        .find(|(_, node)| node.name == name)
                        .cloned()
                    else {
                        continue;
                    };
                    let (color, color_format) = match node.target {
                        NodeTarget::Hdr => (&hdr_current, HDR_FORMAT),
                        NodeTarget::Output => (target_view, self.target_format),
                    };
                    let mut ctx = EncodeContext {
                        device: &self.device,
                        queue: &self.queue,
                        encoder: &mut encoder,
                        layouts: &self.layouts,
                        cache: &self.cache,
                        color,
                        color_format,
                        depth: &self.targets.depth_view,
                        view_bind_group: has_camera.then_some((&view_basic_bg, offsets.main)),
                        view: view_for_ext.as_ref(),
                        width: self.width,
                        height: self.height,
                        tier: self.caps.tier,
                        frame_index: self.frame_index,
                    };
                    let extension = &mut self.extensions[extension_index];
                    if let Err(error) = extension.encode(node.name, &mut ctx) {
                        log::error!(
                            target: "engine::render",
                            "render extension '{}' node '{}' failed: {error}",
                            extension.name(),
                            node.name
                        );
                    }
                }
            }
        }
        drop(pipeline_store);

        self.belt.finish();
        stats.uploaded_bytes = self.belt.bytes_this_frame();
        self.queue.submit(Some(encoder.finish()));
        self.belt.recall();
        self.picking.after_submit();

        // Frame history.
        if let Some(bake) = probe_bake {
            self.probes.slots[bake.slot] = Some(bake.data);
            self.probes.bakes += 1;
            self.pending_probe_clears.push(bake.entity);
        }
        self.prev_view_proj = camera.as_ref().map(|_| unjittered_view_proj);
        self.prev_models = render_world
            .meshes
            .iter()
            .map(|mesh| (mesh.entity, mesh.model))
            .collect();
        self.prev_palettes = render_world
            .meshes
            .iter()
            .filter_map(|mesh| {
                mesh.palette.map(|(offset, count)| {
                    (
                        mesh.entity,
                        render_world.palettes[offset as usize..(offset + count) as usize].to_vec(),
                    )
                })
            })
            .collect();

        let done = Instant::now();
        stats.draw_calls = draw_calls;
        stats.gpu_meshes = self.cache.mesh_count();
        stats.gpu_textures = self.cache.texture_count();
        stats.environment_bakes = self.environment.bake_count;
        stats.probe_bakes = self.probes.bakes;
        stats.cpu_extract_ms = (extract_done - frame_start).as_secs_f32() * 1000.0;
        stats.cpu_prepare_ms = (prepare_done - extract_done).as_secs_f32() * 1000.0;
        stats.cpu_encode_ms = (done - prepare_done).as_secs_f32() * 1000.0;
        stats.extensions = self
            .extensions
            .iter()
            .flat_map(|ext| {
                let name = ext.name();
                ext.stats()
                    .into_iter()
                    .map(move |(key, value)| (format!("{name}.{key}"), value))
            })
            .collect();
        self.stats = stats;
        Ok(())
    }

    fn lit_bind_group(&self) -> wgpu::BindGroup {
        let maps = &self.environment.maps;
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("engine-render-view-lit-bg"),
            layout: &self.layouts.view_lit,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: self.views.buffer(),
                        offset: 0,
                        size: std::num::NonZeroU64::new(self.layouts.view_uniform_size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.lights_buffer.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.cluster_ranges.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.cluster_indices.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.shadow_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&self.shadow_maps.csm_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&self.shadow_maps.atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Sampler(&self.layouts.shadow_compare),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(&maps.irradiance_view),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(&maps.prefiltered_view),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::TextureView(&maps.brdf_lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::TextureView(&self.probes.array_view),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: self.probe_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: wgpu::BindingResource::TextureView(&self.ssao.ao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: wgpu::BindingResource::Sampler(&self.layouts.linear_clamp),
                },
            ],
        })
    }

    fn choose_probe_bake(&self, render_world: &RenderWorld, has_camera: bool) -> Option<ProbeBake> {
        if !has_camera || self.quality.max_probes == 0 {
            return None;
        }
        let key_of = |probe: &crate::extract::ExtractedProbe| {
            probe
                .persistent
                .map(ProbeKey::Persistent)
                .unwrap_or(ProbeKey::Entity(probe.entity))
        };
        render_world
            .probes
            .iter()
            .filter(|probe| {
                let key = key_of(probe);
                let baked = self
                    .probes
                    .slots
                    .iter()
                    .flatten()
                    .any(|slot| slot.key == key && slot.position == probe.position);
                probe.probe.bake || !baked
            })
            .find_map(|probe| {
                let key = key_of(probe);
                let slot = self.probes.slot_for(key, probe.probe.priority)?;
                Some(ProbeBake {
                    entity: probe.entity,
                    slot,
                    data: ProbeSlot {
                        key,
                        position: probe.position,
                        half_extents: probe.probe.half_extents,
                        blend_distance: probe.probe.blend_distance,
                        intensity: probe.probe.intensity,
                        priority: probe.probe.priority,
                    },
                })
            })
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_probe_bake(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bake: &ProbeBake,
        offsets: &ViewOffsets,
        view_lit_bg: &wgpu::BindGroup,
        view_basic_bg: &wgpu::BindGroup,
        instances_bg: &wgpu::BindGroup,
        lists: &[PassDrawList],
        pipelines: &MeshPipelineStore<'_>,
    ) -> u32 {
        let mut draws = 0;
        for face in 0..6u32 {
            let target = self.probes.capture_face_view(face);
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("engine-render-probe-capture"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.probes.capture_depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            let offset = offsets.probe_faces[face as usize];
            rp.set_bind_group(0, view_lit_bg, &[offset]);
            draws += draw_lists(
                &mut rp,
                lists,
                pipelines,
                instances_bg,
                &self.materials,
                &self.cache,
            );
            self.environment
                .draw_skybox(&mut rp, view_basic_bg, offset, 1);
        }
        let mips = self.probes.capture.mip_level_count();
        self.environment.filter.build_mips(
            &self.device,
            encoder,
            &self.probes.capture,
            0,
            mips,
            &self.layouts.linear_clamp,
        );
        self.environment.filter.prefilter(
            &self.device,
            encoder,
            &self.probes.capture_view,
            PROBE_RESOLUTION,
            &self.probes.array,
            bake.slot as u32 * 6,
            PREFILTER_MIPS,
            &self.layouts.linear_clamp,
        );
        draws
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_meshes(
        &mut self,
        render_world: &RenderWorld,
        assets: &AssetServer,
        frustum: &Frustum,
        view_matrix: Mat4,
        cascades: &[Mat4],
        local_matrices: &[Mat4],
        samples: u32,
        encoder: &mut wgpu::CommandEncoder,
        stats: &mut RenderStats,
    ) -> Result<PreparedDraws> {
        let mut instances: Vec<InstanceGpu> = Vec::with_capacity(render_world.meshes.len());
        let mut palettes: Vec<[[f32; 4]; 4]> = render_world.palettes.clone();
        let mut objects: Vec<(DrawObject, Aabb, bool)> = Vec::new();

        for mesh in &render_world.meshes {
            let Some(gpu_mesh) = self.cache.prepare_mesh(&self.device, mesh.mesh, assets) else {
                continue;
            };
            let (local_min, local_max, index_count, has_skin_stream, joint_count) = (
                gpu_mesh.aabb_min,
                gpu_mesh.aabb_max,
                gpu_mesh.index_count,
                gpu_mesh.is_skinned(),
                gpu_mesh.joint_count,
            );
            let material_key = self.materials.prepare(
                &self.device,
                &self.queue,
                &self.layouts,
                &mut self.cache,
                assets,
                mesh.material,
                mesh.texture,
            );
            let Some(material) = self.materials.get(material_key) else {
                continue;
            };
            let (alpha, double_sided) = (material.alpha, material.double_sided);

            let model = Mat4::from_cols_array_2d(&mesh.model);
            let skinned = has_skin_stream
                && mesh
                    .palette
                    .is_some_and(|(_, count)| count >= joint_count && joint_count > 0);
            let mut params = [0u32; 4];
            params[2] = mesh.pick_id;
            if skinned {
                let (offset, count) = mesh.palette.expect("checked above");
                params[0] = offset;
                let previous = self
                    .prev_palettes
                    .get(&mesh.entity)
                    .filter(|prev| prev.len() == count as usize);
                params[1] = match previous {
                    Some(prev) => {
                        let prev_offset = palettes.len() as u32;
                        palettes.extend_from_slice(prev);
                        prev_offset
                    }
                    None => offset,
                };
                params[3] |= crate::uniforms::INSTANCE_FLAG_SKINNED;
                stats.skinned_instances += 1;
            }
            if mesh.receive_shadows {
                params[3] |= INSTANCE_FLAG_RECEIVE_SHADOWS;
            }
            if mesh.selected {
                params[3] |= INSTANCE_FLAG_SELECTED;
            }
            let normal = model.inverse().transpose();
            let object_index = instances.len() as u32;
            instances.push(InstanceGpu {
                model: mesh.model,
                prev_model: self
                    .prev_models
                    .get(&mesh.entity)
                    .copied()
                    .unwrap_or(mesh.model),
                normal: [
                    normal.x_axis.to_array(),
                    normal.y_axis.to_array(),
                    normal.z_axis.to_array(),
                ],
                params,
            });
            let local_aabb = if skinned {
                mesh.skin_bounds.unwrap_or_else(|| {
                    // Unknown pose: inflate the bind-pose bounds.
                    let aabb = Aabb {
                        min: local_min,
                        max: local_max,
                    };
                    Aabb::from_center_extents(
                        aabb.center(),
                        aabb.half_extents() * 2.0 + Vec3::splat(0.5),
                    )
                })
            } else {
                Aabb {
                    min: local_min,
                    max: local_max,
                }
            };
            let world_aabb = local_aabb.transformed(model);
            let depth = -(view_matrix * world_aabb.center().extend(1.0)).z;
            objects.push((
                DrawObject {
                    object_index,
                    mesh: mesh.mesh.id(),
                    material: material_key,
                    skinned,
                    alpha,
                    double_sided,
                    index_count,
                    sort_depth: depth,
                },
                world_aabb,
                mesh.cast_shadows,
            ));
        }

        let mut builder = IndirectionBuilder::new();
        let mut draws = PreparedDraws::default();
        let visible: Vec<&(DrawObject, Aabb, bool)> = objects
            .iter()
            .filter(|(_, aabb, _)| frustum.intersects_aabb(*aabb))
            .collect();
        stats.visible_meshes = visible.len();

        let mut opaque: Vec<DrawObject> = visible
            .iter()
            .filter(|(object, _, _)| object.alpha != AlphaMode::Blend)
            .map(|(object, _, _)| *object)
            .collect();
        let mut transparent: Vec<DrawObject> = visible
            .iter()
            .filter(|(object, _, _)| object.alpha == AlphaMode::Blend)
            .map(|(object, _, _)| *object)
            .collect();
        stats.transparent_meshes = transparent.len();
        let mut all_visible: Vec<DrawObject> =
            visible.iter().map(|(object, _, _)| *object).collect();

        draws.prepass = builder.add_pass(&mut opaque.clone(), MeshPass::Prepass, 1, false);
        draws.main_opaque = builder.add_pass(&mut opaque, MeshPass::Opaque, samples, samples == 1);
        draws.main_transparent =
            builder.add_pass(&mut transparent, MeshPass::Transparent, samples, false);
        draws.picking = builder.add_pass(&mut all_visible.clone(), MeshPass::Picking, 1, false);
        draws.overdraw = builder.add_pass(&mut all_visible, MeshPass::Overdraw, 1, false);

        let casters: Vec<&(DrawObject, Aabb, bool)> = objects
            .iter()
            .filter(|(object, _, cast)| *cast && object.alpha != AlphaMode::Blend)
            .collect();
        let mut caster_count = 0;
        for matrix in cascades {
            let mut list: Vec<DrawObject> = casters
                .iter()
                .filter(|(_, aabb, _)| {
                    aabb_in_clip_volume(
                        *matrix,
                        Vec3::from_array(aabb.min),
                        Vec3::from_array(aabb.max),
                    )
                })
                .map(|(object, _, _)| *object)
                .collect();
            caster_count += list.len();
            draws
                .cascades
                .push(builder.add_pass(&mut list, MeshPass::Shadow, 1, false));
        }
        for matrix in local_matrices {
            let mut list: Vec<DrawObject> = casters
                .iter()
                .filter(|(_, aabb, _)| {
                    aabb_in_clip_volume(
                        *matrix,
                        Vec3::from_array(aabb.min),
                        Vec3::from_array(aabb.max),
                    )
                })
                .map(|(object, _, _)| *object)
                .collect();
            caster_count += list.len();
            draws
                .local_faces
                .push(builder.add_pass(&mut list, MeshPass::Shadow, 1, false));
        }
        stats.shadow_casters = caster_count;

        let mut capture: Vec<DrawObject> = objects
            .iter()
            .filter(|(object, _, _)| object.alpha != AlphaMode::Blend)
            .map(|(object, _, _)| DrawObject {
                double_sided: true,
                ..*object
            })
            .collect();
        draws.probe_capture = builder.add_pass(&mut capture, MeshPass::Opaque, 1, false);

        let mut keys: Vec<MeshPipelineKey> = [
            &draws.prepass,
            &draws.main_opaque,
            &draws.main_transparent,
            &draws.picking,
            &draws.overdraw,
            &draws.probe_capture,
        ]
        .into_iter()
        .chain(draws.cascades.iter())
        .chain(draws.local_faces.iter())
        .flat_map(|lists| {
            lists
                .iter()
                .flat_map(|list| list.batches.iter().map(|b| b.key))
        })
        .collect();
        keys.sort();
        keys.dedup();
        draws.pipeline_keys = keys;

        self.instances.upload(
            &self.device,
            encoder,
            &mut self.belt,
            &instances,
            &palettes,
            builder.bytes(),
        );
        Ok(draws)
    }

    pub fn dump_debug_ppm(&self, path: &std::path::Path) -> Result<()> {
        let w = 64u32;
        let h = 64u32;
        let mut pixels = vec![0u8; (w * h * 4) as usize];
        let counts = &self.last_cluster_counts;
        let max = counts.iter().copied().max().unwrap_or(1).max(1);
        let len = counts.len().max(1);
        for y in 0..h {
            for x in 0..w {
                let idx = ((y * w + x) as usize * len) / (w * h) as usize;
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

    /// Renders one frame into an offscreen texture of the renderer's size
    /// and reads it back as RGBA8 (tests, thumbnails, save-game previews).
    pub fn render_to_image(&mut self, world: &mut World, assets: &AssetServer) -> Result<Vec<u8>> {
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let previous_format = self.target_format;
        self.target_format = format;
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("engine-render-capture"),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let result = self.render_to_view(world, assets, &view);
        self.target_format = previous_format;
        result?;
        read_texture_rgba8(&self.device, &self.queue, &texture)
    }
}

/// Reads an RGBA8 texture back to CPU memory (blocking).
pub fn read_texture_rgba8(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
) -> Result<Vec<u8>> {
    let (width, height) = (texture.width(), texture.height());
    let row = width * 4;
    let padded =
        row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("engine-render-readback"),
        size: (padded * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("engine-render-readback"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));
    let (tx, rx) = std::sync::mpsc::channel();
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
    let _ = device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|_| EngineError::Render("readback channel closed".to_owned()))?
        .map_err(|error| EngineError::Render(format!("readback map failed: {error}")))?;
    let data = buffer.slice(..).get_mapped_range();
    let mut pixels = Vec::with_capacity((row * height) as usize);
    for y in 0..height {
        let start = (y * padded) as usize;
        pixels.extend_from_slice(&data[start..start + row as usize]);
    }
    drop(data);
    buffer.unmap();
    Ok(pixels)
}

fn process_start() -> &'static Instant {
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    START.get_or_init(Instant::now)
}

fn apply_quality_to_graph(graph: &mut RenderGraph, quality: &QualitySettings) {
    graph.set_enabled(PassId("ssao"), quality.ssao_enabled);
    graph.set_enabled(PassId("bloom"), quality.bloom_enabled);
    graph.set_enabled(PassId("taa"), quality.taa_enabled);
    graph.set_enabled(PassId("shadow_local"), quality.local_shadow_slots > 0);
    graph.set_enabled(PassId("shadow_csm"), quality.cascade_count > 0);
    graph.set_enabled(PassId("probe_bake"), quality.max_probes > 0);
    graph.set_enabled(PassId("picking"), true);
}

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
    FrameRenderer::try_new(device, queue, caps, target_format, width, height, config)
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
