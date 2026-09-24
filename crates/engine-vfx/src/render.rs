//! Particle rendering and GPU simulation as a render extension: one
//! `particles` node on the HDR target after transparent geometry.

use std::collections::HashMap;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use bytemuck::{Pod, Zeroable};
use engine_assets::{Assets, Handle, MeshData, TextureData};
use engine_core::{Camera3d, GlobalTransform, PrimaryCamera, Result};
use engine_math::{Affine3A, Mat4, Vec3, Vec4};
use engine_render::extension::{
    EncodeContext, ExtensionNode, ExtractInfo, NodeTarget, PrepareContext, RenderExtension,
};
use engine_render::gpu::assets::{MeshVertexGpu, TextureRole};
use engine_render::uniforms::ViewUniform;
use engine_render::CapabilityTier;

use crate::components::{ActiveBackend, GpuStep, ParticleEffectState, ParticleEmitter};
use crate::effect::{
    BlendMode, EmitterDef, Module, ParticleEffect, RenderMode, Shape, SimulationSpace,
    VelocityDirection,
};
use crate::sim::{pcg, GpuParticle, MeshPoints};
use crate::systems::ParticleSettings;

const COMMON: &str = include_str!("../shaders/particles_common.wgsl");
const SIM: &str = include_str!("../shaders/particles_sim.wgsl");
const SORT: &str = include_str!("../shaders/particles_sort.wgsl");
const DRAW: &str = include_str!("../shaders/particles_draw.wgsl");

pub const NODE: &str = "particles";
const UNIFORM_STRIDE: u64 = 1024;
const SORT_STRIDE: u64 = 256;
const WORKGROUP: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EmitterUniform {
    transform: [[f32; 4]; 4],
    inv_transform: [[f32; 4]; 4],
    counts: [u32; 4],
    shape: [f32; 4],
    shape_extents: [f32; 4],
    lifetime_speed: [f32; 4],
    size_rotation: [f32; 4],
    angular: [f32; 4],
    direction: [f32; 4],
    color: [f32; 4],
    gravity_drag: [f32; 4],
    accel: [f32; 4],
    noise: [f32; 4],
    collision: [f32; 4],
    time: [f32; 4],
    render: [f32; 4],
    flipbook: [f32; 4],
    draw: [u32; 4],
    color_lut: [[f32; 4]; 16],
    size_lut: [[f32; 4]; 4],
    speed_lut: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SceneUniform {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    camera: [f32; 4],
    viewport: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SortUniform {
    k: u32,
    j: u32,
    list: u32,
    capacity: u32,
    camera: [f32; 4],
    flags: [f32; 4],
    transform: [[f32; 4]; 4],
}

/// One emitter instance handed from the world to the renderer.
struct ExtractedEmitter {
    key: (Entity, usize),
    generation: u64,
    backend: ActiveBackend,
    effect: Arc<ParticleEffect>,
    index: usize,
    transform: Affine3A,
    capacity: u32,
    steps: Vec<GpuStep>,
    reset: bool,
    cpu: Vec<GpuParticle>,
    mesh_points: Option<MeshPoints>,
    tint: Vec4,
    seed: u32,
    gravity: Vec3,
}

struct EmitterGpu {
    generation: u64,
    capacity: u32,
    particles: wgpu::Buffer,
    alive: wgpu::Buffer,
    dead: wgpu::Buffer,
    counters: wgpu::Buffer,
    draw_args: wgpu::Buffer,
    mesh_points: wgpu::Buffer,
    mesh_points_len: u32,
    keys: Option<wgpu::Buffer>,
    sort_size: u32,
    /// Source list of the next step (the live list after the last one).
    src: u32,
    spawned: u32,
    needs_reset: bool,
    seen: bool,
}

/// Work recorded in prepare, executed in encode.
struct FrameEmitter {
    key: (Entity, usize),
    backend: ActiveBackend,
    steps: Vec<(u64, u32, u32)>,
    reset: bool,
    reset_offset: u64,
    draw_offset: u64,
    sort_offsets: Vec<u64>,
    blend: BlendMode,
    mesh: Option<Handle<MeshData>>,
    texture: Option<Handle<TextureData>>,
    visible: bool,
    sorted: bool,
}

struct ComputePipelines {
    sim_layout: wgpu::BindGroupLayout,
    sort_layout: wgpu::BindGroupLayout,
    reset: wgpu::ComputePipeline,
    begin: wgpu::ComputePipeline,
    spawn: wgpu::ComputePipeline,
    update: wgpu::ComputePipeline,
    finalize: wgpu::ComputePipeline,
    fill: wgpu::ComputePipeline,
    sort: wgpu::ComputePipeline,
}

struct Pipelines {
    /// `None` on devices without compute (CPU emitters only).
    compute: Option<ComputePipelines>,
    draw_layout: wgpu::BindGroupLayout,
    draw_module: Arc<wgpu::ShaderModule>,
    render: HashMap<(wgpu::TextureFormat, BlendMode, bool), wgpu::RenderPipeline>,
}

#[derive(Default)]
pub struct ParticleRenderer {
    emitters: Vec<ExtractedEmitter>,
    gpu: HashMap<(Entity, usize), EmitterGpu>,
    frame: Vec<FrameEmitter>,
    pipelines: Option<Pipelines>,
    uniforms: Option<(wgpu::Buffer, u64)>,
    sort_uniforms: Option<(wgpu::Buffer, u64)>,
    scene: Option<wgpu::Buffer>,
    dummy: Option<wgpu::Buffer>,
    tier: Option<CapabilityTier>,
    budgets: Option<(u32, u32)>,
    camera: Option<Vec3>,
    stats_cpu: usize,
    stats_gpu_capacity: u64,
    stats_steps: usize,
    stats_drawn: usize,
}

fn mat(m: Mat4) -> [[f32; 4]; 4] {
    m.to_cols_array_2d()
}

fn range(r: crate::effect::Range) -> [f32; 2] {
    [r.min, r.max]
}

fn emitter_uniform(
    extracted: &ExtractedEmitter,
    def: &EmitterDef,
    step: Option<&GpuStep>,
    src: u32,
    seed_base: u32,
    draw: [u32; 4],
    textured: bool,
) -> EmitterUniform {
    let transform = Mat4::from(extracted.transform);
    let local = def.space == SimulationSpace::Local;
    let rotation = extracted.transform.to_scale_rotation_translation().1;
    let (kind, a, b, points, extents) = match &def.shape {
        Shape::Point => (0.0, 0.0, 0.0, 0.0, Vec3::ZERO),
        Shape::Sphere { radius, surface } => (
            1.0,
            *radius,
            if *surface { 1.0 } else { 0.0 },
            0.0,
            Vec3::ZERO,
        ),
        Shape::Cone { angle, radius } => (2.0, *angle, *radius, 0.0, Vec3::ZERO),
        Shape::Box { half_extents } => (3.0, 0.0, 0.0, 0.0, *half_extents),
        Shape::Mesh { .. } => (
            4.0,
            0.0,
            0.0,
            extracted
                .mesh_points
                .as_ref()
                .map_or(0.0, |p| p.len() as f32),
            Vec3::ZERO,
        ),
    };
    let (direction_mode, direction) = match def.init.direction {
        VelocityDirection::Shape => (0.0, Vec3::Y),
        VelocityDirection::Direction(d) => (1.0, d),
    };
    let mut gravity = Vec3::ZERO;
    let mut accel = Vec3::ZERO;
    let mut drag = 0.0;
    let mut noise = [0.0; 4];
    let mut collision = [0.0; 4];
    for module in &def.modules {
        match module {
            Module::Gravity(scale) => gravity += extracted.gravity * *scale,
            Module::Acceleration(a) => accel += *a,
            Module::Drag(k) => drag += k,
            Module::CurlNoise {
                strength,
                frequency,
                scroll_speed,
            } => noise = [*strength, *frequency, *scroll_speed, 1.0],
            Module::Collision {
                bounce,
                friction,
                kill,
            } => collision = [1.0, *bounce, *friction, if *kill { 1.0 } else { 0.0 }],
            _ => {}
        }
    }
    if local {
        gravity = rotation.inverse() * gravity;
    }
    let has_speed = def
        .modules
        .iter()
        .any(|m| matches!(m, Module::SpeedOverLife(_)));
    let color_lut = def.color_lut().map(|c| c.to_array());
    let pack = |values: [f32; 16]| -> [[f32; 4]; 4] {
        std::array::from_fn(|i| {
            [
                values[i * 4],
                values[i * 4 + 1],
                values[i * 4 + 2],
                values[i * 4 + 3],
            ]
        })
    };
    let (mode, stretch) = match &def.render.mode {
        RenderMode::Billboard => (0.0, 0.0),
        RenderMode::Stretched { length_scale } => (1.0, *length_scale),
        RenderMode::Mesh { .. } => (2.0, 0.0),
    };
    let blend = match def.render.blend {
        BlendMode::Additive => 0.0,
        BlendMode::Alpha => 1.0,
        BlendMode::Premultiplied => 2.0,
    };
    let flipbook = def.render.flipbook.map_or([1.0, 1.0, 0.0, 0.0], |f| {
        [f.columns.max(1) as f32, f.rows.max(1) as f32, f.fps, 1.0]
    });
    let [lmin, lmax] = range(def.init.lifetime);
    let [smin, smax] = range(def.init.speed);
    let [zmin, zmax] = range(def.init.size);
    let [rmin, rmax] = range(def.init.rotation);
    let [amin, amax] = range(def.init.angular_velocity);
    EmitterUniform {
        transform: mat(transform),
        inv_transform: mat(transform.inverse()),
        counts: [
            step.map_or(0, |s| s.spawn),
            extracted.capacity,
            seed_base,
            src,
        ],
        shape: [kind, a, b, points],
        shape_extents: extents.extend(0.0).to_array(),
        lifetime_speed: [lmin, lmax, smin, smax],
        size_rotation: [zmin, zmax, rmin, rmax],
        angular: [amin, amax, direction_mode, 0.0],
        direction: direction.extend(0.0).to_array(),
        color: (def.init.color * extracted.tint).to_array(),
        gravity_drag: gravity.extend(drag).to_array(),
        accel: accel.extend(if has_speed { 1.0 } else { 0.0 }).to_array(),
        noise,
        collision,
        time: [
            step.map_or(0.0, |s| s.dt),
            step.map_or(0.0, |s| s.time),
            if local { 1.0 } else { 0.0 },
            def.render.soft_distance.max(0.0),
        ],
        render: [mode, stretch, blend, if textured { 1.0 } else { 0.0 }],
        flipbook,
        draw,
        color_lut,
        size_lut: pack(def.size_lut()),
        speed_lut: pack(def.speed_lut()),
    }
}

fn storage(
    device: &wgpu::Device,
    label: &str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(16),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | usage,
        mapped_at_creation: false,
    })
}

fn layout_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    ty: wgpu::BindingType,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty,
        count: None,
    }
}

fn buffer_type(kind: wgpu::BufferBindingType, dynamic: bool) -> wgpu::BindingType {
    wgpu::BindingType::Buffer {
        ty: kind,
        has_dynamic_offset: dynamic,
        min_binding_size: None,
    }
}

fn depth_type() -> wgpu::BindingType {
    wgpu::BindingType::Texture {
        sample_type: wgpu::TextureSampleType::Depth,
        view_dimension: wgpu::TextureViewDimension::D2,
        multisampled: false,
    }
}

fn rw() -> wgpu::BindingType {
    buffer_type(wgpu::BufferBindingType::Storage { read_only: false }, false)
}

fn ro() -> wgpu::BindingType {
    buffer_type(wgpu::BufferBindingType::Storage { read_only: true }, false)
}

impl ComputePipelines {
    fn new(device: &wgpu::Device, sim: &wgpu::ShaderModule, sort: &wgpu::ShaderModule) -> Self {
        let c = wgpu::ShaderStages::COMPUTE;
        let uniform_dynamic = buffer_type(wgpu::BufferBindingType::Uniform, true);
        let sim_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vfx-sim-layout"),
            entries: &[
                layout_entry(0, c, uniform_dynamic),
                layout_entry(1, c, rw()),
                layout_entry(2, c, rw()),
                layout_entry(3, c, rw()),
                layout_entry(4, c, rw()),
                layout_entry(5, c, rw()),
                layout_entry(6, c, ro()),
                layout_entry(7, c, buffer_type(wgpu::BufferBindingType::Uniform, false)),
                layout_entry(8, c, depth_type()),
            ],
        });
        let sort_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vfx-sort-layout"),
            entries: &[
                layout_entry(0, c, uniform_dynamic),
                layout_entry(1, c, ro()),
                layout_entry(2, c, ro()),
                layout_entry(3, c, ro()),
                layout_entry(4, c, rw()),
            ],
        });
        let compute = |layout: &wgpu::BindGroupLayout, module: &wgpu::ShaderModule, entry: &str| {
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("vfx-compute-layout"),
                bind_group_layouts: &[layout],
                push_constant_ranges: &[],
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self {
            reset: compute(&sim_layout, sim, "reset"),
            begin: compute(&sim_layout, sim, "begin"),
            spawn: compute(&sim_layout, sim, "spawn"),
            update: compute(&sim_layout, sim, "update"),
            finalize: compute(&sim_layout, sim, "finalize"),
            fill: compute(&sort_layout, sort, "fill"),
            sort: compute(&sort_layout, sort, "sort_step"),
            sim_layout,
            sort_layout,
        }
    }
}

impl Pipelines {
    fn new(
        device: &wgpu::Device,
        compute: Option<ComputePipelines>,
        draw: Arc<wgpu::ShaderModule>,
    ) -> Self {
        let uniform_dynamic = buffer_type(wgpu::BufferBindingType::Uniform, true);
        let v = wgpu::ShaderStages::VERTEX;
        let vf = wgpu::ShaderStages::VERTEX_FRAGMENT;
        let f = wgpu::ShaderStages::FRAGMENT;
        let draw_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vfx-draw-layout"),
            entries: &[
                layout_entry(0, vf, uniform_dynamic),
                layout_entry(1, v, ro()),
                layout_entry(2, v, ro()),
                layout_entry(3, v, ro()),
                layout_entry(4, f, depth_type()),
                layout_entry(
                    5,
                    f,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                layout_entry(
                    6,
                    f,
                    wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                ),
            ],
        });
        Self {
            compute,
            draw_layout,
            draw_module: draw,
            render: HashMap::new(),
        }
    }

    fn render_pipeline(
        &mut self,
        device: &wgpu::Device,
        view_layout: &wgpu::BindGroupLayout,
        format: wgpu::TextureFormat,
        blend: BlendMode,
        mesh: bool,
    ) -> &wgpu::RenderPipeline {
        let draw_layout = &self.draw_layout;
        let module = &self.draw_module;
        self.render.entry((format, blend, mesh)).or_insert_with(|| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("vfx-draw-pipeline-layout"),
                bind_group_layouts: &[view_layout, draw_layout],
                push_constant_ranges: &[],
            });
            let state = match blend {
                BlendMode::Additive => wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent::OVER,
                },
                BlendMode::Alpha => wgpu::BlendState::ALPHA_BLENDING,
                BlendMode::Premultiplied => wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING,
            };
            let buffers = [MeshVertexGpu::layout()];
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("vfx-draw"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some(if mesh { "vs_mesh" } else { "vs_quad" }),
                    buffers: if mesh { &buffers } else { &[] },
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(state),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview: None,
                cache: None,
            })
        })
    }
}

fn grow(
    buffer: &mut Option<(wgpu::Buffer, u64)>,
    device: &wgpu::Device,
    label: &str,
    size: u64,
) -> wgpu::Buffer {
    let needed = size.max(UNIFORM_STRIDE);
    if buffer
        .as_ref()
        .is_none_or(|(_, capacity)| *capacity < needed)
    {
        let capacity = needed.next_power_of_two();
        *buffer = Some((
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: capacity,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            capacity,
        ));
    }
    buffer.as_ref().expect("allocated").0.clone()
}

fn uniform_binding(binding: u32, buffer: &wgpu::Buffer, size: u64) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer,
            offset: 0,
            size: wgpu::BufferSize::new(size),
        }),
    }
}

fn aabb_visible(view_proj: &Mat4, min: Vec3, max: Vec3) -> bool {
    let corners = [
        Vec3::new(min.x, min.y, min.z),
        Vec3::new(max.x, min.y, min.z),
        Vec3::new(min.x, max.y, min.z),
        Vec3::new(max.x, max.y, min.z),
        Vec3::new(min.x, min.y, max.z),
        Vec3::new(max.x, min.y, max.z),
        Vec3::new(min.x, max.y, max.z),
        Vec3::new(max.x, max.y, max.z),
    ];
    let clips: Vec<Vec4> = corners.iter().map(|c| *view_proj * c.extend(1.0)).collect();
    let outside = |test: &dyn Fn(&Vec4) -> bool| clips.iter().all(test);
    !(outside(&|c| c.x < -c.w)
        || outside(&|c| c.x > c.w)
        || outside(&|c| c.y < -c.w)
        || outside(&|c| c.y > c.w)
        || outside(&|c| c.z > c.w)
        || outside(&|c| c.w <= 0.0))
}

impl RenderExtension for ParticleRenderer {
    fn name(&self) -> &'static str {
        "engine::vfx::particles"
    }

    fn nodes(&self) -> Vec<ExtensionNode> {
        vec![ExtensionNode {
            name: NODE,
            target: NodeTarget::Hdr,
            after: vec!["transparent_3d"],
            // Before TAA, bloom and tonemapping (which follow sprites).
            before: vec!["sprites_2d"],
        }]
    }

    fn extract(&mut self, world: &mut World, _info: &ExtractInfo) {
        if let (Some(tier), Some(mut settings)) =
            (self.tier, world.get_resource_mut::<ParticleSettings>())
        {
            let gpu = tier == CapabilityTier::Tier1;
            let (max_gpu, max_cpu) = self
                .budgets
                .unwrap_or((settings.max_gpu_particles, settings.max_cpu_particles));
            if settings.gpu_compute != gpu
                || settings.max_gpu_particles != max_gpu
                || settings.max_cpu_particles != max_cpu
            {
                settings.gpu_compute = gpu;
                settings.max_gpu_particles = max_gpu;
                settings.max_cpu_particles = max_cpu;
            }
        }
        let gravity = world
            .get_resource::<engine_physics::PhysicsWorld3D>()
            .map(|p| Vec3::new(p.gravity.x, p.gravity.y, p.gravity.z))
            .unwrap_or(Vec3::new(0.0, -9.81, 0.0));
        let mut cameras =
            world.query_filtered::<(&GlobalTransform, Option<&PrimaryCamera>), With<Camera3d>>();
        self.camera = cameras
            .iter(world)
            .max_by_key(|(_, primary)| primary.is_some())
            .map(|(global, _)| Vec3::from(global.0.translation));
        let camera = self.camera;

        self.emitters.clear();
        let mut query = world.query::<(
            Entity,
            &ParticleEmitter,
            &mut ParticleEffectState,
            Option<&GlobalTransform>,
        )>();
        for (entity, emitter, mut state, global) in query.iter_mut(world) {
            let Some(effect) = state.effect.clone() else {
                continue;
            };
            let generation = state.generation;
            let transform = global.map_or(Affine3A::IDENTITY, |g| g.0);
            for (index, emitter_state) in state.emitters.iter_mut().enumerate() {
                let def = &effect.emitters[index];
                let cpu = match emitter_state.backend {
                    ActiveBackend::Cpu => {
                        let sort = def.render.sort && def.render.blend != BlendMode::Additive;
                        emitter_state.cpu.gpu_records(
                            camera.filter(|_| sort),
                            &transform,
                            def.space == SimulationSpace::Local,
                        )
                    }
                    ActiveBackend::Gpu => Vec::new(),
                };
                let reset = std::mem::take(&mut emitter_state.gpu_reset);
                self.emitters.push(ExtractedEmitter {
                    key: (entity, index),
                    generation,
                    backend: emitter_state.backend,
                    effect: effect.clone(),
                    index,
                    transform,
                    capacity: emitter_state.capacity,
                    steps: std::mem::take(&mut emitter_state.pending_gpu),
                    reset,
                    cpu,
                    mesh_points: emitter_state.mesh_points.clone(),
                    tint: Vec4::from_array(emitter.tint),
                    seed: emitter.seed ^ effect.seed ^ (index as u32).wrapping_mul(0x85EB_CA6B),
                    gravity,
                });
            }
        }
    }

    fn prepare(&mut self, ctx: &mut PrepareContext<'_>) -> Result<()> {
        self.tier = Some(ctx.tier);
        self.budgets = Some((ctx.quality.max_gpu_particles, ctx.quality.max_cpu_particles));
        let device = ctx.device;
        self.frame.clear();
        self.stats_cpu = 0;
        self.stats_gpu_capacity = 0;
        self.stats_steps = 0;
        if self.emitters.is_empty() {
            self.gpu.clear();
            return Ok(());
        }

        if self.pipelines.is_none() {
            for (path, source) in [
                ("vfx/particles_common.wgsl", COMMON),
                ("vfx/particles_sim.wgsl", SIM),
                ("vfx/particles_sort.wgsl", SORT),
                ("vfx/particles_draw.wgsl", DRAW),
            ] {
                ctx.shaders.add_source(path, source);
            }
            let draw = ctx.shaders.module(device, "vfx/particles_draw.wgsl", &[])?;
            let compute = if ctx.tier == CapabilityTier::Tier1 {
                let sim = ctx.shaders.module(device, "vfx/particles_sim.wgsl", &[])?;
                let sort = ctx.shaders.module(device, "vfx/particles_sort.wgsl", &[])?;
                Some(ComputePipelines::new(device, &sim, &sort))
            } else {
                None
            };
            self.pipelines = Some(Pipelines::new(device, compute, draw));
        }
        if self.dummy.is_none() {
            self.dummy = Some(storage(
                device,
                "vfx-dummy",
                64,
                wgpu::BufferUsages::empty(),
            ));
        }
        let scene = self.scene.get_or_insert_with(|| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vfx-scene"),
                size: size_of::<SceneUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        let view_proj = ctx.view.map(|v| Mat4::from_cols_array_2d(&v.view_proj));
        let scene_uniform = SceneUniform {
            view_proj: ctx.view.map_or(mat(Mat4::IDENTITY), |v| v.view_proj),
            inv_view_proj: ctx.view.map_or(mat(Mat4::IDENTITY), |v| v.inv_view_proj),
            camera: ctx.view.map_or([0.0; 4], |v: &ViewUniform| {
                [
                    v.camera_position[0],
                    v.camera_position[1],
                    v.camera_position[2],
                    1.0,
                ]
            }),
            viewport: [
                ctx.width as f32,
                ctx.height as f32,
                1.0 / ctx.width.max(1) as f32,
                1.0 / ctx.height.max(1) as f32,
            ],
        };
        ctx.queue
            .write_buffer(scene, 0, bytemuck::bytes_of(&scene_uniform));

        for gpu in self.gpu.values_mut() {
            gpu.seen = false;
        }
        let mut uniform_bytes: Vec<u8> = Vec::new();
        let mut sort_bytes: Vec<u8> = Vec::new();
        let push = |bytes: &mut Vec<u8>, data: &[u8], stride: u64| -> u64 {
            let offset = bytes.len() as u64;
            bytes.extend_from_slice(data);
            bytes.resize((offset + stride) as usize, 0);
            offset
        };
        let assets: &Assets = ctx.server.assets();

        for extracted in &self.emitters {
            let def = &extracted.effect.emitters[extracted.index];
            let capacity = match extracted.backend {
                ActiveBackend::Gpu => extracted.capacity,
                ActiveBackend::Cpu => {
                    (extracted.cpu.len() as u32).max(extracted.capacity.min(1024))
                }
            }
            .max(1);
            let sorted_gpu = extracted.backend == ActiveBackend::Gpu
                && def.render.sort
                && def.render.blend != BlendMode::Additive;
            let stale = self.gpu.get(&extracted.key).is_none_or(|g| {
                g.generation != extracted.generation
                    || g.capacity < capacity
                    || (extracted.backend == ActiveBackend::Gpu && g.capacity != capacity)
                    || (sorted_gpu && g.keys.is_none())
            });
            if stale {
                let sort_size = if sorted_gpu {
                    capacity.next_power_of_two()
                } else {
                    0
                };
                let points: Vec<[f32; 4]> = extracted
                    .mesh_points
                    .as_ref()
                    .map(|p| {
                        p.iter()
                            .flat_map(|(pos, n)| {
                                [pos.extend(0.0).to_array(), n.extend(0.0).to_array()]
                            })
                            .collect()
                    })
                    .unwrap_or_else(|| vec![[0.0; 4]; 2]);
                let mesh_points = storage(
                    device,
                    "vfx-mesh-points",
                    (points.len() * 16) as u64,
                    wgpu::BufferUsages::empty(),
                );
                ctx.queue
                    .write_buffer(&mesh_points, 0, bytemuck::cast_slice(&points));
                self.gpu.insert(
                    extracted.key,
                    EmitterGpu {
                        generation: extracted.generation,
                        capacity,
                        particles: storage(
                            device,
                            "vfx-particles",
                            capacity as u64 * 64,
                            wgpu::BufferUsages::empty(),
                        ),
                        alive: storage(
                            device,
                            "vfx-alive",
                            capacity as u64 * 8,
                            wgpu::BufferUsages::empty(),
                        ),
                        dead: storage(
                            device,
                            "vfx-dead",
                            capacity as u64 * 4,
                            wgpu::BufferUsages::empty(),
                        ),
                        counters: storage(device, "vfx-counters", 16, wgpu::BufferUsages::empty()),
                        draw_args: storage(
                            device,
                            "vfx-draw-args",
                            32,
                            wgpu::BufferUsages::INDIRECT,
                        ),
                        mesh_points,
                        mesh_points_len: (points.len() / 2) as u32,
                        keys: sorted_gpu.then(|| {
                            storage(
                                device,
                                "vfx-keys",
                                sort_size as u64 * 8,
                                wgpu::BufferUsages::empty(),
                            )
                        }),
                        sort_size,
                        src: 0,
                        spawned: 0,
                        needs_reset: true,
                        seen: true,
                    },
                );
            }
            let gpu = self.gpu.get_mut(&extracted.key).expect("inserted above");
            gpu.seen = true;
            let _ = gpu.mesh_points_len;

            let mesh = match &def.render.mode {
                RenderMode::Mesh { mesh } => Some(assets.request::<MeshData>(mesh)),
                _ => None,
            };
            let index_count = match mesh {
                Some(handle) => ctx
                    .cache
                    .prepare_typed_mesh(device, handle, assets)
                    .map_or(0, |m| m.index_count),
                None => 6,
            };
            let texture = def
                .render
                .texture
                .as_ref()
                .map(|t| assets.request::<TextureData>(t));
            let textured = texture.is_some_and(|handle| {
                ctx.cache
                    .prepare_typed_texture(device, ctx.queue, handle, assets, true)
                    .is_some()
            });
            let draw_info = [
                index_count,
                u32::from(mesh.is_some()),
                u32::from(sorted_gpu),
                gpu.sort_size,
            ];

            let mut frame = FrameEmitter {
                key: extracted.key,
                backend: extracted.backend,
                steps: Vec::new(),
                reset: false,
                reset_offset: 0,
                draw_offset: 0,
                sort_offsets: Vec::new(),
                blend: def.render.blend,
                mesh,
                texture,
                visible: true,
                sorted: sorted_gpu,
            };
            match extracted.backend {
                ActiveBackend::Gpu => {
                    self.stats_gpu_capacity += capacity as u64;
                    if extracted.reset || gpu.needs_reset {
                        gpu.needs_reset = false;
                        gpu.src = 0;
                        frame.reset = true;
                        let uniform =
                            emitter_uniform(extracted, def, None, 0, 0, draw_info, textured);
                        frame.reset_offset = push(
                            &mut uniform_bytes,
                            bytemuck::bytes_of(&uniform),
                            UNIFORM_STRIDE,
                        );
                    }
                    for step in &extracted.steps {
                        let seed_base = pcg(extracted.seed ^ gpu.spawned.wrapping_mul(0x9E37_79B9));
                        gpu.spawned = gpu.spawned.wrapping_add(step.spawn);
                        let uniform = emitter_uniform(
                            extracted,
                            def,
                            Some(step),
                            gpu.src,
                            seed_base,
                            draw_info,
                            textured,
                        );
                        let offset = push(
                            &mut uniform_bytes,
                            bytemuck::bytes_of(&uniform),
                            UNIFORM_STRIDE,
                        );
                        frame.steps.push((offset, step.spawn, gpu.src));
                        gpu.src = 1 - gpu.src;
                        self.stats_steps += 1;
                    }
                    let uniform =
                        emitter_uniform(extracted, def, None, 1 - gpu.src, 0, draw_info, textured);
                    frame.draw_offset = push(
                        &mut uniform_bytes,
                        bytemuck::bytes_of(&uniform),
                        UNIFORM_STRIDE,
                    );
                    if sorted_gpu {
                        let camera = scene_uniform.camera;
                        let base = SortUniform {
                            k: 0,
                            j: 0,
                            list: gpu.src,
                            capacity,
                            camera,
                            flags: [
                                if def.space == SimulationSpace::Local {
                                    1.0
                                } else {
                                    0.0
                                },
                                0.0,
                                0.0,
                                0.0,
                            ],
                            transform: mat(Mat4::from(extracted.transform)),
                        };
                        frame.sort_offsets.push(push(
                            &mut sort_bytes,
                            bytemuck::bytes_of(&base),
                            SORT_STRIDE,
                        ));
                        let mut k = 2;
                        while k <= gpu.sort_size {
                            let mut j = k / 2;
                            while j > 0 {
                                let step = SortUniform { k, j, ..base };
                                frame.sort_offsets.push(push(
                                    &mut sort_bytes,
                                    bytemuck::bytes_of(&step),
                                    SORT_STRIDE,
                                ));
                                j /= 2;
                            }
                            k *= 2;
                        }
                    }
                }
                ActiveBackend::Cpu => {
                    let count = extracted.cpu.len() as u32;
                    self.stats_cpu += count as usize;
                    if count > 0 {
                        ctx.queue.write_buffer(
                            &gpu.particles,
                            0,
                            bytemuck::cast_slice(&extracted.cpu),
                        );
                        let order: Vec<u32> = (0..count).collect();
                        ctx.queue
                            .write_buffer(&gpu.alive, 0, bytemuck::cast_slice(&order));
                    }
                    let args: [u32; 8] = if mesh.is_some() {
                        [index_count, count, 0, 0, 0, count, 0, 0]
                    } else {
                        [6, count, 0, 0, 0, count, 0, 0]
                    };
                    ctx.queue
                        .write_buffer(&gpu.draw_args, 0, bytemuck::cast_slice(&args));
                    // Draw reads list `1 - counts.w` = 0.
                    let uniform = emitter_uniform(
                        extracted,
                        def,
                        None,
                        1,
                        0,
                        [index_count, u32::from(mesh.is_some()), 0, 0],
                        textured,
                    );
                    frame.draw_offset = push(
                        &mut uniform_bytes,
                        bytemuck::bytes_of(&uniform),
                        UNIFORM_STRIDE,
                    );
                    frame.visible = count > 0;
                }
            }
            if let (Some(view_proj), Some((min, max))) = (view_proj, extracted.effect.bounds) {
                let corners = [min, max];
                let world_min = corners.iter().fold(Vec3::splat(f32::MAX), |acc, c| {
                    acc.min(extracted.transform.transform_point3(*c))
                });
                let world_max = corners.iter().fold(Vec3::splat(f32::MIN), |acc, c| {
                    acc.max(extracted.transform.transform_point3(*c))
                });
                if !aabb_visible(&view_proj, world_min, world_max) {
                    frame.visible = false;
                }
            }
            if mesh.is_some() && index_count == 0 {
                frame.visible = false;
            }
            self.frame.push(frame);
        }
        self.gpu.retain(|_, gpu| gpu.seen);

        let buffer = grow(
            &mut self.uniforms,
            device,
            "vfx-uniforms",
            uniform_bytes.len() as u64,
        );
        if !uniform_bytes.is_empty() {
            ctx.queue.write_buffer(&buffer, 0, &uniform_bytes);
        }
        let sort_buffer = grow(
            &mut self.sort_uniforms,
            device,
            "vfx-sort-uniforms",
            sort_bytes.len() as u64,
        );
        if !sort_bytes.is_empty() {
            ctx.queue.write_buffer(&sort_buffer, 0, &sort_bytes);
        }
        Ok(())
    }

    fn encode(&mut self, node: &'static str, ctx: &mut EncodeContext<'_>) -> Result<()> {
        if node != NODE || self.frame.is_empty() {
            self.stats_drawn = 0;
            return Ok(());
        }
        let (Some(pipelines), Some((uniforms, _)), Some(scene), Some(dummy)) = (
            self.pipelines.as_mut(),
            self.uniforms.as_ref(),
            self.scene.as_ref(),
            self.dummy.as_ref(),
        ) else {
            return Ok(());
        };
        let Some((view_group, view_offset)) = ctx.view_bind_group else {
            return Ok(());
        };
        let device = ctx.device;

        // Simulation.
        if let Some(compute) = pipelines.compute.as_ref() {
            let mut pass = ctx
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("vfx-simulate"),
                    timestamp_writes: None,
                });
            for frame in &self.frame {
                if frame.backend != ActiveBackend::Gpu {
                    continue;
                }
                let Some(gpu) = self.gpu.get(&frame.key) else {
                    continue;
                };
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("vfx-sim-group"),
                    layout: &compute.sim_layout,
                    entries: &[
                        uniform_binding(0, uniforms, size_of::<EmitterUniform>() as u64),
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: gpu.particles.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: gpu.alive.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: gpu.dead.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: gpu.counters.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: gpu.draw_args.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: gpu.mesh_points.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: scene.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 8,
                            resource: wgpu::BindingResource::TextureView(ctx.depth),
                        },
                    ],
                });
                let groups = |count: u32| count.div_ceil(WORKGROUP).max(1);
                if frame.reset {
                    pass.set_bind_group(0, &group, &[frame.reset_offset as u32]);
                    pass.set_pipeline(&compute.reset);
                    pass.dispatch_workgroups(groups(gpu.capacity), 1, 1);
                }
                for (offset, spawn, _) in &frame.steps {
                    pass.set_bind_group(0, &group, &[*offset as u32]);
                    pass.set_pipeline(&compute.begin);
                    pass.dispatch_workgroups(1, 1, 1);
                    if *spawn > 0 {
                        pass.set_pipeline(&compute.spawn);
                        pass.dispatch_workgroups(groups(*spawn), 1, 1);
                    }
                    pass.set_pipeline(&compute.update);
                    pass.dispatch_workgroups(groups(gpu.capacity), 1, 1);
                    pass.set_pipeline(&compute.finalize);
                    pass.dispatch_workgroups(1, 1, 1);
                }
                if frame.sorted {
                    let (Some(keys), Some((sort_uniforms, _))) =
                        (gpu.keys.as_ref(), self.sort_uniforms.as_ref())
                    else {
                        continue;
                    };
                    let sort_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("vfx-sort-group"),
                        layout: &compute.sort_layout,
                        entries: &[
                            uniform_binding(0, sort_uniforms, size_of::<SortUniform>() as u64),
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: gpu.particles.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: gpu.alive.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: gpu.draw_args.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: keys.as_entire_binding(),
                            },
                        ],
                    });
                    let mut offsets = frame.sort_offsets.iter();
                    if let Some(first) = offsets.next() {
                        pass.set_bind_group(0, &sort_group, &[*first as u32]);
                        pass.set_pipeline(&compute.fill);
                        pass.dispatch_workgroups(groups(gpu.sort_size), 1, 1);
                    }
                    pass.set_pipeline(&compute.sort);
                    for offset in offsets {
                        pass.set_bind_group(0, &sort_group, &[*offset as u32]);
                        pass.dispatch_workgroups(groups(gpu.sort_size), 1, 1);
                    }
                }
            }
        }

        // Drawing.
        let sampler = &ctx.layouts.linear_clamp;
        let white = &ctx.cache.fallback_for(TextureRole::Color).view;
        let mut drawn = 0;
        let mut draws = Vec::new();
        for frame in &self.frame {
            if !frame.visible {
                continue;
            }
            let Some(gpu) = self.gpu.get(&frame.key) else {
                continue;
            };
            let texture_view = frame
                .texture
                .and_then(|handle| ctx.cache.texture(handle.id(), true))
                .map(|t| &t.view)
                .unwrap_or(white);
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("vfx-draw-group"),
                layout: &pipelines.draw_layout,
                entries: &[
                    uniform_binding(0, uniforms, size_of::<EmitterUniform>() as u64),
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: gpu.particles.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: gpu.alive.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: gpu.keys.as_ref().unwrap_or(dummy).as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(ctx.depth),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            });
            pipelines.render_pipeline(
                device,
                &ctx.layouts.view_basic,
                ctx.color_format,
                frame.blend,
                frame.mesh.is_some(),
            );
            draws.push((frame, group));
        }
        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("vfx-draw"),
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
        pass.set_bind_group(0, view_group, &[view_offset]);
        for (frame, group) in &draws {
            let Some(gpu) = self.gpu.get(&frame.key) else {
                continue;
            };
            let pipeline =
                &pipelines.render[&(ctx.color_format, frame.blend, frame.mesh.is_some())];
            pass.set_pipeline(pipeline);
            pass.set_bind_group(1, group, &[frame.draw_offset as u32]);
            match frame.mesh.and_then(|handle| ctx.cache.mesh(handle.id())) {
                Some(mesh) => {
                    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed_indirect(&gpu.draw_args, 0);
                }
                None if frame.mesh.is_none() => pass.draw_indirect(&gpu.draw_args, 0),
                None => continue,
            }
            drawn += 1;
        }
        self.stats_drawn = drawn;
        Ok(())
    }

    fn stats(&self) -> Vec<(&'static str, f64)> {
        vec![
            ("particle_emitters", self.frame.len() as f64),
            ("particles_cpu", self.stats_cpu as f64),
            ("particles_gpu_capacity", self.stats_gpu_capacity as f64),
            ("particle_gpu_steps", self.stats_steps as f64),
            ("particle_draws", self.stats_drawn as f64),
        ]
    }
}
