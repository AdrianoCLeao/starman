//! Mesh drawing shared by every mesh pass: pipeline variants, material
//! bind groups, the per-frame instance buffer, per-pass indirection lists
//! and instanced draw batches.

use std::collections::HashMap;
use std::mem::size_of;
use std::sync::Arc;

use engine_assets::{
    AssetId, AssetServer, Assets, Handle, MaterialData, MaterialHandle, TextureData, TextureHandle,
};
use engine_core::Result;
use wgpu::util::DeviceExt;

use crate::gpu::assets::{GpuAssetCache, MeshVertexGpu, SkinVertexGpu, TextureRole};
use crate::gpu::{GrowableBuffer, StagingBelt};
use crate::layouts::{SharedLayouts, DEPTH_FORMAT, PICK_FORMAT, VELOCITY_FORMAT};
use crate::shader::ShaderLibrary;
use crate::uniforms::{InstanceGpu, MaterialUniformGpu};

/// Which mesh pass a pipeline serves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MeshPass {
    Prepass,
    Shadow,
    Opaque,
    Transparent,
    Picking,
    Overdraw,
}

impl MeshPass {
    fn define(self) -> &'static str {
        match self {
            Self::Prepass => "PREPASS",
            Self::Shadow => "SHADOW",
            Self::Opaque | Self::Transparent => "LIT",
            Self::Picking => "PICKING",
            Self::Overdraw => "OVERDRAW",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MeshPipelineKey {
    pub pass: MeshPass,
    pub skinned: bool,
    pub alpha_mask: bool,
    pub double_sided: bool,
    pub samples: u32,
    /// Opaque passes after a depth pre-pass test `Equal`-ish (LessEqual,
    /// no depth write).
    pub depth_prepassed: bool,
}

impl MeshPipelineKey {
    pub fn defines(&self) -> Vec<&'static str> {
        let mut defines = vec![self.pass.define()];
        if self.skinned {
            defines.push("SKINNED");
        }
        if self.alpha_mask {
            defines.push("ALPHA_MASK");
        }
        defines
    }
}

pub struct MeshPipelines {
    pipelines: HashMap<MeshPipelineKey, wgpu::RenderPipeline>,
    lit_layout: wgpu::PipelineLayout,
    basic_layout: wgpu::PipelineLayout,
    hdr_format: wgpu::TextureFormat,
}

impl MeshPipelines {
    pub fn new(
        device: &wgpu::Device,
        layouts: &SharedLayouts,
        hdr_format: wgpu::TextureFormat,
    ) -> Self {
        let lit_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("engine-render-mesh-lit-layout"),
            bind_group_layouts: &[&layouts.view_lit, &layouts.instances, &layouts.material],
            push_constant_ranges: &[],
        });
        let basic_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("engine-render-mesh-basic-layout"),
            bind_group_layouts: &[&layouts.view_basic, &layouts.instances, &layouts.material],
            push_constant_ranges: &[],
        });
        Self {
            pipelines: HashMap::new(),
            lit_layout,
            basic_layout,
            hdr_format,
        }
    }

    pub fn get(
        &mut self,
        device: &wgpu::Device,
        shaders: &mut ShaderLibrary,
        key: MeshPipelineKey,
    ) -> Result<&wgpu::RenderPipeline> {
        if !self.pipelines.contains_key(&key) {
            let pipeline = self.create(device, shaders, key)?;
            self.pipelines.insert(key, pipeline);
        }
        Ok(self.pipelines.get(&key).expect("inserted above"))
    }

    /// An already compiled pipeline.
    pub fn lookup(&self, key: &MeshPipelineKey) -> Option<&wgpu::RenderPipeline> {
        self.pipelines.get(key)
    }

    pub fn variant_count(&self) -> usize {
        self.pipelines.len()
    }

    fn create(
        &self,
        device: &wgpu::Device,
        shaders: &mut ShaderLibrary,
        key: MeshPipelineKey,
    ) -> Result<wgpu::RenderPipeline> {
        let defines = key.defines();
        let module: Arc<wgpu::ShaderModule> = shaders.module(device, "mesh.wgsl", &defines)?;
        let mut buffers = vec![MeshVertexGpu::layout()];
        if key.skinned {
            buffers.push(SkinVertexGpu::layout());
        }
        let lit = matches!(key.pass, MeshPass::Opaque | MeshPass::Transparent);
        let layout = if lit {
            &self.lit_layout
        } else {
            &self.basic_layout
        };
        let cull_mode = if key.double_sided {
            None
        } else {
            Some(wgpu::Face::Back)
        };
        let (targets, depth): (Vec<Option<wgpu::ColorTargetState>>, wgpu::DepthStencilState) =
            match key.pass {
                MeshPass::Prepass => (
                    vec![Some(wgpu::ColorTargetState {
                        format: VELOCITY_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    depth_state(true, wgpu::CompareFunction::Less, 0),
                ),
                MeshPass::Shadow => (
                    vec![],
                    depth_state(true, wgpu::CompareFunction::LessEqual, 2),
                ),
                MeshPass::Opaque => (
                    vec![Some(wgpu::ColorTargetState {
                        format: self.hdr_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    if key.depth_prepassed {
                        depth_state(false, wgpu::CompareFunction::LessEqual, 0)
                    } else {
                        depth_state(true, wgpu::CompareFunction::Less, 0)
                    },
                ),
                MeshPass::Transparent => (
                    vec![Some(wgpu::ColorTargetState {
                        format: self.hdr_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    depth_state(false, wgpu::CompareFunction::Less, 0),
                ),
                MeshPass::Picking => (
                    vec![Some(wgpu::ColorTargetState {
                        format: PICK_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    depth_state(true, wgpu::CompareFunction::Less, 0),
                ),
                MeshPass::Overdraw => (
                    vec![Some(wgpu::ColorTargetState {
                        format: self.hdr_format,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent::OVER,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    depth_state(false, wgpu::CompareFunction::Always, 0),
                ),
            };
        let fragment = if key.pass == MeshPass::Shadow && !key.alpha_mask {
            None
        } else {
            Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                targets: &targets,
                compilation_options: Default::default(),
            })
        };
        Ok(
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(&format!("engine-render-mesh-{:?}", key)),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    buffers: &buffers,
                    compilation_options: Default::default(),
                },
                fragment,
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode,
                    front_face: wgpu::FrontFace::Ccw,
                    ..Default::default()
                },
                depth_stencil: Some(depth),
                multisample: wgpu::MultisampleState {
                    count: key.samples.max(1),
                    ..Default::default()
                },
                multiview: None,
                cache: None,
            }),
        )
    }
}

fn depth_state(
    write: bool,
    compare: wgpu::CompareFunction,
    slope_bias: i32,
) -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: write,
        depth_compare: compare,
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState {
            constant: slope_bias,
            slope_scale: slope_bias as f32,
            clamp: 0.0,
        },
    }
}

/// What a material resolved to this frame (for sorting and pipelines).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlphaMode {
    Opaque,
    Mask,
    Blend,
}

impl AlphaMode {
    pub fn parse(mode: &str) -> Self {
        match mode.to_ascii_uppercase().as_str() {
            "MASK" => Self::Mask,
            "BLEND" => Self::Blend,
            _ => Self::Opaque,
        }
    }
}

/// Key identifying a material bind group: the material revision plus the
/// albedo override and the revisions of every texture it samples.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct MaterialBindKey {
    material: AssetId,
    material_revision: u64,
    albedo_override: AssetId,
    textures: [(u64, u64); 5],
}

pub struct GpuMaterial {
    pub bind_group: wgpu::BindGroup,
    pub alpha: AlphaMode,
    pub double_sided: bool,
    key: MaterialBindKey,
    last_used: u64,
}

/// Material bind groups cached by content; rebuilt only when the material
/// or one of its textures changes.
#[derive(Default)]
pub struct MaterialCache {
    materials: HashMap<(AssetId, AssetId), GpuMaterial>,
    texture_requests: HashMap<String, Handle<TextureData>>,
    frame: u64,
}

fn material_texture_flags(material: &MaterialData) -> u32 {
    let mut bits = 0;
    if material.base_color_texture.is_some() {
        bits |= 1;
    }
    if material.metallic_roughness_texture.is_some() {
        bits |= 2;
    }
    if material.normal_texture.is_some() {
        bits |= 4;
    }
    if material.occlusion_texture.is_some() {
        bits |= 8;
    }
    if material.emissive_texture.is_some() {
        bits |= 16;
    }
    bits
}

pub fn material_uniform(material: &MaterialData, has_albedo: bool) -> MaterialUniformGpu {
    let alpha = AlphaMode::parse(&material.alpha_mode);
    let mut flags = material_texture_flags(material);
    if has_albedo {
        flags |= 1;
    }
    MaterialUniformGpu {
        base_color: material.base_color_factor,
        metallic_roughness: [
            material.metallic,
            material.roughness,
            material.normal_scale,
            material.occlusion_strength,
        ],
        emissive: [
            material.emissive_factor[0],
            material.emissive_factor[1],
            material.emissive_factor[2],
            material.alpha_cutoff,
        ],
        flags: [
            match alpha {
                AlphaMode::Opaque => 0,
                AlphaMode::Mask => 1,
                AlphaMode::Blend => 2,
            },
            material.double_sided as u32,
            flags,
            0,
        ],
    }
}

impl MaterialCache {
    pub fn begin_frame(&mut self) {
        self.frame += 1;
        let frame = self.frame;
        self.materials
            .retain(|_, material| frame - material.last_used < 120);
    }

    pub fn len(&self) -> usize {
        self.materials.len()
    }

    pub fn is_empty(&self) -> bool {
        self.materials.is_empty()
    }

    fn texture_handle(&mut self, assets: &Assets, path: &str) -> Handle<TextureData> {
        *self
            .texture_requests
            .entry(path.to_owned())
            .or_insert_with(|| assets.request_path::<TextureData>(path))
    }

    /// Ensures a bind group for `(material, albedo override)` exists and is
    /// current; returns the cache key to look it up with [`Self::get`].
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layouts: &SharedLayouts,
        cache: &mut GpuAssetCache,
        server: &AssetServer,
        material_handle: MaterialHandle,
        albedo_override: TextureHandle,
    ) -> (AssetId, AssetId) {
        let fallback = MaterialData::default();
        let material = server
            .material_payload(material_handle)
            .unwrap_or(&fallback);
        let assets = server.assets();
        let slot_paths = [
            material.base_color_texture.clone(),
            material.metallic_roughness_texture.clone(),
            material.normal_texture.clone(),
            material.occlusion_texture.clone(),
            material.emissive_texture.clone(),
        ];
        let roles = [
            TextureRole::Color,
            TextureRole::Data,
            TextureRole::Normal,
            TextureRole::Data,
            TextureRole::Emissive,
        ];
        let handles: Vec<Option<Handle<TextureData>>> = slot_paths
            .iter()
            .map(|path| {
                path.as_deref()
                    .map(|path| self.texture_handle(assets, path))
            })
            .collect();
        let mut textures = [(0u64, 0u64); 5];
        for (slot, handle) in handles.iter().enumerate() {
            if let Some(handle) = handle {
                textures[slot] = (handle.id().value(), assets.revision(*handle));
            }
        }
        let override_revision = server
            .texture_payload(albedo_override)
            .map(|payload| payload.revision)
            .unwrap_or(0);
        if handles[0].is_none() {
            textures[0] = (albedo_override.id().value(), override_revision);
        }
        let key = MaterialBindKey {
            material: material_handle.id(),
            material_revision: server.material_revision(material_handle).unwrap_or(0),
            albedo_override: albedo_override.id(),
            textures,
        };
        let map_key = (material_handle.id(), albedo_override.id());
        if let Some(existing) = self.materials.get_mut(&map_key) {
            if existing.key == key {
                existing.last_used = self.frame;
                return map_key;
            }
        }

        // Upload whatever is available; missing images use fallbacks and
        // the key (texture revision 0) makes us retry next frame.
        for (slot, handle) in handles.iter().enumerate() {
            if let Some(handle) = handle {
                cache.prepare_typed_texture(device, queue, *handle, assets, roles[slot].is_srgb());
            }
        }
        let has_override = handles[0].is_none()
            && cache
                .prepare_texture(device, queue, albedo_override, server, true)
                .is_some();
        let views: Vec<&wgpu::TextureView> = (0..5)
            .map(|slot| {
                let uploaded = match (&handles[slot], slot) {
                    (Some(handle), _) => cache.texture(handle.id(), roles[slot].is_srgb()),
                    (None, 0) if has_override => cache.texture(albedo_override.id(), true),
                    _ => None,
                };
                uploaded
                    .map(|texture| &texture.view)
                    .unwrap_or(&cache.fallback_for(roles[slot]).view)
            })
            .collect();
        let uniform = material_uniform(material, has_override);
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("engine-render-material"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("engine-render-material-bind-group"),
            layout: &layouts.material,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(views[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(views[2]),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(views[3]),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(views[4]),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(&layouts.linear_repeat_aniso),
                },
            ],
        });
        self.materials.insert(
            map_key,
            GpuMaterial {
                bind_group,
                alpha: AlphaMode::parse(&material.alpha_mode),
                double_sided: material.double_sided,
                key,
                last_used: self.frame,
            },
        );
        map_key
    }

    pub fn get(&self, key: (AssetId, AssetId)) -> Option<&GpuMaterial> {
        self.materials.get(&key)
    }
}

/// One object ready to draw: which mesh, material and instance record.
#[derive(Clone, Copy, Debug)]
pub struct DrawObject {
    pub object_index: u32,
    pub mesh: AssetId,
    pub material: (AssetId, AssetId),
    pub skinned: bool,
    pub alpha: AlphaMode,
    pub double_sided: bool,
    pub index_count: u32,
    /// View depth, for transparent back-to-front sorting.
    pub sort_depth: f32,
}

/// A contiguous instance range drawn with one pipeline/material/mesh.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DrawBatch {
    pub key: MeshPipelineKey,
    pub mesh: AssetId,
    pub material: (AssetId, AssetId),
    pub first: u32,
    pub count: u32,
    pub index_count: u32,
}

/// Builds instanced batches for `objects` in one pass. Opaque passes sort
/// by state to maximise instancing; transparent passes keep a strict
/// back-to-front order (instancing only consecutive identical draws).
pub fn build_batches(
    objects: &mut [DrawObject],
    pass: MeshPass,
    samples: u32,
    depth_prepassed: bool,
    indirection: &mut Vec<u32>,
) -> Vec<DrawBatch> {
    let key_of = |object: &DrawObject| MeshPipelineKey {
        pass,
        skinned: object.skinned,
        alpha_mask: object.alpha == AlphaMode::Mask,
        double_sided: object.double_sided,
        samples,
        depth_prepassed: depth_prepassed && pass == MeshPass::Opaque,
    };
    if pass == MeshPass::Transparent {
        objects.sort_by(|a, b| {
            b.sort_depth
                .partial_cmp(&a.sort_depth)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    } else {
        objects.sort_by(|a, b| {
            key_of(a)
                .cmp(&key_of(b))
                .then(a.material.0.value().cmp(&b.material.0.value()))
                .then(a.material.1.value().cmp(&b.material.1.value()))
                .then(a.mesh.value().cmp(&b.mesh.value()))
                .then(
                    a.sort_depth
                        .partial_cmp(&b.sort_depth)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
        });
    }
    let mut batches: Vec<DrawBatch> = Vec::new();
    for object in objects.iter() {
        let key = key_of(object);
        let first = indirection.len() as u32;
        indirection.push(object.object_index);
        // Skinned objects each carry their own palette offset in the
        // instance record, so they instance fine as well.
        match batches.last_mut() {
            Some(batch)
                if batch.key == key
                    && batch.mesh == object.mesh
                    && batch.material == object.material
                    && batch.first + batch.count == first =>
            {
                batch.count += 1;
            }
            _ => batches.push(DrawBatch {
                key,
                mesh: object.mesh,
                material: object.material,
                first,
                count: 1,
                index_count: object.index_count,
            }),
        }
    }
    batches
}

/// Per-frame instance storage shared by every mesh pass.
pub struct InstanceBuffers {
    pub instances: GrowableBuffer,
    pub palettes: GrowableBuffer,
    pub indirection: GrowableBuffer,
    bind_group: Option<(wgpu::BindGroup, [u64; 3])>,
}

/// Alignment of each pass's indirection list (storage dynamic offsets).
pub const INDIRECTION_ALIGNMENT: u64 = 256;
/// Bytes visible through the dynamic indirection binding (4096 draws).
pub const INDIRECTION_WINDOW: u64 = 16 * 1024;

impl InstanceBuffers {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            instances: GrowableBuffer::new(
                device,
                "engine-render-instances",
                wgpu::BufferUsages::STORAGE,
                (size_of::<InstanceGpu>() * 256) as u64,
            ),
            palettes: GrowableBuffer::new(
                device,
                "engine-render-palettes",
                wgpu::BufferUsages::STORAGE,
                64 * 256,
            ),
            indirection: GrowableBuffer::new(
                device,
                "engine-render-indirection",
                wgpu::BufferUsages::STORAGE,
                4096,
            ),
            bind_group: None,
        }
    }

    pub fn upload(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        belt: &mut StagingBelt,
        instances: &[InstanceGpu],
        palettes: &[[[f32; 4]; 4]],
        indirection: &[u8],
    ) {
        let identity = [crate::uniforms::IDENTITY];
        let palettes: &[[[f32; 4]; 4]] = if palettes.is_empty() {
            &identity
        } else {
            palettes
        };
        let zero = [InstanceGpu {
            model: crate::uniforms::IDENTITY,
            prev_model: crate::uniforms::IDENTITY,
            normal: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            params: [0; 4],
        }];
        let instances: &[InstanceGpu] = if instances.is_empty() {
            &zero
        } else {
            instances
        };
        self.instances
            .upload(device, encoder, belt, bytemuck::cast_slice(instances));
        self.palettes
            .upload(device, encoder, belt, bytemuck::cast_slice(palettes));
        let mut indirection_bytes = indirection.to_vec();
        // Keep room for a full binding window at the last offset.
        indirection_bytes.resize(indirection_bytes.len() + INDIRECTION_WINDOW as usize, 0);
        self.indirection
            .upload(device, encoder, belt, &indirection_bytes);
    }

    pub fn bind_group(
        &mut self,
        device: &wgpu::Device,
        layouts: &SharedLayouts,
    ) -> &wgpu::BindGroup {
        let generations = [
            self.instances.generation(),
            self.palettes.generation(),
            self.indirection.generation(),
        ];
        let stale = self
            .bind_group
            .as_ref()
            .is_none_or(|(_, cached)| *cached != generations);
        if stale {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("engine-render-instances-bind-group"),
                layout: &layouts.instances,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.instances.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.palettes.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: self.indirection.buffer(),
                            offset: 0,
                            size: std::num::NonZeroU64::new(INDIRECTION_WINDOW),
                        }),
                    },
                ],
            });
            self.bind_group = Some((bind_group, generations));
        }
        &self.bind_group.as_ref().expect("created above").0
    }
}

/// A pass's slice of the indirection buffer plus its batches.
#[derive(Clone, Debug, Default)]
pub struct PassDrawList {
    pub byte_offset: u32,
    pub batches: Vec<DrawBatch>,
}

/// Builds indirection lists for several passes into one aligned buffer.
/// Batch `first` values are relative to each pass's own window, and every
/// window is bound through the 256-byte dynamic storage binding.
pub struct IndirectionBuilder {
    bytes: Vec<u8>,
}

impl Default for IndirectionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl IndirectionBuilder {
    pub fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    /// Adds a pass; batches are split so no batch spans more instances
    /// than fit in one binding window (`INDIRECTION_WINDOW / 4`).
    pub fn add_pass(
        &mut self,
        objects: &mut [DrawObject],
        pass: MeshPass,
        samples: u32,
        depth_prepassed: bool,
    ) -> Vec<PassDrawList> {
        let mut indirection = Vec::new();
        let batches = build_batches(objects, pass, samples, depth_prepassed, &mut indirection);
        // One window holds up to `per_window` consecutive indirection
        // entries; batches are packed into windows in order and split when
        // a window fills up.
        let per_window = (INDIRECTION_WINDOW / 4) as u32;
        let mut lists: Vec<PassDrawList> = Vec::new();
        let mut window_start = 0usize;
        let mut window_used = per_window;
        for batch in batches {
            let mut remaining = batch.count;
            let mut cursor = batch.first;
            while remaining > 0 {
                if window_used == per_window {
                    window_start =
                        crate::gpu::align_up(self.bytes.len() as u64, INDIRECTION_ALIGNMENT)
                            as usize;
                    self.bytes.resize(window_start, 0);
                    window_used = 0;
                    lists.push(PassDrawList {
                        byte_offset: window_start as u32,
                        batches: Vec::new(),
                    });
                }
                let take = remaining.min(per_window - window_used);
                for index in cursor..cursor + take {
                    self.bytes
                        .extend_from_slice(&indirection[index as usize].to_le_bytes());
                }
                let list = lists.last_mut().expect("window list exists");
                list.batches.push(DrawBatch {
                    first: window_used,
                    count: take,
                    ..batch
                });
                debug_assert_eq!(
                    self.bytes.len(),
                    window_start + 4 * (window_used + take) as usize
                );
                window_used += take;
                cursor += take;
                remaining -= take;
            }
        }
        lists
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Encodes `lists` into an open render pass. The caller has set group 0.
pub fn draw_lists(
    pass: &mut wgpu::RenderPass<'_>,
    lists: &[PassDrawList],
    pipelines: &MeshPipelineStore<'_>,
    instances_bind_group: &wgpu::BindGroup,
    materials: &MaterialCache,
    cache: &GpuAssetCache,
) -> u32 {
    let mut draws = 0;
    let mut current_pipeline: Option<MeshPipelineKey> = None;
    for list in lists {
        pass.set_bind_group(1, instances_bind_group, &[list.byte_offset]);
        for batch in &list.batches {
            let (Some(mesh), Some(material), Some(pipeline)) = (
                cache.mesh(batch.mesh),
                materials.get(batch.material),
                pipelines.get(&batch.key),
            ) else {
                continue;
            };
            if current_pipeline != Some(batch.key) {
                pass.set_pipeline(pipeline);
                current_pipeline = Some(batch.key);
            }
            pass.set_bind_group(2, &material.bind_group, &[]);
            pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            if batch.key.skinned {
                let Some(skin): Option<&wgpu::Buffer> = mesh.skin_buffer.as_ref() else {
                    continue;
                };
                pass.set_vertex_buffer(1, skin.slice(..));
            }
            pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(
                0..mesh.index_count,
                0,
                batch.first..batch.first + batch.count,
            );
            draws += 1;
        }
    }
    draws
}

/// Borrowed view of compiled pipelines for encoding.
pub type MeshPipelineStore<'a> = HashMap<MeshPipelineKey, &'a wgpu::RenderPipeline>;

#[cfg(test)]
mod tests {
    use super::*;

    fn object(index: u32, mesh: u64, material: u64, alpha: AlphaMode, depth: f32) -> DrawObject {
        DrawObject {
            object_index: index,
            mesh: test_id(mesh),
            material: (test_id(material), test_id(0)),
            skinned: false,
            alpha,
            double_sided: false,
            index_count: 36,
            sort_depth: depth,
        }
    }

    fn test_id(value: u64) -> AssetId {
        AssetId::from_raw(value)
    }

    #[test]
    fn opaque_objects_are_instanced_by_state() {
        let mut objects = vec![
            object(0, 1, 7, AlphaMode::Opaque, 5.0),
            object(1, 2, 7, AlphaMode::Opaque, 1.0),
            object(2, 1, 7, AlphaMode::Opaque, 2.0),
            object(3, 1, 8, AlphaMode::Mask, 3.0),
        ];
        let mut indirection = Vec::new();
        let batches = build_batches(&mut objects, MeshPass::Opaque, 1, true, &mut indirection);
        assert_eq!(batches.len(), 3);
        let instanced = batches.iter().find(|b| b.count == 2).unwrap();
        assert_eq!(instanced.mesh, test_id(1));
        assert!(batches.iter().any(|b| b.key.alpha_mask));
        assert_eq!(indirection.len(), 4);
    }

    #[test]
    fn transparent_objects_are_sorted_back_to_front() {
        let mut objects = vec![
            object(0, 1, 7, AlphaMode::Blend, 1.0),
            object(1, 1, 7, AlphaMode::Blend, 9.0),
            object(2, 1, 7, AlphaMode::Blend, 5.0),
        ];
        let mut indirection = Vec::new();
        build_batches(
            &mut objects,
            MeshPass::Transparent,
            1,
            false,
            &mut indirection,
        );
        assert_eq!(indirection, vec![1, 2, 0]);
    }

    #[test]
    fn indirection_windows_are_aligned_and_split() {
        let mut objects: Vec<DrawObject> = (0..5000)
            .map(|i| object(i, 1 + (i as u64 % 2), 7, AlphaMode::Opaque, i as f32))
            .collect();
        let mut builder = IndirectionBuilder::new();
        let lists = builder.add_pass(&mut objects, MeshPass::Opaque, 1, false);
        assert_eq!(lists.len(), 2);
        let total: u32 = lists.iter().flat_map(|l| &l.batches).map(|b| b.count).sum();
        assert_eq!(total, 5000);
        assert!(lists[0].batches.iter().all(|b| b.first + b.count <= 4096));
        assert_eq!(lists[1].byte_offset % INDIRECTION_ALIGNMENT as u32, 0);
        let mut more = vec![object(9, 1, 7, AlphaMode::Opaque, 0.0)];
        let second = builder.add_pass(&mut more, MeshPass::Shadow, 1, false);
        assert_eq!(second[0].byte_offset % INDIRECTION_ALIGNMENT as u32, 0);
        assert!(second[0].byte_offset as usize >= 5000 * 4);
    }

    #[test]
    fn material_uniform_packs_alpha_and_texture_bits() {
        let material = MaterialData {
            alpha_mode: "MASK".into(),
            normal_texture: Some("n.png".into()),
            ..MaterialData::default()
        };
        let uniform = material_uniform(&material, true);
        assert_eq!(uniform.flags[0], 1);
        assert_eq!(uniform.flags[2], 1 | 4);
    }
}
