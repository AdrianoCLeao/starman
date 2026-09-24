//! GPU copies of mesh and texture assets, uploaded lazily, refreshed when
//! the asset revision changes, and retired through the generational arena
//! (freed a few frames later, once in-flight command buffers are done).

use std::collections::HashMap;
use std::mem::size_of;

use bytemuck::{Pod, Zeroable};
use engine_assets::{
    AssetId, AssetServer, Assets, Handle, MeshData, MeshHandle, TextureData, TextureHandle,
};
use engine_core::{EngineError, Result};
use wgpu::util::DeviceExt;

use super::{GpuArena, GpuHandle};
use crate::texture_upload::prepare_rgba8_upload_data;

/// Interleaved static vertex stream: position, normal, uv, tangent.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub struct MeshVertexGpu {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub tangent: [f32; 4],
}

impl MeshVertexGpu {
    pub const ATTRIBUTES: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
        2 => Float32x2,
        3 => Float32x4
    ];

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<Self>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

/// Second stream for skinned meshes: joint indices and weights.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub struct SkinVertexGpu {
    pub joints: [u16; 4],
    pub weights: [f32; 4],
}

impl SkinVertexGpu {
    pub const ATTRIBUTES: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![4 => Uint16x4, 5 => Float32x4];

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<Self>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

pub struct GpuMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub skin_buffer: Option<wgpu::Buffer>,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    pub joint_count: u32,
    pub revision: u64,
}

impl GpuMesh {
    pub fn is_skinned(&self) -> bool {
        self.skin_buffer.is_some()
    }
}

/// Builds the static vertex stream (generating tangents when missing).
pub fn build_mesh_vertices(mesh: &MeshData) -> Vec<MeshVertexGpu> {
    let tangents = if mesh.has_tangents() {
        None
    } else {
        let mut copy = mesh.clone();
        copy.generate_tangents();
        Some(copy.tangents)
    };
    let tangents = tangents.as_ref().unwrap_or(&mesh.tangents);
    mesh.vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| MeshVertexGpu {
            position: vertex.position,
            normal: vertex.normal,
            uv: vertex.uv,
            tangent: tangents.get(index).copied().unwrap_or([1.0, 0.0, 0.0, 1.0]),
        })
        .collect()
}

pub fn build_skin_vertices(mesh: &MeshData) -> Option<Vec<SkinVertexGpu>> {
    mesh.is_skinned().then(|| {
        mesh.joints
            .iter()
            .zip(&mesh.weights)
            .map(|(joints, weights)| SkinVertexGpu {
                joints: *joints,
                weights: *weights,
            })
            .collect()
    })
}

pub fn compute_index_count(indices_len: usize) -> Result<u32> {
    u32::try_from(indices_len)
        .map_err(|_| EngineError::Render("mesh index count overflow for u32 draw call".to_owned()))
}

fn upload_mesh(device: &wgpu::Device, mesh: &MeshData, revision: u64) -> Result<GpuMesh> {
    if mesh.vertices.is_empty() || mesh.indices.is_empty() {
        return Err(EngineError::Render(format!(
            "mesh '{}' is empty",
            mesh.name
        )));
    }
    let vertices = build_mesh_vertices(mesh);
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("engine-render-mesh-vertices"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let skin_vertices = build_skin_vertices(mesh);
    let joint_count = mesh
        .joints
        .iter()
        .flat_map(|joints| joints.iter())
        .copied()
        .max()
        .map(|max| max as u32 + 1)
        .unwrap_or(0);
    let skin_buffer = skin_vertices.map(|skin| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("engine-render-mesh-skin"),
            contents: bytemuck::cast_slice(&skin),
            usage: wgpu::BufferUsages::VERTEX,
        })
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("engine-render-mesh-indices"),
        contents: bytemuck::cast_slice(&mesh.indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    Ok(GpuMesh {
        vertex_buffer,
        skin_buffer,
        index_buffer,
        index_count: compute_index_count(mesh.indices.len())?,
        aabb_min: mesh.aabb_min,
        aabb_max: mesh.aabb_max,
        joint_count,
        revision,
    })
}

pub struct GpuTexture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
    pub mip_count: u32,
    pub revision: u64,
}

/// Number of mips for a full chain down to 1x1.
pub fn mip_count(width: u32, height: u32) -> u32 {
    32 - width.max(height).max(1).leading_zeros()
}

fn srgb_to_linear_lut() -> &'static [f32; 256] {
    static LUT: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    LUT.get_or_init(|| {
        let mut lut = [0.0; 256];
        for (i, value) in lut.iter_mut().enumerate() {
            let c = i as f32 / 255.0;
            *value = if c <= 0.04045 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            };
        }
        lut
    })
}

fn linear_to_srgb_u8(value: f32) -> u8 {
    let c = value.clamp(0.0, 1.0);
    let s = if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0 + 0.5) as u8
}

/// Generates the full mip chain (2x2 box filter), filtering color in
/// linear space when `srgb` so dark edges do not bleed.
pub fn generate_mips(width: u32, height: u32, base: &[u8], srgb: bool) -> Vec<(u32, u32, Vec<u8>)> {
    let lut = srgb_to_linear_lut();
    let mut levels = vec![(width, height, base.to_vec())];
    let (mut w, mut h) = (width, height);
    while w > 1 || h > 1 {
        let (pw, ph) = (w, h);
        w = (w / 2).max(1);
        h = (h / 2).max(1);
        let previous = &levels.last().expect("base level exists").2;
        let mut next = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let mut acc = [0.0f32; 4];
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let sx = (x * 2 + dx).min(pw - 1);
                    let sy = (y * 2 + dy).min(ph - 1);
                    let i = ((sy * pw + sx) * 4) as usize;
                    for c in 0..4 {
                        let v = previous[i + c];
                        acc[c] += if srgb && c < 3 {
                            lut[v as usize]
                        } else {
                            v as f32 / 255.0
                        };
                    }
                }
                let o = ((y * w + x) * 4) as usize;
                for c in 0..4 {
                    let avg = acc[c] * 0.25;
                    next[o + c] = if srgb && c < 3 {
                        linear_to_srgb_u8(avg)
                    } else {
                        (avg.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
                    };
                }
            }
        }
        levels.push((w, h, next));
    }
    levels
}

/// Uploads RGBA8 pixels with a full mip chain.
pub fn upload_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    data: &TextureData,
    srgb: bool,
    revision: u64,
) -> Result<GpuTexture> {
    let format = if srgb {
        wgpu::TextureFormat::Rgba8UnormSrgb
    } else {
        wgpu::TextureFormat::Rgba8Unorm
    };
    let width = data.width.max(1);
    let height = data.height.max(1);
    let mips = generate_mips(width, height, &data.pixels_rgba8, srgb);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: mips.len() as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (level, (w, h, pixels)) in mips.iter().enumerate() {
        let prepared = prepare_rgba8_upload_data(*w, *h, pixels)?;
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: level as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            prepared.data.as_ref(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(prepared.bytes_per_row),
                rows_per_image: Some(prepared.rows_per_image),
            },
            wgpu::Extent3d {
                width: *w,
                height: *h,
                depth_or_array_layers: 1,
            },
        );
    }
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    Ok(GpuTexture {
        texture,
        view,
        width,
        height,
        mip_count: mips.len() as u32,
        revision,
    })
}

/// Built-in 1x1 textures used when a material slot is empty or its image
/// is still loading.
pub struct FallbackTextures {
    pub white_srgb: GpuTexture,
    pub white_linear: GpuTexture,
    pub black_srgb: GpuTexture,
    /// Tangent-space +Z.
    pub flat_normal: GpuTexture,
}

impl FallbackTextures {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let pixel = |rgba: [u8; 4]| TextureData {
            width: 1,
            height: 1,
            pixels_rgba8: rgba.to_vec(),
            revision: 0,
        };
        let make = |label, rgba, srgb| {
            upload_texture(device, queue, label, &pixel(rgba), srgb, 0)
                .expect("1x1 fallback texture uploads")
        };
        Self {
            white_srgb: make("fallback-white-srgb", [255; 4], true),
            white_linear: make("fallback-white-linear", [255; 4], false),
            black_srgb: make("fallback-black-srgb", [0, 0, 0, 255], true),
            flat_normal: make("fallback-flat-normal", [128, 128, 255, 255], false),
        }
    }
}

/// Which kind of texture a sampler slot expects (decides sRGB decoding and
/// the fallback).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureRole {
    Color,
    Data,
    Normal,
    Emissive,
}

impl TextureRole {
    pub fn is_srgb(self) -> bool {
        matches!(self, Self::Color | Self::Emissive)
    }
}

/// Lazily uploaded GPU meshes and textures for legacy and typed handles.
pub struct GpuAssetCache {
    meshes: GpuArena<GpuMesh>,
    mesh_by_asset: HashMap<AssetId, GpuHandle<GpuMesh>>,
    textures: GpuArena<GpuTexture>,
    texture_by_asset: HashMap<(AssetId, bool), GpuHandle<GpuTexture>>,
    pub fallback: FallbackTextures,
    failed: HashMap<AssetId, u64>,
}

impl GpuAssetCache {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, defer_frames: u64) -> Self {
        Self {
            meshes: GpuArena::new(defer_frames),
            mesh_by_asset: HashMap::new(),
            textures: GpuArena::new(defer_frames),
            texture_by_asset: HashMap::new(),
            fallback: FallbackTextures::new(device, queue),
            failed: HashMap::new(),
        }
    }

    pub fn begin_frame(&mut self) {
        self.meshes.advance_frame();
        self.textures.advance_frame();
    }

    pub fn mesh_count(&self) -> usize {
        self.mesh_by_asset.len()
    }

    pub fn texture_count(&self) -> usize {
        self.texture_by_asset.len()
    }

    /// Uploads (or refreshes) a legacy mesh handle; returns `None` while the
    /// payload is missing or failed to upload.
    pub fn prepare_mesh(
        &mut self,
        device: &wgpu::Device,
        handle: MeshHandle,
        server: &AssetServer,
    ) -> Option<&GpuMesh> {
        let id = handle.id();
        let revision = server.mesh_revision(handle).unwrap_or(0);
        if self.failed.get(&id) == Some(&revision) {
            return None;
        }
        let stale = match self.mesh_by_asset.get(&id) {
            Some(gpu) => self.meshes.get(*gpu).map(|mesh| mesh.revision) != Some(revision),
            None => true,
        };
        if stale {
            let payload = server.mesh_payload(handle)?;
            match upload_mesh(device, payload, revision) {
                Ok(mesh) => {
                    if let Some(old) = self.mesh_by_asset.remove(&id) {
                        self.meshes.retire(old);
                    }
                    let gpu = self.meshes.insert(mesh);
                    self.mesh_by_asset.insert(id, gpu);
                    self.failed.remove(&id);
                }
                Err(error) => {
                    log::warn!(target: "engine::render", "mesh upload failed: {error}");
                    self.failed.insert(id, revision);
                    return None;
                }
            }
        }
        self.mesh(id)
    }

    /// Uploads (or refreshes) a typed mesh handle.
    pub fn prepare_typed_mesh(
        &mut self,
        device: &wgpu::Device,
        handle: Handle<MeshData>,
        assets: &Assets,
    ) -> Option<&GpuMesh> {
        let id = handle.id();
        let revision = assets.revision(handle);
        if revision == 0 || self.failed.get(&id) == Some(&revision) {
            return None;
        }
        let stale = match self.mesh_by_asset.get(&id) {
            Some(gpu) => self.meshes.get(*gpu).map(|mesh| mesh.revision) != Some(revision),
            None => true,
        };
        if stale {
            let payload = assets.get(handle)?;
            match upload_mesh(device, &payload, revision) {
                Ok(mesh) => {
                    if let Some(old) = self.mesh_by_asset.remove(&id) {
                        self.meshes.retire(old);
                    }
                    let gpu = self.meshes.insert(mesh);
                    self.mesh_by_asset.insert(id, gpu);
                }
                Err(error) => {
                    log::warn!(target: "engine::render", "mesh upload failed: {error}");
                    self.failed.insert(id, revision);
                    return None;
                }
            }
        }
        self.mesh(id)
    }

    pub fn mesh(&self, id: AssetId) -> Option<&GpuMesh> {
        self.meshes.get(*self.mesh_by_asset.get(&id)?)
    }

    /// Uploads (or refreshes) a legacy texture handle.
    pub fn prepare_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        handle: TextureHandle,
        server: &AssetServer,
        srgb: bool,
    ) -> Option<&GpuTexture> {
        let payload = server.texture_payload(handle)?;
        self.prepare_texture_data(device, queue, handle.id(), payload, payload.revision, srgb)
    }

    /// Uploads (or refreshes) a typed texture handle.
    pub fn prepare_typed_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        handle: Handle<TextureData>,
        assets: &Assets,
        srgb: bool,
    ) -> Option<&GpuTexture> {
        let revision = assets.revision(handle);
        if revision == 0 {
            return None;
        }
        let key = (handle.id(), srgb);
        let fresh = self
            .texture_by_asset
            .get(&key)
            .and_then(|gpu| self.textures.get(*gpu))
            .is_some_and(|texture| texture.revision == revision);
        if fresh {
            return self.texture(handle.id(), srgb);
        }
        let payload = assets.get(handle)?;
        self.prepare_texture_data(device, queue, handle.id(), &payload, revision, srgb)
    }

    fn prepare_texture_data(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: AssetId,
        payload: &TextureData,
        revision: u64,
        srgb: bool,
    ) -> Option<&GpuTexture> {
        let key = (id, srgb);
        let stale = match self.texture_by_asset.get(&key) {
            Some(gpu) => self.textures.get(*gpu).map(|t| t.revision) != Some(revision),
            None => true,
        };
        if stale {
            match upload_texture(
                device,
                queue,
                "engine-render-texture",
                payload,
                srgb,
                revision,
            ) {
                Ok(texture) => {
                    if let Some(old) = self.texture_by_asset.remove(&key) {
                        self.textures.retire(old);
                    }
                    let gpu = self.textures.insert(texture);
                    self.texture_by_asset.insert(key, gpu);
                }
                Err(error) => {
                    log::warn!(target: "engine::render", "texture upload failed: {error}");
                    return None;
                }
            }
        }
        self.texture(id, srgb)
    }

    pub fn texture(&self, id: AssetId, srgb: bool) -> Option<&GpuTexture> {
        self.textures.get(*self.texture_by_asset.get(&(id, srgb))?)
    }

    /// The fallback texture for an empty slot of `role`.
    pub fn fallback_for(&self, role: TextureRole) -> &GpuTexture {
        match role {
            TextureRole::Color => &self.fallback.white_srgb,
            TextureRole::Data => &self.fallback.white_linear,
            TextureRole::Normal => &self.fallback.flat_normal,
            TextureRole::Emissive => &self.fallback.white_srgb,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_assets::MeshVertex;

    #[test]
    fn mip_chain_reaches_one_pixel_and_filters_in_linear_space() {
        assert_eq!(mip_count(1, 1), 1);
        assert_eq!(mip_count(256, 64), 9);
        let mut pixels = Vec::new();
        for i in 0..4 {
            let v = if i % 2 == 0 { 0 } else { 255 };
            pixels.extend_from_slice(&[v, v, v, 255]);
        }
        let srgb = generate_mips(2, 2, &pixels, true);
        let linear = generate_mips(2, 2, &pixels, false);
        assert_eq!(srgb.len(), 2);
        // 50% linear gray is ~188 in sRGB, not 128.
        assert!((srgb[1].2[0] as i32 - 188).abs() <= 1, "{}", srgb[1].2[0]);
        assert!((linear[1].2[0] as i32 - 128).abs() <= 1);
        assert_eq!(srgb[1].2[3], 255);
    }

    #[test]
    fn vertex_streams_carry_tangents_and_skinning() {
        let mut mesh = MeshData::new(
            "tri",
            vec![
                MeshVertex {
                    position: [0.0, 0.0, 0.0],
                    normal: [0.0, 0.0, 1.0],
                    uv: [0.0, 0.0],
                },
                MeshVertex {
                    position: [1.0, 0.0, 0.0],
                    normal: [0.0, 0.0, 1.0],
                    uv: [1.0, 0.0],
                },
                MeshVertex {
                    position: [0.0, 1.0, 0.0],
                    normal: [0.0, 0.0, 1.0],
                    uv: [0.0, 1.0],
                },
            ],
            vec![0, 1, 2],
        );
        let vertices = build_mesh_vertices(&mesh);
        assert_eq!(vertices.len(), 3);
        for v in &vertices {
            assert!((v.tangent[0] - 1.0).abs() < 1e-5, "{:?}", v.tangent);
            assert_eq!(v.tangent[3], 1.0);
        }
        assert!(build_skin_vertices(&mesh).is_none());
        mesh.joints = vec![[0, 1, 0, 0]; 3];
        mesh.weights = vec![[0.5, 0.5, 0.0, 0.0]; 3];
        let skin = build_skin_vertices(&mesh).unwrap();
        assert_eq!(skin[1].joints, [0, 1, 0, 0]);
        assert_eq!(size_of::<MeshVertexGpu>(), 48);
        assert_eq!(size_of::<SkinVertexGpu>(), 24);
    }

    #[test]
    fn index_count_rejects_overflow() {
        assert_eq!(compute_index_count(42).unwrap(), 42);
        if usize::BITS > 32 {
            assert!(compute_index_count(u32::MAX as usize + 1).is_err());
        }
    }
}
