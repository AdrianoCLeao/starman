//! Bind group layouts and samplers shared by every pass.

use std::mem::size_of;
use std::num::NonZeroU64;

use crate::uniforms::{MaterialUniformGpu, ViewUniform};

pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
pub const VELOCITY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;
pub const PICK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;
pub const AO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

fn uniform_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    dynamic: bool,
    size: u64,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: dynamic,
            min_binding_size: NonZeroU64::new(size),
        },
        count: None,
    }
}

fn storage_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_dynamic_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: true,
            min_binding_size: None,
        },
        count: None,
    }
}

pub fn texture_entry(
    binding: u32,
    dimension: wgpu::TextureViewDimension,
    sample_type: wgpu::TextureSampleType,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            multisampled: false,
            view_dimension: dimension,
            sample_type,
        },
        count: None,
    }
}

pub fn sampler_entry(binding: u32, kind: wgpu::SamplerBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(kind),
        count: None,
    }
}

const FLOAT: wgpu::TextureSampleType = wgpu::TextureSampleType::Float { filterable: true };

pub struct SharedLayouts {
    /// Group 0 for non-lit passes: the view uniform only (dynamic offset).
    pub view_basic: wgpu::BindGroupLayout,
    /// Group 0 for lit passes: view, lights, clusters, shadows, IBL, SSAO.
    pub view_lit: wgpu::BindGroupLayout,
    /// Group 1: instances, joint palettes, per-pass draw indirection.
    pub instances: wgpu::BindGroupLayout,
    /// Group 2: material uniform, five textures, sampler.
    pub material: wgpu::BindGroupLayout,
    pub linear_clamp: wgpu::Sampler,
    pub linear_repeat_aniso: wgpu::Sampler,
    pub shadow_compare: wgpu::Sampler,
    pub view_uniform_size: u64,
}

impl SharedLayouts {
    pub fn new(device: &wgpu::Device) -> Self {
        let vf = wgpu::ShaderStages::VERTEX_FRAGMENT;
        let frag = wgpu::ShaderStages::FRAGMENT;
        let view_size = size_of::<ViewUniform>() as u64;
        let view_basic = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine-render-view-basic-layout"),
            entries: &[uniform_entry(0, vf, true, view_size)],
        });
        let view_lit = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine-render-view-lit-layout"),
            entries: &[
                uniform_entry(0, vf, true, view_size),
                storage_entry(1, frag),
                storage_entry(2, frag),
                storage_entry(3, frag),
                uniform_entry(4, frag, false, 0),
                texture_entry(
                    5,
                    wgpu::TextureViewDimension::D2Array,
                    wgpu::TextureSampleType::Depth,
                ),
                texture_entry(
                    6,
                    wgpu::TextureViewDimension::D2,
                    wgpu::TextureSampleType::Depth,
                ),
                sampler_entry(7, wgpu::SamplerBindingType::Comparison),
                texture_entry(8, wgpu::TextureViewDimension::Cube, FLOAT),
                texture_entry(9, wgpu::TextureViewDimension::Cube, FLOAT),
                texture_entry(10, wgpu::TextureViewDimension::D2, FLOAT),
                texture_entry(11, wgpu::TextureViewDimension::CubeArray, FLOAT),
                uniform_entry(12, frag, false, 0),
                texture_entry(13, wgpu::TextureViewDimension::D2, FLOAT),
                sampler_entry(14, wgpu::SamplerBindingType::Filtering),
            ],
        });
        let instances = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine-render-instances-layout"),
            entries: &[
                storage_entry(0, vf),
                storage_entry(1, wgpu::ShaderStages::VERTEX),
                storage_dynamic_entry(2, wgpu::ShaderStages::VERTEX),
            ],
        });
        let material = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine-render-material-layout"),
            entries: &[
                uniform_entry(0, frag, false, size_of::<MaterialUniformGpu>() as u64),
                texture_entry(1, wgpu::TextureViewDimension::D2, FLOAT),
                texture_entry(2, wgpu::TextureViewDimension::D2, FLOAT),
                texture_entry(3, wgpu::TextureViewDimension::D2, FLOAT),
                texture_entry(4, wgpu::TextureViewDimension::D2, FLOAT),
                texture_entry(5, wgpu::TextureViewDimension::D2, FLOAT),
                sampler_entry(6, wgpu::SamplerBindingType::Filtering),
            ],
        });
        let linear_clamp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("engine-render-linear-clamp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let linear_repeat_aniso = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("engine-render-material-sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            anisotropy_clamp: 8,
            ..Default::default()
        });
        let shadow_compare = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("engine-render-shadow-compare"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        Self {
            view_basic,
            view_lit,
            instances,
            material,
            linear_clamp,
            linear_repeat_aniso,
            shadow_compare,
            view_uniform_size: view_size,
        }
    }
}
