//! `#[repr(C)]` mirrors of the WGSL uniform/storage structs. Field order
//! and padding must match `shaders/common/*.wgsl` exactly; the size tests
//! below catch drift.

use bytemuck::{Pod, Zeroable};

pub type Mat4Cols = [[f32; 4]; 4];

pub const IDENTITY: Mat4Cols = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

pub const FEATURE_SHADOWS_CSM: u32 = 1;
pub const FEATURE_SHADOWS_LOCAL: u32 = 2;
pub const FEATURE_IBL: u32 = 4;
pub const FEATURE_SSAO: u32 = 8;
pub const FEATURE_FOG: u32 = 16;
pub const FEATURE_PROBES: u32 = 32;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ViewUniform {
    pub view: Mat4Cols,
    pub proj: Mat4Cols,
    pub view_proj: Mat4Cols,
    pub inv_view_proj: Mat4Cols,
    pub prev_view_proj: Mat4Cols,
    pub unjittered_view_proj: Mat4Cols,
    pub camera_position: [f32; 4],
    pub viewport: [f32; 4],
    pub jitter: [f32; 4],
    pub near_far: [f32; 4],
    pub fog_color_density: [f32; 4],
    pub fog_params: [f32; 4],
    pub ambient: [f32; 4],
    pub counts: [u32; 4],
    pub cluster_dims: [u32; 4],
    pub cluster_params: [f32; 4],
}

impl Default for ViewUniform {
    fn default() -> Self {
        Self {
            view: IDENTITY,
            proj: IDENTITY,
            view_proj: IDENTITY,
            inv_view_proj: IDENTITY,
            prev_view_proj: IDENTITY,
            unjittered_view_proj: IDENTITY,
            camera_position: [0.0; 4],
            viewport: [1.0, 1.0, 1.0, 1.0],
            jitter: [0.0; 4],
            near_far: [0.1, 1000.0, 1.0, 0.0],
            fog_color_density: [0.0; 4],
            fog_params: [0.0; 4],
            ambient: [1.0, 1.0, 1.0, 1.0],
            counts: [0; 4],
            cluster_dims: [1, 1, 1, 64],
            cluster_params: [0.0; 4],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct InstanceGpu {
    pub model: Mat4Cols,
    pub prev_model: Mat4Cols,
    pub normal: [[f32; 4]; 3],
    /// palette offset, previous palette offset, pick id, flags.
    pub params: [u32; 4],
}

pub const INSTANCE_FLAG_SKINNED: u32 = 1;
pub const INSTANCE_FLAG_RECEIVE_SHADOWS: u32 = 2;
pub const INSTANCE_FLAG_SELECTED: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MaterialUniformGpu {
    pub base_color: [f32; 4],
    pub metallic_roughness: [f32; 4],
    pub emissive: [f32; 4],
    pub flags: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ShadowUniform {
    pub cascades: [Mat4Cols; 4],
    pub cascade_splits: [f32; 4],
    pub csm_params: [f32; 4],
    pub local_params: [f32; 4],
    pub local_matrices: [Mat4Cols; 12],
    pub local_tiles: [[f32; 4]; 12],
}

impl Default for ShadowUniform {
    fn default() -> Self {
        Self {
            cascades: [IDENTITY; 4],
            cascade_splits: [0.0; 4],
            csm_params: [0.0; 4],
            local_params: [0.0; 4],
            local_matrices: [IDENTITY; 12],
            local_tiles: [[0.0; 4]; 12],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct ProbeGpu {
    pub position: [f32; 4],
    pub extents_intensity: [f32; 4],
    pub params: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct ProbeUniform {
    pub probes: [ProbeGpu; 4],
    pub ibl_params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct SkyUniform {
    pub zenith: [f32; 4],
    pub horizon: [f32; 4],
    pub ground: [f32; 4],
    pub sun_direction: [f32; 4],
    pub sun_color: [f32; 4],
    pub params: [f32; 4],
}

/// Converts a glam matrix to column arrays.
pub fn cols(m: engine_math::Mat4) -> Mat4Cols {
    m.to_cols_array_2d()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    #[test]
    fn struct_sizes_match_wgsl() {
        assert_eq!(size_of::<ViewUniform>(), 6 * 64 + 10 * 16);
        assert_eq!(size_of::<InstanceGpu>(), 192);
        assert_eq!(size_of::<MaterialUniformGpu>(), 64);
        assert_eq!(
            size_of::<ShadowUniform>(),
            4 * 64 + 3 * 16 + 12 * 64 + 12 * 16
        );
        assert_eq!(size_of::<ProbeUniform>(), 4 * 48 + 16);
        assert_eq!(size_of::<SkyUniform>(), 96);
    }
}
