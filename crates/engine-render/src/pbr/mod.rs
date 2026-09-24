#![allow(dead_code)]

//! PBR material helpers and alpha modes (ADR 0010).

use engine_assets::MaterialData;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AlphaMode {
    Opaque,
    Mask { cutoff: f32 },
    Blend,
}

impl AlphaMode {
    pub fn from_material(mat: &MaterialData) -> Self {
        match mat.alpha_mode.as_str() {
            "MASK" => Self::Mask {
                cutoff: mat.alpha_cutoff,
            },
            "BLEND" => Self::Blend,
            _ => Self::Opaque,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PbrMaterialUniform {
    pub base_color: [f32; 4],
    pub metallic_roughness: [f32; 4], // metallic, roughness, normal_scale, occlusion
    pub emissive: [f32; 4],           // rgb + alpha_cutoff
    pub flags: [u32; 4],              // alpha_mode, double_sided, has_maps bitmask, pad
}

impl PbrMaterialUniform {
    pub const FLAG_BASE_COLOR: u32 = 1;
    pub const FLAG_MR: u32 = 2;
    pub const FLAG_NORMAL: u32 = 4;
    pub const FLAG_OCCLUSION: u32 = 8;
    pub const FLAG_EMISSIVE: u32 = 16;

    pub fn from_material(mat: &MaterialData) -> Self {
        let alpha_mode = match mat.alpha_mode.as_str() {
            "MASK" => 1u32,
            "BLEND" => 2u32,
            _ => 0u32,
        };
        let mut flags_maps = 0u32;
        if mat.base_color_texture.is_some() {
            flags_maps |= Self::FLAG_BASE_COLOR;
        }
        if mat.metallic_roughness_texture.is_some() {
            flags_maps |= Self::FLAG_MR;
        }
        if mat.normal_texture.is_some() {
            flags_maps |= Self::FLAG_NORMAL;
        }
        if mat.occlusion_texture.is_some() {
            flags_maps |= Self::FLAG_OCCLUSION;
        }
        if mat.emissive_texture.is_some() {
            flags_maps |= Self::FLAG_EMISSIVE;
        }
        Self {
            base_color: mat.base_color_factor,
            metallic_roughness: [
                mat.metallic,
                mat.roughness,
                mat.normal_scale,
                mat.occlusion_strength,
            ],
            emissive: [
                mat.emissive_factor[0],
                mat.emissive_factor[1],
                mat.emissive_factor[2],
                mat.alpha_cutoff,
            ],
            flags: [alpha_mode, u32::from(mat.double_sided), flags_maps, 0],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpha_mask_from_material() {
        let mat = MaterialData {
            alpha_mode: "MASK".into(),
            alpha_cutoff: 0.4,
            ..MaterialData::default()
        };
        assert_eq!(
            AlphaMode::from_material(&mat),
            AlphaMode::Mask { cutoff: 0.4 }
        );
    }
}
