// Material bindings (group 2): glTF metallic-roughness.

struct MaterialUniform {
    base_color: vec4<f32>,
    // metallic, roughness, normal scale, occlusion strength.
    metallic_roughness: vec4<f32>,
    // rgb emissive, alpha cutoff.
    emissive: vec4<f32>,
    // x: alpha mode (0 opaque, 1 mask, 2 blend), y: double sided,
    // z: texture presence bits, w: unused.
    flags: vec4<u32>,
};

const MAT_TEX_BASE: u32 = 1u;
const MAT_TEX_MR: u32 = 2u;
const MAT_TEX_NORMAL: u32 = 4u;
const MAT_TEX_OCCLUSION: u32 = 8u;
const MAT_TEX_EMISSIVE: u32 = 16u;

@group(2) @binding(0) var<uniform> material: MaterialUniform;
@group(2) @binding(1) var t_base_color: texture_2d<f32>;
@group(2) @binding(2) var t_metallic_roughness: texture_2d<f32>;
@group(2) @binding(3) var t_normal: texture_2d<f32>;
@group(2) @binding(4) var t_occlusion: texture_2d<f32>;
@group(2) @binding(5) var t_emissive: texture_2d<f32>;
@group(2) @binding(6) var s_material: sampler;

fn material_base_color(uv: vec2<f32>) -> vec4<f32> {
    return textureSample(t_base_color, s_material, uv) * material.base_color;
}
