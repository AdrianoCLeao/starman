// Per-object data (group 1). Passes draw through an indirection list so
// each pass (camera, cascade, probe face) can cull independently.

struct Instance {
    model: mat4x4<f32>,
    prev_model: mat4x4<f32>,
    // Inverse-transpose of the model's upper 3x3, as three columns.
    normal0: vec4<f32>,
    normal1: vec4<f32>,
    normal2: vec4<f32>,
    // x: palette offset, y: previous palette offset, z: pick id, w: flags.
    params: vec4<u32>,
};

const INSTANCE_FLAG_SKINNED: u32 = 1u;
const INSTANCE_FLAG_RECEIVE_SHADOWS: u32 = 2u;
const INSTANCE_FLAG_SELECTED: u32 = 4u;

@group(1) @binding(0) var<storage, read> instances: array<Instance>;
@group(1) @binding(1) var<storage, read> palettes: array<mat4x4<f32>>;
@group(1) @binding(2) var<storage, read> draw_indirection: array<u32>;

fn instance_normal_matrix(inst: Instance) -> mat3x3<f32> {
    return mat3x3<f32>(inst.normal0.xyz, inst.normal1.xyz, inst.normal2.xyz);
}
