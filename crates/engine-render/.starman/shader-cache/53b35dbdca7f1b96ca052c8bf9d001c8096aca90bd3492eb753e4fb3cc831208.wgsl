// Mesh shader for every mesh pass. Variants (defines):
//   PREPASS     depth + velocity (Rg16Float) pre-pass
//   SHADOW      depth-only shadow pass (light view in `view`)
//   LIT         full PBR shading (opaque and transparent)
//   PICKING     entity pick ids (R32Uint)
//   OVERDRAW    additive constant for the overdraw debug view
//   SKINNED     linear blend skinning from the joint palette
//   ALPHA_MASK  alpha-tested materials discard below the cutoff

// Per-view constants (camera, shadow cascade, probe face). Bound with a
// dynamic offset so one buffer serves every view of the frame.

struct ViewUniform {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    // View-projection without TAA jitter (velocity, sky).
    unjittered_view_proj: mat4x4<f32>,
    // xyz position, w = elapsed seconds.
    camera_position: vec4<f32>,
    // width, height, 1/width, 1/height.
    viewport: vec4<f32>,
    // xy = current jitter (clip), zw = previous jitter.
    jitter: vec4<f32>,
    // near, far, exposure, frame index.
    near_far: vec4<f32>,
    // rgb fog color (linear), density.
    fog_color_density: vec4<f32>,
    // height falloff, base height, start distance, max opacity.
    fog_params: vec4<f32>,
    // rgb ambient tint, ssao intensity.
    ambient: vec4<f32>,
    // x: directional light count, y: base cluster index, z: probe count, w: feature bits.
    counts: vec4<u32>,
    // Cluster grid: x, y, z slices, tile size in pixels.
    cluster_dims: vec4<u32>,
    // Cluster depth slicing: scale, bias (log2 slicing).
    cluster_params: vec4<f32>,
};

const FEATURE_SHADOWS_CSM: u32 = 1u;
const FEATURE_SHADOWS_LOCAL: u32 = 2u;
const FEATURE_IBL: u32 = 4u;
const FEATURE_SSAO: u32 = 8u;
const FEATURE_FOG: u32 = 16u;
const FEATURE_PROBES: u32 = 32u;

fn view_has(view: ViewUniform, feature: u32) -> bool {
    return (view.counts.w & feature) != 0u;
}
@group(0) @binding(0) var<uniform> view: ViewUniform;
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

struct VertexInput {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) @invariant clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) world_tangent: vec4<f32>,
    @location(4) current_clip: vec4<f32>,
    @location(5) previous_clip: vec4<f32>,
    @location(6) @interpolate(flat) object_index: u32,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    let object_index = draw_indirection[input.instance_index];
    let inst = instances[object_index];

    var local_position = vec4<f32>(input.position, 1.0);
    var previous_local = local_position;
    var local_normal = input.normal;
    var local_tangent = input.tangent.xyz;

    let world = inst.model * local_position;
    var out: VertexOutput;
    out.clip_position = view.view_proj * world;
    out.world_position = world.xyz;
    let normal_matrix = instance_normal_matrix(inst);
    out.world_normal = normalize(normal_matrix * local_normal);
    out.world_tangent = vec4<f32>(
        normalize((inst.model * vec4<f32>(local_tangent, 0.0)).xyz),
        input.tangent.w
    );
    out.uv = input.uv;
    out.current_clip = view.unjittered_view_proj * world;
    out.previous_clip = view.prev_view_proj * (inst.prev_model * previous_local);
    out.object_index = object_index;
    return out;
}

fn alpha_test(uv: vec2<f32>) {
    if material_base_color(uv).a < material.emissive.w {
        discard;
    }
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec2<f32> {
    alpha_test(input.uv);
    let current = input.current_clip.xy / input.current_clip.w;
    let previous = input.previous_clip.xy / input.previous_clip.w;
    // Screen-space motion in UV units (y down).
    return (current - previous) * vec2<f32>(0.5, -0.5);
}




