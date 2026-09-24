// Mesh shader for every mesh pass. Variants (defines):
//   PREPASS     depth + velocity (Rg16Float) pre-pass
//   SHADOW      depth-only shadow pass (light view in `view`)
//   LIT         full PBR shading (opaque and transparent)
//   PICKING     entity pick ids (R32Uint)
//   OVERDRAW    additive constant for the overdraw debug view
//   SKINNED     linear blend skinning from the joint palette
//   ALPHA_MASK  alpha-tested materials discard below the cutoff

#ifdef LIT
#include "common/lighting.wgsl"
#else
#include "common/view.wgsl"
@group(0) @binding(0) var<uniform> view: ViewUniform;
#endif
#include "common/instance.wgsl"
#include "common/material.wgsl"
#ifdef SKINNED
#include "common/skinning.wgsl"
#endif

struct VertexInput {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
#ifdef SKINNED
    @location(4) joints: vec4<u32>,
    @location(5) weights: vec4<f32>,
#endif
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
#ifdef SKINNED
    let skin = skin_matrix(inst.params.x, input.joints, input.weights);
    let previous_skin = skin_matrix(inst.params.y, input.joints, input.weights);
    local_position = skin * local_position;
    previous_local = previous_skin * vec4<f32>(input.position, 1.0);
    let skin3 = mat3x3<f32>(skin[0].xyz, skin[1].xyz, skin[2].xyz);
    local_normal = skin3 * input.normal;
    local_tangent = skin3 * input.tangent.xyz;
#endif

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
#ifdef ALPHA_MASK
    if material_base_color(uv).a < material.emissive.w {
        discard;
    }
#endif
}

#ifdef PREPASS
@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec2<f32> {
    alpha_test(input.uv);
    let current = input.current_clip.xy / input.current_clip.w;
    let previous = input.previous_clip.xy / input.previous_clip.w;
    // Screen-space motion in UV units (y down).
    return (current - previous) * vec2<f32>(0.5, -0.5);
}
#endif

#ifdef SHADOW
@fragment
fn fs_main(input: VertexOutput) {
    alpha_test(input.uv);
}
#endif

#ifdef PICKING
@fragment
fn fs_main(input: VertexOutput) -> @location(0) u32 {
    alpha_test(input.uv);
    return instances[input.object_index].params.z;
}
#endif

#ifdef OVERDRAW
@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.1, 0.1, 0.1, 1.0);
}
#endif

#ifdef LIT
fn perturbed_normal(input: VertexOutput, facing: bool) -> vec3<f32> {
    var n = normalize(input.world_normal);
    if material.flags.y == 1u && !facing {
        n = -n;
    }
    if (material.flags.z & MAT_TEX_NORMAL) == 0u {
        return n;
    }
    let t = normalize(input.world_tangent.xyz - n * dot(n, input.world_tangent.xyz));
    let b = cross(n, t) * input.world_tangent.w;
    var tn = textureSample(t_normal, s_material, input.uv).xyz * 2.0 - 1.0;
    tn = vec3<f32>(tn.xy * material.metallic_roughness.z, tn.z);
    return normalize(mat3x3<f32>(t, b, n) * tn);
}

@fragment
fn fs_main(input: VertexOutput, @builtin(front_facing) facing: bool) -> @location(0) vec4<f32> {
    let base = material_base_color(input.uv);
#ifdef ALPHA_MASK
    if base.a < material.emissive.w {
        discard;
    }
#endif
    let mr = textureSample(t_metallic_roughness, s_material, input.uv);
    let metallic = clamp(material.metallic_roughness.x * mr.b, 0.0, 1.0);
    let roughness = clamp(material.metallic_roughness.y * mr.g, 0.04, 1.0);
    let occlusion_sample = textureSample(t_occlusion, s_material, input.uv).r;
    let occlusion = mix(1.0, occlusion_sample, material.metallic_roughness.w);
    let emissive = material.emissive.rgb * textureSample(t_emissive, s_material, input.uv).rgb;
    let inst = instances[input.object_index];

    var s: SurfaceInput;
    s.world_pos = input.world_position;
    s.frag_coord = input.clip_position.xy;
    s.n = perturbed_normal(input, facing);
    s.v = normalize(view.camera_position.xyz - input.world_position);
    s.base_color = base.rgb;
    s.metallic = metallic;
    s.roughness = roughness;
    s.occlusion = occlusion;
    s.emissive = emissive;
    s.receive_shadows = (inst.params.w & INSTANCE_FLAG_RECEIVE_SHADOWS) != 0u;
    var color = shade_surface(s);
    if (inst.params.w & INSTANCE_FLAG_SELECTED) != 0u {
        let rim = pow(1.0 - max(dot(s.n, s.v), 0.0), 3.0);
        color += vec3<f32>(1.0, 0.55, 0.1) * rim * 2.0;
    }
    var alpha = 1.0;
    if material.flags.x == 2u {
        alpha = base.a;
    }
    return vec4<f32>(color, alpha);
}
#endif
