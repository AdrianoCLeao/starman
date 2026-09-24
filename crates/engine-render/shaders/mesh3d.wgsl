#include "common/lighting.wgsl"

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_position: vec4<f32>,
    light_direction: vec4<f32>,
};

struct ModelUniform {
    model: mat4x4<f32>,
    normal: mat4x4<f32>,
};

struct MaterialUniform {
    base_color: vec4<f32>,
    metallic_roughness: vec4<f32>,
    emissive: vec4<f32>,
    flags: vec4<u32>,
};

struct GpuLight {
    position_range: vec4<f32>,
    color_intensity: vec4<f32>,
    direction_cone: vec4<f32>,
    light_type: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> model: ModelUniform;
@group(2) @binding(0) var<uniform> material: MaterialUniform;
@group(2) @binding(1) var t_albedo: texture_2d<f32>;
@group(2) @binding(2) var s_albedo: sampler;
@group(3) @binding(0) var<storage, read> lights: array<GpuLight>;
@group(3) @binding(1) var<uniform> light_count: vec4<u32>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    let world_pos = model.model * vec4<f32>(input.position, 1.0);
    output.clip_position = camera.view_proj * world_pos;
    output.world_position = world_pos.xyz;
    output.world_normal = normalize((model.normal * vec4<f32>(input.normal, 0.0)).xyz);
    output.uv = input.uv;
    return output;
}

fn evaluate_light(light: GpuLight, world_pos: vec3<f32>) -> vec3<f32> {
    if light.light_type == 0u {
        return normalize(-light.direction_cone.xyz);
    }
    let to_light = light.position_range.xyz - world_pos;
    let dist = length(to_light);
    let range = max(light.position_range.w, 0.001);
    if dist > range {
        return vec3<f32>(0.0);
    }
    return normalize(to_light);
}

fn light_radiance(light: GpuLight, world_pos: vec3<f32>, l: vec3<f32>) -> vec3<f32> {
    let color = light.color_intensity.xyz * light.color_intensity.w;
    if light.light_type == 0u {
        return color;
    }
    let dist = length(light.position_range.xyz - world_pos);
    let range = max(light.position_range.w, 0.001);
    let atten = saturate(1.0 - dist / range);
    var spot = 1.0;
    if light.light_type == 2u {
        let dir = normalize(light.direction_cone.xyz);
        let cos_outer = cos(light.direction_cone.w);
        let cos_inner = cos(bitcast<f32>(light.pad0));
        let cd = dot(-l, dir);
        spot = smoothstep(cos_outer, cos_inner, cd);
    }
    return color * (atten * atten) * spot;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let albedo_sample = textureSample(t_albedo, s_albedo, input.uv);
    let base_color = albedo_sample * material.base_color;
    if material.flags.x == 1u && base_color.a < material.emissive.w {
        discard;
    }
    let metallic = clamp(material.metallic_roughness.x, 0.0, 1.0);
    let roughness = clamp(material.metallic_roughness.y, 0.04, 1.0);
    let n = normalize(input.world_normal);
    let v = normalize(camera.camera_position.xyz - input.world_position);

    var color = base_color.rgb * 0.03;
    let count = min(light_count.x, 128u);
    for (var i = 0u; i < count; i = i + 1u) {
        let light = lights[i];
        let l = evaluate_light(light, input.world_position);
        if length(l) < 1e-5 {
            continue;
        }
        let rad = light_radiance(light, input.world_position, l);
        color += shade_pbr_light(base_color.rgb, metallic, roughness, n, v, l, rad);
    }
    // Fallback single directional if no lights uploaded.
    if count == 0u {
        let l = normalize(-camera.light_direction.xyz);
        color += shade_pbr_light(
            base_color.rgb,
            metallic,
            roughness,
            n,
            v,
            l,
            vec3<f32>(1.0),
        );
    }
    color += material.emissive.xyz;
    return vec4<f32>(color, base_color.a);
}
