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
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> model: ModelUniform;
@group(2) @binding(0) var<uniform> material: MaterialUniform;
@group(2) @binding(1) var t_albedo: texture_2d<f32>;
@group(2) @binding(2) var s_albedo: sampler;

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

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let albedo_sample = textureSample(t_albedo, s_albedo, input.uv);
    let base_color = albedo_sample * material.base_color;
    let metallic = clamp(material.metallic_roughness.x, 0.0, 1.0);
    let roughness = clamp(material.metallic_roughness.y, 0.04, 1.0);

    let n = normalize(input.world_normal);
    let l = normalize(-camera.light_direction.xyz);
    let v = normalize(camera.camera_position.xyz - input.world_position);
    let color = shade_blinn_phong(base_color.rgb, n, l, v, metallic, roughness);
    return vec4<f32>(color, base_color.a);
}
