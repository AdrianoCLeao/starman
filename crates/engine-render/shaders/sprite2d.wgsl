struct CameraUniform {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var t_sprite: texture_2d<f32>;
@group(1) @binding(1) var s_sprite: sampler;

struct VertexInput {
    @location(0) local_position: vec2<f32>,
    @location(1) quad_uv: vec2<f32>,
    @location(2) model_0: vec4<f32>,
    @location(3) model_1: vec4<f32>,
    @location(4) model_2: vec4<f32>,
    @location(5) model_3: vec4<f32>,
    @location(6) color: vec4<f32>,
    @location(7) uv_rect: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    let model = mat4x4<f32>(input.model_0, input.model_1, input.model_2, input.model_3);
    let world_position = model * vec4<f32>(input.local_position, 0.0, 1.0);
    output.clip_position = camera.view_proj * world_position;
    output.uv = mix(input.uv_rect.xy, input.uv_rect.zw, input.quad_uv);
    output.color = input.color;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let texel = textureSample(t_sprite, s_sprite, input.uv);
    return texel * input.color;
}
