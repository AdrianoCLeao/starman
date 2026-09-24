// Skybox: samples the environment cubemap along the view ray, drawn at
// the far plane behind all opaque geometry.

#include "common/view.wgsl"
#include "common/fullscreen.wgsl"

@group(0) @binding(0) var<uniform> view: ViewUniform;
@group(1) @binding(0) var t_environment: texture_cube<f32>;
@group(1) @binding(1) var s_environment: sampler;

struct SkyOut {
    @builtin(position) position: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> SkyOut {
    let tri = fullscreen_triangle(index);
    var out: SkyOut;
    // Depth 1.0 (far plane); passes LessEqual only where nothing was drawn.
    out.position = vec4<f32>(tri.position.xy, 1.0, 1.0);
    out.ndc = tri.position.xy;
    return out;
}

@fragment
fn fs_main(input: SkyOut) -> @location(0) vec4<f32> {
    let far = view.inv_view_proj * vec4<f32>(input.ndc, 1.0, 1.0);
    let near = view.inv_view_proj * vec4<f32>(input.ndc, 0.0, 1.0);
    let dir = normalize(far.xyz / far.w - near.xyz / near.w);
    let color = textureSampleLevel(t_environment, s_environment, dir, 0.0).rgb;
    return vec4<f32>(color, 1.0);
}
