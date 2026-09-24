// Depth-aware 4x4 blur of the raw SSAO term.

#include "common/fullscreen.wgsl"

@group(0) @binding(0) var t_ao: texture_2d<f32>;
@group(0) @binding(1) var t_depth: texture_depth_2d;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> FullscreenOut {
    return fullscreen_triangle(index);
}

@fragment
fn fs_main(input: FullscreenOut) -> @location(0) vec4<f32> {
    let size = vec2<i32>(textureDimensions(t_ao));
    let center = vec2<i32>(input.uv * vec2<f32>(size));
    let center_depth = textureLoad(t_depth, center, 0);
    var sum = 0.0;
    var weight = 0.0;
    for (var y = -2; y < 2; y = y + 1) {
        for (var x = -2; x < 2; x = x + 1) {
            let p = clamp(center + vec2<i32>(x, y), vec2<i32>(0), size - vec2<i32>(1));
            let d = textureLoad(t_depth, p, 0);
            let w = 1.0 / (1e-4 + abs(d - center_depth) * 1000.0);
            sum += textureLoad(t_ao, p, 0).r * w;
            weight += w;
        }
    }
    let ao = sum / max(weight, 1e-4);
    return vec4<f32>(ao, ao, ao, 1.0);
}
