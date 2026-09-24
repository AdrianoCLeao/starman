// Dual-filter bloom: thresholded 13-tap downsample chain followed by a
// 3x3 tent upsample accumulated additively (Jimenez 2014).

#include "common/fullscreen.wgsl"
#include "common/color.wgsl"

struct BloomUniform {
    // x: threshold, y: soft knee, z: upsample radius, w: 1 = first downsample.
    params: vec4<f32>,
    // xy: source texel size.
    texel: vec4<f32>,
};

@group(0) @binding(0) var t_source: texture_2d<f32>;
@group(0) @binding(1) var s_source: sampler;
@group(0) @binding(2) var<uniform> bloom: BloomUniform;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> FullscreenOut {
    return fullscreen_triangle(index);
}

fn karis_weight(c: vec3<f32>) -> f32 {
    return 1.0 / (1.0 + luminance(c));
}

fn threshold(c: vec3<f32>) -> vec3<f32> {
    let brightness = max(max(c.r, c.g), c.b);
    let knee = bloom.params.x * bloom.params.y + 1e-5;
    var soft = brightness - bloom.params.x + knee;
    soft = clamp(soft, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee);
    let contribution = max(soft, brightness - bloom.params.x) / max(brightness, 1e-5);
    return c * contribution;
}

@fragment
fn fs_downsample(input: FullscreenOut) -> @location(0) vec4<f32> {
    let t = bloom.texel.xy;
    let uv = input.uv;
    let a = textureSampleLevel(t_source, s_source, uv + t * vec2<f32>(-2.0, -2.0), 0.0).rgb;
    let b = textureSampleLevel(t_source, s_source, uv + t * vec2<f32>(0.0, -2.0), 0.0).rgb;
    let c = textureSampleLevel(t_source, s_source, uv + t * vec2<f32>(2.0, -2.0), 0.0).rgb;
    let d = textureSampleLevel(t_source, s_source, uv + t * vec2<f32>(-2.0, 0.0), 0.0).rgb;
    let e = textureSampleLevel(t_source, s_source, uv, 0.0).rgb;
    let f = textureSampleLevel(t_source, s_source, uv + t * vec2<f32>(2.0, 0.0), 0.0).rgb;
    let g = textureSampleLevel(t_source, s_source, uv + t * vec2<f32>(-2.0, 2.0), 0.0).rgb;
    let h = textureSampleLevel(t_source, s_source, uv + t * vec2<f32>(0.0, 2.0), 0.0).rgb;
    let i = textureSampleLevel(t_source, s_source, uv + t * vec2<f32>(2.0, 2.0), 0.0).rgb;
    let j = textureSampleLevel(t_source, s_source, uv + t * vec2<f32>(-1.0, -1.0), 0.0).rgb;
    let k = textureSampleLevel(t_source, s_source, uv + t * vec2<f32>(1.0, -1.0), 0.0).rgb;
    let l = textureSampleLevel(t_source, s_source, uv + t * vec2<f32>(-1.0, 1.0), 0.0).rgb;
    let m = textureSampleLevel(t_source, s_source, uv + t * vec2<f32>(1.0, 1.0), 0.0).rgb;

    var color: vec3<f32>;
    if bloom.params.w > 0.5 {
        // First pass: threshold and Karis-average groups to kill fireflies.
        let g0 = (a + b + d + e) * 0.25;
        let g1 = (b + c + e + f) * 0.25;
        let g2 = (d + e + g + h) * 0.25;
        let g3 = (e + f + h + i) * 0.25;
        let g4 = (j + k + l + m) * 0.25;
        let w0 = karis_weight(g0) * 0.125;
        let w1 = karis_weight(g1) * 0.125;
        let w2 = karis_weight(g2) * 0.125;
        let w3 = karis_weight(g3) * 0.125;
        let w4 = karis_weight(g4) * 0.5;
        color = (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / (w0 + w1 + w2 + w3 + w4);
        color = threshold(color);
    } else {
        color = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625
            + (j + k + l + m) * 0.125;
    }
    return vec4<f32>(max(color, vec3<f32>(0.0)), 1.0);
}

@fragment
fn fs_upsample(input: FullscreenOut) -> @location(0) vec4<f32> {
    let t = bloom.texel.xy * bloom.params.z;
    let uv = input.uv;
    var color = textureSampleLevel(t_source, s_source, uv, 0.0).rgb * 4.0;
    color += textureSampleLevel(t_source, s_source, uv + vec2<f32>(-t.x, 0.0), 0.0).rgb * 2.0;
    color += textureSampleLevel(t_source, s_source, uv + vec2<f32>(t.x, 0.0), 0.0).rgb * 2.0;
    color += textureSampleLevel(t_source, s_source, uv + vec2<f32>(0.0, -t.y), 0.0).rgb * 2.0;
    color += textureSampleLevel(t_source, s_source, uv + vec2<f32>(0.0, t.y), 0.0).rgb * 2.0;
    color += textureSampleLevel(t_source, s_source, uv + vec2<f32>(-t.x, -t.y), 0.0).rgb;
    color += textureSampleLevel(t_source, s_source, uv + vec2<f32>(t.x, -t.y), 0.0).rgb;
    color += textureSampleLevel(t_source, s_source, uv + vec2<f32>(-t.x, t.y), 0.0).rgb;
    color += textureSampleLevel(t_source, s_source, uv + vec2<f32>(t.x, t.y), 0.0).rgb;
    return vec4<f32>(color / 16.0, 1.0);
}
