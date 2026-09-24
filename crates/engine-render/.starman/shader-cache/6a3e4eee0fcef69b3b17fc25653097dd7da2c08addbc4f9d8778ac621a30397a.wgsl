// Temporal anti-aliasing resolve: reprojects history with pre-pass
// velocity, clamps it to the current 3x3 neighborhood in YCoCg, and
// blends with a luminance-weighted exponential history.

// Full-screen triangle helpers.

struct FullscreenOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

fn fullscreen_triangle(index: u32) -> FullscreenOut {
    var out: FullscreenOut;
    let x = f32(i32(index & 1u) * 4 - 1);
    let y = f32(i32(index >> 1u) * 4 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}
// Color space helpers.

fn linear_to_srgb(x: vec3<f32>) -> vec3<f32> {
    let lo = x * 12.92;
    let hi = 1.055 * pow(max(x, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, x <= vec3<f32>(0.0031308));
}

fn srgb_to_linear(x: vec3<f32>) -> vec3<f32> {
    let lo = x / 12.92;
    let hi = pow((max(x, vec3<f32>(0.0)) + 0.055) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, x <= vec3<f32>(0.04045));
}

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn rgb_to_ycocg(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        0.25 * c.r + 0.5 * c.g + 0.25 * c.b,
        0.5 * c.r - 0.5 * c.b,
        -0.25 * c.r + 0.5 * c.g - 0.25 * c.b
    );
}

fn ycocg_to_rgb(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

struct TaaUniform {
    // x: history weight, y: 1 = history valid, zw: texel size.
    params: vec4<f32>,
};

@group(0) @binding(0) var t_current: texture_2d<f32>;
@group(0) @binding(1) var t_history: texture_2d<f32>;
@group(0) @binding(2) var t_velocity: texture_2d<f32>;
@group(0) @binding(3) var s_linear: sampler;
@group(0) @binding(4) var<uniform> taa: TaaUniform;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> FullscreenOut {
    return fullscreen_triangle(index);
}

@fragment
fn fs_main(input: FullscreenOut) -> @location(0) vec4<f32> {
    let size = vec2<i32>(textureDimensions(t_current));
    let pixel = vec2<i32>(input.uv * vec2<f32>(size));
    let current = textureLoad(t_current, pixel, 0).rgb;
    if taa.params.y < 0.5 {
        return vec4<f32>(current, 1.0);
    }

    var box_min = vec3<f32>(1e9);
    var box_max = vec3<f32>(-1e9);
    var m1 = vec3<f32>(0.0);
    var m2 = vec3<f32>(0.0);
    // Velocity of the closest depth in the neighborhood would be ideal;
    // the center velocity is used with a dilated neighborhood clamp.
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let p = clamp(pixel + vec2<i32>(x, y), vec2<i32>(0), size - vec2<i32>(1));
            let c = rgb_to_ycocg(textureLoad(t_current, p, 0).rgb);
            box_min = min(box_min, c);
            box_max = max(box_max, c);
            m1 += c;
            m2 += c * c;
        }
    }
    let mean = m1 / 9.0;
    let sigma = sqrt(max(m2 / 9.0 - mean * mean, vec3<f32>(0.0)));
    box_min = max(box_min, mean - sigma * 1.25);
    box_max = min(box_max, mean + sigma * 1.25);

    let velocity = textureLoad(t_velocity, pixel, 0).xy;
    let history_uv = input.uv - velocity;
    if any(history_uv < vec2<f32>(0.0)) || any(history_uv > vec2<f32>(1.0)) {
        return vec4<f32>(current, 1.0);
    }
    var history = rgb_to_ycocg(textureSampleLevel(t_history, s_linear, history_uv, 0.0).rgb);
    history = clamp(history, box_min, box_max);
    let history_rgb = ycocg_to_rgb(history);

    let weight = taa.params.x;
    let current_w = (1.0 - weight) / (1.0 + luminance(current));
    let history_w = weight / (1.0 + luminance(history_rgb));
    let resolved = (current * current_w + history_rgb * history_w) / (current_w + history_w);
    return vec4<f32>(max(resolved, vec3<f32>(0.0)), 1.0);
}
