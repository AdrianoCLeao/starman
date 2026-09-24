// Exposure, bloom composite, ACES filmic tonemap (Narkowicz 2015) and
// output encoding. With OUTPUT_SRGB_ENCODE the shader applies the sRGB
// transfer itself (non-sRGB swapchain formats); otherwise the target's
// sRGB view format encodes on store.

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

struct TonemapUniforms {
    // x: exposure, y: bloom intensity, z: 1 = bloom present, w: dither.
    params: vec4<f32>,
};

@group(0) @binding(0) var hdr_tex: texture_2d<f32>;
@group(0) @binding(1) var hdr_sampler: sampler;
@group(0) @binding(2) var<uniform> tonemap: TonemapUniforms;
@group(0) @binding(3) var bloom_tex: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> FullscreenOut {
    return fullscreen_triangle(idx);
}

fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(input: FullscreenOut) -> @location(0) vec4<f32> {
    var hdr = textureSampleLevel(hdr_tex, hdr_sampler, input.uv, 0.0).rgb;
    if tonemap.params.z > 0.5 {
        hdr += textureSampleLevel(bloom_tex, hdr_sampler, input.uv, 0.0).rgb * tonemap.params.y;
    }
    var mapped = aces_tonemap(hdr * tonemap.params.x);
    mapped = linear_to_srgb(mapped);
    // Tiny ordered dither against banding in 8-bit targets.
    let noise = fract(52.9829189 * fract(dot(input.position.xy, vec2<f32>(0.06711056, 0.00583715))));
    mapped += (noise - 0.5) / 255.0 * tonemap.params.w;
    return vec4<f32>(mapped, 1.0);
}
