// Full-screen debug visualizations over the final image.

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

struct DebugUniform {
    // x: mode, y: overdraw scale.
    mode: vec4<u32>,
};

@group(0) @binding(0) var<uniform> view: ViewUniform;
@group(0) @binding(1) var t_depth: texture_depth_2d;
@group(0) @binding(2) var t_aux: texture_2d<f32>;
@group(0) @binding(3) var<storage, read> cluster_ranges: array<vec2<u32>>;
@group(0) @binding(4) var<uniform> debug: DebugUniform;

const MODE_DEPTH: u32 = 1u;
const MODE_NORMALS: u32 = 2u;
const MODE_CLUSTERS: u32 = 3u;
const MODE_OVERDRAW: u32 = 4u;
const MODE_LIGHT_HEAT: u32 = 5u;
const MODE_SSAO: u32 = 6u;
const MODE_VELOCITY: u32 = 7u;
const MODE_SHADOW_CASCADES: u32 = 8u;

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> FullscreenOut {
    return fullscreen_triangle(idx);
}

fn linear_depth(depth: f32) -> f32 {
    let near = view.near_far.x;
    let far = view.near_far.y;
    return near * far / (far - depth * (far - near));
}

fn world_at(pixel: vec2<i32>) -> vec3<f32> {
    let size = vec2<i32>(textureDimensions(t_depth));
    let p = clamp(pixel, vec2<i32>(0), size - vec2<i32>(1));
    let depth = textureLoad(t_depth, p, 0);
    let uv = (vec2<f32>(p) + 0.5) / vec2<f32>(size);
    let world = view.inv_view_proj * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    return world.xyz / world.w;
}

fn heat(t: f32) -> vec3<f32> {
    let x = clamp(t, 0.0, 1.0);
    return vec3<f32>(smoothstep(0.3, 0.8, x), smoothstep(0.0, 0.5, x) - smoothstep(0.7, 1.0, x), 1.0 - smoothstep(0.0, 0.4, x));
}

@fragment
fn fs_main(input: FullscreenOut) -> @location(0) vec4<f32> {
    let size = vec2<f32>(textureDimensions(t_depth));
    let pixel = vec2<i32>(input.uv * size);
    let depth = textureLoad(t_depth, pixel, 0);
    var color = vec3<f32>(0.0);
    switch debug.mode.x {
        case 1u: {
            color = vec3<f32>(1.0 - clamp(linear_depth(depth) / 100.0, 0.0, 1.0));
        }
        case 2u: {
            let p = world_at(pixel);
            let n = normalize(cross(world_at(pixel + vec2<i32>(0, 1)) - p, world_at(pixel + vec2<i32>(1, 0)) - p));
            color = select(n * 0.5 + 0.5, vec3<f32>(0.0), depth >= 1.0);
        }
        case 3u, 5u: {
            let dims = view.cluster_dims;
            let tile = vec2<u32>(input.position.xy) / max(dims.w, 1u);
            let d = linear_depth(depth);
            let slice_f = log2(max(d, 1e-4)) * view.cluster_params.x - view.cluster_params.y;
            let slice = u32(clamp(slice_f, 0.0, f32(dims.z - 1u)));
            let index = (slice * dims.y + min(tile.y, dims.y - 1u)) * dims.x + min(tile.x, dims.x - 1u);
            let count = f32(cluster_ranges[index].y + view.counts.x);
            color = heat(count / 16.0);
            if debug.mode.x == 3u {
                let edge = any(vec2<u32>(input.position.xy) % max(dims.w, 1u) == vec2<u32>(0u));
                color = select(color * 0.8, vec3<f32>(1.0), edge);
            }
        }
        case 4u: {
            color = heat(textureLoad(t_aux, pixel, 0).r);
        }
        case 6u: {
            color = vec3<f32>(textureLoad(t_aux, pixel, 0).r);
        }
        case 7u: {
            let v = textureLoad(t_aux, pixel, 0).xy * 50.0;
            color = vec3<f32>(0.5 + v.x, 0.5 + v.y, 0.5);
        }
        default: {
            color = vec3<f32>(1.0, 0.0, 1.0);
        }
    }
    return vec4<f32>(color, 1.0);
}
