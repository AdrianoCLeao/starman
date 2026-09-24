// Screen-space ambient occlusion from the pre-pass depth (normals are
// reconstructed from depth), hemisphere sampling with per-pixel rotation.

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

struct SsaoUniform {
    // x: radius (world), y: bias, z: power, w: sample count.
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> view: ViewUniform;
@group(0) @binding(1) var t_depth: texture_depth_2d;
@group(0) @binding(2) var<uniform> ssao: SsaoUniform;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> FullscreenOut {
    return fullscreen_triangle(index);
}

fn view_position(pixel: vec2<i32>) -> vec3<f32> {
    let size = vec2<i32>(textureDimensions(t_depth));
    let p = clamp(pixel, vec2<i32>(0), size - vec2<i32>(1));
    let depth = textureLoad(t_depth, p, 0);
    let uv = (vec2<f32>(p) + 0.5) / vec2<f32>(size);
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let world = view.inv_view_proj * ndc;
    let world_pos = world.xyz / world.w;
    return (view.view * vec4<f32>(world_pos, 1.0)).xyz;
}

fn interleaved_gradient_noise(p: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

@fragment
fn fs_main(input: FullscreenOut) -> @location(0) vec4<f32> {
    let size = vec2<f32>(textureDimensions(t_depth));
    let pixel = vec2<i32>(input.uv * size);
    let depth = textureLoad(t_depth, pixel, 0);
    if depth >= 1.0 {
        return vec4<f32>(1.0);
    }
    let p = view_position(pixel);
    // Pick the smaller-difference neighbors for a stable normal on edges.
    let pr = view_position(pixel + vec2<i32>(1, 0));
    let pl = view_position(pixel - vec2<i32>(1, 0));
    let pd = view_position(pixel + vec2<i32>(0, 1));
    let pu = view_position(pixel - vec2<i32>(0, 1));
    let dx = select(p - pl, pr - p, abs(pr.z - p.z) < abs(p.z - pl.z));
    let dy = select(p - pu, pd - p, abs(pd.z - p.z) < abs(p.z - pu.z));
    let n = normalize(cross(dy, dx));

    let radius = ssao.params.x;
    let bias = ssao.params.y;
    let count = u32(ssao.params.w);
    let noise = interleaved_gradient_noise(input.position.xy) * 6.2831853;
    let random = vec3<f32>(cos(noise), sin(noise), 0.0);
    let tangent = normalize(random - n * dot(random, n));
    let bitangent = cross(n, tangent);
    let tbn = mat3x3<f32>(tangent, bitangent, n);

    var occlusion = 0.0;
    for (var i = 0u; i < count; i = i + 1u) {
        let fi = f32(i);
        // Deterministic hemisphere kernel, denser near the origin.
        let a = fi * 2.39996323;
        let r = sqrt((fi + 0.5) / f32(count));
        let z = sqrt(max(1.0 - r * r, 0.0));
        var sample_dir = vec3<f32>(cos(a) * r, sin(a) * r, z);
        let scale = mix(0.1, 1.0, (fi / f32(count)) * (fi / f32(count)));
        let sample_pos = p + tbn * sample_dir * radius * scale;
        let clip = view.proj * vec4<f32>(sample_pos, 1.0);
        let ndc = clip.xy / clip.w;
        let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) {
            continue;
        }
        let scene = view_position(vec2<i32>(uv * size));
        let range = smoothstep(0.0, 1.0, radius / max(abs(p.z - scene.z), 1e-4));
        if scene.z >= sample_pos.z + bias {
            occlusion += range;
        }
    }
    let ao = pow(clamp(1.0 - occlusion / f32(max(count, 1u)), 0.0, 1.0), ssao.params.z);
    return vec4<f32>(ao, ao, ao, 1.0);
}
