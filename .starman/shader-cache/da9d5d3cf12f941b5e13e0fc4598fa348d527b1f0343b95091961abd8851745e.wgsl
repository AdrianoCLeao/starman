// Skybox: samples the environment cubemap along the view ray, drawn at
// the far plane behind all opaque geometry.

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
