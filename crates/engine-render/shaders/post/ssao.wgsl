// Screen-space ambient occlusion from the pre-pass depth (normals are
// reconstructed from depth), hemisphere sampling with per-pixel rotation.

#include "common/view.wgsl"
#include "common/fullscreen.wgsl"

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
