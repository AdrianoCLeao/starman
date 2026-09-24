// Full-screen debug visualizations over the final image.

#include "common/view.wgsl"
#include "common/fullscreen.wgsl"
#include "common/color.wgsl"

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
#ifdef OUTPUT_SRGB_ENCODE
    color = linear_to_srgb(color);
#endif
    return vec4<f32>(color, 1.0);
}
