// Lit-pass group 0 bindings and the full lighting model: clustered local
// lights, directional lights, cascaded + local shadows, IBL with blended
// box-projected reflection probes, SSAO and height fog.

#include "common/view.wgsl"
#include "common/brdf.wgsl"

struct GpuLight {
    position_range: vec4<f32>,
    color_intensity: vec4<f32>,
    direction_cone: vec4<f32>,
    // x: type (0 dir, 1 point, 2 spot), y: first shadow matrix or 0xffffffff,
    // z: bitcast inner cone angle, w: unused.
    params: vec4<u32>,
};

struct ShadowUniform {
    cascades: array<mat4x4<f32>, 4>,
    // View-space far distance of each cascade.
    cascade_splits: vec4<f32>,
    // x: 1 / resolution, y: depth bias, z: normal bias, w: cascade count.
    csm_params: vec4<f32>,
    // x: 1 / atlas resolution, y: depth bias, z: normal bias, w: pcf radius.
    local_params: vec4<f32>,
    local_matrices: array<mat4x4<f32>, 12>,
    // xy: atlas uv offset, zw: atlas uv scale.
    local_tiles: array<vec4<f32>, 12>,
};

struct Probe {
    // xyz position, w unused.
    position: vec4<f32>,
    // xyz box half extents, w intensity.
    extents_intensity: vec4<f32>,
    // x: cube array slot, y: blend distance bits.
    params: vec4<u32>,
};

struct ProbeUniform {
    probes: array<Probe, 4>,
    // x: IBL intensity, y: prefiltered mip count, z: sky intensity.
    ibl_params: vec4<f32>,
};

const LIGHT_DIRECTIONAL: u32 = 0u;
const LIGHT_POINT: u32 = 1u;
const LIGHT_SPOT: u32 = 2u;
const NO_SHADOW: u32 = 0xffffffffu;

@group(0) @binding(0) var<uniform> view: ViewUniform;
@group(0) @binding(1) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(2) var<storage, read> cluster_ranges: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> cluster_indices: array<u32>;
@group(0) @binding(4) var<uniform> shadows: ShadowUniform;
@group(0) @binding(5) var t_csm: texture_depth_2d_array;
@group(0) @binding(6) var t_local_shadows: texture_depth_2d;
@group(0) @binding(7) var s_shadow: sampler_comparison;
@group(0) @binding(8) var t_irradiance: texture_cube<f32>;
@group(0) @binding(9) var t_prefiltered: texture_cube<f32>;
@group(0) @binding(10) var t_brdf_lut: texture_2d<f32>;
@group(0) @binding(11) var t_probes: texture_cube_array<f32>;
@group(0) @binding(12) var<uniform> probes: ProbeUniform;
@group(0) @binding(13) var t_ssao: texture_2d<f32>;
@group(0) @binding(14) var s_linear: sampler;

fn view_depth(world_pos: vec3<f32>) -> f32 {
    return -(view.view * vec4<f32>(world_pos, 1.0)).z;
}

fn cluster_index(frag_coord: vec2<f32>, depth: f32) -> u32 {
    let dims = view.cluster_dims;
    let tile = vec2<u32>(frag_coord) / max(dims.w, 1u);
    let slice_f = log2(max(depth, 1e-4)) * view.cluster_params.x - view.cluster_params.y;
    let slice = u32(clamp(slice_f, 0.0, f32(dims.z - 1u)));
    let x = min(tile.x, dims.x - 1u);
    let y = min(tile.y, dims.y - 1u);
    // counts.y: base cluster (non-zero for views that use a single
    // "every local light" cluster, e.g. reflection probe faces).
    return view.counts.y + (slice * dims.y + y) * dims.x + x;
}

fn pcf_csm(layer: i32, uv: vec2<f32>, depth: f32) -> f32 {
    let texel = shadows.csm_params.x;
    var sum = 0.0;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel;
            sum += textureSampleCompareLevel(t_csm, s_shadow, uv + offset, layer, depth);
        }
    }
    return sum / 9.0;
}

fn directional_shadow(world_pos: vec3<f32>, n: vec3<f32>, l: vec3<f32>) -> f32 {
    if !view_has(view, FEATURE_SHADOWS_CSM) {
        return 1.0;
    }
    let depth = view_depth(world_pos);
    let count = u32(shadows.csm_params.w);
    var cascade = count;
    for (var i = 0u; i < count; i = i + 1u) {
        if depth <= shadows.cascade_splits[i] {
            cascade = i;
            break;
        }
    }
    if cascade >= count {
        return 1.0;
    }
    let n_dot_l = clamp(dot(n, l), 0.0, 1.0);
    let normal_offset = n * shadows.csm_params.z * (1.0 - n_dot_l) * (f32(cascade) + 1.0);
    let clip = shadows.cascades[cascade] * vec4<f32>(world_pos + normal_offset, 1.0);
    let ndc = clip.xyz / clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) || ndc.z > 1.0 {
        return 1.0;
    }
    return pcf_csm(i32(cascade), uv, ndc.z - shadows.csm_params.y);
}

fn point_face(dir: vec3<f32>) -> u32 {
    let a = abs(dir);
    if a.x >= a.y && a.x >= a.z {
        return select(1u, 0u, dir.x > 0.0);
    }
    if a.y >= a.z {
        return select(3u, 2u, dir.y > 0.0);
    }
    return select(5u, 4u, dir.z > 0.0);
}

fn local_shadow(light: GpuLight, world_pos: vec3<f32>, n: vec3<f32>) -> f32 {
    if !view_has(view, FEATURE_SHADOWS_LOCAL) || light.params.y == NO_SHADOW {
        return 1.0;
    }
    var matrix_index = light.params.y;
    if light.params.x == LIGHT_POINT {
        matrix_index += point_face(world_pos - light.position_range.xyz);
    }
    let biased = world_pos + n * shadows.local_params.z;
    let clip = shadows.local_matrices[matrix_index] * vec4<f32>(biased, 1.0);
    if clip.w <= 0.0 {
        return 1.0;
    }
    let ndc = clip.xyz / clip.w;
    let local_uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if any(local_uv < vec2<f32>(0.0)) || any(local_uv > vec2<f32>(1.0)) || ndc.z > 1.0 {
        return 1.0;
    }
    let tile = shadows.local_tiles[matrix_index];
    let texel = shadows.local_params.x;
    let border = texel * 1.5;
    let uv = tile.xy + clamp(local_uv, vec2<f32>(border / tile.z), vec2<f32>(1.0 - border / tile.z)) * tile.zw;
    var sum = 0.0;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel;
            sum += textureSampleCompareLevel(
                t_local_shadows,
                s_shadow,
                uv + offset,
                ndc.z - shadows.local_params.y
            );
        }
    }
    return sum / 9.0;
}

fn light_attenuation(light: GpuLight, world_pos: vec3<f32>, l: vec3<f32>) -> f32 {
    let dist = length(light.position_range.xyz - world_pos);
    let range = max(light.position_range.w, 0.001);
    // Smooth window * inverse square (UE4 style), normalized so the
    // authored intensity is the radiance at 1 m.
    let ratio = dist / range;
    let window = clamp(1.0 - ratio * ratio * ratio * ratio, 0.0, 1.0);
    var atten = window * window / (dist * dist + 1.0);
    if light.params.x == LIGHT_SPOT {
        let cos_outer = cos(light.direction_cone.w);
        let cos_inner = cos(bitcast<f32>(light.params.z));
        let cd = dot(-l, normalize(light.direction_cone.xyz));
        atten *= smoothstep(cos_outer, cos_inner, cd);
    }
    return atten;
}

fn box_projected_direction(r: vec3<f32>, world_pos: vec3<f32>, probe: Probe) -> vec3<f32> {
    let box_min = probe.position.xyz - probe.extents_intensity.xyz;
    let box_max = probe.position.xyz + probe.extents_intensity.xyz;
    let first = (box_max - world_pos) / r;
    let second = (box_min - world_pos) / r;
    let furthest = max(first, second);
    let dist = min(min(furthest.x, furthest.y), furthest.z);
    let hit = world_pos + r * dist;
    return hit - probe.position.xyz;
}

fn probe_weight(world_pos: vec3<f32>, probe: Probe) -> f32 {
    let local = abs(world_pos - probe.position.xyz);
    let extents = max(probe.extents_intensity.xyz, vec3<f32>(1e-3));
    let blend = max(bitcast<f32>(probe.params.y), 1e-3);
    let inside = (extents - local) / blend;
    return clamp(min(min(inside.x, inside.y), inside.z), 0.0, 1.0);
}

fn ambient_occlusion(frag_coord: vec2<f32>) -> f32 {
    if !view_has(view, FEATURE_SSAO) {
        return 1.0;
    }
    let uv = frag_coord * view.viewport.zw;
    let ao = textureSampleLevel(t_ssao, s_linear, uv, 0.0).r;
    return mix(1.0, ao, clamp(view.ambient.w, 0.0, 1.0));
}

fn image_based_lighting(
    base_color: vec3<f32>,
    metallic: f32,
    roughness: f32,
    n: vec3<f32>,
    v: vec3<f32>,
    world_pos: vec3<f32>,
) -> vec3<f32> {
    let n_dot_v = max(dot(n, v), 1e-4);
    let f0 = mix(vec3<f32>(0.04), base_color, metallic);
    let f = fresnel_schlick_roughness(n_dot_v, f0, roughness);
    let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
    if !view_has(view, FEATURE_IBL) {
        return kd * base_color * view.ambient.rgb;
    }
    let intensity = probes.ibl_params.x;
    let irradiance = textureSampleLevel(t_irradiance, s_linear, n, 0.0).rgb;
    let diffuse = kd * base_color * irradiance;

    let r = reflect(-v, n);
    let max_mip = max(probes.ibl_params.y - 1.0, 0.0);
    let lod = roughness * max_mip;
    var specular_env = textureSampleLevel(t_prefiltered, s_linear, r, lod).rgb;
    if view_has(view, FEATURE_PROBES) {
        var accumulated = vec3<f32>(0.0);
        var total = 0.0;
        for (var i = 0u; i < min(view.counts.z, 4u); i = i + 1u) {
            let probe = probes.probes[i];
            let w = probe_weight(world_pos, probe) * (1.0 - total);
            if w <= 0.0 {
                continue;
            }
            let dir = box_projected_direction(r, world_pos, probe);
            let sample = textureSampleLevel(t_probes, s_linear, dir, i32(probe.params.x), lod).rgb;
            accumulated += sample * probe.extents_intensity.w * w;
            total += w;
        }
        specular_env = accumulated + specular_env * (1.0 - total);
    }
    let brdf = textureSampleLevel(t_brdf_lut, s_linear, vec2<f32>(n_dot_v, roughness), 0.0).rg;
    let specular = specular_env * (f * brdf.x + brdf.y);
    return (diffuse + specular) * intensity * view.ambient.rgb;
}

fn apply_fog(color: vec3<f32>, world_pos: vec3<f32>) -> vec3<f32> {
    if !view_has(view, FEATURE_FOG) {
        return color;
    }
    let camera = view.camera_position.xyz;
    let to_point = world_pos - camera;
    let distance = max(length(to_point) - view.fog_params.z, 0.0);
    let density = view.fog_color_density.w;
    let falloff = max(view.fog_params.x, 1e-4);
    // Analytic exponential height fog integral along the view ray.
    let height_camera = camera.y - view.fog_params.y;
    let dir_y = to_point.y / max(length(to_point), 1e-4);
    var integral = distance * exp(-falloff * height_camera);
    if abs(dir_y) > 1e-4 {
        let t = falloff * dir_y * distance;
        integral = integral * (1.0 - exp(-t)) / t;
    }
    let fog = clamp(1.0 - exp(-density * integral), 0.0, view.fog_params.w);
    return mix(color, view.fog_color_density.rgb, fog);
}

struct SurfaceInput {
    world_pos: vec3<f32>,
    frag_coord: vec2<f32>,
    n: vec3<f32>,
    v: vec3<f32>,
    base_color: vec3<f32>,
    metallic: f32,
    roughness: f32,
    occlusion: f32,
    emissive: vec3<f32>,
    receive_shadows: bool,
};

fn shade_surface(s: SurfaceInput) -> vec3<f32> {
    var color = vec3<f32>(0.0);
    let dir_count = view.counts.x;
    for (var i = 0u; i < dir_count; i = i + 1u) {
        let light = lights[i];
        let l = normalize(-light.direction_cone.xyz);
        var shadow = 1.0;
        if i == 0u && s.receive_shadows {
            shadow = directional_shadow(s.world_pos, s.n, l);
        }
        let radiance = light.color_intensity.rgb * light.color_intensity.w * shadow;
        color += shade_pbr_light(s.base_color, s.metallic, s.roughness, s.n, s.v, l, radiance);
    }

    let cluster = cluster_ranges[cluster_index(s.frag_coord, view_depth(s.world_pos))];
    for (var j = 0u; j < cluster.y; j = j + 1u) {
        let light = lights[cluster_indices[cluster.x + j]];
        let to_light = light.position_range.xyz - s.world_pos;
        let dist = length(to_light);
        if dist > light.position_range.w {
            continue;
        }
        let l = to_light / max(dist, 1e-5);
        var shadow = 1.0;
        if s.receive_shadows {
            shadow = local_shadow(light, s.world_pos, s.n);
        }
        let radiance = light.color_intensity.rgb * light.color_intensity.w
            * light_attenuation(light, s.world_pos, l) * shadow;
        color += shade_pbr_light(s.base_color, s.metallic, s.roughness, s.n, s.v, l, radiance);
    }

    let ao = ambient_occlusion(s.frag_coord) * s.occlusion;
    color += image_based_lighting(s.base_color, s.metallic, s.roughness, s.n, s.v, s.world_pos) * ao;
    color += s.emissive;
    return apply_fog(color, s.world_pos);
}
