// Mesh shader for every mesh pass. Variants (defines):
//   PREPASS     depth + velocity (Rg16Float) pre-pass
//   SHADOW      depth-only shadow pass (light view in `view`)
//   LIT         full PBR shading (opaque and transparent)
//   PICKING     entity pick ids (R32Uint)
//   OVERDRAW    additive constant for the overdraw debug view
//   SKINNED     linear blend skinning from the joint palette
//   ALPHA_MASK  alpha-tested materials discard below the cutoff

// Lit-pass group 0 bindings and the full lighting model: clustered local
// lights, directional lights, cascaded + local shadows, IBL with blended
// box-projected reflection probes, SSAO and height fog.

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
// Cook-Torrance microfacet BRDF (GGX / Smith-Schlick / Schlick Fresnel).

const PI: f32 = 3.14159265;

fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d + 1e-7);
}

fn geometry_schlick_ggx(n_dot_x: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return n_dot_x / (n_dot_x * (1.0 - k) + k + 1e-7);
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    return geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn fresnel_schlick_roughness(cos_theta: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
    return f0 + (max(vec3<f32>(1.0 - roughness), f0) - f0)
        * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn shade_pbr_light(
    base_color: vec3<f32>,
    metallic: f32,
    roughness: f32,
    n: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>,
    radiance: vec3<f32>,
) -> vec3<f32> {
    let h = normalize(v + l);
    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_v = max(dot(n, v), 1e-4);
    let n_dot_h = max(dot(n, h), 0.0);
    let h_dot_v = max(dot(h, v), 0.0);

    let f0 = mix(vec3<f32>(0.04), base_color, metallic);
    let d = distribution_ggx(n_dot_h, roughness);
    let g = geometry_smith(n_dot_v, n_dot_l, roughness);
    let f = fresnel_schlick(h_dot_v, f0);
    let specular = (d * g * f) / max(4.0 * n_dot_v * n_dot_l, 1e-4);
    let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
    let diffuse = kd * base_color / PI;
    return (diffuse + specular) * radiance * n_dot_l;
}

// Hammersley point set and GGX importance sampling (IBL precomputation).
fn radical_inverse_vdc(bits_in: u32) -> f32 {
    var bits = bits_in;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10;
}

fn hammersley(i: u32, n: u32) -> vec2<f32> {
    return vec2<f32>(f32(i) / f32(n), radical_inverse_vdc(i));
}

fn importance_sample_ggx(xi: vec2<f32>, n: vec3<f32>, roughness: f32) -> vec3<f32> {
    let a = roughness * roughness;
    let phi = 2.0 * PI * xi.x;
    let cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    let h = vec3<f32>(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
    let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(n.z) < 0.999);
    let tangent = normalize(cross(up, n));
    let bitangent = cross(n, tangent);
    return normalize(tangent * h.x + bitangent * h.y + n * h.z);
}

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
// Per-object data (group 1). Passes draw through an indirection list so
// each pass (camera, cascade, probe face) can cull independently.

struct Instance {
    model: mat4x4<f32>,
    prev_model: mat4x4<f32>,
    // Inverse-transpose of the model's upper 3x3, as three columns.
    normal0: vec4<f32>,
    normal1: vec4<f32>,
    normal2: vec4<f32>,
    // x: palette offset, y: previous palette offset, z: pick id, w: flags.
    params: vec4<u32>,
};

const INSTANCE_FLAG_SKINNED: u32 = 1u;
const INSTANCE_FLAG_RECEIVE_SHADOWS: u32 = 2u;
const INSTANCE_FLAG_SELECTED: u32 = 4u;

@group(1) @binding(0) var<storage, read> instances: array<Instance>;
@group(1) @binding(1) var<storage, read> palettes: array<mat4x4<f32>>;
@group(1) @binding(2) var<storage, read> draw_indirection: array<u32>;

fn instance_normal_matrix(inst: Instance) -> mat3x3<f32> {
    return mat3x3<f32>(inst.normal0.xyz, inst.normal1.xyz, inst.normal2.xyz);
}
// Material bindings (group 2): glTF metallic-roughness.

struct MaterialUniform {
    base_color: vec4<f32>,
    // metallic, roughness, normal scale, occlusion strength.
    metallic_roughness: vec4<f32>,
    // rgb emissive, alpha cutoff.
    emissive: vec4<f32>,
    // x: alpha mode (0 opaque, 1 mask, 2 blend), y: double sided,
    // z: texture presence bits, w: unused.
    flags: vec4<u32>,
};

const MAT_TEX_BASE: u32 = 1u;
const MAT_TEX_MR: u32 = 2u;
const MAT_TEX_NORMAL: u32 = 4u;
const MAT_TEX_OCCLUSION: u32 = 8u;
const MAT_TEX_EMISSIVE: u32 = 16u;

@group(2) @binding(0) var<uniform> material: MaterialUniform;
@group(2) @binding(1) var t_base_color: texture_2d<f32>;
@group(2) @binding(2) var t_metallic_roughness: texture_2d<f32>;
@group(2) @binding(3) var t_normal: texture_2d<f32>;
@group(2) @binding(4) var t_occlusion: texture_2d<f32>;
@group(2) @binding(5) var t_emissive: texture_2d<f32>;
@group(2) @binding(6) var s_material: sampler;

fn material_base_color(uv: vec2<f32>) -> vec4<f32> {
    return textureSample(t_base_color, s_material, uv) * material.base_color;
}

struct VertexInput {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) @invariant clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) world_tangent: vec4<f32>,
    @location(4) current_clip: vec4<f32>,
    @location(5) previous_clip: vec4<f32>,
    @location(6) @interpolate(flat) object_index: u32,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    let object_index = draw_indirection[input.instance_index];
    let inst = instances[object_index];

    var local_position = vec4<f32>(input.position, 1.0);
    var previous_local = local_position;
    var local_normal = input.normal;
    var local_tangent = input.tangent.xyz;

    let world = inst.model * local_position;
    var out: VertexOutput;
    out.clip_position = view.view_proj * world;
    out.world_position = world.xyz;
    let normal_matrix = instance_normal_matrix(inst);
    out.world_normal = normalize(normal_matrix * local_normal);
    out.world_tangent = vec4<f32>(
        normalize((inst.model * vec4<f32>(local_tangent, 0.0)).xyz),
        input.tangent.w
    );
    out.uv = input.uv;
    out.current_clip = view.unjittered_view_proj * world;
    out.previous_clip = view.prev_view_proj * (inst.prev_model * previous_local);
    out.object_index = object_index;
    return out;
}

fn alpha_test(uv: vec2<f32>) {
    if material_base_color(uv).a < material.emissive.w {
        discard;
    }
}





fn perturbed_normal(input: VertexOutput, facing: bool) -> vec3<f32> {
    var n = normalize(input.world_normal);
    if material.flags.y == 1u && !facing {
        n = -n;
    }
    if (material.flags.z & MAT_TEX_NORMAL) == 0u {
        return n;
    }
    let t = normalize(input.world_tangent.xyz - n * dot(n, input.world_tangent.xyz));
    let b = cross(n, t) * input.world_tangent.w;
    var tn = textureSample(t_normal, s_material, input.uv).xyz * 2.0 - 1.0;
    tn = vec3<f32>(tn.xy * material.metallic_roughness.z, tn.z);
    return normalize(mat3x3<f32>(t, b, n) * tn);
}

@fragment
fn fs_main(input: VertexOutput, @builtin(front_facing) facing: bool) -> @location(0) vec4<f32> {
    let base = material_base_color(input.uv);
    if base.a < material.emissive.w {
        discard;
    }
    let mr = textureSample(t_metallic_roughness, s_material, input.uv);
    let metallic = clamp(material.metallic_roughness.x * mr.b, 0.0, 1.0);
    let roughness = clamp(material.metallic_roughness.y * mr.g, 0.04, 1.0);
    let occlusion_sample = textureSample(t_occlusion, s_material, input.uv).r;
    let occlusion = mix(1.0, occlusion_sample, material.metallic_roughness.w);
    let emissive = material.emissive.rgb * textureSample(t_emissive, s_material, input.uv).rgb;
    let inst = instances[input.object_index];

    var s: SurfaceInput;
    s.world_pos = input.world_position;
    s.frag_coord = input.clip_position.xy;
    s.n = perturbed_normal(input, facing);
    s.v = normalize(view.camera_position.xyz - input.world_position);
    s.base_color = base.rgb;
    s.metallic = metallic;
    s.roughness = roughness;
    s.occlusion = occlusion;
    s.emissive = emissive;
    s.receive_shadows = (inst.params.w & INSTANCE_FLAG_RECEIVE_SHADOWS) != 0u;
    var color = shade_surface(s);
    if (inst.params.w & INSTANCE_FLAG_SELECTED) != 0u {
        let rim = pow(1.0 - max(dot(s.n, s.v), 0.0), 3.0);
        color += vec3<f32>(1.0, 0.55, 0.1) * rim * 2.0;
    }
    var alpha = 1.0;
    if material.flags.x == 2u {
        alpha = base.a;
    }
    return vec4<f32>(color, alpha);
}
