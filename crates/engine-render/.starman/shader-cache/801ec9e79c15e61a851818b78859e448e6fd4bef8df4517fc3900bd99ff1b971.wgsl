// GGX-prefiltered specular environment for one face and one roughness
// (mip level), with PDF-based source-mip selection to avoid fireflies.

// Procedural sky model shared by the skybox (when no HDR environment is
// loaded) and the environment cubemap bake.

struct SkyUniform {
    // rgb zenith color (linear), intensity.
    zenith: vec4<f32>,
    // rgb horizon color, horizon sharpness.
    horizon: vec4<f32>,
    // rgb ground color, unused.
    ground: vec4<f32>,
    // xyz direction *towards* the sun, sun angular radius (radians).
    sun_direction: vec4<f32>,
    // rgb sun color * intensity, sun glow falloff.
    sun_color: vec4<f32>,
    // x: mode (0 procedural, 1 hdr), y: rotation (radians), z: hdr intensity.
    params: vec4<f32>,
};

fn procedural_sky(dir_in: vec3<f32>, sky: SkyUniform) -> vec3<f32> {
    let dir = normalize(dir_in);
    let up = dir.y;
    let sharp = max(sky.horizon.w, 0.1);
    var color: vec3<f32>;
    if up >= 0.0 {
        let t = pow(1.0 - up, sharp);
        color = mix(sky.zenith.rgb, sky.horizon.rgb, t);
    } else {
        let t = pow(1.0 + up, sharp * 4.0);
        color = mix(sky.ground.rgb, sky.horizon.rgb, t);
    }
    let sun_dir = normalize(sky.sun_direction.xyz);
    let cos_angle = dot(dir, sun_dir);
    let radius = max(sky.sun_direction.w, 1e-3);
    let disk = smoothstep(cos(radius * 1.2), cos(radius), cos_angle);
    let glow = pow(max(cos_angle, 0.0), max(sky.sun_color.w, 1.0));
    color += sky.sun_color.rgb * (disk * 20.0 + glow * 0.25);
    return color * sky.zenith.w;
}

fn rotate_y(dir: vec3<f32>, angle: f32) -> vec3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec3<f32>(c * dir.x + s * dir.z, dir.y, -s * dir.x + c * dir.z);
}

fn direction_to_equirect(dir: vec3<f32>) -> vec2<f32> {
    let d = normalize(dir);
    let u = atan2(d.z, d.x) / (2.0 * 3.14159265) + 0.5;
    let v = acos(clamp(d.y, -1.0, 1.0)) / 3.14159265;
    return vec2<f32>(u, v);
}

// Direction through a cube face texel. `face` follows the wgpu/D3D cube
// layout (+X, -X, +Y, -Y, +Z, -Z); `uv` in [0, 1].
fn cube_direction(face: u32, uv: vec2<f32>) -> vec3<f32> {
    let st = uv * 2.0 - 1.0;
    switch face {
        case 0u: { return normalize(vec3<f32>(1.0, -st.y, -st.x)); }
        case 1u: { return normalize(vec3<f32>(-1.0, -st.y, st.x)); }
        case 2u: { return normalize(vec3<f32>(st.x, 1.0, st.y)); }
        case 3u: { return normalize(vec3<f32>(st.x, -1.0, -st.y)); }
        case 4u: { return normalize(vec3<f32>(st.x, -st.y, 1.0)); }
        default: { return normalize(vec3<f32>(-st.x, -st.y, -1.0)); }
    }
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

struct PrefilterUniform {
    // x: face, y: sample count.
    face: vec4<u32>,
    // x: roughness, y: source resolution.
    params: vec4<f32>,
};

@group(0) @binding(0) var t_source: texture_cube<f32>;
@group(0) @binding(1) var s_source: sampler;
@group(0) @binding(2) var<uniform> prefilter: PrefilterUniform;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> FullscreenOut {
    return fullscreen_triangle(index);
}

@fragment
fn fs_main(input: FullscreenOut) -> @location(0) vec4<f32> {
    let n = cube_direction(prefilter.face.x, input.uv);
    let v = n;
    let roughness = prefilter.params.x;
    if roughness < 0.01 {
        return vec4<f32>(textureSampleLevel(t_source, s_source, n, 0.0).rgb, 1.0);
    }
    let count = prefilter.face.y;
    let resolution = prefilter.params.y;
    var color = vec3<f32>(0.0);
    var weight = 0.0;
    for (var i = 0u; i < count; i = i + 1u) {
        let xi = hammersley(i, count);
        let h = importance_sample_ggx(xi, n, roughness);
        let l = normalize(2.0 * dot(v, h) * h - v);
        let n_dot_l = dot(n, l);
        if n_dot_l > 0.0 {
            let n_dot_h = max(dot(n, h), 0.0);
            let d = distribution_ggx(n_dot_h, roughness);
            let pdf = d * n_dot_h / (4.0 * n_dot_h) + 1e-4;
            let sa_texel = 4.0 * PI / (6.0 * resolution * resolution);
            let sa_sample = 1.0 / (f32(count) * pdf + 1e-4);
            let mip = select(0.5 * log2(sa_sample / sa_texel), 0.0, roughness == 0.0);
            color += textureSampleLevel(t_source, s_source, l, max(mip, 0.0)).rgb * n_dot_l;
            weight += n_dot_l;
        }
    }
    return vec4<f32>(color / max(weight, 1e-4), 1.0);
}
