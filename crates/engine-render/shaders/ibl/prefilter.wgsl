// GGX-prefiltered specular environment for one face and one roughness
// (mip level), with PDF-based source-mip selection to avoid fireflies.

#include "sky/sky_common.wgsl"
#include "common/brdf.wgsl"
#include "common/fullscreen.wgsl"

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
