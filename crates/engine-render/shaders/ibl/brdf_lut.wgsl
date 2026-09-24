// Split-sum environment BRDF lookup table (Karis 2013).

#include "common/brdf.wgsl"
#include "common/fullscreen.wgsl"

fn geometry_schlick_ggx_ibl(n_dot_x: f32, roughness: f32) -> f32 {
    let k = (roughness * roughness) / 2.0;
    return n_dot_x / (n_dot_x * (1.0 - k) + k);
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> FullscreenOut {
    return fullscreen_triangle(index);
}

@fragment
fn fs_main(input: FullscreenOut) -> @location(0) vec4<f32> {
    let n_dot_v = max(input.uv.x, 1e-3);
    let roughness = input.uv.y;
    let v = vec3<f32>(sqrt(1.0 - n_dot_v * n_dot_v), 0.0, n_dot_v);
    let n = vec3<f32>(0.0, 0.0, 1.0);
    var a = 0.0;
    var b = 0.0;
    let count = 256u;
    for (var i = 0u; i < count; i = i + 1u) {
        let xi = hammersley(i, count);
        let h = importance_sample_ggx(xi, n, roughness);
        let l = normalize(2.0 * dot(v, h) * h - v);
        let n_dot_l = max(l.z, 0.0);
        let n_dot_h = max(h.z, 0.0);
        let v_dot_h = max(dot(v, h), 0.0);
        if n_dot_l > 0.0 {
            let g = geometry_schlick_ggx_ibl(n_dot_v, roughness)
                * geometry_schlick_ggx_ibl(n_dot_l, roughness);
            let g_vis = (g * v_dot_h) / max(n_dot_h * n_dot_v, 1e-4);
            let fc = pow(1.0 - v_dot_h, 5.0);
            a += (1.0 - fc) * g_vis;
            b += fc * g_vis;
        }
    }
    return vec4<f32>(a / f32(count), b / f32(count), 0.0, 1.0);
}
