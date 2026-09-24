// Diffuse irradiance convolution of the environment cubemap (one face).

#include "sky/sky_common.wgsl"
#include "common/fullscreen.wgsl"

struct FaceUniform {
    face: vec4<u32>,
};

@group(0) @binding(0) var t_source: texture_cube<f32>;
@group(0) @binding(1) var s_source: sampler;
@group(0) @binding(2) var<uniform> face: FaceUniform;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> FullscreenOut {
    return fullscreen_triangle(index);
}

@fragment
fn fs_main(input: FullscreenOut) -> @location(0) vec4<f32> {
    let n = cube_direction(face.face.x, input.uv);
    let up0 = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.y) < 0.999);
    let right = normalize(cross(up0, n));
    let up = cross(n, right);
    var irradiance = vec3<f32>(0.0);
    var samples = 0.0;
    let delta = 0.05;
    for (var phi = 0.0; phi < 6.2831853; phi = phi + delta * 2.0) {
        for (var theta = 0.0; theta < 1.5707963; theta = theta + delta) {
            let tangent = vec3<f32>(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
            let dir = tangent.x * right + tangent.y * up + tangent.z * n;
            irradiance += textureSampleLevel(t_source, s_source, dir, 2.0).rgb
                * cos(theta) * sin(theta);
            samples += 1.0;
        }
    }
    return vec4<f32>(3.14159265 * irradiance / samples, 1.0);
}
