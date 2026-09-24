// Builds the next mip of a cube face by filtering the previous level.

#include "sky/sky_common.wgsl"
#include "common/fullscreen.wgsl"

struct FaceUniform {
    // x: face, y: source mip.
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
    let size = f32(textureDimensions(t_source, face.face.y).x);
    let texel = 0.5 / size;
    let lod = f32(face.face.y);
    var color = vec3<f32>(0.0);
    for (var y = -1; y <= 1; y = y + 2) {
        for (var x = -1; x <= 1; x = x + 2) {
            let uv = input.uv + vec2<f32>(f32(x), f32(y)) * texel;
            color += textureSampleLevel(t_source, s_source, cube_direction(face.face.x, uv), lod).rgb;
        }
    }
    return vec4<f32>(color * 0.25, 1.0);
}
