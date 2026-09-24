// Renders the source environment (procedural sky or equirect HDR) into
// one cube face. Face index and uv come from the full-screen triangle.

#include "sky/sky_common.wgsl"
#include "common/fullscreen.wgsl"

struct FaceUniform {
    // x: face index.
    face: vec4<u32>,
};

@group(0) @binding(0) var<uniform> sky: SkyUniform;
@group(0) @binding(1) var<uniform> face: FaceUniform;
@group(0) @binding(2) var t_equirect: texture_2d<f32>;
@group(0) @binding(3) var s_equirect: sampler;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> FullscreenOut {
    return fullscreen_triangle(index);
}

@fragment
fn fs_main(input: FullscreenOut) -> @location(0) vec4<f32> {
    let dir = cube_direction(face.face.x, input.uv);
    if sky.params.x > 0.5 {
        let rotated = rotate_y(dir, sky.params.y);
        let uv = direction_to_equirect(rotated);
        let color = textureSampleLevel(t_equirect, s_equirect, uv, 0.0).rgb * sky.params.z;
        return vec4<f32>(color, 1.0);
    }
    return vec4<f32>(procedural_sky(dir, sky), 1.0);
}
