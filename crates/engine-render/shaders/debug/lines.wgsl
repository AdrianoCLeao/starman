// Immediate-mode debug lines (physics, navigation, AI, gizmos). Colors
// are authored in sRGB and converted to linear for sRGB targets.

#include "common/view.wgsl"
#include "common/color.wgsl"

@group(0) @binding(0) var<uniform> view: ViewUniform;

struct LineVertex {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
};

struct LineOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(input: LineVertex) -> LineOut {
    var out: LineOut;
    out.position = view.unjittered_view_proj * vec4<f32>(input.position, 1.0);
    // Pull lines slightly towards the camera so coplanar edges win.
    out.position.z = out.position.z - 1e-4 * out.position.w;
    out.color = input.color;
    return out;
}

@fragment
fn fs_main(input: LineOut) -> @location(0) vec4<f32> {
#ifdef OUTPUT_SRGB_ENCODE
    return input.color;
#else
    return vec4<f32>(srgb_to_linear(input.color.rgb), input.color.a);
#endif
}
