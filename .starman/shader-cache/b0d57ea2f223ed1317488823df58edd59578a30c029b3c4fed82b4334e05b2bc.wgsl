// Renders the source environment (procedural sky or equirect HDR) into
// one cube face. Face index and uv come from the full-screen triangle.

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
