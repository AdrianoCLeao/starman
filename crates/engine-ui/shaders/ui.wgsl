// Game UI quads: SDF rounded boxes with borders, images and glyph
// coverage. Colors are authored in sRGB; blending happens in linear space
// and the output is encoded for the target format.

struct Screen {
    // width, height of the target in pixels
    size: vec2<f32>,
    // glyph atlas size in pixels
    atlas_size: vec2<f32>,
    // x: 1 when the target encodes sRGB itself
    flags: vec4<u32>,
};

@group(0) @binding(0) var<uniform> screen: Screen;
@group(1) @binding(0) var quad_texture: texture_2d<f32>;
@group(1) @binding(1) var quad_sampler: sampler;

struct QuadIn {
    @location(0) rect: vec4<f32>,
    @location(1) uv: vec4<f32>,
    @location(2) color: vec4<f32>,
    @location(3) border_color: vec4<f32>,
    // corner radius, border width, mode (0 solid, 1 image, 2 glyph), unused
    @location(4) params: vec4<f32>,
};

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) size: vec2<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
    @location(4) border_color: vec4<f32>,
    @location(5) params: vec4<f32>,
};

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let low = c / 12.92;
    let high = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(high, low, c <= vec3<f32>(0.04045));
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let low = c * 12.92;
    let high = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    return select(high, low, c <= vec3<f32>(0.0031308));
}

@vertex
fn vs_main(@builtin(vertex_index) vertex: u32, quad: QuadIn) -> VertexOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
    );
    let corner = corners[vertex % 6u];
    let pixel = quad.rect.xy + corner * quad.rect.zw;
    var out: VertexOut;
    let ndc = pixel / screen.size * 2.0 - vec2<f32>(1.0);
    out.clip = vec4<f32>(ndc.x, -ndc.y, 0.0, 1.0);
    out.local = corner * quad.rect.zw;
    out.size = quad.rect.zw;
    var uv = mix(quad.uv.xy, quad.uv.zw, corner);
    if (u32(quad.params.z) == 2u) {
        uv = uv / max(screen.atlas_size, vec2<f32>(1.0));
    }
    out.uv = uv;
    out.color = vec4<f32>(srgb_to_linear(quad.color.rgb), quad.color.a);
    out.border_color = vec4<f32>(srgb_to_linear(quad.border_color.rgb), quad.border_color.a);
    out.params = quad.params;
    return out;
}

fn rounded_box(p: vec2<f32>, half_size: vec2<f32>, radius: f32) -> f32 {
    let q = abs(p) - half_size + vec2<f32>(radius);
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - radius;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let mode = u32(in.params.z);
    var color: vec4<f32>;
    if (mode == 2u) {
        let coverage = textureSample(quad_texture, quad_sampler, in.uv).r;
        color = vec4<f32>(in.color.rgb, in.color.a * coverage);
    } else if (mode == 1u) {
        color = textureSample(quad_texture, quad_sampler, in.uv) * in.color;
    } else {
        let half_size = in.size * 0.5;
        let radius = min(in.params.x, min(half_size.x, half_size.y));
        let d = rounded_box(in.local - half_size, half_size, radius);
        let outside = clamp(d + 0.5, 0.0, 1.0);
        let border = in.params.y;
        var fill = in.color;
        if (border > 0.0) {
            // Border band: between the outer edge and `border` inwards.
            let inner = clamp(d + border + 0.5, 0.0, 1.0);
            fill = mix(in.color, in.border_color, inner);
        }
        color = vec4<f32>(fill.rgb, fill.a * (1.0 - outside));
    }
    if (screen.flags.x == 0u) {
        color = vec4<f32>(linear_to_srgb(max(color.rgb, vec3<f32>(0.0))), color.a);
    }
    return color;
}
