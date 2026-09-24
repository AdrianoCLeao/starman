// Particle rendering: billboards, velocity-stretched quads or instanced
// meshes, with flipbooks and soft depth fade against the scene depth.

#include "common/view.wgsl"
#include "vfx/particles_common.wgsl"

@group(0) @binding(0) var<uniform> view: ViewUniform;

@group(1) @binding(0) var<uniform> emitter: Emitter;
@group(1) @binding(1) var<storage, read> particles: array<Particle>;
@group(1) @binding(2) var<storage, read> order: array<u32>;
@group(1) @binding(3) var<storage, read> keys: array<vec2<u32>>;
@group(1) @binding(4) var scene_depth: texture_depth_2d;
@group(1) @binding(5) var particle_texture: texture_2d<f32>;
@group(1) @binding(6) var particle_sampler: sampler;

fn over_life_color(t: f32) -> vec4<f32> {
    let x = lut_position(t);
    let i = u32(x.x);
    return mix(emitter.color_lut[i], emitter.color_lut[i + 1u], x.y);
}

fn over_life_size(t: f32) -> f32 {
    let x = lut_position(t);
    let i = u32(x.x);
    return mix(lane(emitter.size_lut[i / 4u], i % 4u), lane(emitter.size_lut[(i + 1u) / 4u], (i + 1u) % 4u), x.y);
}

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) view_depth: f32,
    @location(3) @interpolate(flat) frame: u32,
};

fn particle_index(instance: u32) -> u32 {
    if (emitter.draw.z == 1u) {
        return keys[instance].y;
    }
    return order[(1u - emitter.counts.w) * emitter.counts.y + instance];
}

fn to_world(p: vec3<f32>) -> vec3<f32> {
    if (emitter.time.z > 0.5) {
        return (emitter.transform * vec4<f32>(p, 1.0)).xyz;
    }
    return p;
}

fn to_world_dir(v: vec3<f32>) -> vec3<f32> {
    if (emitter.time.z > 0.5) {
        return (emitter.transform * vec4<f32>(v, 0.0)).xyz;
    }
    return v;
}

fn common_out(p: Particle, world: vec3<f32>, uv: vec2<f32>) -> VertexOut {
    var out: VertexOut;
    let life = p.position_age.w / p.velocity_lifetime.w;
    out.clip = view.view_proj * vec4<f32>(world, 1.0);
    out.color = p.color * over_life_color(life);
    out.uv = uv;
    out.view_depth = dot(world - view.camera_position.xyz, -vec3<f32>(view.view[0].z, view.view[1].z, view.view[2].z));
    let frames = max(u32(emitter.flipbook.x * emitter.flipbook.y), 1u);
    var frame = 0u;
    if (emitter.flipbook.w > 0.5) {
        if (emitter.flipbook.z > 0.0) {
            frame = u32(p.position_age.w * emitter.flipbook.z) % frames;
        } else {
            frame = min(u32(life * f32(frames)), frames - 1u);
        }
    }
    out.frame = frame;
    return out;
}

@vertex
fn vs_quad(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> VertexOut {
    let p = particles[particle_index(instance)];
    let life = p.position_age.w / p.velocity_lifetime.w;
    let size = p.size_rotation.x * over_life_size(life);
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let corner = corners[vertex % 6u];
    let center = to_world(p.position_age.xyz);
    let camera_right = vec3<f32>(view.view[0].x, view.view[1].x, view.view[2].x);
    let camera_up = vec3<f32>(view.view[0].y, view.view[1].y, view.view[2].y);
    var offset: vec3<f32>;
    if (emitter.render.x > 0.5) {
        // Stretched along velocity.
        let velocity = to_world_dir(p.velocity_lifetime.xyz);
        let speed = length(velocity);
        let axis = select(camera_up, velocity / max(speed, 1e-5), speed > 1e-4);
        let to_camera = normalize(view.camera_position.xyz - center);
        let side = normalize(cross(axis, to_camera) + vec3<f32>(1e-6, 0.0, 0.0));
        let stretch_length = size + speed * emitter.render.y;
        offset = side * corner.x * size * 0.5 + axis * corner.y * stretch_length * 0.5;
    } else {
        let angle = p.size_rotation.y;
        let c = cos(angle);
        let s = sin(angle);
        let rotated = vec2<f32>(corner.x * c - corner.y * s, corner.x * s + corner.y * c);
        offset = (camera_right * rotated.x + camera_up * rotated.y) * size * 0.5;
    }
    return common_out(p, center + offset, corner * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5));
}

struct MeshVertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
};

@vertex
fn vs_mesh(vertex: MeshVertex, @builtin(instance_index) instance: u32) -> VertexOut {
    let p = particles[particle_index(instance)];
    let life = p.position_age.w / p.velocity_lifetime.w;
    let size = p.size_rotation.x * over_life_size(life);
    let angle = p.size_rotation.y;
    let c = cos(angle);
    let s = sin(angle);
    let local = vec3<f32>(vertex.position.x * c + vertex.position.z * s, vertex.position.y, -vertex.position.x * s + vertex.position.z * c) * size;
    var out = common_out(p, to_world(p.position_age.xyz) + to_world_dir(local), vertex.uv);
    let n = normalize(vec3<f32>(vertex.normal.x * c + vertex.normal.z * s, vertex.normal.y, -vertex.normal.x * s + vertex.normal.z * c));
    out.color = vec4<f32>(out.color.rgb * (0.55 + 0.45 * max(n.y, 0.0)), out.color.a);
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // Manual depth test and soft fade against the scene depth.
    let pixel = vec2<i32>(in.clip.xy);
    let depth = textureLoad(scene_depth, pixel, 0);
    let uv = (vec2<f32>(pixel) + vec2<f32>(0.5)) * view.viewport.zw;
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let world = view.inv_view_proj * ndc;
    let scene_point = world.xyz / world.w;
    let forward = -vec3<f32>(view.view[0].z, view.view[1].z, view.view[2].z);
    let scene_depth_view = dot(scene_point - view.camera_position.xyz, forward);
    if (depth < 1.0 && in.view_depth > scene_depth_view) {
        discard;
    }
    var fade = 1.0;
    if (emitter.time.w > 0.0 && depth < 1.0) {
        fade = clamp((scene_depth_view - in.view_depth) / emitter.time.w, 0.0, 1.0);
    }

    var sample_uv = in.uv;
    if (emitter.flipbook.w > 0.5) {
        let columns = max(emitter.flipbook.x, 1.0);
        let rows = max(emitter.flipbook.y, 1.0);
        let column = f32(in.frame % u32(columns));
        let row = f32(in.frame / u32(columns));
        sample_uv = (in.uv + vec2<f32>(column, row)) / vec2<f32>(columns, rows);
    }
    var texel = vec4<f32>(1.0);
    if (emitter.render.w > 0.5) {
        texel = textureSample(particle_texture, particle_sampler, sample_uv);
    } else if (emitter.render.x < 1.5) {
        // Untextured quads are soft discs.
        let d = length(in.uv * 2.0 - vec2<f32>(1.0));
        texel = vec4<f32>(1.0, 1.0, 1.0, clamp(1.0 - d, 0.0, 1.0));
    }
    let color = in.color * texel;
    let alpha = color.a * fade;
    let blend = u32(emitter.render.z);
    if (blend == 0u) {
        // Additive.
        return vec4<f32>(color.rgb * alpha, 0.0);
    }
    if (blend == 2u) {
        // Premultiplied.
        return vec4<f32>(color.rgb * fade, alpha);
    }
    return vec4<f32>(color.rgb, alpha);
}
