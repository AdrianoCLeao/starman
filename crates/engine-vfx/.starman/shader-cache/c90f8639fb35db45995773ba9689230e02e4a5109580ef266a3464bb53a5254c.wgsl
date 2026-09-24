// GPU particle simulation: reset, begin, spawn, update, finalize and the
// depth-buffer collision. Alive lists are double buffered (source list
// `counts.w`), the dead list is a stack.

// Shared particle data and helpers. Random numbers and noise mirror
// `engine_vfx::sim` so both backends behave alike.

struct Particle {
    // xyz position, w age
    position_age: vec4<f32>,
    // xyz velocity, w lifetime
    velocity_lifetime: vec4<f32>,
    color: vec4<f32>,
    // size, rotation, angular velocity, seed bits
    size_rotation: vec4<f32>,
};

struct Emitter {
    transform: mat4x4<f32>,
    inv_transform: mat4x4<f32>,
    // x spawn count, y capacity, z seed base, w source list (0/1)
    counts: vec4<u32>,
    // x shape (0 point, 1 sphere, 2 cone, 3 box, 4 mesh), y radius/angle, z surface flag/cone radius, w mesh points
    shape: vec4<f32>,
    shape_extents: vec4<f32>,
    // lifetime min/max, speed min/max
    lifetime_speed: vec4<f32>,
    // size min/max, rotation min/max
    size_rotation: vec4<f32>,
    // angular min/max, direction mode (0 shape, 1 fixed), 0
    angular: vec4<f32>,
    direction: vec4<f32>,
    color: vec4<f32>,
    // gravity (scaled, simulation space) xyz, drag
    gravity_drag: vec4<f32>,
    // constant acceleration xyz, has speed curve
    accel: vec4<f32>,
    // strength, frequency, scroll speed, enabled
    noise: vec4<f32>,
    // enabled, bounce, friction, kill
    collision: vec4<f32>,
    // dt, time, local space, soft distance
    time: vec4<f32>,
    // mode (0 billboard, 1 stretched, 2 mesh), stretch scale, blend, textured
    render: vec4<f32>,
    // columns, rows, fps, enabled
    flipbook: vec4<f32>,
    // x vertex/index count, y indexed, z sorted, w sort size
    draw: vec4<u32>,
    color_lut: array<vec4<f32>, 16>,
    size_lut: array<vec4<f32>, 4>,
    speed_lut: array<vec4<f32>, 4>,
};

fn pcg(value: u32) -> u32 {
    let state = value * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn random(seed: u32, channel: u32) -> f32 {
    return f32(pcg(seed ^ pcg(channel + 0x9E3779B9u)) >> 8u) / 16777216.0;
}

fn lattice(c: vec3<i32>, seed: u32) -> f32 {
    let h = pcg((u32(c.x) * 73856093u) ^ (u32(c.y) * 19349663u) ^ (u32(c.z) * 83492791u) ^ seed);
    return f32(h >> 8u) / 16777216.0 * 2.0 - 1.0;
}

fn value_noise(p: vec3<f32>, seed: u32) -> f32 {
    let i = floor(p);
    let f = p - i;
    let u = f * f * (vec3<f32>(3.0) - 2.0 * f);
    let c = vec3<i32>(i);
    var result = 0.0;
    for (var dz = 0; dz < 2; dz++) {
        for (var dy = 0; dy < 2; dy++) {
            for (var dx = 0; dx < 2; dx++) {
                let w = select(1.0 - u.x, u.x, dx == 1)
                    * select(1.0 - u.y, u.y, dy == 1)
                    * select(1.0 - u.z, u.z, dz == 1);
                result += w * lattice(c + vec3<i32>(dx, dy, dz), seed);
            }
        }
    }
    return result;
}

fn potential(q: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        value_noise(q, 11u),
        value_noise(q + vec3<f32>(31.4, 17.1, 5.9), 23u),
        value_noise(q + vec3<f32>(-9.2, 44.7, 13.3), 37u),
    );
}

fn curl_noise(p: vec3<f32>) -> vec3<f32> {
    let e = 0.01;
    let dx = vec3<f32>(e, 0.0, 0.0);
    let dy = vec3<f32>(0.0, e, 0.0);
    let dz = vec3<f32>(0.0, 0.0, e);
    let px0 = potential(p - dx);
    let px1 = potential(p + dx);
    let py0 = potential(p - dy);
    let py1 = potential(p + dy);
    let pz0 = potential(p - dz);
    let pz1 = potential(p + dz);
    return vec3<f32>(
        (py1.z - py0.z) - (pz1.y - pz0.y),
        (pz1.x - pz0.x) - (px1.z - px0.z),
        (px1.y - px0.y) - (py1.x - py0.x),
    ) / (2.0 * e);
}

fn lut_color(e: Emitter, t: f32) -> vec4<f32> {
    let x = clamp(t, 0.0, 1.0) * 15.0;
    let i = min(u32(floor(x)), 14u);
    let f = x - f32(i);
    return mix(e.color_lut[i], e.color_lut[i + 1u], f);
}

fn lut_scalar(table: array<vec4<f32>, 4>, t: f32) -> f32 {
    let x = clamp(t, 0.0, 1.0) * 15.0;
    let i = min(u32(floor(x)), 14u);
    let f = x - f32(i);
    let a = table[i / 4u][i % 4u];
    let b = table[(i + 1u) / 4u][(i + 1u) % 4u];
    return mix(a, b, f);
}

struct Counters {
    alive: array<atomic<u32>, 2>,
    dead: atomic<i32>,
    pad: u32,
};

struct Scene {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    // xyz camera position, w has depth
    camera: vec4<f32>,
    // width, height, 1/width, 1/height
    viewport: vec4<f32>,
};

@group(0) @binding(0) var<uniform> emitter: Emitter;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> alive: array<u32>;
@group(0) @binding(3) var<storage, read_write> dead: array<u32>;
@group(0) @binding(4) var<storage, read_write> counters: Counters;
@group(0) @binding(5) var<storage, read_write> draw_args: array<u32, 8>;
@group(0) @binding(6) var<storage, read> mesh_points: array<vec4<f32>>;
@group(0) @binding(7) var<uniform> scene: Scene;
@group(0) @binding(8) var scene_depth: texture_depth_2d;

@compute @workgroup_size(64)
fn reset(@builtin(global_invocation_id) id: vec3<u32>) {
    let capacity = emitter.counts.y;
    if (id.x < capacity) {
        dead[id.x] = capacity - 1u - id.x;
    }
    if (id.x == 0u) {
        atomicStore(&counters.alive[0], 0u);
        atomicStore(&counters.alive[1], 0u);
        atomicStore(&counters.dead, i32(capacity));
    }
}

@compute @workgroup_size(1)
fn begin() {
    let dst = 1u - emitter.counts.w;
    atomicStore(&counters.alive[dst], 0u);
}

fn unit_sphere(seed: u32, channel: u32) -> vec3<f32> {
    let z = random(seed, channel) * 2.0 - 1.0;
    let angle = random(seed, channel + 1u) * 6.28318530718;
    let r = sqrt(max(1.0 - z * z, 0.0));
    return vec3<f32>(r * cos(angle), z, r * sin(angle));
}

// Returns local position (xyz) and writes the outward direction.
fn sample_shape(seed: u32, outward: ptr<function, vec3<f32>>) -> vec3<f32> {
    let kind = u32(emitter.shape.x);
    switch (kind) {
        case 1u: {
            let dir = unit_sphere(seed, 10u);
            *outward = dir;
            let distance = select(emitter.shape.y * pow(random(seed, 12u), 1.0 / 3.0), emitter.shape.y, emitter.shape.z > 0.5);
            return dir * distance;
        }
        case 2u: {
            let spin = random(seed, 10u) * 6.28318530718;
            let spread = sqrt(random(seed, 11u));
            let tilt = clamp(emitter.shape.y, 0.0, 3.14159265) * spread;
            *outward = vec3<f32>(sin(tilt) * cos(spin), cos(tilt), sin(tilt) * sin(spin));
            return vec3<f32>(cos(spin), 0.0, sin(spin)) * emitter.shape.z * spread;
        }
        case 3u: {
            *outward = vec3<f32>(0.0, 1.0, 0.0);
            let p = vec3<f32>(random(seed, 10u), random(seed, 11u), random(seed, 12u)) * 2.0 - vec3<f32>(1.0);
            return p * emitter.shape_extents.xyz;
        }
        case 4u: {
            let count = u32(emitter.shape.w);
            if (count == 0u) {
                *outward = vec3<f32>(0.0, 1.0, 0.0);
                return vec3<f32>(0.0);
            }
            let index = u32(random(seed, 10u) * f32(count)) % count;
            *outward = mesh_points[index * 2u + 1u].xyz;
            return mesh_points[index * 2u].xyz;
        }
        default: {
            *outward = unit_sphere(seed, 10u);
            return vec3<f32>(0.0);
        }
    }
}

@compute @workgroup_size(64)
fn spawn(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= emitter.counts.x) {
        return;
    }
    let previous = atomicSub(&counters.dead, 1);
    if (previous <= 0) {
        atomicAdd(&counters.dead, 1);
        return;
    }
    let index = dead[u32(previous - 1)];
    let seed = pcg(emitter.counts.z + id.x);
    var outward = vec3<f32>(0.0, 1.0, 0.0);
    var position = sample_shape(seed, &outward);
    var direction = outward;
    if (emitter.angular.z > 0.5) {
        direction = normalize(emitter.direction.xyz + vec3<f32>(1e-6));
    }
    var velocity = direction * mix(emitter.lifetime_speed.z, emitter.lifetime_speed.w, random(seed, 1u));
    if (emitter.time.z < 0.5) {
        // World-space simulation: spawn in world.
        position = (emitter.transform * vec4<f32>(position, 1.0)).xyz;
        velocity = (emitter.transform * vec4<f32>(velocity, 0.0)).xyz;
    }
    var particle: Particle;
    particle.position_age = vec4<f32>(position, 0.0);
    particle.velocity_lifetime = vec4<f32>(
        velocity,
        max(mix(emitter.lifetime_speed.x, emitter.lifetime_speed.y, random(seed, 2u)), 0.001),
    );
    particle.color = emitter.color;
    particle.size_rotation = vec4<f32>(
        mix(emitter.size_rotation.x, emitter.size_rotation.y, random(seed, 3u)),
        mix(emitter.size_rotation.z, emitter.size_rotation.w, random(seed, 4u)),
        mix(emitter.angular.x, emitter.angular.y, random(seed, 5u)),
        bitcast<f32>(seed),
    );
    particles[index] = particle;
    let src = emitter.counts.w;
    let slot = atomicAdd(&counters.alive[src], 1u);
    alive[src * emitter.counts.y + slot] = index;
}

fn world_position(p: vec3<f32>) -> vec3<f32> {
    if (emitter.time.z > 0.5) {
        return (emitter.transform * vec4<f32>(p, 1.0)).xyz;
    }
    return p;
}

fn depth_at(pixel: vec2<i32>) -> f32 {
    let size = vec2<i32>(textureDimensions(scene_depth));
    return textureLoad(scene_depth, clamp(pixel, vec2<i32>(0), size - vec2<i32>(1)), 0);
}

fn reconstruct(pixel: vec2<i32>, depth: f32) -> vec3<f32> {
    let uv = (vec2<f32>(pixel) + vec2<f32>(0.5)) * scene.viewport.zw;
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let world = scene.inv_view_proj * ndc;
    return world.xyz / world.w;
}

@compute @workgroup_size(64)
fn update(@builtin(global_invocation_id) id: vec3<u32>) {
    let src = emitter.counts.w;
    let dst = 1u - src;
    let capacity = emitter.counts.y;
    if (id.x >= atomicLoad(&counters.alive[src])) {
        return;
    }
    let index = alive[src * capacity + id.x];
    var p = particles[index];
    let dt = emitter.time.x;
    p.position_age.w += dt;
    if (p.position_age.w >= p.velocity_lifetime.w) {
        let slot = atomicAdd(&counters.dead, 1);
        dead[u32(slot)] = index;
        return;
    }
    let life = p.position_age.w / p.velocity_lifetime.w;
    var velocity = p.velocity_lifetime.xyz;
    velocity += emitter.gravity_drag.xyz * dt + emitter.accel.xyz * dt;
    velocity *= max(1.0 - emitter.gravity_drag.w * dt, 0.0);
    if (emitter.noise.w > 0.5) {
        let q = p.position_age.xyz * emitter.noise.y + vec3<f32>(emitter.time.y * emitter.noise.z);
        velocity += q * 0.0;
    }
    var speed_scale = 1.0;
    if (emitter.accel.w > 0.5) {
        speed_scale = life;
    }
    var next = p.position_age.xyz + velocity * speed_scale * dt;

    if (false) {
        let world = world_position(next);
        let clip = scene.view_proj * vec4<f32>(world, 1.0);
        if (clip.w > 0.0) {
            let ndc = clip.xyz / clip.w;
            if (all(abs(ndc.xy) < vec2<f32>(1.0))) {
                let pixel = vec2<i32>(vec2<f32>((ndc.x * 0.5 + 0.5), (0.5 - ndc.y * 0.5)) * scene.viewport.xy);
                let surface_depth = depth_at(pixel);
                let surface = reconstruct(pixel, surface_depth);
                let to_camera = scene.camera.xyz - world;
                let behind = dot(scene.camera.xyz - surface, normalize(to_camera)) < length(to_camera);
                let thickness = distance(world, surface);
                if (behind && thickness < 0.5 + length(velocity) * dt * 2.0) {
                    if (emitter.collision.w > 0.5) {
                        let slot = atomicAdd(&counters.dead, 1);
                        dead[u32(slot)] = index;
                        return;
                    }
                    let right = reconstruct(pixel + vec2<i32>(1, 0), depth_at(pixel + vec2<i32>(1, 0)));
                    let down = reconstruct(pixel + vec2<i32>(0, 1), depth_at(pixel + vec2<i32>(0, 1)));
                    var normal = normalize(cross(down - surface, right - surface));
                    if (dot(normal, to_camera) < 0.0) {
                        normal = -normal;
                    }
                    var n = normal;
                    if (emitter.time.z > 0.5) {
                        n = normalize((emitter.inv_transform * vec4<f32>(normal, 0.0)).xyz);
                    }
                    let normal_part = n * dot(velocity, n);
                    let tangent = velocity - normal_part;
                    velocity = tangent * (1.0 - clamp(emitter.collision.z, 0.0, 1.0)) - normal_part * emitter.collision.y;
                    next = p.position_age.xyz;
                }
            }
        }
    }

    p.position_age = vec4<f32>(next, p.position_age.w);
    p.velocity_lifetime = vec4<f32>(velocity, p.velocity_lifetime.w);
    p.size_rotation.y += p.size_rotation.z * dt;
    particles[index] = p;
    let slot = atomicAdd(&counters.alive[dst], 1u);
    alive[dst * capacity + slot] = index;
}

@compute @workgroup_size(1)
fn finalize() {
    let dst = 1u - emitter.counts.w;
    let count = atomicLoad(&counters.alive[dst]);
    if (emitter.draw.y == 1u) {
        // DrawIndexedIndirect: index count, instances, first index, base vertex, first instance.
        draw_args[0] = emitter.draw.x;
        draw_args[1] = count;
        draw_args[2] = 0u;
        draw_args[3] = 0u;
        draw_args[4] = 0u;
    } else {
        draw_args[0] = emitter.draw.x;
        draw_args[1] = count;
        draw_args[2] = 0u;
        draw_args[3] = 0u;
    }
    // Sort input size.
    draw_args[5] = count;
}
