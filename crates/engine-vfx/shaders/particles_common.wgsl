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

// Over-life lookup tables are read straight from the uniform in each
// shader (never copied into function arguments: some drivers mishandle
// dynamically indexed by-value arrays).

// Integer segment (x) and blend factor (y) for normalized age `t`.
fn lut_position(t: f32) -> vec2<f32> {
    let x = clamp(t, 0.0, 1.0) * 15.0;
    let i = min(floor(x), 14.0);
    return vec2<f32>(i, x - i);
}

fn lane(v: vec4<f32>, i: u32) -> f32 {
    return select(select(select(v.w, v.z, i == 2u), v.y, i == 1u), v.x, i == 0u);
}
