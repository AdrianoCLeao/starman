// Back-to-front ordering for alpha-blended GPU particles: fill keys from
// the live list, then a bitonic sort over a power-of-two key array.

#include "vfx/particles_common.wgsl"

struct SortStep {
    k: u32,
    j: u32,
    list: u32,
    capacity: u32,
    camera: vec4<f32>,
    // x local space
    flags: vec4<f32>,
    transform: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> step: SortStep;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;
@group(0) @binding(2) var<storage, read> alive: array<u32>;
@group(0) @binding(3) var<storage, read> draw_args: array<u32, 8>;
@group(0) @binding(4) var<storage, read_write> keys: array<vec2<u32>>;

@compute @workgroup_size(64)
fn fill(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = arrayLength(&keys);
    if (id.x >= size) {
        return;
    }
    let count = draw_args[5];
    if (id.x >= count) {
        keys[id.x] = vec2<u32>(0xFFFFFFFFu, 0u);
        return;
    }
    let index = alive[step.list * step.capacity + id.x];
    var position = particles[index].position_age.xyz;
    if (step.flags.x > 0.5) {
        position = (step.transform * vec4<f32>(position, 1.0)).xyz;
    }
    let d = distance(position, step.camera.xyz);
    // Descending distance == ascending inverted bits (distances are >= 0).
    keys[id.x] = vec2<u32>(~bitcast<u32>(d), index);
}

@compute @workgroup_size(64)
fn sort_step(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let size = arrayLength(&keys);
    let partner = i ^ step.j;
    if (i >= size || partner <= i || partner >= size) {
        return;
    }
    let a = keys[i];
    let b = keys[partner];
    let ascending = (i & step.k) == 0u;
    if ((a.x > b.x) == ascending) {
        keys[i] = b;
        keys[partner] = a;
    }
}
