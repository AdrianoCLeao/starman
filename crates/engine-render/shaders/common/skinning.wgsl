// Linear blend skinning against the joint palette (model-space skin
// matrices: inverse(mesh global) * joint global * inverse bind).

fn skin_matrix(offset: u32, joints: vec4<u32>, weights: vec4<f32>) -> mat4x4<f32> {
    return palettes[offset + joints.x] * weights.x
        + palettes[offset + joints.y] * weights.y
        + palettes[offset + joints.z] * weights.z
        + palettes[offset + joints.w] * weights.w;
}
