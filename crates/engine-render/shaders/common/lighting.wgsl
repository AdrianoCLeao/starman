// Cook-Torrance PBR + clustered light evaluation helpers (WGSL).

fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (3.14159265 * d * d + 1e-7);
}

fn geometry_schlick_ggx(n_dot_x: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return n_dot_x / (n_dot_x * (1.0 - k) + k + 1e-7);
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    return geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn shade_pbr_light(
    base_color: vec3<f32>,
    metallic: f32,
    roughness: f32,
    n: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>,
    radiance: vec3<f32>,
) -> vec3<f32> {
    let h = normalize(v + l);
    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_v = max(dot(n, v), 0.0);
    let n_dot_h = max(dot(n, h), 0.0);
    let h_dot_v = max(dot(h, v), 0.0);

    let f0 = mix(vec3<f32>(0.04), base_color, metallic);
    let d = distribution_ggx(n_dot_h, roughness);
    let g = geometry_smith(n_dot_v, n_dot_l, roughness);
    let f = fresnel_schlick(h_dot_v, f0);
    let specular = (d * g * f) / max(4.0 * n_dot_v * n_dot_l, 1e-4);
    let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
    let diffuse = kd * base_color / 3.14159265;
    return (diffuse + specular) * radiance * n_dot_l;
}

fn shade_blinn_phong(
    base_color: vec3<f32>,
    n: vec3<f32>,
    l: vec3<f32>,
    v: vec3<f32>,
    metallic: f32,
    roughness: f32,
) -> vec3<f32> {
    let h = normalize(l + v);
    let ndotl = max(dot(n, l), 0.0);
    let ndoth = max(dot(n, h), 0.0);
    let spec_power = mix(256.0, 4.0, roughness);
    let specular_strength = pow(ndoth, spec_power);
    let diffuse = base_color * ndotl * (1.0 - metallic);
    let specular = vec3<f32>(specular_strength) * mix(0.04, 1.0, metallic);
    let ambient = base_color * 0.08;
    return ambient + diffuse + specular;
}
