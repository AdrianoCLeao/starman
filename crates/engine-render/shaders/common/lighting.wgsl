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
