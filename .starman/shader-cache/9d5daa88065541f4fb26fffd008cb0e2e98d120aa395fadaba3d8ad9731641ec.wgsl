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

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_position: vec4<f32>,
    light_direction: vec4<f32>,
};

struct ModelUniform {
    model: mat4x4<f32>,
    normal: mat4x4<f32>,
};

struct MaterialUniform {
    base_color: vec4<f32>,
    metallic_roughness: vec4<f32>,
    emissive: vec4<f32>,
    flags: vec4<u32>,
};

struct GpuLight {
    position_range: vec4<f32>,
    color_intensity: vec4<f32>,
    direction_cone: vec4<f32>,
    light_type: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> model: ModelUniform;
@group(2) @binding(0) var<uniform> material: MaterialUniform;
@group(2) @binding(1) var t_albedo: texture_2d<f32>;
@group(2) @binding(2) var s_albedo: sampler;
@group(3) @binding(0) var<storage, read> lights: array<GpuLight>;
@group(3) @binding(1) var<uniform> light_count: vec4<u32>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    let world_pos = model.model * vec4<f32>(input.position, 1.0);
    output.clip_position = camera.view_proj * world_pos;
    output.world_position = world_pos.xyz;
    output.world_normal = normalize((model.normal * vec4<f32>(input.normal, 0.0)).xyz);
    output.uv = input.uv;
    return output;
}

fn evaluate_light(light: GpuLight, world_pos: vec3<f32>) -> vec3<f32> {
    if light.light_type == 0u {
        return normalize(-light.direction_cone.xyz);
    }
    let to_light = light.position_range.xyz - world_pos;
    let dist = length(to_light);
    let range = max(light.position_range.w, 0.001);
    if dist > range {
        return vec3<f32>(0.0);
    }
    return normalize(to_light);
}

fn light_radiance(light: GpuLight, world_pos: vec3<f32>, l: vec3<f32>) -> vec3<f32> {
    let color = light.color_intensity.xyz * light.color_intensity.w;
    if light.light_type == 0u {
        return color;
    }
    let dist = length(light.position_range.xyz - world_pos);
    let range = max(light.position_range.w, 0.001);
    let atten = saturate(1.0 - dist / range);
    var spot = 1.0;
    if light.light_type == 2u {
        let dir = normalize(light.direction_cone.xyz);
        let cos_outer = cos(light.direction_cone.w);
        let cos_inner = cos(bitcast<f32>(light.pad0));
        let cd = dot(-l, dir);
        spot = smoothstep(cos_outer, cos_inner, cd);
    }
    return color * (atten * atten) * spot;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let albedo_sample = textureSample(t_albedo, s_albedo, input.uv);
    let base_color = albedo_sample * material.base_color;
    if material.flags.x == 1u && base_color.a < material.emissive.w {
        discard;
    }
    let metallic = clamp(material.metallic_roughness.x, 0.0, 1.0);
    let roughness = clamp(material.metallic_roughness.y, 0.04, 1.0);
    let n = normalize(input.world_normal);
    let v = normalize(camera.camera_position.xyz - input.world_position);

    var color = base_color.rgb * 0.03;
    let count = min(light_count.x, 128u);
    for (var i = 0u; i < count; i = i + 1u) {
        let light = lights[i];
        let l = evaluate_light(light, input.world_position);
        if length(l) < 1e-5 {
            continue;
        }
        let rad = light_radiance(light, input.world_position, l);
        color += shade_pbr_light(base_color.rgb, metallic, roughness, n, v, l, rad);
    }
    // Fallback single directional if no lights uploaded.
    if count == 0u {
        let l = normalize(-camera.light_direction.xyz);
        color += shade_pbr_light(
            base_color.rgb,
            metallic,
            roughness,
            n,
            v,
            l,
            vec3<f32>(1.0),
        );
    }
    color += material.emissive.xyz;
    return vec4<f32>(color, base_color.a);
}
