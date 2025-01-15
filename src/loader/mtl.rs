use nalgebra::Vector3;

#[derive(Clone)]
pub struct MtlMaterial {
    pub name: String,
    pub ambiant_texture: Option<String>,
    pub diffuse_texture: Option<String>,
    pub specular_texture: Option<String>,
    pub opacity_map: Option<String>,
    pub ambiant: Vector3<f32>,
    pub diffuse: Vector3<f32>,
    pub specular: Vector3<f32>,
    pub shininess: f32,
    pub alpha: f32,
}

impl MtlMaterial {
    pub fn new_default(name: String) -> MtlMaterial {
        MtlMaterial {
            name,
            shininess: 60.0,
            alpha: 1.0,
            ambiant_texture: None,
            diffuse_texture: None,
            specular_texture: None,
            opacity_map: None,
            ambiant: Vector3::new(1.0, 1.0, 1.0),
            diffuse: Vector3::new(1.0, 1.0, 1.0),
            specular: Vector3::new(1.0, 1.0, 1.0),
        }
    }

    pub fn new(
        name: String,
        shininess: f32,
        alpha: f32,
        ambiant: Vector3<f32>,
        diffuse: Vector3<f32>,
        specular: Vector3<f32>,
        ambiant_texture: Option<String>,
        diffuse_texture: Option<String>,
        specular_texture: Option<String>,
        opacity_map: Option<String>,
    ) -> MtlMaterial {
        MtlMaterial {
            name,
            ambiant,
            diffuse,
            specular,
            ambiant_texture,
            diffuse_texture,
            specular_texture,
            opacity_map,
            shininess,
            alpha,
        }
    }
}