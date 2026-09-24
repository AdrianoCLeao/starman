use engine_core::{EngineError, HardeningConfig, Result};
use image::GenericImageView;
use std::path::Path;

use crate::{MaterialData, MeshData, MeshVertex, TextureData};

pub(crate) fn load_texture_payload(
    path: &Path,
    hardening: &HardeningConfig,
) -> Result<TextureData> {
    let image = image::open(path).map_err(|error| EngineError::AssetLoad {
        path: path.display().to_string(),
        reason: error.to_string(),
    })?;

    let (width, height) = image.dimensions();
    if width > hardening.max_texture_dimension || height > hardening.max_texture_dimension {
        return Err(EngineError::AssetLoad {
            path: path.display().to_string(),
            reason: format!(
                "texture dimensions {}x{} exceed configured limit {}",
                width, height, hardening.max_texture_dimension
            ),
        });
    }

    let estimated_payload_bytes = width as u64 * height as u64 * 4;
    if estimated_payload_bytes > hardening.max_texture_payload_bytes as u64 {
        return Err(EngineError::AssetLoad {
            path: path.display().to_string(),
            reason: format!(
                "texture payload {} bytes exceeds configured limit {}",
                estimated_payload_bytes, hardening.max_texture_payload_bytes
            ),
        });
    }

    let pixels_rgba8 = image.to_rgba8().into_raw();

    Ok(TextureData {
        width,
        height,
        pixels_rgba8,
        revision: 0,
    })
}

/// Loads every mesh in a glTF/glb file as a separate [`MeshData`].
pub(crate) fn load_mesh_payloads(
    path: &Path,
    hardening: &HardeningConfig,
) -> Result<Vec<MeshData>> {
    let (document, buffers, _images) =
        gltf::import(path).map_err(|error| EngineError::AssetLoad {
            path: path.display().to_string(),
            reason: error.to_string(),
        })?;

    let fallback_name = path
        .file_stem()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unnamed-mesh".to_owned());

    let mut meshes = Vec::new();

    for (mesh_index, mesh) in document.meshes().enumerate() {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
            let positions: Vec<[f32; 3]> = reader
                .read_positions()
                .ok_or_else(|| EngineError::AssetLoad {
                    path: path.display().to_string(),
                    reason: "gltf primitive is missing POSITION attribute".to_owned(),
                })?
                .collect();

            let normals: Vec<[f32; 3]> = reader
                .read_normals()
                .map(|iter| iter.collect())
                .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);
            let uvs: Vec<[f32; 2]> = reader
                .read_tex_coords(0)
                .map(|iter| iter.into_f32().collect())
                .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);

            if normals.len() != positions.len() || uvs.len() != positions.len() {
                return Err(EngineError::AssetLoad {
                    path: path.display().to_string(),
                    reason: "gltf vertex attribute lengths do not match".to_owned(),
                });
            }

            let base_index = u32::try_from(vertices.len()).map_err(|_| EngineError::AssetLoad {
                path: path.display().to_string(),
                reason: "mesh has too many vertices for u32 index buffer".to_owned(),
            })?;
            let primitive_vertex_count =
                u32::try_from(positions.len()).map_err(|_| EngineError::AssetLoad {
                    path: path.display().to_string(),
                    reason: "primitive vertex count overflow".to_owned(),
                })?;

            if vertices.len() + positions.len() > hardening.max_mesh_vertices {
                return Err(EngineError::AssetLoad {
                    path: path.display().to_string(),
                    reason: format!(
                        "mesh vertex count exceeds configured limit {}",
                        hardening.max_mesh_vertices
                    ),
                });
            }

            for ((position, normal), uv) in positions.into_iter().zip(normals).zip(uvs) {
                vertices.push(MeshVertex {
                    position,
                    normal,
                    uv,
                });
            }

            let primitive_indices: Vec<u32> = if let Some(read_indices) = reader.read_indices() {
                read_indices
                    .into_u32()
                    .map(|index| base_index + index)
                    .collect()
            } else {
                (0..primitive_vertex_count)
                    .map(|index| base_index + index)
                    .collect()
            };

            if indices.len() + primitive_indices.len() > hardening.max_mesh_indices {
                return Err(EngineError::AssetLoad {
                    path: path.display().to_string(),
                    reason: format!(
                        "mesh index count exceeds configured limit {}",
                        hardening.max_mesh_indices
                    ),
                });
            }

            indices.extend(primitive_indices);
        }

        if vertices.is_empty() {
            continue;
        }

        let name = mesh
            .name()
            .map(str::to_owned)
            .unwrap_or_else(|| format!("{fallback_name}-{mesh_index}"));

        let mut mesh_data = MeshData {
            name,
            vertices,
            indices,
            aabb_min: [0.0; 3],
            aabb_max: [0.0; 3],
        };
        mesh_data.recompute_bounds();
        meshes.push(mesh_data);
    }

    if meshes.is_empty() {
        return Err(EngineError::AssetLoad {
            path: path.display().to_string(),
            reason: "gltf file contains no renderable primitives".to_owned(),
        });
    }

    Ok(meshes)
}

pub(crate) fn load_mesh_payload_merged(
    path: &Path,
    hardening: &HardeningConfig,
) -> Result<MeshData> {
    let meshes = load_mesh_payloads(path, hardening)?;

    let mut name = None;
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for mesh in meshes {
        if name.is_none() {
            name = Some(mesh.name);
        }

        let base_index = u32::try_from(vertices.len()).map_err(|_| EngineError::AssetLoad {
            path: path.display().to_string(),
            reason: "mesh has too many vertices for u32 index buffer".to_owned(),
        })?;

        vertices.extend(mesh.vertices);
        indices.extend(mesh.indices.into_iter().map(|index| base_index + index));
    }

    let fallback_name = path
        .file_stem()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unnamed-mesh".to_owned());

    let mut mesh_data = MeshData {
        name: name.unwrap_or(fallback_name),
        vertices,
        indices,
        aabb_min: [0.0; 3],
        aabb_max: [0.0; 3],
    };
    mesh_data.recompute_bounds();
    Ok(mesh_data)
}

/// Extract glTF PBR materials as Starman [`MaterialData`] (texture paths empty;
/// callers may resolve embedded images separately).
pub fn extract_gltf_materials(path: &Path) -> Result<Vec<MaterialData>> {
    let (document, _buffers, _images) =
        gltf::import(path).map_err(|error| EngineError::AssetLoad {
            path: path.display().to_string(),
            reason: error.to_string(),
        })?;

    let mut out = Vec::new();
    for material in document.materials() {
        let pbr = material.pbr_metallic_roughness();
        let base = pbr.base_color_factor();
        let alpha_mode = match material.alpha_mode() {
            gltf::material::AlphaMode::Opaque => "OPAQUE",
            gltf::material::AlphaMode::Mask => "MASK",
            gltf::material::AlphaMode::Blend => "BLEND",
        };
        let emissive = material.emissive_factor();
        out.push(MaterialData {
            base_color_factor: base,
            metallic: pbr.metallic_factor(),
            roughness: pbr.roughness_factor(),
            emissive_factor: emissive,
            normal_scale: material.normal_texture().map(|n| n.scale()).unwrap_or(1.0),
            occlusion_strength: material
                .occlusion_texture()
                .map(|o| o.strength())
                .unwrap_or(1.0),
            alpha_mode: alpha_mode.to_owned(),
            alpha_cutoff: material.alpha_cutoff().unwrap_or(0.5),
            double_sided: material.double_sided(),
            base_color_texture: None,
            metallic_roughness_texture: None,
            normal_texture: None,
            occlusion_texture: None,
            emissive_texture: None,
        });
    }
    Ok(out)
}

pub(crate) fn load_material_payload(path: &Path) -> Result<MaterialData> {
    let source = std::fs::read_to_string(path).map_err(|error| EngineError::AssetLoad {
        path: path.display().to_string(),
        reason: error.to_string(),
    })?;

    #[derive(serde::Deserialize)]
    struct RawMaterial {
        base_color_factor: Vec<f32>,
        metallic: f32,
        roughness: f32,
        #[serde(default)]
        emissive_factor: Option<Vec<f32>>,
        #[serde(default)]
        normal_scale: Option<f32>,
        #[serde(default)]
        occlusion_strength: Option<f32>,
        #[serde(default)]
        alpha_mode: Option<String>,
        #[serde(default)]
        alpha_cutoff: Option<f32>,
        #[serde(default)]
        double_sided: Option<bool>,
        #[serde(default)]
        base_color_texture: Option<String>,
        #[serde(default)]
        metallic_roughness_texture: Option<String>,
        #[serde(default)]
        normal_texture: Option<String>,
        #[serde(default)]
        occlusion_texture: Option<String>,
        #[serde(default)]
        emissive_texture: Option<String>,
    }

    // Accept either `Some("path")` or a bare `"path"` for texture slots via a
    // two-pass: try full RON Value map for string-or-option fields.
    let value: ron::Value = ron::from_str(&source).map_err(|error| EngineError::AssetLoad {
        path: path.display().to_string(),
        reason: format!("failed to parse material: {error}"),
    })?;

    let raw: RawMaterial = match ron::from_str(&source) {
        Ok(raw) => raw,
        Err(_) => {
            // Fallback: coerce bare string texture fields into Options.
            let mut map = match value {
                ron::Value::Map(m) => m,
                _ => {
                    return Err(EngineError::AssetLoad {
                        path: path.display().to_string(),
                        reason: "material root must be a map".to_owned(),
                    });
                }
            };
            for key in [
                "base_color_texture",
                "metallic_roughness_texture",
                "normal_texture",
                "occlusion_texture",
                "emissive_texture",
            ] {
                let k = ron::Value::String(key.into());
                if let Some(ron::Value::String(s)) = map.remove(&k) {
                    map.insert(
                        k,
                        ron::Value::Option(Some(Box::new(ron::Value::String(s)))),
                    );
                }
            }
            let rewritten = ron::to_string(&ron::Value::Map(map)).map_err(|e| {
                EngineError::AssetLoad {
                    path: path.display().to_string(),
                    reason: e.to_string(),
                }
            })?;
            ron::from_str(&rewritten).map_err(|error| EngineError::AssetLoad {
                path: path.display().to_string(),
                reason: format!("failed to parse material: {error}"),
            })?
        }
    };

    let base_color_factor: [f32; 4] =
        raw.base_color_factor
            .as_slice()
            .try_into()
            .map_err(|_| EngineError::AssetLoad {
                path: path.display().to_string(),
                reason: format!(
                    "base_color_factor must have 4 components, found {}",
                    raw.base_color_factor.len()
                ),
            })?;

    let emissive_factor = match raw.emissive_factor {
        Some(v) if v.len() == 3 => [v[0], v[1], v[2]],
        Some(v) => {
            return Err(EngineError::AssetLoad {
                path: path.display().to_string(),
                reason: format!("emissive_factor must have 3 components, found {}", v.len()),
            });
        }
        None => [0.0; 3],
    };

    let material = MaterialData {
        base_color_factor,
        metallic: raw.metallic,
        roughness: raw.roughness,
        emissive_factor,
        normal_scale: raw.normal_scale.unwrap_or(1.0),
        occlusion_strength: raw.occlusion_strength.unwrap_or(1.0),
        alpha_mode: raw.alpha_mode.unwrap_or_else(|| "OPAQUE".to_owned()),
        alpha_cutoff: raw.alpha_cutoff.unwrap_or(0.5),
        double_sided: raw.double_sided.unwrap_or(false),
        base_color_texture: raw.base_color_texture,
        metallic_roughness_texture: raw.metallic_roughness_texture,
        normal_texture: raw.normal_texture,
        occlusion_texture: raw.occlusion_texture,
        emissive_texture: raw.emissive_texture,
    };

    let all_finite = material
        .base_color_factor
        .iter()
        .chain(material.emissive_factor.iter())
        .all(|value| value.is_finite())
        && material.metallic.is_finite()
        && material.roughness.is_finite();
    if !all_finite {
        return Err(EngineError::AssetLoad {
            path: path.display().to_string(),
            reason: "material values must be finite numbers".to_owned(),
        });
    }

    Ok(material)
}
