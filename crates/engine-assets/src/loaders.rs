use engine_core::{EngineError, HardeningConfig, Result};
use image::GenericImageView;
use std::path::Path;

use crate::{MaterialData, MeshData, MeshVertex, SubMesh, TextureData};

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

/// Loads every mesh in a glTF/glb file as a separate [`MeshData`], one
/// submesh per primitive, with tangents (read or generated), skinning
/// streams (`JOINTS_0`/`WEIGHTS_0`, weights renormalized) and the skin the
/// mesh is instanced with. Mesh indices without primitives are kept as
/// empty placeholders so `mesh:<i>` keys stay aligned with the file.
pub(crate) fn load_mesh_payloads(
    path: &Path,
    hardening: &HardeningConfig,
) -> Result<Vec<MeshData>> {
    let (document, buffers, _images) =
        gltf::import(path).map_err(|error| EngineError::AssetLoad {
            path: path.display().to_string(),
            reason: error.to_string(),
        })?;
    let meshes = decode_gltf_meshes(path, &document, &buffers, hardening)?;
    if meshes.iter().all(|mesh| mesh.vertices.is_empty()) {
        return Err(EngineError::AssetLoad {
            path: path.display().to_string(),
            reason: "gltf file contains no renderable primitives".to_owned(),
        });
    }
    Ok(meshes)
}

fn load_error(path: &Path, reason: impl Into<String>) -> EngineError {
    EngineError::AssetLoad {
        path: path.display().to_string(),
        reason: reason.into(),
    }
}

/// Decodes every mesh of an already-imported glTF document.
pub fn decode_gltf_meshes(
    path: &Path,
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    hardening: &HardeningConfig,
) -> Result<Vec<MeshData>> {
    let fallback_name = path
        .file_stem()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unnamed-mesh".to_owned());

    let mut skin_of_mesh: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    for node in document.nodes() {
        if let (Some(mesh), Some(skin)) = (node.mesh(), node.skin()) {
            skin_of_mesh.entry(mesh.index()).or_insert(skin.index());
        }
    }

    let mut meshes = Vec::new();
    for (mesh_index, mesh) in document.meshes().enumerate() {
        let mut data = MeshData::new(
            mesh.name()
                .map(str::to_owned)
                .unwrap_or_else(|| format!("{fallback_name}-{mesh_index}")),
            Vec::new(),
            Vec::new(),
        );
        let mut all_tangents = true;
        let mut all_skinned = true;

        for primitive in mesh.primitives() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                log::warn!(
                    target: "engine::assets",
                    "{}: skipping non-triangle primitive in mesh {mesh_index}",
                    path.display()
                );
                continue;
            }
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
            let positions: Vec<[f32; 3]> = reader
                .read_positions()
                .ok_or_else(|| load_error(path, "gltf primitive is missing POSITION attribute"))?
                .collect();
            let count = positions.len();

            let normals: Vec<[f32; 3]> = reader
                .read_normals()
                .map(|iter| iter.collect())
                .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; count]);
            let uvs: Vec<[f32; 2]> = reader
                .read_tex_coords(0)
                .map(|iter| iter.into_f32().collect())
                .unwrap_or_else(|| vec![[0.0, 0.0]; count]);
            let tangents: Option<Vec<[f32; 4]>> = reader.read_tangents().map(|iter| iter.collect());
            let joints: Option<Vec<[u16; 4]>> =
                reader.read_joints(0).map(|iter| iter.into_u16().collect());
            let weights: Option<Vec<[f32; 4]>> =
                reader.read_weights(0).map(|iter| iter.into_f32().collect());

            if normals.len() != count
                || uvs.len() != count
                || tangents.as_ref().is_some_and(|t| t.len() != count)
                || joints.as_ref().is_some_and(|j| j.len() != count)
                || weights.as_ref().is_some_and(|w| w.len() != count)
            {
                return Err(load_error(
                    path,
                    "gltf vertex attribute lengths do not match",
                ));
            }

            let base_index = u32::try_from(data.vertices.len())
                .map_err(|_| load_error(path, "mesh has too many vertices for u32 index buffer"))?;
            let primitive_vertex_count = u32::try_from(count)
                .map_err(|_| load_error(path, "primitive vertex count overflow"))?;

            if data.vertices.len() + count > hardening.max_mesh_vertices {
                return Err(load_error(
                    path,
                    format!(
                        "mesh vertex count exceeds configured limit {}",
                        hardening.max_mesh_vertices
                    ),
                ));
            }

            for ((position, normal), uv) in positions.into_iter().zip(normals).zip(uvs) {
                data.vertices.push(MeshVertex {
                    position,
                    normal,
                    uv,
                });
            }
            match tangents {
                Some(tangents) if all_tangents => data.tangents.extend(tangents),
                _ => all_tangents = false,
            }
            match (joints, weights) {
                (Some(joints), Some(weights)) if all_skinned => {
                    data.joints.extend(joints);
                    data.weights
                        .extend(weights.into_iter().map(normalize_weights));
                }
                _ => all_skinned = false,
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
            if primitive_indices
                .iter()
                .any(|index| *index >= base_index + primitive_vertex_count)
            {
                return Err(load_error(path, "gltf index references a missing vertex"));
            }
            if data.indices.len() + primitive_indices.len() > hardening.max_mesh_indices {
                return Err(load_error(
                    path,
                    format!(
                        "mesh index count exceeds configured limit {}",
                        hardening.max_mesh_indices
                    ),
                ));
            }

            data.submeshes.push(SubMesh {
                first_index: data.indices.len() as u32,
                index_count: primitive_indices.len() as u32,
                material: primitive.material().index(),
            });
            data.indices.extend(primitive_indices);
        }

        if !all_tangents || data.tangents.len() != data.vertices.len() {
            data.tangents.clear();
        }
        if !all_skinned || data.joints.len() != data.vertices.len() {
            data.joints.clear();
            data.weights.clear();
        }
        if !data.vertices.is_empty() && !data.has_tangents() {
            data.generate_tangents();
        }
        if data.is_skinned() {
            data.skin = skin_of_mesh.get(&mesh_index).copied();
            if data.skin.is_none() {
                log::warn!(
                    target: "engine::assets",
                    "{}: mesh {mesh_index} has joint weights but no node binds it to a skin",
                    path.display()
                );
            }
        }
        data.recompute_bounds();
        meshes.push(data);
    }
    Ok(meshes)
}

fn normalize_weights(weights: [f32; 4]) -> [f32; 4] {
    let sum: f32 = weights.iter().map(|w| w.max(0.0)).sum();
    if sum <= 1e-8 {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        weights.map(|w| w.max(0.0) / sum)
    }
}

pub(crate) fn load_mesh_payload_merged(
    path: &Path,
    hardening: &HardeningConfig,
) -> Result<MeshData> {
    let meshes: Vec<MeshData> = load_mesh_payloads(path, hardening)?
        .into_iter()
        .filter(|mesh| !mesh.vertices.is_empty())
        .collect();
    merge_meshes(path, meshes)
}

/// Concatenates meshes into one. Skinning streams survive only when every
/// mesh is skinned against the same skin.
pub(crate) fn merge_meshes(path: &Path, meshes: Vec<MeshData>) -> Result<MeshData> {
    let fallback_name = path
        .file_stem()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unnamed-mesh".to_owned());
    let shared_skin = meshes.first().and_then(|mesh| mesh.skin);
    let keep_skin = !meshes.is_empty()
        && meshes
            .iter()
            .all(|mesh| mesh.is_skinned() && mesh.skin == shared_skin);

    let mut merged = MeshData::new(
        meshes
            .first()
            .map(|mesh| mesh.name.clone())
            .unwrap_or(fallback_name),
        Vec::new(),
        Vec::new(),
    );
    for mesh in meshes {
        let base_index = u32::try_from(merged.vertices.len())
            .map_err(|_| load_error(path, "mesh has too many vertices for u32 index buffer"))?;
        let base_first = merged.indices.len() as u32;
        let tangents = if mesh.has_tangents() {
            mesh.tangents.clone()
        } else {
            let mut copy = mesh.clone();
            copy.generate_tangents();
            copy.tangents
        };
        merged.tangents.extend(tangents);
        if keep_skin {
            merged.joints.extend(mesh.joints);
            merged.weights.extend(mesh.weights);
        }
        let submeshes = if mesh.submeshes.is_empty() {
            vec![SubMesh {
                first_index: 0,
                index_count: mesh.indices.len() as u32,
                material: None,
            }]
        } else {
            mesh.submeshes
        };
        merged
            .submeshes
            .extend(submeshes.into_iter().map(|submesh| SubMesh {
                first_index: submesh.first_index + base_first,
                ..submesh
            }));
        merged.vertices.extend(mesh.vertices);
        merged
            .indices
            .extend(mesh.indices.into_iter().map(|index| base_index + index));
    }
    if keep_skin {
        merged.skin = shared_skin;
    }
    merged.recompute_bounds();
    Ok(merged)
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
    parse_material_payload(path, &source)
}

pub(crate) fn parse_material_payload(path: &Path, source: &str) -> Result<MaterialData> {
    let source = source.to_owned();

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
                    map.insert(k, ron::Value::Option(Some(Box::new(ron::Value::String(s)))));
                }
            }
            let rewritten =
                ron::to_string(&ron::Value::Map(map)).map_err(|e| EngineError::AssetLoad {
                    path: path.display().to_string(),
                    reason: e.to_string(),
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
