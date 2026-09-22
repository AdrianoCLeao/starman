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

/// Loads every mesh in a glTF/glb file as a separate [`MeshData`] — one per
/// `document.meshes()` entry, in that same (stable) order, which is what
/// mesh sub-asset keys (`"mesh:<index>"`, see
/// `AssetDatabase::extract_mesh_sub_asset_keys`) index into. Primitives
/// *within* a single mesh are still merged into that mesh's one
/// vertex/index buffer, as they always were.
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

        meshes.push(MeshData {
            name,
            vertices,
            indices,
        });
    }

    if meshes.is_empty() {
        return Err(EngineError::AssetLoad {
            path: path.display().to_string(),
            reason: "gltf file contains no renderable primitives".to_owned(),
        });
    }

    Ok(meshes)
}

/// Loads every mesh in the file (see [`load_mesh_payloads`]) and flattens
/// them into one merged blob — the long-standing behavior of
/// `AssetServer::load_mesh_handle`, preserved unchanged for callers that
/// just want a single renderable mesh for the whole file.
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

    Ok(MeshData {
        name: name.unwrap_or(fallback_name),
        vertices,
        indices,
    })
}

pub(crate) fn load_material_payload(path: &Path) -> Result<MaterialData> {
    let source = std::fs::read_to_string(path).map_err(|error| EngineError::AssetLoad {
        path: path.display().to_string(),
        reason: error.to_string(),
    })?;

    // RON reads fixed-size arrays as tuples (`(1.0, ..)`), but material files
    // are authored with lists (`[1.0, ..]`), so parse through a `Vec`.
    #[derive(serde::Deserialize)]
    struct RawMaterial {
        base_color_factor: Vec<f32>,
        metallic: f32,
        roughness: f32,
    }

    let raw: RawMaterial = ron::from_str(&source).map_err(|error| EngineError::AssetLoad {
        path: path.display().to_string(),
        reason: format!("failed to parse material: {error}"),
    })?;

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

    let material = MaterialData {
        base_color_factor,
        metallic: raw.metallic,
        roughness: raw.roughness,
    };

    let all_finite = material
        .base_color_factor
        .iter()
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
