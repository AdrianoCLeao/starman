use gltf::Gltf;
use gltf::mesh::Mode;
use nalgebra::{Point3, Vector3};
use std::path::Path;
use std::fs;

pub struct Mesh {
    pub name: Option<String>,
    pub vertices: Vec<Point3<f32>>,
    pub normals: Vec<Vector3<f32>>,
    pub indices: Vec<u32>,
}

pub fn load_glb(file_path: &Path) -> Result<Vec<Mesh>, String> {
    if !file_path.exists() {
        return Err(format!("File not found: {:?}", file_path));
    }

    let file_data = fs::read(file_path).map_err(|e| format!("Failed to read file: {}", e))?;

    let gltf = Gltf::from_slice(&file_data).map_err(|e| format!("Failed to parse GLB: {}", e))?;

    let mut meshes = Vec::new();

    for mesh in gltf.meshes() {
        for primitive in mesh.primitives() {
            if primitive.mode() != Mode::Triangles {
                return Err(format!("Unsupported primitive mode: {:?}", primitive.mode()));
            }

            let reader = primitive.reader(|buffer| {
                gltf.blob.as_deref()
            });

            let vertices: Vec<Point3<f32>> = reader
                .read_positions()
                .ok_or("Failed to read positions")?
                .map(|v| Point3::new(v[0], v[1], v[2]))
                .collect();

            let normals: Vec<Vector3<f32>> = reader
                .read_normals()
                .ok_or("Failed to read normals")?
                .map(|n| Vector3::new(n[0], n[1], n[2]))
                .collect();

            let indices: Vec<u32> = reader
                .read_indices()
                .ok_or("Failed to read indices")?
                .into_u32()
                .collect();

            meshes.push(Mesh {
                name: mesh.name().map(|s| s.to_string()),
                vertices,
                normals,
                indices,
            });
        }
    }

    Ok(meshes)
}
