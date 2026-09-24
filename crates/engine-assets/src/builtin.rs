//! Typed loaders for the built-in texture, mesh and material payloads, and
//! the runtime plugin that installs the shared [`Assets`] store.

use std::path::Path;

use engine_core::{GameRuntime, Result, RuntimePlugin};
use image::GenericImageView;

use crate::loaders::{decode_gltf_meshes, merge_meshes, parse_material_payload};
use crate::typed::{Asset, AssetLoader, Assets, LoadContext};
use crate::{MaterialData, MeshData, TextureData};

impl Asset for TextureData {
    const TYPE_NAME: &'static str = "Texture";
}

impl Asset for MeshData {
    const TYPE_NAME: &'static str = "Mesh";
}

impl Asset for MaterialData {
    const TYPE_NAME: &'static str = "Material";
}

/// PNG/JPEG → RGBA8 [`TextureData`], within hardening limits.
pub struct TextureLoader;

impl AssetLoader for TextureLoader {
    type Asset = TextureData;

    fn extensions(&self) -> &'static [&'static str] {
        &["png", "jpg", "jpeg"]
    }

    fn load(&self, bytes: &[u8], ctx: &mut LoadContext<'_>) -> Result<TextureData> {
        let image = image::load_from_memory(bytes).map_err(|error| ctx.error(error.to_string()))?;
        let (width, height) = image.dimensions();
        let limit = ctx.hardening.max_texture_dimension;
        if width > limit || height > limit {
            return Err(ctx.error(format!(
                "texture dimensions {width}x{height} exceed configured limit {limit}"
            )));
        }
        if width as u64 * height as u64 * 4 > ctx.hardening.max_texture_payload_bytes as u64 {
            return Err(ctx.error("texture payload exceeds configured limit"));
        }
        Ok(TextureData {
            width,
            height,
            pixels_rgba8: image.to_rgba8().into_raw(),
            revision: 0,
        })
    }
}

/// glTF/glb → [`MeshData`]. `file#mesh:<i>` addresses one mesh; the bare
/// file merges every mesh (the legacy `load_mesh_handle` behavior).
pub struct MeshLoader;

impl AssetLoader for MeshLoader {
    type Asset = MeshData;

    fn extensions(&self) -> &'static [&'static str] {
        &["glb", "gltf"]
    }

    fn load(&self, bytes: &[u8], ctx: &mut LoadContext<'_>) -> Result<MeshData> {
        let base = ctx.disk_path.parent().map(Path::to_path_buf);
        let (document, buffers, _images) = if ctx.disk_path.extension().is_some_and(|e| e == "gltf")
        {
            // External buffers resolve relative to the file.
            gltf::import(ctx.disk_path).map_err(|error| ctx.error(error.to_string()))?
        } else {
            gltf::import_slice(bytes).map_err(|error| ctx.error(error.to_string()))?
        };
        let _ = base;
        let meshes = decode_gltf_meshes(ctx.disk_path, &document, &buffers, ctx.hardening)?;
        match ctx.sub_key {
            None => merge_meshes(
                ctx.disk_path,
                meshes
                    .into_iter()
                    .filter(|mesh| !mesh.vertices.is_empty())
                    .collect(),
            ),
            Some(key) => {
                let index = key
                    .strip_prefix("mesh:")
                    .and_then(|index| index.parse::<usize>().ok())
                    .ok_or_else(|| ctx.error(format!("'{key}' is not a mesh sub-asset key")))?;
                meshes
                    .into_iter()
                    .nth(index)
                    .filter(|mesh| !mesh.vertices.is_empty())
                    .ok_or_else(|| ctx.error(format!("mesh {index} has no triangles")))
            }
        }
    }
}

/// `.ron` material documents → [`MaterialData`] (legacy string texture
/// slots accepted). Claims only `material.ron`/`mat.ron` so it does not
/// shadow other RON asset types; the legacy handle API still loads any
/// `.ron` as a material.
pub struct MaterialLoader;

impl AssetLoader for MaterialLoader {
    type Asset = MaterialData;

    fn extensions(&self) -> &'static [&'static str] {
        &["material.ron", "mat.ron", "ron"]
    }

    fn load(&self, bytes: &[u8], ctx: &mut LoadContext<'_>) -> Result<MaterialData> {
        let source = std::str::from_utf8(bytes).map_err(|_| ctx.error("material is not UTF-8"))?;
        parse_material_payload(ctx.disk_path, source)
    }
}

/// High dynamic range image (Radiance `.hdr`), linear RGBA32F.
#[derive(Clone, Debug)]
pub struct HdrImageData {
    pub width: u32,
    pub height: u32,
    /// Linear RGBA, row-major, top row first.
    pub rgba: Vec<f32>,
}

impl Asset for HdrImageData {
    const TYPE_NAME: &'static str = "HdrImage";
}

/// Radiance `.hdr` → [`HdrImageData`] (environment panoramas).
pub struct HdrImageLoader;

impl AssetLoader for HdrImageLoader {
    type Asset = HdrImageData;

    fn extensions(&self) -> &'static [&'static str] {
        &["hdr"]
    }

    fn load(&self, bytes: &[u8], ctx: &mut LoadContext<'_>) -> Result<HdrImageData> {
        let image = image::load_from_memory_with_format(bytes, image::ImageFormat::Hdr)
            .map_err(|error| ctx.error(error.to_string()))?;
        let (width, height) = image.dimensions();
        let limit = ctx.hardening.max_texture_dimension;
        if width > limit || height > limit {
            return Err(ctx.error(format!(
                "HDR dimensions {width}x{height} exceed configured limit {limit}"
            )));
        }
        Ok(HdrImageData {
            width,
            height,
            rgba: image.to_rgba32f().into_raw(),
        })
    }
}

/// Installs the shared typed-asset store (unless the host already inserted
/// the one its `AssetServer` fills) and the built-in loaders.
#[derive(Default)]
pub struct AssetsPlugin;

impl RuntimePlugin for AssetsPlugin {
    fn name(&self) -> &'static str {
        "engine::assets"
    }

    fn build(&self, runtime: &mut GameRuntime) {
        runtime.init_resource::<Assets>();
        let assets = runtime.world.resource::<Assets>().clone();
        assets.register_loader(TextureLoader);
        assets.register_loader(MeshLoader);
        assets.register_loader(MaterialLoader);
        assets.register_loader(HdrImageLoader);
        runtime.register_type::<crate::AssetRef>();
    }
}
