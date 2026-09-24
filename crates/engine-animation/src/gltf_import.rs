//! Shared glTF decoding helpers for skeletons and clips.

use engine_assets::LoadContext;
use engine_core::Result;
use engine_math::Mat4;

/// Parses a `.glb`/`.gltf` payload (external buffers resolve next to the
/// file).
pub fn gltf_document(
    bytes: &[u8],
    ctx: &LoadContext<'_>,
) -> Result<(gltf::Document, Vec<gltf::buffer::Data>)> {
    let (document, buffers, _images) = if ctx.disk_path.extension().is_some_and(|e| e == "gltf") {
        gltf::import(ctx.disk_path).map_err(|error| ctx.error(error.to_string()))?
    } else {
        gltf::import_slice(bytes).map_err(|error| ctx.error(error.to_string()))?
    };
    Ok((document, buffers))
}

/// Parent node index of every node.
pub fn node_parents(document: &gltf::Document) -> Vec<Option<usize>> {
    let mut parents = vec![None; document.nodes().len()];
    for node in document.nodes() {
        for child in node.children() {
            parents[child.index()] = Some(node.index());
        }
    }
    parents
}

/// Scene-space transform of every node.
pub fn node_globals(document: &gltf::Document, parents: &[Option<usize>]) -> Vec<Mat4> {
    let locals: Vec<Mat4> = document
        .nodes()
        .map(|node| Mat4::from_cols_array_2d(&node.transform().matrix()))
        .collect();
    let mut globals: Vec<Option<Mat4>> = vec![None; locals.len()];
    fn resolve(
        node: usize,
        locals: &[Mat4],
        parents: &[Option<usize>],
        globals: &mut Vec<Option<Mat4>>,
        depth: usize,
    ) -> Mat4 {
        if let Some(global) = globals[node] {
            return global;
        }
        let global = match parents[node] {
            Some(parent) if depth < 512 => {
                resolve(parent, locals, parents, globals, depth + 1) * locals[node]
            }
            _ => locals[node],
        };
        globals[node] = Some(global);
        global
    }
    (0..locals.len())
        .map(|node| resolve(node, &locals, parents, &mut globals, 0))
        .collect()
}
