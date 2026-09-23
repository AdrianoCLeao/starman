//! Builds the nested-scene dependency graph from scene documents.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use engine_assets::{AssetDatabase, SceneEntityData, SceneFile};
use engine_core::{EngineError, Result, SourceAssetId};
use uuid::Uuid;

use crate::CycleDetector;

/// Directed graph of scene asset dependencies used for cycle checks and
/// bulk reload planning.
#[derive(Debug, Default, Clone)]
pub struct CompositionGraph {
    pub edges: HashMap<SourceAssetId, Vec<SourceAssetId>>,
    pub paths: HashMap<SourceAssetId, PathBuf>,
}

impl CompositionGraph {
    /// Loads `root_path`, follows nested `SceneInstance` references through
    /// `database` (and `scene_path` fallbacks), and fails if a cycle is present.
    pub fn build_from_file(root_path: &Path, database: &AssetDatabase) -> Result<Self> {
        let mut graph = Self::default();
        Self::walk_file(root_path, database, &mut graph)?;
        CycleDetector::check(&graph.edges)?;
        Ok(graph)
    }

    /// Builds from an already-parsed root scene and a loader callback.
    pub fn build_with_loader(
        root_id: SourceAssetId,
        root: &SceneFile,
        mut load: impl FnMut(SourceAssetId) -> Result<(PathBuf, SceneFile)>,
    ) -> Result<Self> {
        let mut graph = Self::default();
        Self::walk_parsed(root_id, None, root, &mut load, &mut graph)?;
        CycleDetector::check(&graph.edges)?;
        Ok(graph)
    }

    fn walk_file(
        path: &Path,
        database: &AssetDatabase,
        graph: &mut CompositionGraph,
    ) -> Result<()> {
        let source = std::fs::read_to_string(path).map_err(|error| EngineError::AssetLoad {
            path: path.display().to_string(),
            reason: error.to_string(),
        })?;
        let scene = parse_scene_for_graph(path, &source)?;

        let relative = relative_under_assets(database, path)
            .unwrap_or_else(|| path.display().to_string());
        let id = database
            .resolve_id(&relative)
            .unwrap_or_else(|| path_derived_id(&relative));

        if graph.paths.values().any(|known| known == path) {
            return Ok(());
        }

        graph.paths.insert(id, path.to_path_buf());
        let mut child_ids = Vec::new();

        for child_rel in collect_instance_paths(&scene.entities) {
            let child_path = database.assets_root().join(&child_rel);
            if !child_path.is_file() {
                return Err(EngineError::AssetLoad {
                    path: child_rel,
                    reason: "nested scene path referenced by an instance does not exist"
                        .to_owned(),
                });
            }
            let child_id = database
                .resolve_id(&child_rel)
                .unwrap_or_else(|| path_derived_id(&child_rel));
            child_ids.push(child_id);
            Self::walk_file(&child_path, database, graph)?;
        }

        for dep in scene.direct_scene_dependencies() {
            if let Some(rel) = database.resolve_relative_path(dep) {
                let child_path = database.assets_root().join(rel);
                if child_path.is_file() {
                    child_ids.push(dep);
                    Self::walk_file(&child_path, database, graph)?;
                }
            }
        }

        child_ids.sort();
        child_ids.dedup();
        graph.edges.insert(id, child_ids);
        Ok(())
    }

    fn walk_parsed(
        id: SourceAssetId,
        path: Option<PathBuf>,
        scene: &SceneFile,
        load: &mut impl FnMut(SourceAssetId) -> Result<(PathBuf, SceneFile)>,
        graph: &mut CompositionGraph,
    ) -> Result<()> {
        if graph.edges.contains_key(&id) {
            return Ok(());
        }
        if let Some(path) = path {
            graph.paths.insert(id, path);
        }
        let deps = scene.direct_scene_dependencies();
        graph.edges.insert(id, deps.clone());
        for dep in deps {
            if graph.edges.contains_key(&dep) {
                continue;
            }
            let (child_path, child) = load(dep)?;
            Self::walk_parsed(dep, Some(child_path), &child, load, graph)?;
        }
        Ok(())
    }
}

fn path_derived_id(relative: &str) -> SourceAssetId {
    SourceAssetId::from_uuid(Uuid::new_v5(&Uuid::NAMESPACE_OID, relative.as_bytes()))
}

fn collect_instance_paths(entities: &[SceneEntityData]) -> Vec<String> {
    let mut paths = Vec::new();
    fn walk(entities: &[SceneEntityData], paths: &mut Vec<String>) {
        for entity in entities {
            if let Some(instance) = &entity.instance {
                if let Some(path) = &instance.scene_path {
                    paths.push(path.clone());
                }
                for added in &instance.added {
                    walk(std::slice::from_ref(&added.entity), paths);
                }
            }
            walk(&entity.children, paths);
        }
    }
    walk(entities, &mut paths);
    paths.sort();
    paths.dedup();
    paths
}

fn parse_scene_for_graph(path: &Path, source: &str) -> Result<SceneFile> {
    let mut scene: SceneFile =
        ron::from_str(source).map_err(|error| EngineError::AssetLoad {
            path: path.display().to_string(),
            reason: format!("failed to parse scene for composition graph: {error}"),
        })?;
    scene.version = SceneFile::CURRENT_VERSION;
    Ok(scene)
}

fn relative_under_assets(database: &AssetDatabase, path: &Path) -> Option<String> {
    path.strip_prefix(database.assets_root())
        .ok()
        .map(|p| p.to_string_lossy().replace('\\', "/"))
}
