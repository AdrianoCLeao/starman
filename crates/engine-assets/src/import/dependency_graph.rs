//! Tracks "who depends on this asset" so that changing one asset (e.g. a
//! texture) can be propagated to everything that references it (e.g. a
//! material), without every caller re-deriving that relationship itself.

use std::collections::{HashMap, HashSet};

use engine_core::SourceAssetId;

#[derive(Default)]
pub(crate) struct DependencyGraph {
    /// dependency -> the set of assets that declared a dependency on it.
    dependents: HashMap<SourceAssetId, HashSet<SourceAssetId>>,
}

impl DependencyGraph {
    /// Replaces every dependency edge previously recorded for `asset` with
    /// `dependencies`. Safe to call repeatedly as an asset is re-imported.
    pub fn set_dependencies(&mut self, asset: SourceAssetId, dependencies: &[SourceAssetId]) {
        self.remove_asset(asset);
        for dependency in dependencies {
            self.dependents
                .entry(*dependency)
                .or_default()
                .insert(asset);
        }
    }

    /// Removes every edge involving `asset`, whether as a dependency or a
    /// dependent. Used when an asset is deleted or fully re-scanned.
    pub fn remove_asset(&mut self, asset: SourceAssetId) {
        for dependents in self.dependents.values_mut() {
            dependents.remove(&asset);
        }
        self.dependents.remove(&asset);
    }

    /// The assets that directly depend on `asset`.
    pub fn direct_dependents_of(&self, asset: SourceAssetId) -> Vec<SourceAssetId> {
        self.dependents
            .get(&asset)
            .map(|set| set.iter().copied().collect())
            .unwrap_or_default()
    }

    /// Every asset that transitively depends on `asset` (i.e. everything
    /// that would need to be reconsidered if `asset` changed), in
    /// breadth-first order, each appearing once.
    pub fn transitive_dependents_of(&self, asset: SourceAssetId) -> Vec<SourceAssetId> {
        let mut visited = HashSet::new();
        let mut queue: Vec<SourceAssetId> = self.direct_dependents_of(asset);
        let mut ordered = Vec::new();

        while let Some(current) = queue.pop() {
            if !visited.insert(current) {
                continue;
            }
            ordered.push(current);
            queue.extend(self.direct_dependents_of(current));
        }

        ordered
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id() -> SourceAssetId {
        SourceAssetId::new_v4()
    }

    #[test]
    fn direct_dependents_are_reported() {
        let texture = id();
        let material = id();
        let mut graph = DependencyGraph::default();

        graph.set_dependencies(material, &[texture]);

        assert_eq!(graph.direct_dependents_of(texture), vec![material]);
        assert!(graph.direct_dependents_of(material).is_empty());
    }

    #[test]
    fn transitive_dependents_follow_the_chain() {
        let texture = id();
        let material = id();
        let mesh_renderer_scene = id();
        let mut graph = DependencyGraph::default();

        graph.set_dependencies(material, &[texture]);
        graph.set_dependencies(mesh_renderer_scene, &[material]);

        let mut transitive = graph.transitive_dependents_of(texture);
        transitive.sort_by_key(|id| id.to_string());

        let mut expected = vec![material, mesh_renderer_scene];
        expected.sort_by_key(|id| id.to_string());

        assert_eq!(transitive, expected);
    }

    #[test]
    fn set_dependencies_replaces_previous_edges() {
        let texture_a = id();
        let texture_b = id();
        let material = id();
        let mut graph = DependencyGraph::default();

        graph.set_dependencies(material, &[texture_a]);
        graph.set_dependencies(material, &[texture_b]);

        assert!(graph.direct_dependents_of(texture_a).is_empty());
        assert_eq!(graph.direct_dependents_of(texture_b), vec![material]);
    }
}
