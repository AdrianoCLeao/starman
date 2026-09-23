//! Detects cycles in the nested-scene dependency graph.

use std::collections::{HashMap, HashSet};

use engine_core::{EngineError, Result, SourceAssetId};

/// Walks a directed graph of scene → nested scene references and reports
/// the first cycle found as an actionable error.
pub struct CycleDetector;

impl CycleDetector {
    /// `edges` maps each scene to the scenes it directly instances.
    pub fn check(edges: &HashMap<SourceAssetId, Vec<SourceAssetId>>) -> Result<()> {
        let mut visiting = HashSet::new();
        let mut visited = HashSet::new();
        let mut stack = Vec::new();

        for &node in edges.keys() {
            if visited.contains(&node) {
                continue;
            }
            Self::dfs(node, edges, &mut visiting, &mut visited, &mut stack)?;
        }

        Ok(())
    }

    fn dfs(
        node: SourceAssetId,
        edges: &HashMap<SourceAssetId, Vec<SourceAssetId>>,
        visiting: &mut HashSet<SourceAssetId>,
        visited: &mut HashSet<SourceAssetId>,
        stack: &mut Vec<SourceAssetId>,
    ) -> Result<()> {
        if visited.contains(&node) {
            return Ok(());
        }
        if !visiting.insert(node) {
            let mut cycle: Vec<String> = stack
                .iter()
                .skip_while(|id| **id != node)
                .map(ToString::to_string)
                .collect();
            cycle.push(node.to_string());
            return Err(EngineError::AssetLoad {
                path: node.to_string(),
                reason: format!(
                    "nested scene cycle detected: {} — a scene cannot instance itself \
                     (directly or through other scenes)",
                    cycle.join(" → ")
                ),
            });
        }

        stack.push(node);
        if let Some(children) = edges.get(&node) {
            for &child in children {
                Self::dfs(child, edges, visiting, visited, stack)?;
            }
        }
        stack.pop();
        visiting.remove(&node);
        visited.insert(node);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_dag() {
        let a = SourceAssetId::new_v4();
        let b = SourceAssetId::new_v4();
        let c = SourceAssetId::new_v4();
        let mut edges = HashMap::new();
        edges.insert(a, vec![b, c]);
        edges.insert(b, vec![c]);
        edges.insert(c, vec![]);
        CycleDetector::check(&edges).unwrap();
    }

    #[test]
    fn rejects_cycle() {
        let a = SourceAssetId::new_v4();
        let b = SourceAssetId::new_v4();
        let mut edges = HashMap::new();
        edges.insert(a, vec![b]);
        edges.insert(b, vec![a]);
        let err = CycleDetector::check(&edges).unwrap_err();
        assert!(err.to_string().contains("cycle"));
    }
}
