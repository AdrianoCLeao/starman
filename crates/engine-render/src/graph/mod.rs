//! Declarative render graph with topological execution.

use std::collections::{HashMap, HashSet, VecDeque};

use engine_core::{EngineError, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PassId(pub &'static str);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ResourceId(pub &'static str);

#[derive(Clone, Debug)]
pub enum ResourceKind {
    ColorTarget,
    Depth,
    Buffer,
    Texture,
}

#[derive(Clone, Debug)]
pub struct GraphResourceDecl {
    pub id: ResourceId,
    pub kind: ResourceKind,
}

#[derive(Clone, Debug)]
pub struct PassNode {
    pub id: PassId,
    pub reads: Vec<ResourceId>,
    pub writes: Vec<ResourceId>,
    /// Explicit ordering edges (runs after these passes).
    pub after: Vec<PassId>,
    pub enabled: bool,
}

#[derive(Default, Debug)]
pub struct RenderGraph {
    resources: HashMap<ResourceId, GraphResourceDecl>,
    passes: Vec<PassNode>,
}

impl RenderGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_resource(&mut self, decl: GraphResourceDecl) {
        self.resources.insert(decl.id, decl);
    }

    pub fn add_pass(&mut self, pass: PassNode) {
        self.passes.push(pass);
    }

    pub fn set_enabled(&mut self, id: PassId, enabled: bool) {
        if let Some(pass) = self.passes.iter_mut().find(|p| p.id == id) {
            pass.enabled = enabled;
        }
    }

    pub fn passes(&self) -> &[PassNode] {
        &self.passes
    }

    pub fn resources(&self) -> impl Iterator<Item = &GraphResourceDecl> {
        self.resources.values()
    }

    /// Topological order from explicit `after` edges (+ declaration tie-break).
    pub fn schedule(&self) -> Result<Vec<PassId>> {
        let enabled: Vec<&PassNode> = self.passes.iter().filter(|p| p.enabled).collect();
        let ids: HashSet<PassId> = enabled.iter().map(|p| p.id).collect();

        let mut dependents: HashMap<PassId, Vec<PassId>> = HashMap::new();
        let mut indegree: HashMap<PassId, usize> = HashMap::new();
        for pass in &enabled {
            indegree.entry(pass.id).or_insert(0);
        }

        for pass in &enabled {
            for pred in &pass.after {
                if !ids.contains(pred) {
                    continue;
                }
                dependents.entry(*pred).or_default().push(pass.id);
                *indegree.entry(pass.id).or_insert(0) += 1;
            }
        }

        let order_index: HashMap<PassId, usize> =
            enabled.iter().enumerate().map(|(i, p)| (p.id, i)).collect();

        let mut queue: VecDeque<PassId> = indegree
            .iter()
            .filter(|(_, d)| **d == 0)
            .map(|(id, _)| *id)
            .collect();
        let mut scheduled = Vec::new();

        while !queue.is_empty() {
            queue
                .make_contiguous()
                .sort_by_key(|id| order_index.get(id).copied().unwrap_or(usize::MAX));
            let next = queue.pop_front().unwrap();
            scheduled.push(next);
            if let Some(children) = dependents.get(&next) {
                for child in children {
                    if let Some(deg) = indegree.get_mut(child) {
                        *deg = deg.saturating_sub(1);
                        if *deg == 0 {
                            queue.push_back(*child);
                        }
                    }
                }
            }
        }

        if scheduled.len() != enabled.len() {
            return Err(EngineError::Render(
                "render graph contains a cycle among enabled passes".to_owned(),
            ));
        }
        Ok(scheduled)
    }

    /// Default M4 Forward+ graph topology.
    pub fn default_forward_plus() -> Self {
        let mut graph = Self::new();
        for (id, kind) in [
            (ResourceId("color"), ResourceKind::ColorTarget),
            (ResourceId("depth"), ResourceKind::Depth),
            (ResourceId("id_buffer"), ResourceKind::Texture),
            (ResourceId("lights"), ResourceKind::Buffer),
            (ResourceId("clusters"), ResourceKind::Buffer),
        ] {
            graph.add_resource(GraphResourceDecl { id, kind });
        }

        graph.add_pass(PassNode {
            id: PassId("clear"),
            reads: vec![],
            writes: vec![ResourceId("color"), ResourceId("depth")],
            after: vec![],
            enabled: true,
        });
        graph.add_pass(PassNode {
            id: PassId("id_pick"),
            reads: vec![ResourceId("depth")],
            writes: vec![ResourceId("id_buffer")],
            after: vec![PassId("clear")],
            enabled: false,
        });
        graph.add_pass(PassNode {
            id: PassId("cluster_cull"),
            reads: vec![ResourceId("lights")],
            writes: vec![ResourceId("clusters")],
            after: vec![PassId("clear")],
            enabled: true,
        });
        graph.add_pass(PassNode {
            id: PassId("opaque_forward_plus"),
            reads: vec![ResourceId("clusters"), ResourceId("lights")],
            writes: vec![ResourceId("color"), ResourceId("depth")],
            after: vec![PassId("cluster_cull")],
            enabled: true,
        });
        graph.add_pass(PassNode {
            id: PassId("transparent_2d"),
            reads: vec![ResourceId("color")],
            writes: vec![ResourceId("color")],
            after: vec![PassId("opaque_forward_plus")],
            enabled: true,
        });
        graph.add_pass(PassNode {
            id: PassId("overlay"),
            reads: vec![ResourceId("color"), ResourceId("depth")],
            writes: vec![ResourceId("color")],
            after: vec![PassId("transparent_2d")],
            enabled: true,
        });
        graph.add_pass(PassNode {
            id: PassId("debug_blit"),
            reads: vec![
                ResourceId("color"),
                ResourceId("depth"),
                ResourceId("clusters"),
            ],
            writes: vec![ResourceId("color")],
            after: vec![PassId("overlay")],
            enabled: false,
        });
        graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedules_default_graph() {
        let graph = RenderGraph::default_forward_plus();
        let order = graph.schedule().unwrap();
        let clear = order.iter().position(|p| *p == PassId("clear")).unwrap();
        let opaque = order
            .iter()
            .position(|p| *p == PassId("opaque_forward_plus"))
            .unwrap();
        assert!(clear < opaque);
    }

    #[test]
    fn detects_cycles() {
        let mut g2 = RenderGraph::new();
        g2.add_pass(PassNode {
            id: PassId("a"),
            reads: vec![],
            writes: vec![],
            after: vec![PassId("b")],
            enabled: true,
        });
        g2.add_pass(PassNode {
            id: PassId("b"),
            reads: vec![],
            writes: vec![],
            after: vec![PassId("a")],
            enabled: true,
        });
        assert!(g2.schedule().is_err());
    }
}
