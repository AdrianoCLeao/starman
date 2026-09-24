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

    /// Default M5 Forward+ / HDR graph.
    pub fn default_forward_plus() -> Self {
        let mut graph = Self::new();
        for (id, kind) in [
            (ResourceId("hdr_color"), ResourceKind::ColorTarget),
            (ResourceId("color"), ResourceKind::ColorTarget),
            (ResourceId("depth"), ResourceKind::Depth),
            (ResourceId("id_buffer"), ResourceKind::Texture),
            (ResourceId("lights"), ResourceKind::Buffer),
            (ResourceId("clusters"), ResourceKind::Buffer),
            (ResourceId("shadow_csm"), ResourceKind::Texture),
            (ResourceId("shadow_local"), ResourceKind::Texture),
            (ResourceId("velocity"), ResourceKind::Texture),
            (ResourceId("history"), ResourceKind::Texture),
        ] {
            graph.add_resource(GraphResourceDecl { id, kind });
        }

        let add =
            |graph: &mut RenderGraph, id: &'static str, after: &[&'static str], enabled: bool| {
                graph.add_pass(PassNode {
                    id: PassId(id),
                    reads: vec![],
                    writes: vec![],
                    after: after.iter().map(|p| PassId(p)).collect(),
                    enabled,
                });
            };

        add(&mut graph, "clear", &[], true);
        add(&mut graph, "shadow_csm", &["clear"], true);
        add(&mut graph, "shadow_local", &["clear"], true);
        add(&mut graph, "id_pick", &["clear"], false);
        add(&mut graph, "cluster_cull", &["clear"], true);
        add(
            &mut graph,
            "opaque_forward_plus",
            &["cluster_cull", "shadow_csm", "shadow_local"],
            true,
        );
        add(&mut graph, "skybox", &["opaque_forward_plus"], true);
        add(&mut graph, "transparent_2d", &["skybox"], true);
        add(&mut graph, "ssao", &["transparent_2d"], false);
        add(&mut graph, "bloom", &["ssao", "transparent_2d"], false);
        add(&mut graph, "taa", &["bloom", "transparent_2d"], false);
        add(
            &mut graph,
            "tonemap_aces",
            &["taa", "bloom", "transparent_2d", "ssao"],
            true,
        );
        add(&mut graph, "overlay", &["tonemap_aces"], true);
        add(&mut graph, "debug_blit", &["overlay"], false);
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
        let tonemap = order
            .iter()
            .position(|p| *p == PassId("tonemap_aces"))
            .unwrap();
        assert!(clear < tonemap);
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
