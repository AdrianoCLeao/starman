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

    /// Adds `pass` and makes every node in `before` depend on it.
    pub fn add_pass_before(&mut self, pass: PassNode, before: &[PassId]) {
        let id = pass.id;
        self.passes.retain(|existing| existing.id != id);
        self.passes.push(pass);
        for target in before {
            if let Some(node) = self.passes.iter_mut().find(|p| p.id == *target) {
                if !node.after.contains(&id) {
                    node.after.push(id);
                }
            }
        }
    }

    pub fn contains(&self, id: PassId) -> bool {
        self.passes.iter().any(|p| p.id == id)
    }

    pub fn is_enabled(&self, id: PassId) -> bool {
        self.passes.iter().any(|p| p.id == id && p.enabled)
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

    /// The built-in Forward+ / HDR graph (M6). Resource names document
    /// data flow for inspection; ordering comes from `after`.
    pub fn default_forward_plus() -> Self {
        let mut graph = Self::new();
        for (id, kind) in [
            (ResourceId("hdr_color"), ResourceKind::ColorTarget),
            (ResourceId("output"), ResourceKind::ColorTarget),
            (ResourceId("depth"), ResourceKind::Depth),
            (ResourceId("velocity"), ResourceKind::Texture),
            (ResourceId("ssao"), ResourceKind::Texture),
            (ResourceId("id_buffer"), ResourceKind::Texture),
            (ResourceId("lights"), ResourceKind::Buffer),
            (ResourceId("clusters"), ResourceKind::Buffer),
            (ResourceId("instances"), ResourceKind::Buffer),
            (ResourceId("shadow_csm"), ResourceKind::Texture),
            (ResourceId("shadow_local"), ResourceKind::Texture),
            (ResourceId("environment"), ResourceKind::Texture),
            (ResourceId("probes"), ResourceKind::Texture),
            (ResourceId("taa_history"), ResourceKind::Texture),
            (ResourceId("bloom_chain"), ResourceKind::Texture),
        ] {
            graph.add_resource(GraphResourceDecl { id, kind });
        }

        let add = |graph: &mut RenderGraph,
                   id: &'static str,
                   after: &[&'static str],
                   reads: &[&'static str],
                   writes: &[&'static str],
                   enabled: bool| {
            graph.add_pass(PassNode {
                id: PassId(id),
                reads: reads.iter().map(|r| ResourceId(r)).collect(),
                writes: writes.iter().map(|w| ResourceId(w)).collect(),
                after: after.iter().map(|p| PassId(p)).collect(),
                enabled,
            });
        };

        add(
            &mut graph,
            "shadow_csm",
            &[],
            &["instances"],
            &["shadow_csm"],
            true,
        );
        add(
            &mut graph,
            "shadow_local",
            &[],
            &["instances"],
            &["shadow_local"],
            true,
        );
        add(
            &mut graph,
            "probe_bake",
            &["shadow_csm", "shadow_local"],
            &[
                "instances",
                "lights",
                "shadow_csm",
                "shadow_local",
                "environment",
            ],
            &["probes"],
            true,
        );
        add(
            &mut graph,
            "prepass",
            &[],
            &["instances"],
            &["depth", "velocity"],
            true,
        );
        add(
            &mut graph,
            "ssao",
            &["prepass"],
            &["depth"],
            &["ssao"],
            false,
        );
        add(
            &mut graph,
            "opaque",
            &[
                "prepass",
                "ssao",
                "shadow_csm",
                "shadow_local",
                "probe_bake",
            ],
            &[
                "instances",
                "lights",
                "clusters",
                "shadow_csm",
                "shadow_local",
                "environment",
                "probes",
                "ssao",
                "depth",
            ],
            &["hdr_color"],
            true,
        );
        add(
            &mut graph,
            "skybox",
            &["opaque"],
            &["environment", "depth"],
            &["hdr_color"],
            true,
        );
        add(
            &mut graph,
            "transparent_3d",
            &["skybox"],
            &["instances", "lights", "clusters", "depth"],
            &["hdr_color"],
            true,
        );
        add(
            &mut graph,
            "sprites_2d",
            &["transparent_3d"],
            &[],
            &["hdr_color"],
            true,
        );
        add(
            &mut graph,
            "taa",
            &["sprites_2d"],
            &["hdr_color", "velocity", "taa_history"],
            &["hdr_color", "taa_history"],
            false,
        );
        add(
            &mut graph,
            "bloom",
            &["taa", "sprites_2d"],
            &["hdr_color"],
            &["bloom_chain"],
            false,
        );
        add(
            &mut graph,
            "tonemap",
            &["taa", "bloom", "sprites_2d"],
            &["hdr_color", "bloom_chain"],
            &["output"],
            true,
        );
        add(
            &mut graph,
            "overdraw",
            &["tonemap"],
            &["instances"],
            &["output"],
            false,
        );
        add(
            &mut graph,
            "debug_view",
            &["tonemap", "overdraw"],
            &["depth", "clusters"],
            &["output"],
            false,
        );
        add(
            &mut graph,
            "debug_lines",
            &["tonemap", "debug_view"],
            &["depth"],
            &["output"],
            true,
        );
        add(
            &mut graph,
            "picking",
            &["prepass"],
            &["instances"],
            &["id_buffer"],
            false,
        );
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
        let position = |name| order.iter().position(|p| *p == PassId(name)).unwrap();
        assert!(position("prepass") < position("opaque"));
        assert!(position("shadow_csm") < position("opaque"));
        assert!(position("opaque") < position("skybox"));
        assert!(position("skybox") < position("transparent_3d"));
        assert!(position("transparent_3d") < position("tonemap"));
        assert!(position("tonemap") < position("debug_lines"));
    }

    #[test]
    fn extension_nodes_slot_between_builtins() {
        let mut graph = RenderGraph::default_forward_plus();
        graph.add_pass_before(
            PassNode {
                id: PassId("particles"),
                reads: vec![],
                writes: vec![],
                after: vec![PassId("transparent_3d")],
                enabled: true,
            },
            &[PassId("sprites_2d")],
        );
        let order = graph.schedule().unwrap();
        let position = |name| order.iter().position(|p| *p == PassId(name)).unwrap();
        assert!(position("transparent_3d") < position("particles"));
        assert!(position("particles") < position("sprites_2d"));
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
