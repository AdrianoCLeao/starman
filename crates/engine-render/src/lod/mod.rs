#![allow(dead_code)]

//! LOD selection from authored distance bands + quality bias.

use bevy_ecs::prelude::Component;
use engine_assets::MeshHandle;

#[derive(Clone, Debug)]
pub struct LodLevel {
    pub max_distance: f32,
    pub mesh: MeshHandle,
}

#[derive(Component, Clone, Debug, Default)]
pub struct LodGroup {
    pub levels: Vec<LodLevel>,
}

impl LodGroup {
    pub fn select(&self, distance: f32, lod_bias: f32) -> Option<MeshHandle> {
        if self.levels.is_empty() {
            return None;
        }
        let d = (distance + lod_bias).max(0.0);
        let mut chosen = self.levels[0].mesh;
        for level in &self.levels {
            if d <= level.max_distance {
                return Some(level.mesh);
            }
            chosen = level.mesh;
        }
        Some(chosen)
    }
}

/// Parse simple MSFT_lod-style distance extras: list of (distance, mesh_index).
pub fn parse_lod_distances(extras: &[(f32, usize)]) -> Vec<(f32, usize)> {
    let mut v = extras.to_vec();
    v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_sorts_distances() {
        let parsed = parse_lod_distances(&[(10.0, 1), (2.0, 0)]);
        assert_eq!(parsed[0], (2.0, 0));
        assert_eq!(parsed[1], (10.0, 1));
    }

    #[test]
    fn empty_group_returns_none() {
        let g = LodGroup::default();
        assert!(g.select(5.0, 0.0).is_none());
    }
}
