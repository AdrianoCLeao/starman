//! Stable path identifying which nested instance owns an override.

use engine_core::EntityId;
use serde::{Deserialize, Serialize};

/// Sequence of instance-root [`EntityId`]s from the opened scene document
/// down to the instance that stores an override. Deterministic and
/// merge-friendly.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, Default)]
pub struct InstancePath {
    pub roots: Vec<EntityId>,
}

impl InstancePath {
    pub fn new(roots: impl IntoIterator<Item = EntityId>) -> Self {
        Self {
            roots: roots.into_iter().collect(),
        }
    }

    pub fn push(&self, root: EntityId) -> Self {
        let mut roots = self.roots.clone();
        roots.push(root);
        Self { roots }
    }

    pub fn parent(&self) -> Option<InstancePath> {
        if self.roots.is_empty() {
            return None;
        }
        let mut roots = self.roots.clone();
        roots.pop();
        Some(Self { roots })
    }

    pub fn leaf(&self) -> Option<EntityId> {
        self.roots.last().copied()
    }

    pub fn is_empty(&self) -> bool {
        self.roots.is_empty()
    }
}

impl std::fmt::Display for InstancePath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.roots.is_empty() {
            return write!(f, "/");
        }
        for (index, id) in self.roots.iter().enumerate() {
            if index > 0 {
                write!(f, "/")?;
            }
            write!(f, "{id}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_push_and_parent_round_trip() {
        let a = EntityId::new_v4();
        let b = EntityId::new_v4();
        let path = InstancePath::new([a]).push(b);
        assert_eq!(path.roots, vec![a, b]);
        assert_eq!(path.parent().unwrap().roots, vec![a]);
    }

    #[test]
    fn serde_round_trip() {
        let path = InstancePath::new([EntityId::new_v4(), EntityId::new_v4()]);
        let encoded = ron::to_string(&path).unwrap();
        let decoded: InstancePath = ron::from_str(&encoded).unwrap();
        assert_eq!(path, decoded);
    }
}
