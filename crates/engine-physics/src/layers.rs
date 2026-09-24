//! Named collision layers and the project collision matrix.

use bevy_ecs::prelude::Resource;
use rapier3d::prelude::{Group, InteractionGroups};

pub const MAX_LAYERS: usize = 32;

/// A set of layers (bit `i` = layer `i`), used by spatial query filters.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LayerMask(pub u32);

impl LayerMask {
    pub const ALL: Self = Self(u32::MAX);
    pub const NONE: Self = Self(0);

    pub fn from_layer(layer: u8) -> Self {
        Self(1u32 << (layer as u32 % MAX_LAYERS as u32))
    }

    pub fn with(self, layer: u8) -> Self {
        Self(self.0 | Self::from_layer(layer).0)
    }

    pub fn without(self, layer: u8) -> Self {
        Self(self.0 & !Self::from_layer(layer).0)
    }

    pub fn contains(self, layer: u8) -> bool {
        self.0 & Self::from_layer(layer).0 != 0
    }
}

impl Default for LayerMask {
    fn default() -> Self {
        Self::ALL
    }
}

/// The project's physics layers: names by index and, per layer, the mask
/// of layers it collides with. Pairs interact only when *both* rows allow
/// it, so an asymmetric matrix can never produce one-sided contacts.
#[derive(Resource, Clone, Debug, PartialEq)]
pub struct PhysicsLayers {
    names: Vec<String>,
    matrix: [u32; MAX_LAYERS],
}

impl Default for PhysicsLayers {
    fn default() -> Self {
        Self {
            names: vec!["default".to_owned()],
            matrix: [u32::MAX; MAX_LAYERS],
        }
    }
}

impl PhysicsLayers {
    /// From project settings: `names[i]` names layer `i`; `matrix[i]` is
    /// the mask layer `i` collides with (missing rows collide with all).
    pub fn new(names: &[String], matrix: &[u32]) -> Self {
        let mut rows = [u32::MAX; MAX_LAYERS];
        for (row, value) in rows.iter_mut().zip(matrix) {
            *row = *value;
        }
        let mut names: Vec<String> = names.iter().take(MAX_LAYERS).cloned().collect();
        if names.is_empty() {
            names.push("default".to_owned());
        }
        Self {
            names,
            matrix: rows,
        }
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    pub fn name(&self, layer: u8) -> Option<&str> {
        self.names.get(layer as usize).map(String::as_str)
    }

    pub fn index_of(&self, name: &str) -> Option<u8> {
        self.names.iter().position(|n| n == name).map(|i| i as u8)
    }

    /// Mask of the named layers (unknown names are ignored).
    pub fn mask_of<'a>(&self, names: impl IntoIterator<Item = &'a str>) -> LayerMask {
        LayerMask(
            names
                .into_iter()
                .filter_map(|name| self.index_of(name))
                .fold(0, |mask, layer| mask | LayerMask::from_layer(layer).0),
        )
    }

    /// Layers `layer` collides with.
    pub fn collides_with(&self, layer: u8) -> LayerMask {
        LayerMask(self.matrix[layer as usize % MAX_LAYERS])
    }

    pub fn set_collides(&mut self, a: u8, b: u8, collide: bool) {
        let (a, b) = (a as usize % MAX_LAYERS, b as usize % MAX_LAYERS);
        for (row, bit) in [(a, b), (b, a)] {
            if collide {
                self.matrix[row] |= 1 << bit;
            } else {
                self.matrix[row] &= !(1 << bit);
            }
        }
    }

    pub fn interact(&self, a: u8, b: u8) -> bool {
        self.collides_with(a).contains(b) && self.collides_with(b).contains(a)
    }

    /// Rapier interaction groups for a collider on `layer`.
    pub fn groups(&self, layer: u8) -> InteractionGroups {
        InteractionGroups::new(
            Group::from_bits_retain(LayerMask::from_layer(layer).0),
            Group::from_bits_retain(self.collides_with(layer).0),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_rows_default_to_everything_and_pairs_need_both_sides() {
        let names: Vec<String> = ["default", "player", "enemy", "trigger"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let mut layers = PhysicsLayers::new(&names, &[u32::MAX, !0b1000]);
        assert!(layers.interact(0, 3));
        assert!(!layers.interact(1, 3), "player row excludes triggers");
        assert!(!layers.interact(3, 1), "symmetric by construction");
        layers.set_collides(1, 3, true);
        assert!(layers.interact(1, 3));
        assert_eq!(
            layers.mask_of(["player", "enemy", "nope"]),
            LayerMask(0b110)
        );
        let groups = layers.groups(2);
        assert_eq!(groups.memberships.bits(), 0b100);
    }
}
