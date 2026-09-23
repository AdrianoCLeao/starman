use std::collections::HashSet;

use bevy_ecs::entity::Entity;

#[derive(Default, Clone, Debug)]
pub struct Selection {
    primary: Option<Entity>,
    secondary: HashSet<Entity>,
}

impl Selection {
    pub fn select_single(&mut self, entity: Entity) {
        self.primary = Some(entity);
        self.secondary.clear();
    }

    pub fn deselect(&mut self) {
        self.primary = None;
        self.secondary.clear();
    }

    /// Ctrl/Cmd-click: toggle membership. Becomes primary if nothing selected.
    pub fn toggle(&mut self, entity: Entity) {
        if self.primary == Some(entity) {
            if let Some(next) = self.secondary.iter().next().copied() {
                self.secondary.remove(&next);
                self.primary = Some(next);
            } else {
                self.primary = None;
            }
            return;
        }
        if self.secondary.remove(&entity) {
            return;
        }
        if self.primary.is_none() {
            self.primary = Some(entity);
        } else {
            self.secondary.insert(entity);
        }
    }

    /// Shift-click range is caller-defined; this adds without clearing.
    pub fn add(&mut self, entity: Entity) {
        if self.primary.is_none() {
            self.primary = Some(entity);
            return;
        }
        if self.primary != Some(entity) {
            self.secondary.insert(entity);
        }
    }

    pub fn primary(&self) -> Option<Entity> {
        self.primary
    }

    pub fn has_selection(&self) -> bool {
        self.primary.is_some() || !self.secondary.is_empty()
    }

    pub fn len(&self) -> usize {
        self.secondary.len() + usize::from(self.primary.is_some())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn contains(&self, entity: Entity) -> bool {
        self.primary == Some(entity) || self.secondary.contains(&entity)
    }

    pub fn all(&self) -> impl Iterator<Item = Entity> + '_ {
        self.primary
            .iter()
            .copied()
            .chain(self.secondary.iter().copied())
    }

    pub fn secondary(&self) -> &HashSet<Entity> {
        &self.secondary
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::world::World;

    #[test]
    fn toggle_builds_multi_selection() {
        let mut world = World::new();
        let a = world.spawn_empty().id();
        let b = world.spawn_empty().id();
        let mut selection = Selection::default();
        selection.select_single(a);
        selection.toggle(b);
        assert_eq!(selection.len(), 2);
        assert!(selection.contains(a));
        assert!(selection.contains(b));
    }
}
