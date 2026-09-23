//! Ownership of components/systems registered by a dynamic plugin or Lua.

use std::collections::HashMap;

use crate::loader::PluginId;

/// Tracks which dynamic registrations belong to which plugin/script owner.
#[derive(Default, Debug)]
pub struct DynamicRegistration {
    /// Schedule name → system names registered by each owner.
    pub systems_by_owner: HashMap<PluginId, Vec<(String, String)>>,
    /// Component type names registered by each owner (metadata only in M3).
    pub components_by_owner: HashMap<PluginId, Vec<String>>,
}

impl DynamicRegistration {
    pub fn record_system(&mut self, owner: PluginId, schedule: &str, name: &str) {
        self.systems_by_owner
            .entry(owner)
            .or_default()
            .push((schedule.to_owned(), name.to_owned()));
    }

    pub fn record_component(&mut self, owner: PluginId, component: &str) {
        self.components_by_owner
            .entry(owner)
            .or_default()
            .push(component.to_owned());
    }

    pub fn take_owner_systems(&mut self, owner: PluginId) -> Vec<(String, String)> {
        self.systems_by_owner.remove(&owner).unwrap_or_default()
    }

    pub fn clear_owner(&mut self, owner: PluginId) {
        self.systems_by_owner.remove(&owner);
        self.components_by_owner.remove(&owner);
    }
}
