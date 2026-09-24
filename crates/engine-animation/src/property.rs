//! Property tracks: animated values written into reflected component
//! fields of the animated entity or its named descendants.

use std::collections::HashMap;

use bevy_ecs::prelude::*;
use bevy_reflect::{GetPath, PartialReflect};
use engine_core::{Children, EntityName};
use engine_math::{Quat, Vec2, Vec3, Vec4};
use engine_reflect::{ComponentDescriptor, ComponentRegistry};

use crate::clip::PropertyValue;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PropertyKey {
    pub entity: Entity,
    pub target: String,
    pub component: String,
    pub field: String,
}

/// Property values gathered this frame, blended by weight.
#[derive(Resource, Default, Debug)]
pub struct PropertyWrites {
    values: HashMap<PropertyKey, (PropertyValue, f32)>,
    order: Vec<PropertyKey>,
    /// Problems reported once per key (missing target/component/field).
    warned: std::collections::HashSet<PropertyKey>,
}

impl PropertyWrites {
    pub fn push(&mut self, key: PropertyKey, value: PropertyValue, weight: f32) {
        if weight <= 1e-5 {
            return;
        }
        match self.values.get_mut(&key) {
            Some((current, total)) => {
                *total += weight;
                *current = current.blend(value, weight / *total);
            }
            None => {
                self.order.push(key.clone());
                self.values.insert(key, (value, weight));
            }
        }
    }

    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    fn drain(&mut self) -> Vec<(PropertyKey, PropertyValue)> {
        let mut values = std::mem::take(&mut self.values);
        self.order
            .drain(..)
            .filter_map(|key| values.remove(&key).map(|(value, _)| (key, value)))
            .collect()
    }
}

/// Resolves `path` (`"A/B"`, names of descendants) from `root`.
pub fn resolve_target(world: &World, root: Entity, path: &str) -> Option<Entity> {
    let mut current = root;
    for name in path.split('/').filter(|part| !part.is_empty()) {
        let children = world.get::<Children>(current)?;
        current = children
            .0
            .iter()
            .copied()
            .find(|child| world.get::<EntityName>(*child).is_some_and(|n| n.0 == name))?;
    }
    Some(current)
}

fn find_component<'a>(
    registry: &'a ComponentRegistry,
    name: &str,
) -> Option<&'a ComponentDescriptor> {
    registry.by_name(name).or_else(|| {
        let suffix = format!("::{name}");
        registry
            .all()
            .iter()
            .find(|descriptor| descriptor.name.ends_with(&suffix))
    })
}

/// Writes `value` into `field`; supports scalar, glam vector/quaternion,
/// `[f32; N]` color and bool fields.
pub fn write_value(field: &mut dyn PartialReflect, value: PropertyValue) -> bool {
    macro_rules! set {
        ($ty:ty, $v:expr) => {
            if let Some(slot) = field.try_downcast_mut::<$ty>() {
                *slot = $v;
                return true;
            }
        };
    }
    match value {
        PropertyValue::Float(v) => {
            set!(f32, v);
            set!(f64, v as f64);
        }
        PropertyValue::Vec2(v) => {
            set!(Vec2, v);
            set!([f32; 2], v.to_array());
        }
        PropertyValue::Vec3(v) => {
            set!(Vec3, v);
            set!([f32; 3], v.to_array());
        }
        PropertyValue::Vec4(v) => {
            set!(Vec4, v);
            set!([f32; 4], v.to_array());
        }
        PropertyValue::Quat(v) => {
            set!(Quat, v);
        }
        PropertyValue::Bool(v) => {
            set!(bool, v);
        }
    }
    false
}

/// Applies this frame's property writes through reflection.
pub fn apply_property_writes(world: &mut World) {
    let writes = match world.get_resource_mut::<PropertyWrites>() {
        Some(mut writes) if !writes.is_empty() => writes.drain(),
        _ => return,
    };
    let Some(registry) = world.remove_resource::<ComponentRegistry>() else {
        return;
    };
    let mut failures = Vec::new();
    for (key, value) in writes {
        let Some(target) = resolve_target(world, key.entity, &key.target) else {
            failures.push((key, "target not found"));
            continue;
        };
        let Some(descriptor) = find_component(&registry, &key.component) else {
            failures.push((key, "component type is not registered"));
            continue;
        };
        let Some(component) = descriptor.get_reflect_mut(target, world) else {
            failures.push((key, "entity lacks the component"));
            continue;
        };
        let written = match component.reflect_path_mut(key.field.as_str()) {
            Ok(field) => write_value(field, value),
            Err(_) => false,
        };
        if !written {
            failures.push((key, "field missing or of another type"));
        }
    }
    world.insert_resource(registry);
    if failures.is_empty() {
        return;
    }
    if let Some(mut writes) = world.get_resource_mut::<PropertyWrites>() {
        for (key, reason) in failures {
            if writes.warned.insert(key.clone()) {
                log::warn!(
                    target: "engine::animation",
                    "property track {}:{}.{} on {:?}: {reason}",
                    key.target,
                    key.component,
                    key.field,
                    key.entity
                );
            }
        }
    }
}
