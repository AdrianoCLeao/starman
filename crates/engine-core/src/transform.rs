use bevy_ecs::prelude::{Bundle, Changed, Component, Entity, Or, Query, RemovedComponents};
use bevy_reflect::Reflect;
use engine_math::glam::{Affine3A, Quat, Vec3};

#[derive(Component, Clone, Debug, Reflect, engine_reflect::RegisterReflect)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    #[engine_reflect(range(min = 0.001, max = 100.0))]
    pub scale: Vec3,
}

impl Transform {
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    pub fn from_xyz(x: f32, y: f32, z: f32) -> Self {
        Self {
            translation: Vec3::new(x, y, z),
            ..Self::IDENTITY
        }
    }

    pub fn from_translation(translation: Vec3) -> Self {
        Self {
            translation,
            ..Self::IDENTITY
        }
    }

    pub fn with_rotation(mut self, rotation: Quat) -> Self {
        self.rotation = rotation;
        self
    }

    pub fn with_scale(mut self, scale: Vec3) -> Self {
        self.scale = scale;
        self
    }

    /// Rotates so local -Z points at `target` (no-op when degenerate).
    pub fn looking_at(mut self, target: Vec3, up: Vec3) -> Self {
        let forward = (target - self.translation).normalize_or_zero();
        if forward.length_squared() < 1e-8 {
            return self;
        }
        let right = up.cross(-forward).normalize_or_zero();
        if right.length_squared() < 1e-8 {
            return self;
        }
        let up = (-forward).cross(right);
        self.rotation = Quat::from_mat3(&engine_math::glam::Mat3::from_cols(right, up, -forward));
        self
    }

    pub fn forward(&self) -> Vec3 {
        self.rotation * Vec3::NEG_Z
    }

    pub fn right(&self) -> Vec3 {
        self.rotation * Vec3::X
    }

    pub fn to_affine(&self) -> Affine3A {
        Affine3A::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

#[derive(Component, Clone, Debug)]
pub struct GlobalTransform(pub Affine3A);

impl Default for GlobalTransform {
    fn default() -> Self {
        Self(Affine3A::IDENTITY)
    }
}

impl GlobalTransform {
    pub fn from_translation(translation: Vec3) -> Self {
        Self(Affine3A::from_translation(translation))
    }

    pub fn translation(&self) -> Vec3 {
        self.0.translation.into()
    }

    pub fn rotation(&self) -> Quat {
        self.0.to_scale_rotation_translation().1
    }

    pub fn scale(&self) -> Vec3 {
        self.0.to_scale_rotation_translation().0
    }

    /// Decomposes into a [`Transform`] (world space).
    pub fn compute_transform(&self) -> Transform {
        let (scale, rotation, translation) = self.0.to_scale_rotation_translation();
        Transform {
            translation,
            rotation,
            scale,
        }
    }

    /// World-space forward (-Z), normalized.
    pub fn forward(&self) -> Vec3 {
        self.0.transform_vector3(Vec3::NEG_Z).normalize_or_zero()
    }

    pub fn right(&self) -> Vec3 {
        self.0.transform_vector3(Vec3::X).normalize_or_zero()
    }

    pub fn up(&self) -> Vec3 {
        self.0.transform_vector3(Vec3::Y).normalize_or_zero()
    }

    pub fn transform_point(&self, point: Vec3) -> Vec3 {
        self.0.transform_point3(point)
    }
}

#[derive(Component, Clone, Debug)]
pub struct Parent(pub Entity);

#[derive(Component, Clone, Debug, Default)]
pub struct Children(pub Vec<Entity>);

#[derive(Component, Clone, Debug, Reflect, engine_reflect::RegisterReflect, PartialEq, Eq)]
pub struct EntityName(pub String);

impl EntityName {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }
}

impl std::fmt::Display for EntityName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Default for EntityName {
    fn default() -> Self {
        Self("Entity".to_owned())
    }
}

#[derive(Bundle, Default)]
pub struct SpatialBundle {
    pub transform: Transform,
    pub global_transform: GlobalTransform,
}

#[derive(Bundle, Default)]
pub struct EditorEntityBundle {
    pub name: EntityName,
    pub transform: Transform,
    pub global_transform: GlobalTransform,
}

/// Recomputes [`GlobalTransform`] for every entity whose local transform,
/// parent link, or ancestor changed since the last run. Unchanged subtrees
/// are skipped entirely; a frame with no hierarchy/transform change costs
/// a single change-detection scan.
#[allow(clippy::type_complexity)]
pub fn propagate_transforms(
    changed: Query<Entity, Or<(Changed<Transform>, Changed<Parent>)>>,
    mut removed_parents: RemovedComponents<Parent>,
    locals: Query<(Entity, &Transform, Option<&Parent>)>,
    mut globals: Query<&mut GlobalTransform>,
) {
    use std::collections::{HashMap, HashSet};

    let mut dirty_roots: Vec<Entity> = changed.iter().collect();
    dirty_roots.extend(removed_parents.read());
    if dirty_roots.is_empty() {
        return;
    }

    let mut children_of: HashMap<Entity, Vec<Entity>> = HashMap::new();
    let mut nodes: HashMap<Entity, (Option<Entity>, Affine3A)> = HashMap::new();
    for (entity, local, parent) in &locals {
        let parent = parent.map(|value| value.0);
        if let Some(parent) = parent {
            children_of.entry(parent).or_default().push(entity);
        }
        nodes.insert(entity, (parent, local.to_affine()));
    }

    // Expand dirty roots to every descendant.
    let mut dirty: HashSet<Entity> = HashSet::with_capacity(dirty_roots.len());
    while let Some(entity) = dirty_roots.pop() {
        if !dirty.insert(entity) {
            continue;
        }
        if let Some(children) = children_of.get(&entity) {
            dirty_roots.extend(children.iter().copied());
        }
    }

    fn resolve(
        entity: Entity,
        nodes: &HashMap<Entity, (Option<Entity>, Affine3A)>,
        dirty: &HashSet<Entity>,
        cache: &mut HashMap<Entity, Affine3A>,
        visiting: &mut HashSet<Entity>,
        globals: &Query<&mut GlobalTransform>,
    ) -> Affine3A {
        if let Some(cached) = cache.get(&entity) {
            return *cached;
        }
        let Some((parent, local)) = nodes.get(&entity).copied() else {
            return globals
                .get(entity)
                .map(|global| global.0)
                .unwrap_or(Affine3A::IDENTITY);
        };
        if !visiting.insert(entity) {
            log::warn!(
                target: "engine::ecs",
                "Cycle detected in transform hierarchy at entity {:?}; treating as root",
                entity
            );
            return local;
        }
        let global = match parent {
            Some(parent) if dirty.contains(&parent) => {
                resolve(parent, nodes, dirty, cache, visiting, globals) * local
            }
            Some(parent) => {
                let parent_global = globals
                    .get(parent)
                    .map(|global| global.0)
                    .unwrap_or(Affine3A::IDENTITY);
                parent_global * local
            }
            None => local,
        };
        visiting.remove(&entity);
        cache.insert(entity, global);
        global
    }

    let mut cache = HashMap::with_capacity(dirty.len());
    let mut visiting = HashSet::new();
    for entity in dirty.iter().copied() {
        visiting.clear();
        let _ = resolve(entity, &nodes, &dirty, &mut cache, &mut visiting, &globals);
    }

    for (entity, global) in cache {
        if let Ok(mut current) = globals.get_mut(entity) {
            current.0 = global;
        }
    }
}

#[cfg(test)]
#[path = "transform_tests.rs"]
mod tests;
