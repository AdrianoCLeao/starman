//! Spatial queries: raycasts, shape casts, overlaps and point projection
//! with layer filters and entity exclusion.

use bevy_ecs::prelude::{Entity, Res};
use bevy_ecs::system::SystemParam;
use engine_math::{Quat, Vec3};
use rapier3d::parry::query::ShapeCastOptions;
use rapier3d::prelude::{
    ColliderHandle, Point, QueryFilter, QueryFilterFlags, Ray, Real, SharedShape,
};
use std::collections::HashMap;

use crate::layers::LayerMask;
use crate::mapping::{ColliderEntityMap3D, PhysicsEntityHandles3D};
use crate::pose::{from_vector, to_isometry, to_vector};
use crate::world3d::PhysicsWorld3D;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RaycastHit {
    /// The collider's entity.
    pub entity: Entity,
    /// The entity owning the collider's rigid body, if any.
    pub body: Option<Entity>,
    pub point: Vec3,
    pub normal: Vec3,
    pub distance: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShapeCastHit {
    pub entity: Entity,
    pub body: Option<Entity>,
    /// Distance travelled along the direction until first contact.
    pub distance: f32,
    /// Contact point on the hit collider.
    pub point: Vec3,
    /// Contact normal on the hit collider.
    pub normal: Vec3,
}

/// Shapes usable in casts and overlaps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QueryShape {
    Sphere { radius: f32 },
    Box { half_extents: Vec3 },
    Capsule { half_height: f32, radius: f32 },
}

impl QueryShape {
    fn to_shared(self) -> SharedShape {
        match self {
            Self::Sphere { radius } => SharedShape::ball(radius.max(0.0001)),
            Self::Box { half_extents } => SharedShape::cuboid(
                half_extents.x.max(0.0001),
                half_extents.y.max(0.0001),
                half_extents.z.max(0.0001),
            ),
            Self::Capsule {
                half_height,
                radius,
            } => SharedShape::capsule_y(half_height.max(0.0001), radius.max(0.0001)),
        }
    }
}

/// Which colliders a query may hit.
#[derive(Clone, Debug, PartialEq)]
pub struct SpatialQueryFilter {
    pub layers: LayerMask,
    /// Entities to ignore; a body entity excludes all its colliders.
    pub exclude: Vec<Entity>,
    pub include_sensors: bool,
}

impl Default for SpatialQueryFilter {
    fn default() -> Self {
        Self {
            layers: LayerMask::ALL,
            exclude: Vec::new(),
            include_sensors: false,
        }
    }
}

impl SpatialQueryFilter {
    pub fn with_layers(mut self, layers: LayerMask) -> Self {
        self.layers = layers;
        self
    }

    pub fn excluding(mut self, entity: Entity) -> Self {
        self.exclude.push(entity);
        self
    }

    pub fn with_sensors(mut self, include: bool) -> Self {
        self.include_sensors = include;
        self
    }
}

/// Read-only query access over a physics world (usable from systems via
/// [`SpatialQuery`] or from exclusive/scripting contexts directly).
#[derive(Clone, Copy)]
pub struct PhysicsQueries<'a> {
    pub physics: &'a PhysicsWorld3D,
    pub colliders: &'a ColliderEntityMap3D,
    pub handles: &'a PhysicsEntityHandles3D,
}

impl<'a> PhysicsQueries<'a> {
    pub fn new(
        physics: &'a PhysicsWorld3D,
        colliders: &'a ColliderEntityMap3D,
        handles: &'a PhysicsEntityHandles3D,
    ) -> Self {
        Self {
            physics,
            colliders,
            handles,
        }
    }

    fn body_of(&self, handle: ColliderHandle, entity: Entity) -> Option<Entity> {
        self.physics
            .collider_set
            .get(handle)
            .and_then(|collider| collider.parent())
            .and_then(|body| self.handles.body_entity(body))
            .or_else(|| self.handles.body(entity).map(|_| entity))
    }

    fn with_filter<R>(&self, filter: &SpatialQueryFilter, run: impl FnOnce(QueryFilter) -> R) -> R {
        let excluded_bodies: Vec<_> = filter
            .exclude
            .iter()
            .filter_map(|entity| self.handles.body(*entity))
            .collect();
        let excluded_colliders: Vec<_> = filter
            .exclude
            .iter()
            .filter_map(|entity| self.handles.collider(*entity))
            .collect();
        let layers = filter.layers.0;
        // Layer filtering tests membership only: a query for "enemies"
        // must hit enemies even if their own matrix row ignores raycasts.
        let predicate = |handle: ColliderHandle, collider: &rapier3d::prelude::Collider| {
            collider.collision_groups().memberships.bits() & layers != 0
                && !excluded_colliders.contains(&handle)
                && collider
                    .parent()
                    .is_none_or(|body| !excluded_bodies.contains(&body))
        };
        let mut query = QueryFilter::new().predicate(&predicate);
        if !filter.include_sensors {
            query.flags |= QueryFilterFlags::EXCLUDE_SENSORS;
        }
        run(query)
    }

    /// Closest hit along a ray.
    pub fn raycast(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_distance: f32,
        filter: &SpatialQueryFilter,
    ) -> Option<RaycastHit> {
        let dir = direction.normalize_or_zero();
        if dir == Vec3::ZERO || max_distance <= 0.0 {
            return None;
        }
        let ray = Ray::new(Point::from(to_vector(origin)), to_vector(dir));
        let (handle, hit) = self.with_filter(filter, |query| {
            self.physics.query_pipeline.cast_ray_and_get_normal(
                &self.physics.rigid_body_set,
                &self.physics.collider_set,
                &ray,
                max_distance,
                true,
                query,
            )
        })?;
        let entity = self.colliders.get(&handle)?;
        Some(RaycastHit {
            entity,
            body: self.body_of(handle, entity),
            point: from_vector(&ray.point_at(hit.time_of_impact).coords),
            normal: from_vector(&hit.normal),
            distance: hit.time_of_impact,
        })
    }

    /// Every hit along a ray, sorted by distance.
    pub fn raycast_all(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_distance: f32,
        filter: &SpatialQueryFilter,
    ) -> Vec<RaycastHit> {
        let dir = direction.normalize_or_zero();
        if dir == Vec3::ZERO || max_distance <= 0.0 {
            return Vec::new();
        }
        let ray = Ray::new(Point::from(to_vector(origin)), to_vector(dir));
        let mut hits = Vec::new();
        self.with_filter(filter, |query| {
            self.physics.query_pipeline.intersections_with_ray(
                &self.physics.rigid_body_set,
                &self.physics.collider_set,
                &ray,
                max_distance,
                true,
                query,
                |handle, hit| {
                    if let Some(entity) = self.colliders.get(&handle) {
                        hits.push(RaycastHit {
                            entity,
                            body: self.body_of(handle, entity),
                            point: from_vector(&ray.point_at(hit.time_of_impact).coords),
                            normal: from_vector(&hit.normal),
                            distance: hit.time_of_impact,
                        });
                    }
                    true
                },
            )
        });
        hits.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        hits
    }

    /// First hit of `shape` swept from `origin` along `direction`.
    pub fn shapecast(
        &self,
        shape: QueryShape,
        origin: Vec3,
        rotation: Quat,
        direction: Vec3,
        max_distance: f32,
        filter: &SpatialQueryFilter,
    ) -> Option<ShapeCastHit> {
        let dir = direction.normalize_or_zero();
        if dir == Vec3::ZERO || max_distance <= 0.0 {
            return None;
        }
        let shared = shape.to_shared();
        let pose = to_isometry((origin, rotation));
        let (handle, hit) = self.with_filter(filter, |query| {
            self.physics.query_pipeline.cast_shape(
                &self.physics.rigid_body_set,
                &self.physics.collider_set,
                &pose,
                &to_vector(dir),
                &*shared,
                ShapeCastOptions {
                    max_time_of_impact: max_distance,
                    stop_at_penetration: true,
                    ..ShapeCastOptions::default()
                },
                query,
            )
        })?;
        let entity = self.colliders.get(&handle)?;
        // The query pipeline reports witness/normal 1 on the hit collider,
        // in world space.
        Some(ShapeCastHit {
            entity,
            body: self.body_of(handle, entity),
            distance: hit.time_of_impact,
            point: from_vector(&hit.witness1.coords),
            normal: from_vector(&hit.normal1),
        })
    }

    /// Entities whose colliders intersect `shape` at the given pose.
    pub fn overlap(
        &self,
        shape: QueryShape,
        position: Vec3,
        rotation: Quat,
        filter: &SpatialQueryFilter,
    ) -> Vec<Entity> {
        let shared = shape.to_shared();
        let pose = to_isometry((position, rotation));
        let mut found = Vec::new();
        self.with_filter(filter, |query| {
            self.physics.query_pipeline.intersections_with_shape(
                &self.physics.rigid_body_set,
                &self.physics.collider_set,
                &pose,
                &*shared,
                query,
                |handle| {
                    if let Some(entity) = self.colliders.get(&handle) {
                        found.push(entity);
                    }
                    true
                },
            )
        });
        found
    }

    /// Closest point on any collider to `point` (and whether `point` is inside).
    pub fn project_point(
        &self,
        point: Vec3,
        max_distance: f32,
        filter: &SpatialQueryFilter,
    ) -> Option<(Entity, Vec3, bool)> {
        let target = Point::from(to_vector(point));
        let (handle, projection) = self.with_filter(filter, |query| {
            self.physics.query_pipeline.project_point(
                &self.physics.rigid_body_set,
                &self.physics.collider_set,
                &target,
                true,
                query,
            )
        })?;
        let projected = from_vector(&projection.point.coords);
        if projected.distance(point) > max_distance {
            return None;
        }
        Some((
            self.colliders.get(&handle)?,
            projected,
            projection.is_inside,
        ))
    }
}

/// System parameter for spatial queries.
#[derive(SystemParam)]
pub struct SpatialQuery<'w> {
    physics: Option<Res<'w, PhysicsWorld3D>>,
    colliders: Option<Res<'w, ColliderEntityMap3D>>,
    handles: Option<Res<'w, PhysicsEntityHandles3D>>,
}

impl SpatialQuery<'_> {
    /// `None` when physics is not installed.
    pub fn get(&self) -> Option<PhysicsQueries<'_>> {
        Some(PhysicsQueries::new(
            self.physics.as_deref()?,
            self.colliders.as_deref()?,
            self.handles.as_deref()?,
        ))
    }
}

/// Closest ray hit against every collider (editor picking, legacy API).
pub fn raycast(
    origin: Vec3,
    direction: Vec3,
    max_distance: f32,
    physics: &PhysicsWorld3D,
    entity_map: &HashMap<ColliderHandle, Entity>,
) -> Option<RaycastHit> {
    let max_distance: Real = max_distance.max(0.0);
    let dir = direction.normalize_or_zero();
    if max_distance <= 0.0 || dir == Vec3::ZERO {
        return None;
    }
    let ray = Ray::new(Point::from(to_vector(origin)), to_vector(dir));
    let (handle, hit) = physics.query_pipeline.cast_ray_and_get_normal(
        &physics.rigid_body_set,
        &physics.collider_set,
        &ray,
        max_distance,
        true,
        QueryFilter::default(),
    )?;
    let entity = *entity_map.get(&handle)?;
    Some(RaycastHit {
        entity,
        body: None,
        point: from_vector(&ray.point_at(hit.time_of_impact).coords),
        normal: from_vector(&hit.normal),
        distance: hit.time_of_impact,
    })
}
