//! Physics debug visualisation through [`DebugDraw`] (category
//! `Physics`): colliders by body kind, sensors, AABBs, contacts, joints
//! and character controllers.

use bevy_ecs::prelude::*;
use engine_core::{DebugCategory, DebugColor, DebugDraw};
use engine_math::{Quat, Vec3};
use rapier3d::prelude::{Collider, Isometry, Real, TypedShape};

use crate::character::CharacterState;
use crate::components::RigidBodyHandle3D;
use crate::pose::{from_isometry, from_vector};
use crate::world3d::PhysicsWorld3D;

/// Which physics debug layers are drawn (when the `Physics` debug
/// category is enabled).
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct PhysicsDebugSettings {
    pub colliders: bool,
    pub aabbs: bool,
    pub contacts: bool,
    pub joints: bool,
    pub characters: bool,
    /// Triangles drawn per mesh collider at most (big level meshes).
    pub max_mesh_triangles: usize,
}

impl Default for PhysicsDebugSettings {
    fn default() -> Self {
        Self {
            colliders: true,
            aabbs: false,
            contacts: true,
            joints: true,
            characters: true,
            max_mesh_triangles: 4096,
        }
    }
}

const STATIC: DebugColor = [0.6, 0.6, 0.6, 1.0];
const DYNAMIC: DebugColor = [0.2, 0.9, 0.3, 1.0];
const SLEEPING: DebugColor = [0.1, 0.45, 0.15, 1.0];
const KINEMATIC: DebugColor = [0.3, 0.55, 1.0, 1.0];
const SENSOR: DebugColor = [1.0, 0.85, 0.1, 1.0];
const AABB: DebugColor = [0.8, 0.3, 0.8, 1.0];
const CONTACT: DebugColor = [1.0, 0.2, 0.2, 1.0];
const JOINT: DebugColor = [1.0, 0.55, 0.0, 1.0];
const GROUNDED: DebugColor = [0.2, 1.0, 1.0, 1.0];
const AIRBORNE: DebugColor = [1.0, 0.4, 0.8, 1.0];

fn collider_color(physics: &PhysicsWorld3D, collider: &Collider) -> DebugColor {
    if collider.is_sensor() {
        return SENSOR;
    }
    match collider
        .parent()
        .and_then(|body| physics.rigid_body_set.get(body))
    {
        None => STATIC,
        Some(body) if body.is_fixed() => STATIC,
        Some(body) if body.is_kinematic() => KINEMATIC,
        Some(body) if body.is_sleeping() => SLEEPING,
        Some(_) => DYNAMIC,
    }
}

fn draw_shape(
    draw: &mut DebugDraw,
    shape: TypedShape<'_>,
    pose: &Isometry<Real>,
    color: DebugColor,
    max_triangles: usize,
) {
    let (center, rotation) = from_isometry(pose);
    let to_world = |p: &rapier3d::prelude::Point<Real>| from_vector(&(pose * p).coords);
    match shape {
        TypedShape::Ball(ball) => draw.sphere(center, ball.radius, color),
        TypedShape::Cuboid(cuboid) => {
            draw.oriented_box(center, rotation, from_vector(&cuboid.half_extents), color)
        }
        TypedShape::Capsule(capsule) => {
            let a = to_world(&capsule.segment.a);
            let b = to_world(&capsule.segment.b);
            let axis = b - a;
            let axis_rotation = if axis.length_squared() > 1e-8 {
                Quat::from_rotation_arc(Vec3::Y, axis.normalize())
            } else {
                rotation
            };
            draw.capsule(
                (a + b) * 0.5,
                axis_rotation,
                axis.length() * 0.5,
                capsule.radius,
                color,
            );
        }
        TypedShape::Cylinder(cylinder) => draw_round(
            draw,
            center,
            rotation,
            cylinder.half_height,
            cylinder.radius,
            cylinder.radius,
            color,
        ),
        TypedShape::Cone(cone) => draw_round(
            draw,
            center,
            rotation,
            cone.half_height,
            cone.radius,
            0.0,
            color,
        ),
        TypedShape::TriMesh(mesh) => {
            let vertices = mesh.vertices();
            for triangle in mesh.indices().iter().take(max_triangles) {
                let [a, b, c] = triangle.map(|i| to_world(&vertices[i as usize]));
                draw.line(a, b, color);
                draw.line(b, c, color);
                draw.line(c, a, color);
            }
        }
        TypedShape::ConvexPolyhedron(hull) => {
            let points = hull.points();
            for edge in hull.edges() {
                let (a, b) = (edge.vertices[0] as usize, edge.vertices[1] as usize);
                draw.line(to_world(&points[a]), to_world(&points[b]), color);
            }
        }
        TypedShape::Compound(compound) => {
            for (sub_pose, sub_shape) in compound.shapes() {
                draw_shape(
                    draw,
                    sub_shape.as_typed_shape(),
                    &(pose * sub_pose),
                    color,
                    max_triangles,
                );
            }
        }
        _ => draw.cross(center, 0.25, color),
    }
}

/// Cylinder (`top_radius == bottom_radius`) or cone (`top_radius == 0`).
fn draw_round(
    draw: &mut DebugDraw,
    center: Vec3,
    rotation: Quat,
    half_height: f32,
    bottom_radius: f32,
    top_radius: f32,
    color: DebugColor,
) {
    let up = rotation * Vec3::Y;
    let top = center + up * half_height;
    let bottom = center - up * half_height;
    draw.circle(bottom, up, bottom_radius, color, 20);
    if top_radius > 0.0 {
        draw.circle(top, up, top_radius, color, 20);
    }
    for side in [Vec3::X, -Vec3::X, Vec3::Z, -Vec3::Z] {
        let side = rotation * side;
        draw.line(
            bottom + side * bottom_radius,
            top + side * top_radius,
            color,
        );
    }
}

/// Emits the physics debug geometry for this frame.
pub fn draw_physics_debug(
    physics: Option<Res<PhysicsWorld3D>>,
    settings: Option<Res<PhysicsDebugSettings>>,
    characters: Query<(&RigidBodyHandle3D, &CharacterState)>,
    mut draw: Option<ResMut<DebugDraw>>,
) {
    let (Some(physics), Some(draw)) = (physics, draw.as_deref_mut()) else {
        return;
    };
    if !draw.is_enabled(DebugCategory::Physics) {
        return;
    }
    let settings = settings.map(|s| *s).unwrap_or_default();

    for (_, collider) in physics.collider_set.iter() {
        if !collider.is_enabled() {
            continue;
        }
        if settings.colliders {
            let color = collider_color(&physics, collider);
            draw_shape(
                draw,
                collider.shape().as_typed_shape(),
                collider.position(),
                color,
                settings.max_mesh_triangles,
            );
        }
        if settings.aabbs {
            let aabb = collider.compute_aabb();
            draw.aabb(
                from_vector(&aabb.mins.coords),
                from_vector(&aabb.maxs.coords),
                AABB,
            );
        }
    }

    if settings.contacts {
        for pair in physics.narrow_phase.contact_pairs() {
            if !pair.has_any_active_contact {
                continue;
            }
            let Some(collider) = physics.collider_set.get(pair.collider1) else {
                continue;
            };
            let pose = collider.position();
            for manifold in &pair.manifolds {
                let normal = from_vector(&(pose * manifold.local_n1));
                for point in &manifold.points {
                    let world = from_vector(&(pose * point.local_p1).coords);
                    draw.cross(world, 0.05, CONTACT);
                    draw.line_overlay(world, world + normal * 0.25, CONTACT);
                }
            }
        }
    }

    if settings.joints {
        for (_, joint) in physics.impulse_joint_set.iter() {
            let (Some(b1), Some(b2)) = (
                physics.rigid_body_set.get(joint.body1),
                physics.rigid_body_set.get(joint.body2),
            ) else {
                continue;
            };
            let a = from_vector(&(b1.position() * joint.data.local_anchor1()).coords);
            let b = from_vector(&(b2.position() * joint.data.local_anchor2()).coords);
            draw.line_overlay(from_vector(b1.translation()), a, JOINT);
            draw.line_overlay(a, b, JOINT);
            draw.cross(a, 0.08, JOINT);
        }
    }

    if settings.characters {
        for (handle, state) in &characters {
            let Some(body) = physics.rigid_body_set.get(handle.0) else {
                continue;
            };
            let position = from_vector(body.translation());
            let color = if state.grounded { GROUNDED } else { AIRBORNE };
            draw.arrow(position, position + state.velocity * 0.25, color);
            draw.circle(position, Vec3::Y, 0.15, color, 12);
        }
    }
}
