use super::{collect_descendants, despawn_recursive, is_ancestor_of, set_parent, set_parent_with};
use crate::{Children, GlobalTransform, Parent, Transform};
use bevy_ecs::world::World;
use engine_math::Vec3;

fn spawn(world: &mut World, x: f32) -> bevy_ecs::entity::Entity {
    world
        .spawn((Transform::from_xyz(x, 0.0, 0.0), GlobalTransform::default()))
        .id()
}

#[test]
fn set_parent_keeps_both_sides_coherent() {
    let mut world = World::new();
    let a = spawn(&mut world, 0.0);
    let b = spawn(&mut world, 0.0);
    let child = spawn(&mut world, 0.0);

    assert!(set_parent(&mut world, child, Some(a)));
    assert_eq!(world.get::<Children>(a).unwrap().0, vec![child]);
    assert!(set_parent(&mut world, child, Some(b)));
    assert!(world.get::<Children>(a).unwrap().0.is_empty());
    assert_eq!(world.get::<Children>(b).unwrap().0, vec![child]);
    assert_eq!(world.get::<Parent>(child).unwrap().0, b);
    assert!(set_parent(&mut world, child, None));
    assert!(world.get::<Parent>(child).is_none());
    assert!(world.get::<Children>(b).unwrap().0.is_empty());
}

#[test]
fn set_parent_rejects_cycles() {
    let mut world = World::new();
    let a = spawn(&mut world, 0.0);
    let b = spawn(&mut world, 0.0);
    assert!(set_parent(&mut world, b, Some(a)));
    assert!(is_ancestor_of(&world, a, b));
    assert!(!set_parent(&mut world, a, Some(b)));
    assert!(!set_parent(&mut world, a, Some(a)));
}

#[test]
fn keep_world_transform_recomputes_local() {
    let mut world = World::new();
    let parent = spawn(&mut world, 10.0);
    world.get_mut::<GlobalTransform>(parent).unwrap().0 =
        engine_math::glam::Affine3A::from_translation(Vec3::new(10.0, 0.0, 0.0));
    let child = spawn(&mut world, 3.0);
    world.get_mut::<GlobalTransform>(child).unwrap().0 =
        engine_math::glam::Affine3A::from_translation(Vec3::new(3.0, 0.0, 0.0));
    assert!(set_parent_with(&mut world, child, Some(parent), true));
    let local = world.get::<Transform>(child).unwrap().translation;
    assert!((local - Vec3::new(-7.0, 0.0, 0.0)).length() < 1e-5);
}

#[test]
fn despawn_recursive_removes_subtree_and_detaches() {
    let mut world = World::new();
    let root = spawn(&mut world, 0.0);
    let mid = spawn(&mut world, 0.0);
    let leaf = spawn(&mut world, 0.0);
    let other = spawn(&mut world, 0.0);
    set_parent(&mut world, mid, Some(root));
    set_parent(&mut world, leaf, Some(mid));
    set_parent(&mut world, root, Some(other));
    assert_eq!(collect_descendants(&world, root), vec![root, mid, leaf]);
    despawn_recursive(&mut world, root);
    assert!(world.get_entity(root).is_err());
    assert!(world.get_entity(leaf).is_err());
    assert!(world.get::<Children>(other).unwrap().0.is_empty());
}
