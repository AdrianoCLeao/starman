//! Host bus unit tests.

use bevy_ecs::world::World;
use engine_reflect::{ComponentRegistry, ReflectMetadataRegistry, ReflectTypeRegistry};

use engine_plugin::{
    HostBus, HostCommand, HostQuery, PermissionGuard, PluginId, ProjectPermissions, ScheduleName,
};

#[test]
fn spawn_and_set_transform_field_roundtrip() {
    let mut world = World::new();
    let mut components = ComponentRegistry::default();
    let mut types = ReflectTypeRegistry::default();
    let mut meta = ReflectMetadataRegistry::default();
    engine_core::register_core_reflection_types(&mut types, &mut components, &mut meta);

    let guard = PermissionGuard::new("/tmp/proj", ProjectPermissions::default());
    let mut bus = HostBus::new(guard);

    let handle = bus
        .execute(
            &mut world,
            &components,
            HostCommand::Spawn {
                name: Some("demo".into()),
            },
        )
        .unwrap();
    let id = handle.as_u64().unwrap();

    bus.execute(
        &mut world,
        &components,
        HostCommand::SetField {
            entity: starman_plugin_sdk::EntityHandle(id),
            component: "Transform".into(),
            field_path: "translation.y".into(),
            value: serde_json::json!(1.5),
        },
    )
    .unwrap();

    let value = bus
        .query(
            &world,
            &components,
            &types,
            HostQuery::GetField {
                entity: starman_plugin_sdk::EntityHandle(id),
                component: "Transform".into(),
                field_path: "translation.y".into(),
            },
        )
        .unwrap();
    assert!((value.as_f64().unwrap() - 1.5).abs() < 0.001);

    bus.execute(
        &mut world,
        &components,
        HostCommand::RegisterSystem {
            owner: PluginId(7),
            schedule: ScheduleName::Update,
            name: "tick".into(),
            callback_id: 1,
        },
    )
    .unwrap();
    assert_eq!(bus.registrations.systems_by_owner[&PluginId(7)].len(), 1);
    bus.registrations.clear_owner(PluginId(7));
    assert!(!bus.registrations.systems_by_owner.contains_key(&PluginId(7)));
}
