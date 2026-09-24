use super::{FrameHooks, GameClock, GameRuntime, RuntimePlugin, MAX_FIXED_STEPS_PER_FRAME};
use crate::schedule::{FixedSet, ScheduleKind, UpdateSet};
use crate::{FrameTime, GlobalTransform, Transform};
use bevy_ecs::event::{Event, EventReader, EventWriter, Events};
use bevy_ecs::prelude::{IntoSystemConfigs, Res, ResMut, Resource, World};

#[derive(Resource, Default)]
struct Trace(Vec<&'static str>);

#[derive(Event, Clone, Copy, Debug, PartialEq)]
struct Ping(u32);

#[derive(Resource, Default)]
struct Received(Vec<u32>);

struct TracePlugin;

impl RuntimePlugin for TracePlugin {
    fn name(&self) -> &'static str {
        "test::trace"
    }

    fn build(&self, runtime: &mut GameRuntime) {
        runtime
            .init_resource::<Trace>()
            .add_systems(
                ScheduleKind::FixedUpdate,
                (
                    (|mut t: ResMut<Trace>| t.0.push("fixed-ai")).in_set(FixedSet::Ai),
                    (|mut t: ResMut<Trace>| t.0.push("fixed-scripts")).in_set(FixedSet::Scripts),
                ),
            )
            .add_systems(
                ScheduleKind::Update,
                (
                    (|mut t: ResMut<Trace>| t.0.push("ui")).in_set(UpdateSet::UiLayout),
                    (|mut t: ResMut<Trace>| t.0.push("input")).in_set(UpdateSet::InputActions),
                ),
            );
    }
}

#[test]
fn system_sets_order_plugin_systems() {
    let mut runtime = GameRuntime::with_fixed_timestep(0.5);
    runtime.add_plugin(TracePlugin);
    runtime.step(0.5);
    assert_eq!(
        runtime.world.resource::<Trace>().0,
        vec!["fixed-scripts", "fixed-ai", "input", "ui"]
    );
}

#[test]
fn duplicate_plugins_are_installed_once() {
    let mut runtime = GameRuntime::new();
    runtime.add_plugin(TracePlugin).add_plugin(TracePlugin);
    assert_eq!(runtime.plugins(), &["test::trace"]);
    assert!(runtime.has_plugin("test::trace"));
}

#[test]
fn fixed_steps_accumulate_and_are_capped() {
    let mut runtime = GameRuntime::with_fixed_timestep(0.1);
    assert_eq!(runtime.step(0.25).fixed_steps, 2);
    // 0.05 left over + 0.05 → one more step.
    assert_eq!(runtime.step(0.05).fixed_steps, 1);
    assert_eq!(runtime.step(10.0).fixed_steps, MAX_FIXED_STEPS_PER_FRAME);
    // The backlog was dropped, not carried over.
    assert_eq!(runtime.step(0.0).fixed_steps, 0);
}

#[test]
fn paused_clock_stops_game_time_but_not_real_time() {
    let mut runtime = GameRuntime::with_fixed_timestep(0.1);
    runtime.world.resource_mut::<GameClock>().paused = true;
    let frame = runtime.step(0.2);
    assert_eq!(frame.fixed_steps, 0);
    let time = *runtime.world.resource::<FrameTime>();
    assert_eq!(time.delta_seconds, 0.0);
    assert!((time.real_delta_seconds - 0.2).abs() < 1e-6);

    let mut clock = runtime.world.resource_mut::<GameClock>();
    clock.paused = false;
    clock.scale = 0.5;
    let frame = runtime.step(0.2);
    assert_eq!(frame.fixed_steps, 1);
    assert!((frame.delta_seconds - 0.1).abs() < 1e-6);
}

#[test]
fn events_sent_in_fixed_are_visible_in_update_then_expire() {
    let mut runtime = GameRuntime::with_fixed_timestep(0.1);
    runtime
        .add_event::<Ping>()
        .init_resource::<Received>()
        .add_systems(
            ScheduleKind::FixedUpdate,
            (|mut writer: EventWriter<Ping>, time: Res<FrameTime>| {
                writer.send(Ping(time.frame_count as u32));
            })
            .in_set(FixedSet::Gameplay),
        )
        .add_systems(
            ScheduleKind::Update,
            (|mut reader: EventReader<Ping>, mut received: ResMut<Received>| {
                received.0.extend(reader.read().map(|ping| ping.0));
            })
            .in_set(UpdateSet::Gameplay),
        );
    runtime.step(0.1);
    runtime.step(0.0);
    runtime.step(0.0);
    assert_eq!(runtime.world.resource::<Received>().0, vec![1]);
    assert!(runtime.world.resource::<Events<Ping>>().is_empty());
}

#[test]
fn pre_render_propagates_transforms() {
    let mut runtime = GameRuntime::new();
    let entity = runtime
        .world
        .spawn((
            Transform::from_xyz(1.0, 2.0, 3.0),
            GlobalTransform::default(),
        ))
        .id();
    runtime.step(0.0);
    let global = runtime.world.get::<GlobalTransform>(entity).unwrap();
    assert_eq!(global.translation(), engine_math::Vec3::new(1.0, 2.0, 3.0));
}

struct CountingHooks {
    begin: u32,
    fixed: u32,
    update: u32,
    render: u32,
}

impl FrameHooks for CountingHooks {
    fn begin_frame(&mut self, _world: &mut World) -> crate::Result<()> {
        self.begin += 1;
        Ok(())
    }

    fn after_fixed(&mut self, _world: &mut World, _dt: f32) -> crate::Result<()> {
        self.fixed += 1;
        Ok(())
    }

    fn after_update(&mut self, _world: &mut World, _dt: f32) -> crate::Result<()> {
        self.update += 1;
        Ok(())
    }

    fn render(&mut self, _world: &mut World, alpha: f32) -> crate::Result<()> {
        assert!((0.0..=1.0).contains(&alpha));
        self.render += 1;
        Ok(())
    }
}

#[test]
fn hooks_are_called_around_schedules() {
    let mut runtime = GameRuntime::with_fixed_timestep(0.1);
    let mut hooks = CountingHooks {
        begin: 0,
        fixed: 0,
        update: 0,
        render: 0,
    };
    runtime.run_frame(0.35, &mut hooks).unwrap();
    assert_eq!(
        (hooks.begin, hooks.fixed, hooks.update, hooks.render),
        (1, 3, 1, 1)
    );
}
