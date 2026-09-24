use super::*;
use crate::actions::tests_support::SAMPLE;
use crate::actions::InputSource;
use crate::rebind::RebindRequest;
use engine_core::GameRuntime;
use gilrs::{Axis, Button};
use winit::event::ElementState;
use winit::keyboard::KeyCode;

const DT: f32 = 1.0 / 60.0;

fn sample() -> InputActions {
    ron::from_str(SAMPLE).expect("sample parses")
}

struct Harness {
    input: InputState,
    actions: InputActions,
    players: LocalPlayers,
    user: InputUserSettingsStore,
    rebinding: Rebinding,
}

impl Harness {
    fn new(max_players: u8) -> Self {
        Self {
            input: InputState::default(),
            actions: sample(),
            players: LocalPlayers::new(max_players),
            user: InputUserSettingsStore::default(),
            rebinding: Rebinding::default(),
        }
    }

    /// Runs one frame with the given raw input and returns the emitted events.
    fn frame(&mut self, feed: impl FnOnce(&mut InputState)) -> Vec<Emitted> {
        self.input.begin_frame();
        feed(&mut self.input);
        let mut emitted = Vec::new();
        step_actions(
            &self.input,
            DT,
            Some(&self.actions),
            &mut self.players,
            &mut self.user,
            &mut self.rebinding,
            &mut |event| emitted.push(event),
        );
        emitted
    }
}

fn phases(events: &[Emitted], action: &str) -> Vec<ActionPhase> {
    events
        .iter()
        .filter_map(|event| match event {
            Emitted::Action(event) if event.action == action => Some(event.phase),
            _ => None,
        })
        .collect()
}

#[test]
fn press_and_release_drive_action_state_and_events() {
    let mut harness = Harness::new(1);
    let events = harness
        .frame(|input| input.process_key_input(KeyCode::Space, ElementState::Pressed, false));
    assert_eq!(
        phases(&events, "jump"),
        vec![ActionPhase::Started, ActionPhase::Performed]
    );
    assert!(harness.players.primary().just_pressed("jump"));

    let events = harness.frame(|_| {});
    assert!(phases(&events, "jump").is_empty());
    assert!(harness.players.primary().pressed("jump"));
    assert!(!harness.players.primary().just_pressed("jump"));

    harness.frame(|input| input.process_key_input(KeyCode::Space, ElementState::Released, false));
    assert!(harness.players.primary().just_released("jump"));
}

#[test]
fn hold_performs_after_duration_and_cancels_when_released_early() {
    let mut harness = Harness::new(1);
    let events =
        harness.frame(|input| input.process_key_input(KeyCode::KeyE, ElementState::Pressed, false));
    assert_eq!(phases(&events, "interact"), vec![ActionPhase::Started]);
    let events = harness
        .frame(|input| input.process_key_input(KeyCode::KeyE, ElementState::Released, false));
    assert_eq!(phases(&events, "interact"), vec![ActionPhase::Canceled]);

    harness.frame(|input| input.process_key_input(KeyCode::KeyE, ElementState::Pressed, false));
    let mut performed = false;
    for _ in 0..40 {
        let events = harness.frame(|_| {});
        performed |= phases(&events, "interact").contains(&ActionPhase::Performed);
    }
    assert!(performed, "hold for 0.5s performs");
}

#[test]
fn blocking_context_releases_lower_actions() {
    let mut harness = Harness::new(1);
    harness.frame(|input| input.process_key_input(KeyCode::KeyW, ElementState::Pressed, false));
    assert!(harness.players.primary().axis2d("move").y > 0.9);

    harness.players.primary_mut().push_context("menu");
    let events = harness.frame(|_| {});
    assert_eq!(phases(&events, "move"), vec![ActionPhase::Canceled]);
    assert_eq!(harness.players.primary().axis2d("move"), Vec2::ZERO);

    let events = harness
        .frame(|input| input.process_key_input(KeyCode::Enter, ElementState::Pressed, false));
    assert!(phases(&events, "ui_submit").contains(&ActionPhase::Performed));

    harness.players.primary_mut().pop_context("menu");
    harness.frame(|_| {});
    assert!(
        harness.players.primary().axis2d("move").y > 0.9,
        "held key resumes"
    );
}

#[test]
fn scheme_switches_follow_the_driving_device() {
    let mut harness = Harness::new(1);
    harness.input.set_gamepad_connected(0, true);
    let events = harness.frame(|input| input.process_gamepad_axis_input(0, Axis::LeftStickX, 0.9));
    assert!(events.iter().any(|event| matches!(
        event,
        Emitted::Device(PlayerDeviceEvent {
            player: 0,
            device: InputDevice::Gamepad(0),
            change: PlayerDeviceChange::Assigned
        })
    )));
    assert!(events.contains(&Emitted::Scheme(ControlSchemeChanged {
        player: 0,
        scheme: ControlScheme::Gamepad
    })));
    assert!(harness.players.primary().axis2d("move").x > 0.5);

    harness.frame(|input| input.process_gamepad_axis_input(0, Axis::LeftStickX, 0.0));
    let events =
        harness.frame(|input| input.process_key_input(KeyCode::KeyA, ElementState::Pressed, false));
    assert!(events.contains(&Emitted::Scheme(ControlSchemeChanged {
        player: 0,
        scheme: ControlScheme::KeyboardMouse
    })));
}

#[test]
fn join_on_press_gives_the_second_player_its_own_actions() {
    let mut harness = Harness::new(2);
    harness.input.set_gamepad_connected(3, true);
    let events = harness.frame(|input| input.process_gamepad_button_input(3, Button::South, true));
    assert!(events.contains(&Emitted::Device(PlayerDeviceEvent {
        player: 1,
        device: InputDevice::Gamepad(3),
        change: PlayerDeviceChange::Joined,
    })));
    // The joining press itself is evaluated for the new player next frame.
    harness.frame(|_| {});
    assert!(harness.players.player(1).unwrap().pressed("jump"));
    assert!(
        !harness.players.primary().pressed("jump"),
        "keyboard player is unaffected"
    );

    let events = harness.frame(|input| input.set_gamepad_connected(3, false));
    assert!(events.contains(&Emitted::Device(PlayerDeviceEvent {
        player: 1,
        device: InputDevice::Gamepad(3),
        change: PlayerDeviceChange::Lost,
    })));
}

#[test]
fn rebinding_captures_the_next_press_and_swallows_it() {
    let mut harness = Harness::new(1);
    harness.rebinding.start(RebindRequest::new(
        "gameplay",
        "jump",
        ControlScheme::KeyboardMouse,
    ));
    let events =
        harness.frame(|input| input.process_key_input(KeyCode::KeyJ, ElementState::Pressed, false));
    assert!(
        phases(&events, "jump").is_empty(),
        "capture frame is swallowed"
    );
    assert!(events.iter().any(|event| matches!(
        event,
        Emitted::Rebind(RebindEvent(RebindOutcome::Bound {
            source: InputSource::Key(KeyCode::KeyJ),
            ..
        }))
    )));
    assert!(!harness.rebinding.is_active());

    harness.frame(|input| input.process_key_input(KeyCode::KeyJ, ElementState::Released, false));
    let events =
        harness.frame(|input| input.process_key_input(KeyCode::KeyJ, ElementState::Pressed, false));
    assert!(phases(&events, "jump").contains(&ActionPhase::Performed));
    let events = harness
        .frame(|input| input.process_key_input(KeyCode::Space, ElementState::Pressed, false));
    assert!(phases(&events, "jump").is_empty(), "old key was replaced");
}

#[test]
fn rebinding_can_be_canceled() {
    let mut harness = Harness::new(1);
    harness.rebinding.start(RebindRequest::new(
        "gameplay",
        "jump",
        ControlScheme::KeyboardMouse,
    ));
    let events = harness
        .frame(|input| input.process_key_input(KeyCode::Escape, ElementState::Pressed, false));
    assert!(events.contains(&Emitted::Rebind(RebindEvent(RebindOutcome::Canceled))));
}

#[test]
fn plugin_runs_before_fixed_steps_and_latches_presses() {
    let mut runtime = GameRuntime::with_fixed_timestep(1.0 / 60.0);
    runtime.add_plugin(InputPlugin);
    runtime
        .world
        .resource_mut::<InputActionsSource>()
        .set_inline(sample());

    #[derive(Resource, Default)]
    struct SeenInFixed(u32);
    runtime.init_resource::<SeenInFixed>();
    runtime.add_systems(
        ScheduleKind::FixedUpdate,
        (|players: Res<LocalPlayers>, mut seen: ResMut<SeenInFixed>| {
            if players
                .primary()
                .action("jump")
                .is_some_and(|state| state.fixed_just_pressed)
            {
                seen.0 += 1;
            }
        })
        .in_set(FixedSet::Gameplay),
    );

    runtime.step(DT);
    {
        let mut input = runtime.world.resource_mut::<InputState>();
        input.begin_frame();
        input.process_key_input(KeyCode::Space, ElementState::Pressed, false);
    }
    runtime.step(DT);
    assert_eq!(
        runtime.world.resource::<SeenInFixed>().0,
        1,
        "seen in the same frame"
    );
    runtime.step(DT);
    assert_eq!(runtime.world.resource::<SeenInFixed>().0, 1, "latched once");

    let events = runtime.world.resource::<Events<ActionEvent>>();
    let mut reader = events.get_cursor();
    assert!(reader
        .read(events)
        .any(|event| event.action == "jump" && event.phase == ActionPhase::Performed));
}
