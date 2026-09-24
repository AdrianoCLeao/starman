//! The input runtime plugin: evaluates action maps for every local player
//! once per frame (before the fixed simulation), assigns devices, detects
//! control-scheme switches, drives rebinding sessions and latches
//! presses for the fixed step.

use std::collections::HashSet;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_assets::{AssetRef, Assets, Handle};
use engine_core::{FirstSet, FixedSet, FrameTime, GameRuntime, RuntimePlugin, ScheduleKind};
use engine_math::Vec2;

use crate::actions::{ControlScheme, InputActions, InputActionsLoader};
use crate::evaluate::evaluate_bindings;
use crate::players::LocalPlayers;
use crate::rebind::{RebindEvent, RebindOutcome, Rebinding};
use crate::settings::InputUserSettingsStore;
use crate::{InputDevice, InputState};

pub use crate::state::ActionPhase;

/// An action changed phase for a player.
#[derive(Event, Clone, Debug, PartialEq)]
pub struct ActionEvent {
    pub player: u8,
    pub context: String,
    pub action: String,
    pub phase: ActionPhase,
    pub value: Vec2,
}

/// A player's active control scheme changed (UI swaps button prompts).
#[derive(Event, Clone, Copy, Debug, PartialEq, Eq)]
pub struct ControlSchemeChanged {
    pub player: u8,
    pub scheme: ControlScheme,
}

/// What happened to a device/player pairing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlayerDeviceChange {
    /// An existing player received the device.
    Assigned,
    /// A new local player joined with the device.
    Joined,
    /// The device was disconnected and left the player.
    Lost,
}

#[derive(Event, Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlayerDeviceEvent {
    pub player: u8,
    pub device: InputDevice,
    pub change: PlayerDeviceChange,
}

/// Where the active [`InputActions`] come from: a project asset (resolved
/// and hot-reloaded through [`Assets`]) or an inline value (tests, tools).
#[derive(Resource, Default)]
pub struct InputActionsSource {
    asset: Option<AssetRef>,
    handle: Option<Handle<InputActions>>,
    revision: u64,
    current: Option<Arc<InputActions>>,
}

impl InputActionsSource {
    /// Uses the project asset `asset` (loaded on demand).
    pub fn set_asset(&mut self, asset: AssetRef) {
        if self.asset.as_ref() != Some(&asset) {
            self.asset = Some(asset);
            self.handle = None;
            self.revision = 0;
            self.current = None;
        }
    }

    /// Uses `actions` directly.
    pub fn set_inline(&mut self, actions: InputActions) {
        self.asset = None;
        self.handle = None;
        self.revision = 0;
        self.current = Some(Arc::new(actions));
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }

    pub fn asset(&self) -> Option<&AssetRef> {
        self.asset.as_ref()
    }

    /// The actions in effect, once loaded.
    pub fn actions(&self) -> Option<&Arc<InputActions>> {
        self.current.as_ref()
    }

    /// Picks up a (re)loaded asset. Returns `true` when the actions changed.
    fn refresh(&mut self, assets: Option<&Assets>) -> bool {
        let (Some(asset), Some(assets)) = (&self.asset, assets) else {
            return false;
        };
        let handle = *self
            .handle
            .get_or_insert_with(|| assets.request::<InputActions>(asset));
        let revision = assets.revision(handle);
        if revision == self.revision && self.current.is_some() {
            return false;
        }
        match assets.get(handle) {
            Some(actions) => {
                self.revision = revision;
                self.current = Some(actions);
                true
            }
            None => false,
        }
    }
}

/// Installs input resources, events and the action systems.
#[derive(Default)]
pub struct InputPlugin;

impl RuntimePlugin for InputPlugin {
    fn name(&self) -> &'static str {
        "engine::input"
    }

    fn build(&self, runtime: &mut GameRuntime) {
        if let Some(assets) = runtime.world.get_resource::<Assets>() {
            assets.register_loader(InputActionsLoader);
        }
        runtime
            .init_resource::<InputState>()
            .init_resource::<LocalPlayers>()
            .init_resource::<InputActionsSource>()
            .init_resource::<InputUserSettingsStore>()
            .init_resource::<Rebinding>()
            .add_event::<ActionEvent>()
            .add_event::<ControlSchemeChanged>()
            .add_event::<PlayerDeviceEvent>()
            .add_event::<RebindEvent>()
            .add_systems(ScheduleKind::First, update_actions.in_set(FirstSet::Input))
            .add_systems(
                ScheduleKind::FixedUpdate,
                latch_fixed_actions.in_set(FixedSet::Input),
            );
    }
}

#[derive(bevy_ecs::system::SystemParam)]
struct ActionEventWriters<'w> {
    actions: EventWriter<'w, ActionEvent>,
    schemes: EventWriter<'w, ControlSchemeChanged>,
    devices: EventWriter<'w, PlayerDeviceEvent>,
    rebinds: EventWriter<'w, RebindEvent>,
}

#[allow(clippy::too_many_arguments)]
fn update_actions(
    input: Res<InputState>,
    time: Option<Res<FrameTime>>,
    assets: Option<Res<Assets>>,
    mut source: ResMut<InputActionsSource>,
    mut players: ResMut<LocalPlayers>,
    mut user: ResMut<InputUserSettingsStore>,
    mut rebinding: ResMut<Rebinding>,
    mut events: ActionEventWriters,
) {
    let dt = time.map_or(0.0, |time| time.real_delta_seconds);
    source.refresh(assets.as_deref());
    let actions = source.actions().cloned();
    let events = &mut events;
    step_actions(
        &input,
        dt,
        actions.as_deref(),
        &mut players,
        &mut user,
        &mut rebinding,
        &mut |event| match event {
            Emitted::Action(event) => {
                events.actions.send(event);
            }
            Emitted::Scheme(event) => {
                events.schemes.send(event);
            }
            Emitted::Device(event) => {
                events.devices.send(event);
            }
            Emitted::Rebind(event) => {
                events.rebinds.send(event);
            }
        },
    );
}

fn latch_fixed_actions(mut players: ResMut<LocalPlayers>) {
    for player in players.players_mut() {
        for state in player.states.values_mut() {
            state.latch_fixed();
        }
    }
}

/// Events produced by [`step_actions`].
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Emitted {
    Action(ActionEvent),
    Scheme(ControlSchemeChanged),
    Device(PlayerDeviceEvent),
    Rebind(RebindEvent),
}

/// One frame of the action layer, independent of the ECS so it can be
/// unit-tested and reused by tools (editor input tester).
pub(crate) fn step_actions(
    input: &InputState,
    dt: f32,
    actions: Option<&InputActions>,
    players: &mut LocalPlayers,
    user: &mut InputUserSettingsStore,
    rebinding: &mut Rebinding,
    emit: &mut dyn FnMut(Emitted),
) {
    players.now += dt;
    let now = players.now;

    // Device ownership.
    let before: Vec<(u8, Vec<InputDevice>)> = players
        .players()
        .iter()
        .map(|player| (player.index, player.devices.clone()))
        .collect();
    let mut pressed_devices: Vec<InputDevice> = Vec::new();
    for (device, _) in input.new_presses() {
        if !pressed_devices.contains(device) {
            pressed_devices.push(*device);
        }
    }
    let assigned = players.assign_devices(&input.connected_gamepads(), &pressed_devices);
    for (player, devices) in &before {
        let Some(current) = players.player(*player) else {
            continue;
        };
        for device in devices {
            if !current.owns(*device) {
                emit(Emitted::Device(PlayerDeviceEvent {
                    player: *player,
                    device: *device,
                    change: PlayerDeviceChange::Lost,
                }));
            }
        }
    }
    for (player, device, joined) in assigned {
        emit(Emitted::Device(PlayerDeviceEvent {
            player,
            device,
            change: if joined {
                PlayerDeviceChange::Joined
            } else {
                PlayerDeviceChange::Assigned
            },
        }));
    }

    if let Some(actions) = actions {
        players.adopt_defaults(actions);
    }

    // A rebinding session swallows all input of the frames it is active in
    // (including the capturing press) so gameplay never sees it.
    let suppress = rebinding.is_active();
    if suppress {
        let owners: Vec<(u8, Vec<InputDevice>)> = players
            .players()
            .iter()
            .map(|player| (player.index, player.devices.clone()))
            .collect();
        let owned = |player: u8, device: InputDevice| {
            owners
                .iter()
                .any(|(index, devices)| *index == player && devices.contains(&device))
        };
        if let Some(outcome) =
            rebinding.update(dt, input.new_presses(), &owned, actions, &mut user.settings)
        {
            if matches!(outcome, RebindOutcome::Bound { .. }) {
                user.dirty = true;
                if let Err(error) = user.save() {
                    log::warn!(target: "engine::input", "could not save input settings: {error}");
                }
            }
            emit(Emitted::Rebind(RebindEvent(outcome)));
        }
    }

    let Some(actions) = actions else {
        return;
    };
    for player in players.players_mut() {
        // Contexts from the top of the stack down to the first blocking one.
        let mut active: Vec<&str> = Vec::new();
        for name in player.contexts.iter().rev() {
            let Some(context) = actions.context(name) else {
                continue;
            };
            active.push(&context.name);
            if context.blocks_lower {
                break;
            }
        }
        let active_set: HashSet<&str> = active.iter().copied().collect();

        // Blocked or removed contexts release their actions.
        for ((context, action), state) in player.states.iter_mut() {
            if !active_set.contains(context.as_str()) && state.reset() {
                emit(Emitted::Action(ActionEvent {
                    player: player.index,
                    context: context.clone(),
                    action: action.clone(),
                    phase: ActionPhase::Canceled,
                    value: Vec2::ZERO,
                }));
            }
        }

        let mut driving_scheme: Option<(ControlScheme, f32)> = None;
        for context_name in active {
            let Some(context) = actions.context(context_name) else {
                continue;
            };
            for def in &context.actions {
                let bindings = user.settings.bindings_for(&context.name, def);
                let evaluated = if suppress {
                    Default::default()
                } else {
                    evaluate_bindings(input, bindings, def.kind, &player.devices, &user.settings)
                };
                let state = player
                    .states
                    .entry((context.name.clone(), def.name.clone()))
                    .or_default();
                if let Some(device) = evaluated.device {
                    state.scheme = Some(device.scheme());
                    let magnitude = evaluated.value.length();
                    if driving_scheme.is_none_or(|(_, best)| magnitude > best) {
                        driving_scheme = Some((device.scheme(), magnitude));
                    }
                }
                for phase in state.update(def, evaluated.value, dt, now) {
                    emit(Emitted::Action(ActionEvent {
                        player: player.index,
                        context: context.name.clone(),
                        action: def.name.clone(),
                        phase,
                        value: state.value,
                    }));
                }
            }
        }

        // Scheme switch: an actuated action, else any press on an owned
        // device (menus with no bound action still swap prompts).
        let scheme = driving_scheme.map(|(scheme, _)| scheme).or_else(|| {
            input
                .new_presses()
                .iter()
                .find(|(device, _)| player.owns(*device))
                .map(|(device, _)| device.scheme())
        });
        if let Some(scheme) = scheme {
            if scheme != player.scheme {
                player.scheme = scheme;
                emit(Emitted::Scheme(ControlSchemeChanged {
                    player: player.index,
                    scheme,
                }));
            }
        }
    }
}

#[cfg(test)]
#[path = "plugin_tests.rs"]
mod tests;
