//! Local players: device ownership, context stacks and per-action state.

use std::collections::HashMap;

use bevy_ecs::prelude::Resource;
use engine_math::Vec2;

use crate::actions::{ControlScheme, InputActions};
use crate::state::ActionState;
use crate::InputDevice;

/// How new devices are assigned to players.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum JoinPolicy {
    /// Player 0 owns every device (single-player; any device controls it).
    #[default]
    SinglePlayerAllDevices,
    /// Player 0 starts with keyboard/mouse; pressing a button on an
    /// unowned gamepad joins the next free player.
    JoinOnPress,
}

#[derive(Clone, Debug, Default)]
pub struct PlayerInput {
    pub index: u8,
    pub devices: Vec<InputDevice>,
    /// Scheme of the device that last drove any action.
    pub scheme: ControlScheme,
    /// Active contexts, bottom first.
    pub contexts: Vec<String>,
    /// `(context, action)` → state.
    pub(crate) states: HashMap<(String, String), ActionState>,
}

impl PlayerInput {
    pub fn owns(&self, device: InputDevice) -> bool {
        self.devices.contains(&device)
    }

    /// The state of `action` in the top-most active context declaring it.
    pub fn action(&self, action: &str) -> Option<&ActionState> {
        self.contexts
            .iter()
            .rev()
            .find_map(|context| self.states.get(&(context.clone(), action.to_owned())))
    }

    pub fn action_mut(&mut self, action: &str) -> Option<&mut ActionState> {
        let context = self
            .contexts
            .iter()
            .rev()
            .find(|context| {
                self.states
                    .contains_key(&((*context).clone(), action.to_owned()))
            })?
            .clone();
        self.states.get_mut(&(context, action.to_owned()))
    }

    pub fn pressed(&self, action: &str) -> bool {
        self.action(action).is_some_and(ActionState::pressed)
    }

    pub fn just_pressed(&self, action: &str) -> bool {
        self.action(action).is_some_and(|state| state.just_pressed)
    }

    pub fn just_released(&self, action: &str) -> bool {
        self.action(action).is_some_and(|state| state.just_released)
    }

    pub fn performed(&self, action: &str) -> bool {
        self.action(action).is_some_and(|state| state.performed)
    }

    pub fn axis(&self, action: &str) -> f32 {
        self.action(action).map(ActionState::axis).unwrap_or(0.0)
    }

    pub fn axis2d(&self, action: &str) -> Vec2 {
        self.action(action)
            .map(ActionState::axis2d)
            .unwrap_or(Vec2::ZERO)
    }

    pub fn has_context(&self, context: &str) -> bool {
        self.contexts.iter().any(|c| c == context)
    }

    /// Pushes `context` on top (moving it there if already active).
    pub fn push_context(&mut self, context: impl Into<String>) {
        let context = context.into();
        self.contexts.retain(|c| *c != context);
        self.contexts.push(context);
    }

    /// Removes `context`, resetting its action states.
    pub fn pop_context(&mut self, context: &str) {
        self.contexts.retain(|c| c != context);
        self.states.retain(|(ctx, _), _| ctx != context);
    }

    pub fn set_contexts(&mut self, contexts: Vec<String>) {
        self.contexts = contexts;
        let active = self.contexts.clone();
        self.states.retain(|(ctx, _), _| active.contains(ctx));
    }
}

#[derive(Resource, Clone, Debug)]
pub struct LocalPlayers {
    players: Vec<PlayerInput>,
    pub max_players: u8,
    pub join_policy: JoinPolicy,
    /// Real seconds since start (interaction timing).
    pub(crate) now: f32,
    default_contexts: Vec<String>,
}

impl Default for LocalPlayers {
    fn default() -> Self {
        Self::new(1)
    }
}

impl LocalPlayers {
    pub fn new(max_players: u8) -> Self {
        let max_players = max_players.clamp(1, 8);
        let mut players = Self {
            players: Vec::new(),
            max_players,
            join_policy: if max_players > 1 {
                JoinPolicy::JoinOnPress
            } else {
                JoinPolicy::SinglePlayerAllDevices
            },
            now: 0.0,
            default_contexts: Vec::new(),
        };
        players.players.push(PlayerInput {
            index: 0,
            devices: vec![InputDevice::KeyboardMouse],
            ..Default::default()
        });
        players
    }

    pub fn players(&self) -> &[PlayerInput] {
        &self.players
    }

    pub fn players_mut(&mut self) -> &mut [PlayerInput] {
        &mut self.players
    }

    pub fn player(&self, index: u8) -> Option<&PlayerInput> {
        self.players.get(index as usize)
    }

    pub fn player_mut(&mut self, index: u8) -> Option<&mut PlayerInput> {
        self.players.get_mut(index as usize)
    }

    /// Player 0 (the common single-player case).
    pub fn primary(&self) -> &PlayerInput {
        &self.players[0]
    }

    pub fn primary_mut(&mut self) -> &mut PlayerInput {
        &mut self.players[0]
    }

    /// Applies an asset's default contexts to players that have none yet.
    pub(crate) fn adopt_defaults(&mut self, actions: &InputActions) {
        if self.default_contexts != actions.default_contexts {
            self.default_contexts = actions.default_contexts.clone();
            for player in &mut self.players {
                if player.contexts.is_empty() {
                    player.contexts = actions.default_contexts.clone();
                }
            }
        }
    }

    /// Owner of `device`, if any.
    pub fn owner_of(&self, device: InputDevice) -> Option<u8> {
        self.players
            .iter()
            .find(|player| player.owns(device))
            .map(|player| player.index)
    }

    /// Assigns connected/used devices according to the join policy.
    /// Returns `(player, device, joined_new_player)` for every assignment.
    pub(crate) fn assign_devices(
        &mut self,
        connected_gamepads: &[usize],
        pressed_devices: &[InputDevice],
    ) -> Vec<(u8, InputDevice, bool)> {
        let mut changes = Vec::new();
        match self.join_policy {
            JoinPolicy::SinglePlayerAllDevices => {
                let player = &mut self.players[0];
                for slot in connected_gamepads {
                    let device = InputDevice::Gamepad(*slot);
                    if !player.owns(device) {
                        player.devices.push(device);
                        changes.push((0, device, false));
                    }
                }
                player.devices.retain(|device| match device {
                    InputDevice::Gamepad(slot) => connected_gamepads.contains(slot),
                    InputDevice::KeyboardMouse => true,
                });
            }
            JoinPolicy::JoinOnPress => {
                for player in &mut self.players {
                    player.devices.retain(|device| match device {
                        InputDevice::Gamepad(slot) => connected_gamepads.contains(slot),
                        InputDevice::KeyboardMouse => true,
                    });
                }
                for device in pressed_devices {
                    if self.owner_of(*device).is_some() {
                        continue;
                    }
                    // Fill a player without devices first, else add one.
                    if let Some(player) = self
                        .players
                        .iter_mut()
                        .find(|player| player.devices.is_empty())
                    {
                        player.devices.push(*device);
                        changes.push((player.index, *device, false));
                    } else if (self.players.len() as u8) < self.max_players {
                        let index = self.players.len() as u8;
                        self.players.push(PlayerInput {
                            index,
                            devices: vec![*device],
                            scheme: device.scheme(),
                            contexts: self.default_contexts.clone(),
                            states: HashMap::new(),
                        });
                        changes.push((index, *device, true));
                    }
                }
            }
        }
        changes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_player_owns_every_connected_gamepad() {
        let mut players = LocalPlayers::new(1);
        let changes = players.assign_devices(&[0, 3], &[]);
        assert_eq!(changes.len(), 2);
        assert!(players.primary().owns(InputDevice::Gamepad(3)));
        players.assign_devices(&[3], &[]);
        assert!(!players.primary().owns(InputDevice::Gamepad(0)));
    }

    #[test]
    fn join_on_press_adds_players_up_to_the_limit() {
        let mut players = LocalPlayers::new(2);
        assert_eq!(players.join_policy, JoinPolicy::JoinOnPress);
        let changes = players.assign_devices(&[0, 1], &[InputDevice::Gamepad(0)]);
        assert_eq!(changes, vec![(1, InputDevice::Gamepad(0), true)]);
        let changes = players.assign_devices(&[0, 1], &[InputDevice::Gamepad(1)]);
        assert!(changes.is_empty(), "limit reached");
        // Unplugging frees the device; the empty player gets the next one.
        players.assign_devices(&[1], &[]);
        let changes = players.assign_devices(&[1], &[InputDevice::Gamepad(1)]);
        assert_eq!(changes, vec![(1, InputDevice::Gamepad(1), false)]);
    }

    #[test]
    fn context_stack_resolves_top_down() {
        let mut player = PlayerInput::default();
        player.set_contexts(vec!["gameplay".into()]);
        player
            .states
            .insert(("gameplay".into(), "jump".into()), ActionState::default());
        player.push_context("menu");
        let mut active = ActionState::default();
        active.active = true;
        player.states.insert(("menu".into(), "jump".into()), active);
        assert!(player.pressed("jump"));
        player.pop_context("menu");
        assert!(!player.pressed("jump"));
        assert!(!player.states.contains_key(&("menu".into(), "jump".into())));
    }
}
