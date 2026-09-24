//! Input: raw device state ([`InputState`]), the OS pump
//! ([`InputModule`]), and the action layer on top — action maps
//! ([`actions`]), local players with device assignment ([`players`]),
//! interactions ([`state`]), rebinding with persisted user overrides
//! ([`rebind`], [`settings`]) and the runtime plugin ([`plugin`]).

pub mod actions;
pub mod evaluate;
pub mod players;
pub mod plugin;
pub mod rebind;
pub mod settings;
pub mod state;

pub use actions::{
    ActionContext, ActionDef, ActionKind, Binding, ControlScheme, InputActions, InputActionsLoader,
    InputSource, Interaction, Modifier, Stick, INPUT_ACTIONS_VERSION,
};
pub use players::{JoinPolicy, LocalPlayers, PlayerInput};
pub use plugin::{
    ActionEvent, ActionPhase, ControlSchemeChanged, InputActionsSource, InputPlugin,
    PlayerDeviceChange, PlayerDeviceEvent,
};
pub use rebind::{RebindEvent, RebindOutcome, RebindRequest, Rebinding};
pub use settings::{InputUserSettings, InputUserSettingsStore, INPUT_USER_SETTINGS_VERSION};
pub use state::ActionState;

use bevy_ecs::prelude::Resource;
use engine_core::{HardeningConfig, Result};
use engine_math::Vec2;
use gilrs::{Axis, Button, EventType, Gilrs};
use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
};
use winit::event::{DeviceEvent, ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

/// A physical device family instance a player can own.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum InputDevice {
    KeyboardMouse,
    Gamepad(usize),
}

impl InputDevice {
    pub fn scheme(self) -> ControlScheme {
        match self {
            Self::KeyboardMouse => ControlScheme::KeyboardMouse,
            Self::Gamepad(_) => ControlScheme::Gamepad,
        }
    }
}

pub const DEFAULT_GAMEPAD_DEADZONE: f32 = 0.15;

#[derive(Debug, Clone)]
struct ButtonState<T>
where
    T: Eq + Hash,
{
    held: HashSet<T>,
    pressed: HashSet<T>,
    released: HashSet<T>,
}

impl<T> Default for ButtonState<T>
where
    T: Eq + Hash,
{
    fn default() -> Self {
        Self {
            held: HashSet::new(),
            pressed: HashSet::new(),
            released: HashSet::new(),
        }
    }
}

impl<T> ButtonState<T>
where
    T: Eq + Hash + Copy,
{
    fn begin_frame(&mut self) {
        self.pressed.clear();
        self.released.clear();
    }

    fn press(&mut self, value: T) {
        if self.held.insert(value) {
            self.pressed.insert(value);
        }
    }

    fn release(&mut self, value: T) {
        if self.held.remove(&value) {
            self.released.insert(value);
        }
    }

    fn clear_all(&mut self) {
        self.held.clear();
        self.pressed.clear();
        self.released.clear();
    }

    fn held(&self, value: T) -> bool {
        self.held.contains(&value)
    }

    fn just_pressed(&self, value: T) -> bool {
        self.pressed.contains(&value)
    }

    fn just_released(&self, value: T) -> bool {
        self.released.contains(&value)
    }
}

#[derive(Debug, Clone, Default)]
pub struct GamepadState {
    connected: bool,
    buttons: ButtonState<Button>,
    axes: HashMap<Axis, f32>,
    /// Analog value of pressure-sensitive buttons (triggers).
    button_values: HashMap<Button, f32>,
}

#[derive(Resource, Debug, Clone)]
pub struct InputState {
    keys: ButtonState<KeyCode>,
    mouse_buttons: ButtonState<MouseButton>,
    pub mouse_position: Vec2,
    pub mouse_delta: Vec2,
    /// Raw relative mouse motion this frame (works with a locked cursor).
    pub mouse_motion: Vec2,
    pub scroll_delta: f32,
    gamepads: HashMap<usize, GamepadState>,
    gamepad_deadzone: f32,
    last_cursor_position: Option<Vec2>,
    /// Devices that produced any press or significant motion this frame.
    active_devices: Vec<InputDevice>,
    /// Digital sources newly pressed this frame, with their device.
    new_presses: Vec<(InputDevice, InputSource)>,
    /// Gamepad connection changes this frame.
    gamepad_changes: Vec<(usize, bool)>,
    /// Characters typed this frame (text fields), control characters removed.
    typed_text: String,
}

impl Default for InputState {
    fn default() -> Self {
        Self {
            keys: ButtonState::default(),
            mouse_buttons: ButtonState::default(),
            mouse_position: Vec2::ZERO,
            mouse_delta: Vec2::ZERO,
            mouse_motion: Vec2::ZERO,
            scroll_delta: 0.0,
            gamepads: HashMap::new(),
            gamepad_deadzone: DEFAULT_GAMEPAD_DEADZONE,
            last_cursor_position: None,
            active_devices: Vec::new(),
            new_presses: Vec::new(),
            gamepad_changes: Vec::new(),
            typed_text: String::new(),
        }
    }
}

impl InputState {
    pub fn begin_frame(&mut self) {
        self.keys.begin_frame();
        self.mouse_buttons.begin_frame();
        self.mouse_delta = Vec2::ZERO;
        self.mouse_motion = Vec2::ZERO;
        self.scroll_delta = 0.0;
        self.active_devices.clear();
        self.new_presses.clear();
        self.gamepad_changes.clear();
        self.typed_text.clear();

        for gamepad in self.gamepads.values_mut() {
            gamepad.buttons.begin_frame();
        }
    }

    pub fn process_window_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::KeyboardInput { event, .. } => self.process_key_event(event),
            WindowEvent::MouseInput { state, button, .. } => {
                self.process_mouse_button_input(*button, *state);
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.process_cursor_position(position.x as f32, position.y as f32);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(position) => position.y as f32 / 120.0,
                };
                self.process_scroll_delta(scroll);
            }
            _ => {}
        }
    }

    pub fn process_key_event(&mut self, event: &winit::event::KeyEvent) {
        if let PhysicalKey::Code(code) = event.physical_key {
            self.process_key_input(code, event.state, event.repeat);
        }
        if event.state == ElementState::Pressed {
            if let Some(text) = &event.text {
                self.process_text(text);
            }
        }
    }

    /// Text produced by the keyboard (layout- and IME-aware).
    pub fn process_text(&mut self, text: &str) {
        self.typed_text
            .extend(text.chars().filter(|c| !c.is_control()));
    }

    /// Characters typed this frame.
    pub fn typed_text(&self) -> &str {
        &self.typed_text
    }

    pub fn process_key_input(&mut self, code: KeyCode, state: ElementState, repeat: bool) {
        match state {
            ElementState::Pressed if !repeat => {
                if !self.keys.held(code) {
                    self.note_press(InputDevice::KeyboardMouse, InputSource::Key(code));
                }
                self.keys.press(code);
            }
            ElementState::Released => {
                self.keys.release(code);
            }
            _ => {}
        }
    }

    pub fn process_mouse_button_input(&mut self, button: MouseButton, state: ElementState) {
        match state {
            ElementState::Pressed => {
                if !self.mouse_buttons.held(button) {
                    self.note_press(InputDevice::KeyboardMouse, InputSource::MouseButton(button));
                }
                self.mouse_buttons.press(button)
            }
            ElementState::Released => self.mouse_buttons.release(button),
        }
    }

    pub fn process_cursor_position(&mut self, x: f32, y: f32) {
        let next = Vec2::new(x, y);

        if let Some(last) = self.last_cursor_position {
            self.mouse_delta += next - last;
        }

        self.last_cursor_position = Some(next);
        self.mouse_position = next;
    }

    pub fn process_scroll_delta(&mut self, delta: f32) {
        self.scroll_delta += delta;
        if delta != 0.0 {
            self.note_activity(InputDevice::KeyboardMouse);
        }
    }

    /// Raw relative motion (e.g. `DeviceEvent::MouseMotion`).
    pub fn process_mouse_motion(&mut self, dx: f32, dy: f32) {
        self.mouse_motion += Vec2::new(dx, dy);
        if dx.abs() + dy.abs() > 2.0 {
            self.note_activity(InputDevice::KeyboardMouse);
        }
    }

    fn note_activity(&mut self, device: InputDevice) {
        if !self.active_devices.contains(&device) {
            self.active_devices.push(device);
        }
    }

    fn note_press(&mut self, device: InputDevice, source: InputSource) {
        self.note_activity(device);
        self.new_presses.push((device, source));
    }

    /// Devices that were used this frame, in first-use order.
    pub fn active_devices(&self) -> &[InputDevice] {
        &self.active_devices
    }

    /// Digital inputs pressed this frame (rebinding, "press any key").
    pub fn new_presses(&self) -> &[(InputDevice, InputSource)] {
        &self.new_presses
    }

    /// Gamepad connections (`true`) and disconnections this frame.
    pub fn gamepad_changes(&self) -> &[(usize, bool)] {
        &self.gamepad_changes
    }

    /// Connected gamepad slots, ascending.
    pub fn connected_gamepads(&self) -> Vec<usize> {
        let mut slots: Vec<usize> = self
            .gamepads
            .iter()
            .filter(|(_, state)| state.connected)
            .map(|(slot, _)| *slot)
            .collect();
        slots.sort_unstable();
        slots
    }

    /// Analog value of a gamepad button (1.0/0.0 for digital buttons).
    pub fn gamepad_button_value(&self, gamepad_slot: usize, button: Button) -> f32 {
        let Some(state) = self.gamepads.get(&gamepad_slot) else {
            return 0.0;
        };
        state
            .button_values
            .get(&button)
            .copied()
            .unwrap_or(if state.buttons.held(button) { 1.0 } else { 0.0 })
    }

    /// Releases everything held (focus loss, entering a menu).
    pub fn release_all(&mut self) {
        let keys: Vec<KeyCode> = self.keys.held.iter().copied().collect();
        for key in keys {
            self.keys.release(key);
        }
        let buttons: Vec<MouseButton> = self.mouse_buttons.held.iter().copied().collect();
        for button in buttons {
            self.mouse_buttons.release(button);
        }
        for gamepad in self.gamepads.values_mut() {
            let held: Vec<Button> = gamepad.buttons.held.iter().copied().collect();
            for button in held {
                gamepad.buttons.release(button);
            }
            gamepad.axes.clear();
            gamepad.button_values.clear();
        }
    }

    pub fn set_gamepad_connected(&mut self, gamepad_slot: usize, connected: bool) {
        let gamepad = self.gamepads.entry(gamepad_slot).or_default();
        if gamepad.connected != connected {
            self.gamepad_changes.push((gamepad_slot, connected));
        }
        gamepad.connected = connected;
        if !connected {
            gamepad.buttons.clear_all();
            gamepad.axes.clear();
        }
    }

    pub fn process_gamepad_button_input(
        &mut self,
        gamepad_slot: usize,
        button: Button,
        pressed: bool,
    ) {
        let gamepad = self.gamepads.entry(gamepad_slot).or_default();
        gamepad.connected = true;
        let was_held = gamepad.buttons.held(button);
        if pressed {
            gamepad.buttons.press(button);
        } else {
            gamepad.buttons.release(button);
        }
        if pressed && !was_held {
            self.note_press(
                InputDevice::Gamepad(gamepad_slot),
                InputSource::GamepadButton(button),
            );
        }
    }

    /// Analog button value (triggers); crossing 0.5 counts as a press.
    pub fn process_gamepad_button_value(
        &mut self,
        gamepad_slot: usize,
        button: Button,
        value: f32,
    ) {
        let value = value.clamp(0.0, 1.0);
        self.gamepads
            .entry(gamepad_slot)
            .or_default()
            .button_values
            .insert(button, value);
        self.process_gamepad_button_input(gamepad_slot, button, value >= 0.5);
    }

    pub fn process_gamepad_axis_input(&mut self, gamepad_slot: usize, axis: Axis, value: f32) {
        let normalized = apply_deadzone(value, self.gamepad_deadzone);
        let gamepad = self.gamepads.entry(gamepad_slot).or_default();
        gamepad.connected = true;
        let previous = gamepad.axes.insert(axis, normalized).unwrap_or(0.0);
        if normalized.abs() > 0.5 && previous.abs() <= 0.5 {
            self.note_press(
                InputDevice::Gamepad(gamepad_slot),
                InputSource::GamepadAxis(axis),
            );
        } else if normalized != 0.0 {
            self.note_activity(InputDevice::Gamepad(gamepad_slot));
        }
    }

    pub fn set_gamepad_deadzone(&mut self, deadzone: f32) {
        self.gamepad_deadzone = deadzone.clamp(0.0, 1.0);
    }

    pub fn key_held(&self, key: KeyCode) -> bool {
        self.keys.held(key)
    }

    pub fn key_just_pressed(&self, key: KeyCode) -> bool {
        self.keys.just_pressed(key)
    }

    pub fn key_just_released(&self, key: KeyCode) -> bool {
        self.keys.just_released(key)
    }

    pub fn mouse_held(&self, button: MouseButton) -> bool {
        self.mouse_buttons.held(button)
    }

    pub fn mouse_just_pressed(&self, button: MouseButton) -> bool {
        self.mouse_buttons.just_pressed(button)
    }

    pub fn mouse_just_released(&self, button: MouseButton) -> bool {
        self.mouse_buttons.just_released(button)
    }

    pub fn first_connected_gamepad(&self) -> Option<usize> {
        self.gamepads
            .iter()
            .find_map(|(slot, state)| state.connected.then_some(*slot))
    }

    pub fn gamepad_button_held(&self, gamepad_slot: usize, button: Button) -> bool {
        self.gamepads
            .get(&gamepad_slot)
            .is_some_and(|state| state.buttons.held(button))
    }

    pub fn gamepad_button_just_pressed(&self, gamepad_slot: usize, button: Button) -> bool {
        self.gamepads
            .get(&gamepad_slot)
            .is_some_and(|state| state.buttons.just_pressed(button))
    }

    pub fn gamepad_button_just_released(&self, gamepad_slot: usize, button: Button) -> bool {
        self.gamepads
            .get(&gamepad_slot)
            .is_some_and(|state| state.buttons.just_released(button))
    }

    pub fn gamepad_axis(&self, gamepad_slot: usize, axis: Axis) -> f32 {
        self.gamepads
            .get(&gamepad_slot)
            .and_then(|state| state.axes.get(&axis).copied())
            .unwrap_or(0.0)
    }
}

fn apply_deadzone(value: f32, deadzone: f32) -> f32 {
    if value.abs() < deadzone {
        0.0
    } else {
        value
    }
}

#[derive(Debug, Clone)]
enum BufferedWindowEvent {
    Key {
        code: KeyCode,
        state: ElementState,
        repeat: bool,
    },
    MouseButton {
        button: MouseButton,
        state: ElementState,
    },
    CursorMoved {
        x: f32,
        y: f32,
    },
    MouseWheel {
        delta: f32,
    },
    MouseMotion {
        dx: f32,
        dy: f32,
    },
    Text(String),
}

pub struct InputModule {
    buffered_events: Vec<BufferedWindowEvent>,
    gilrs: Option<Gilrs>,
    max_buffered_events: usize,
    max_gamepads: usize,
    buffer_overflow_warned: bool,
    gamepad_overflow_warned: bool,
}

impl Default for InputModule {
    fn default() -> Self {
        Self::new()
    }
}

impl InputModule {
    pub fn new() -> Self {
        let hardening = HardeningConfig::default();

        let gilrs = match Gilrs::new() {
            Ok(gilrs) => Some(gilrs),
            Err(error) => {
                log::warn!(
                    target: "engine::input",
                    "Gamepad backend unavailable at startup: {}",
                    error
                );
                None
            }
        };

        Self {
            buffered_events: Vec::new(),
            gilrs,
            max_buffered_events: hardening.max_buffered_input_events.max(1),
            max_gamepads: hardening.max_registered_gamepads.max(1),
            buffer_overflow_warned: false,
            gamepad_overflow_warned: false,
        }
    }

    pub fn configure_hardening(&mut self, hardening: HardeningConfig) {
        self.max_buffered_events = hardening.max_buffered_input_events.max(1);
        self.max_gamepads = hardening.max_registered_gamepads.max(1);
    }

    pub fn handle_window_event(&mut self, event: &WindowEvent) -> Result<()> {
        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                let PhysicalKey::Code(code) = event.physical_key else {
                    return Ok(());
                };

                self.push_buffered_event(BufferedWindowEvent::Key {
                    code,
                    state: event.state,
                    repeat: event.repeat,
                });
                if event.state == ElementState::Pressed {
                    if let Some(text) = &event.text {
                        self.push_buffered_event(BufferedWindowEvent::Text(text.to_string()));
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                self.push_buffered_event(BufferedWindowEvent::MouseButton {
                    button: *button,
                    state: *state,
                });
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.push_buffered_event(BufferedWindowEvent::CursorMoved {
                    x: position.x as f32,
                    y: position.y as f32,
                });
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(position) => position.y as f32 / 120.0,
                };

                self.push_buffered_event(BufferedWindowEvent::MouseWheel { delta: scroll });
            }
            _ => {}
        }

        Ok(())
    }

    /// Buffers raw device events (relative mouse motion).
    pub fn handle_device_event(&mut self, event: &DeviceEvent) -> Result<()> {
        if let DeviceEvent::MouseMotion { delta } = event {
            self.push_buffered_event(BufferedWindowEvent::MouseMotion {
                dx: delta.0 as f32,
                dy: delta.1 as f32,
            });
        }
        Ok(())
    }

    pub fn pump(&mut self, input_state: &mut InputState) -> Result<()> {
        input_state.begin_frame();
        self.buffer_overflow_warned = false;
        self.gamepad_overflow_warned = false;

        for event in self.buffered_events.drain(..) {
            match event {
                BufferedWindowEvent::Key {
                    code,
                    state,
                    repeat,
                } => input_state.process_key_input(code, state, repeat),
                BufferedWindowEvent::MouseButton { button, state } => {
                    input_state.process_mouse_button_input(button, state)
                }
                BufferedWindowEvent::CursorMoved { x, y } => {
                    input_state.process_cursor_position(x, y)
                }
                BufferedWindowEvent::MouseWheel { delta } => {
                    input_state.process_scroll_delta(delta)
                }
                BufferedWindowEvent::MouseMotion { dx, dy } => {
                    input_state.process_mouse_motion(dx, dy)
                }
                BufferedWindowEvent::Text(text) => input_state.process_text(&text),
            }
        }

        if let Some(gilrs) = self.gilrs.as_mut() {
            while let Some(event) = gilrs.next_event() {
                let gamepad_slot: usize = event.id.into();
                let is_known_gamepad = input_state.gamepads.contains_key(&gamepad_slot);

                if !is_known_gamepad && input_state.gamepads.len() >= self.max_gamepads {
                    if !self.gamepad_overflow_warned {
                        log::warn!(
                            target: "engine::input",
                            "Gamepad capacity reached ({}); skipping additional gamepad events",
                            self.max_gamepads
                        );
                        self.gamepad_overflow_warned = true;
                    }
                    continue;
                }

                match event.event {
                    EventType::Connected => input_state.set_gamepad_connected(gamepad_slot, true),
                    EventType::Disconnected => {
                        input_state.set_gamepad_connected(gamepad_slot, false)
                    }
                    EventType::ButtonPressed(button, _) | EventType::ButtonRepeated(button, _) => {
                        input_state.process_gamepad_button_input(gamepad_slot, button, true)
                    }
                    EventType::ButtonReleased(button, _) => {
                        input_state.process_gamepad_button_input(gamepad_slot, button, false)
                    }
                    EventType::ButtonChanged(button, value, _) => {
                        input_state.process_gamepad_button_value(gamepad_slot, button, value);
                    }
                    EventType::AxisChanged(axis, value, _) => {
                        input_state.process_gamepad_axis_input(gamepad_slot, axis, value)
                    }
                    EventType::Dropped | EventType::ForceFeedbackEffectCompleted => {}
                    _ => {}
                }
            }
        }

        log::trace!(target: "engine::input", "Input events pumped");
        Ok(())
    }

    fn push_buffered_event(&mut self, event: BufferedWindowEvent) {
        if self.buffered_events.len() >= self.max_buffered_events {
            if !self.buffer_overflow_warned {
                log::warn!(
                    target: "engine::input",
                    "Input event buffer capacity reached ({}); skipping additional events until next pump",
                    self.max_buffered_events
                );
                self.buffer_overflow_warned = true;
            }
            return;
        }

        self.buffered_events.push(event);
    }

    pub fn backend_type_names(&self) -> (&'static str, &'static str) {
        (
            std::any::type_name::<winit::event::ElementState>(),
            std::any::type_name::<gilrs::GamepadId>(),
        )
    }
}

pub fn module_name() -> &'static str {
    "engine-input"
}

#[cfg(test)]
mod tests;
