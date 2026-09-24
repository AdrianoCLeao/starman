use bevy_ecs::prelude::Resource;
use engine_core::{HardeningConfig, Result};
use engine_math::Vec2;
use gilrs::{Axis, Button, EventType, Gilrs};
use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
};
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

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
}

#[derive(Resource, Debug, Clone)]
pub struct InputState {
    keys: ButtonState<KeyCode>,
    mouse_buttons: ButtonState<MouseButton>,
    pub mouse_position: Vec2,
    pub mouse_delta: Vec2,
    pub scroll_delta: f32,
    gamepads: HashMap<usize, GamepadState>,
    gamepad_deadzone: f32,
    last_cursor_position: Option<Vec2>,
}

impl Default for InputState {
    fn default() -> Self {
        Self {
            keys: ButtonState::default(),
            mouse_buttons: ButtonState::default(),
            mouse_position: Vec2::ZERO,
            mouse_delta: Vec2::ZERO,
            scroll_delta: 0.0,
            gamepads: HashMap::new(),
            gamepad_deadzone: DEFAULT_GAMEPAD_DEADZONE,
            last_cursor_position: None,
        }
    }
}

impl InputState {
    pub fn begin_frame(&mut self) {
        self.keys.begin_frame();
        self.mouse_buttons.begin_frame();
        self.mouse_delta = Vec2::ZERO;
        self.scroll_delta = 0.0;

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
    }

    pub fn process_key_input(&mut self, code: KeyCode, state: ElementState, repeat: bool) {
        match state {
            ElementState::Pressed if !repeat => {
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
            ElementState::Pressed => self.mouse_buttons.press(button),
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
    }

    pub fn set_gamepad_connected(&mut self, gamepad_slot: usize, connected: bool) {
        let gamepad = self.gamepads.entry(gamepad_slot).or_default();
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
        if pressed {
            gamepad.buttons.press(button);
        } else {
            gamepad.buttons.release(button);
        }
    }

    pub fn process_gamepad_axis_input(&mut self, gamepad_slot: usize, axis: Axis, value: f32) {
        let normalized = apply_deadzone(value, self.gamepad_deadzone);
        let gamepad = self.gamepads.entry(gamepad_slot).or_default();
        gamepad.axes.insert(axis, normalized);
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
                        input_state.process_gamepad_button_input(
                            gamepad_slot,
                            button,
                            value >= input_state.gamepad_deadzone,
                        );
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

/// Installs the raw [`InputState`] resource into a runtime. The OS event
/// pump ([`InputModule`]) stays with the windowed host.
#[derive(Default)]
pub struct InputPlugin;

impl engine_core::RuntimePlugin for InputPlugin {
    fn name(&self) -> &'static str {
        "engine::input"
    }

    fn build(&self, runtime: &mut engine_core::GameRuntime) {
        runtime.init_resource::<InputState>();
    }
}
