//! Reading bindings from raw device state and applying modifiers.

use engine_math::Vec2;
use gilrs::{Axis, Button};

use crate::actions::{ActionKind, Binding, InputSource, Modifier, Stick};
use crate::settings::InputUserSettings;
use crate::{InputDevice, InputState};

/// Raw value of `source` on `device` (1D sources use `x`).
pub fn source_value(state: &InputState, source: &InputSource, device: InputDevice) -> Vec2 {
    match (source, device) {
        (InputSource::Key(code), InputDevice::KeyboardMouse) => {
            Vec2::new(if state.key_held(*code) { 1.0 } else { 0.0 }, 0.0)
        }
        (InputSource::MouseButton(button), InputDevice::KeyboardMouse) => {
            Vec2::new(if state.mouse_held(*button) { 1.0 } else { 0.0 }, 0.0)
        }
        (InputSource::MouseMotion, InputDevice::KeyboardMouse) => {
            // Prefer raw motion (locked cursor); fall back to cursor delta.
            if state.mouse_motion != Vec2::ZERO {
                state.mouse_motion
            } else {
                state.mouse_delta
            }
        }
        (InputSource::MouseWheel, InputDevice::KeyboardMouse) => Vec2::new(state.scroll_delta, 0.0),
        (InputSource::GamepadButton(button), InputDevice::Gamepad(slot)) => {
            Vec2::new(state.gamepad_button_value(slot, *button), 0.0)
        }
        (InputSource::GamepadAxis(axis), InputDevice::Gamepad(slot)) => {
            Vec2::new(state.gamepad_axis(slot, *axis), 0.0)
        }
        (InputSource::GamepadStick(stick), InputDevice::Gamepad(slot)) => match stick {
            Stick::Left => Vec2::new(
                state.gamepad_axis(slot, Axis::LeftStickX),
                state.gamepad_axis(slot, Axis::LeftStickY),
            ),
            Stick::Right => Vec2::new(
                state.gamepad_axis(slot, Axis::RightStickX),
                state.gamepad_axis(slot, Axis::RightStickY),
            ),
            Stick::DPad => {
                let axis = Vec2::new(
                    state.gamepad_axis(slot, Axis::DPadX),
                    state.gamepad_axis(slot, Axis::DPadY),
                );
                let buttons = Vec2::new(
                    state.gamepad_button_value(slot, Button::DPadRight)
                        - state.gamepad_button_value(slot, Button::DPadLeft),
                    state.gamepad_button_value(slot, Button::DPadUp)
                        - state.gamepad_button_value(slot, Button::DPadDown),
                );
                if buttons.length_squared() > axis.length_squared() {
                    buttons
                } else {
                    axis
                }
            }
        },
        (InputSource::Composite1D { negative, positive }, device) => {
            let value = scalar(state, positive, device) - scalar(state, negative, device);
            Vec2::new(value, 0.0)
        }
        (
            InputSource::Composite2D {
                up,
                down,
                left,
                right,
            },
            device,
        ) => Vec2::new(
            scalar(state, right, device) - scalar(state, left, device),
            scalar(state, up, device) - scalar(state, down, device),
        ),
        _ => Vec2::ZERO,
    }
}

fn scalar(state: &InputState, source: &InputSource, device: InputDevice) -> f32 {
    source_value(state, source, device).x.abs().min(1.0)
}

/// Applies `modifiers` in order.
pub fn apply_modifiers(mut value: Vec2, modifiers: &[Modifier], dimension: u8) -> Vec2 {
    for modifier in modifiers {
        value = match modifier {
            Modifier::Deadzone(deadzone) => {
                let magnitude = if dimension == 2 {
                    value.length()
                } else {
                    value.x.abs()
                };
                if magnitude <= *deadzone || magnitude <= f32::EPSILON {
                    Vec2::ZERO
                } else {
                    let rescaled = ((magnitude - deadzone) / (1.0 - deadzone)).min(1.0);
                    value * (rescaled / magnitude)
                }
            }
            Modifier::Invert => -value,
            Modifier::InvertX => Vec2::new(-value.x, value.y),
            Modifier::InvertY => Vec2::new(value.x, -value.y),
            Modifier::Scale(factor) => value * *factor,
            Modifier::ScaleXY(x, y) => Vec2::new(value.x * x, value.y * y),
            Modifier::Normalize => {
                let length = value.length();
                if length > 1.0 {
                    value / length
                } else {
                    value
                }
            }
            Modifier::Swizzle => Vec2::new(value.y, value.x),
        };
    }
    value
}

/// Applies per-user preferences (sensitivity, invert Y) to look-style
/// sources.
pub fn apply_user_preferences(value: Vec2, source: &InputSource, user: &InputUserSettings) -> Vec2 {
    match source {
        InputSource::MouseMotion => {
            let mut v = value * user.mouse_sensitivity;
            if user.invert_y {
                v.y = -v.y;
            }
            v
        }
        InputSource::GamepadStick(Stick::Right) => {
            let mut v = value * user.gamepad_look_sensitivity;
            if user.invert_y {
                v.y = -v.y;
            }
            v
        }
        _ => value,
    }
}

/// Result of evaluating one action for one player.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Evaluated {
    pub value: Vec2,
    /// Device of the strongest binding, when any was actuated.
    pub device: Option<InputDevice>,
}

/// Evaluates `bindings` over `devices`, keeping the strongest input.
pub fn evaluate_bindings(
    state: &InputState,
    bindings: &[Binding],
    kind: ActionKind,
    devices: &[InputDevice],
    user: &InputUserSettings,
) -> Evaluated {
    let mut best = Evaluated::default();
    let mut best_magnitude = 0.0f32;
    for binding in bindings {
        let dimension = binding.source.dimension();
        for device in devices {
            if device.scheme() != binding.source.scheme() {
                continue;
            }
            let raw = source_value(state, &binding.source, *device);
            let value = apply_user_preferences(
                apply_modifiers(raw, &binding.modifiers, dimension),
                &binding.source,
                user,
            );
            let value = match kind {
                ActionKind::Axis2D => value,
                ActionKind::Button | ActionKind::Axis1D => Vec2::new(value.x, 0.0),
            };
            let magnitude = value.length();
            if magnitude > best_magnitude {
                best_magnitude = magnitude;
                best = Evaluated {
                    value,
                    device: Some(*device),
                };
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use winit::event::ElementState;
    use winit::keyboard::KeyCode;

    fn wasd() -> InputSource {
        InputSource::Composite2D {
            up: Box::new(InputSource::Key(KeyCode::KeyW)),
            down: Box::new(InputSource::Key(KeyCode::KeyS)),
            left: Box::new(InputSource::Key(KeyCode::KeyA)),
            right: Box::new(InputSource::Key(KeyCode::KeyD)),
        }
    }

    #[test]
    fn composites_and_normalize() {
        let mut state = InputState::default();
        state.process_key_input(KeyCode::KeyW, ElementState::Pressed, false);
        state.process_key_input(KeyCode::KeyD, ElementState::Pressed, false);
        let raw = source_value(&state, &wasd(), InputDevice::KeyboardMouse);
        assert_eq!(raw, Vec2::new(1.0, 1.0));
        let normalized = apply_modifiers(raw, &[Modifier::Normalize], 2);
        assert!((normalized.length() - 1.0).abs() < 1e-5);
        // Keyboard sources read nothing on a gamepad device.
        assert_eq!(
            source_value(&state, &wasd(), InputDevice::Gamepad(0)),
            Vec2::ZERO
        );
    }

    #[test]
    fn radial_deadzone_rescales() {
        let value = apply_modifiers(Vec2::new(0.1, 0.0), &[Modifier::Deadzone(0.2)], 2);
        assert_eq!(value, Vec2::ZERO);
        let value = apply_modifiers(Vec2::new(0.6, 0.0), &[Modifier::Deadzone(0.2)], 2);
        assert!((value.x - 0.5).abs() < 1e-5);
        let value = apply_modifiers(
            Vec2::new(1.0, 0.0),
            &[Modifier::Deadzone(0.2), Modifier::Invert],
            2,
        );
        assert!((value.x + 1.0).abs() < 1e-5);
    }

    #[test]
    fn strongest_device_wins() {
        let mut state = InputState::default();
        state.process_gamepad_axis_input(1, Axis::LeftStickX, 0.9);
        state.process_key_input(KeyCode::KeyA, ElementState::Pressed, false);
        let bindings = vec![
            Binding::new(wasd()),
            Binding::new(InputSource::GamepadStick(Stick::Left)),
        ];
        let result = evaluate_bindings(
            &state,
            &bindings,
            ActionKind::Axis2D,
            &[InputDevice::KeyboardMouse, InputDevice::Gamepad(1)],
            &InputUserSettings::default(),
        );
        assert_eq!(result.device, Some(InputDevice::KeyboardMouse));
        assert_eq!(result.value, Vec2::new(-1.0, 0.0));
    }

    #[test]
    fn user_preferences_scale_and_invert_look() {
        let user = InputUserSettings {
            mouse_sensitivity: 2.0,
            invert_y: true,
            ..Default::default()
        };
        let value = apply_user_preferences(Vec2::new(1.0, 1.0), &InputSource::MouseMotion, &user);
        assert_eq!(value, Vec2::new(2.0, -2.0));
    }
}
