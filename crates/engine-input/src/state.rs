//! Per-action runtime state and the interaction state machines.

use engine_math::Vec2;

use crate::actions::{ActionDef, ActionKind, ControlScheme, Interaction};

/// Lifecycle phase reported through `ActionEvent`s.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActionPhase {
    /// Actuation (or the interaction) began.
    Started,
    /// The interaction completed (press, release, hold time reached, tap,
    /// multi-tap).
    Performed,
    /// Actuation ended without performing (hold released early, tap held
    /// too long).
    Canceled,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ActionState {
    /// Current value (`x` for buttons/1D).
    pub value: Vec2,
    /// Whether the action is actuated beyond its press point.
    pub active: bool,
    pub just_pressed: bool,
    pub just_released: bool,
    /// The interaction performed this frame.
    pub performed: bool,
    /// Seconds the current actuation has lasted.
    pub held_seconds: f32,
    /// Scheme of the device that last drove this action.
    pub scheme: Option<ControlScheme>,
    /// Latched for the fixed-step simulation: set by `latch_fixed` when a
    /// press/perform happened since the previous fixed step.
    pub fixed_just_pressed: bool,
    pub fixed_performed: bool,
    presses_since_fixed: u32,
    performs_since_fixed: u32,
    hold_fired: bool,
    tap_count: u32,
    last_tap_end: f32,
}

impl ActionState {
    /// Digital state.
    pub fn pressed(&self) -> bool {
        self.active
    }

    pub fn axis(&self) -> f32 {
        self.value.x
    }

    pub fn axis2d(&self) -> Vec2 {
        self.value
    }

    /// Advances the state with this frame's evaluated `value`. Returns the
    /// phases entered this frame.
    pub fn update(&mut self, def: &ActionDef, value: Vec2, dt: f32, now: f32) -> Vec<ActionPhase> {
        let mut phases = Vec::new();
        let magnitude = match def.kind {
            ActionKind::Axis2D => value.length(),
            _ => value.x.abs(),
        };
        let was_active = self.active;
        let active = magnitude >= def.press_point.max(1e-4);
        self.value = match def.kind {
            ActionKind::Button => Vec2::new(if active { 1.0 } else { 0.0 }, 0.0),
            _ => value,
        };
        self.active = active;
        self.just_pressed = active && !was_active;
        self.just_released = !active && was_active;
        self.performed = false;
        let released_after = self.held_seconds;
        if active {
            self.held_seconds = if was_active {
                self.held_seconds + dt
            } else {
                0.0
            };
        }

        if self.just_pressed {
            self.presses_since_fixed += 1;
            phases.push(ActionPhase::Started);
            self.hold_fired = false;
        }

        match &def.interaction {
            Interaction::Press => {
                if self.just_pressed {
                    self.perform(&mut phases);
                }
            }
            Interaction::Release => {
                if self.just_released {
                    self.perform(&mut phases);
                }
            }
            Interaction::Hold { seconds } => {
                if active && !self.hold_fired && self.held_seconds >= *seconds {
                    self.hold_fired = true;
                    self.perform(&mut phases);
                }
                if self.just_released && !self.hold_fired {
                    phases.push(ActionPhase::Canceled);
                }
            }
            Interaction::Tap { max_seconds } => {
                if self.just_released {
                    if released_after <= *max_seconds {
                        self.perform(&mut phases);
                    } else {
                        phases.push(ActionPhase::Canceled);
                    }
                }
            }
            Interaction::MultiTap { count, max_gap } => {
                if self.just_pressed && self.tap_count > 0 && now - self.last_tap_end > *max_gap {
                    self.tap_count = 0;
                }
                if self.just_released {
                    self.tap_count += 1;
                    self.last_tap_end = now;
                    if self.tap_count >= *count {
                        self.tap_count = 0;
                        self.perform(&mut phases);
                    }
                }
                if !active && self.tap_count > 0 && now - self.last_tap_end > *max_gap {
                    self.tap_count = 0;
                    phases.push(ActionPhase::Canceled);
                }
            }
        }
        if !active {
            self.held_seconds = 0.0;
        }
        phases
    }

    fn perform(&mut self, phases: &mut Vec<ActionPhase>) {
        self.performed = true;
        self.performs_since_fixed += 1;
        phases.push(ActionPhase::Performed);
    }

    /// Latches presses/performs that happened since the previous fixed
    /// step (run at the start of every fixed step).
    pub fn latch_fixed(&mut self) {
        self.fixed_just_pressed = self.presses_since_fixed > 0;
        self.fixed_performed = self.performs_since_fixed > 0;
        self.presses_since_fixed = 0;
        self.performs_since_fixed = 0;
    }

    /// Clears everything (context deactivated).
    pub fn reset(&mut self) -> bool {
        let was_active = self.active;
        *self = Self::default();
        was_active
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actions::ActionDef;

    fn run(
        state: &mut ActionState,
        def: &ActionDef,
        values: &[f32],
        dt: f32,
    ) -> Vec<Vec<ActionPhase>> {
        let mut now = 0.0;
        values
            .iter()
            .map(|value| {
                now += dt;
                state.update(def, Vec2::new(*value, 0.0), dt, now)
            })
            .collect()
    }

    #[test]
    fn press_performs_once_on_actuation() {
        let def = ActionDef::new("jump", ActionKind::Button);
        let mut state = ActionState::default();
        let phases = run(&mut state, &def, &[0.0, 1.0, 1.0, 0.0], 0.1);
        assert_eq!(
            phases[1],
            vec![ActionPhase::Started, ActionPhase::Performed]
        );
        assert!(phases[2].is_empty());
        assert!(state.just_released);
    }

    #[test]
    fn hold_performs_after_duration_or_cancels() {
        let def = ActionDef::new("interact", ActionKind::Button)
            .with_interaction(Interaction::Hold { seconds: 0.25 });
        let mut state = ActionState::default();
        let phases = run(&mut state, &def, &[1.0, 1.0, 1.0, 1.0, 0.0], 0.1);
        assert!(
            phases
                .iter()
                .flatten()
                .filter(|p| **p == ActionPhase::Performed)
                .count()
                == 1
        );
        let mut state = ActionState::default();
        let phases = run(&mut state, &def, &[1.0, 1.0, 0.0], 0.1);
        assert_eq!(phases[2], vec![ActionPhase::Canceled]);
    }

    #[test]
    fn tap_requires_quick_release() {
        let def = ActionDef::new("attack", ActionKind::Button)
            .with_interaction(Interaction::Tap { max_seconds: 0.15 });
        let mut state = ActionState::default();
        let phases = run(&mut state, &def, &[1.0, 1.0, 0.0], 0.1);
        assert!(phases[2].contains(&ActionPhase::Performed));
        let mut state = ActionState::default();
        let phases = run(&mut state, &def, &[1.0, 1.0, 1.0, 0.0], 0.1);
        assert!(phases[3].contains(&ActionPhase::Canceled));
    }

    #[test]
    fn double_tap_within_gap() {
        let def =
            ActionDef::new("dodge", ActionKind::Button).with_interaction(Interaction::MultiTap {
                count: 2,
                max_gap: 0.25,
            });
        let mut state = ActionState::default();
        let phases = run(&mut state, &def, &[1.0, 0.0, 1.0, 0.0], 0.1);
        assert!(phases[3].contains(&ActionPhase::Performed));
        let mut state = ActionState::default();
        let phases = run(&mut state, &def, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 0.1);
        assert!(!phases
            .iter()
            .flatten()
            .any(|p| *p == ActionPhase::Performed));
    }

    #[test]
    fn fixed_latch_survives_frames_without_fixed_steps() {
        let def = ActionDef::new("jump", ActionKind::Button);
        let mut state = ActionState::default();
        run(&mut state, &def, &[1.0, 0.0], 0.1);
        state.latch_fixed();
        assert!(state.fixed_just_pressed);
        assert!(state.fixed_performed);
        state.latch_fixed();
        assert!(!state.fixed_just_pressed);
    }

    #[test]
    fn axes_keep_analog_values() {
        let def = ActionDef::new("zoom", ActionKind::Axis1D);
        let mut state = ActionState::default();
        state.update(&def, Vec2::new(0.3, 0.0), 0.016, 0.016);
        assert_eq!(state.axis(), 0.3);
        assert!(!state.active);
    }
}
