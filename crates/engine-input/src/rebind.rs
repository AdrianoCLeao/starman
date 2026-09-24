//! Interactive rebinding: listen for the next input on the requested
//! scheme, detect conflicts, and write a user override.

use bevy_ecs::prelude::{Event, Resource};

use crate::actions::{ActionKind, Binding, ControlScheme, InputActions, InputSource};
use crate::settings::InputUserSettings;
use crate::InputDevice;

#[derive(Clone, Debug, PartialEq)]
pub struct RebindRequest {
    pub player: u8,
    pub context: String,
    pub action: String,
    /// Binding (of `scheme`) to replace; `None` replaces the first binding
    /// of that scheme or appends one.
    pub binding: Option<usize>,
    pub scheme: ControlScheme,
    /// Inputs that cancel instead of binding (Escape, Start).
    pub cancel_sources: Vec<InputSource>,
    pub timeout_seconds: f32,
    /// Remove the captured input from conflicting actions of the context.
    pub swap_conflicts: bool,
}

impl RebindRequest {
    pub fn new(
        context: impl Into<String>,
        action: impl Into<String>,
        scheme: ControlScheme,
    ) -> Self {
        Self {
            player: 0,
            context: context.into(),
            action: action.into(),
            binding: None,
            scheme,
            cancel_sources: vec![
                InputSource::Key(winit::keyboard::KeyCode::Escape),
                InputSource::GamepadButton(gilrs::Button::Start),
            ],
            timeout_seconds: 8.0,
            swap_conflicts: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum RebindOutcome {
    Bound {
        context: String,
        action: String,
        source: InputSource,
        /// Other actions of the context that also use `source`.
        conflicts: Vec<String>,
    },
    Canceled,
    TimedOut,
    /// The request named an action that does not exist or cannot be rebound.
    Rejected(String),
}

#[derive(Event, Clone, Debug, PartialEq)]
pub struct RebindEvent(pub RebindOutcome);

#[derive(Resource, Default, Debug)]
pub struct Rebinding {
    active: Option<(RebindRequest, f32)>,
    pub last_outcome: Option<RebindOutcome>,
}

impl Rebinding {
    pub fn start(&mut self, request: RebindRequest) {
        self.active = Some((request, 0.0));
        self.last_outcome = None;
    }

    pub fn cancel(&mut self) -> bool {
        self.active.take().is_some()
    }

    pub fn is_active(&self) -> bool {
        self.active.is_some()
    }

    pub fn request(&self) -> Option<&RebindRequest> {
        self.active.as_ref().map(|(request, _)| request)
    }

    /// Advances the session with this frame's presses (of devices owned by
    /// the requesting player). Returns an outcome when the session ends.
    pub(crate) fn update(
        &mut self,
        dt: f32,
        presses: &[(InputDevice, InputSource)],
        owned: &dyn Fn(u8, InputDevice) -> bool,
        actions: Option<&InputActions>,
        user: &mut InputUserSettings,
    ) -> Option<RebindOutcome> {
        let (request, elapsed) = self.active.as_mut()?;
        *elapsed += dt;
        let request = request.clone();
        let elapsed = *elapsed;
        let outcome = (|| {
            let Some(actions) = actions else {
                return Some(RebindOutcome::Rejected(
                    "no input actions are loaded".to_owned(),
                ));
            };
            let Some(def) = actions.action(&request.context, &request.action) else {
                return Some(RebindOutcome::Rejected(format!(
                    "unknown action '{}/{}'",
                    request.context, request.action
                )));
            };
            if !def.rebindable {
                return Some(RebindOutcome::Rejected(format!(
                    "action '{}/{}' is not rebindable",
                    request.context, request.action
                )));
            }
            for (device, source) in presses {
                if !owned(request.player, *device) {
                    continue;
                }
                if request.cancel_sources.contains(source) {
                    return Some(RebindOutcome::Canceled);
                }
                if source.scheme() != request.scheme {
                    continue;
                }
                // A digital capture only fits 1D/button actions; 2D actions
                // are rebound part by part through composites by the UI.
                if def.kind == ActionKind::Axis2D {
                    continue;
                }
                let mut bindings = user.bindings_for(&request.context, def).to_vec();
                let slot = request
                    .binding
                    .filter(|index| *index < bindings.len())
                    .or_else(|| {
                        bindings
                            .iter()
                            .position(|binding| binding.source.scheme() == request.scheme)
                    });
                match slot {
                    Some(index) => bindings[index].source = source.clone(),
                    None => bindings.push(Binding::new(source.clone())),
                }
                user.set_bindings(&request.context, &request.action, bindings);
                let conflicts =
                    conflicting_actions(actions, user, &request.context, &request.action, source);
                if request.swap_conflicts {
                    for other in &conflicts {
                        if let Some(other_def) = actions.action(&request.context, other) {
                            let kept: Vec<Binding> = user
                                .bindings_for(&request.context, other_def)
                                .iter()
                                .filter(|binding| !binding.source.leaves().contains(&source))
                                .cloned()
                                .collect();
                            user.set_bindings(&request.context, other, kept);
                        }
                    }
                }
                return Some(RebindOutcome::Bound {
                    context: request.context.clone(),
                    action: request.action.clone(),
                    source: source.clone(),
                    conflicts: if request.swap_conflicts {
                        Vec::new()
                    } else {
                        conflicts
                    },
                });
            }
            (elapsed >= request.timeout_seconds).then_some(RebindOutcome::TimedOut)
        })();
        if outcome.is_some() {
            self.active = None;
            self.last_outcome = outcome.clone();
        }
        outcome
    }
}

/// Other actions of `context` whose bindings contain `source`.
pub fn conflicting_actions(
    actions: &InputActions,
    user: &InputUserSettings,
    context: &str,
    action: &str,
    source: &InputSource,
) -> Vec<String> {
    let Some(context_def) = actions.context(context) else {
        return Vec::new();
    };
    context_def
        .actions
        .iter()
        .filter(|def| def.name != action)
        .filter(|def| {
            user.bindings_for(context, def)
                .iter()
                .any(|binding| binding.source.leaves().contains(&source))
        })
        .map(|def| def.name.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use winit::keyboard::KeyCode;

    fn actions() -> InputActions {
        ron::from_str(crate::actions::tests_support::SAMPLE).unwrap()
    }

    #[test]
    fn captures_the_next_press_and_reports_conflicts() {
        let actions = actions();
        let mut user = InputUserSettings::default();
        let mut rebinding = Rebinding::default();
        rebinding.start(RebindRequest::new(
            "gameplay",
            "jump",
            ControlScheme::KeyboardMouse,
        ));
        let owned = |_: u8, _: InputDevice| true;
        // A gamepad press is ignored for a keyboard rebind.
        let none = rebinding.update(
            0.1,
            &[(
                InputDevice::Gamepad(0),
                InputSource::GamepadButton(gilrs::Button::North),
            )],
            &owned,
            Some(&actions),
            &mut user,
        );
        assert!(none.is_none());
        let outcome = rebinding
            .update(
                0.1,
                &[(InputDevice::KeyboardMouse, InputSource::Key(KeyCode::KeyE))],
                &owned,
                Some(&actions),
                &mut user,
            )
            .unwrap();
        match outcome {
            RebindOutcome::Bound { conflicts, .. } => {
                assert_eq!(conflicts, vec!["interact".to_owned()])
            }
            other => panic!("unexpected {other:?}"),
        }
        let jump = actions.action("gameplay", "jump").unwrap();
        assert_eq!(
            user.bindings_for("gameplay", jump)[0].source,
            InputSource::Key(KeyCode::KeyE)
        );
        // The gamepad binding is untouched.
        assert_eq!(user.bindings_for("gameplay", jump).len(), 2);
    }

    #[test]
    fn swap_removes_the_input_from_the_other_action() {
        let actions = actions();
        let mut user = InputUserSettings::default();
        let mut rebinding = Rebinding::default();
        let mut request = RebindRequest::new("gameplay", "jump", ControlScheme::KeyboardMouse);
        request.swap_conflicts = true;
        rebinding.start(request);
        rebinding.update(
            0.1,
            &[(InputDevice::KeyboardMouse, InputSource::Key(KeyCode::KeyE))],
            &|_, _| true,
            Some(&actions),
            &mut user,
        );
        let interact = actions.action("gameplay", "interact").unwrap();
        assert!(user.bindings_for("gameplay", interact).is_empty());
    }

    #[test]
    fn cancel_and_timeout() {
        let actions = actions();
        let mut user = InputUserSettings::default();
        let mut rebinding = Rebinding::default();
        rebinding.start(RebindRequest::new(
            "gameplay",
            "jump",
            ControlScheme::KeyboardMouse,
        ));
        let outcome = rebinding.update(
            0.1,
            &[(
                InputDevice::KeyboardMouse,
                InputSource::Key(KeyCode::Escape),
            )],
            &|_, _| true,
            Some(&actions),
            &mut user,
        );
        assert_eq!(outcome, Some(RebindOutcome::Canceled));
        rebinding.start(RebindRequest::new(
            "gameplay",
            "jump",
            ControlScheme::KeyboardMouse,
        ));
        assert_eq!(
            rebinding.update(100.0, &[], &|_, _| true, Some(&actions), &mut user),
            Some(RebindOutcome::TimedOut)
        );
        rebinding.start(RebindRequest::new(
            "gameplay",
            "nope",
            ControlScheme::KeyboardMouse,
        ));
        assert!(matches!(
            rebinding.update(0.1, &[], &|_, _| true, Some(&actions), &mut user),
            Some(RebindOutcome::Rejected(_))
        ));
    }
}
