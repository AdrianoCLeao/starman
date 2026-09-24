//! Input action maps (ADR 0017): the `InputActions` asset (`*.input.ron`).
//!
//! Gameplay reads *actions* ("move", "jump", "ui_submit"), never raw keys.
//! Actions live in named contexts that can be pushed/popped per player;
//! each action has typed bindings for one or more control schemes, with
//! modifiers (deadzone, invert, scale, normalize, …) and an interaction
//! (press, release, hold, tap, multi-tap) deciding when it *performs*.

use std::collections::HashSet;

use engine_assets::{Asset, AssetLoader, LoadContext};
use engine_core::Result;
use gilrs::{Axis, Button};
use serde::{Deserialize, Serialize};
use winit::event::MouseButton;
use winit::keyboard::KeyCode;

/// Current `InputActions` format version.
pub const INPUT_ACTIONS_VERSION: u32 = 1;

/// What kind of value an action produces.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionKind {
    /// Digital (analog sources are thresholded at `press_point`).
    #[default]
    Button,
    /// One signed axis in [-1, 1] (or unbounded for mouse sources).
    Axis1D,
    /// Two axes (stick, WASD composite, mouse motion).
    Axis2D,
}

/// Which physical family of devices a binding belongs to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ControlScheme {
    #[default]
    KeyboardMouse,
    Gamepad,
}

impl ControlScheme {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::KeyboardMouse => "keyboard_mouse",
            Self::Gamepad => "gamepad",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Stick {
    Left,
    Right,
    DPad,
}

/// A physical input.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum InputSource {
    /// Physical key by W3C UI Events code name (`KeyW`, `Space`, …).
    Key(KeyCode),
    MouseButton(MouseButton),
    /// Relative mouse motion (pixels this frame; 2D).
    MouseMotion,
    /// Vertical wheel (lines this frame; 1D).
    MouseWheel,
    GamepadButton(Button),
    /// One gamepad axis in [-1, 1].
    GamepadAxis(Axis),
    /// A full stick (2D).
    GamepadStick(Stick),
    /// Two digital sources as an axis (negative, positive).
    Composite1D {
        negative: Box<InputSource>,
        positive: Box<InputSource>,
    },
    /// Four digital sources as a 2D axis (WASD, arrows, d-pad buttons).
    Composite2D {
        up: Box<InputSource>,
        down: Box<InputSource>,
        left: Box<InputSource>,
        right: Box<InputSource>,
    },
}

impl InputSource {
    /// The scheme this source belongs to (composites take their first
    /// part's scheme; validation rejects mixed composites).
    pub fn scheme(&self) -> ControlScheme {
        match self {
            Self::Key(_) | Self::MouseButton(_) | Self::MouseMotion | Self::MouseWheel => {
                ControlScheme::KeyboardMouse
            }
            Self::GamepadButton(_) | Self::GamepadAxis(_) | Self::GamepadStick(_) => {
                ControlScheme::Gamepad
            }
            Self::Composite1D { negative, .. } => negative.scheme(),
            Self::Composite2D { up, .. } => up.scheme(),
        }
    }

    /// Natural dimension of the source (1 or 2).
    pub fn dimension(&self) -> u8 {
        match self {
            Self::MouseMotion | Self::GamepadStick(_) | Self::Composite2D { .. } => 2,
            _ => 1,
        }
    }

    /// Whether the source is a single digital-capable input (valid as a
    /// composite part).
    pub fn is_simple(&self) -> bool {
        matches!(
            self,
            Self::Key(_) | Self::MouseButton(_) | Self::GamepadButton(_) | Self::GamepadAxis(_)
        )
    }

    /// Short human-readable label for UI prompts (`W`, `Mouse Left`,
    /// `South`, `Left Stick`).
    pub fn display_name(&self) -> String {
        match self {
            Self::Key(code) => {
                let name = format!("{code:?}");
                name.strip_prefix("Key")
                    .or_else(|| name.strip_prefix("Digit"))
                    .map(str::to_owned)
                    .unwrap_or(name)
            }
            Self::MouseButton(button) => format!("Mouse {button:?}"),
            Self::MouseMotion => "Mouse".to_owned(),
            Self::MouseWheel => "Mouse Wheel".to_owned(),
            Self::GamepadButton(button) => format!("{button:?}"),
            Self::GamepadAxis(axis) => format!("{axis:?}"),
            Self::GamepadStick(stick) => format!("{stick:?} Stick"),
            Self::Composite1D { negative, positive } => {
                format!("{}/{}", negative.display_name(), positive.display_name())
            }
            Self::Composite2D {
                up,
                left,
                down,
                right,
            } => format!(
                "{}{}{}{}",
                up.display_name(),
                left.display_name(),
                down.display_name(),
                right.display_name()
            ),
        }
    }

    /// Simple sources contained in this one (itself, or composite parts).
    pub fn leaves(&self) -> Vec<&InputSource> {
        match self {
            Self::Composite1D { negative, positive } => {
                let mut out = negative.leaves();
                out.extend(positive.leaves());
                out
            }
            Self::Composite2D {
                up,
                down,
                left,
                right,
            } => {
                let mut out = up.leaves();
                out.extend(down.leaves());
                out.extend(left.leaves());
                out.extend(right.leaves());
                out
            }
            other => vec![other],
        }
    }
}

/// Post-processing applied to a binding's raw value, in order.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Modifier {
    /// Radial deadzone for 2D, magnitude deadzone for 1D; rescales the
    /// remaining range to [0, 1].
    Deadzone(f32),
    /// Negate both axes.
    Invert,
    InvertX,
    InvertY,
    /// Multiply by a factor (sensitivity).
    Scale(f32),
    ScaleXY(f32, f32),
    /// Clamp 2D length to 1 (diagonal WASD is not faster).
    Normalize,
    /// Swap X and Y.
    Swizzle,
}

/// When an actuated action *performs*.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum Interaction {
    /// Performs when actuation starts (and on every frame for axes while
    /// actuated).
    #[default]
    Press,
    /// Performs when actuation ends.
    Release,
    /// Performs once after being held for `seconds`.
    Hold { seconds: f32 },
    /// Performs on release when held shorter than `max_seconds`.
    Tap { max_seconds: f32 },
    /// Performs on the `count`-th tap, taps at most `max_gap` apart.
    MultiTap { count: u32, max_gap: f32 },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Binding {
    pub source: InputSource,
    #[serde(default)]
    pub modifiers: Vec<Modifier>,
}

impl Binding {
    pub fn new(source: InputSource) -> Self {
        Self {
            source,
            modifiers: Vec::new(),
        }
    }

    pub fn with(mut self, modifier: Modifier) -> Self {
        self.modifiers.push(modifier);
        self
    }
}

fn default_press_point() -> f32 {
    0.5
}

fn default_true() -> bool {
    true
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionDef {
    pub name: String,
    #[serde(default)]
    pub kind: ActionKind,
    #[serde(default)]
    pub bindings: Vec<Binding>,
    #[serde(default)]
    pub interaction: Interaction,
    /// Actuation threshold for digital interpretation of analog sources.
    #[serde(default = "default_press_point")]
    pub press_point: f32,
    /// Players may rebind this action from the options menu.
    #[serde(default = "default_true")]
    pub rebindable: bool,
    /// Localization key for the action's display name (options menu).
    #[serde(default)]
    pub label: Option<String>,
}

impl ActionDef {
    pub fn new(name: impl Into<String>, kind: ActionKind) -> Self {
        Self {
            name: name.into(),
            kind,
            bindings: Vec::new(),
            interaction: Interaction::Press,
            press_point: default_press_point(),
            rebindable: true,
            label: None,
        }
    }

    pub fn bind(mut self, binding: Binding) -> Self {
        self.bindings.push(binding);
        self
    }

    pub fn with_interaction(mut self, interaction: Interaction) -> Self {
        self.interaction = interaction;
        self
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionContext {
    pub name: String,
    /// While active, contexts below it on a player's stack are not
    /// evaluated (a pause menu blocks gameplay).
    #[serde(default)]
    pub blocks_lower: bool,
    #[serde(default)]
    pub actions: Vec<ActionDef>,
}

/// The `*.input.ron` asset.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InputActions {
    pub version: u32,
    pub contexts: Vec<ActionContext>,
    /// Contexts every player starts with, bottom first.
    #[serde(default)]
    pub default_contexts: Vec<String>,
}

impl Asset for InputActions {
    const TYPE_NAME: &'static str = "InputActions";
}

impl InputActions {
    pub fn context(&self, name: &str) -> Option<&ActionContext> {
        self.contexts.iter().find(|context| context.name == name)
    }

    pub fn action(&self, context: &str, action: &str) -> Option<&ActionDef> {
        self.context(context)?
            .actions
            .iter()
            .find(|def| def.name == action)
    }

    /// Structural validation with actionable messages.
    pub fn validate(&self) -> std::result::Result<(), Vec<String>> {
        let mut errors = Vec::new();
        if self.version != INPUT_ACTIONS_VERSION {
            errors.push(format!(
                "unsupported InputActions version {} (expected {INPUT_ACTIONS_VERSION})",
                self.version
            ));
        }
        let mut context_names = HashSet::new();
        for context in &self.contexts {
            if context.name.trim().is_empty() {
                errors.push("a context has an empty name".to_owned());
            }
            if !context_names.insert(context.name.as_str()) {
                errors.push(format!("context '{}' is declared twice", context.name));
            }
            let mut action_names = HashSet::new();
            for action in &context.actions {
                let at = format!("{}/{}", context.name, action.name);
                if action.name.trim().is_empty() {
                    errors.push(format!(
                        "context '{}' has an action with an empty name",
                        context.name
                    ));
                }
                if !action_names.insert(action.name.as_str()) {
                    errors.push(format!("action '{at}' is declared twice"));
                }
                if !(0.0..=1.0).contains(&action.press_point) {
                    errors.push(format!("action '{at}' press_point must be in [0, 1]"));
                }
                match &action.interaction {
                    Interaction::Hold { seconds } if *seconds <= 0.0 => {
                        errors.push(format!("action '{at}' hold duration must be positive"))
                    }
                    Interaction::Tap { max_seconds } if *max_seconds <= 0.0 => {
                        errors.push(format!("action '{at}' tap window must be positive"))
                    }
                    Interaction::MultiTap { count, max_gap } if *count < 2 || *max_gap <= 0.0 => {
                        errors.push(format!(
                            "action '{at}' multi-tap needs count >= 2 and a positive gap"
                        ))
                    }
                    _ => {}
                }
                for (index, binding) in action.bindings.iter().enumerate() {
                    validate_binding(&at, index, action.kind, binding, &mut errors);
                }
            }
        }
        for name in &self.default_contexts {
            if !context_names.contains(name.as_str()) {
                errors.push(format!("default context '{name}' is not declared"));
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

fn validate_binding(
    at: &str,
    index: usize,
    kind: ActionKind,
    binding: &Binding,
    errors: &mut Vec<String>,
) {
    let source = &binding.source;
    let dimension = source.dimension();
    if kind == ActionKind::Axis2D && dimension != 2 {
        errors.push(format!(
            "action '{at}' binding {index}: Axis2D needs a 2D source (stick, mouse motion, Composite2D)"
        ));
    }
    if kind != ActionKind::Axis2D && dimension == 2 {
        errors.push(format!(
            "action '{at}' binding {index}: a 2D source cannot drive a {kind:?} action"
        ));
    }
    let leaves = source.leaves();
    if matches!(
        source,
        InputSource::Composite1D { .. } | InputSource::Composite2D { .. }
    ) {
        if leaves.iter().any(|leaf| !leaf.is_simple()) {
            errors.push(format!(
                "action '{at}' binding {index}: composite parts must be keys or buttons"
            ));
        }
        let scheme = source.scheme();
        if leaves.iter().any(|leaf| leaf.scheme() != scheme) {
            errors.push(format!(
                "action '{at}' binding {index}: composite mixes keyboard/mouse and gamepad parts"
            ));
        }
    }
    for modifier in &binding.modifiers {
        match modifier {
            Modifier::Deadzone(value) if !(0.0..1.0).contains(value) => errors.push(format!(
                "action '{at}' binding {index}: deadzone must be in [0, 1)"
            )),
            Modifier::Scale(value) if !value.is_finite() => errors.push(format!(
                "action '{at}' binding {index}: scale must be finite"
            )),
            Modifier::ScaleXY(x, y) if !x.is_finite() || !y.is_finite() => errors.push(format!(
                "action '{at}' binding {index}: scale must be finite"
            )),
            _ => {}
        }
    }
}

/// Loads and validates `*.input.ron`.
pub struct InputActionsLoader;

impl AssetLoader for InputActionsLoader {
    type Asset = InputActions;

    fn extensions(&self) -> &'static [&'static str] {
        &["input.ron"]
    }

    fn load(&self, bytes: &[u8], ctx: &mut LoadContext<'_>) -> Result<InputActions> {
        let text =
            std::str::from_utf8(bytes).map_err(|_| ctx.error("input actions are not UTF-8"))?;
        let actions: InputActions = ron::from_str(text)
            .map_err(|error| ctx.error(format!("invalid input actions: {error}")))?;
        actions
            .validate()
            .map_err(|errors| ctx.error(errors.join("; ")))?;
        Ok(actions)
    }
}

#[cfg(test)]
pub(crate) mod tests_support {
    pub(crate) const SAMPLE: &str = r#"(
        version: 1,
        default_contexts: ["gameplay"],
        contexts: [
            (
                name: "gameplay",
                actions: [
                    (
                        name: "move",
                        kind: Axis2D,
                        bindings: [
                            (source: Composite2D(up: Key(KeyW), down: Key(KeyS), left: Key(KeyA), right: Key(KeyD)), modifiers: [Normalize]),
                            (source: GamepadStick(Left), modifiers: [Deadzone(0.2)]),
                        ],
                    ),
                    (name: "jump", bindings: [(source: Key(Space)), (source: GamepadButton(South))]),
                    (name: "attack", bindings: [(source: MouseButton(Left))], interaction: Tap(max_seconds: 0.3)),
                    (name: "interact", bindings: [(source: Key(KeyE))], interaction: Hold(seconds: 0.5)),
                    (name: "dodge", bindings: [(source: Key(ShiftLeft))], interaction: MultiTap(count: 2, max_gap: 0.3)),
                    (name: "zoom", kind: Axis1D, bindings: [(source: MouseWheel, modifiers: [Scale(0.5)])]),
                ],
            ),
            (
                name: "menu",
                blocks_lower: true,
                actions: [
                    (name: "ui_submit", bindings: [(source: Key(Enter)), (source: GamepadButton(South))]),
                    (name: "ui_cancel", bindings: [(source: Key(Escape)), (source: GamepadButton(East))]),
                ],
            ),
        ],
    )"#;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> InputActions {
        ron::from_str(tests_support::SAMPLE).expect("sample parses")
    }

    #[test]
    fn sample_asset_parses_and_validates() {
        let actions = sample();
        actions.validate().expect("valid");
        assert_eq!(
            actions.action("gameplay", "move").unwrap().kind,
            ActionKind::Axis2D
        );
        assert!(actions.context("menu").unwrap().blocks_lower);
        assert_eq!(
            actions.action("gameplay", "jump").unwrap().bindings[1]
                .source
                .scheme(),
            ControlScheme::Gamepad
        );
    }

    #[test]
    fn validation_reports_every_problem() {
        let mut actions = sample();
        actions.contexts[0]
            .actions
            .push(ActionDef::new("jump", ActionKind::Button));
        actions.contexts[0].actions[0]
            .bindings
            .push(Binding::new(InputSource::Key(KeyCode::KeyQ)));
        actions.contexts[0].actions[1]
            .bindings
            .push(Binding::new(InputSource::Composite1D {
                negative: Box::new(InputSource::Key(KeyCode::KeyQ)),
                positive: Box::new(InputSource::GamepadButton(Button::South)),
            }));
        actions.default_contexts.push("missing".into());
        let errors = actions.validate().unwrap_err();
        let joined = errors.join("\n");
        assert!(joined.contains("declared twice"), "{joined}");
        assert!(joined.contains("Axis2D needs a 2D source"), "{joined}");
        assert!(
            joined.contains("mixes keyboard/mouse and gamepad"),
            "{joined}"
        );
        assert!(joined.contains("default context 'missing'"), "{joined}");
    }

    #[test]
    fn display_names_are_short() {
        assert_eq!(InputSource::Key(KeyCode::KeyW).display_name(), "W");
        assert_eq!(InputSource::Key(KeyCode::Digit1).display_name(), "1");
        assert_eq!(
            InputSource::GamepadStick(Stick::Left).display_name(),
            "Left Stick"
        );
    }
}
