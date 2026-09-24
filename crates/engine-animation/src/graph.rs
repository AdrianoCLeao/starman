//! Animation graphs (`*.animgraph.ron`): parameters, layers with bone
//! masks and override/additive blending, state machines (nested through
//! sub-state machines) with conditional/exit-time transitions and
//! any-state transitions, and 1D/2D blend trees.

use std::collections::HashSet;

use engine_assets::{Asset, AssetLoader, AssetRef, LoadContext};
use engine_core::Result;
use engine_math::Vec2;
use serde::{Deserialize, Serialize};

pub const ANIM_GRAPH_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ParameterValue {
    Float(f32),
    Int(i32),
    Bool(bool),
    /// Set by gameplay, consumed by the first transition that uses it.
    Trigger(bool),
}

impl ParameterValue {
    pub fn as_float(self) -> f32 {
        match self {
            Self::Float(v) => v,
            Self::Int(v) => v as f32,
            Self::Bool(v) | Self::Trigger(v) => {
                if v {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    pub fn as_bool(self) -> bool {
        match self {
            Self::Float(v) => v != 0.0,
            Self::Int(v) => v != 0,
            Self::Bool(v) | Self::Trigger(v) => v,
        }
    }

    pub fn same_kind(self, other: Self) -> bool {
        std::mem::discriminant(&self) == std::mem::discriminant(&other)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterDef {
    pub name: String,
    /// Kind and default value.
    pub default: ParameterValue,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerBlend {
    /// Replaces the pose of the layers below (per the mask and weight).
    #[default]
    Override,
    /// Adds the difference between the clip and its first frame.
    Additive,
}

/// Bones a layer affects.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BoneMask {
    pub bones: Vec<String>,
    /// Also include every descendant of the listed bones.
    #[serde(default = "yes")]
    pub include_descendants: bool,
}

fn yes() -> bool {
    true
}

fn one() -> f32 {
    1.0
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphLayer {
    pub name: String,
    #[serde(default = "one")]
    pub weight: f32,
    /// Float parameter driving the weight (overrides `weight`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight_parameter: Option<String>,
    #[serde(default)]
    pub blend: LayerBlend,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mask: Option<BoneMask>,
    pub state_machine: StateMachine,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct StateMachine {
    /// Initial state name (first state when empty).
    #[serde(default)]
    pub entry: String,
    pub states: Vec<State>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub transitions: Vec<Transition>,
    /// Transitions that may fire from any state of this machine.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub any_state: Vec<Transition>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct State {
    pub name: String,
    pub motion: Motion,
    #[serde(default = "one")]
    pub speed: f32,
    /// Float parameter multiplying `speed`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed_parameter: Option<String>,
    /// Editor canvas position.
    #[serde(default)]
    pub position: Vec2,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Motion {
    /// Nothing (the layer contributes no pose while here).
    Empty,
    Clip(AssetRef),
    /// Children placed on one parameter axis.
    Blend1D {
        parameter: String,
        children: Vec<BlendChild1D>,
    },
    /// Children placed on a 2D parameter plane.
    Blend2D {
        parameter_x: String,
        parameter_y: String,
        #[serde(default)]
        mode: Blend2DMode,
        children: Vec<BlendChild2D>,
    },
    /// A nested state machine (runs while this state is active).
    StateMachine(Box<StateMachine>),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BlendChild1D {
    pub threshold: f32,
    pub motion: Motion,
    #[serde(default = "one")]
    pub speed: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BlendChild2D {
    pub position: Vec2,
    pub motion: Motion,
    #[serde(default = "one")]
    pub speed: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Blend2DMode {
    /// Gradient-band interpolation in polar space: best for locomotion
    /// (directions at several speeds, with an idle at the origin).
    #[default]
    FreeformDirectional,
    /// Gradient-band interpolation in Cartesian space.
    FreeformCartesian,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransitionCurve {
    #[default]
    Linear,
    EaseInOut,
}

impl TransitionCurve {
    pub fn apply(self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => t,
            Self::EaseInOut => t * t * (3.0 - 2.0 * t),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Condition {
    If(String),
    IfNot(String),
    Greater(String, f32),
    Less(String, f32),
    Equals(String, i32),
    NotEquals(String, i32),
}

impl Condition {
    pub fn parameter(&self) -> &str {
        match self {
            Self::If(p)
            | Self::IfNot(p)
            | Self::Greater(p, _)
            | Self::Less(p, _)
            | Self::Equals(p, _)
            | Self::NotEquals(p, _) => p,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Transition {
    /// Source state (ignored for any-state transitions).
    #[serde(default)]
    pub from: String,
    pub to: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub conditions: Vec<Condition>,
    /// Normalized time of the source state (fraction of a loop) after
    /// which the transition may fire; `None` = any time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_time: Option<f32>,
    /// Crossfade seconds.
    #[serde(default)]
    pub duration: f32,
    #[serde(default)]
    pub curve: TransitionCurve,
    /// Whether another transition may interrupt this one.
    #[serde(default)]
    pub interruptible: bool,
    /// Higher priorities are tested first.
    #[serde(default)]
    pub priority: i32,
    /// Any-state only: allow re-entering the current state.
    #[serde(default)]
    pub can_transition_to_self: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnimGraph {
    #[serde(default = "graph_version")]
    pub version: u32,
    #[serde(default)]
    pub parameters: Vec<ParameterDef>,
    pub layers: Vec<GraphLayer>,
}

fn graph_version() -> u32 {
    ANIM_GRAPH_VERSION
}

impl Asset for AnimGraph {
    const TYPE_NAME: &'static str = "AnimGraph";
}

impl Default for AnimGraph {
    fn default() -> Self {
        Self {
            version: ANIM_GRAPH_VERSION,
            parameters: Vec::new(),
            layers: Vec::new(),
        }
    }
}

impl Motion {
    /// Every clip referenced by this motion (recursively).
    pub fn clips<'a>(&'a self, out: &mut Vec<&'a AssetRef>) {
        match self {
            Self::Empty => {}
            Self::Clip(clip) => out.push(clip),
            Self::Blend1D { children, .. } => {
                for child in children {
                    child.motion.clips(out);
                }
            }
            Self::Blend2D { children, .. } => {
                for child in children {
                    child.motion.clips(out);
                }
            }
            Self::StateMachine(machine) => machine.clips(out),
        }
    }
}

impl StateMachine {
    pub fn state_index(&self, name: &str) -> Option<usize> {
        self.states.iter().position(|state| state.name == name)
    }

    pub fn entry_index(&self) -> usize {
        self.state_index(&self.entry).unwrap_or(0)
    }

    pub fn clips<'a>(&'a self, out: &mut Vec<&'a AssetRef>) {
        for state in &self.states {
            state.motion.clips(out);
        }
    }
}

impl AnimGraph {
    pub fn parameter(&self, name: &str) -> Option<&ParameterDef> {
        self.parameters.iter().find(|p| p.name == name)
    }

    /// Every clip the graph can play (for preloading).
    pub fn clips(&self) -> Vec<&AssetRef> {
        let mut out = Vec::new();
        for layer in &self.layers {
            layer.state_machine.clips(&mut out);
        }
        out
    }

    pub fn validate(&self) -> std::result::Result<(), Vec<String>> {
        let mut errors = Vec::new();
        if self.version > ANIM_GRAPH_VERSION {
            errors.push(format!(
                "graph version {} is newer than supported {ANIM_GRAPH_VERSION}",
                self.version
            ));
        }
        let mut names = HashSet::new();
        for parameter in &self.parameters {
            if !names.insert(parameter.name.as_str()) {
                errors.push(format!("parameter '{}' is declared twice", parameter.name));
            }
        }
        if self.layers.is_empty() {
            errors.push("a graph needs at least one layer".to_owned());
        }
        for layer in &self.layers {
            if let Some(parameter) = &layer.weight_parameter {
                self.check_float(
                    parameter,
                    &format!("layer '{}' weight", layer.name),
                    &mut errors,
                );
            }
            self.validate_machine(&layer.state_machine, &layer.name, &mut errors);
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    fn check_float(&self, name: &str, context: &str, errors: &mut Vec<String>) {
        match self.parameter(name).map(|p| p.default) {
            None => errors.push(format!("{context}: unknown parameter '{name}'")),
            Some(ParameterValue::Float(_) | ParameterValue::Int(_)) => {}
            Some(_) => errors.push(format!("{context}: parameter '{name}' is not numeric")),
        }
    }

    fn validate_machine(&self, machine: &StateMachine, path: &str, errors: &mut Vec<String>) {
        if machine.states.is_empty() {
            errors.push(format!("{path}: state machine has no states"));
            return;
        }
        let mut names = HashSet::new();
        for state in &machine.states {
            if !names.insert(state.name.as_str()) {
                errors.push(format!("{path}: state '{}' is declared twice", state.name));
            }
            if let Some(parameter) = &state.speed_parameter {
                self.check_float(parameter, &format!("{path}/{} speed", state.name), errors);
            }
            self.validate_motion(&state.motion, &format!("{path}/{}", state.name), errors);
        }
        if !machine.entry.is_empty() && machine.state_index(&machine.entry).is_none() {
            errors.push(format!(
                "{path}: entry state '{}' does not exist",
                machine.entry
            ));
        }
        for (transition, any) in machine
            .transitions
            .iter()
            .map(|t| (t, false))
            .chain(machine.any_state.iter().map(|t| (t, true)))
        {
            if !any && machine.state_index(&transition.from).is_none() {
                errors.push(format!(
                    "{path}: transition from unknown state '{}'",
                    transition.from
                ));
            }
            if machine.state_index(&transition.to).is_none() {
                errors.push(format!(
                    "{path}: transition to unknown state '{}'",
                    transition.to
                ));
            }
            if transition.conditions.is_empty() && transition.exit_time.is_none() {
                errors.push(format!(
                    "{path}: transition '{}' -> '{}' has neither conditions nor an exit time",
                    transition.from, transition.to
                ));
            }
            for condition in &transition.conditions {
                let parameter = condition.parameter();
                let Some(def) = self.parameter(parameter) else {
                    errors.push(format!(
                        "{path}: condition on unknown parameter '{parameter}'"
                    ));
                    continue;
                };
                let fits = matches!(
                    (condition, def.default),
                    (
                        Condition::If(_) | Condition::IfNot(_),
                        ParameterValue::Bool(_) | ParameterValue::Trigger(_)
                    ) | (
                        Condition::Greater(..) | Condition::Less(..),
                        ParameterValue::Float(_) | ParameterValue::Int(_)
                    ) | (
                        Condition::Equals(..) | Condition::NotEquals(..),
                        ParameterValue::Int(_)
                    )
                );
                if !fits {
                    errors.push(format!(
                        "{path}: condition {condition:?} does not fit parameter '{parameter}'"
                    ));
                }
            }
        }
    }

    fn validate_motion(&self, motion: &Motion, path: &str, errors: &mut Vec<String>) {
        match motion {
            Motion::Empty => {}
            Motion::Clip(clip) => {
                if clip.is_empty() {
                    errors.push(format!("{path}: clip reference is empty"));
                }
            }
            Motion::Blend1D {
                parameter,
                children,
            } => {
                self.check_float(parameter, path, errors);
                if children.is_empty() {
                    errors.push(format!("{path}: blend tree has no children"));
                }
                for (i, child) in children.iter().enumerate() {
                    self.validate_motion(&child.motion, &format!("{path}[{i}]"), errors);
                }
            }
            Motion::Blend2D {
                parameter_x,
                parameter_y,
                children,
                ..
            } => {
                self.check_float(parameter_x, path, errors);
                self.check_float(parameter_y, path, errors);
                if children.is_empty() {
                    errors.push(format!("{path}: blend tree has no children"));
                }
                for (i, child) in children.iter().enumerate() {
                    self.validate_motion(&child.motion, &format!("{path}[{i}]"), errors);
                }
            }
            Motion::StateMachine(machine) => self.validate_machine(machine, path, errors),
        }
    }

    pub fn to_ron(&self) -> String {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default().struct_names(false))
            .unwrap_or_default()
    }
}

/// Loads and validates `*.animgraph.ron`.
pub struct AnimGraphLoader;

impl AssetLoader for AnimGraphLoader {
    type Asset = AnimGraph;

    fn extensions(&self) -> &'static [&'static str] {
        &["animgraph.ron"]
    }

    fn load(&self, bytes: &[u8], ctx: &mut LoadContext<'_>) -> Result<AnimGraph> {
        let text = std::str::from_utf8(bytes).map_err(|_| ctx.error("graph is not UTF-8"))?;
        let graph: AnimGraph =
            ron::from_str(text).map_err(|error| ctx.error(format!("invalid graph: {error}")))?;
        graph
            .validate()
            .map_err(|errors| ctx.error(errors.join("; ")))?;
        Ok(graph)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) const LOCOMOTION: &str = r#"(
        parameters: [
            (name: "speed", default: Float(0.0)),
            (name: "attack", default: Trigger(false)),
            (name: "dead", default: Bool(false)),
            (name: "aim", default: Float(0.0)),
        ],
        layers: [
            (
                name: "base",
                state_machine: (
                    entry: "move",
                    states: [
                        (name: "move", motion: Blend1D(parameter: "speed", children: [
                            (threshold: 0.0, motion: Clip((path: "idle.anim.ron"))),
                            (threshold: 2.0, motion: Clip((path: "walk.anim.ron"))),
                            (threshold: 6.0, motion: Clip((path: "run.anim.ron"))),
                        ])),
                        (name: "attack", motion: Clip((path: "attack.anim.ron"))),
                        (name: "death", motion: Clip((path: "death.anim.ron"))),
                    ],
                    transitions: [
                        (from: "move", to: "attack", conditions: [If("attack")], duration: 0.1),
                        (from: "attack", to: "move", exit_time: Some(0.9), duration: 0.2),
                    ],
                    any_state: [
                        (to: "death", conditions: [If("dead")], duration: 0.2, priority: 10),
                    ],
                ),
            ),
            (
                name: "upper",
                weight_parameter: Some("aim"),
                blend: Additive,
                mask: Some((bones: ["spine"])),
                state_machine: (states: [(name: "aim", motion: Clip((path: "aim.anim.ron")))]),
            ),
        ],
    )"#;

    pub(crate) fn locomotion() -> AnimGraph {
        ron::from_str(LOCOMOTION).expect("graph parses")
    }

    #[test]
    fn sample_graph_validates_and_lists_clips() {
        let graph = locomotion();
        graph.validate().unwrap();
        assert_eq!(graph.clips().len(), 6);
        let parsed: AnimGraph = ron::from_str(&graph.to_ron()).unwrap();
        assert_eq!(parsed, graph);
    }

    #[test]
    fn validation_reports_bad_references() {
        let mut graph = locomotion();
        let machine = &mut graph.layers[0].state_machine;
        machine.transitions.push(Transition {
            from: "nowhere".into(),
            to: "move".into(),
            conditions: vec![Condition::Greater("dead".into(), 1.0)],
            exit_time: None,
            duration: 0.0,
            curve: TransitionCurve::Linear,
            interruptible: false,
            priority: 0,
            can_transition_to_self: false,
        });
        machine.entry = "missing".into();
        let errors = graph.validate().unwrap_err().join("\n");
        assert!(errors.contains("unknown state 'nowhere'"), "{errors}");
        assert!(errors.contains("does not fit parameter 'dead'"), "{errors}");
        assert!(errors.contains("entry state 'missing'"), "{errors}");
    }
}
