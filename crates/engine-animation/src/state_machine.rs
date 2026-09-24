//! Runtime evaluation of animation graphs: parameters, state machine
//! stepping (transitions, crossfades, interruption, nesting) and blend
//! tree weights. Independent of the ECS so it is unit-testable and
//! reusable by the editor preview.

use std::collections::HashMap;
use std::sync::Arc;

use engine_math::Vec2;

use crate::clip::{AnimationClip, ClipBinding};
use crate::graph::{
    AnimGraph, Blend2DMode, BlendChild1D, BlendChild2D, Condition, Motion, ParameterValue,
    StateMachine, Transition, TransitionCurve,
};

/// A clip resolved for one skeleton (or none, for property-only use).
#[derive(Clone)]
pub struct ResolvedClip {
    pub clip: Arc<AnimationClip>,
    pub binding: Arc<ClipBinding>,
}

/// Looks up clips by their graph reference key.
pub trait ClipSource {
    fn clip(&self, key: &str) -> Option<&ResolvedClip>;
}

impl ClipSource for HashMap<String, ResolvedClip> {
    fn clip(&self, key: &str) -> Option<&ResolvedClip> {
        self.get(key)
    }
}

/// One clip contributing to a layer this frame.
#[derive(Clone)]
pub struct ClipInstance {
    pub clip: ResolvedClip,
    /// Playhead (seconds, wrapped/clamped) this frame and last frame.
    pub time: f32,
    pub previous_time: f32,
    pub weight: f32,
    /// Whether timeline events of this instance fire (the dominant child
    /// of a blend tree).
    pub fire_events: bool,
}

#[derive(Clone, Debug, PartialEq)]
struct StateRuntime {
    index: usize,
    /// Normalized time (1.0 = one cycle), unwrapped.
    phase: f32,
    previous_phase: f32,
    sub: Option<Box<MachineRuntime>>,
}

#[derive(Clone, Debug, PartialEq)]
struct TransitionRuntime {
    from: StateRuntime,
    elapsed: f32,
    duration: f32,
    curve: TransitionCurve,
    interruptible: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MachineRuntime {
    current: StateRuntime,
    transition: Option<TransitionRuntime>,
}

/// Snapshot of a machine for inspection (editor, debugger, save-game).
#[derive(Clone, Debug, PartialEq)]
pub struct MachineStatus {
    pub state: String,
    pub normalized_time: f32,
    pub transition_from: Option<String>,
    pub transition_progress: f32,
    pub sub: Option<Box<MachineStatus>>,
}

fn new_state(machine: &StateMachine, index: usize) -> StateRuntime {
    let sub = match machine.states.get(index).map(|s| &s.motion) {
        Some(Motion::StateMachine(inner)) => Some(Box::new(MachineRuntime::new(inner))),
        _ => None,
    };
    StateRuntime {
        index,
        phase: 0.0,
        previous_phase: 0.0,
        sub,
    }
}

impl MachineRuntime {
    pub fn new(machine: &StateMachine) -> Self {
        Self {
            current: new_state(machine, machine.entry_index()),
            transition: None,
        }
    }

    pub fn status(&self, machine: &StateMachine) -> MachineStatus {
        let name = |index: usize| {
            machine
                .states
                .get(index)
                .map(|s| s.name.clone())
                .unwrap_or_default()
        };
        let sub = match (
            &self.current.sub,
            machine.states.get(self.current.index).map(|s| &s.motion),
        ) {
            (Some(sub), Some(Motion::StateMachine(inner))) => Some(Box::new(sub.status(inner))),
            _ => None,
        };
        MachineStatus {
            state: name(self.current.index),
            normalized_time: self.current.phase,
            transition_from: self.transition.as_ref().map(|t| name(t.from.index)),
            transition_progress: self
                .transition
                .as_ref()
                .map_or(1.0, |t| (t.elapsed / t.duration.max(1e-6)).min(1.0)),
            sub,
        }
    }

    /// Current state index.
    pub fn current_state(&self) -> usize {
        self.current.index
    }

    /// Jumps to `state` immediately (or crossfading over `fade` seconds).
    pub fn play(&mut self, machine: &StateMachine, state: usize, fade: f32) {
        let next = new_state(machine, state.min(machine.states.len().saturating_sub(1)));
        let from = std::mem::replace(&mut self.current, next);
        self.transition = (fade > 0.0).then_some(TransitionRuntime {
            from,
            elapsed: 0.0,
            duration: fade,
            curve: TransitionCurve::Linear,
            interruptible: true,
        });
    }
}

/// Parameter values of one animator.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Parameters {
    values: HashMap<String, ParameterValue>,
}

impl Parameters {
    pub fn from_graph(graph: &AnimGraph, previous: &Parameters) -> Self {
        let mut values = HashMap::new();
        for def in &graph.parameters {
            let value = previous
                .values
                .get(&def.name)
                .copied()
                .filter(|v| v.same_kind(def.default))
                .or_else(|| {
                    // Convert values set before the graph loaded.
                    previous.values.get(&def.name).map(|v| match def.default {
                        ParameterValue::Float(_) => ParameterValue::Float(v.as_float()),
                        ParameterValue::Int(_) => ParameterValue::Int(v.as_float() as i32),
                        ParameterValue::Bool(_) => ParameterValue::Bool(v.as_bool()),
                        ParameterValue::Trigger(_) => ParameterValue::Trigger(v.as_bool()),
                    })
                })
                .unwrap_or(def.default);
            values.insert(def.name.clone(), value);
        }
        Self { values }
    }

    pub fn get(&self, name: &str) -> Option<ParameterValue> {
        self.values.get(name).copied()
    }

    pub fn float(&self, name: &str) -> f32 {
        self.get(name).map_or(0.0, ParameterValue::as_float)
    }

    pub fn set(&mut self, name: impl Into<String>, value: ParameterValue) {
        self.values.insert(name.into(), value);
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &ParameterValue)> {
        self.values.iter()
    }

    fn check(&self, condition: &Condition) -> bool {
        let value = |name: &str| self.values.get(name).copied();
        match condition {
            Condition::If(p) => value(p).is_some_and(ParameterValue::as_bool),
            Condition::IfNot(p) => !value(p).is_some_and(ParameterValue::as_bool),
            Condition::Greater(p, v) => value(p).is_some_and(|x| x.as_float() > *v),
            Condition::Less(p, v) => value(p).is_some_and(|x| x.as_float() < *v),
            Condition::Equals(p, v) => value(p).is_some_and(|x| x.as_float() as i32 == *v),
            Condition::NotEquals(p, v) => value(p).is_some_and(|x| x.as_float() as i32 != *v),
        }
    }

    fn consume_triggers(&mut self, conditions: &[Condition]) {
        for condition in conditions {
            if let Some(value @ ParameterValue::Trigger(true)) =
                self.values.get_mut(condition.parameter())
            {
                *value = ParameterValue::Trigger(false);
            }
        }
    }
}

/// Weights of 1D blend children for `value` (piecewise linear between
/// neighbouring thresholds, clamped at the ends).
pub fn blend_1d_weights(children: &[BlendChild1D], value: f32) -> Vec<f32> {
    let mut weights = vec![0.0; children.len()];
    if children.is_empty() {
        return weights;
    }
    let mut order: Vec<usize> = (0..children.len()).collect();
    order.sort_by(|a, b| children[*a].threshold.total_cmp(&children[*b].threshold));
    let first = order[0];
    let last = *order.last().expect("non-empty");
    if value <= children[first].threshold {
        weights[first] = 1.0;
        return weights;
    }
    if value >= children[last].threshold {
        weights[last] = 1.0;
        return weights;
    }
    for pair in order.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let (ta, tb) = (children[a].threshold, children[b].threshold);
        if value >= ta && value <= tb {
            let t = if tb > ta {
                (value - ta) / (tb - ta)
            } else {
                0.0
            };
            weights[a] = 1.0 - t;
            weights[b] = t;
            break;
        }
    }
    weights
}

fn signed_angle(a: Vec2, b: Vec2) -> f32 {
    a.perp_dot(b).atan2(a.dot(b))
}

/// Gradient-band weights of 2D blend children (Cartesian or polar).
pub fn blend_2d_weights(children: &[BlendChild2D], mode: Blend2DMode, point: Vec2) -> Vec<f32> {
    let n = children.len();
    let mut weights = vec![0.0; n];
    if n == 0 {
        return weights;
    }
    if n == 1 {
        weights[0] = 1.0;
        return weights;
    }
    const ANGULAR: f32 = 2.0;
    for i in 0..n {
        let pi = children[i].position;
        let mut weight = f32::MAX;
        for (j, child) in children.iter().enumerate() {
            if i == j {
                continue;
            }
            let pj = child.position;
            let polar = mode == Blend2DMode::FreeformDirectional
                && pi.length() > 1e-4
                && pj.length() > 1e-4;
            let h = if polar {
                let mean = (pi.length() + pj.length()) * 0.5;
                let vij = Vec2::new(
                    (pj.length() - pi.length()) / mean,
                    signed_angle(pi, pj) * ANGULAR,
                );
                let vip = Vec2::new(
                    (point.length() - pi.length()) / mean,
                    if point.length() > 1e-4 {
                        signed_angle(pi, point) * ANGULAR
                    } else {
                        0.0
                    },
                );
                let len2 = vij.length_squared();
                if len2 < 1e-8 {
                    continue;
                }
                1.0 - vip.dot(vij) / len2
            } else {
                let vij = pj - pi;
                let len2 = vij.length_squared();
                if len2 < 1e-8 {
                    continue;
                }
                1.0 - (point - pi).dot(vij) / len2
            };
            weight = weight.min(h.clamp(0.0, 1.0));
        }
        weights[i] = if weight == f32::MAX { 0.0 } else { weight };
    }
    let total: f32 = weights.iter().sum();
    if total > 1e-6 {
        for weight in &mut weights {
            *weight /= total;
        }
    } else {
        // Degenerate layout: nearest child.
        let nearest = (0..n)
            .min_by(|a, b| {
                children[*a]
                    .position
                    .distance_squared(point)
                    .total_cmp(&children[*b].position.distance_squared(point))
            })
            .expect("non-empty");
        weights[nearest] = 1.0;
    }
    weights
}

/// Seconds for one cycle of `motion` (weighted for blend trees).
fn motion_duration(motion: &Motion, params: &Parameters, clips: &dyn ClipSource) -> f32 {
    match motion {
        Motion::Empty | Motion::StateMachine(_) => 1.0,
        Motion::Clip(reference) => clips
            .clip(&reference.request_key())
            .map_or(1.0, |c| c.clip.length().max(1e-3)),
        Motion::Blend1D {
            parameter,
            children,
        } => {
            let weights = blend_1d_weights(children, params.float(parameter));
            children
                .iter()
                .zip(weights)
                .map(|(child, w)| {
                    w * motion_duration(&child.motion, params, clips) / child.speed.abs().max(1e-3)
                })
                .sum::<f32>()
                .max(1e-3)
        }
        Motion::Blend2D {
            parameter_x,
            parameter_y,
            mode,
            children,
        } => {
            let point = Vec2::new(params.float(parameter_x), params.float(parameter_y));
            let weights = blend_2d_weights(children, *mode, point);
            children
                .iter()
                .zip(weights)
                .map(|(child, w)| {
                    w * motion_duration(&child.motion, params, clips) / child.speed.abs().max(1e-3)
                })
                .sum::<f32>()
                .max(1e-3)
        }
    }
}

fn motion_loops(motion: &Motion, clips: &dyn ClipSource) -> bool {
    match motion {
        Motion::Empty | Motion::StateMachine(_) => true,
        Motion::Clip(reference) => clips
            .clip(&reference.request_key())
            .is_none_or(|c| c.clip.looping),
        Motion::Blend1D { children, .. } => children.iter().any(|c| motion_loops(&c.motion, clips)),
        Motion::Blend2D { children, .. } => children.iter().any(|c| motion_loops(&c.motion, clips)),
    }
}

fn advance_state(
    state: &mut StateRuntime,
    machine: &StateMachine,
    params: &mut Parameters,
    clips: &dyn ClipSource,
    dt: f32,
) {
    let Some(def) = machine.states.get(state.index) else {
        return;
    };
    let speed = def.speed
        * def
            .speed_parameter
            .as_ref()
            .map_or(1.0, |p| params.float(p));
    let duration = motion_duration(&def.motion, params, clips);
    state.previous_phase = state.phase;
    state.phase += dt * speed / duration;
    if let (Some(sub), Motion::StateMachine(inner)) = (&mut state.sub, &def.motion) {
        sub.step(inner, params, clips, dt * speed);
    }
}

fn exit_time_reached(state: &StateRuntime, exit: f32, looping: bool) -> bool {
    if looping {
        let exit = exit.fract();
        (state.phase - exit).floor() > (state.previous_phase - exit).floor()
    } else {
        state.phase >= exit
    }
}

impl MachineRuntime {
    /// Advances time, fires transitions and returns nothing; outputs are
    /// read with [`MachineRuntime::collect`].
    pub fn step(
        &mut self,
        machine: &StateMachine,
        params: &mut Parameters,
        clips: &dyn ClipSource,
        dt: f32,
    ) {
        advance_state(&mut self.current, machine, params, clips, dt);
        if let Some(transition) = &mut self.transition {
            advance_state(&mut transition.from, machine, params, clips, dt);
            transition.elapsed += dt;
            if transition.elapsed >= transition.duration {
                self.transition = None;
            }
        }
        if self.transition.as_ref().is_some_and(|t| !t.interruptible) {
            return;
        }

        let current = self.current.index;
        let looping = machine
            .states
            .get(current)
            .is_some_and(|s| motion_loops(&s.motion, clips));
        let mut candidates: Vec<&Transition> = machine
            .any_state
            .iter()
            .filter(|t| t.can_transition_to_self || machine.state_index(&t.to) != Some(current))
            .chain(
                machine
                    .transitions
                    .iter()
                    .filter(|t| machine.state_index(&t.from) == Some(current)),
            )
            .collect();
        candidates.sort_by_key(|t| std::cmp::Reverse(t.priority));
        let fired = candidates.into_iter().find(|t| {
            t.conditions.iter().all(|c| params.check(c))
                && t.exit_time
                    .is_none_or(|exit| exit_time_reached(&self.current, exit, looping))
        });
        let Some(transition) = fired else {
            return;
        };
        let Some(to) = machine.state_index(&transition.to) else {
            return;
        };
        params.consume_triggers(&transition.conditions);
        let next = new_state(machine, to);
        let from = std::mem::replace(&mut self.current, next);
        self.transition = (transition.duration > 0.0).then_some(TransitionRuntime {
            from,
            elapsed: 0.0,
            duration: transition.duration,
            curve: transition.curve,
            interruptible: transition.interruptible,
        });
    }

    /// Appends this machine's weighted clip instances.
    pub fn collect(
        &self,
        machine: &StateMachine,
        params: &Parameters,
        clips: &dyn ClipSource,
        weight: f32,
        out: &mut Vec<ClipInstance>,
    ) {
        match &self.transition {
            Some(transition) => {
                let blend = transition
                    .curve
                    .apply(transition.elapsed / transition.duration.max(1e-6));
                collect_state(
                    &transition.from,
                    machine,
                    params,
                    clips,
                    weight * (1.0 - blend),
                    out,
                );
                collect_state(&self.current, machine, params, clips, weight * blend, out);
            }
            None => collect_state(&self.current, machine, params, clips, weight, out),
        }
    }
}

fn collect_state(
    state: &StateRuntime,
    machine: &StateMachine,
    params: &Parameters,
    clips: &dyn ClipSource,
    weight: f32,
    out: &mut Vec<ClipInstance>,
) {
    if weight <= 1e-4 {
        return;
    }
    let Some(def) = machine.states.get(state.index) else {
        return;
    };
    if let (Some(sub), Motion::StateMachine(inner)) = (&state.sub, &def.motion) {
        sub.collect(inner, params, clips, weight, out);
        return;
    }
    collect_motion(
        &def.motion,
        state.phase,
        state.previous_phase,
        params,
        clips,
        weight,
        true,
        out,
    );
}

#[allow(clippy::too_many_arguments)]
fn collect_motion(
    motion: &Motion,
    phase: f32,
    previous_phase: f32,
    params: &Parameters,
    clips: &dyn ClipSource,
    weight: f32,
    fire_events: bool,
    out: &mut Vec<ClipInstance>,
) {
    if weight <= 1e-4 {
        return;
    }
    match motion {
        Motion::Empty | Motion::StateMachine(_) => {}
        Motion::Clip(reference) => {
            let Some(resolved) = clips.clip(&reference.request_key()) else {
                return;
            };
            let length = resolved.clip.length();
            out.push(ClipInstance {
                clip: resolved.clone(),
                time: resolved.clip.wrap_time(phase * length),
                previous_time: resolved.clip.wrap_time(previous_phase * length),
                weight,
                fire_events,
            });
        }
        Motion::Blend1D {
            parameter,
            children,
        } => {
            let weights = blend_1d_weights(children, params.float(parameter));
            let dominant = dominant(&weights);
            for (i, (child, w)) in children.iter().zip(weights).enumerate() {
                collect_motion(
                    &child.motion,
                    phase,
                    previous_phase,
                    params,
                    clips,
                    weight * w,
                    fire_events && Some(i) == dominant,
                    out,
                );
            }
        }
        Motion::Blend2D {
            parameter_x,
            parameter_y,
            mode,
            children,
        } => {
            let point = Vec2::new(params.float(parameter_x), params.float(parameter_y));
            let weights = blend_2d_weights(children, *mode, point);
            let dominant = dominant(&weights);
            for (i, (child, w)) in children.iter().zip(weights).enumerate() {
                collect_motion(
                    &child.motion,
                    phase,
                    previous_phase,
                    params,
                    clips,
                    weight * w,
                    fire_events && Some(i) == dominant,
                    out,
                );
            }
        }
    }
}

fn dominant(weights: &[f32]) -> Option<usize> {
    weights
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::AnimationClip;
    use crate::graph::tests::locomotion;
    use engine_assets::AssetRef;

    fn clips(lengths: &[(&str, f32, bool)]) -> HashMap<String, ResolvedClip> {
        lengths
            .iter()
            .map(|(path, length, looping)| {
                (
                    AssetRef::from_path(*path).request_key(),
                    ResolvedClip {
                        clip: Arc::new(AnimationClip {
                            name: path.to_string(),
                            duration: *length,
                            looping: *looping,
                            ..Default::default()
                        }),
                        binding: Arc::new(ClipBinding::default()),
                    },
                )
            })
            .collect()
    }

    fn library() -> HashMap<String, ResolvedClip> {
        clips(&[
            ("idle.anim.ron", 2.0, true),
            ("walk.anim.ron", 1.0, true),
            ("run.anim.ron", 0.5, true),
            ("attack.anim.ron", 1.0, false),
            ("death.anim.ron", 1.0, false),
            ("aim.anim.ron", 1.0, true),
        ])
    }

    fn weights(out: &[ClipInstance]) -> Vec<(String, f32)> {
        out.iter()
            .map(|i| (i.clip.clip.name.clone(), (i.weight * 100.0).round() / 100.0))
            .collect()
    }

    #[test]
    fn blend_1d_interpolates_neighbours() {
        let graph = locomotion();
        let machine = &graph.layers[0].state_machine;
        let clips = library();
        let mut params = Parameters::from_graph(&graph, &Parameters::default());
        params.set("speed", ParameterValue::Float(4.0));
        let mut runtime = MachineRuntime::new(machine);
        runtime.step(machine, &mut params, &clips, 0.1);
        let mut out = Vec::new();
        runtime.collect(machine, &params, &clips, 1.0, &mut out);
        assert_eq!(
            weights(&out),
            vec![("walk.anim.ron".into(), 0.5), ("run.anim.ron".into(), 0.5)]
        );
        // Synced phase: blended cycle is 0.75s; 0.1s → phase 0.1333.
        assert!((out[0].time - 0.1333).abs() < 1e-3, "{}", out[0].time);
        assert!((out[1].time - 0.0667).abs() < 1e-3, "{}", out[1].time);
        assert_eq!(out.iter().filter(|i| i.fire_events).count(), 1);
    }

    #[test]
    fn trigger_transition_crossfades_and_exit_time_returns() {
        let graph = locomotion();
        let machine = &graph.layers[0].state_machine;
        let clips = library();
        let mut params = Parameters::from_graph(&graph, &Parameters::default());
        let mut runtime = MachineRuntime::new(machine);
        runtime.step(machine, &mut params, &clips, 0.1);
        params.set("attack", ParameterValue::Trigger(true));
        runtime.step(machine, &mut params, &clips, 0.05);
        assert_eq!(runtime.status(machine).state, "attack");
        assert_eq!(
            params.get("attack"),
            Some(ParameterValue::Trigger(false)),
            "consumed"
        );
        runtime.step(machine, &mut params, &clips, 0.05);
        let mut out = Vec::new();
        runtime.collect(machine, &params, &clips, 1.0, &mut out);
        assert_eq!(
            weights(&out),
            vec![
                ("idle.anim.ron".into(), 0.5),
                ("attack.anim.ron".into(), 0.5)
            ]
        );

        // Attack is 1s, exit at 0.9.
        for _ in 0..16 {
            runtime.step(machine, &mut params, &clips, 0.05);
        }
        assert_eq!(runtime.status(machine).state, "attack");
        runtime.step(machine, &mut params, &clips, 0.06);
        let status = runtime.status(machine);
        assert_eq!(status.state, "move");
        assert_eq!(status.transition_from.as_deref(), Some("attack"));
    }

    #[test]
    fn any_state_has_priority_and_skips_self() {
        let graph = locomotion();
        let machine = &graph.layers[0].state_machine;
        let clips = library();
        let mut params = Parameters::from_graph(&graph, &Parameters::default());
        let mut runtime = MachineRuntime::new(machine);
        params.set("dead", ParameterValue::Bool(true));
        params.set("attack", ParameterValue::Trigger(true));
        runtime.step(machine, &mut params, &clips, 0.1);
        assert_eq!(runtime.status(machine).state, "death");
        assert_eq!(
            params.get("attack"),
            Some(ParameterValue::Trigger(true)),
            "not consumed"
        );
        runtime.step(machine, &mut params, &clips, 0.5);
        runtime.step(machine, &mut params, &clips, 0.5);
        let status = runtime.status(machine);
        assert_eq!(status.state, "death");
        assert!(status.transition_from.is_none(), "no re-entry into itself");
    }

    #[test]
    fn blend_2d_directional_weights() {
        let child = |x: f32, y: f32| BlendChild2D {
            position: Vec2::new(x, y),
            motion: Motion::Empty,
            speed: 1.0,
        };
        let children = vec![
            child(0.0, 0.0),
            child(0.0, 1.0),
            child(1.0, 0.0),
            child(0.0, -1.0),
            child(-1.0, 0.0),
        ];
        for mode in [
            Blend2DMode::FreeformDirectional,
            Blend2DMode::FreeformCartesian,
        ] {
            let w = blend_2d_weights(&children, mode, Vec2::new(0.0, 1.0));
            assert!((w[1] - 1.0).abs() < 1e-4, "{mode:?} {w:?}");
            let w = blend_2d_weights(&children, mode, Vec2::ZERO);
            assert!((w[0] - 1.0).abs() < 1e-4, "{mode:?} {w:?}");
            let w = blend_2d_weights(&children, mode, Vec2::new(0.5, 0.5));
            assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-4);
            assert!((w[1] - w[2]).abs() < 1e-3, "symmetric: {w:?}");
            assert!(w[3] < 1e-4 && w[4] < 1e-4);
        }
    }

    #[test]
    fn parameters_convert_values_set_before_the_graph_loaded() {
        let graph = locomotion();
        let mut early = Parameters::default();
        early.set("speed", ParameterValue::Int(3));
        early.set("unknown", ParameterValue::Bool(true));
        let params = Parameters::from_graph(&graph, &early);
        assert_eq!(params.get("speed"), Some(ParameterValue::Float(3.0)));
        assert_eq!(params.get("dead"), Some(ParameterValue::Bool(false)));
        assert!(params.get("unknown").is_none());
    }
}
