//! Interaction: pointer hit-testing (hover, press, drag, wheel), focus
//! and directional navigation (explicit links or spatial search), widget
//! behaviour and the events/model writes they produce.

use engine_math::Vec2;

use crate::document::{BindTarget, NodeKind};
use crate::instance::UiInstance;
use crate::model::UiValue;
use crate::style::State;

/// Pointer input for one frame (physical pixels).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PointerInput {
    pub position: Option<Vec2>,
    pub held: bool,
    pub just_pressed: bool,
    pub just_released: bool,
    /// Wheel lines (positive = up).
    pub wheel: f32,
}

/// Keyboard/gamepad navigation input for one frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NavInput {
    pub up: bool,
    pub down: bool,
    pub left: bool,
    pub right: bool,
    pub submit: bool,
    pub cancel: bool,
    pub typed: String,
    pub backspace: bool,
}

impl NavInput {
    pub fn any_direction(&self) -> bool {
        self.up || self.down || self.left || self.right
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum UiEventKind {
    Click,
    Changed(UiValue),
    Focus,
    Blur,
    HoverEnter,
    HoverLeave,
    /// Enter pressed in a text input.
    Submit(String),
    /// `ui_cancel` (Escape / B) while the document has input.
    Cancel,
}

/// An event produced by a node (resolved to ECS events by the plugin).
#[derive(Clone, Debug, PartialEq)]
pub struct NodeEvent {
    pub node: String,
    pub item: Option<usize>,
    pub kind: UiEventKind,
}

/// Interaction state of one document.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Interaction {
    pub focus: Option<usize>,
    pub hover: Option<usize>,
    pub pressed: Option<usize>,
    pub dragging: Option<usize>,
    /// Whether the pointer is over any visible UI element this frame.
    pub pointer_over: bool,
}

/// What an interaction step produced.
#[derive(Default, Debug)]
pub struct InteractionOutput {
    pub events: Vec<NodeEvent>,
    pub model_writes: Vec<(String, UiValue)>,
    /// Styles depend on changed states.
    pub restyle: bool,
}

fn is_interactive(instance: &UiInstance, index: usize) -> bool {
    let node = &instance.nodes[index];
    node.part.is_none() && node.focusable && instance.is_shown(index) && instance.is_enabled(index)
}

/// Topmost node under `point` (draw order), and topmost interactive one.
pub fn hit_test(instance: &UiInstance, point: Vec2) -> (Option<usize>, Option<usize>) {
    let mut any = None;
    let mut interactive = None;
    fn visit(
        instance: &UiInstance,
        index: usize,
        point: Vec2,
        any: &mut Option<usize>,
        interactive: &mut Option<usize>,
    ) {
        let node = &instance.nodes[index];
        if !node.visible || node.style.display == Some(crate::style::Display::None) {
            return;
        }
        let inside = node.rect.contains(point) && node.clip.contains(point);
        if inside {
            let solid = node.visual.background[3] > 0.01
                || node.text.is_some()
                || node.image.is_some()
                || node.focusable
                || matches!(node.kind, NodeKind::ScrollView { .. });
            if solid {
                *any = Some(index);
            }
        }
        for child in instance.draw_children(index) {
            visit(instance, child, point, any, interactive);
        }
        if inside && is_interactive(instance, index) {
            // Later (higher) interactive nodes win unless a descendant is
            // already the target.
            let descendant_hit = interactive.is_some_and(|hit| is_descendant(instance, hit, index));
            if !descendant_hit {
                *interactive = Some(index);
            }
        }
    }
    fn is_descendant(instance: &UiInstance, node: usize, ancestor: usize) -> bool {
        let mut current = instance.nodes[node].parent;
        while let Some(i) = current {
            if i == ancestor {
                return true;
            }
            current = instance.nodes[i].parent;
        }
        false
    }
    visit(instance, instance.root, point, &mut any, &mut interactive);
    (any, interactive)
}

fn emit(output: &mut InteractionOutput, instance: &UiInstance, index: usize, kind: UiEventKind) {
    let node = &instance.nodes[index];
    output.events.push(NodeEvent {
        node: node.id.clone(),
        item: node.item_index,
        kind,
    });
}

fn bound_key(instance: &UiInstance, index: usize, target: BindTarget) -> Option<String> {
    instance.nodes[index]
        .bind
        .iter()
        .find(|b| b.target == target)
        .map(|b| b.key.clone())
}

fn set_focus(
    instance: &mut UiInstance,
    state: &mut Interaction,
    target: Option<usize>,
    output: &mut InteractionOutput,
) {
    if state.focus == target {
        return;
    }
    if let Some(old) = state.focus {
        if old < instance.nodes.len() {
            instance.nodes[old].set_state(State::Focused, false);
            emit(output, instance, old, UiEventKind::Blur);
        }
    }
    state.focus = target;
    if let Some(new) = target {
        instance.nodes[new].set_state(State::Focused, true);
        emit(output, instance, new, UiEventKind::Focus);
        scroll_into_view(instance, new);
    }
    output.restyle = true;
}

fn scroll_into_view(instance: &mut UiInstance, index: usize) {
    let rect = instance.nodes[index].rect;
    let mut current = instance.nodes[index].parent;
    while let Some(i) = current {
        if matches!(instance.nodes[i].kind, NodeKind::ScrollView { .. }) {
            let view = instance.nodes[i].rect;
            let scroll = &mut instance.nodes[i].widget.scroll;
            if rect.y < view.y {
                scroll.y -= view.y - rect.y;
            } else if rect.y + rect.height > view.y + view.height {
                scroll.y += rect.y + rect.height - (view.y + view.height);
            }
            if rect.x < view.x {
                scroll.x -= view.x - rect.x;
            } else if rect.x + rect.width > view.x + view.width {
                scroll.x += rect.x + rect.width - (view.x + view.width);
            }
            scroll.x = scroll.x.max(0.0);
            scroll.y = scroll.y.max(0.0);
            instance.relayout = true;
        }
        current = instance.nodes[i].parent;
    }
}

fn change_value(
    instance: &mut UiInstance,
    index: usize,
    value: f32,
    output: &mut InteractionOutput,
) {
    let NodeKind::Slider { min, max, step, .. } = instance.nodes[index].kind else {
        return;
    };
    let mut value = value.clamp(min, max);
    if step > 0.0 {
        value = min + ((value - min) / step).round() * step;
        value = value.clamp(min, max);
    }
    if (instance.nodes[index].widget.value - value).abs() < 1e-6 {
        return;
    }
    instance.nodes[index].widget.value = value;
    instance.relayout = true;
    emit(
        output,
        instance,
        index,
        UiEventKind::Changed(UiValue::Number(value as f64)),
    );
    if let Some(key) = bound_key(instance, index, BindTarget::Value) {
        output
            .model_writes
            .push((key, UiValue::Number(value as f64)));
    }
}

fn slider_value_at(instance: &UiInstance, index: usize, x: f32) -> Option<f32> {
    let NodeKind::Slider { min, max, .. } = instance.nodes[index].kind else {
        return None;
    };
    let rect = instance.nodes[index].rect;
    let t = ((x - rect.x) / rect.width.max(1.0)).clamp(0.0, 1.0);
    Some(min + (max - min) * t)
}

fn activate(instance: &mut UiInstance, index: usize, output: &mut InteractionOutput) {
    match instance.nodes[index].kind.clone() {
        NodeKind::Toggle { .. } => {
            let checked = !instance.nodes[index].widget.checked;
            instance.nodes[index].widget.checked = checked;
            instance.restyle = true;
            instance.relayout = true;
            emit(
                output,
                instance,
                index,
                UiEventKind::Changed(UiValue::Bool(checked)),
            );
            if let Some(key) = bound_key(instance, index, BindTarget::Checked) {
                output.model_writes.push((key, UiValue::Bool(checked)));
            }
            emit(output, instance, index, UiEventKind::Click);
        }
        NodeKind::TextInput { .. } => {
            let text = instance.nodes[index].widget.input.clone();
            emit(output, instance, index, UiEventKind::Submit(text));
        }
        _ => emit(output, instance, index, UiEventKind::Click),
    }
}

/// Best focus candidate from `from` in `direction` (explicit links first).
fn navigate(instance: &UiInstance, from: usize, direction: Vec2) -> Option<usize> {
    let node = &instance.nodes[from];
    let link = if direction.y < 0.0 {
        &node.nav.up
    } else if direction.y > 0.0 {
        &node.nav.down
    } else if direction.x < 0.0 {
        &node.nav.left
    } else {
        &node.nav.right
    };
    if let Some(target) = link.as_ref().and_then(|id| instance.by_id.get(id)) {
        if is_interactive(instance, *target) {
            return Some(*target);
        }
    }
    let origin = node.rect.center();
    let mut best: Option<(f32, usize)> = None;
    for index in 0..instance.nodes.len() {
        if index == from || !is_interactive(instance, index) {
            continue;
        }
        let delta = instance.nodes[index].rect.center() - origin;
        let along = delta.dot(direction);
        if along <= 1.0 {
            continue;
        }
        let across = (delta - direction * along).length();
        let score = along + across * 2.0;
        if best.is_none_or(|(s, _)| score < s) {
            best = Some((score, index));
        }
    }
    best.map(|(_, index)| index)
}

fn first_focusable(instance: &UiInstance) -> Option<usize> {
    instance
        .preorder()
        .into_iter()
        .find(|i| is_interactive(instance, *i))
}

/// One frame of interaction. `accepts_input` is false for documents that
/// only display (HUDs) or are covered by a modal document.
pub fn interact(
    instance: &mut UiInstance,
    state: &mut Interaction,
    pointer: &PointerInput,
    nav: &NavInput,
    accepts_input: bool,
    scale: f32,
) -> InteractionOutput {
    let mut output = InteractionOutput::default();
    // Drop stale indices after rebuilds.
    for slot in [
        &mut state.focus,
        &mut state.hover,
        &mut state.pressed,
        &mut state.dragging,
    ] {
        if slot.is_some_and(|i| i >= instance.nodes.len()) {
            *slot = None;
        }
    }
    if state.focus.is_some_and(|f| !is_interactive(instance, f)) {
        let focus = state.focus;
        state.focus = None;
        if let Some(f) = focus {
            instance.nodes[f].set_state(State::Focused, false);
            output.restyle = true;
        }
    }

    let (any, target) = match pointer.position {
        Some(point) => hit_test(instance, point),
        None => (None, None),
    };
    state.pointer_over = any.is_some();
    if !accepts_input {
        return output;
    }

    // Hover.
    if state.hover != target {
        if let Some(old) = state.hover {
            instance.nodes[old].set_state(State::Hover, false);
            emit(&mut output, instance, old, UiEventKind::HoverLeave);
        }
        if let Some(new) = target {
            instance.nodes[new].set_state(State::Hover, true);
            emit(&mut output, instance, new, UiEventKind::HoverEnter);
        }
        state.hover = target;
        output.restyle = true;
    }

    // Press / drag / release.
    if pointer.just_pressed {
        match target {
            Some(index) => {
                state.pressed = Some(index);
                instance.nodes[index].set_state(State::Pressed, true);
                set_focus(instance, state, Some(index), &mut output);
                if let Some(value) =
                    slider_value_at(instance, index, pointer.position.map_or(0.0, |p| p.x))
                {
                    state.dragging = Some(index);
                    change_value(instance, index, value, &mut output);
                }
                output.restyle = true;
            }
            None if any.is_none() => set_focus(instance, state, None, &mut output),
            None => {}
        }
    }
    if let (Some(index), Some(point)) = (state.dragging, pointer.position) {
        if pointer.held {
            if let Some(value) = slider_value_at(instance, index, point.x) {
                change_value(instance, index, value, &mut output);
            }
        }
    }
    if pointer.just_released || (!pointer.held && state.pressed.is_some()) {
        if let Some(index) = state.pressed.take() {
            instance.nodes[index].set_state(State::Pressed, false);
            output.restyle = true;
            if Some(index) == target && state.dragging != Some(index) {
                activate(instance, index, &mut output);
            }
        }
        state.dragging = None;
    }

    // Wheel scrolling of the scroll view under the pointer.
    if pointer.wheel != 0.0 {
        let mut current = any;
        while let Some(i) = current {
            if matches!(instance.nodes[i].kind, NodeKind::ScrollView { .. }) {
                let node = &mut instance.nodes[i];
                node.widget.scroll.y =
                    (node.widget.scroll.y - pointer.wheel * 40.0 * scale).max(0.0);
                instance.relayout = true;
                break;
            }
            current = instance.nodes[i].parent;
        }
    }

    // Keyboard / gamepad navigation.
    let focused_input = state
        .focus
        .filter(|f| matches!(instance.nodes[*f].kind, NodeKind::TextInput { .. }));
    if let Some(index) = focused_input {
        let NodeKind::TextInput { max_length, .. } = instance.nodes[index].kind else {
            unreachable!()
        };
        let mut text = instance.nodes[index].widget.input.clone();
        let before = text.clone();
        if nav.backspace {
            text.pop();
        }
        for c in nav.typed.chars() {
            if text.chars().count() < max_length {
                text.push(c);
            }
        }
        if text != before {
            instance.nodes[index].widget.input = text.clone();
            instance.relayout = true;
            emit(
                &mut output,
                instance,
                index,
                UiEventKind::Changed(UiValue::Text(text.clone())),
            );
            if let Some(key) = bound_key(instance, index, BindTarget::Input) {
                output.model_writes.push((key, UiValue::Text(text)));
            }
        }
    }
    if nav.any_direction() || nav.submit {
        if state.focus.is_none() {
            let first = first_focusable(instance);
            set_focus(instance, state, first, &mut output);
        } else if let Some(focus) = state.focus {
            let horizontal = if nav.left {
                -1.0
            } else if nav.right {
                1.0
            } else {
                0.0
            };
            let slider = matches!(instance.nodes[focus].kind, NodeKind::Slider { .. });
            let text_field = focused_input.is_some();
            if slider && horizontal != 0.0 {
                if let NodeKind::Slider { min, max, step, .. } = instance.nodes[focus].kind {
                    let increment = if step > 0.0 { step } else { (max - min) * 0.05 };
                    let value = instance.nodes[focus].widget.value + horizontal * increment;
                    change_value(instance, focus, value, &mut output);
                }
            } else if nav.any_direction() && !(text_field && horizontal != 0.0) {
                let direction = if nav.up {
                    Vec2::new(0.0, -1.0)
                } else if nav.down {
                    Vec2::new(0.0, 1.0)
                } else {
                    Vec2::new(horizontal, 0.0)
                };
                if let Some(next) = navigate(instance, focus, direction) {
                    set_focus(instance, state, Some(next), &mut output);
                }
            }
            if nav.submit {
                if let Some(focus) = state.focus {
                    activate(instance, focus, &mut output);
                }
            }
        }
    }
    if nav.cancel {
        let node = state.focus.unwrap_or(instance.root);
        emit(&mut output, instance, node, UiEventKind::Cancel);
    }
    output
}
