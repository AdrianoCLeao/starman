//! A live UI document: the runtime node tree built from a [`UiLayout`]
//! (widget parts generated, lists instantiated from the view model),
//! resolved styles, bound values, text blocks and widget state.

use std::collections::{BTreeMap, HashMap};

use engine_localization::{LocArg, Localization};
use engine_math::Vec2;

use crate::document::{
    BindTarget, Binding, NavLinks, NodeKind, TextSource, UiLayout, UiNode, ValueSource,
};
use crate::model::{lookup, UiModel, UiValue};
use crate::style::{Color, CompiledRules, State, Style, TextAlign};
use crate::text::{TextBlock, TextParams, UiFonts, DEFAULT_FAMILY};

/// Parts generated for built-in widgets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    Label,
    ToggleBox,
    ToggleCheck,
    SliderFill,
    ProgressFill,
    InputText,
}

/// Interactive state that survives rebuilds (keyed by node id).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WidgetState {
    pub value: f32,
    pub checked: bool,
    pub input: String,
    pub scroll: Vec2,
}

/// Visual properties animated by style transitions.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Visual {
    pub background: Color,
    pub border_color: Color,
    pub text_color: Color,
    pub opacity: f32,
}

impl Visual {
    fn from_style(style: &Style) -> Self {
        Self {
            background: style.background.unwrap_or([0.0; 4]),
            border_color: style.border_color.unwrap_or([0.0; 4]),
            text_color: style.text_color.unwrap_or([1.0; 4]),
            opacity: style.opacity.unwrap_or(1.0).clamp(0.0, 1.0),
        }
    }

    fn approach(&mut self, target: &Visual, t: f32) {
        let mix = |a: &mut Color, b: &Color| {
            for i in 0..4 {
                a[i] += (b[i] - a[i]) * t;
            }
        };
        mix(&mut self.background, &target.background);
        mix(&mut self.border_color, &target.border_color);
        mix(&mut self.text_color, &target.text_color);
        self.opacity += (target.opacity - self.opacity) * t;
    }
}

/// Screen rectangle in physical pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Rect {
    pub fn contains(&self, point: Vec2) -> bool {
        point.x >= self.x
            && point.y >= self.y
            && point.x < self.x + self.width
            && point.y < self.y + self.height
    }

    pub fn intersect(&self, other: &Rect) -> Rect {
        let x0 = self.x.max(other.x);
        let y0 = self.y.max(other.y);
        let x1 = (self.x + self.width).min(other.x + other.width);
        let y1 = (self.y + self.height).min(other.y + other.height);
        Rect {
            x: x0,
            y: y0,
            width: (x1 - x0).max(0.0),
            height: (y1 - y0).max(0.0),
        }
    }

    pub fn center(&self) -> Vec2 {
        Vec2::new(self.x + self.width * 0.5, self.y + self.height * 0.5)
    }

    pub const INFINITE: Rect = Rect {
        x: -1e9,
        y: -1e9,
        width: 2e9,
        height: 2e9,
    };
}

pub struct RuntimeNode {
    pub id: String,
    pub kind: NodeKind,
    pub part: Option<Part>,
    pub classes: Vec<String>,
    pub bound_classes: Vec<String>,
    pub inline: Style,
    pub bind: Vec<Binding>,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub visible: bool,
    pub enabled: bool,
    pub focusable: bool,
    pub nav: NavLinks,
    /// List item scope (index into [`UiInstance::items`]).
    pub item: Option<usize>,
    /// Index within its list (for events).
    pub item_index: Option<usize>,
    pub text_source: Option<TextSource>,
    pub text: Option<usize>,
    pub style: Style,
    pub states: u8,
    pub visual: Visual,
    pub target_visual: Visual,
    pub widget: WidgetState,
    pub image: Option<engine_assets::AssetRef>,
    pub rect: Rect,
    /// `rect` minus border and padding.
    pub content: Rect,
    pub clip: Rect,
    pub content_size: Vec2,
    pub(crate) taffy: Option<taffy::NodeId>,
}

impl RuntimeNode {
    fn new(id: String, kind: NodeKind) -> Self {
        Self {
            id,
            kind,
            part: None,
            classes: Vec::new(),
            bound_classes: Vec::new(),
            inline: Style::default(),
            bind: Vec::new(),
            parent: None,
            children: Vec::new(),
            visible: true,
            enabled: true,
            focusable: false,
            nav: NavLinks::default(),
            item: None,
            item_index: None,
            text_source: None,
            text: None,
            style: Style::default(),
            states: 0,
            visual: Visual::default(),
            target_visual: Visual::default(),
            widget: WidgetState::default(),
            image: None,
            rect: Rect::default(),
            content: Rect::default(),
            clip: Rect::INFINITE,
            content_size: Vec2::ZERO,
            taffy: None,
        }
    }

    /// Kind name for selectors (parts style as panels/texts).
    pub fn kind_name(&self) -> &'static str {
        match self.part {
            Some(Part::Label) | Some(Part::InputText) => "Text",
            Some(_) => "Panel",
            None => self.kind.name(),
        }
    }

    pub fn has_state(&self, state: State) -> bool {
        self.states & state.bit() != 0
    }

    pub fn set_state(&mut self, state: State, on: bool) -> bool {
        let before = self.states;
        if on {
            self.states |= state.bit();
        } else {
            self.states &= !state.bit();
        }
        before != self.states
    }
}

/// A built document.
pub struct UiInstance {
    pub nodes: Vec<RuntimeNode>,
    pub root: usize,
    pub by_id: HashMap<String, usize>,
    pub items: Vec<BTreeMap<String, UiValue>>,
    pub texts: Vec<TextBlock>,
    pub rules: CompiledRules,
    /// List contents the tree was built from (rebuild when they change).
    pub list_sources: Vec<(String, Vec<BTreeMap<String, UiValue>>)>,
    pub restyle: bool,
    pub relayout: bool,
}

fn list_items(model: &UiModel, key: &str) -> Vec<BTreeMap<String, UiValue>> {
    match model.get(key) {
        Some(UiValue::List(items)) => items.clone(),
        _ => Vec::new(),
    }
}

impl UiInstance {
    /// Builds the runtime tree. `preserved` carries widget state by id
    /// across rebuilds.
    pub fn build(
        layout: &UiLayout,
        rules: CompiledRules,
        model: &UiModel,
        preserved: &HashMap<String, WidgetState>,
    ) -> Self {
        let mut instance = Self {
            nodes: Vec::new(),
            root: 0,
            by_id: HashMap::new(),
            items: Vec::new(),
            texts: Vec::new(),
            rules,
            list_sources: Vec::new(),
            restyle: true,
            relayout: true,
        };
        instance.root = instance.add(&layout.root, None, None, None, "", model, preserved);
        instance
    }

    /// Whether model lists changed since the build.
    pub fn lists_changed(&self, model: &UiModel) -> bool {
        self.list_sources
            .iter()
            .any(|(key, items)| list_items(model, key) != *items)
    }

    pub fn widget_states(&self) -> HashMap<String, WidgetState> {
        self.nodes
            .iter()
            .filter(|n| n.part.is_none() && !n.id.is_empty())
            .map(|n| (n.id.clone(), n.widget.clone()))
            .collect()
    }

    fn push(&mut self, mut node: RuntimeNode, parent: Option<usize>) -> usize {
        node.parent = parent;
        let index = self.nodes.len();
        if !node.id.is_empty() {
            self.by_id.entry(node.id.clone()).or_insert(index);
        }
        self.nodes.push(node);
        if let Some(parent) = parent {
            self.nodes[parent].children.push(index);
        }
        index
    }

    fn part(&mut self, parent: usize, part: Part, class: &str, text: Option<TextSource>) -> usize {
        let id = format!("{}.{class}", self.nodes[parent].id);
        let kind = if text.is_some() {
            NodeKind::Text {
                text: text.clone().unwrap_or_default(),
            }
        } else {
            NodeKind::Panel
        };
        let mut node = RuntimeNode::new(id, kind);
        node.part = Some(part);
        node.classes.push(class.to_owned());
        node.item = self.nodes[parent].item;
        node.item_index = self.nodes[parent].item_index;
        node.text_source = text;
        self.push(node, Some(parent))
    }

    #[allow(clippy::too_many_arguments)]
    fn add(
        &mut self,
        def: &UiNode,
        parent: Option<usize>,
        item: Option<usize>,
        item_index: Option<usize>,
        prefix: &str,
        model: &UiModel,
        preserved: &HashMap<String, WidgetState>,
    ) -> usize {
        let id = if prefix.is_empty() || def.id.is_empty() {
            if prefix.is_empty() {
                def.id.clone()
            } else {
                prefix.to_owned()
            }
        } else {
            format!("{prefix}.{}", def.id)
        };
        let mut node = RuntimeNode::new(id.clone(), def.kind.clone());
        node.classes = def.classes.clone();
        node.inline = def.style.clone();
        node.bind = def.bind.clone();
        node.visible = def.visible;
        node.enabled = def.enabled;
        node.focusable = def.is_focusable();
        node.nav = def.nav.clone();
        node.item = item;
        node.item_index = item_index;
        node.widget = match &def.kind {
            NodeKind::Slider { value, .. } | NodeKind::ProgressBar { value } => WidgetState {
                value: *value,
                ..Default::default()
            },
            NodeKind::Toggle { value, .. } => WidgetState {
                checked: *value,
                ..Default::default()
            },
            NodeKind::TextInput { value, .. } => WidgetState {
                input: value.clone(),
                ..Default::default()
            },
            _ => WidgetState::default(),
        };
        if let Some(saved) = preserved.get(&id) {
            node.widget = saved.clone();
        }
        if let NodeKind::Text { text } = &def.kind {
            node.text_source = Some(text.clone());
        }
        if let NodeKind::Image { image, .. } = &def.kind {
            node.image = Some(image.clone());
        }
        let index = self.push(node, parent);

        match &def.kind {
            NodeKind::Button { text: Some(text) } => {
                self.part(index, Part::Label, "label", Some(text.clone()));
            }
            NodeKind::Toggle { text, .. } => {
                let checkbox = self.part(index, Part::ToggleBox, "toggle-box", None);
                self.part(checkbox, Part::ToggleCheck, "toggle-check", None);
                if let Some(text) = text {
                    self.part(index, Part::Label, "label", Some(text.clone()));
                }
            }
            NodeKind::Slider { .. } => {
                self.part(index, Part::SliderFill, "slider-fill", None);
            }
            NodeKind::ProgressBar { .. } => {
                self.part(index, Part::ProgressFill, "progress-fill", None);
            }
            NodeKind::TextInput { .. } => {
                self.part(
                    index,
                    Part::InputText,
                    "input-text",
                    Some(TextSource::default()),
                );
            }
            NodeKind::List { source, template } => {
                let items = list_items(model, source);
                for (i, item_values) in items.iter().enumerate() {
                    let item_slot = self.items.len();
                    self.items.push(item_values.clone());
                    let item_prefix = format!("{id}[{i}]");
                    self.add(
                        template,
                        Some(index),
                        Some(item_slot),
                        Some(i),
                        &item_prefix,
                        model,
                        preserved,
                    );
                }
                self.list_sources.push((source.clone(), items));
            }
            _ => {}
        }
        for child in &def.children {
            self.add(
                child,
                Some(index),
                item,
                item_index,
                prefix,
                model,
                preserved,
            );
        }
        index
    }

    fn value<'a>(&'a self, model: &'a UiModel, node: usize, key: &str) -> Option<&'a UiValue> {
        let item = self.nodes[node].item.and_then(|i| self.items.get(i));
        lookup(model, item, key)
    }

    fn resolve_text(
        &self,
        source: &TextSource,
        node: usize,
        model: &UiModel,
        l10n: Option<&Localization>,
    ) -> String {
        match source {
            TextSource::Literal(text) => text.clone(),
            TextSource::Bind(key) => self
                .value(model, node, key)
                .map(UiValue::to_text)
                .unwrap_or_default(),
            TextSource::Loc { key, args } => {
                let args: Vec<(&str, LocArg)> = args
                    .iter()
                    .map(|(name, value)| {
                        let value = match value {
                            ValueSource::Text(text) => LocArg::Text(text.clone()),
                            ValueSource::Number(number) => LocArg::Number(*number),
                            ValueSource::Bind(key) => match self.value(model, node, key) {
                                Some(UiValue::Number(number)) => LocArg::Number(*number),
                                Some(other) => LocArg::Text(other.to_text()),
                                None => LocArg::Text(String::new()),
                            },
                        };
                        (name.as_str(), value)
                    })
                    .collect();
                match l10n {
                    Some(l10n) => l10n.tr_args(key, &args),
                    None => key.clone(),
                }
            }
        }
    }

    /// Applies model bindings and localized texts. Returns whether layout
    /// may have changed.
    pub fn apply_bindings(&mut self, model: &UiModel) -> bool {
        let mut changed = false;
        for index in 0..self.nodes.len() {
            let bindings = self.nodes[index].bind.clone();
            for binding in &bindings {
                let value = self.value(model, index, &binding.key).cloned();
                let Some(value) = value else {
                    continue;
                };
                if binding.target == BindTarget::Text {
                    // Buttons/toggles show their label part.
                    let label = self.nodes[index]
                        .children
                        .iter()
                        .copied()
                        .find(|c| self.nodes[*c].part == Some(Part::Label));
                    let target = label.unwrap_or(index);
                    self.nodes[target].text_source = Some(TextSource::Literal(value.to_text()));
                    continue;
                }
                let node = &mut self.nodes[index];
                match &binding.target {
                    BindTarget::Text => {}
                    BindTarget::Value => {
                        let v = value.as_number() as f32;
                        if node.widget.value != v {
                            node.widget.value = v;
                            changed = true;
                        }
                    }
                    BindTarget::Checked => {
                        node.widget.checked = value.as_bool();
                    }
                    BindTarget::Input => {
                        let text = value.to_text();
                        if node.widget.input != text {
                            node.widget.input = text;
                        }
                    }
                    BindTarget::Visible => {
                        if node.visible != value.as_bool() {
                            node.visible = value.as_bool();
                            changed = true;
                        }
                    }
                    BindTarget::Enabled => {
                        node.enabled = value.as_bool();
                    }
                    BindTarget::Class(class) => {
                        let has = node.bound_classes.contains(class);
                        if value.as_bool() && !has {
                            node.bound_classes.push(class.clone());
                            self.restyle = true;
                        } else if !value.as_bool() && has {
                            node.bound_classes.retain(|c| c != class);
                            self.restyle = true;
                        }
                    }
                    BindTarget::Image => {
                        let path = value.to_text();
                        let reference = engine_assets::AssetRef::from_path(path);
                        if node.image.as_ref() != Some(&reference) {
                            node.image = Some(reference);
                        }
                    }
                }
            }
        }
        // Widget-driven part state.
        for index in 0..self.nodes.len() {
            let Some(part) = self.nodes[index].part else {
                continue;
            };
            let Some(parent) = self.nodes[index].parent else {
                continue;
            };
            let owner = match part {
                Part::ToggleCheck => self.nodes[parent].parent.unwrap_or(parent),
                _ => parent,
            };
            let widget = self.nodes[owner].widget.clone();
            let kind = self.nodes[owner].kind.clone();
            let node = &mut self.nodes[index];
            match part {
                Part::ToggleCheck => {
                    if node.visible != widget.checked {
                        node.visible = widget.checked;
                        changed = true;
                    }
                }
                Part::SliderFill | Part::ProgressFill => {
                    let fraction = match kind {
                        NodeKind::Slider { min, max, .. } => {
                            ((widget.value - min) / (max - min).max(1e-6)).clamp(0.0, 1.0)
                        }
                        _ => widget.value.clamp(0.0, 1.0),
                    };
                    let width = Some(crate::style::Val::Percent(fraction * 100.0));
                    if node.inline.width != width {
                        node.inline.width = width;
                        changed = true;
                    }
                }
                Part::InputText => {
                    let (text, placeholder) = if widget.input.is_empty() {
                        match &kind {
                            NodeKind::TextInput { placeholder, .. } => (placeholder.clone(), true),
                            _ => (TextSource::default(), true),
                        }
                    } else {
                        (TextSource::Literal(widget.input.clone()), false)
                    };
                    node.text_source = Some(text);
                    let has = node.bound_classes.iter().any(|c| c == "placeholder");
                    if placeholder != has {
                        if placeholder {
                            node.bound_classes.push("placeholder".to_owned());
                        } else {
                            node.bound_classes.retain(|c| c != "placeholder");
                        }
                        self.restyle = true;
                    }
                }
                Part::Label | Part::ToggleBox => {}
            }
        }
        // Checked state for selectors.
        for node in &mut self.nodes {
            if matches!(node.kind, NodeKind::Toggle { .. })
                && node.part.is_none()
                && node.set_state(State::Checked, node.widget.checked)
            {
                self.restyle = true;
            }
            if node.set_state(State::Disabled, !node.enabled) {
                self.restyle = true;
            }
        }
        changed
    }

    /// Resolves styles through the cascade (with text inheritance).
    pub fn resolve_styles(&mut self) {
        let order = self.preorder();
        for index in order {
            let node = &self.nodes[index];
            let mut classes = node.classes.clone();
            classes.extend(node.bound_classes.iter().cloned());
            let mut style = self.rules.resolve(
                node.kind_name(),
                &node.id,
                &classes,
                node.states,
                &node.inline,
            );
            if let Some(parent) = node.parent {
                let parent_style = self.nodes[parent].style.clone();
                style.inherit_text(&parent_style);
            }
            let target = Visual::from_style(&style);
            let node = &mut self.nodes[index];
            if node.style.transition.unwrap_or(0.0) <= 0.0 && style.transition.unwrap_or(0.0) <= 0.0
                || node.target_visual == Visual::default()
            {
                node.visual = target;
            }
            node.target_visual = target;
            node.style = style;
        }
        self.restyle = false;
    }

    /// Advances style transitions.
    pub fn animate(&mut self, dt: f32) {
        for node in &mut self.nodes {
            if node.visual == node.target_visual {
                continue;
            }
            let duration = node.style.transition.unwrap_or(0.0);
            let t = if duration <= 0.0 {
                1.0
            } else {
                (dt / duration).clamp(0.0, 1.0)
            };
            let target = node.target_visual;
            node.visual.approach(&target, t);
            if t >= 1.0 {
                node.visual = target;
            }
        }
    }

    /// Updates text blocks from sources and resolved styles (physical
    /// pixel scale). Returns whether any text changed.
    pub fn update_texts(
        &mut self,
        fonts: &mut UiFonts,
        model: &UiModel,
        l10n: Option<&Localization>,
        scale: f32,
    ) -> bool {
        let mut changed = false;
        for index in 0..self.nodes.len() {
            let Some(source) = self.nodes[index].text_source.clone() else {
                continue;
            };
            let text = self.resolve_text(&source, index, model, l10n);
            let style = &self.nodes[index].style;
            let font_size = style.font_size.unwrap_or(20.0) * scale;
            let params = TextParams {
                text,
                font_size,
                line_height: font_size * style.line_height.unwrap_or(1.25),
                family: style
                    .font
                    .clone()
                    .unwrap_or_else(|| DEFAULT_FAMILY.to_owned()),
                bold: style.bold.unwrap_or(false),
                align: style.text_align.unwrap_or(TextAlign::Left),
            };
            match self.nodes[index].text {
                Some(slot) => {
                    if self.texts[slot].update(fonts, params) {
                        changed = true;
                    }
                }
                None => {
                    self.texts.push(TextBlock::new(fonts, params));
                    self.nodes[index].text = Some(self.texts.len() - 1);
                    changed = true;
                }
            }
        }
        changed
    }

    /// Nodes parent-first.
    pub fn preorder(&self) -> Vec<usize> {
        let mut order = Vec::with_capacity(self.nodes.len());
        let mut stack = vec![self.root];
        while let Some(index) = stack.pop() {
            order.push(index);
            for child in self.nodes[index].children.iter().rev() {
                stack.push(*child);
            }
        }
        order
    }

    /// Children in draw order (stable by z-index).
    pub fn draw_children(&self, index: usize) -> Vec<usize> {
        let mut children = self.nodes[index].children.clone();
        children.sort_by_key(|c| self.nodes[*c].style.z_index.unwrap_or(0));
        children
    }

    /// Whether the node and all its ancestors are visible.
    pub fn is_shown(&self, index: usize) -> bool {
        let mut current = Some(index);
        while let Some(i) = current {
            let node = &self.nodes[i];
            if !node.visible || node.style.display == Some(crate::style::Display::None) {
                return false;
            }
            current = node.parent;
        }
        true
    }

    /// Whether the node and all its ancestors are enabled.
    pub fn is_enabled(&self, index: usize) -> bool {
        let mut current = Some(index);
        while let Some(i) = current {
            if !self.nodes[i].enabled {
                return false;
            }
            current = self.nodes[i].parent;
        }
        true
    }

    pub fn node(&self, id: &str) -> Option<&RuntimeNode> {
        self.by_id.get(id).map(|i| &self.nodes[*i])
    }

    pub fn node_mut(&mut self, id: &str) -> Option<&mut RuntimeNode> {
        let index = *self.by_id.get(id)?;
        Some(&mut self.nodes[index])
    }

    /// Resolved text of a node (after [`Self::update_texts`]).
    pub fn text_of(&self, id: &str) -> Option<&str> {
        let node = self.node(id)?;
        let slot = node.text.or_else(|| {
            node.children
                .iter()
                .find(|c| self.nodes[**c].part == Some(Part::Label))
                .and_then(|c| self.nodes[*c].text)
        })?;
        Some(&self.texts[slot].params.text)
    }
}
