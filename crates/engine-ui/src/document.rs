//! UI layout documents (`*.ui.ron`): a tree of nodes with stable ids,
//! widget kinds, classes, inline styles, data bindings and navigation
//! overrides. Retained and versioned; the editor edits these files.

use std::collections::HashSet;

use engine_assets::{Asset, AssetLoader, AssetRef, LoadContext};
use engine_core::Result;
use serde::{Deserialize, Serialize};

use crate::style::{Selector, Style, StyleRule};

pub const UI_LAYOUT_VERSION: u32 = 1;

/// A value used in text arguments.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ValueSource {
    Text(String),
    Number(f64),
    /// A view-model key (`hud.keys`, `item.name` inside lists).
    Bind(String),
}

/// Where a text comes from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TextSource {
    Literal(String),
    /// A localization key with arguments.
    Loc {
        key: String,
        #[serde(default)]
        args: Vec<(String, ValueSource)>,
    },
    /// A view-model value, formatted.
    Bind(String),
}

impl Default for TextSource {
    fn default() -> Self {
        Self::Literal(String::new())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum NodeKind {
    #[default]
    Panel,
    Text {
        text: TextSource,
    },
    Image {
        image: AssetRef,
        /// Nine-slice borders in source pixels: left, right, top, bottom.
        #[serde(default)]
        nine_slice: Option<[f32; 4]>,
    },
    Button {
        #[serde(default)]
        text: Option<TextSource>,
    },
    Toggle {
        #[serde(default)]
        text: Option<TextSource>,
        #[serde(default)]
        value: bool,
    },
    Slider {
        #[serde(default)]
        min: f32,
        #[serde(default = "one")]
        max: f32,
        #[serde(default)]
        step: f32,
        #[serde(default)]
        value: f32,
    },
    ProgressBar {
        #[serde(default)]
        value: f32,
    },
    ScrollView {
        #[serde(default)]
        horizontal: bool,
        #[serde(default = "yes")]
        vertical: bool,
    },
    TextInput {
        #[serde(default)]
        placeholder: TextSource,
        #[serde(default)]
        value: String,
        #[serde(default = "default_max_length")]
        max_length: usize,
    },
    /// Instantiates `template` once per item of the list at model key
    /// `source`; inside, `item.<field>` binds to the item.
    List {
        source: String,
        template: Box<UiNode>,
    },
}

fn one() -> f32 {
    1.0
}

fn yes() -> bool {
    true
}

fn default_max_length() -> usize {
    256
}

impl NodeKind {
    /// Kind name used by style selectors.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Panel => "Panel",
            Self::Text { .. } => "Text",
            Self::Image { .. } => "Image",
            Self::Button { .. } => "Button",
            Self::Toggle { .. } => "Toggle",
            Self::Slider { .. } => "Slider",
            Self::ProgressBar { .. } => "ProgressBar",
            Self::ScrollView { .. } => "ScrollView",
            Self::TextInput { .. } => "TextInput",
            Self::List { .. } => "List",
        }
    }

    /// Widgets that take focus by default.
    pub fn focusable(&self) -> bool {
        matches!(
            self,
            Self::Button { .. }
                | Self::Toggle { .. }
                | Self::Slider { .. }
                | Self::TextInput { .. }
        )
    }
}

/// Properties a binding can drive.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum BindTarget {
    /// Text of Text/Button/Toggle nodes.
    Text,
    /// Slider/progress value; two-way for sliders.
    Value,
    /// Toggle state; two-way.
    Checked,
    /// Text input content; two-way.
    Input,
    Visible,
    Enabled,
    /// Adds the class while the value is truthy.
    Class(String),
    /// Image asset path.
    Image,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Binding {
    pub target: BindTarget,
    pub key: String,
}

/// Explicit focus neighbours (node ids), overriding spatial navigation.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct NavLinks {
    #[serde(default)]
    pub up: Option<String>,
    #[serde(default)]
    pub down: Option<String>,
    #[serde(default)]
    pub left: Option<String>,
    #[serde(default)]
    pub right: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UiNode {
    /// Stable id: events, scripts, styles (`#id`) and the editor use it.
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub kind: NodeKind,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub classes: Vec<String>,
    #[serde(default, skip_serializing_if = "is_default_style")]
    pub style: Style,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<UiNode>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bind: Vec<Binding>,
    #[serde(default = "yes")]
    pub visible: bool,
    #[serde(default = "yes")]
    pub enabled: bool,
    /// Overrides the kind's default focusability.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub focusable: Option<bool>,
    #[serde(default, skip_serializing_if = "is_default_nav")]
    pub nav: NavLinks,
}

fn is_default_style(style: &Style) -> bool {
    *style == Style::default()
}

fn is_default_nav(nav: &NavLinks) -> bool {
    *nav == NavLinks::default()
}

impl Default for UiNode {
    fn default() -> Self {
        Self {
            id: String::new(),
            kind: NodeKind::Panel,
            classes: Vec::new(),
            style: Style::default(),
            children: Vec::new(),
            bind: Vec::new(),
            visible: true,
            enabled: true,
            focusable: None,
            nav: NavLinks::default(),
        }
    }
}

impl UiNode {
    pub fn new(id: impl Into<String>, kind: NodeKind) -> Self {
        Self {
            id: id.into(),
            kind,
            ..Default::default()
        }
    }

    pub fn with_children(mut self, children: Vec<UiNode>) -> Self {
        self.children = children;
        self
    }

    pub fn with_style(mut self, style: Style) -> Self {
        self.style = style;
        self
    }

    pub fn with_class(mut self, class: impl Into<String>) -> Self {
        self.classes.push(class.into());
        self
    }

    pub fn with_binding(mut self, target: BindTarget, key: impl Into<String>) -> Self {
        self.bind.push(Binding {
            target,
            key: key.into(),
        });
        self
    }

    pub fn is_focusable(&self) -> bool {
        self.focusable.unwrap_or_else(|| self.kind.focusable())
    }

    /// Depth-first visit (including list templates).
    pub fn visit<'a>(&'a self, f: &mut dyn FnMut(&'a UiNode)) {
        f(self);
        if let NodeKind::List { template, .. } = &self.kind {
            template.visit(f);
        }
        for child in &self.children {
            child.visit(f);
        }
    }

    /// Finds a node by id (outside list templates).
    pub fn find(&self, id: &str) -> Option<&UiNode> {
        if self.id == id {
            return Some(self);
        }
        self.children.iter().find_map(|child| child.find(id))
    }

    pub fn find_mut(&mut self, id: &str) -> Option<&mut UiNode> {
        if self.id == id {
            return Some(self);
        }
        self.children
            .iter_mut()
            .find_map(|child| child.find_mut(id))
    }
}

/// A UI layout asset.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UiLayout {
    #[serde(default = "layout_version")]
    pub version: u32,
    /// Style sheets (`*.uistyle.ron`), applied in order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub styles: Vec<AssetRef>,
    /// Rules local to this layout (after the sheets).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rules: Vec<StyleRule>,
    pub root: UiNode,
}

fn layout_version() -> u32 {
    UI_LAYOUT_VERSION
}

impl Asset for UiLayout {
    const TYPE_NAME: &'static str = "UiLayout";
}

impl Default for UiLayout {
    fn default() -> Self {
        Self {
            version: UI_LAYOUT_VERSION,
            styles: Vec::new(),
            rules: Vec::new(),
            root: UiNode::new("root", NodeKind::Panel),
        }
    }
}

impl UiLayout {
    pub fn validate(&self) -> std::result::Result<(), Vec<String>> {
        let mut errors = Vec::new();
        if self.version > UI_LAYOUT_VERSION {
            errors.push(format!(
                "layout version {} is newer than supported {UI_LAYOUT_VERSION}",
                self.version
            ));
        }
        for rule in &self.rules {
            if let Err(error) = Selector::parse(&rule.selector) {
                errors.push(format!("selector '{}': {error}", rule.selector));
            }
        }
        let mut ids = HashSet::new();
        let mut all_ids = HashSet::new();
        self.root.visit(&mut |node| {
            if !node.id.is_empty() {
                all_ids.insert(node.id.clone());
            }
        });
        check_ids(&self.root, &mut ids, &mut errors, false);
        self.root.visit(&mut |node| {
            for link in [
                &node.nav.up,
                &node.nav.down,
                &node.nav.left,
                &node.nav.right,
            ]
            .into_iter()
            .flatten()
            {
                if !all_ids.contains(link) {
                    errors.push(format!(
                        "node '{}' navigates to unknown id '{link}'",
                        node.id
                    ));
                }
            }
            if let NodeKind::Slider { min, max, step, .. } = &node.kind {
                if max <= min || *step < 0.0 {
                    errors.push(format!(
                        "slider '{}' needs min < max and step >= 0",
                        node.id
                    ));
                }
            }
            if let NodeKind::List { source, .. } = &node.kind {
                if source.is_empty() {
                    errors.push(format!("list '{}' needs a source key", node.id));
                }
            }
        });
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    pub fn to_ron(&self) -> String {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default().struct_names(false))
            .unwrap_or_default()
    }
}

fn check_ids(
    node: &UiNode,
    ids: &mut HashSet<String>,
    errors: &mut Vec<String>,
    in_template: bool,
) {
    if !node.id.is_empty() && !in_template && !ids.insert(node.id.clone()) {
        errors.push(format!("id '{}' is used twice", node.id));
    }
    if let NodeKind::List { template, .. } = &node.kind {
        check_ids(template, ids, errors, true);
    }
    for child in &node.children {
        check_ids(child, ids, errors, in_template);
    }
}

/// Loads and validates `*.ui.ron`.
pub struct UiLayoutLoader;

impl AssetLoader for UiLayoutLoader {
    type Asset = UiLayout;

    fn extensions(&self) -> &'static [&'static str] {
        &["ui.ron"]
    }

    fn load(&self, bytes: &[u8], ctx: &mut LoadContext<'_>) -> Result<UiLayout> {
        let text = std::str::from_utf8(bytes).map_err(|_| ctx.error("layout is not UTF-8"))?;
        let layout: UiLayout =
            ron::from_str(text).map_err(|error| ctx.error(format!("invalid layout: {error}")))?;
        layout
            .validate()
            .map_err(|errors| ctx.error(errors.join("; ")))?;
        for sheet in &layout.styles {
            ctx.add_dependency(sheet.path.clone());
        }
        Ok(layout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) const PAUSE_MENU: &str = r#"(
        rules: [
            (selector: "Button", style: (padding: Some((left: Px(12.0), right: Px(12.0), top: Px(6.0), bottom: Px(6.0))), background: Some((0.2, 0.2, 0.25, 1.0)))),
            (selector: "Button:focused", style: (background: Some((0.4, 0.4, 0.8, 1.0)))),
        ],
        root: (
            id: "pause",
            style: (direction: Some(Column), width: Some(Percent(100.0)), height: Some(Percent(100.0)), justify_content: Some(Center), align_items: Some(Center), gap: Some((8.0, 8.0))),
            children: [
                (id: "title", kind: Text(text: Loc(key: "menu-title"))),
                (id: "resume", kind: Button(text: Some(Loc(key: "menu-resume")))),
                (id: "volume", kind: Slider(min: 0.0, max: 1.0, step: 0.1, value: 0.5), bind: [(target: Value, key: "settings.volume")]),
                (id: "subtitles", kind: Toggle(text: Some(Literal("Subtitles")))),
                (id: "slots", kind: List(source: "saves", template: (kind: Button(text: Some(Bind("item.label")))))),
                (id: "quit", kind: Button(text: Some(Loc(key: "menu-quit"))), nav: (down: Some("resume"))),
            ],
        ),
    )"#;

    #[test]
    fn sample_layout_parses_validates_and_round_trips() {
        let layout: UiLayout = ron::from_str(PAUSE_MENU).unwrap();
        layout.validate().unwrap();
        assert_eq!(layout.root.children.len(), 6);
        assert!(layout.root.find("volume").unwrap().is_focusable());
        assert!(!layout.root.find("title").unwrap().is_focusable());
        let text = layout.to_ron();
        let parsed: UiLayout = ron::from_str(&text).unwrap();
        assert_eq!(parsed, layout);
    }

    #[test]
    fn validation_catches_duplicate_ids_and_bad_links() {
        let mut layout: UiLayout = ron::from_str(PAUSE_MENU).unwrap();
        layout
            .root
            .children
            .push(UiNode::new("resume", NodeKind::Panel));
        layout.root.find_mut("quit").unwrap().nav.up = Some("nowhere".into());
        layout.root.children.push(UiNode::new(
            "bad",
            NodeKind::Slider {
                min: 1.0,
                max: 0.0,
                step: 0.0,
                value: 0.0,
            },
        ));
        let errors = layout.validate().unwrap_err().join("\n");
        assert!(errors.contains("'resume' is used twice"), "{errors}");
        assert!(errors.contains("unknown id 'nowhere'"), "{errors}");
        assert!(errors.contains("slider 'bad'"), "{errors}");
    }
}
