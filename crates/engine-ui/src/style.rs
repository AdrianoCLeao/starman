//! Styles: layout (flexbox/grid via taffy) and visual properties, style
//! sheets (`*.uistyle.ron`) with kind/class/id/state selectors, and the
//! cascade that resolves a node's final style.

use engine_assets::{Asset, AssetLoader, LoadContext};
use engine_core::Result;
use serde::{Deserialize, Serialize};

/// Linear-light RGBA in the sRGB color space as authored (0..1); the UI
/// renderer converts to the output encoding.
pub type Color = [f32; 4];

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum Val {
    #[default]
    Auto,
    Px(f32),
    Percent(f32),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Edges {
    #[serde(default)]
    pub left: Val,
    #[serde(default)]
    pub right: Val,
    #[serde(default)]
    pub top: Val,
    #[serde(default)]
    pub bottom: Val,
}

impl Edges {
    pub fn all(value: Val) -> Self {
        Self {
            left: value,
            right: value,
            top: value,
            bottom: value,
        }
    }

    pub fn px(value: f32) -> Self {
        Self::all(Val::Px(value))
    }

    pub fn symmetric(horizontal: Val, vertical: Val) -> Self {
        Self {
            left: horizontal,
            right: horizontal,
            top: vertical,
            bottom: vertical,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Display {
    #[default]
    Flex,
    Grid,
    None,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum PositionType {
    #[default]
    Relative,
    Absolute,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlexDirection {
    #[default]
    Row,
    Column,
    RowReverse,
    ColumnReverse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Justify {
    Start,
    End,
    Center,
    SpaceBetween,
    SpaceAround,
    SpaceEvenly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Align {
    Start,
    End,
    Center,
    Stretch,
    Baseline,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum TextAlign {
    #[default]
    Left,
    Center,
    Right,
    Justified,
}

/// A grid track size.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Track {
    Px(f32),
    Percent(f32),
    Fr(f32),
    Auto,
}

/// Every property is optional so styles cascade: later/more specific
/// sources override only what they set.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Style {
    // Layout.
    pub display: Option<Display>,
    pub position: Option<PositionType>,
    pub direction: Option<FlexDirection>,
    pub wrap: Option<bool>,
    pub justify_content: Option<Justify>,
    pub align_items: Option<Align>,
    pub align_self: Option<Align>,
    /// Row and column gap (px).
    pub gap: Option<(f32, f32)>,
    pub padding: Option<Edges>,
    pub margin: Option<Edges>,
    /// Offsets for absolute positioning.
    pub inset: Option<Edges>,
    pub width: Option<Val>,
    pub height: Option<Val>,
    pub min_width: Option<Val>,
    pub min_height: Option<Val>,
    pub max_width: Option<Val>,
    pub max_height: Option<Val>,
    pub flex_grow: Option<f32>,
    pub flex_shrink: Option<f32>,
    pub flex_basis: Option<Val>,
    pub aspect_ratio: Option<f32>,
    pub grid_columns: Option<Vec<Track>>,
    pub grid_rows: Option<Vec<Track>>,
    /// 1-based start line and span.
    pub grid_column: Option<(i16, u16)>,
    pub grid_row: Option<(i16, u16)>,
    /// Clip children to the node's rectangle.
    pub clip: Option<bool>,
    // Visuals.
    pub background: Option<Color>,
    pub border_color: Option<Color>,
    pub border_width: Option<f32>,
    pub corner_radius: Option<f32>,
    pub opacity: Option<f32>,
    pub text_color: Option<Color>,
    pub font_size: Option<f32>,
    /// Font family name (project fonts or the bundled "Inter").
    pub font: Option<String>,
    pub bold: Option<bool>,
    pub text_align: Option<TextAlign>,
    /// Line height as a multiple of the font size.
    pub line_height: Option<f32>,
    pub image_tint: Option<Color>,
    /// Seconds to animate color/opacity changes (state transitions).
    pub transition: Option<f32>,
    /// Draw order among siblings.
    pub z_index: Option<i32>,
}

macro_rules! merge_fields {
    ($target:ident, $source:ident, $($field:ident),* $(,)?) => {
        $(
            if $source.$field.is_some() {
                $target.$field = $source.$field.clone();
            }
        )*
    };
}

impl Style {
    /// Overrides every property `other` sets.
    pub fn merge(&mut self, other: &Style) {
        merge_fields!(
            self,
            other,
            display,
            position,
            direction,
            wrap,
            justify_content,
            align_items,
            align_self,
            gap,
            padding,
            margin,
            inset,
            width,
            height,
            min_width,
            min_height,
            max_width,
            max_height,
            flex_grow,
            flex_shrink,
            flex_basis,
            aspect_ratio,
            grid_columns,
            grid_rows,
            grid_column,
            grid_row,
            clip,
            background,
            border_color,
            border_width,
            corner_radius,
            opacity,
            text_color,
            font_size,
            font,
            bold,
            text_align,
            line_height,
            image_tint,
            transition,
            z_index,
        );
    }

    /// Text properties inherit from the parent when unset.
    pub fn inherit_text(&mut self, parent: &Style) {
        if self.text_color.is_none() {
            self.text_color = parent.text_color;
        }
        if self.font_size.is_none() {
            self.font_size = parent.font_size;
        }
        if self.font.is_none() {
            self.font = parent.font.clone();
        }
        if self.bold.is_none() {
            self.bold = parent.bold;
        }
        if self.text_align.is_none() {
            self.text_align = parent.text_align;
        }
        if self.line_height.is_none() {
            self.line_height = parent.line_height;
        }
    }
}

/// Interaction states a selector can match.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum State {
    Hover,
    Pressed,
    Focused,
    Disabled,
    Checked,
}

impl State {
    fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "hover" => Self::Hover,
            "pressed" | "active" => Self::Pressed,
            "focused" | "focus" => Self::Focused,
            "disabled" => Self::Disabled,
            "checked" => Self::Checked,
            _ => return None,
        })
    }

    pub fn bit(self) -> u8 {
        1 << self as u8
    }
}

/// A parsed selector: `Kind.class1.class2#id:state`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Selector {
    pub kind: Option<String>,
    pub classes: Vec<String>,
    pub id: Option<String>,
    pub states: Vec<State>,
}

impl Selector {
    pub fn parse(text: &str) -> std::result::Result<Self, String> {
        let mut selector = Self::default();
        let text = text.trim();
        if text.is_empty() {
            return Err("empty selector".to_owned());
        }
        let mut rest = text;
        let (head, states) = match rest.split_once(':') {
            Some((head, states)) => (head, Some(states)),
            None => (rest, None),
        };
        rest = head;
        if let Some(states) = states {
            for state in states.split(':') {
                selector
                    .states
                    .push(State::parse(state).ok_or_else(|| format!("unknown state ':{state}'"))?);
            }
        }
        // Kind prefix up to the first '.' or '#'.
        let kind_end = rest.find(['.', '#']).unwrap_or(rest.len());
        if kind_end > 0 {
            selector.kind = Some(rest[..kind_end].to_owned());
        }
        rest = &rest[kind_end..];
        while !rest.is_empty() {
            let marker = rest.chars().next().expect("non-empty");
            let body = &rest[1..];
            let end = body.find(['.', '#']).unwrap_or(body.len());
            let name = &body[..end];
            if name.is_empty() {
                return Err(format!("empty name after '{marker}' in '{text}'"));
            }
            match marker {
                '.' => selector.classes.push(name.to_owned()),
                '#' => selector.id = Some(name.to_owned()),
                _ => unreachable!(),
            }
            rest = &body[end..];
        }
        Ok(selector)
    }

    /// (ids, classes + states, kinds) — compared lexicographically.
    pub fn specificity(&self) -> (u32, u32, u32) {
        (
            u32::from(self.id.is_some()),
            (self.classes.len() + self.states.len()) as u32,
            u32::from(self.kind.is_some()),
        )
    }

    pub fn matches(&self, kind: &str, id: &str, classes: &[String], states: u8) -> bool {
        self.kind.as_deref().is_none_or(|k| k == kind)
            && self.id.as_deref().is_none_or(|i| i == id)
            && self.classes.iter().all(|c| classes.contains(c))
            && self.states.iter().all(|s| states & s.bit() != 0)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StyleRule {
    pub selector: String,
    pub style: Style,
}

pub const UI_STYLE_VERSION: u32 = 1;

/// A style sheet asset (`*.uistyle.ron`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct UiStyleSheet {
    #[serde(default)]
    pub version: u32,
    pub rules: Vec<StyleRule>,
}

impl Asset for UiStyleSheet {
    const TYPE_NAME: &'static str = "UiStyleSheet";
}

/// Rules with parsed selectors, ordered for the cascade.
#[derive(Clone, Debug, Default)]
pub struct CompiledRules {
    rules: Vec<(Selector, Style, usize)>,
}

impl CompiledRules {
    /// Compiles rules in source order (earlier sheets first); invalid
    /// selectors are reported and skipped.
    pub fn compile<'a>(rules: impl IntoIterator<Item = &'a StyleRule>) -> (Self, Vec<String>) {
        let mut compiled = Vec::new();
        let mut errors = Vec::new();
        for (order, rule) in rules.into_iter().enumerate() {
            match Selector::parse(&rule.selector) {
                Ok(selector) => compiled.push((selector, rule.style.clone(), order)),
                Err(error) => errors.push(format!("selector '{}': {error}", rule.selector)),
            }
        }
        // Lower specificity first, then source order: later merges win.
        compiled.sort_by_key(|(selector, _, order)| (selector.specificity(), *order));
        (Self { rules: compiled }, errors)
    }

    /// Cascaded style of a node: matching rules, then the inline style.
    pub fn resolve(
        &self,
        kind: &str,
        id: &str,
        classes: &[String],
        states: u8,
        inline: &Style,
    ) -> Style {
        let mut style = Style::default();
        for (selector, rule_style, _) in &self.rules {
            if selector.matches(kind, id, classes, states) {
                style.merge(rule_style);
            }
        }
        style.merge(inline);
        style
    }

    /// Whether any rule depends on `state` (skip restyling otherwise).
    pub fn uses_state(&self, state: State) -> bool {
        self.rules.iter().any(|(s, _, _)| s.states.contains(&state))
    }

    pub fn len(&self) -> usize {
        self.rules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }
}

impl UiStyleSheet {
    pub fn validate(&self) -> std::result::Result<(), Vec<String>> {
        let mut errors = Vec::new();
        if self.version > UI_STYLE_VERSION {
            errors.push(format!(
                "style sheet version {} is newer than supported",
                self.version
            ));
        }
        for rule in &self.rules {
            if let Err(error) = Selector::parse(&rule.selector) {
                errors.push(format!("selector '{}': {error}", rule.selector));
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// Loads `*.uistyle.ron`.
pub struct UiStyleSheetLoader;

impl AssetLoader for UiStyleSheetLoader {
    type Asset = UiStyleSheet;

    fn extensions(&self) -> &'static [&'static str] {
        &["uistyle.ron"]
    }

    fn load(&self, bytes: &[u8], ctx: &mut LoadContext<'_>) -> Result<UiStyleSheet> {
        let text = std::str::from_utf8(bytes).map_err(|_| ctx.error("style sheet is not UTF-8"))?;
        let sheet: UiStyleSheet = ron::from_str(text)
            .map_err(|error| ctx.error(format!("invalid style sheet: {error}")))?;
        sheet
            .validate()
            .map_err(|errors| ctx.error(errors.join("; ")))?;
        Ok(sheet)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selectors_parse_and_match() {
        let selector = Selector::parse("Button.primary.big#save:hover:focused").unwrap();
        assert_eq!(selector.kind.as_deref(), Some("Button"));
        assert_eq!(selector.classes, vec!["primary", "big"]);
        assert_eq!(selector.id.as_deref(), Some("save"));
        assert_eq!(selector.states, vec![State::Hover, State::Focused]);
        let classes = vec!["primary".to_owned(), "big".to_owned(), "x".to_owned()];
        let states = State::Hover.bit() | State::Focused.bit();
        assert!(selector.matches("Button", "save", &classes, states));
        assert!(!selector.matches("Button", "save", &classes, State::Hover.bit()));
        assert!(Selector::parse(".a:wobbly").is_err());
        assert!(Selector::parse("Button.").is_err());
    }

    #[test]
    fn cascade_respects_specificity_then_order_then_inline() {
        let rule = |selector: &str, style: Style| StyleRule {
            selector: selector.into(),
            style,
        };
        let red = [1.0, 0.0, 0.0, 1.0];
        let green = [0.0, 1.0, 0.0, 1.0];
        let blue = [0.0, 0.0, 1.0, 1.0];
        let rules = vec![
            rule(
                "#save",
                Style {
                    background: Some(blue),
                    ..Default::default()
                },
            ),
            rule(
                ".primary",
                Style {
                    background: Some(red),
                    font_size: Some(20.0),
                    ..Default::default()
                },
            ),
            rule(
                "Button",
                Style {
                    background: Some(green),
                    padding: Some(Edges::px(4.0)),
                    ..Default::default()
                },
            ),
            rule(
                ".primary:hover",
                Style {
                    background: Some(green),
                    ..Default::default()
                },
            ),
        ];
        let (compiled, errors) = CompiledRules::compile(&rules);
        assert!(errors.is_empty());
        let classes = vec!["primary".to_owned()];
        let style = compiled.resolve("Button", "other", &classes, 0, &Style::default());
        assert_eq!(style.background, Some(red), "class beats kind");
        assert_eq!(
            style.padding,
            Some(Edges::px(4.0)),
            "kind still contributes"
        );
        let style = compiled.resolve(
            "Button",
            "other",
            &classes,
            State::Hover.bit(),
            &Style::default(),
        );
        assert_eq!(style.background, Some(green), "class+state beats class");
        let style = compiled.resolve(
            "Button",
            "save",
            &classes,
            State::Hover.bit(),
            &Style::default(),
        );
        assert_eq!(style.background, Some(blue), "id beats everything");
        let inline = Style {
            font_size: Some(9.0),
            ..Default::default()
        };
        assert_eq!(
            compiled
                .resolve("Button", "save", &classes, 0, &inline)
                .font_size,
            Some(9.0)
        );
        assert!(compiled.uses_state(State::Hover));
        assert!(!compiled.uses_state(State::Pressed));
    }
}
