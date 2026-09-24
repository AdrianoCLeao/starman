//! Layout: styles → taffy (flexbox/grid) with text measurement, then
//! absolute rectangles, clip rectangles and scroll content sizes.

use engine_math::Vec2;
use taffy::prelude::{
    auto, fr, length, line, percent, span, AvailableSpace, Dimension, LengthPercentage,
    LengthPercentageAuto, Size, TaffyTree,
};
use taffy::{GridPlacement, GridTemplateComponent, Line, Overflow, Point};

use crate::document::NodeKind;
use crate::instance::{Rect, UiInstance};
use crate::style::{
    Align, Display, Edges, FlexDirection, Justify, PositionType, Style, Track, Val,
};
use crate::text::UiFonts;

/// Text measurement context per taffy leaf.
pub type NodeContext = usize;

fn dimension(value: Option<Val>, scale: f32) -> Dimension {
    match value.unwrap_or(Val::Auto) {
        Val::Auto => auto(),
        Val::Px(v) => length(v * scale),
        Val::Percent(p) => percent(p / 100.0),
    }
}

fn lpa(value: Val, scale: f32) -> LengthPercentageAuto {
    match value {
        Val::Auto => auto(),
        Val::Px(v) => length(v * scale),
        Val::Percent(p) => percent(p / 100.0),
    }
}

fn lp(value: Val, scale: f32) -> LengthPercentage {
    match value {
        Val::Auto => length(0.0),
        Val::Px(v) => length(v * scale),
        Val::Percent(p) => percent(p / 100.0),
    }
}

fn rect_lpa(edges: Option<Edges>, scale: f32, default: Val) -> taffy::Rect<LengthPercentageAuto> {
    let e = edges.unwrap_or(Edges::all(default));
    taffy::Rect {
        left: lpa(e.left, scale),
        right: lpa(e.right, scale),
        top: lpa(e.top, scale),
        bottom: lpa(e.bottom, scale),
    }
}

fn rect_lp(edges: Option<Edges>, scale: f32) -> taffy::Rect<LengthPercentage> {
    let e = edges.unwrap_or(Edges::all(Val::Px(0.0)));
    taffy::Rect {
        left: lp(e.left, scale),
        right: lp(e.right, scale),
        top: lp(e.top, scale),
        bottom: lp(e.bottom, scale),
    }
}

fn track(track: &Track, scale: f32) -> GridTemplateComponent<String> {
    match track {
        Track::Px(v) => length(v * scale),
        Track::Percent(p) => percent(p / 100.0),
        Track::Fr(f) => fr(*f),
        Track::Auto => auto(),
    }
}

fn placement(value: Option<(i16, u16)>) -> Line<GridPlacement<String>> {
    match value {
        Some((start, count)) => Line {
            start: line(start),
            end: span(count.max(1)),
        },
        None => Line {
            start: auto(),
            end: auto(),
        },
    }
}

fn align(value: Align) -> taffy::AlignItems {
    match value {
        Align::Start => taffy::AlignItems::FLEX_START,
        Align::End => taffy::AlignItems::FLEX_END,
        Align::Center => taffy::AlignItems::CENTER,
        Align::Stretch => taffy::AlignItems::STRETCH,
        Align::Baseline => taffy::AlignItems::BASELINE,
    }
}

/// Converts a resolved style; `visible` false collapses the node.
pub fn to_taffy(
    style: &Style,
    kind: &NodeKind,
    visible: bool,
    scale: f32,
    is_root: bool,
    viewport: Vec2,
) -> taffy::Style<String> {
    let display = if !visible {
        taffy::Display::None
    } else {
        match style.display.unwrap_or_default() {
            Display::Flex => taffy::Display::Flex,
            Display::Grid => taffy::Display::Grid,
            Display::None => taffy::Display::None,
        }
    };
    let scroll = matches!(kind, NodeKind::ScrollView { .. });
    let (scroll_x, scroll_y) = match kind {
        NodeKind::ScrollView {
            horizontal,
            vertical,
        } => (*horizontal, *vertical),
        _ => (false, false),
    };
    let overflow = |on: bool| {
        if on {
            Overflow::Scroll
        } else if style.clip.unwrap_or(false) || scroll {
            Overflow::Clip
        } else {
            Overflow::Visible
        }
    };
    let mut size = Size {
        width: dimension(style.width, scale),
        height: dimension(style.height, scale),
    };
    if is_root {
        if style.width.is_none() {
            size.width = length(viewport.x);
        }
        if style.height.is_none() {
            size.height = length(viewport.y);
        }
    }
    let border = style.border_width.unwrap_or(0.0) * scale;
    taffy::Style {
        display,
        position: match style.position.unwrap_or_default() {
            PositionType::Relative => taffy::Position::Relative,
            PositionType::Absolute => taffy::Position::Absolute,
        },
        overflow: Point {
            x: overflow(scroll_x),
            y: overflow(scroll_y),
        },
        scrollbar_width: 0.0,
        inset: rect_lpa(style.inset, scale, Val::Auto),
        size,
        min_size: Size {
            width: lpa(style.min_width.unwrap_or(Val::Auto), scale),
            height: lpa(style.min_height.unwrap_or(Val::Auto), scale),
        },
        max_size: Size {
            width: lpa(style.max_width.unwrap_or(Val::Auto), scale),
            height: lpa(style.max_height.unwrap_or(Val::Auto), scale),
        },
        aspect_ratio: style.aspect_ratio,
        margin: rect_lpa(style.margin, scale, Val::Px(0.0)),
        padding: rect_lp(style.padding, scale),
        border: taffy::Rect {
            left: length(border),
            right: length(border),
            top: length(border),
            bottom: length(border),
        },
        align_items: style.align_items.map(align),
        align_self: style.align_self.map(align),
        justify_content: style.justify_content.map(|j| match j {
            Justify::Start => taffy::JustifyContent::FLEX_START,
            Justify::End => taffy::JustifyContent::FLEX_END,
            Justify::Center => taffy::JustifyContent::CENTER,
            Justify::SpaceBetween => taffy::JustifyContent::SPACE_BETWEEN,
            Justify::SpaceAround => taffy::JustifyContent::SPACE_AROUND,
            Justify::SpaceEvenly => taffy::JustifyContent::SPACE_EVENLY,
        }),
        gap: {
            let (row, column) = style.gap.unwrap_or((0.0, 0.0));
            Size {
                width: length(column * scale),
                height: length(row * scale),
            }
        },
        flex_direction: match style.direction.unwrap_or_default() {
            FlexDirection::Row => taffy::FlexDirection::Row,
            FlexDirection::Column => taffy::FlexDirection::Column,
            FlexDirection::RowReverse => taffy::FlexDirection::RowReverse,
            FlexDirection::ColumnReverse => taffy::FlexDirection::ColumnReverse,
        },
        flex_wrap: if style.wrap.unwrap_or(false) {
            taffy::FlexWrap::Wrap
        } else {
            taffy::FlexWrap::NoWrap
        },
        flex_basis: dimension(style.flex_basis, scale),
        flex_grow: style.flex_grow.unwrap_or(0.0),
        flex_shrink: style.flex_shrink.unwrap_or(if scroll { 0.0 } else { 1.0 }),
        grid_template_columns: style
            .grid_columns
            .as_ref()
            .map(|tracks| tracks.iter().map(|t| track(t, scale)).collect())
            .unwrap_or_default(),
        grid_template_rows: style
            .grid_rows
            .as_ref()
            .map(|tracks| tracks.iter().map(|t| track(t, scale)).collect())
            .unwrap_or_default(),
        grid_column: placement(style.grid_column),
        grid_row: placement(style.grid_row),
        ..Default::default()
    }
}

/// Lays out `instance` for a `viewport` (physical px) at `scale`.
pub fn compute_layout(
    instance: &mut UiInstance,
    taffy: &mut TaffyTree<NodeContext>,
    fonts: &mut UiFonts,
    scale: f32,
    viewport: Vec2,
) {
    // (Re)build the taffy tree: cheap for UI-sized trees and keeps it in
    // sync with rebuilds, visibility and style changes.
    taffy.clear();
    let order = instance.preorder();
    for &index in order.iter().rev() {
        let node = &instance.nodes[index];
        let shown = node.visible;
        let style = to_taffy(
            &node.style,
            &node.kind,
            shown,
            scale,
            index == instance.root,
            viewport,
        );
        let children: Vec<taffy::NodeId> = instance
            .draw_children(index)
            .into_iter()
            .filter_map(|c| instance.nodes[c].taffy)
            .collect();
        let id = match node.text {
            Some(text) if children.is_empty() => taffy
                .new_leaf_with_context(style, text)
                .expect("taffy leaf"),
            _ => taffy
                .new_with_children(style, &children)
                .expect("taffy node"),
        };
        instance.nodes[index].taffy = Some(id);
    }
    let Some(root) = instance.nodes[instance.root].taffy else {
        return;
    };
    let texts = &mut instance.texts;
    let _ = taffy.compute_layout_with_measure(
        root,
        Size {
            width: AvailableSpace::Definite(viewport.x),
            height: AvailableSpace::Definite(viewport.y),
        },
        |inputs, _node, context, style| {
            taffy::compute_leaf_layout(
                inputs,
                style,
                |_, _| 0.0,
                |known, available| {
                    let Some(&mut slot) = context else {
                        return Size::ZERO;
                    };
                    let Some(block) = texts.get_mut(slot) else {
                        return Size::ZERO;
                    };
                    let width = known.width.or(match available.width {
                        AvailableSpace::Definite(w) => Some(w),
                        AvailableSpace::MinContent => Some(0.0),
                        AvailableSpace::MaxContent => None,
                    });
                    let (w, h) = block.measure(fonts, width);
                    Size {
                        width: known.width.unwrap_or(w),
                        height: known.height.unwrap_or(h),
                    }
                },
            )
        },
    );

    // Absolute rectangles, clips and scroll offsets.
    fn place(
        instance: &mut UiInstance,
        taffy: &TaffyTree<NodeContext>,
        index: usize,
        origin: Vec2,
        clip: Rect,
    ) {
        let Some(id) = instance.nodes[index].taffy else {
            return;
        };
        let Ok(layout) = taffy.layout(id) else {
            return;
        };
        let rect = Rect {
            x: origin.x + layout.location.x,
            y: origin.y + layout.location.y,
            width: layout.size.width,
            height: layout.size.height,
        };
        let content = Vec2::new(
            layout.scrollable_overflow_rect.right.max(layout.size.width),
            layout
                .scrollable_overflow_rect
                .bottom
                .max(layout.size.height),
        );
        let inner = |side: f32, pad: f32| side + pad;
        let content_rect = Rect {
            x: inner(rect.x, layout.border.left + layout.padding.left),
            y: inner(rect.y, layout.border.top + layout.padding.top),
            width: (rect.width
                - layout.border.left
                - layout.border.right
                - layout.padding.left
                - layout.padding.right)
                .max(0.0),
            height: (rect.height
                - layout.border.top
                - layout.border.bottom
                - layout.padding.top
                - layout.padding.bottom)
                .max(0.0),
        };
        let node = &mut instance.nodes[index];
        node.rect = rect;
        node.content = content_rect;
        node.clip = clip;
        node.content_size = content;
        // Keep scroll offsets within the content.
        let max_scroll = Vec2::new(
            (content.x - rect.width).max(0.0),
            (content.y - rect.height).max(0.0),
        );
        node.widget.scroll = node.widget.scroll.clamp(Vec2::ZERO, max_scroll);
        let clips =
            node.style.clip.unwrap_or(false) || matches!(node.kind, NodeKind::ScrollView { .. });
        let child_clip = if clips { clip.intersect(&rect) } else { clip };
        let child_origin = Vec2::new(rect.x, rect.y) - node.widget.scroll;
        let children = node.children.clone();
        for child in children {
            place(instance, taffy, child, child_origin, child_clip);
        }
    }
    let root_index = instance.root;
    place(instance, taffy, root_index, Vec2::ZERO, Rect::INFINITE);
    instance.relayout = false;
}

/// Scale factor for a viewport from the reference resolution and mode.
pub fn scale_factor(mode: ScaleMode, reference: Vec2, viewport: Vec2, dpi: f32) -> f32 {
    let (rx, ry) = (reference.x.max(1.0), reference.y.max(1.0));
    let factor = match mode {
        ScaleMode::MatchHeight => viewport.y / ry,
        ScaleMode::MatchWidth => viewport.x / rx,
        ScaleMode::Fit => (viewport.x / rx).min(viewport.y / ry),
        ScaleMode::ConstantPixelSize => dpi,
    };
    factor.max(0.05)
}

/// How UI units map to pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ScaleMode {
    #[default]
    MatchHeight,
    MatchWidth,
    Fit,
    ConstantPixelSize,
}
