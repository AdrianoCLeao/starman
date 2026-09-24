//! Game UI (ADR 0015): retained, versioned layouts (`*.ui.ron`) and style
//! sheets (`*.uistyle.ron`), flexbox/grid layout (taffy), shaped text
//! (cosmic-text) with a bundled OFL font, pointer and focus navigation
//! driven by input actions, data binding to a view model, localization
//! through Fluent and a batched render node.

// ECS query tuples are inherently long; aliasing each one hurts more than helps.
#![allow(clippy::type_complexity)]

pub mod document;
pub mod draw;
pub mod instance;
pub mod interact;
pub mod layout;
pub mod model;
pub mod plugin;
pub mod render;
pub mod style;
pub mod systems;
pub mod text;

pub use document::{
    BindTarget, Binding, NavLinks, NodeKind, TextSource, UiLayout, UiLayoutLoader, UiNode,
    ValueSource, UI_LAYOUT_VERSION,
};
pub use draw::{QuadTexture, UiDrawList, UiQuad};
pub use instance::{Rect, UiInstance};
pub use interact::UiEventKind;
pub use layout::ScaleMode;
pub use model::{UiModel, UiValue};
pub use plugin::UiPlugin;
pub use render::UiRenderer;
pub use style::{
    Align, Color, Display, Edges, FlexDirection, Justify, PositionType, State, Style, StyleRule,
    TextAlign, Track, UiStyleSheet, UiStyleSheetLoader, Val,
};
pub use systems::{UiDocument, UiDocumentState, UiEvent, UiInputCapture, UiSettings};
pub use text::{FontData, FontLoader, UiFonts};
