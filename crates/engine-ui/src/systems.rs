//! ECS integration: documents, settings, events and the per-frame UI
//! update (assets → rebuild → input → bindings → style → text → layout →
//! draw list).

use std::collections::HashMap;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use bevy_reflect::Reflect;
use engine_assets::{AssetRef, Assets, Handle};
use engine_core::{
    Camera3d, CursorGrab, CursorState, FrameTime, GlobalTransform, PrimaryCamera, WindowSize,
};
use engine_input::{InputState, LocalPlayers};
use engine_localization::Localization;
use engine_math::{Mat4, Vec2, Vec3};
use gilrs::Button;
use taffy::TaffyTree;
use winit::event::MouseButton;
use winit::keyboard::KeyCode;

use crate::document::UiLayout;
use crate::draw::{build_quads, UiDrawList};
use crate::instance::UiInstance;
use crate::interact::{interact, Interaction, NavInput, PointerInput, UiEventKind};
use crate::layout::{compute_layout, scale_factor, NodeContext, ScaleMode};
use crate::model::UiModel;
use crate::style::{CompiledRules, StyleRule, UiStyleSheet};
use crate::text::{FontData, UiFonts};

/// Displays a UI layout (`*.ui.ron`).
#[derive(Component, Clone, Debug, PartialEq, Reflect, engine_reflect::RegisterReflect)]
pub struct UiDocument {
    pub layout: AssetRef,
    /// Higher documents draw on top and get input first.
    pub order: i32,
    pub visible: bool,
    /// Receives pointer and navigation input.
    pub interactive: bool,
    /// Blocks input to documents below while visible (menus).
    pub modal: bool,
    /// Positions the root at the entity's projected position (health bars).
    pub world_anchor: bool,
    pub anchor_offset: Vec3,
    /// Local player whose `ui_*` actions navigate this document.
    pub player: u8,
}

impl Default for UiDocument {
    fn default() -> Self {
        Self {
            layout: AssetRef::default(),
            order: 0,
            visible: true,
            interactive: true,
            modal: false,
            world_anchor: false,
            anchor_offset: Vec3::ZERO,
            player: 0,
        }
    }
}

/// A UI event from a document node.
#[derive(Event, Clone, Debug, PartialEq)]
pub struct UiEvent {
    pub document: Entity,
    pub node: String,
    /// Index within a list, for nodes instantiated from a list template.
    pub item: Option<usize>,
    pub kind: UiEventKind,
}

/// Scaling and fonts.
#[derive(Resource, Clone, Debug, PartialEq)]
pub struct UiSettings {
    pub reference_resolution: Vec2,
    pub scale_mode: ScaleMode,
    /// Extra multiplier (accessibility text/UI scale).
    pub user_scale: f32,
    /// Project fonts to load (`.ttf`/`.otf`).
    pub fonts: Vec<AssetRef>,
}

impl Default for UiSettings {
    fn default() -> Self {
        Self {
            reference_resolution: Vec2::new(1920.0, 1080.0),
            scale_mode: ScaleMode::MatchHeight,
            user_scale: 1.0,
            fonts: Vec::new(),
        }
    }
}

/// Whether UI is consuming input this frame (gameplay should ignore it).
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UiInputCapture {
    pub pointer_over_ui: bool,
    /// A document has keyboard/gamepad focus.
    pub navigation: bool,
    /// A modal document is visible.
    pub modal: bool,
    /// A text input is focused (gameplay must not read keys).
    pub text_input: bool,
}

/// Layout trees per document. Kept out of components because taffy styles
/// are not `Send` (they can hold `calc()` pointers); the UI update runs on
/// the main thread.
#[derive(Default)]
pub struct UiLayoutTrees {
    trees: HashMap<Entity, TaffyTree<NodeContext>>,
}

/// Runtime state of a [`UiDocument`].
#[derive(Component, Default)]
pub struct UiDocumentState {
    reference: AssetRef,
    handle: Option<Handle<UiLayout>>,
    revision: u64,
    sheets: Vec<(Handle<UiStyleSheet>, u64)>,
    layout: Option<Arc<UiLayout>>,
    pub instance: Option<UiInstance>,
    pub interaction: Interaction,
    model_revision: u64,
    l10n_revision: u64,
    fonts_revision: u64,
    scale: f32,
    viewport: Vec2,
    caret_time: f32,
    /// Style/layout problems (shown by the editor).
    pub errors: Vec<String>,
}

impl UiDocumentState {
    /// Text of a node by id (tests, tools).
    pub fn text(&self, id: &str) -> Option<&str> {
        self.instance.as_ref()?.text_of(id)
    }

    /// Currently focused node id.
    pub fn focused(&self) -> Option<&str> {
        let instance = self.instance.as_ref()?;
        self.interaction
            .focus
            .map(|i| instance.nodes[i].id.as_str())
    }

    /// Moves focus to `id` (scripts, menus opening).
    pub fn focus(&mut self, id: &str) -> bool {
        let Some(instance) = self.instance.as_mut() else {
            return false;
        };
        let Some(&index) = instance.by_id.get(id) else {
            return false;
        };
        if let Some(old) = self.interaction.focus {
            if old < instance.nodes.len() {
                instance.nodes[old].set_state(crate::style::State::Focused, false);
            }
        }
        instance.nodes[index].set_state(crate::style::State::Focused, true);
        instance.restyle = true;
        self.interaction.focus = Some(index);
        true
    }
}

fn default_rules() -> Vec<StyleRule> {
    static THEME: std::sync::OnceLock<Vec<StyleRule>> = std::sync::OnceLock::new();
    THEME
        .get_or_init(|| {
            ron::from_str::<UiStyleSheet>(include_str!("theme.uistyle.ron"))
                .expect("built-in theme parses")
                .rules
        })
        .clone()
}

fn nav_input(input: &InputState, players: Option<&LocalPlayers>, player: u8) -> NavInput {
    let action = |name: &str| {
        players
            .and_then(|p| p.player(player))
            .is_some_and(|p| p.just_pressed(name))
    };
    let pad = |button: Button| {
        input
            .connected_gamepads()
            .iter()
            .any(|slot| input.gamepad_button_just_pressed(*slot, button))
    };
    NavInput {
        up: action("ui_up") || input.key_just_pressed(KeyCode::ArrowUp) || pad(Button::DPadUp),
        down: action("ui_down")
            || input.key_just_pressed(KeyCode::ArrowDown)
            || pad(Button::DPadDown),
        left: action("ui_left")
            || input.key_just_pressed(KeyCode::ArrowLeft)
            || pad(Button::DPadLeft),
        right: action("ui_right")
            || input.key_just_pressed(KeyCode::ArrowRight)
            || pad(Button::DPadRight),
        submit: action("ui_submit")
            || input.key_just_pressed(KeyCode::Enter)
            || input.key_just_pressed(KeyCode::NumpadEnter)
            || pad(Button::South),
        cancel: action("ui_cancel") || input.key_just_pressed(KeyCode::Escape) || pad(Button::East),
        typed: input.typed_text().to_owned(),
        backspace: input.key_just_pressed(KeyCode::Backspace),
    }
}

/// Resolves the layout and its style sheets; returns whether they changed.
fn refresh_assets(state: &mut UiDocumentState, document: &UiDocument, assets: &Assets) -> bool {
    if state.reference != document.layout {
        state.reference = document.layout.clone();
        state.handle = None;
        state.layout = None;
        state.instance = None;
        state.sheets.clear();
    }
    if state.reference.is_empty() {
        return false;
    }
    let handle = *state
        .handle
        .get_or_insert_with(|| assets.request::<UiLayout>(&state.reference));
    let revision = assets.revision(handle);
    let mut changed = false;
    if state.layout.is_none() || revision != state.revision {
        if let Some(layout) = assets.get(handle) {
            state.revision = revision;
            state.sheets = layout
                .styles
                .iter()
                .map(|sheet| (assets.request::<UiStyleSheet>(sheet), 0))
                .collect();
            state.layout = Some(layout);
            changed = true;
        }
    }
    for (sheet, seen) in &mut state.sheets {
        let revision = assets.revision(*sheet);
        if revision != *seen && assets.get(*sheet).is_some() {
            *seen = revision;
            changed = true;
        }
    }
    changed
}

fn compile_rules(state: &mut UiDocumentState, assets: &Assets) -> CompiledRules {
    let mut rules = default_rules();
    for (sheet, _) in &state.sheets {
        if let Some(sheet) = assets.get(*sheet) {
            rules.extend(sheet.rules.iter().cloned());
        }
    }
    if let Some(layout) = &state.layout {
        rules.extend(layout.rules.iter().cloned());
    }
    let (compiled, errors) = CompiledRules::compile(&rules);
    state.errors = errors;
    compiled
}

fn project(view_proj: &Mat4, world: Vec3, viewport: Vec2) -> Option<Vec2> {
    let clip = *view_proj * world.extend(1.0);
    if clip.w <= 0.0 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    Some(Vec2::new(
        (ndc.x * 0.5 + 0.5) * viewport.x,
        (0.5 - ndc.y * 0.5) * viewport.y,
    ))
}

/// Frame inputs of the UI update.
#[derive(bevy_ecs::system::SystemParam)]
pub struct UiFrameInputs<'w> {
    input: Option<Res<'w, InputState>>,
    players: Option<Res<'w, LocalPlayers>>,
    cursor: Option<Res<'w, CursorState>>,
    window: Option<Res<'w, WindowSize>>,
    time: Option<Res<'w, FrameTime>>,
    l10n: Option<Res<'w, Localization>>,
}

/// Frame outputs of the UI update.
#[derive(bevy_ecs::system::SystemParam)]
pub struct UiFrameOutputs<'w> {
    draw_list: ResMut<'w, UiDrawList>,
    capture: ResMut<'w, UiInputCapture>,
    events: EventWriter<'w, UiEvent>,
}

/// The per-frame UI update.
#[allow(clippy::too_many_arguments)]
pub fn update_ui(
    mut commands: Commands,
    assets: Option<Res<Assets>>,
    settings: Res<UiSettings>,
    mut fonts: ResMut<UiFonts>,
    mut model: ResMut<UiModel>,
    frame: UiFrameInputs,
    cameras: Query<(&Camera3d, &GlobalTransform, Option<&PrimaryCamera>)>,
    mut documents: Query<(
        Entity,
        &UiDocument,
        Option<&mut UiDocumentState>,
        Option<&GlobalTransform>,
    )>,
    outputs: UiFrameOutputs,
    mut font_handles: Local<Vec<(AssetRef, Handle<FontData>)>>,
    mut trees: NonSendMut<UiLayoutTrees>,
) {
    let UiFrameInputs {
        input,
        players,
        cursor,
        window,
        time,
        l10n,
    } = frame;
    let UiFrameOutputs {
        mut draw_list,
        mut capture,
        mut events,
    } = outputs;
    let Some(assets) = assets else {
        return;
    };
    let dt = time.map_or(0.0, |t| t.real_delta_seconds);
    let viewport = window
        .map(|w| Vec2::new(w.width.max(1) as f32, w.height.max(1) as f32))
        .unwrap_or(Vec2::new(1280.0, 720.0));
    let scale = scale_factor(
        settings.scale_mode,
        settings.reference_resolution,
        viewport,
        1.0,
    ) * settings.user_scale.max(0.1);

    // Project fonts.
    for font in &settings.fonts {
        if !font_handles.iter().any(|(r, _)| r == font) {
            font_handles.push((font.clone(), assets.request::<FontData>(font)));
        }
    }
    for (_, handle) in font_handles.iter() {
        if let Some(data) = assets.get(*handle) {
            fonts.add_font(handle.id(), &data);
        }
    }

    let view_proj = cameras
        .iter()
        .max_by_key(|(_, _, primary)| primary.is_some())
        .map(|(camera, global, _)| camera.projection_matrix() * Mat4::from(global.0.inverse()));

    // Input routing: topmost documents first.
    let pointer_allowed = cursor.is_none_or(|c| c.visible && c.grab != CursorGrab::Locked);
    let mut pointer = PointerInput::default();
    let mut nav_by_player: HashMap<u8, NavInput> = HashMap::new();
    if let Some(input) = input.as_deref() {
        if pointer_allowed {
            pointer = PointerInput {
                position: Some(input.mouse_position),
                held: input.mouse_held(MouseButton::Left),
                just_pressed: input.mouse_just_pressed(MouseButton::Left),
                just_released: input.mouse_just_released(MouseButton::Left),
                wheel: input.scroll_delta,
            };
        }
        for (_, document, _, _) in &documents {
            nav_by_player
                .entry(document.player)
                .or_insert_with(|| nav_input(input, players.as_deref(), document.player));
        }
    }
    let mut order: Vec<(i32, Entity)> = documents
        .iter()
        .map(|(entity, document, _, _)| (document.order, entity))
        .collect();
    order.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));

    let mut new_capture = UiInputCapture::default();
    let mut pointer_taken = false;
    let mut nav_taken: Vec<u8> = Vec::new();
    let mut blocked = false;
    let mut model_writes = Vec::new();
    let mut per_document_quads: Vec<(i32, Entity, Vec<crate::draw::UiQuad>)> = Vec::new();

    for (_, entity) in order {
        let Ok((entity, document, state, global)) = documents.get_mut(entity) else {
            continue;
        };
        let Some(mut state) = state else {
            commands.entity(entity).insert(UiDocumentState::default());
            continue;
        };
        let state = &mut *state;
        if refresh_assets(state, document, &assets) {
            state.instance = None;
        }
        let Some(layout) = state.layout.clone() else {
            continue;
        };

        // (Re)build when needed, keeping widget state across rebuilds.
        let rebuild = state
            .instance
            .as_ref()
            .is_none_or(|i| i.lists_changed(&model));
        if rebuild {
            let preserved = state
                .instance
                .as_ref()
                .map(|i| i.widget_states())
                .unwrap_or_default();
            let focused = state.focused().map(str::to_owned);
            let rules = compile_rules(state, &assets);
            state.instance = Some(UiInstance::build(&layout, rules, &model, &preserved));
            state.interaction = Interaction::default();
            if let Some(id) = focused {
                state.focus(&id);
            }
            state.model_revision = u64::MAX;
        }
        let Some(instance) = state.instance.as_mut() else {
            continue;
        };

        // Input.
        let accepts = document.visible && document.interactive && !blocked;
        let has_focusables = instance
            .nodes
            .iter()
            .any(|n| n.focusable && n.part.is_none());
        let doc_pointer = if accepts && !pointer_taken {
            pointer
        } else {
            PointerInput {
                position: None,
                ..Default::default()
            }
        };
        let doc_nav = if accepts && has_focusables && !nav_taken.contains(&document.player) {
            nav_taken.push(document.player);
            nav_by_player
                .get(&document.player)
                .cloned()
                .unwrap_or_default()
        } else {
            NavInput::default()
        };
        let output = interact(
            instance,
            &mut state.interaction,
            &doc_pointer,
            &doc_nav,
            accepts,
            state.scale.max(0.01),
        );
        if state.interaction.pointer_over && document.visible {
            pointer_taken = true;
            new_capture.pointer_over_ui = true;
        }
        if accepts && state.interaction.focus.is_some() {
            new_capture.navigation = true;
            if state.interaction.focus.is_some_and(|f| {
                matches!(
                    instance.nodes[f].kind,
                    crate::document::NodeKind::TextInput { .. }
                )
            }) {
                new_capture.text_input = true;
            }
        }
        if document.visible && document.modal {
            blocked = true;
            new_capture.modal = true;
        }
        if output.restyle {
            instance.restyle = true;
        }
        for event in output.events {
            events.send(UiEvent {
                document: entity,
                node: event.node,
                item: event.item,
                kind: event.kind,
            });
        }
        model_writes.extend(output.model_writes);

        // Bindings, styles, texts, layout.
        let l10n_revision = l10n.as_ref().map_or(0, |l| l.revision());
        let bindings_dirty =
            state.model_revision != model.revision() || state.l10n_revision != l10n_revision;
        if bindings_dirty {
            if instance.apply_bindings(&model) {
                instance.relayout = true;
            }
        } else if output.restyle || instance.relayout {
            // Widget changes (toggle, slider) update their parts.
            instance.apply_bindings(&model);
        }
        if instance.restyle {
            instance.resolve_styles();
            instance.relayout = true;
        }
        instance.animate(dt);
        let scale_changed = (state.scale - scale).abs() > 1e-4 || state.viewport != viewport;
        let texts_changed = if bindings_dirty
            || scale_changed
            || fonts.revision != state.fonts_revision
            || rebuild
            || instance.relayout
        {
            instance.update_texts(&mut fonts, &model, l10n.as_deref(), scale)
        } else {
            false
        };
        if texts_changed || scale_changed || instance.relayout || rebuild {
            let tree = trees.trees.entry(entity).or_insert_with(TaffyTree::new);
            compute_layout(instance, tree, &mut fonts, scale, viewport);
        }
        state.model_revision = model.revision();
        state.l10n_revision = l10n_revision;
        state.fonts_revision = fonts.revision;
        state.scale = scale;
        state.viewport = viewport;

        // Draw.
        if !document.visible {
            continue;
        }
        let mut origin = Vec2::ZERO;
        if document.world_anchor {
            let world =
                global.map_or(Vec3::ZERO, |g| Vec3::from(g.0.translation)) + document.anchor_offset;
            let Some(point) = view_proj.and_then(|vp| project(&vp, world, viewport)) else {
                continue;
            };
            let root = instance.nodes[instance.root].rect;
            origin = point - Vec2::new(root.x + root.width * 0.5, root.y + root.height);
        }
        state.caret_time = (state.caret_time + dt) % 1.0;
        let mut quads = Vec::new();
        build_quads(
            instance,
            &mut fonts,
            Some(&assets),
            state.interaction.focus,
            state.caret_time < 0.5,
            scale,
            origin,
            &mut quads,
        );
        per_document_quads.push((document.order, entity, quads));
    }

    // Forget trees of despawned documents.
    trees.trees.retain(|entity, _| documents.contains(*entity));
    for (key, value) in model_writes {
        model.set(key, value);
    }
    // Paint bottom-up.
    per_document_quads.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
    draw_list.quads.clear();
    for (_, _, quads) in per_document_quads {
        draw_list.quads.extend(quads);
    }
    draw_list.sync_atlas(&fonts);
    if *capture != new_capture {
        *capture = new_capture;
    }
}
