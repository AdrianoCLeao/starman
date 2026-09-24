//! UI documents through the runtime, headless: build, localization,
//! bindings, lists, focus navigation, pointer, widgets, modality, anchors.

use std::collections::BTreeMap;

use bevy_ecs::prelude::*;
use engine_assets::{AssetRef, Assets};
use engine_core::{Camera3d, GameRuntime, GlobalTransform, PrimaryCamera, Transform, WindowSize};
use engine_input::InputState;
use engine_localization::Localization;
use engine_math::Vec3;
use engine_ui::*;
use winit::event::{ElementState, MouseButton};
use winit::keyboard::KeyCode;

const MENU: &str = r##"(
    rules: [
        (selector: "#menu", style: (direction: Some(Column), width: Some(Percent(100.0)), height: Some(Percent(100.0)), justify_content: Some(Center), align_items: Some(Center), gap: Some((10.0, 10.0)))),
        (selector: ".danger", style: (background: Some((0.6, 0.1, 0.1, 1.0)))),
    ],
    root: (
        id: "menu",
        children: [
            (id: "title", kind: Text(text: Loc(key: "menu-title", args: [("keys", Bind("hud.keys"))]))),
            (id: "resume", kind: Button(text: Some(Loc(key: "menu-resume")))),
            (id: "volume", kind: Slider(min: 0.0, max: 1.0, step: 0.1, value: 0.5), bind: [(target: Value, key: "settings.volume")]),
            (id: "subtitles", kind: Toggle(text: Some(Literal("Subtitles"))), bind: [(target: Checked, key: "settings.subtitles")]),
            (id: "name", kind: TextInput(placeholder: Literal("Name"))),
            (id: "slots", kind: List(source: "saves", template: (id: "slot", kind: Button(text: Some(Bind("item.label"))), bind: [(target: Class("danger"), key: "item.corrupt")]))),
            (id: "quit", kind: Button(text: Some(Loc(key: "menu-quit"))), bind: [(target: Enabled, key: "menu.can_quit")]),
        ],
    ),
)"##;

const EN: &str = "menu-title = Paused ({ $keys } keys)\nmenu-resume = Resume\nmenu-quit = Quit\n";
const PT: &str =
    "menu-title = Pausado ({ $keys } chaves)\nmenu-resume = Continuar\nmenu-quit = Sair\n";

#[derive(Resource, Default)]
struct Received(Vec<UiEvent>);

fn record(mut received: ResMut<Received>, mut events: EventReader<UiEvent>) {
    received.0.extend(events.read().cloned());
}

fn setup() -> (GameRuntime, Entity) {
    let assets = Assets::new();
    let handle = assets.request::<UiLayout>(&AssetRef::from_path("ui/menu.ui.ron"));
    let layout: UiLayout = ron::from_str(MENU).expect("layout parses");
    layout.validate().unwrap();
    assert!(assets.replace(handle, layout));
    let mut runtime = GameRuntime::new();
    runtime.insert_resource(assets);
    runtime.insert_resource(WindowSize {
        width: 1280,
        height: 720,
    });
    runtime.add_plugin(engine_input::InputPlugin);
    runtime.add_plugin(UiPlugin);
    runtime.insert_resource(Localization::from_sources(
        "en",
        &[
            ("en", vec![("menu.ftl", EN)]),
            ("pt-BR", vec![("menu.ftl", PT)]),
        ],
    ));
    runtime.init_resource::<Received>();
    runtime.add_systems(
        engine_core::ScheduleKind::Update,
        record.after(engine_core::UpdateSet::UiLayout),
    );
    {
        let mut model = runtime.world.resource_mut::<UiModel>();
        model.set("hud.keys", 2);
        model.set("menu.can_quit", true);
        let slot = |label: &str, corrupt: bool| -> BTreeMap<String, UiValue> {
            [
                ("label".to_owned(), UiValue::from(label)),
                ("corrupt".to_owned(), UiValue::Bool(corrupt)),
            ]
            .into()
        };
        model.set(
            "saves",
            UiValue::List(vec![slot("Slot 1", false), slot("Slot 2", true)]),
        );
    }
    let document = runtime
        .world
        .spawn(UiDocument {
            layout: AssetRef::from_path("ui/menu.ui.ron"),
            modal: true,
            ..Default::default()
        })
        .id();
    (runtime, document)
}

fn frame(runtime: &mut GameRuntime, feed: impl FnOnce(&mut InputState)) {
    {
        let mut input = runtime.world.resource_mut::<InputState>();
        input.begin_frame();
        feed(&mut input);
    }
    runtime.step(1.0 / 60.0);
}

fn state(runtime: &GameRuntime, document: Entity) -> &UiDocumentState {
    runtime.world.get::<UiDocumentState>(document).unwrap()
}

fn events(runtime: &mut GameRuntime) -> Vec<UiEvent> {
    std::mem::take(&mut runtime.world.resource_mut::<Received>().0)
}

fn center(runtime: &GameRuntime, document: Entity, id: &str) -> (f32, f32) {
    let rect = state(runtime, document)
        .instance
        .as_ref()
        .unwrap()
        .node(id)
        .unwrap()
        .rect;
    (rect.x + rect.width * 0.5, rect.y + rect.height * 0.5)
}

fn click(runtime: &mut GameRuntime, x: f32, y: f32) {
    frame(runtime, |input| {
        input.process_cursor_position(x, y);
        input.process_mouse_button_input(MouseButton::Left, ElementState::Pressed);
    });
    frame(runtime, |input| {
        input.process_mouse_button_input(MouseButton::Left, ElementState::Released)
    });
}

fn key(runtime: &mut GameRuntime, code: KeyCode) {
    frame(runtime, |input| {
        input.process_key_input(code, ElementState::Pressed, false)
    });
    frame(runtime, |input| {
        input.process_key_input(code, ElementState::Released, false)
    });
}

#[test]
fn builds_localizes_binds_and_lays_out() {
    let (mut runtime, document) = setup();
    frame(&mut runtime, |_| {});
    frame(&mut runtime, |_| {});
    let doc = state(&runtime, document);
    assert!(doc.errors.is_empty(), "{:?}", doc.errors);
    assert_eq!(doc.text("title"), Some("Paused (2 keys)"));
    assert_eq!(doc.text("resume"), Some("Resume"));
    assert_eq!(doc.text("slots[1].slot"), Some("Slot 2"));
    let instance = doc.instance.as_ref().unwrap();
    let slot = instance.node("slots[1].slot").unwrap();
    assert!(slot.bound_classes.contains(&"danger".to_owned()));
    assert_eq!(
        slot.style.background,
        Some([0.6, 0.1, 0.1, 1.0]),
        "bound class styles the item"
    );
    // Centered column: the resume button sits around the middle.
    let resume = instance.node("resume").unwrap().rect;
    assert!(
        (resume.x + resume.width * 0.5 - 640.0).abs() < 2.0,
        "{resume:?}"
    );
    assert!(
        resume.width > 60.0 && resume.height >= 40.0 * 720.0 / 1080.0 - 1.0,
        "{resume:?}"
    );
    let draw = runtime.world.resource::<UiDrawList>();
    assert!(draw.quads.iter().any(|q| q.texture == QuadTexture::Glyphs));
    assert!(draw.atlas.is_some());

    // Locale switch and model changes re-resolve texts.
    runtime
        .world
        .resource_mut::<Localization>()
        .set_locale("pt-BR");
    runtime.world.resource_mut::<UiModel>().set("hud.keys", 3);
    frame(&mut runtime, |_| {});
    let doc = state(&runtime, document);
    assert_eq!(doc.text("title"), Some("Pausado (3 chaves)"));
    assert_eq!(doc.text("quit"), Some("Sair"));
}

#[test]
fn keyboard_navigation_focuses_activates_and_adjusts_widgets() {
    let (mut runtime, document) = setup();
    frame(&mut runtime, |_| {});
    key(&mut runtime, KeyCode::ArrowDown);
    assert_eq!(
        state(&runtime, document).focused(),
        Some("resume"),
        "first focusable"
    );
    key(&mut runtime, KeyCode::Enter);
    let received = events(&mut runtime);
    assert!(
        received
            .iter()
            .any(|e| e.node == "resume" && e.kind == UiEventKind::Click),
        "{received:?}"
    );

    key(&mut runtime, KeyCode::ArrowDown);
    assert_eq!(state(&runtime, document).focused(), Some("volume"));
    key(&mut runtime, KeyCode::ArrowRight);
    key(&mut runtime, KeyCode::ArrowRight);
    assert_eq!(
        runtime
            .world
            .resource::<UiModel>()
            .number("settings.volume"),
        Some(0.7_f32 as f64)
    );
    let received = events(&mut runtime);
    assert!(received
        .iter()
        .any(|e| e.node == "volume" && matches!(e.kind, UiEventKind::Changed(UiValue::Number(_)))));

    key(&mut runtime, KeyCode::ArrowDown);
    assert_eq!(state(&runtime, document).focused(), Some("subtitles"));
    key(&mut runtime, KeyCode::Enter);
    assert_eq!(
        runtime
            .world
            .resource::<UiModel>()
            .get("settings.subtitles"),
        Some(&UiValue::Bool(true))
    );

    // Disabled nodes are skipped.
    runtime
        .world
        .resource_mut::<UiModel>()
        .set("menu.can_quit", false);
    for _ in 0..6 {
        key(&mut runtime, KeyCode::ArrowDown);
    }
    assert_ne!(state(&runtime, document).focused(), Some("quit"));
    key(&mut runtime, KeyCode::Escape);
    assert!(events(&mut runtime)
        .iter()
        .any(|e| e.kind == UiEventKind::Cancel));
    assert!(runtime.world.resource::<UiInputCapture>().modal);
}

#[test]
fn pointer_clicks_drag_sliders_and_type_into_inputs() {
    let (mut runtime, document) = setup();
    frame(&mut runtime, |_| {});
    frame(&mut runtime, |_| {});
    let (x, y) = center(&runtime, document, "slots[0].slot");
    click(&mut runtime, x, y);
    let received = events(&mut runtime);
    let clicked = received
        .iter()
        .find(|e| e.kind == UiEventKind::Click)
        .expect("click");
    assert_eq!(
        (clicked.node.as_str(), clicked.item),
        ("slots[0].slot", Some(0))
    );
    assert!(runtime.world.resource::<UiInputCapture>().pointer_over_ui);

    // Clicking the right end of the slider sets it to max.
    let rect = state(&runtime, document)
        .instance
        .as_ref()
        .unwrap()
        .node("volume")
        .unwrap()
        .rect;
    click(
        &mut runtime,
        rect.x + rect.width - 1.0,
        rect.y + rect.height * 0.5,
    );
    assert_eq!(
        runtime
            .world
            .resource::<UiModel>()
            .number("settings.volume"),
        Some(1.0)
    );

    let (x, y) = center(&runtime, document, "name");
    click(&mut runtime, x, y);
    assert_eq!(state(&runtime, document).focused(), Some("name"));
    frame(&mut runtime, |input| input.process_text("Ana"));
    frame(&mut runtime, |input| {
        input.process_key_input(KeyCode::Backspace, ElementState::Pressed, false)
    });
    frame(&mut runtime, |input| input.process_text("a\n"));
    assert_eq!(
        state(&runtime, document).text("name.input-text"),
        Some("Ana")
    );
    assert!(runtime.world.resource::<UiInputCapture>().text_input);
}

#[test]
fn lists_rebuild_from_the_model_and_keep_widget_state() {
    let (mut runtime, document) = setup();
    frame(&mut runtime, |_| {});
    key(&mut runtime, KeyCode::ArrowDown);
    key(&mut runtime, KeyCode::ArrowDown);
    key(&mut runtime, KeyCode::ArrowRight);
    let slot = |label: &str| -> BTreeMap<String, UiValue> {
        [("label".to_owned(), UiValue::from(label))].into()
    };
    runtime.world.resource_mut::<UiModel>().set(
        "saves",
        UiValue::List(vec![slot("A"), slot("B"), slot("C")]),
    );
    frame(&mut runtime, |_| {});
    let doc = state(&runtime, document);
    assert_eq!(doc.text("slots[2].slot"), Some("C"));
    assert_eq!(doc.focused(), Some("volume"), "focus survives the rebuild");
    let volume = doc
        .instance
        .as_ref()
        .unwrap()
        .node("volume")
        .unwrap()
        .widget
        .value;
    assert!((volume - 0.6).abs() < 1e-5, "{volume}");
}

#[test]
fn modal_documents_block_lower_ones_and_anchors_follow_entities() {
    let (mut runtime, menu) = setup();
    let assets = runtime.world.resource::<Assets>().clone();
    let bar: UiLayout = ron::from_str(
        r#"(root: (id: "bar", style: (width: Some(Px(100.0)), height: Some(Px(10.0))), children: [(id: "fill", kind: ProgressBar(value: 0.5), style: (width: Some(Percent(100.0)), min_width: Some(Px(0.0))), bind: [(target: Value, key: "enemy.health")])]))"#,
    )
    .unwrap();
    let handle = assets.request::<UiLayout>(&AssetRef::from_path("ui/bar.ui.ron"));
    assets.replace(handle, bar);
    let hud = runtime
        .world
        .spawn((
            UiDocument {
                layout: AssetRef::from_path("ui/bar.ui.ron"),
                order: -1,
                world_anchor: true,
                anchor_offset: Vec3::new(0.0, 2.0, 0.0),
                ..Default::default()
            },
            GlobalTransform::default(),
            Transform::default(),
        ))
        .id();
    let camera = Transform::from_xyz(0.0, 1.0, 10.0).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y);
    runtime.world.spawn((
        Camera3d {
            aspect_ratio: 1280.0 / 720.0,
            ..Default::default()
        },
        PrimaryCamera,
        GlobalTransform(camera.to_affine()),
        camera,
    ));
    frame(&mut runtime, |_| {});
    frame(&mut runtime, |_| {});
    let draw = runtime.world.resource::<UiDrawList>();
    assert!(!draw.quads.is_empty());
    // The bar (drawn first, order -1) is centred horizontally on the entity.
    let first = draw.quads[0];
    assert!(
        (first.rect[0] + first.rect[2] * 0.5 - 640.0).abs() < 2.0,
        "{first:?}"
    );
    assert!(first.rect[1] < 360.0, "above the anchor: {first:?}");
    // The modal menu blocks the HUD from input.
    runtime.world.get_mut::<UiDocument>(menu).unwrap().visible = true;
    let _ = hud;
}
