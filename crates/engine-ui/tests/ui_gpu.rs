//! The UI render node under GPU validation, on the final target.

use engine_assets::{AssetRef, AssetServer};
use engine_core::{Camera3d, GameRuntime, GlobalTransform, PrimaryCamera, Transform, WindowSize};
use engine_localization::Localization;
use engine_math::Vec3;
use engine_render::{FrameRenderer, FrameRendererConfig, QualityPreset};
use engine_ui::*;

const WIDTH: u32 = 320;
const HEIGHT: u32 = 180;

fn renderer() -> Option<FrameRenderer> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))?;
    let tier = engine_render::negotiate_tier(&adapter);
    let info = adapter.get_info();
    let (device, queue) =
        pollster::block_on(adapter.request_device(&Default::default(), None)).ok()?;
    let caps = engine_render::NegotiatedCapabilities {
        tier,
        adapter_name: info.name,
        backend: format!("{:?}", info.backend),
        features: adapter.features(),
        limits: adapter.limits(),
    };
    Some(
        FrameRenderer::try_new(
            device,
            queue,
            caps,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            WIDTH,
            HEIGHT,
            FrameRendererConfig {
                quality: QualityPreset::Low,
                shader_cache: std::env::temp_dir().join("starman-ui-test-shader-cache"),
                ..Default::default()
            },
        )
        .expect("renderer"),
    )
}

#[test]
fn menu_renders_text_boxes_and_images_without_validation_errors() {
    let Some(mut frame) = renderer() else {
        eprintln!("skipping GPU test: no adapter");
        return;
    };
    let root = std::env::temp_dir().join(format!(
        "starman-ui-gpu-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(root.join("ui")).unwrap();
    let reference = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/reference-project/assets");
    std::fs::copy(
        reference.join("textures/placeholder.png"),
        root.join("ui/panel.png"),
    )
    .unwrap();
    std::fs::write(
        root.join("ui/menu.ui.ron"),
        r##"(
            rules: [(selector: "#menu", style: (direction: Some(Column), width: Some(Percent(100.0)), height: Some(Percent(100.0)), justify_content: Some(Center), align_items: Some(Center), gap: Some((12.0, 12.0)), background: Some((0.02, 0.02, 0.05, 0.6))))],
            root: (id: "menu", children: [
                (id: "title", kind: Text(text: Literal("Pausado — ação!")), style: (font_size: Some(48.0), bold: Some(true))),
                (id: "resume", kind: Button(text: Some(Literal("Continuar")))),
                (id: "volume", kind: Slider(value: 0.7)),
                (id: "health", kind: ProgressBar(value: 0.4)),
                (id: "icon", kind: Image(image: (path: "ui/panel.png"), nine_slice: Some((4.0, 4.0, 4.0, 4.0))), style: (width: Some(Px(160.0)), height: Some(Px(60.0)))),
            ]),
        )"##,
    )
    .unwrap();
    let mut server = AssetServer::new(root.to_string_lossy().to_string());
    let mut runtime = GameRuntime::new();
    runtime.insert_resource(server.assets().clone());
    runtime.insert_resource(WindowSize {
        width: WIDTH,
        height: HEIGHT,
    });
    runtime.add_plugin(engine_assets::AssetsPlugin);
    runtime.add_plugin(engine_input::InputPlugin);
    runtime.add_plugin(UiPlugin);
    runtime.insert_resource(Localization::default());
    let camera = Transform::from_xyz(0.0, 1.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y);
    runtime.world.spawn((
        Camera3d::default(),
        PrimaryCamera,
        GlobalTransform(camera.to_affine()),
        camera,
    ));
    let document = runtime
        .world
        .spawn(UiDocument {
            layout: AssetRef::from_path("ui/menu.ui.ron"),
            ..Default::default()
        })
        .id();
    runtime
        .world
        .get_mut::<UiDocument>(document)
        .unwrap()
        .visible = false;

    let mut render = |runtime: &mut GameRuntime, server: &mut AssetServer| {
        for _ in 0..3 {
            server.update_blocking();
            runtime.step(1.0 / 60.0);
        }
        frame.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let image = frame
            .render_to_image(&mut runtime.world, server)
            .expect("frame");
        let error = pollster::block_on(frame.device.pop_error_scope());
        assert!(error.is_none(), "GPU validation error: {error:?}");
        image
    };
    let hidden = render(&mut runtime, &mut server);
    runtime
        .world
        .get_mut::<UiDocument>(document)
        .unwrap()
        .visible = true;
    let image = render(&mut runtime, &mut server);
    if let Ok(path) = std::env::var("STARMAN_UI_DUMP") {
        image::save_buffer(&path, &image, WIDTH, HEIGHT, image::ColorType::Rgba8).unwrap();
    }
    let stats = frame.stats().clone();
    let stat = |name: &str| {
        stats
            .extensions
            .iter()
            .find(|(n, _)| n.ends_with(name))
            .map_or(-1.0, |(_, v)| *v)
    };
    assert!(stat("ui_quads") > 30.0, "{:?}", stats.extensions);
    assert!(
        stat("ui_draws") >= 3.0,
        "solid, glyph and image batches: {:?}",
        stats.extensions
    );
    // Bright glyph pixels appear in the title band; the overlay darkens.
    let luminance = |p: &[u8]| (p[0] as u32 + p[1] as u32 + p[2] as u32) / 3;
    let bright = image
        .chunks_exact(4)
        .enumerate()
        .filter(|(i, p)| (i / WIDTH as usize) < (HEIGHT as usize / 2) && luminance(p) > 200)
        .count();
    assert!(bright > 40, "title text is visible: {bright}");
    let darker = hidden
        .chunks_exact(4)
        .zip(image.chunks_exact(4))
        .filter(|(a, b)| luminance(b) + 5 < luminance(a))
        .count();
    assert!(
        darker > (WIDTH * HEIGHT / 2) as usize,
        "the menu overlay darkens the scene: {darker}"
    );
    let _ = std::fs::remove_dir_all(&root);
}
