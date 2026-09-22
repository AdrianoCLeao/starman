use std::path::PathBuf;

use engine_editor::EditorApp;

/// Defaults to the Starman reference project so `cargo run -p engine-editor`
/// from the repo root keeps working with no arguments, matching today's
/// behavior; pass a path explicitly to open a different project. There is
/// no "open project" dialog yet (M7 scope) — an invalid path fails fast.
fn resolve_project_path() -> PathBuf {
    std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("examples/reference-project"))
}

fn main() {
    let _ = engine_diagnostics::initialize(engine_diagnostics::DiagnosticsConfig::for_application(
        "starman-editor",
    ));

    let project_path = resolve_project_path();

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Starman Editor")
            .with_inner_size([1600.0, 900.0])
            .with_min_inner_size([800.0, 600.0]),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };

    if let Err(error) = eframe::run_native(
        "Starman Editor",
        native_options,
        Box::new(move |cc| {
            EditorApp::try_new(cc, project_path)
                .map(|app| Box::new(app) as Box<dyn eframe::App>)
                .map_err(|error| Box::new(error) as Box<dyn std::error::Error + Send + Sync>)
        }),
    ) {
        log::error!(target: "engine::editor", "Failed to start editor: {}", error);
        std::process::exit(1);
    }
}
