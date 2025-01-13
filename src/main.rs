use eframe::egui;
use eframe::{run_native, NativeOptions};

fn main() {
    let native_options = NativeOptions {
        initial_window_size: Some(egui::vec2(800.0, 600.0)),
        ..Default::default()
    };

    let _ = run_native(
        "3D Engine GUI",
        native_options,
        Box::new(|_cc| Box::new(MyApp::default())),
    );
}

struct MyApp {
    bottom_panel_height: f32,
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            bottom_panel_height: 250.0,
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("top_bar")
            .frame(
                egui::Frame::default()
                    .fill(egui::Color32::from_rgb(15, 15, 15))
                    .inner_margin(egui::Margin::symmetric(4.0, 0.0)),
            )
            .show(ctx, |ui| {
                egui::menu::bar(ui, |ui| {
                    ui.menu_button("File", |ui| {
                        if ui.button("New").clicked() {
                            println!("New");
                        }
                        if ui.button("Open").clicked() {
                            println!("Open");
                        }
                        if ui.button("Exit").clicked() {
                            println!("Exit");
                        }
                    });

                    ui.menu_button("Edit", |ui| {
                        if ui.button("Undo").clicked() {
                            println!("Undo");
                        }
                        if ui.button("Redo").clicked() {
                            println!("Redo");
                        }
                    });
                });
            });

        egui::SidePanel::left("scene_graph")
            .resizable(true)
            .frame(
                egui::Frame::default()
                    .fill(egui::Color32::from_rgb(25, 25, 25))
                    .inner_margin(egui::Margin::symmetric(10.0, 10.0)),
            )
            .show(ctx, |ui| {
                ui.heading("Scene Graph");
                ui.label("Here will be displayed the hierarchy of the scene.");
            });

        egui::SidePanel::right("properties")
            .resizable(true)
            .frame(
                egui::Frame::default()
                    .fill(egui::Color32::from_rgb(25, 25, 25))
                    .inner_margin(egui::Margin::symmetric(10.0, 10.0)),
            )
            .show(ctx, |ui| {
                ui.heading("Properties");
                ui.label("Here we'll have the properties of the selected object.");
            });

        egui::CentralPanel::default()
            .frame(
                egui::Frame::default()
                    .fill(egui::Color32::from_rgb(35, 35, 35))
                    .inner_margin(egui::Margin::symmetric(10.0, 10.0)),
            )
            .show(ctx, |ui| {
                ui.heading("Scene");
                ui.label("This is where the main content will be displayed.");
            });

        egui::TopBottomPanel::bottom("asset_browser")
            .resizable(true)
            .frame(
                egui::Frame::default()
                    .fill(egui::Color32::from_rgb(25, 25, 25))
                    .inner_margin(egui::Margin::symmetric(10.0, 10.0)),
            )
            .min_height(50.0)
            .default_height(self.bottom_panel_height)
            .show(ctx, |ui| {
                ui.vertical(|ui| {
                    ui.label("The assets and directories navigation will be centered here.");
                });
            });
    }
}
