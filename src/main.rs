use eframe::egui;
use eframe::{run_native, NativeOptions};
use std::fs;
use std::path::PathBuf;

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

enum ViewMode {
    List,
    Icons,
}

struct MyApp {
    current_dir: PathBuf,
    bottom_panel_height: f32,
    view_mode: ViewMode,
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            current_dir: std::env::current_dir().unwrap(),
            bottom_panel_height: 150.0,
            view_mode: ViewMode::List,
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
                ui.heading("Asset Browser");
                ui.separator();

                ui.horizontal(|ui| {
                    ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                        if let Some(parent_dir) = self.current_dir.parent() {
                            if ui.button("⬅️ Back").clicked() {
                                self.current_dir = parent_dir.to_path_buf();
                            }
                        }

                        if ui
                            .button(match self.view_mode {
                                ViewMode::List => "Switch to Icons",
                                ViewMode::Icons => "Switch to List",
                            })
                            .clicked()
                        {
                            self.view_mode = match self.view_mode {
                                ViewMode::List => ViewMode::Icons,
                                ViewMode::Icons => ViewMode::List,
                            };
                        }
                    });
                });

                ui.separator();

                if let Ok(entries) = fs::read_dir(&self.current_dir) {
                    let entries: Vec<_> = entries
                        .flatten()
                        .filter(|entry| {
                            let binding = entry.file_name();
                            let file_name = binding.to_string_lossy();
                            !file_name.starts_with('.') 
                                && file_name != "target"
                        })
                        .collect();

                    match self.view_mode {
                        ViewMode::List => {
                            for entry in entries {
                                let path = entry.path();
                                let name = entry.file_name().to_string_lossy().to_string();

                                if path.is_dir() {
                                    if ui.button(format!("📁 {}", name)).clicked() {
                                        self.current_dir = path;
                                    }
                                } else {
                                    ui.label(format!("📄 {}", name));
                                }
                            }
                        }
                        ViewMode::Icons => {
                            ui.horizontal_wrapped(|ui| {
                                for entry in entries {
                                    let path = entry.path();
                                    let name = entry.file_name().to_string_lossy().to_string();

                                    let label = if path.is_dir() {
                                        format!("📁 {}", name)
                                    } else {
                                        format!("📄 {}", name)
                                    };

                                    if ui.button(&label).clicked() {
                                        if path.is_dir() {
                                            self.current_dir = path;
                                        } else {
                                            if let Err(err) = open::that(&path) {
                                                eprintln!("Failed to open file: {}", err);
                                            }
                                        }
                                    }
                                }
                            });
                        }
                    }
                }
            });
    }
}
