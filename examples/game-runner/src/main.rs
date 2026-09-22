use std::io::{self, BufRead};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};

use engine_assets::AssetDatabase;
use engine_core::{EngineError, Result, WindowConfig};
use engine_project::Project;
use engine_runner::{RunnerControlCommand, RunnerOptions};

fn parse_control_command(line: &str) -> Option<RunnerControlCommand> {
    match line.trim().to_ascii_lowercase().as_str() {
        "pause" => Some(RunnerControlCommand::Pause),
        "resume" => Some(RunnerControlCommand::Resume),
        "stop" => Some(RunnerControlCommand::Stop),
        _ => None,
    }
}

fn spawn_control_channel() -> Receiver<RunnerControlCommand> {
    let (tx, rx) = mpsc::channel::<RunnerControlCommand>();

    if let Err(error) = std::thread::Builder::new()
        .name("runner-control-stdin".to_owned())
        .spawn(move || {
            let stdin = io::stdin();
            let mut reader = io::BufReader::new(stdin.lock());
            let mut line = String::new();

            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => {
                        if let Some(command) = parse_control_command(&line) {
                            if tx.send(command).is_err() {
                                break;
                            }
                        }
                    }
                    Err(read_error) => {
                        log::warn!(
                            target: "engine::runner",
                            "Control channel read error: {}",
                            read_error
                        );
                        break;
                    }
                }
            }
        })
    {
        log::warn!(
            target: "engine::runner",
            "Failed to spawn control channel thread: {}",
            error
        );
    }

    rx
}

fn parse_args() -> Result<(PathBuf, String)> {
    let mut args = std::env::args_os();
    let _binary = args.next();

    let Some(scene_path) = args.next() else {
        return Err(EngineError::Config(
            "usage: game-runner <scene_path> [assets_root]".to_owned(),
        ));
    };

    let assets_root = args
        .next()
        .map(|value| PathBuf::from(value).to_string_lossy().into_owned())
        .unwrap_or_else(|| "examples/reference-project/assets".to_owned());

    Ok((PathBuf::from(scene_path), assets_root))
}

/// Best-effort: if `assets_root` looks like `<project>/assets` and that
/// project has a manifest, opens an `AssetDatabase` for it. This is how
/// `game-runner` gains project awareness (kept `.meta`/cache during hot
/// reload) without changing its argv contract — the Editor's Play Mode
/// depends on that contract staying exactly `<scene_path> [assets_root]`,
/// since it runs an out-of-project scene snapshot through it. Returns
/// `None` silently whenever there is no project to find; that is the
/// normal case for an ad hoc `assets_root` and is not an error.
fn discover_project_database(assets_root: &str) -> Option<AssetDatabase> {
    let project_root = Path::new(assets_root).parent()?;
    let project = Project::open(project_root).ok()?;
    AssetDatabase::open(project.paths.assets_dir(), project.paths.imported_dir()).ok()
}

fn run() -> Result<()> {
    let (scene_path, assets_root) = parse_args()?;
    let control_rx = spawn_control_channel();

    let scene_label = scene_path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("scene");
    let window_title = format!("Motley Play Mode - {}", scene_label);

    let mut options = RunnerOptions::new(window_title.clone())
        .with_window(
            WindowConfig::default()
                .with_title(window_title)
                .with_size(1280, 720)
                .with_resizable(true)
                .with_vsync(true),
        )
        .with_control_rx(control_rx);

    if let Some(database) = discover_project_database(&assets_root) {
        log::info!(
            target: "engine::runner",
            "Attached asset database for project near '{}'",
            assets_root
        );
        options = options.with_database(database);
    }

    engine_runner::run_scene_windowed(&assets_root, &scene_path, options)
}

fn main() {
    let _ = engine_diagnostics::initialize(
        engine_diagnostics::DiagnosticsConfig::for_application("game-runner").json_stdout(true),
    );

    if let Err(error) = run() {
        log::error!(target: "engine::runner", "Startup failed: {}", error);
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_control_command, RunnerControlCommand};

    #[test]
    fn parse_control_command_is_case_insensitive_and_trimmed() {
        assert_eq!(
            parse_control_command(" pause\n"),
            Some(RunnerControlCommand::Pause)
        );
        assert_eq!(
            parse_control_command("RESUME"),
            Some(RunnerControlCommand::Resume)
        );
        assert_eq!(
            parse_control_command(" stop  "),
            Some(RunnerControlCommand::Stop)
        );
    }

    #[test]
    fn parse_control_command_rejects_unknown_values() {
        assert_eq!(parse_control_command(""), None);
        assert_eq!(parse_control_command("play"), None);
        assert_eq!(parse_control_command("foobar"), None);
    }
}
