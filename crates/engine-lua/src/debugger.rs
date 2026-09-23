//! Minimal Lua debugger: breakpoints, step, continue.

use std::collections::HashSet;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DebuggerCommand {
    Continue,
    Step,
    Break { file: PathBuf, line: u32 },
    ClearBreak { file: PathBuf, line: u32 },
    ClearAll,
}

#[derive(Debug, Default)]
pub struct LuaDebugger {
    pub breakpoints: HashSet<(PathBuf, u32)>,
    pub paused: bool,
    pub step_once: bool,
    pub last_file: Option<PathBuf>,
    pub last_line: Option<u32>,
}

impl LuaDebugger {
    pub fn apply(&mut self, command: DebuggerCommand) {
        match command {
            DebuggerCommand::Continue => {
                self.paused = false;
                self.step_once = false;
            }
            DebuggerCommand::Step => {
                self.paused = false;
                self.step_once = true;
            }
            DebuggerCommand::Break { file, line } => {
                self.breakpoints.insert((file, line));
            }
            DebuggerCommand::ClearBreak { file, line } => {
                self.breakpoints.remove(&(file, line));
            }
            DebuggerCommand::ClearAll => {
                self.breakpoints.clear();
            }
        }
    }

    /// Called from the Lua debug hook; returns true if execution should pause.
    pub fn should_pause(&mut self, file: &str, line: u32) -> bool {
        let path = PathBuf::from(file);
        self.last_file = Some(path.clone());
        self.last_line = Some(line);
        if self.step_once {
            self.step_once = false;
            self.paused = true;
            return true;
        }
        if self.breakpoints.contains(&(path, line)) {
            self.paused = true;
            return true;
        }
        self.paused
    }
}
