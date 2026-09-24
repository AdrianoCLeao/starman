//! The shared gameplay runtime assembly (ADR 0012).
//!
//! Every host that simulates a game — the windowed runner, editor
//! play-mode, `starman test`, `engine-smoke` and the benchmarks — builds
//! its [`GameRuntime`] through [`build_runtime`], so the installed
//! subsystems, their ordering and their reflected types are identical
//! everywhere.

pub use engine_core::{GameRuntime, RuntimePlugin};

use engine_core::DEFAULT_FIXED_TIMESTEP_SECONDS;

/// Options for [`build_runtime`].
#[derive(Clone, Debug)]
pub struct RuntimeOptions {
    pub fixed_timestep_seconds: f64,
}

impl Default for RuntimeOptions {
    fn default() -> Self {
        Self {
            fixed_timestep_seconds: DEFAULT_FIXED_TIMESTEP_SECONDS,
        }
    }
}

/// The engine's standard plugin set, in installation order.
pub fn default_plugins() -> Vec<Box<dyn RuntimePlugin>> {
    vec![
        Box::new(engine_input::InputPlugin),
        Box::new(engine_physics::PhysicsPlugin),
    ]
}

/// Installs [`default_plugins`] into `runtime`.
pub fn install_default_plugins(runtime: &mut GameRuntime) {
    runtime.add_plugins(default_plugins());
}

/// Builds a runtime with every standard subsystem installed.
pub fn build_runtime(options: &RuntimeOptions) -> GameRuntime {
    let mut runtime = GameRuntime::with_fixed_timestep(options.fixed_timestep_seconds);
    install_default_plugins(&mut runtime);
    runtime
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_runtime_installs_every_standard_plugin() {
        let runtime = build_runtime(&RuntimeOptions::default());
        for plugin in default_plugins() {
            assert!(runtime.has_plugin(plugin.name()), "{}", plugin.name());
        }
    }

    #[test]
    fn default_runtime_steps_headless() {
        let mut runtime = build_runtime(&RuntimeOptions::default());
        for _ in 0..10 {
            runtime.step(1.0 / 60.0);
        }
        assert_eq!(runtime.frame_count(), 10);
    }
}
