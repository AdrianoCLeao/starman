use crate::{EngineError, Result};
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{DeviceEvent, DeviceId, ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowAttributes, WindowId},
};

use crate::app_control::{CursorGrab, CursorState};

const DEFAULT_VSYNC_FALLBACK_REFRESH_RATE_MILLIHZ: u32 = 60_000;
const ESCAPE_CONFIRM_WINDOW: Duration = Duration::from_secs(2);

fn default_frame_interval() -> Duration {
    Duration::from_secs_f64(1000.0 / DEFAULT_VSYNC_FALLBACK_REFRESH_RATE_MILLIHZ as f64)
}

pub trait WindowLoop {
    fn window_created(&mut self, _window: Arc<Window>, _config: &WindowConfig) -> Result<()> {
        Ok(())
    }

    fn tick(&mut self) -> Result<()>;

    fn window_event(&mut self, _event: &WindowEvent) -> Result<()> {
        Ok(())
    }

    fn resized(&mut self, _width: u32, _height: u32) -> Result<()> {
        Ok(())
    }

    fn title(&self) -> String {
        "Starman".to_owned()
    }

    /// Raw device input (mouse motion while the cursor is locked, …).
    fn device_event(&mut self, _event: &DeviceEvent) -> Result<()> {
        Ok(())
    }

    /// Whether the application asked to quit (e.g. a "Quit" menu entry).
    fn wants_exit(&self) -> bool {
        false
    }

    /// Desired cursor state; applied to the window when it changes.
    fn cursor(&self) -> Option<CursorState> {
        None
    }
}

#[derive(Debug, Clone)]
pub struct WindowConfig {
    pub title: String,
    pub width: u32,
    pub height: u32,
    pub resizable: bool,
    pub vsync: bool,
    /// Pressing Escape twice within two seconds quits. Convenient for tools
    /// and samples; games that use Escape (pause menus) turn it off and
    /// quit through [`crate::AppControl`].
    pub escape_to_exit: bool,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            title: "Starman".to_owned(),
            width: 1280,
            height: 720,
            resizable: true,
            vsync: true,
            escape_to_exit: true,
        }
    }
}

impl WindowConfig {
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    pub fn with_resizable(mut self, resizable: bool) -> Self {
        self.resizable = resizable;
        self
    }

    pub fn with_vsync(mut self, vsync: bool) -> Self {
        self.vsync = vsync;
        self
    }

    pub fn with_escape_to_exit(mut self, escape_to_exit: bool) -> Self {
        self.escape_to_exit = escape_to_exit;
        self
    }
}

pub fn run_windowed<A>(config: WindowConfig, app: A) -> Result<()>
where
    A: WindowLoop + 'static,
{
    let initial_control_flow = if config.vsync {
        ControlFlow::Wait
    } else {
        ControlFlow::Poll
    };

    let event_loop = EventLoop::new()
        .map_err(|error| EngineError::Window(format!("failed to create event loop: {error}")))?;

    event_loop.set_control_flow(initial_control_flow);

    let mut runner = WindowRunner::new(config, app);
    let event_loop_result = event_loop.run_app(&mut runner);

    if let Some(error) = runner.take_error() {
        return Err(error);
    }

    event_loop_result
        .map_err(|error| EngineError::Window(format!("event loop exited with error: {error}")))
}

struct WindowRunner<A: WindowLoop> {
    config: WindowConfig,
    app: A,
    window: Option<Arc<Window>>,
    window_id: Option<WindowId>,
    frame_interval: Option<Duration>,
    next_redraw_deadline: Option<Instant>,
    last_escape_press: Option<Instant>,
    error: Option<EngineError>,
    applied_cursor: Option<CursorState>,
}

impl<A: WindowLoop> WindowRunner<A> {
    fn new(config: WindowConfig, app: A) -> Self {
        Self {
            frame_interval: frame_interval_from_refresh_rate(config.vsync, None),
            config,
            app,
            window: None,
            window_id: None,
            next_redraw_deadline: None,
            last_escape_press: None,
            error: None,
            applied_cursor: None,
        }
    }

    fn maybe_confirm_escape_exit(
        &mut self,
        event_loop: &ActiveEventLoop,
        event: &WindowEvent,
    ) -> bool {
        let WindowEvent::KeyboardInput { event, .. } = event else {
            return false;
        };

        if event.state != ElementState::Pressed || event.repeat {
            return false;
        }

        if !self.config.escape_to_exit
            || !matches!(event.physical_key, PhysicalKey::Code(KeyCode::Escape))
        {
            return false;
        }

        let now = Instant::now();
        if let Some(last_press) = self.last_escape_press {
            if now.duration_since(last_press) <= ESCAPE_CONFIRM_WINDOW {
                log::info!(target: "engine::window", "Escape confirmation received, exiting application");
                event_loop.exit();
                self.last_escape_press = None;
                return true;
            }
        }

        self.last_escape_press = Some(now);
        log::warn!(
            target: "engine::window",
            "Press Escape again within 2 seconds to exit"
        );
        true
    }

    fn take_error(&mut self) -> Option<EngineError> {
        self.error.take()
    }

    fn fail(&mut self, event_loop: &ActiveEventLoop, error: EngineError) {
        self.error = Some(error);
        event_loop.exit();
    }
}

impl<A: WindowLoop> ApplicationHandler for WindowRunner<A> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attributes = WindowAttributes::default()
            .with_title(self.config.title.clone())
            .with_inner_size(LogicalSize::new(self.config.width, self.config.height))
            .with_resizable(self.config.resizable);

        match event_loop.create_window(attributes) {
            Ok(window) => {
                let window = Arc::new(window);
                self.frame_interval = frame_interval_from_refresh_rate(
                    self.config.vsync,
                    window
                        .current_monitor()
                        .and_then(|monitor| monitor.refresh_rate_millihertz()),
                );
                self.window_id = Some(window.id());
                self.next_redraw_deadline = Some(Instant::now());

                if let Err(error) = self.app.window_created(window.clone(), &self.config) {
                    self.fail(event_loop, error);
                    return;
                }

                if !self.config.vsync {
                    event_loop.set_control_flow(ControlFlow::Poll);
                }

                window.request_redraw();
                self.window = Some(window);
            }
            Err(error) => {
                self.fail(
                    event_loop,
                    EngineError::Window(format!("failed to create window: {error}")),
                );
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if self.window_id != Some(window_id) {
            return;
        }

        if let Err(error) = self.app.window_event(&event) {
            self.fail(event_loop, error);
            return;
        }

        if self.maybe_confirm_escape_exit(event_loop, &event) {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Err(error) = self.app.resized(size.width, size.height) {
                    self.fail(event_loop, error);
                }
            }
            WindowEvent::RedrawRequested => {
                if let Err(error) = self.app.tick() {
                    self.fail(event_loop, error);
                    return;
                }

                if self.app.wants_exit() {
                    log::info!(target: "engine::window", "application requested exit");
                    event_loop.exit();
                    return;
                }
                if let Some(window) = self.window.as_ref() {
                    window.set_title(&self.app.title());
                    let desired = self.app.cursor();
                    if let Some(state) = desired.filter(|_| desired != self.applied_cursor) {
                        apply_cursor(window, state);
                        self.applied_cursor = desired;
                    }
                }
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let Err(error) = self.app.device_event(&event) {
            self.fail(event_loop, error);
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if !self.config.vsync {
            event_loop.set_control_flow(ControlFlow::Poll);
            if let Some(window) = self.window.as_ref() {
                window.request_redraw();
            }
            return;
        }

        let Some(window) = self.window.as_ref() else {
            return;
        };

        let frame_interval = self.frame_interval.unwrap_or_else(|| {
            debug_assert!(
                false,
                "vsync enabled but frame_interval was not initialized"
            );
            default_frame_interval()
        });

        let now = Instant::now();
        let deadline = self.next_redraw_deadline.unwrap_or(now);

        if now >= deadline {
            window.request_redraw();
            let next_deadline = now + frame_interval;
            self.next_redraw_deadline = Some(next_deadline);
            event_loop.set_control_flow(ControlFlow::WaitUntil(next_deadline));
            return;
        }

        event_loop.set_control_flow(ControlFlow::WaitUntil(deadline));
    }
}

fn apply_cursor(window: &Window, state: CursorState) {
    window.set_cursor_visible(state.visible);
    let modes: &[CursorGrabMode] = match state.grab {
        CursorGrab::None => &[CursorGrabMode::None],
        // Locked is not supported everywhere (X11); fall back to confined.
        CursorGrab::Locked => &[CursorGrabMode::Locked, CursorGrabMode::Confined],
        CursorGrab::Confined => &[CursorGrabMode::Confined, CursorGrabMode::Locked],
    };
    for mode in modes {
        if window.set_cursor_grab(*mode).is_ok() {
            return;
        }
    }
    log::warn!(target: "engine::window", "cursor grab {:?} is not supported here", state.grab);
}

fn frame_interval_from_refresh_rate(
    vsync_enabled: bool,
    refresh_rate_millihertz: Option<u32>,
) -> Option<Duration> {
    if !vsync_enabled {
        return None;
    }

    let millihertz = refresh_rate_millihertz
        .filter(|rate| *rate > 0)
        .unwrap_or(DEFAULT_VSYNC_FALLBACK_REFRESH_RATE_MILLIHZ);

    Some(Duration::from_secs_f64(1000.0 / millihertz as f64))
}

#[cfg(test)]
#[path = "window_tests.rs"]
mod tests;
