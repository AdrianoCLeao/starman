use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::panic::PanicHookInfo;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::sync::{Arc, Mutex, Once, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::field::{Field, Visit};
use tracing::{Event, Level, Subscriber};
use tracing_log::LogTracer;
use tracing_subscriber::layer::Context;
use tracing_subscriber::prelude::*;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::Layer;

const DEFAULT_RING_CAPACITY: usize = 4096;
const DEFAULT_FILE_SIZE_LIMIT: u64 = 10 * 1024 * 1024;
const DEFAULT_RETAINED_FILES: usize = 5;
const DEFAULT_LEVEL_FILTER: &str = "info";

static GLOBAL_STATE: OnceLock<Arc<DiagnosticsState>> = OnceLock::new();
static PANIC_HOOK_INIT: Once = Once::new();

#[derive(Debug, Clone)]
pub struct DiagnosticsConfig {
    pub application: String,
    pub log_directory: PathBuf,
    pub level_filter: String,
    pub ring_capacity: usize,
    pub file_size_limit: u64,
    pub retained_files: usize,
    pub terminal: bool,
    pub file: bool,
    pub stdout_json: bool,
    pub install_panic_hook: bool,
}

impl DiagnosticsConfig {
    pub fn for_application(application: impl Into<String>) -> Self {
        let application = application.into();
        Self {
            log_directory: default_log_directory(&application),
            application,
            ..Self::default()
        }
    }

    pub fn with_log_directory(mut self, log_directory: impl Into<PathBuf>) -> Self {
        self.log_directory = log_directory.into();
        self
    }

    pub fn json_stdout(mut self, enabled: bool) -> Self {
        self.stdout_json = enabled;
        if enabled {
            self.terminal = false;
        }
        self
    }
}

impl Default for DiagnosticsConfig {
    fn default() -> Self {
        Self {
            application: "starman".to_owned(),
            log_directory: default_log_directory("starman"),
            level_filter: DEFAULT_LEVEL_FILTER.to_owned(),
            ring_capacity: DEFAULT_RING_CAPACITY,
            file_size_limit: DEFAULT_FILE_SIZE_LIMIT,
            retained_files: DEFAULT_RETAINED_FILES,
            terminal: true,
            file: true,
            stdout_json: false,
            install_panic_hook: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum DiagnosticLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl DiagnosticLevel {
    fn from_tracing(level: &Level) -> Self {
        match *level {
            Level::TRACE => Self::Trace,
            Level::DEBUG => Self::Debug,
            Level::INFO => Self::Info,
            Level::WARN => Self::Warn,
            Level::ERROR => Self::Error,
        }
    }
}

impl fmt::Display for DiagnosticLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Trace => "trace",
            Self::Debug => "debug",
            Self::Info => "info",
            Self::Warn => "warn",
            Self::Error => "error",
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DiagnosticEvent {
    pub sequence: u64,
    pub timestamp_millis: u128,
    pub level: DiagnosticLevel,
    pub target: String,
    pub message: String,
    pub fields: BTreeMap<String, String>,
    pub thread: Option<String>,
}

pub struct DiagnosticsHandle {
    state: Arc<DiagnosticsState>,
}

impl DiagnosticsHandle {
    pub fn recent_events_after(&self, cursor: u64) -> RecentDiagnosticEvents {
        self.state.recent_events_after(cursor)
    }

    pub fn cursor(&self) -> u64 {
        self.state.sequence.load(Ordering::Relaxed)
    }
}

#[derive(Clone, Debug, Default)]
pub struct RecentDiagnosticEvents {
    pub events: Vec<DiagnosticEvent>,
    pub next_cursor: u64,
    pub dropped_events: u64,
}

#[derive(Debug, Error)]
pub enum DiagnosticsInitError {
    #[error("diagnostics has already been initialized")]
    AlreadyInitialized,
    #[error("failed to create diagnostics directory '{path}': {source}")]
    CreateDirectory { path: PathBuf, source: io::Error },
    #[error("failed to initialize tracing subscriber: {0}")]
    Subscriber(String),
}

struct DiagnosticsState {
    ring: Mutex<EventRing>,
    writer: Option<SyncSender<DiagnosticEvent>>,
    writer_dropped: AtomicU64,
    sequence: AtomicU64,
    home_dir: Option<PathBuf>,
}

impl DiagnosticsState {
    fn record(&self, mut event: DiagnosticEvent) {
        event.sequence = self.sequence.fetch_add(1, Ordering::Relaxed) + 1;
        event.message = self.sanitize(&event.message);
        event.target = self.sanitize(&event.target);
        event.fields = event
            .fields
            .into_iter()
            .map(|(key, value)| (key, self.sanitize(&value)))
            .collect();

        if let Ok(mut ring) = self.ring.lock() {
            ring.push(event.clone());
        }

        if let Some(writer) = &self.writer {
            match writer.try_send(event) {
                Ok(()) => {}
                Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                    self.writer_dropped.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }

    fn recent_events_after(&self, cursor: u64) -> RecentDiagnosticEvents {
        let Ok(ring) = self.ring.lock() else {
            return RecentDiagnosticEvents {
                next_cursor: self.sequence.load(Ordering::Relaxed),
                ..RecentDiagnosticEvents::default()
            };
        };

        let events = ring
            .events
            .iter()
            .filter(|event| event.sequence > cursor)
            .cloned()
            .collect::<Vec<_>>();

        RecentDiagnosticEvents {
            events,
            next_cursor: self.sequence.load(Ordering::Relaxed),
            dropped_events: ring.dropped + self.writer_dropped.load(Ordering::Relaxed),
        }
    }

    fn sanitize(&self, value: &str) -> String {
        let Some(home_dir) = &self.home_dir else {
            return value.to_owned();
        };

        let home = home_dir.to_string_lossy();
        if home.is_empty() {
            value.to_owned()
        } else {
            value.replace(home.as_ref(), "<home>")
        }
    }
}

struct EventRing {
    events: VecDeque<DiagnosticEvent>,
    capacity: usize,
    dropped: u64,
}

impl EventRing {
    fn new(capacity: usize) -> Self {
        Self {
            events: VecDeque::with_capacity(capacity),
            capacity,
            dropped: 0,
        }
    }

    fn push(&mut self, event: DiagnosticEvent) {
        if self.capacity == 0 {
            self.dropped += 1;
            return;
        }

        if self.events.len() == self.capacity {
            self.events.pop_front();
            self.dropped += 1;
        }

        self.events.push_back(event);
    }
}

struct FileSink {
    application: String,
    directory: PathBuf,
    size_limit: u64,
    retained_files: usize,
    current_file: File,
    current_path: PathBuf,
    bytes_written: u64,
    rotation_sequence: u64,
}

impl FileSink {
    fn new(
        application: &str,
        directory: PathBuf,
        size_limit: u64,
        retained_files: usize,
    ) -> io::Result<Self> {
        fs::create_dir_all(&directory)?;
        let current_path = directory.join(format!("{application}.jsonl"));
        let current_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&current_path)?;
        let bytes_written = current_file
            .metadata()
            .map(|metadata| metadata.len())
            .unwrap_or(0);

        Ok(Self {
            application: application.to_owned(),
            directory,
            size_limit: size_limit.max(1024),
            retained_files: retained_files.max(1),
            current_file,
            current_path,
            bytes_written,
            rotation_sequence: 0,
        })
    }

    fn write_event(&mut self, event: &DiagnosticEvent) -> io::Result<()> {
        let line = serde_json::to_string(event).map_err(io::Error::other)?;
        if self.bytes_written + line.len() as u64 + 1 > self.size_limit {
            self.rotate()?;
        }

        writeln!(self.current_file, "{line}")?;
        self.bytes_written += line.len() as u64 + 1;
        Ok(())
    }

    fn rotate(&mut self) -> io::Result<()> {
        self.current_file.flush()?;
        self.rotation_sequence += 1;
        let timestamp = unix_millis();
        let rotated_path = self.directory.join(format!(
            "{}-{timestamp}-{}.jsonl",
            self.application, self.rotation_sequence
        ));
        fs::rename(&self.current_path, rotated_path)?;
        self.current_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.current_path)?;
        self.bytes_written = 0;
        self.prune_old_files()
    }

    fn prune_old_files(&self) -> io::Result<()> {
        let mut entries = fs::read_dir(&self.directory)?
            .filter_map(Result::ok)
            .filter(|entry| {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                name.starts_with(&format!("{}-", self.application)) && name.ends_with(".jsonl")
            })
            .collect::<Vec<_>>();

        entries.sort_by_key(|entry| entry.file_name());

        let excess = entries.len().saturating_sub(self.retained_files);
        for entry in entries.into_iter().take(excess) {
            let _ = fs::remove_file(entry.path());
        }

        Ok(())
    }
}

fn spawn_writer(
    mut file_sink: Option<FileSink>,
    terminal: bool,
    stdout_json: bool,
    capacity: usize,
) -> Option<SyncSender<DiagnosticEvent>> {
    if !terminal && !stdout_json && file_sink.is_none() {
        return None;
    }

    let (sender, receiver) = mpsc::sync_channel::<DiagnosticEvent>(capacity.max(1));
    let _ = std::thread::Builder::new()
        .name("engine-diagnostics-writer".to_owned())
        .spawn(move || {
            while let Ok(event) = receiver.recv() {
                if terminal {
                    let _ = writeln!(
                        io::stderr(),
                        "[{}] [{}] {}",
                        event.level,
                        event.target,
                        event.message
                    );
                }

                if stdout_json {
                    if let Ok(line) = serde_json::to_string(&event) {
                        let _ = writeln!(io::stdout(), "{line}");
                    }
                }

                if let Some(sink) = file_sink.as_mut() {
                    let _ = sink.write_event(&event);
                }
            }
        });

    Some(sender)
}

struct DiagnosticsLayer {
    state: Arc<DiagnosticsState>,
}

impl<S> Layer<S> for DiagnosticsLayer
where
    S: Subscriber,
{
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let metadata = event.metadata();
        let mut visitor = EventVisitor::default();
        event.record(&mut visitor);

        let message = visitor
            .message
            .clone()
            .unwrap_or_else(|| visitor.fields_as_message());

        self.state.record(DiagnosticEvent {
            sequence: 0,
            timestamp_millis: unix_millis(),
            level: DiagnosticLevel::from_tracing(metadata.level()),
            target: metadata.target().to_owned(),
            message,
            fields: visitor.fields,
            thread: std::thread::current().name().map(str::to_owned),
        });
    }
}

#[derive(Default)]
struct EventVisitor {
    message: Option<String>,
    fields: BTreeMap<String, String>,
}

impl EventVisitor {
    fn fields_as_message(&self) -> String {
        self.fields
            .iter()
            .map(|(key, value)| format!("{key}={value}"))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

impl Visit for EventVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        let value = format!("{value:?}");
        if field.name() == "message" {
            self.message = Some(trim_debug_string(&value));
        } else {
            self.fields
                .insert(field.name().to_owned(), trim_debug_string(&value));
        }
    }
}

pub fn initialize(config: DiagnosticsConfig) -> Result<DiagnosticsHandle, DiagnosticsInitError> {
    if GLOBAL_STATE.get().is_some() {
        return Err(DiagnosticsInitError::AlreadyInitialized);
    }

    let application = normalize_application_name(&config.application);
    fs::create_dir_all(&config.log_directory).map_err(|source| {
        DiagnosticsInitError::CreateDirectory {
            path: config.log_directory.clone(),
            source,
        }
    })?;

    let file_sink = if config.file {
        Some(
            FileSink::new(
                &application,
                config.log_directory.clone(),
                config.file_size_limit,
                config.retained_files,
            )
            .map_err(|source| DiagnosticsInitError::CreateDirectory {
                path: config.log_directory.clone(),
                source,
            })?,
        )
    } else {
        None
    };

    let writer = spawn_writer(
        file_sink,
        config.terminal,
        config.stdout_json,
        config.ring_capacity,
    );

    let env_filter = std::env::var("STARMAN_LOG")
        .or_else(|_| std::env::var("RUST_LOG"))
        .unwrap_or_else(|_| config.level_filter.clone());
    let env_filter =
        EnvFilter::try_new(env_filter).unwrap_or_else(|_| EnvFilter::new(DEFAULT_LEVEL_FILTER));

    let state = Arc::new(DiagnosticsState {
        ring: Mutex::new(EventRing::new(config.ring_capacity)),
        writer,
        writer_dropped: AtomicU64::new(0),
        sequence: AtomicU64::new(0),
        home_dir: std::env::var_os("HOME").map(PathBuf::from),
    });

    let layer = DiagnosticsLayer {
        state: state.clone(),
    };

    LogTracer::init().map_err(|error| DiagnosticsInitError::Subscriber(error.to_string()))?;
    log::set_max_level(log::LevelFilter::Trace);

    tracing_subscriber::registry()
        .with(env_filter)
        .with(layer)
        .try_init()
        .map_err(|error| DiagnosticsInitError::Subscriber(error.to_string()))?;

    GLOBAL_STATE
        .set(state.clone())
        .map_err(|_| DiagnosticsInitError::AlreadyInitialized)?;

    if config.install_panic_hook {
        install_panic_hook_once(application, config.log_directory, config.retained_files);
    }

    Ok(DiagnosticsHandle { state })
}

pub fn global_handle() -> Option<DiagnosticsHandle> {
    GLOBAL_STATE
        .get()
        .cloned()
        .map(|state| DiagnosticsHandle { state })
}

pub fn recent_events_after(cursor: u64) -> RecentDiagnosticEvents {
    global_handle()
        .map(|handle| handle.recent_events_after(cursor))
        .unwrap_or_default()
}

pub fn current_cursor() -> u64 {
    global_handle().map(|handle| handle.cursor()).unwrap_or(0)
}

pub fn event_from_json_line(line: &str) -> serde_json::Result<DiagnosticEvent> {
    serde_json::from_str(line)
}

pub fn event_to_json_line(event: &DiagnosticEvent) -> serde_json::Result<String> {
    serde_json::to_string(event)
}

pub fn write_panic_report(
    application: &str,
    log_directory: &Path,
    panic_message: &str,
    location: Option<&str>,
) -> io::Result<PathBuf> {
    let crash_dir =
        log_directory
            .join("crashes")
            .join(format!("{}-{}", unix_millis(), std::process::id()));
    fs::create_dir_all(&crash_dir)?;

    let report = serde_json::json!({
        "schema": 1,
        "application": normalize_application_name(application),
        "timestamp_millis": unix_millis(),
        "process_id": std::process::id(),
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "panic": {
            "message": panic_message,
            "location": location,
        },
        "backtrace": std::backtrace::Backtrace::force_capture().to_string(),
    });

    fs::write(
        crash_dir.join("report.json"),
        serde_json::to_vec_pretty(&report).map_err(io::Error::other)?,
    )?;

    if let Some(handle) = global_handle() {
        let recent = handle.recent_events_after(0);
        let mut file = File::create(crash_dir.join("recent-events.jsonl"))?;
        for event in recent.events {
            writeln!(
                file,
                "{}",
                serde_json::to_string(&event).map_err(io::Error::other)?
            )?;
        }
    }

    Ok(crash_dir)
}

fn install_panic_hook_once(application: String, log_directory: PathBuf, retained_crashes: usize) {
    PANIC_HOOK_INIT.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let message = panic_message(info);
            let location = info.location().map(|location| {
                format!(
                    "{}:{}:{}",
                    location.file(),
                    location.line(),
                    location.column()
                )
            });
            let _ = write_panic_report(&application, &log_directory, &message, location.as_deref());
            let _ = prune_crash_directories(&log_directory.join("crashes"), retained_crashes);
            previous(info);
        }));
    });
}

fn prune_crash_directories(crashes_dir: &Path, retained_crashes: usize) -> io::Result<()> {
    let mut entries = fs::read_dir(crashes_dir)?
        .filter_map(Result::ok)
        .collect::<Vec<_>>();
    entries.sort_by_key(|entry| entry.file_name());

    let excess = entries.len().saturating_sub(retained_crashes.max(1));
    for entry in entries.into_iter().take(excess) {
        if entry
            .file_type()
            .map(|file_type| file_type.is_dir())
            .unwrap_or(false)
        {
            let _ = fs::remove_dir_all(entry.path());
        }
    }

    Ok(())
}

fn panic_message(info: &PanicHookInfo<'_>) -> String {
    if let Some(message) = info.payload().downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = info.payload().downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_owned()
    }
}

fn default_log_directory(application: &str) -> PathBuf {
    std::env::var_os("STARMAN_LOG_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("logs").join(normalize_application_name(application)))
}

fn normalize_application_name(application: &str) -> String {
    let normalized = application
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>();

    normalized.trim_matches('-').to_owned()
}

fn trim_debug_string(value: &str) -> String {
    value
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
        .unwrap_or(value)
        .replace("\\\"", "\"")
}

fn unix_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(sequence: u64, message: &str) -> DiagnosticEvent {
        DiagnosticEvent {
            sequence,
            timestamp_millis: 1,
            level: DiagnosticLevel::Info,
            target: "engine::test".to_owned(),
            message: message.to_owned(),
            fields: BTreeMap::new(),
            thread: None,
        }
    }

    #[test]
    fn event_ring_keeps_recent_events_and_counts_drops() {
        let mut ring = EventRing::new(2);
        ring.push(event(1, "first"));
        ring.push(event(2, "second"));
        ring.push(event(3, "third"));

        let messages = ring
            .events
            .iter()
            .map(|event| event.message.as_str())
            .collect::<Vec<_>>();

        assert_eq!(messages, vec!["second", "third"]);
        assert_eq!(ring.dropped, 1);
    }

    #[test]
    fn diagnostic_event_roundtrips_as_json_line() {
        let original = event(42, "hello");
        let line = event_to_json_line(&original).expect("event should serialize");
        let parsed = event_from_json_line(&line).expect("event should parse");

        assert_eq!(parsed.sequence, 42);
        assert_eq!(parsed.message, "hello");
        assert_eq!(parsed.target, "engine::test");
    }

    #[test]
    fn panic_report_writes_schema_and_recent_events_file() {
        let dir = std::env::temp_dir().join(format!(
            "starman-diagnostics-test-{}-{}",
            std::process::id(),
            unix_millis()
        ));

        let crash_dir = write_panic_report("starman-test", &dir, "boom", Some("test.rs:1:1"))
            .expect("panic report should be written");

        assert!(crash_dir.join("report.json").exists());
        let report = fs::read_to_string(crash_dir.join("report.json")).expect("report json");
        assert!(report.contains("\"schema\": 1"));
        assert!(report.contains("\"message\": \"boom\""));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn file_sink_rotation_names_do_not_collide() {
        let dir = std::env::temp_dir().join(format!(
            "starman-diagnostics-rotation-test-{}-{}",
            std::process::id(),
            unix_millis()
        ));
        let mut sink =
            FileSink::new("rotation-test", dir.clone(), 1024, 10).expect("file sink should open");

        sink.rotate().expect("first rotation should succeed");
        sink.rotate().expect("second rotation should succeed");

        let rotated_files = fs::read_dir(&dir)
            .expect("rotation dir should exist")
            .filter_map(Result::ok)
            .filter(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with("rotation-test-")
            })
            .count();

        assert_eq!(rotated_files, 2);
        let _ = fs::remove_dir_all(dir);
    }
}
