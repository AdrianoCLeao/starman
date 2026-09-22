//! Coalesces bursts of filesystem events (an editor saving a file often
//! produces several Create/Modify/Rename events in a row) into a single
//! notification per path, emitted once the path has been quiet for a
//! configurable window. The clock is injected so this is fully
//! deterministic under test.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

pub(crate) struct Debouncer {
    quiet_window: Duration,
    last_event: HashMap<PathBuf, Instant>,
}

impl Debouncer {
    pub fn new(quiet_window: Duration) -> Self {
        Self {
            quiet_window,
            last_event: HashMap::new(),
        }
    }

    /// Records an event for `path` at `now`, restarting its quiet window.
    pub fn push(&mut self, path: PathBuf, now: Instant) {
        self.last_event.insert(path, now);
    }

    /// Removes and returns every path that has been quiet for at least the
    /// configured window as of `now`, in a stable (sorted) order.
    pub fn drain_ready(&mut self, now: Instant) -> Vec<PathBuf> {
        let window = self.quiet_window;
        let mut ready: Vec<PathBuf> = self
            .last_event
            .iter()
            .filter(|(_, last)| now.saturating_duration_since(**last) >= window)
            .map(|(path, _)| path.clone())
            .collect();
        ready.sort();

        for path in &ready {
            self.last_event.remove(path);
        }

        ready
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WINDOW: Duration = Duration::from_millis(100);

    #[test]
    fn nothing_is_emitted_before_the_window_elapses() {
        let start = Instant::now();
        let mut debouncer = Debouncer::new(WINDOW);
        debouncer.push(PathBuf::from("a.png"), start);

        assert!(debouncer
            .drain_ready(start + Duration::from_millis(99))
            .is_empty());
    }

    #[test]
    fn a_burst_of_events_collapses_into_one_emission() {
        let start = Instant::now();
        let mut debouncer = Debouncer::new(WINDOW);
        for offset in 0..10 {
            debouncer.push(
                PathBuf::from("a.png"),
                start + Duration::from_millis(offset * 10),
            );
        }

        // Last event at +90ms; the window restarts with each event.
        assert!(debouncer
            .drain_ready(start + Duration::from_millis(150))
            .is_empty());
        let ready = debouncer.drain_ready(start + Duration::from_millis(190));
        assert_eq!(ready, vec![PathBuf::from("a.png")]);
        assert!(debouncer
            .drain_ready(start + Duration::from_millis(500))
            .is_empty());
    }

    #[test]
    fn distinct_paths_are_debounced_independently() {
        let start = Instant::now();
        let mut debouncer = Debouncer::new(WINDOW);
        debouncer.push(PathBuf::from("a.png"), start);
        debouncer.push(PathBuf::from("b.png"), start + Duration::from_millis(80));

        let first = debouncer.drain_ready(start + Duration::from_millis(100));
        assert_eq!(first, vec![PathBuf::from("a.png")]);

        let second = debouncer.drain_ready(start + Duration::from_millis(180));
        assert_eq!(second, vec![PathBuf::from("b.png")]);
    }

    #[test]
    fn zero_window_emits_immediately() {
        let start = Instant::now();
        let mut debouncer = Debouncer::new(Duration::ZERO);
        debouncer.push(PathBuf::from("a.png"), start);

        assert_eq!(debouncer.drain_ready(start), vec![PathBuf::from("a.png")]);
    }
}
