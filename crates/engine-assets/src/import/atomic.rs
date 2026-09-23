//! Atomic file writes: readers either see the previous complete file or
//! the new complete file, never a partially written one.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_path_for(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "file".to_owned());
    let unique = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let temp_name = format!("{file_name}.tmp-{}-{unique}", std::process::id());
    path.with_file_name(temp_name)
}

/// True for the temporary files [`write_atomic`] creates, so scanners and
/// watchers can ignore them.
pub(crate) fn is_temp_file(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.contains(".tmp-"))
}

/// Writes `bytes` to a sibling temporary file and renames it over `path`.
/// On failure the temporary file is removed and `path` is left untouched.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let temp_path = temp_path_for(path);

    if let Err(error) = fs::write(&temp_path, bytes) {
        let _ = fs::remove_file(&temp_path);
        return Err(error);
    }

    if let Err(error) = fs::rename(&temp_path, path) {
        let _ = fs::remove_file(&temp_path);
        return Err(error);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn scratch_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be valid")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("{prefix}-{nanos}"));
        fs::create_dir_all(&dir).expect("scratch dir should be created");
        dir
    }

    #[test]
    fn write_atomic_replaces_contents_and_leaves_no_temp_file() {
        let dir = scratch_dir("starman-atomic-ok");
        let target = dir.join("value.txt");
        fs::write(&target, "old").expect("seed file should be written");

        write_atomic(&target, b"new").expect("atomic write should succeed");

        assert_eq!(fs::read_to_string(&target).unwrap(), "new");
        let leftovers = fs::read_dir(&dir)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| is_temp_file(&entry.path()))
            .count();
        assert_eq!(leftovers, 0);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn failed_write_leaves_target_and_directory_clean() {
        let dir = scratch_dir("starman-atomic-fail");
        // The destination is a directory, so the final rename must fail.
        let target = dir.join("occupied");
        fs::create_dir_all(&target).expect("target dir should be created");

        assert!(write_atomic(&target, b"data").is_err());

        assert!(target.is_dir());
        let leftovers = fs::read_dir(&dir)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| is_temp_file(&entry.path()))
            .count();
        assert_eq!(leftovers, 0);

        let _ = fs::remove_dir_all(&dir);
    }
}
