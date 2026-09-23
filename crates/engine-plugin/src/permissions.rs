//! Project-level permission grants for scripts and plugins.

use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Deny-by-default permissions declared in `project.ron`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectPermissions {
    /// Glob-like prefixes relative to the project root (e.g. `assets/**`).
    #[serde(default)]
    pub filesystem: Vec<String>,
    #[serde(default)]
    pub process: bool,
    #[serde(default)]
    pub network: bool,
}

impl Default for ProjectPermissions {
    fn default() -> Self {
        Self {
            filesystem: vec![
                "assets/**".to_owned(),
                "scripts/**".to_owned(),
                "saves/**".to_owned(),
            ],
            process: false,
            network: false,
        }
    }
}

/// Runtime checker used by the host bus before sensitive operations.
#[derive(Debug, Clone)]
pub struct PermissionGuard {
    project_root: PathBuf,
    permissions: ProjectPermissions,
}

impl PermissionGuard {
    pub fn new(project_root: impl Into<PathBuf>, permissions: ProjectPermissions) -> Self {
        Self {
            project_root: project_root.into(),
            permissions,
        }
    }

    pub fn permissions(&self) -> &ProjectPermissions {
        &self.permissions
    }

    pub fn allow_process(&self) -> bool {
        self.permissions.process
    }

    pub fn allow_network(&self) -> bool {
        self.permissions.network
    }

    /// Returns true if `relative` (project-relative) is covered by a grant.
    pub fn allow_filesystem_relative(&self, relative: &str) -> bool {
        let normalized = normalize_relative(relative);
        if normalized.is_none() {
            return false;
        }
        let relative = normalized.unwrap();
        self.permissions
            .filesystem
            .iter()
            .any(|grant| path_matches_grant(&relative, grant))
    }

    /// Resolves `path` under the project root and checks grants.
    pub fn allow_filesystem_path(&self, path: &Path) -> bool {
        let Ok(canonical_root) = self.project_root.canonicalize() else {
            // Fall back to non-canonical matching when the root does not exist yet.
            return path
                .strip_prefix(&self.project_root)
                .ok()
                .map(|rel| {
                    let s = rel.to_string_lossy().replace('\\', "/");
                    self.allow_filesystem_relative(&s)
                })
                .unwrap_or(false);
        };
        let Ok(canonical) = path.canonicalize() else {
            // Path may not exist yet — check the relative form.
            return path
                .strip_prefix(&self.project_root)
                .ok()
                .map(|rel| {
                    let s = rel.to_string_lossy().replace('\\', "/");
                    self.allow_filesystem_relative(&s)
                })
                .unwrap_or(false);
        };
        canonical
            .strip_prefix(&canonical_root)
            .ok()
            .map(|rel| {
                let s = rel.to_string_lossy().replace('\\', "/");
                self.allow_filesystem_relative(&s)
            })
            .unwrap_or(false)
    }
}

fn normalize_relative(relative: &str) -> Option<String> {
    let path = Path::new(relative);
    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            Component::Normal(s) => parts.push(s.to_string_lossy().into_owned()),
            Component::CurDir => {}
            Component::ParentDir => {
                parts.pop()?;
            }
            Component::RootDir | Component::Prefix(_) => return None,
        }
    }
    Some(parts.join("/"))
}

fn path_matches_grant(relative: &str, grant: &str) -> bool {
    let grant = grant.trim_start_matches("./");
    if let Some(prefix) = grant.strip_suffix("/**") {
        if prefix.is_empty() {
            return true;
        }
        relative == prefix || relative.starts_with(&format!("{prefix}/"))
    } else if let Some(prefix) = grant.strip_suffix("/*") {
        if let Some(rest) = relative.strip_prefix(prefix) {
            let rest = rest.trim_start_matches('/');
            return !rest.is_empty() && !rest.contains('/');
        }
        false
    } else {
        relative == grant
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_path_escape() {
        let guard = PermissionGuard::new("/tmp/proj", ProjectPermissions::default());
        assert!(!guard.allow_filesystem_relative("../etc/passwd"));
        assert!(!guard.allow_filesystem_relative("/etc/passwd"));
    }

    #[test]
    fn allows_granted_prefixes() {
        let guard = PermissionGuard::new("/tmp/proj", ProjectPermissions::default());
        assert!(guard.allow_filesystem_relative("assets/textures/a.png"));
        assert!(guard.allow_filesystem_relative("scripts/main.lua"));
        assert!(!guard.allow_filesystem_relative("secrets/key.pem"));
        assert!(!guard.allow_process());
        assert!(!guard.allow_network());
    }
}
