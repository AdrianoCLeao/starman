//! Project validation: a lenient check that reports every problem it finds,
//! rather than stopping at the first one, so a user (or the CLI) sees the
//! full picture in one pass.

use std::fmt;

/// Severity of a single validation finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// The project cannot be opened or trusted as-is.
    Error,
    /// The project can still be opened, but something is missing or
    /// unexpected (e.g. a generated directory that will simply be
    /// recreated on demand).
    Warning,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationIssue {
    pub severity: Severity,
    pub message: String,
}

impl fmt::Display for ValidationIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self.severity {
            Severity::Error => "error",
            Severity::Warning => "warning",
        };
        write!(f, "[{label}] {}", self.message)
    }
}

/// The result of validating a project directory: every issue found, in the
/// order they were checked.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ValidationReport {
    issues: Vec<ValidationIssue>,
}

impl ValidationReport {
    pub(crate) fn new() -> Self {
        Self { issues: Vec::new() }
    }

    pub(crate) fn push_error(&mut self, message: impl Into<String>) {
        self.issues.push(ValidationIssue {
            severity: Severity::Error,
            message: message.into(),
        });
    }

    pub(crate) fn push_warning(&mut self, message: impl Into<String>) {
        self.issues.push(ValidationIssue {
            severity: Severity::Warning,
            message: message.into(),
        });
    }

    /// True if no error-level issue was found (warnings are still allowed).
    pub fn is_valid(&self) -> bool {
        !self
            .issues
            .iter()
            .any(|issue| issue.severity == Severity::Error)
    }

    pub fn issues(&self) -> &[ValidationIssue] {
        &self.issues
    }

    pub fn errors(&self) -> impl Iterator<Item = &ValidationIssue> {
        self.issues
            .iter()
            .filter(|issue| issue.severity == Severity::Error)
    }

    pub fn warnings(&self) -> impl Iterator<Item = &ValidationIssue> {
        self.issues
            .iter()
            .filter(|issue| issue.severity == Severity::Warning)
    }
}

impl fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.issues.is_empty() {
            return write!(f, "project is valid");
        }

        for (index, issue) in self.issues.iter().enumerate() {
            if index > 0 {
                writeln!(f)?;
            }
            write!(f, "{issue}")?;
        }

        Ok(())
    }
}
