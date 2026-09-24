//! WGSL shader library (ADR 0009): `#include`, preprocessor variants,
//! compiled-module cache, disk cache of expanded variants, and compile
//! errors mapped back to the original file and line.
//!
//! Supported directives (one per line, `#` must be the first non-blank):
//!
//! * `#include "relative/path.wgsl"` — textual include, each file at most
//!   once per expansion (include guards are implicit), cycles rejected;
//! * `#define NAME` — defines a flag for the rest of the expansion;
//! * `#ifdef NAME`, `#ifndef NAME`, `#if defined(A) && !defined(B)`
//!   (`&&`/`||`/`!` over `defined(...)`, no parentheses nesting),
//!   `#else`, `#endif` — nestable conditionals.
//!
//! Variants are selected by a set of defines; each distinct set compiles
//! to its own module, cached by `(path, defines)`.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use blake3::Hasher;
use engine_core::{EngineError, Result};

use crate::capabilities::CapabilityTier;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ShaderId(pub &'static str);

/// Legacy variant key (id + feature bits + tier); kept for API stability.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ShaderVariantKey {
    pub id: ShaderId,
    pub feature_bits: u64,
    pub tier: CapabilityTier,
}

/// Where a line of expanded source came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceLocation {
    pub file: String,
    pub line: usize,
}

/// Fully preprocessed source plus a line map back to the originals.
#[derive(Clone, Debug)]
pub struct ExpandedShader {
    pub source: String,
    /// `line_map[i]` is the origin of expanded line `i` (0-based).
    pub line_map: Vec<SourceLocation>,
}

impl ExpandedShader {
    /// Maps a 1-based expanded line number to its origin.
    pub fn origin_of(&self, expanded_line: usize) -> Option<&SourceLocation> {
        expanded_line
            .checked_sub(1)
            .and_then(|index| self.line_map.get(index))
    }
}

type VariantKey = (String, Vec<String>);

pub struct ShaderLibrary {
    root: PathBuf,
    cache_dir: PathBuf,
    /// In-memory sources registered by extensions (checked before disk),
    /// keyed by virtual path (`vfx/particles.wgsl`).
    virtual_sources: HashMap<String, String>,
    /// Expanded (include-resolved, preprocessed) sources per variant.
    expanded: HashMap<VariantKey, ExpandedShader>,
    modules: HashMap<VariantKey, Arc<wgpu::ShaderModule>>,
    legacy_modules: HashMap<ShaderVariantKey, Arc<wgpu::ShaderModule>>,
}

impl ShaderLibrary {
    pub fn new(root: impl Into<PathBuf>, cache_dir: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            cache_dir: cache_dir.into(),
            virtual_sources: HashMap::new(),
            expanded: HashMap::new(),
            modules: HashMap::new(),
            legacy_modules: HashMap::new(),
        }
    }

    /// The library rooted at the engine's built-in shader directory.
    pub fn builtin() -> Self {
        Self::new(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders"),
            PathBuf::from(".starman/shader-cache"),
        )
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Registers an in-memory source under `path` (extensions ship their
    /// WGSL with `include_str!`; it may `#include` the built-in modules).
    /// Re-registering different text invalidates cached expansions.
    pub fn add_source(&mut self, path: impl Into<String>, source: impl Into<String>) {
        let path = path.into();
        let source = source.into();
        if self.virtual_sources.get(&path) != Some(&source) {
            self.virtual_sources.insert(path, source);
            self.invalidate();
        }
    }

    pub fn has_source(&self, path: &str) -> bool {
        self.virtual_sources.contains_key(path)
    }

    /// Expands `#include`s only (no defines), for tools.
    pub fn expand_source(&mut self, relative: &str) -> Result<String> {
        Ok(self.expand_variant(relative, &[])?.source)
    }

    /// Expands `relative` with `defines` active.
    pub fn expand_variant(&mut self, relative: &str, defines: &[&str]) -> Result<ExpandedShader> {
        let key = variant_key(relative, defines);
        if let Some(cached) = self.expanded.get(&key) {
            return Ok(cached.clone());
        }
        let mut state = Preprocessor {
            root: &self.root,
            virtual_sources: &self.virtual_sources,
            defines: defines.iter().map(|d| (*d).to_owned()).collect(),
            included: HashSet::new(),
            stack: Vec::new(),
            output: ExpandedShader {
                source: String::new(),
                line_map: Vec::new(),
            },
        };
        state.process_file(relative)?;
        self.expanded.insert(key, state.output.clone());
        Ok(state.output)
    }

    /// Compiles (or returns the cached) module for `relative` + `defines`.
    /// Validation errors are reported with the original file and line.
    pub fn module(
        &mut self,
        device: &wgpu::Device,
        relative: &str,
        defines: &[&str],
    ) -> Result<Arc<wgpu::ShaderModule>> {
        let key = variant_key(relative, defines);
        if let Some(module) = self.modules.get(&key) {
            return Ok(Arc::clone(module));
        }
        let expanded = self.expand_variant(relative, defines)?;
        validate_wgsl(relative, defines, &expanded)?;
        let hash = hash_source(&expanded.source, &key.1);
        let _ = self.write_cache(&hash, &expanded.source);

        let label = if defines.is_empty() {
            relative.to_owned()
        } else {
            format!("{relative}[{}]", key.1.join(","))
        };
        let module = Arc::new(device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&label),
            source: wgpu::ShaderSource::Wgsl(expanded.source.into()),
        }));
        self.modules.insert(key, Arc::clone(&module));
        Ok(module)
    }

    /// Legacy entry point (M4 API): compiles `relative` with no defines.
    pub fn compile(
        &mut self,
        device: &wgpu::Device,
        key: ShaderVariantKey,
        relative: &str,
    ) -> Result<Arc<wgpu::ShaderModule>> {
        if let Some(module) = self.legacy_modules.get(&key) {
            return Ok(Arc::clone(module));
        }
        let module = self.module(device, relative, &[])?;
        self.legacy_modules.insert(key, Arc::clone(&module));
        Ok(module)
    }

    /// Number of distinct compiled variants (diagnostics).
    pub fn compiled_variant_count(&self) -> usize {
        self.modules.len()
    }

    /// Drops every cached expansion/module (shader hot reload).
    pub fn invalidate(&mut self) {
        self.expanded.clear();
        self.modules.clear();
        self.legacy_modules.clear();
    }

    fn write_cache(&self, hash: &str, source: &str) -> Result<()> {
        let _ = std::fs::create_dir_all(&self.cache_dir);
        let path = self.cache_dir.join(format!("{hash}.wgsl"));
        if !path.is_file() {
            std::fs::write(&path, source)
                .map_err(|error| EngineError::Render(error.to_string()))?;
        }
        Ok(())
    }
}

fn variant_key(relative: &str, defines: &[&str]) -> VariantKey {
    let set: BTreeSet<String> = defines.iter().map(|d| (*d).to_owned()).collect();
    (relative.to_owned(), set.into_iter().collect())
}

fn hash_source(source: &str, defines: &[String]) -> String {
    let mut hasher = Hasher::new();
    hasher.update(source.as_bytes());
    for define in defines {
        hasher.update(define.as_bytes());
        hasher.update(b"\0");
    }
    hasher.finalize().to_hex().to_string()
}

/// Parses with naga so errors point at the authored file/line instead of
/// an opaque device error at pipeline creation.
fn validate_wgsl(relative: &str, defines: &[&str], expanded: &ExpandedShader) -> Result<()> {
    let module = naga::front::wgsl::parse_str(&expanded.source).map_err(|error| {
        let location = error
            .location(&expanded.source)
            .and_then(|loc| expanded.origin_of(loc.line_number as usize));
        let at = location
            .map(|loc| format!("{}:{}", loc.file, loc.line))
            .unwrap_or_else(|| relative.to_owned());
        EngineError::Render(format!(
            "shader '{relative}' {defines:?}: {at}: {}",
            error.message()
        ))
    })?;
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator.validate(&module).map_err(|error| {
        let span = error
            .spans()
            .next()
            .map(|(span, _)| span.location(&expanded.source).line_number as usize);
        let at = span
            .and_then(|line| expanded.origin_of(line))
            .map(|loc| format!("{}:{}", loc.file, loc.line))
            .unwrap_or_else(|| relative.to_owned());
        EngineError::Render(format!(
            "shader '{relative}' {defines:?}: {at}: validation failed: {}",
            error.as_inner()
        ))
    })?;
    Ok(())
}

struct Preprocessor<'a> {
    root: &'a Path,
    virtual_sources: &'a HashMap<String, String>,
    defines: HashSet<String>,
    included: HashSet<String>,
    stack: Vec<String>,
    output: ExpandedShader,
}

/// One level of `#if` nesting.
struct Conditional {
    /// Whether the enclosing scope is emitting.
    parent_active: bool,
    /// Whether this branch is emitting.
    active: bool,
    /// Whether some branch of this conditional has already been taken.
    taken: bool,
    else_seen: bool,
}

impl Preprocessor<'_> {
    fn process_file(&mut self, relative: &str) -> Result<()> {
        if self.stack.iter().any(|entry| entry == relative) {
            return Err(EngineError::Render(format!(
                "shader include cycle: {} -> {relative}",
                self.stack.join(" -> ")
            )));
        }
        if !self.included.insert(relative.to_owned()) {
            // Implicit include guard.
            return Ok(());
        }
        let text = match self.virtual_sources.get(relative) {
            Some(text) => text.clone(),
            None => {
                let path = self.root.join(relative);
                std::fs::read_to_string(&path).map_err(|error| {
                    EngineError::Render(format!(
                        "failed to read shader '{}': {error}",
                        path.display()
                    ))
                })?
            }
        };
        self.stack.push(relative.to_owned());

        let mut conditionals: Vec<Conditional> = Vec::new();
        let is_active =
            |conditionals: &Vec<Conditional>| conditionals.last().map(|c| c.active).unwrap_or(true);

        for (line_index, line) in text.lines().enumerate() {
            let line_no = line_index + 1;
            let trimmed = line.trim_start();
            let error =
                |message: String| EngineError::Render(format!("{relative}:{line_no}: {message}"));
            if let Some(directive) = trimmed.strip_prefix('#') {
                let (name, rest) = directive
                    .split_once(char::is_whitespace)
                    .map(|(name, rest)| (name, rest.trim()))
                    .unwrap_or((directive.trim(), ""));
                match name {
                    "include" => {
                        if is_active(&conditionals) {
                            let include = rest
                                .trim_matches(|c| c == '"' || c == '\'' || c == '<' || c == '>')
                                .trim();
                            if include.is_empty() {
                                return Err(error("empty #include".to_owned()));
                            }
                            self.process_file(include).map_err(|nested| {
                                error(format!("while including '{include}': {nested}"))
                            })?;
                        }
                    }
                    "define" => {
                        if is_active(&conditionals) {
                            let symbol = rest.split_whitespace().next().unwrap_or("");
                            if symbol.is_empty() {
                                return Err(error("#define needs a name".to_owned()));
                            }
                            self.defines.insert(symbol.to_owned());
                        }
                    }
                    "ifdef" | "ifndef" | "if" => {
                        let parent_active = is_active(&conditionals);
                        let value = match name {
                            "ifdef" => self.defines.contains(rest),
                            "ifndef" => !self.defines.contains(rest),
                            _ => evaluate_condition(rest, &self.defines).map_err(&error)?,
                        };
                        conditionals.push(Conditional {
                            parent_active,
                            active: parent_active && value,
                            taken: value,
                            else_seen: false,
                        });
                    }
                    "elif" => {
                        let defines = &self.defines;
                        let top = conditionals
                            .last_mut()
                            .ok_or_else(|| error("#elif without #if".to_owned()))?;
                        if top.else_seen {
                            return Err(error("#elif after #else".to_owned()));
                        }
                        let value = evaluate_condition(rest, defines).map_err(&error)?;
                        top.active = top.parent_active && !top.taken && value;
                        top.taken |= value;
                    }
                    "else" => {
                        let top = conditionals
                            .last_mut()
                            .ok_or_else(|| error("#else without #if".to_owned()))?;
                        if top.else_seen {
                            return Err(error("duplicate #else".to_owned()));
                        }
                        top.else_seen = true;
                        top.active = top.parent_active && !top.taken;
                        top.taken = true;
                    }
                    "endif" => {
                        conditionals
                            .pop()
                            .ok_or_else(|| error("#endif without #if".to_owned()))?;
                    }
                    _ => {
                        // Not a directive we own (WGSL has no `#`), keep it
                        // so the parser reports it with a mapped location.
                        if is_active(&conditionals) {
                            self.emit(line, relative, line_no);
                        }
                    }
                }
                continue;
            }
            if is_active(&conditionals) {
                self.emit(line, relative, line_no);
            }
        }
        if !conditionals.is_empty() {
            return Err(EngineError::Render(format!(
                "{relative}: {} unterminated #if block(s)",
                conditionals.len()
            )));
        }
        self.stack.pop();
        Ok(())
    }

    fn emit(&mut self, line: &str, file: &str, line_no: usize) {
        self.output.source.push_str(line);
        self.output.source.push('\n');
        self.output.line_map.push(SourceLocation {
            file: file.to_owned(),
            line: line_no,
        });
    }
}

/// Evaluates `defined(A) && !defined(B) || defined(C)` (`&&` binds
/// tighter than `||`; no parentheses beyond `defined(...)`).
fn evaluate_condition(
    expression: &str,
    defines: &HashSet<String>,
) -> std::result::Result<bool, String> {
    if expression.trim().is_empty() {
        return Err("#if needs a condition".to_owned());
    }
    let mut any = false;
    for disjunct in expression.split("||") {
        let mut all = true;
        for term in disjunct.split("&&") {
            let term = term.trim();
            let (negated, term) = match term.strip_prefix('!') {
                Some(rest) => (true, rest.trim()),
                None => (false, term),
            };
            let value = if let Some(inner) = term
                .strip_prefix("defined(")
                .and_then(|rest| rest.strip_suffix(')'))
            {
                defines.contains(inner.trim())
            } else if term == "1" || term == "true" {
                true
            } else if term == "0" || term == "false" {
                false
            } else {
                return Err(format!("unsupported #if term '{term}'"));
            };
            all &= value != negated;
        }
        any |= all;
    }
    Ok(any)
}

#[cfg(test)]
#[path = "shader_tests.rs"]
mod tests;
