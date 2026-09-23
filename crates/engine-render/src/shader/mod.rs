#![allow(dead_code)]

//! WGSL shader library with `#include`, variants, and disk cache.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use blake3::Hasher;
use engine_core::{EngineError, Result};

use crate::capabilities::CapabilityTier;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ShaderId(pub &'static str);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ShaderVariantKey {
    pub id: ShaderId,
    pub feature_bits: u64,
    pub tier: CapabilityTier,
}

pub struct ShaderLibrary {
    root: PathBuf,
    cache_dir: PathBuf,
    /// Expanded source cache.
    expanded: HashMap<String, String>,
    modules: HashMap<ShaderVariantKey, wgpu::ShaderModule>,
}

impl ShaderLibrary {
    pub fn new(root: impl Into<PathBuf>, cache_dir: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            cache_dir: cache_dir.into(),
            expanded: HashMap::new(),
            modules: HashMap::new(),
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Expand `#include "path"` recursively with cycle detection.
    pub fn expand_source(&mut self, relative: &str) -> Result<String> {
        if let Some(cached) = self.expanded.get(relative) {
            return Ok(cached.clone());
        }
        let mut visiting = HashSet::new();
        let expanded = expand_includes(&self.root, relative, &mut visiting)?;
        self.expanded.insert(relative.to_owned(), expanded.clone());
        Ok(expanded)
    }

    pub fn compile(
        &mut self,
        device: &wgpu::Device,
        key: ShaderVariantKey,
        relative: &str,
    ) -> Result<&wgpu::ShaderModule> {
        if self.modules.contains_key(&key) {
            return Ok(self.modules.get(&key).expect("just inserted"));
        }
        let source = self.expand_source(relative)?;
        let hash = hash_source(&source, key.feature_bits, key.tier);
        let _ = self.write_cache(&hash, &source);

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(key.id.0),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        self.modules.insert(key, module);
        Ok(self.modules.get(&key).expect("just inserted"))
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

fn hash_source(source: &str, features: u64, tier: CapabilityTier) -> String {
    let mut hasher = Hasher::new();
    hasher.update(source.as_bytes());
    hasher.update(&features.to_le_bytes());
    hasher.update(tier.as_str().as_bytes());
    hasher.finalize().to_hex().to_string()
}

fn expand_includes(root: &Path, relative: &str, visiting: &mut HashSet<String>) -> Result<String> {
    if !visiting.insert(relative.to_owned()) {
        return Err(EngineError::Render(format!(
            "shader include cycle involving '{relative}'"
        )));
    }
    let path = root.join(relative);
    let text = std::fs::read_to_string(&path).map_err(|error| {
        EngineError::Render(format!(
            "failed to read shader '{}': {error}",
            path.display()
        ))
    })?;

    let mut output = String::new();
    for (line_no, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("#include") {
            let rest = rest.trim();
            let include_path = rest
                .trim_matches(|c| c == '"' || c == '\'' || c == '<' || c == '>')
                .trim();
            if include_path.is_empty() {
                return Err(EngineError::Render(format!(
                    "{}:{}: empty #include",
                    relative,
                    line_no + 1
                )));
            }
            let nested = expand_includes(root, include_path, visiting).map_err(|error| {
                EngineError::Render(format!(
                    "{}:{}: while including '{include_path}': {error}",
                    relative,
                    line_no + 1
                ))
            })?;
            output.push_str(&nested);
            if !nested.ends_with('\n') {
                output.push('\n');
            }
        } else {
            output.push_str(line);
            output.push('\n');
        }
    }
    visiting.remove(relative);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn expands_includes_and_detects_cycles() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("starman-shader-{nanos}"));
        std::fs::create_dir_all(root.join("common")).unwrap();
        std::fs::write(root.join("common/math.wgsl"), "const PI: f32 = 3.14;\n").unwrap();
        std::fs::write(
            root.join("mesh.wgsl"),
            "#include \"common/math.wgsl\"\nfn f() {}\n",
        )
        .unwrap();
        let mut lib = ShaderLibrary::new(&root, root.join("cache"));
        let expanded = lib.expand_source("mesh.wgsl").unwrap();
        assert!(expanded.contains("PI"));
        assert!(expanded.contains("fn f()"));

        std::fs::write(root.join("a.wgsl"), "#include \"b.wgsl\"\n").unwrap();
        std::fs::write(root.join("b.wgsl"), "#include \"a.wgsl\"\n").unwrap();
        assert!(lib.expand_source("a.wgsl").is_err());
        let _ = std::fs::remove_dir_all(&root);
    }
}
