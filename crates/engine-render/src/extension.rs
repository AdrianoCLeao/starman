//! Render extensions (ADR 0009 addendum): subsystems outside the renderer
//! (particles, game UI, editor overlays, project plugins) add their own
//! extract/prepare/encode stages and graph nodes without the renderer
//! knowing about them. Runtime plugins register a factory in the
//! [`RenderExtensions`] resource; every renderer drawing that world
//! instantiates it once.

use std::sync::{Arc, Mutex};

use bevy_ecs::prelude::{Resource, World};
use engine_assets::AssetServer;
use engine_core::Result;

use crate::capabilities::CapabilityTier;
use crate::gpu::assets::GpuAssetCache;
use crate::gpu::StagingBelt;
use crate::layouts::SharedLayouts;
use crate::quality::QualitySettings;
use crate::shader::ShaderLibrary;
use crate::uniforms::ViewUniform;

/// Where an extension node renders.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeTarget {
    /// Linear HDR color before post-processing, with the scene depth bound
    /// read-only (world-space effects: particles, decals).
    Hdr,
    /// The final (tonemapped, display-encoded) target, with the scene depth
    /// bound read-only (screen-space UI, overlays).
    Output,
}

/// A graph node contributed by an extension.
#[derive(Clone, Debug)]
pub struct ExtensionNode {
    pub name: &'static str,
    pub target: NodeTarget,
    /// Built-in or extension nodes this one must run after.
    pub after: Vec<&'static str>,
    /// Nodes this one must run before.
    pub before: Vec<&'static str>,
}

/// Information available during extract.
#[derive(Clone, Copy, Debug)]
pub struct ExtractInfo {
    pub width: u32,
    pub height: u32,
    pub frame_index: u64,
}

/// Everything an extension may use to upload data for this frame.
pub struct PrepareContext<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub encoder: &'a mut wgpu::CommandEncoder,
    pub belt: &'a mut StagingBelt,
    pub shaders: &'a mut ShaderLibrary,
    pub cache: &'a mut GpuAssetCache,
    pub server: &'a AssetServer,
    pub layouts: &'a SharedLayouts,
    /// The main camera view, when a 3D camera exists.
    pub view: Option<&'a ViewUniform>,
    pub width: u32,
    pub height: u32,
    pub target_format: wgpu::TextureFormat,
    pub tier: CapabilityTier,
    pub quality: &'a QualitySettings,
    pub frame_index: u64,
}

/// Everything an extension node may use while encoding.
pub struct EncodeContext<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub encoder: &'a mut wgpu::CommandEncoder,
    pub layouts: &'a SharedLayouts,
    pub cache: &'a GpuAssetCache,
    /// The color target for this node (HDR or final, see [`NodeTarget`]).
    pub color: &'a wgpu::TextureView,
    pub color_format: wgpu::TextureFormat,
    /// Single-sample scene depth (read-only attachment compatible).
    pub depth: &'a wgpu::TextureView,
    /// Basic view bind group (group 0 layout `view_basic`) and its offset;
    /// `None` without a 3D camera.
    pub view_bind_group: Option<(&'a wgpu::BindGroup, u32)>,
    pub view: Option<&'a ViewUniform>,
    pub width: u32,
    pub height: u32,
    pub tier: CapabilityTier,
    pub frame_index: u64,
}

pub trait RenderExtension: Send + 'static {
    /// Unique name (one instance per renderer).
    fn name(&self) -> &'static str;

    /// Graph nodes this extension encodes.
    fn nodes(&self) -> Vec<ExtensionNode>;

    /// Copy what the extension needs out of the world.
    fn extract(&mut self, world: &mut World, info: &ExtractInfo);

    /// Upload GPU data for this frame.
    fn prepare(&mut self, ctx: &mut PrepareContext<'_>) -> Result<()>;

    /// Encode `node` (one of [`Self::nodes`]).
    fn encode(&mut self, node: &'static str, ctx: &mut EncodeContext<'_>) -> Result<()>;

    /// Named counters for stats overlays and the smoke report.
    fn stats(&self) -> Vec<(&'static str, f64)> {
        Vec::new()
    }
}

type Factory = Arc<dyn Fn() -> Box<dyn RenderExtension> + Send + Sync>;

/// Registry of extension factories, stored in the ECS world.
#[derive(Resource, Clone, Default)]
pub struct RenderExtensions {
    factories: Arc<Mutex<Vec<(&'static str, Factory)>>>,
}

impl RenderExtensions {
    /// Registers `factory` under `name` (replacing an existing one).
    pub fn register(
        &self,
        name: &'static str,
        factory: impl Fn() -> Box<dyn RenderExtension> + Send + Sync + 'static,
    ) {
        let mut factories = self.factories.lock().unwrap_or_else(|p| p.into_inner());
        factories.retain(|(existing, _)| *existing != name);
        factories.push((name, Arc::new(factory)));
    }

    pub fn names(&self) -> Vec<&'static str> {
        self.factories
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .map(|(name, _)| *name)
            .collect()
    }

    /// Instantiates every registered extension whose name is not in
    /// `existing`.
    pub fn instantiate_missing(&self, existing: &[&'static str]) -> Vec<Box<dyn RenderExtension>> {
        self.factories
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .filter(|(name, _)| !existing.contains(name))
            .map(|(_, factory)| factory())
            .collect()
    }
}
