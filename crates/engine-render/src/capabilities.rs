//! Capability tiers for wgpu backends (ADR 0009).

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CapabilityTier {
    /// No compute clusters; CPU light assignment / simple forward.
    Tier0,
    /// Compute + storage buffers for Forward+ clustered culling.
    Tier1,
}

impl CapabilityTier {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Tier0 => "tier0",
            Self::Tier1 => "tier1",
        }
    }

    pub fn requires_compute(self) -> bool {
        matches!(self, Self::Tier1)
    }
}

#[derive(Clone, Debug)]
pub struct NegotiatedCapabilities {
    pub tier: CapabilityTier,
    pub adapter_name: String,
    pub backend: String,
    pub features: wgpu::Features,
    pub limits: wgpu::Limits,
}

/// Features required for Tier 1 Forward+ clusters.
pub fn tier1_features() -> wgpu::Features {
    wgpu::Features::empty() // compute is in core WebGPU; storage buffers too.
}

/// Prefer Tier 1 when the adapter exposes usable compute + storage limits.
pub fn negotiate_tier(adapter: &wgpu::Adapter) -> CapabilityTier {
    let limits = adapter.limits();
    // wgpu always exposes compute on native Tier-1 backends we ship.
    // Guard on storage buffer binding size as a proxy for cluster SSBOs.
    if limits.max_storage_buffer_binding_size >= 64 * 1024
        && limits.max_compute_workgroup_storage_size > 0
    {
        CapabilityTier::Tier1
    } else {
        CapabilityTier::Tier0
    }
}

pub fn request_device(
    adapter: &wgpu::Adapter,
) -> Result<(wgpu::Device, wgpu::Queue, NegotiatedCapabilities), wgpu::RequestDeviceError> {
    let tier = negotiate_tier(adapter);
    let info = adapter.get_info();
    let limits = adapter.limits();
    let desc = wgpu::DeviceDescriptor {
        label: Some("starman-render-device"),
        required_features: tier1_features(),
        required_limits: wgpu::Limits::default(),
        memory_hints: Default::default(),
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&desc, None))?;
    Ok((
        device,
        queue,
        NegotiatedCapabilities {
            tier,
            adapter_name: info.name.clone(),
            backend: format!("{:?}", info.backend),
            features: adapter.features(),
            limits,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_strings() {
        assert_eq!(CapabilityTier::Tier1.as_str(), "tier1");
        assert!(CapabilityTier::Tier1.requires_compute());
    }
}
