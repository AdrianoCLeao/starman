//! Quality presets and resolved render budgets (ADR 0010).

use serde::{Deserialize, Serialize};

use crate::capabilities::CapabilityTier;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QualityPreset {
    Low,
    #[default]
    Medium,
    High,
    Ultra,
}

impl QualityPreset {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Ultra => "ultra",
        }
    }

    pub fn all() -> [Self; 4] {
        [Self::Low, Self::Medium, Self::High, Self::Ultra]
    }
}

/// Resolved knobs applied to the render graph and GPU budgets each frame.
#[derive(Clone, Debug, PartialEq)]
pub struct QualitySettings {
    pub preset: QualityPreset,
    pub shadow_map_resolution: u32,
    pub cascade_count: u32,
    pub local_shadow_slots: u32,
    pub msaa_samples: u32,
    pub taa_enabled: bool,
    pub ssao_enabled: bool,
    pub bloom_enabled: bool,
    pub fog_enabled: bool,
    pub max_lights: usize,
    pub max_probes: u32,
    pub probe_resolution: u32,
    pub lod_bias: f32,
    pub exposure: f32,
    /// Distance covered by cascaded shadows (meters).
    pub shadow_distance: f32,
    pub ssao_samples: u32,
    /// GPU particle budget (per frame, all systems).
    pub max_gpu_particles: u32,
    /// CPU particle budget (Tier 0 and CPU-simulated effects).
    pub max_cpu_particles: u32,
}

impl QualitySettings {
    pub fn from_preset(preset: QualityPreset) -> Self {
        match preset {
            QualityPreset::Low => Self {
                preset,
                shadow_map_resolution: 512,
                cascade_count: 2,
                local_shadow_slots: 0,
                msaa_samples: 4,
                taa_enabled: false,
                ssao_enabled: false,
                bloom_enabled: false,
                fog_enabled: true,
                max_lights: 32,
                max_probes: 1,
                probe_resolution: 64,
                lod_bias: 1.0,
                exposure: 1.0,
                shadow_distance: 40.0,
                ssao_samples: 8,
                max_gpu_particles: 20_000,
                max_cpu_particles: 4_000,
            },
            QualityPreset::Medium => Self {
                preset,
                shadow_map_resolution: 1024,
                cascade_count: 3,
                local_shadow_slots: 1,
                msaa_samples: 1,
                taa_enabled: true,
                ssao_enabled: true,
                bloom_enabled: true,
                fog_enabled: true,
                max_lights: 64,
                max_probes: 2,
                probe_resolution: 128,
                lod_bias: 0.0,
                exposure: 1.0,
                shadow_distance: 60.0,
                ssao_samples: 12,
                max_gpu_particles: 100_000,
                max_cpu_particles: 10_000,
            },
            QualityPreset::High => Self {
                preset,
                shadow_map_resolution: 2048,
                cascade_count: 4,
                local_shadow_slots: 2,
                msaa_samples: 1,
                taa_enabled: true,
                ssao_enabled: true,
                bloom_enabled: true,
                fog_enabled: true,
                max_lights: 128,
                max_probes: 4,
                probe_resolution: 256,
                lod_bias: -0.5,
                exposure: 1.0,
                shadow_distance: 90.0,
                ssao_samples: 16,
                max_gpu_particles: 250_000,
                max_cpu_particles: 20_000,
            },
            QualityPreset::Ultra => Self {
                preset,
                shadow_map_resolution: 2048,
                cascade_count: 4,
                local_shadow_slots: 2,
                msaa_samples: 1,
                taa_enabled: true,
                ssao_enabled: true,
                bloom_enabled: true,
                fog_enabled: true,
                max_lights: 128,
                max_probes: 4,
                probe_resolution: 256,
                lod_bias: -1.0,
                exposure: 1.0,
                shadow_distance: 150.0,
                ssao_samples: 24,
                max_gpu_particles: 500_000,
                max_cpu_particles: 40_000,
            },
        }
    }

    /// When TAA is on, MSAA is forced to 1× (ADR 0010).
    pub fn effective_msaa_samples(&self) -> u32 {
        if self.taa_enabled {
            1
        } else {
            self.msaa_samples.max(1)
        }
    }

    /// Tier 0 collapses to a Low-equivalent deterministic subset.
    pub fn apply_tier_fallback(mut self, tier: CapabilityTier) -> Self {
        if tier == CapabilityTier::Tier0 {
            let low = Self::from_preset(QualityPreset::Low);
            self.shadow_map_resolution = low.shadow_map_resolution;
            self.cascade_count = low.cascade_count.min(self.cascade_count);
            self.local_shadow_slots = 0;
            self.msaa_samples = 1;
            self.taa_enabled = false;
            self.ssao_enabled = false;
            self.bloom_enabled = false;
            self.max_lights = self.max_lights.min(low.max_lights);
            self.max_probes = self.max_probes.min(1);
            self.probe_resolution = self.probe_resolution.min(64);
            self.shadow_distance = self.shadow_distance.min(low.shadow_distance);
            // No compute on Tier 0: particles simulate on the CPU only.
            self.max_gpu_particles = 0;
            self.max_cpu_particles = self.max_cpu_particles.min(low.max_cpu_particles);
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn taa_forces_msaa_off() {
        let q = QualitySettings::from_preset(QualityPreset::High);
        assert!(q.taa_enabled);
        assert_eq!(q.effective_msaa_samples(), 1);
    }

    #[test]
    fn tier0_disables_heavy_features() {
        let q = QualitySettings::from_preset(QualityPreset::Ultra)
            .apply_tier_fallback(CapabilityTier::Tier0);
        assert!(!q.taa_enabled);
        assert!(!q.ssao_enabled);
        assert_eq!(q.local_shadow_slots, 0);
    }
}
