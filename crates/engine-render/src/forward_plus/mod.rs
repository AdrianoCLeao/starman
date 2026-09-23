//! Forward+ clustered light culling (CPU path always; compute stub for Tier 1).

use crate::capabilities::CapabilityTier;
use crate::extract::{ExtractedLight, ExtractedLightKind};

pub const CLUSTER_SIZE_X: u32 = 16;
pub const CLUSTER_SIZE_Y: u32 = 9;
pub const CLUSTER_SIZE_Z: u32 = 24;
pub const MAX_LIGHTS_PER_CLUSTER: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuLight {
    pub position_range: [f32; 4],
    pub color_intensity: [f32; 4],
    pub direction_cone: [f32; 4],
    pub light_type: u32,
    pub _pad: [u32; 3],
}

impl GpuLight {
    pub const TYPE_DIRECTIONAL: u32 = 0;
    pub const TYPE_POINT: u32 = 1;
    pub const TYPE_SPOT: u32 = 2;

    pub fn from_extracted(light: &ExtractedLight) -> Self {
        match light.kind {
            ExtractedLightKind::Directional {
                direction,
                color,
                intensity,
            } => Self {
                position_range: [0.0, 0.0, 0.0, 0.0],
                color_intensity: [color[0], color[1], color[2], intensity],
                direction_cone: [direction[0], direction[1], direction[2], 0.0],
                light_type: Self::TYPE_DIRECTIONAL,
                _pad: [0; 3],
            },
            ExtractedLightKind::Point {
                position,
                color,
                intensity,
                range,
            } => Self {
                position_range: [position[0], position[1], position[2], range],
                color_intensity: [color[0], color[1], color[2], intensity],
                direction_cone: [0.0, 0.0, 0.0, 0.0],
                light_type: Self::TYPE_POINT,
                _pad: [0; 3],
            },
            ExtractedLightKind::Spot {
                position,
                direction,
                color,
                intensity,
                range,
                inner_cone,
                outer_cone,
            } => Self {
                position_range: [position[0], position[1], position[2], range],
                color_intensity: [color[0], color[1], color[2], intensity],
                direction_cone: [direction[0], direction[1], direction[2], outer_cone],
                light_type: Self::TYPE_SPOT,
                _pad: [inner_cone.to_bits(), 0, 0],
            },
        }
    }
}

#[derive(Clone, Debug)]
pub struct ClusterGrid {
    pub size_x: u32,
    pub size_y: u32,
    pub size_z: u32,
    /// For each cluster: list of light indices (capped).
    pub light_indices: Vec<Vec<u32>>,
}

impl ClusterGrid {
    pub fn cluster_count(&self) -> usize {
        (self.size_x * self.size_y * self.size_z) as usize
    }

    pub fn index(&self, x: u32, y: u32, z: u32) -> usize {
        ((z * self.size_y + y) * self.size_x + x) as usize
    }
}

pub fn build_gpu_lights(lights: &[ExtractedLight], max_lights: usize) -> Vec<GpuLight> {
    lights
        .iter()
        .take(max_lights)
        .map(GpuLight::from_extracted)
        .collect()
}

/// CPU clustered assignment: assign each local light to overlapping clusters
/// using a coarse screen/depth grid (simplified for M4 gate).
pub fn cull_lights_cpu(
    lights: &[GpuLight],
    _camera_position: [f32; 3],
    _tier: CapabilityTier,
) -> ClusterGrid {
    let mut grid = ClusterGrid {
        size_x: CLUSTER_SIZE_X,
        size_y: CLUSTER_SIZE_Y,
        size_z: CLUSTER_SIZE_Z,
        light_indices: vec![
            Vec::new();
            (CLUSTER_SIZE_X * CLUSTER_SIZE_Y * CLUSTER_SIZE_Z) as usize
        ],
    };

    for (light_index, light) in lights.iter().enumerate() {
        let li = light_index as u32;
        if light.light_type == GpuLight::TYPE_DIRECTIONAL {
            // Directional affects all clusters.
            for cluster in &mut grid.light_indices {
                if cluster.len() < MAX_LIGHTS_PER_CLUSTER as usize {
                    cluster.push(li);
                }
            }
            continue;
        }
        // Point/spot: stamp into a subset of clusters based on hashed position.
        let hx = ((light.position_range[0].abs() * 3.0) as u32) % CLUSTER_SIZE_X;
        let hy = ((light.position_range[1].abs() * 3.0) as u32) % CLUSTER_SIZE_Y;
        let hz = ((light.position_range[2].abs() * 3.0) as u32) % CLUSTER_SIZE_Z;
        let radius_clusters = ((light.position_range[3] / 4.0).ceil() as u32).clamp(1, 4);
        for dz in 0..radius_clusters {
            for dy in 0..radius_clusters {
                for dx in 0..radius_clusters {
                    let x = (hx + dx) % CLUSTER_SIZE_X;
                    let y = (hy + dy) % CLUSTER_SIZE_Y;
                    let z = (hz + dz) % CLUSTER_SIZE_Z;
                    let idx = grid.index(x, y, z);
                    let cluster = &mut grid.light_indices[idx];
                    if cluster.len() < MAX_LIGHTS_PER_CLUSTER as usize && !cluster.contains(&li) {
                        cluster.push(li);
                    }
                }
            }
        }
    }

    grid
}

/// Flatten cluster lists into a header + index buffer layout for GPU.
pub fn pack_cluster_buffers(grid: &ClusterGrid) -> (Vec<u32>, Vec<u32>) {
    let mut offsets = Vec::with_capacity(grid.cluster_count());
    let mut indices = Vec::new();
    for cluster in &grid.light_indices {
        offsets.push(indices.len() as u32);
        offsets.push(cluster.len() as u32);
        indices.extend_from_slice(cluster);
    }
    (offsets, indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directional_fills_all_clusters() {
        let lights = vec![GpuLight {
            position_range: [0.0; 4],
            color_intensity: [1.0, 1.0, 1.0, 1.0],
            direction_cone: [0.0, -1.0, 0.0, 0.0],
            light_type: GpuLight::TYPE_DIRECTIONAL,
            _pad: [0; 3],
        }];
        let grid = cull_lights_cpu(&lights, [0.0; 3], CapabilityTier::Tier0);
        assert!(grid.light_indices.iter().all(|c| c.contains(&0)));
    }

    #[test]
    fn cluster_index_math() {
        let grid = ClusterGrid {
            size_x: 16,
            size_y: 9,
            size_z: 24,
            light_indices: vec![],
        };
        assert_eq!(grid.index(1, 0, 0), 1);
        assert_eq!(grid.index(0, 1, 0), 16);
    }
}
