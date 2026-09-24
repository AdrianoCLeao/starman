//! Forward+ clustered light assignment (ADR 0009).
//!
//! The view frustum is split into screen tiles (`tile_px` pixels) and
//! exponentially spaced depth slices. Each local light's bounding sphere is
//! projected to a conservative tile rectangle and depth-slice range, and
//! its index is appended to every covered cluster. The fragment shader
//! finds its cluster from `frag_coord` and view depth and loops only over
//! that cluster's lights. Directional lights are not clustered: they come
//! first in the light buffer and are counted in the view uniform.

use engine_math::{Mat4, Vec3, Vec4};

use crate::capabilities::CapabilityTier;
use crate::extract::{ExtractedLight, ExtractedLightKind};

pub const DEFAULT_TILE_PX: u32 = 64;
pub const DEFAULT_DEPTH_SLICES: u32 = 24;
pub const MAX_LIGHTS_PER_CLUSTER: usize = 96;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable, PartialEq)]
pub struct GpuLight {
    pub position_range: [f32; 4],
    pub color_intensity: [f32; 4],
    pub direction_cone: [f32; 4],
    /// type, first shadow matrix (or [`GpuLight::NO_SHADOW`]), inner cone
    /// bits, unused.
    pub params: [u32; 4],
}

impl GpuLight {
    pub const TYPE_DIRECTIONAL: u32 = 0;
    pub const TYPE_POINT: u32 = 1;
    pub const TYPE_SPOT: u32 = 2;
    pub const NO_SHADOW: u32 = u32::MAX;

    pub fn light_type(&self) -> u32 {
        self.params[0]
    }

    pub fn from_extracted(light: &ExtractedLight) -> Self {
        match light.kind {
            ExtractedLightKind::Directional {
                direction,
                color,
                intensity,
            } => Self {
                position_range: [0.0; 4],
                color_intensity: [color[0], color[1], color[2], intensity],
                direction_cone: [direction[0], direction[1], direction[2], 0.0],
                params: [Self::TYPE_DIRECTIONAL, Self::NO_SHADOW, 0, 0],
            },
            ExtractedLightKind::Point {
                position,
                color,
                intensity,
                range,
            } => Self {
                position_range: [position[0], position[1], position[2], range],
                color_intensity: [color[0], color[1], color[2], intensity],
                direction_cone: [0.0; 4],
                params: [Self::TYPE_POINT, Self::NO_SHADOW, 0, 0],
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
                params: [Self::TYPE_SPOT, Self::NO_SHADOW, inner_cone.to_bits(), 0],
            },
        }
    }
}

/// Orders lights directional-first (as the shader expects) and caps the
/// total. Returns the lights and the directional count.
pub fn build_gpu_lights(lights: &[ExtractedLight], max_lights: usize) -> (Vec<GpuLight>, u32) {
    let mut directional: Vec<GpuLight> = Vec::new();
    let mut local: Vec<GpuLight> = Vec::new();
    for light in lights {
        let gpu = GpuLight::from_extracted(light);
        if gpu.light_type() == GpuLight::TYPE_DIRECTIONAL {
            directional.push(gpu);
        } else {
            local.push(gpu);
        }
    }
    let dir_count = directional.len().min(max_lights);
    directional.truncate(dir_count);
    local.truncate(max_lights - dir_count);
    let mut all = directional;
    all.extend(local);
    (all, dir_count as u32)
}

/// Cluster grid description shared with the shader.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClusterLayout {
    pub tiles_x: u32,
    pub tiles_y: u32,
    pub slices: u32,
    pub tile_px: u32,
    /// `slice = log2(depth) * scale - bias`.
    pub scale: f32,
    pub bias: f32,
    pub near: f32,
    pub far: f32,
}

impl ClusterLayout {
    pub fn new(width: u32, height: u32, near: f32, far: f32, tile_px: u32, slices: u32) -> Self {
        let tile_px = tile_px.max(8);
        let near = near.max(1e-3);
        let far = far.max(near * 1.001);
        let log_ratio = (far / near).log2();
        let scale = slices as f32 / log_ratio;
        let bias = slices as f32 * near.log2() / log_ratio;
        Self {
            tiles_x: width.max(1).div_ceil(tile_px),
            tiles_y: height.max(1).div_ceil(tile_px),
            slices: slices.max(1),
            tile_px,
            scale,
            bias,
            near,
            far,
        }
    }

    pub fn cluster_count(&self) -> usize {
        (self.tiles_x * self.tiles_y * self.slices) as usize
    }

    pub fn index(&self, x: u32, y: u32, z: u32) -> usize {
        ((z * self.tiles_y + y) * self.tiles_x + x) as usize
    }

    /// Depth slice of a positive view-space depth.
    pub fn slice_of(&self, depth: f32) -> u32 {
        let slice = depth.max(1e-4).log2() * self.scale - self.bias;
        (slice.max(0.0) as u32).min(self.slices - 1)
    }

    pub fn dims(&self) -> [u32; 4] {
        [self.tiles_x, self.tiles_y, self.slices, self.tile_px]
    }

    pub fn params(&self) -> [f32; 4] {
        [self.scale, self.bias, self.near, self.far]
    }
}

/// Packed cluster → light lists, ready to upload.
#[derive(Clone, Debug, Default)]
pub struct ClusterGrid {
    pub layout: Option<ClusterLayout>,
    /// Per cluster: (offset into `indices`, count).
    pub ranges: Vec<[u32; 2]>,
    pub indices: Vec<u32>,
    /// Kept for the debug dump and older callers.
    pub light_indices: Vec<Vec<u32>>,
}

impl ClusterGrid {
    pub fn cluster_count(&self) -> usize {
        self.ranges.len()
    }

    pub fn max_lights_in_cluster(&self) -> u32 {
        self.ranges.iter().map(|range| range[1]).max().unwrap_or(0)
    }
}

/// Assigns local lights (indices into `lights`, which is directional-first)
/// to clusters.
pub fn assign_clusters(
    layout: ClusterLayout,
    lights: &[GpuLight],
    view: Mat4,
    proj: Mat4,
    width: u32,
    height: u32,
) -> ClusterGrid {
    let count = layout.cluster_count();
    let mut lists: Vec<Vec<u32>> = vec![Vec::new(); count];
    let (w, h) = (width.max(1) as f32, height.max(1) as f32);

    for (index, light) in lights.iter().enumerate() {
        if light.light_type() == GpuLight::TYPE_DIRECTIONAL {
            continue;
        }
        let range = light.position_range[3].max(0.0);
        let center = view.transform_point3(Vec3::new(
            light.position_range[0],
            light.position_range[1],
            light.position_range[2],
        ));
        let depth = -center.z;
        if depth + range < layout.near || depth - range > layout.far {
            continue;
        }
        let z0 = layout.slice_of((depth - range).max(layout.near));
        let z1 = layout.slice_of((depth + range).min(layout.far));

        // Conservative screen rectangle from the sphere's view-space AABB.
        let mut min = [f32::MAX; 2];
        let mut max = [f32::MIN; 2];
        let mut full_screen = false;
        for corner in 0..8 {
            let offset = Vec3::new(
                if corner & 1 == 0 { -range } else { range },
                if corner & 2 == 0 { -range } else { range },
                if corner & 4 == 0 { -range } else { range },
            );
            let p = center + offset;
            if -p.z <= layout.near {
                full_screen = true;
                break;
            }
            let clip = proj * Vec4::new(p.x, p.y, p.z, 1.0);
            let ndc = [clip.x / clip.w, clip.y / clip.w];
            let px = [(ndc[0] * 0.5 + 0.5) * w, (0.5 - ndc[1] * 0.5) * h];
            for axis in 0..2 {
                min[axis] = min[axis].min(px[axis]);
                max[axis] = max[axis].max(px[axis]);
            }
        }
        let (x0, x1, y0, y1) = if full_screen {
            (0, layout.tiles_x - 1, 0, layout.tiles_y - 1)
        } else {
            if max[0] < 0.0 || max[1] < 0.0 || min[0] > w || min[1] > h {
                continue;
            }
            let tile = layout.tile_px as f32;
            let clamp_x = |v: f32| ((v / tile).max(0.0) as u32).min(layout.tiles_x - 1);
            let clamp_y = |v: f32| ((v / tile).max(0.0) as u32).min(layout.tiles_y - 1);
            (
                clamp_x(min[0]),
                clamp_x(max[0]),
                clamp_y(min[1]),
                clamp_y(max[1]),
            )
        };
        for z in z0..=z1 {
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let list = &mut lists[layout.index(x, y, z)];
                    if list.len() < MAX_LIGHTS_PER_CLUSTER {
                        list.push(index as u32);
                    }
                }
            }
        }
    }

    let mut ranges = Vec::with_capacity(count);
    let mut indices = Vec::new();
    for list in &lists {
        ranges.push([indices.len() as u32, list.len() as u32]);
        indices.extend_from_slice(list);
    }
    if indices.is_empty() {
        // Storage bindings may not be empty.
        indices.push(0);
    }
    ClusterGrid {
        layout: Some(layout),
        ranges,
        indices,
        light_indices: lists,
    }
}

/// Back-compat helper (benchmarks, debug): clusters `lights` for a default
/// 1280x720 view looking down -Z from `camera_position`.
pub fn cull_lights_cpu(
    lights: &[GpuLight],
    camera_position: [f32; 3],
    _tier: CapabilityTier,
) -> ClusterGrid {
    let eye = Vec3::from_array(camera_position);
    let view = Mat4::look_at_rh(eye, eye + Vec3::NEG_Z, Vec3::Y);
    let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 16.0 / 9.0, 0.1, 500.0);
    let layout = ClusterLayout::new(1280, 720, 0.1, 500.0, DEFAULT_TILE_PX, DEFAULT_DEPTH_SLICES);
    assign_clusters(layout, lights, view, proj, 1280, 720)
}

/// Legacy packing API: (offsets, flat indices).
pub fn pack_cluster_buffers(grid: &ClusterGrid) -> (Vec<u32>, Vec<u32>) {
    let offsets = grid.ranges.iter().map(|range| range[0]).collect();
    (offsets, grid.indices.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(position: [f32; 3], range: f32) -> GpuLight {
        GpuLight {
            position_range: [position[0], position[1], position[2], range],
            color_intensity: [1.0; 4],
            direction_cone: [0.0; 4],
            params: [GpuLight::TYPE_POINT, GpuLight::NO_SHADOW, 0, 0],
        }
    }

    #[test]
    fn slices_are_monotonic_and_bounded() {
        let layout = ClusterLayout::new(1920, 1080, 0.1, 1000.0, 64, 24);
        assert_eq!(layout.tiles_x, 30);
        assert_eq!(layout.tiles_y, 17);
        assert_eq!(layout.slice_of(0.1), 0);
        assert_eq!(layout.slice_of(1000.0), 23);
        let mut previous = 0;
        for depth in [0.2, 1.0, 5.0, 20.0, 100.0, 900.0] {
            let slice = layout.slice_of(depth);
            assert!(slice >= previous);
            previous = slice;
        }
    }

    #[test]
    fn a_light_in_front_of_the_camera_lands_in_the_center_clusters_only() {
        let view = Mat4::IDENTITY;
        let proj = Mat4::perspective_rh(1.0, 1.0, 0.1, 100.0);
        let layout = ClusterLayout::new(512, 512, 0.1, 100.0, 64, 16);
        let lights = vec![point([0.0, 0.0, -10.0], 1.0)];
        let grid = assign_clusters(layout, &lights, view, proj, 512, 512);
        let covered: Vec<usize> = grid
            .ranges
            .iter()
            .enumerate()
            .filter(|(_, range)| range[1] > 0)
            .map(|(index, _)| index)
            .collect();
        assert!(!covered.is_empty());
        // Nothing in the corner tile.
        let slice = layout.slice_of(10.0);
        assert_eq!(grid.ranges[layout.index(0, 0, slice)][1], 0);
        assert!(grid.ranges[layout.index(4, 4, slice)][1] >= 1);
        // Nothing in far slices.
        assert_eq!(grid.ranges[layout.index(4, 4, 15)][1], 0);
    }

    #[test]
    fn lights_behind_the_camera_are_skipped_and_overlapping_ones_fill_the_screen() {
        let proj = Mat4::perspective_rh(1.0, 1.0, 0.1, 100.0);
        let layout = ClusterLayout::new(256, 256, 0.1, 100.0, 64, 8);
        let behind = assign_clusters(
            layout,
            &[point([0.0, 0.0, 20.0], 2.0)],
            Mat4::IDENTITY,
            proj,
            256,
            256,
        );
        assert!(behind.ranges.iter().all(|range| range[1] == 0));
        let around = assign_clusters(
            layout,
            &[point([0.0, 0.0, 0.0], 5.0)],
            Mat4::IDENTITY,
            proj,
            256,
            256,
        );
        let near_slice = layout.slice_of(0.2);
        assert!(around.ranges[layout.index(0, 0, near_slice)][1] == 1);
    }

    #[test]
    fn directional_lights_come_first() {
        let lights = vec![
            ExtractedLight {
                entity: bevy_ecs::entity::Entity::from_raw(1),
                kind: ExtractedLightKind::Point {
                    position: [0.0; 3],
                    color: [1.0; 3],
                    intensity: 1.0,
                    range: 1.0,
                },
                cast_shadows: false,
            },
            ExtractedLight {
                entity: bevy_ecs::entity::Entity::from_raw(2),
                kind: ExtractedLightKind::Directional {
                    direction: [0.0, -1.0, 0.0],
                    color: [1.0; 3],
                    intensity: 1.0,
                },
                cast_shadows: true,
            },
        ];
        let (gpu, dir) = build_gpu_lights(&lights, 8);
        assert_eq!(dir, 1);
        assert_eq!(gpu[0].light_type(), GpuLight::TYPE_DIRECTIONAL);
        assert_eq!(gpu[1].light_type(), GpuLight::TYPE_POINT);
    }
}
