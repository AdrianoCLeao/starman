#![allow(dead_code)]

//! Cascaded shadow maps + local shadow atlas/cubemap budgets.

use engine_math::{Mat4, Vec3};

pub const MAX_CASCADES: usize = 4;
pub const MAX_LOCAL_SHADOWS: usize = 2;

#[derive(Clone, Debug)]
pub struct CascadeSplit {
    pub near: f32,
    pub far: f32,
    pub view_proj: Mat4,
}

#[derive(Clone, Debug)]
pub struct CsmData {
    pub cascades: Vec<CascadeSplit>,
    pub light_direction: Vec3,
}

/// Practical split scheme (lambda blends log and uniform).
pub fn cascade_split_depths(
    camera_near: f32,
    camera_far: f32,
    cascade_count: u32,
    lambda: f32,
) -> Vec<f32> {
    let n = cascade_count.max(1);
    let mut splits = Vec::with_capacity(n as usize);
    let ratio = camera_far / camera_near.max(1e-3);
    for i in 1..=n {
        let p = i as f32 / n as f32;
        let log_d = camera_near * ratio.powf(p);
        let uni_d = camera_near + (camera_far - camera_near) * p;
        splits.push(lambda * log_d + (1.0 - lambda) * uni_d);
    }
    splits
}

/// Orthographic light view-projection covering a camera frustum slice.
pub fn cascade_view_proj(light_dir: Vec3, center: Vec3, radius: f32) -> Mat4 {
    let light_dir = light_dir.normalize_or_zero();
    let up = if light_dir.y.abs() > 0.99 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let eye = center - light_dir * (radius * 2.0);
    let view = Mat4::look_at_rh(eye, center, up);
    let ortho = Mat4::orthographic_rh(-radius, radius, -radius, radius, 0.1, radius * 4.0);
    ortho * view
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalShadowKind {
    Spot,
    Point,
}

#[derive(Clone, Debug)]
pub struct LocalShadowSlot {
    pub light_index: u32,
    pub kind: LocalShadowKind,
    pub atlas_index: u32,
}

/// Pick up to `budget` local shadow casters by intensity × rough importance.
pub fn select_local_shadow_casters(
    lights: &[(u32, f32, LocalShadowKind)],
    budget: u32,
) -> Vec<LocalShadowSlot> {
    let mut ranked = lights.to_vec();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked
        .into_iter()
        .take(budget as usize)
        .enumerate()
        .map(|(i, (light_index, _, kind))| LocalShadowSlot {
            light_index,
            kind,
            atlas_index: i as u32,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_are_monotonic() {
        let splits = cascade_split_depths(0.1, 100.0, 4, 0.5);
        assert_eq!(splits.len(), 4);
        for w in splits.windows(2) {
            assert!(w[1] > w[0]);
        }
    }

    #[test]
    fn local_budget_respected() {
        let lights = vec![
            (0, 10.0, LocalShadowKind::Spot),
            (1, 5.0, LocalShadowKind::Point),
            (2, 1.0, LocalShadowKind::Spot),
        ];
        let slots = select_local_shadow_casters(&lights, 2);
        assert_eq!(slots.len(), 2);
        assert_eq!(slots[0].light_index, 0);
    }
}
