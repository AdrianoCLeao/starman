#![allow(dead_code)]

//! Image-based lighting: env maps, BRDF LUT, skybox, probe blend weights.

use engine_math::Vec3;

#[derive(Clone, Copy, Debug)]
pub struct ReflectionProbeData {
    pub position: Vec3,
    pub radius: f32,
    pub priority: i32,
    pub intensity: f32,
}

/// Distance-based blend weights for up to `max_probes` probes (sorted by priority then distance).
pub fn probe_blend_weights(
    camera_pos: Vec3,
    probes: &[ReflectionProbeData],
    max_probes: usize,
) -> Vec<(usize, f32)> {
    let mut scored: Vec<(usize, f32, i32)> = probes
        .iter()
        .enumerate()
        .filter_map(|(i, p)| {
            let d = (camera_pos - p.position).length();
            if d > p.radius.max(1e-3) {
                return None;
            }
            let w = (1.0 - d / p.radius) * p.intensity;
            Some((i, w, p.priority))
        })
        .collect();
    scored.sort_by(|a, b| {
        b.2.cmp(&a.2)
            .then_with(|| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
    });
    scored
        .into_iter()
        .take(max_probes)
        .map(|(i, w, _)| (i, w))
        .collect()
}

/// Rough irradiance SH L0 term from a solid color env (placeholder bake input).
pub fn solid_irradiance(color: [f32; 3]) -> [f32; 3] {
    [color[0] * 0.5, color[1] * 0.5, color[2] * 0.5]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blend_picks_nearest_inside_radius() {
        let probes = [
            ReflectionProbeData {
                position: Vec3::ZERO,
                radius: 10.0,
                priority: 0,
                intensity: 1.0,
            },
            ReflectionProbeData {
                position: Vec3::new(100.0, 0.0, 0.0),
                radius: 5.0,
                priority: 10,
                intensity: 1.0,
            },
        ];
        let w = probe_blend_weights(Vec3::new(1.0, 0.0, 0.0), &probes, 4);
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].0, 0);
        assert!(w[0].1 > 0.0);
    }
}
