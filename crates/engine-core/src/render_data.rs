//! Render-facing data that gameplay subsystems produce without depending
//! on the renderer: immediate-mode debug geometry and skinning palettes.

use bevy_ecs::prelude::{Component, Resource};
use engine_math::glam::{Mat4, Quat, Vec3};

/// sRGB color with alpha, `[r, g, b, a]` in 0..1.
pub type DebugColor = [f32; 4];

/// One debug line segment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DebugLine {
    pub start: Vec3,
    pub end: Vec3,
    pub color: DebugColor,
    /// Hidden behind geometry when `true`; drawn on top otherwise.
    pub depth_test: bool,
}

/// Named debug categories that can be toggled from the editor/runner.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DebugCategory {
    Physics,
    Navigation,
    Ai,
    Animation,
    Audio,
    Particles,
    Ui,
    Gameplay,
}

impl DebugCategory {
    pub const ALL: [Self; 8] = [
        Self::Physics,
        Self::Navigation,
        Self::Ai,
        Self::Animation,
        Self::Audio,
        Self::Particles,
        Self::Ui,
        Self::Gameplay,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Physics => "physics",
            Self::Navigation => "navigation",
            Self::Ai => "ai",
            Self::Animation => "animation",
            Self::Audio => "audio",
            Self::Particles => "particles",
            Self::Ui => "ui",
            Self::Gameplay => "gameplay",
        }
    }

    fn bit(self) -> u32 {
        1 << self as u32
    }
}

/// Immediate-mode debug drawing. Systems push shapes every frame they
/// want them visible; the renderer consumes and clears the buffer after
/// drawing (or the runtime clears it at frame start when nothing renders).
#[derive(Resource, Debug)]
pub struct DebugDraw {
    lines: Vec<DebugLine>,
    enabled: u32,
    /// Upper bound on buffered lines (hardening); extra lines are dropped.
    pub max_lines: usize,
    dropped: usize,
}

impl Default for DebugDraw {
    fn default() -> Self {
        Self {
            lines: Vec::new(),
            enabled: 0,
            max_lines: 262_144,
            dropped: 0,
        }
    }
}

impl DebugDraw {
    pub fn is_enabled(&self, category: DebugCategory) -> bool {
        self.enabled & category.bit() != 0
    }

    pub fn set_enabled(&mut self, category: DebugCategory, enabled: bool) {
        if enabled {
            self.enabled |= category.bit();
        } else {
            self.enabled &= !category.bit();
        }
    }

    pub fn lines(&self) -> &[DebugLine] {
        &self.lines
    }

    /// Lines dropped because `max_lines` was exceeded since the last clear.
    pub fn dropped(&self) -> usize {
        self.dropped
    }

    pub fn clear(&mut self) {
        self.lines.clear();
        self.dropped = 0;
    }

    /// Takes every buffered line, leaving the buffer empty.
    pub fn take_lines(&mut self) -> Vec<DebugLine> {
        self.dropped = 0;
        std::mem::take(&mut self.lines)
    }

    pub fn line(&mut self, start: Vec3, end: Vec3, color: DebugColor) {
        self.push(start, end, color, true);
    }

    pub fn line_overlay(&mut self, start: Vec3, end: Vec3, color: DebugColor) {
        self.push(start, end, color, false);
    }

    fn push(&mut self, start: Vec3, end: Vec3, color: DebugColor, depth_test: bool) {
        if self.lines.len() >= self.max_lines {
            self.dropped += 1;
            return;
        }
        self.lines.push(DebugLine {
            start,
            end,
            color,
            depth_test,
        });
    }

    pub fn arrow(&mut self, start: Vec3, end: Vec3, color: DebugColor) {
        self.line(start, end, color);
        let dir = end - start;
        let len = dir.length();
        if len < 1e-5 {
            return;
        }
        let dir = dir / len;
        let side = if dir.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
        let a = dir.cross(side).normalize() * len * 0.1;
        let back = end - dir * len * 0.2;
        self.line(end, back + a, color);
        self.line(end, back - a, color);
    }

    pub fn cross(&mut self, center: Vec3, size: f32, color: DebugColor) {
        for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
            self.line(center - axis * size, center + axis * size, color);
        }
    }

    /// An oriented box from its transform (`half_extents` in local space).
    pub fn oriented_box(
        &mut self,
        center: Vec3,
        rotation: Quat,
        half_extents: Vec3,
        color: DebugColor,
    ) {
        let corner =
            |x: f32, y: f32, z: f32| center + rotation * (half_extents * Vec3::new(x, y, z));
        let c = [
            corner(-1.0, -1.0, -1.0),
            corner(1.0, -1.0, -1.0),
            corner(1.0, 1.0, -1.0),
            corner(-1.0, 1.0, -1.0),
            corner(-1.0, -1.0, 1.0),
            corner(1.0, -1.0, 1.0),
            corner(1.0, 1.0, 1.0),
            corner(-1.0, 1.0, 1.0),
        ];
        for (a, b) in [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ] {
            self.line(c[a], c[b], color);
        }
    }

    pub fn aabb(&mut self, min: Vec3, max: Vec3, color: DebugColor) {
        self.oriented_box((min + max) * 0.5, Quat::IDENTITY, (max - min) * 0.5, color);
    }

    pub fn circle(
        &mut self,
        center: Vec3,
        normal: Vec3,
        radius: f32,
        color: DebugColor,
        segments: usize,
    ) {
        let normal = normal.normalize_or_zero();
        if normal == Vec3::ZERO {
            return;
        }
        let side = if normal.y.abs() < 0.9 {
            Vec3::Y
        } else {
            Vec3::X
        };
        let u = normal.cross(side).normalize() * radius;
        let v = normal.cross(u.normalize()) * radius;
        let segments = segments.max(3);
        let mut previous = center + u;
        for i in 1..=segments {
            let angle = i as f32 / segments as f32 * std::f32::consts::TAU;
            let point = center + u * angle.cos() + v * angle.sin();
            self.line(previous, point, color);
            previous = point;
        }
    }

    pub fn sphere(&mut self, center: Vec3, radius: f32, color: DebugColor) {
        for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
            self.circle(center, axis, radius, color, 24);
        }
    }

    /// A Y-aligned capsule (as Rapier's `capsule_y`), rotated by `rotation`.
    pub fn capsule(
        &mut self,
        center: Vec3,
        rotation: Quat,
        half_height: f32,
        radius: f32,
        color: DebugColor,
    ) {
        let up = rotation * Vec3::Y;
        let top = center + up * half_height;
        let bottom = center - up * half_height;
        self.circle(top, up, radius, color, 16);
        self.circle(bottom, up, radius, color, 16);
        let side_a = rotation * Vec3::X * radius;
        let side_b = rotation * Vec3::Z * radius;
        for side in [side_a, -side_a, side_b, -side_b] {
            self.line(top + side, bottom + side, color);
        }
        // Hemisphere arcs.
        for side in [side_a, side_b] {
            let mut prev_top = top + side;
            let mut prev_bottom = bottom + side;
            for i in 1..=8 {
                let angle = i as f32 / 8.0 * std::f32::consts::PI;
                let offset = side * angle.cos();
                let lift = up * radius * angle.sin();
                let p_top = top + offset + lift;
                let p_bottom = bottom + offset - lift;
                self.line(prev_top, p_top, color);
                self.line(prev_bottom, p_bottom, color);
                prev_top = p_top;
                prev_bottom = p_bottom;
            }
        }
    }

    /// A view cone (e.g. AI sight): apex, direction, half angle, length.
    pub fn cone(
        &mut self,
        apex: Vec3,
        direction: Vec3,
        half_angle: f32,
        length: f32,
        color: DebugColor,
    ) {
        let dir = direction.normalize_or_zero();
        if dir == Vec3::ZERO {
            return;
        }
        let radius = half_angle.tan() * length;
        let base = apex + dir * length;
        self.circle(base, dir, radius, color, 20);
        let side = if dir.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
        let u = dir.cross(side).normalize() * radius;
        let v = dir.cross(u.normalize()) * radius;
        for offset in [u, -u, v, -v] {
            self.line(apex, base + offset, color);
        }
    }

    /// A polyline through `points`.
    pub fn path(&mut self, points: &[Vec3], color: DebugColor) {
        for pair in points.windows(2) {
            self.line(pair[0], pair[1], color);
        }
    }
}

/// Model-space skinning matrices of a skinned mesh instance, written by the
/// animation system after transform propagation and read by the renderer.
/// `joint_matrices[i] = inverse(mesh global) * joint_i global * inverse_bind_i`.
#[derive(Component, Clone, Debug, Default)]
pub struct SkinPalette {
    pub joint_matrices: Vec<Mat4>,
    /// Model-space bounds enclosing the posed mesh (for culling).
    pub bounds: Option<(Vec3, Vec3)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn categories_toggle_independently() {
        let mut draw = DebugDraw::default();
        draw.set_enabled(DebugCategory::Physics, true);
        assert!(draw.is_enabled(DebugCategory::Physics));
        assert!(!draw.is_enabled(DebugCategory::Navigation));
        draw.set_enabled(DebugCategory::Physics, false);
        assert!(!draw.is_enabled(DebugCategory::Physics));
    }

    #[test]
    fn shapes_emit_lines_and_respect_the_cap() {
        let mut draw = DebugDraw::default();
        draw.aabb(Vec3::ZERO, Vec3::ONE, [1.0; 4]);
        assert_eq!(draw.lines().len(), 12);
        draw.sphere(Vec3::ZERO, 1.0, [1.0; 4]);
        assert_eq!(draw.lines().len(), 12 + 72);
        draw.max_lines = 90;
        draw.cross(Vec3::ZERO, 1.0, [1.0; 4]);
        assert_eq!(draw.lines().len(), 87);
        draw.line(Vec3::ZERO, Vec3::X, [1.0; 4]);
        draw.line(Vec3::ZERO, Vec3::X, [1.0; 4]);
        draw.line(Vec3::ZERO, Vec3::X, [1.0; 4]);
        draw.line(Vec3::ZERO, Vec3::X, [1.0; 4]);
        assert_eq!(draw.lines().len(), 90);
        assert_eq!(draw.dropped(), 1);
        assert_eq!(draw.take_lines().len(), 90);
        assert!(draw.lines().is_empty());
    }
}
