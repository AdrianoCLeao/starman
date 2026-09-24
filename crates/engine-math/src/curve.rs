//! Keyframed curves with step, linear and cubic (Hermite) interpolation,
//! shared by animation tracks, particle modules and the editor's curve
//! widgets. A `Curve<Vec4>` doubles as a color gradient.

use crate::{Quat, Vec2, Vec3, Vec4};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Interpolation {
    Step,
    #[default]
    Linear,
    /// Cubic Hermite with explicit in/out tangents per key (glTF
    /// `CUBICSPLINE` semantics: tangents are per second).
    Cubic,
}

/// A value that can be keyed, blended and interpolated.
pub trait Animatable: Copy + PartialEq + Default + Send + Sync + 'static {
    fn lerp_value(a: Self, b: Self, t: f32) -> Self;
    /// Hermite interpolation between `p0` (out tangent `m0`) and `p1` (in
    /// tangent `m1`) over a segment of `dt` seconds.
    fn hermite(p0: Self, m0: Self, p1: Self, m1: Self, t: f32, dt: f32) -> Self;
}

fn hermite_weights(t: f32) -> (f32, f32, f32, f32) {
    let t2 = t * t;
    let t3 = t2 * t;
    (
        2.0 * t3 - 3.0 * t2 + 1.0,
        t3 - 2.0 * t2 + t,
        -2.0 * t3 + 3.0 * t2,
        t3 - t2,
    )
}

macro_rules! vector_animatable {
    ($($ty:ty),*) => {$(
        impl Animatable for $ty {
            fn lerp_value(a: Self, b: Self, t: f32) -> Self {
                a + (b - a) * t
            }

            fn hermite(p0: Self, m0: Self, p1: Self, m1: Self, t: f32, dt: f32) -> Self {
                let (h00, h10, h01, h11) = hermite_weights(t);
                p0 * h00 + m0 * (h10 * dt) + p1 * h01 + m1 * (h11 * dt)
            }
        }
    )*};
}

vector_animatable!(f32, Vec2, Vec3, Vec4);

impl Animatable for Quat {
    fn lerp_value(a: Self, b: Self, t: f32) -> Self {
        a.slerp(b, t)
    }

    fn hermite(p0: Self, m0: Self, p1: Self, m1: Self, t: f32, dt: f32) -> Self {
        let v = Vec4::hermite(
            Vec4::from(p0),
            Vec4::from(m0),
            Vec4::from(p1),
            Vec4::from(m1),
            t,
            dt,
        );
        Quat::from_vec4(v).normalize()
    }
}

impl Animatable for bool {
    fn lerp_value(a: Self, b: Self, t: f32) -> Self {
        if t < 1.0 {
            a
        } else {
            b
        }
    }

    fn hermite(p0: Self, _m0: Self, p1: Self, _m1: Self, t: f32, _dt: f32) -> Self {
        Self::lerp_value(p0, p1, t)
    }
}

/// Keys at strictly increasing `times` (seconds).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(bound(serialize = "T: Serialize", deserialize = "T: Deserialize<'de>"))]
pub struct Curve<T> {
    #[serde(default)]
    pub interpolation: Interpolation,
    pub times: Vec<f32>,
    pub values: Vec<T>,
    /// Per-key in tangents (cubic only).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub in_tangents: Vec<T>,
    /// Per-key out tangents (cubic only).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub out_tangents: Vec<T>,
}

impl<T: Animatable> Default for Curve<T> {
    fn default() -> Self {
        Self::new(Interpolation::Linear)
    }
}

impl<T: Animatable> Curve<T> {
    pub fn new(interpolation: Interpolation) -> Self {
        Self {
            interpolation,
            times: Vec::new(),
            values: Vec::new(),
            in_tangents: Vec::new(),
            out_tangents: Vec::new(),
        }
    }

    pub fn constant(value: T) -> Self {
        let mut curve = Self::new(Interpolation::Step);
        curve.times.push(0.0);
        curve.values.push(value);
        curve
    }

    pub fn linear(keys: impl IntoIterator<Item = (f32, T)>) -> Self {
        let mut curve = Self::new(Interpolation::Linear);
        for (time, value) in keys {
            curve.insert_key(time, value);
        }
        curve
    }

    pub fn len(&self) -> usize {
        self.times.len()
    }

    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    pub fn duration(&self) -> f32 {
        self.times.last().copied().unwrap_or(0.0)
    }

    fn cubic(&self) -> bool {
        self.interpolation == Interpolation::Cubic
            && self.in_tangents.len() == self.values.len()
            && self.out_tangents.len() == self.values.len()
    }

    /// Inserts (or replaces) the key at `time`, keeping keys sorted.
    /// Returns the key index.
    pub fn insert_key(&mut self, time: f32, value: T) -> usize {
        let index = self.times.partition_point(|t| *t < time);
        if self
            .times
            .get(index)
            .is_some_and(|t| (*t - time).abs() < 1e-5)
        {
            self.values[index] = value;
            return index;
        }
        self.times.insert(index, time);
        self.values.insert(index, value);
        if self.interpolation == Interpolation::Cubic {
            self.in_tangents
                .insert(index.min(self.in_tangents.len()), T::default());
            self.out_tangents
                .insert(index.min(self.out_tangents.len()), T::default());
        }
        index
    }

    pub fn remove_key(&mut self, index: usize) {
        if index >= self.times.len() {
            return;
        }
        self.times.remove(index);
        self.values.remove(index);
        if index < self.in_tangents.len() {
            self.in_tangents.remove(index);
        }
        if index < self.out_tangents.len() {
            self.out_tangents.remove(index);
        }
    }

    /// Value at `time` (clamped to the first/last key).
    pub fn sample(&self, time: f32) -> Option<T> {
        let last = self.times.len().checked_sub(1)?;
        if last == 0 || time <= self.times[0] {
            return self.values.first().copied();
        }
        if time >= self.times[last] {
            return self.values.get(last).copied();
        }
        let next = self.times.partition_point(|t| *t <= time).min(last);
        let prev = next - 1;
        let (t0, t1) = (self.times[prev], self.times[next]);
        let dt = (t1 - t0).max(1e-6);
        let s = ((time - t0) / dt).clamp(0.0, 1.0);
        let (a, b) = (self.values[prev], self.values[next]);
        Some(match self.interpolation {
            Interpolation::Step => a,
            Interpolation::Linear => T::lerp_value(a, b, s),
            Interpolation::Cubic if self.cubic() => {
                T::hermite(a, self.out_tangents[prev], b, self.in_tangents[next], s, dt)
            }
            Interpolation::Cubic => T::lerp_value(a, b, s),
        })
    }

    /// Checks structural invariants (for loaders and editor validation).
    pub fn validate(&self) -> Result<(), String> {
        if self.times.len() != self.values.len() {
            return Err(format!(
                "{} times but {} values",
                self.times.len(),
                self.values.len()
            ));
        }
        if self
            .times
            .windows(2)
            .any(|w| w[1] <= w[0] || !w[0].is_finite())
        {
            return Err("key times must be finite and strictly increasing".to_owned());
        }
        if self.interpolation == Interpolation::Cubic
            && (!self.in_tangents.is_empty() || !self.out_tangents.is_empty())
            && !self.cubic()
        {
            return Err("cubic curves need one in and out tangent per key".to_owned());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_step_and_clamping() {
        let curve = Curve::linear([(0.0, 0.0f32), (1.0, 10.0), (2.0, 0.0)]);
        assert_eq!(curve.sample(-1.0), Some(0.0));
        assert_eq!(curve.sample(0.5), Some(5.0));
        assert_eq!(curve.sample(1.5), Some(5.0));
        assert_eq!(curve.sample(9.0), Some(0.0));
        let mut step = curve.clone();
        step.interpolation = Interpolation::Step;
        assert_eq!(step.sample(0.99), Some(0.0));
        assert_eq!(step.sample(1.0), Some(10.0));
        assert!(Curve::<f32>::default().sample(0.0).is_none());
    }

    #[test]
    fn cubic_uses_tangents() {
        let mut curve = Curve::new(Interpolation::Cubic);
        curve.insert_key(0.0, 0.0f32);
        curve.insert_key(1.0, 0.0);
        curve.out_tangents[0] = 4.0;
        let mid = curve.sample(0.5).unwrap();
        assert!((mid - 0.5).abs() < 1e-5, "{mid}");
        curve.validate().unwrap();
    }

    #[test]
    fn quaternions_stay_normalized() {
        let curve = Curve::linear([(0.0, Quat::IDENTITY), (1.0, Quat::from_rotation_y(2.0))]);
        let q = curve.sample(0.5).unwrap();
        assert!((q.length() - 1.0).abs() < 1e-5);
        assert!(q.angle_between(Quat::from_rotation_y(1.0)) < 1e-4);
    }

    #[test]
    fn keys_stay_sorted_and_replace_in_place() {
        let mut curve = Curve::linear([(1.0, 1.0f32)]);
        curve.insert_key(0.0, 0.0);
        curve.insert_key(1.0, 2.0);
        assert_eq!(curve.times, vec![0.0, 1.0]);
        assert_eq!(curve.values, vec![0.0, 2.0]);
        curve.remove_key(0);
        assert_eq!(curve.len(), 1);
        let bad = Curve {
            interpolation: Interpolation::Linear,
            times: vec![1.0, 0.5],
            values: vec![0.0f32, 1.0],
            in_tangents: vec![],
            out_tangents: vec![],
        };
        assert!(bad.validate().is_err());
    }
}
