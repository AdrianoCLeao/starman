//! Animation clips: bone TRS tracks, generic property tracks (any
//! reflected field), timeline events and root-motion settings.
//!
//! Clips come from glTF (`file.glb#anim:<i>`) or from `*.anim.ron`
//! documents authored in the editor. An `.anim.ron` may name a glTF
//! animation as its `source` and add looping, events, property tracks and
//! root motion on top (bone tracks it declares replace the source's).

use std::path::Path;

use engine_assets::{Asset, AssetLoader, AssetRef, LoadContext};
use engine_core::Result;
use engine_math::{Quat, Vec2, Vec3, Vec4};
use serde::{Deserialize, Serialize};

use crate::curve::{Animatable, Curve, Interpolation};
use crate::gltf_import::gltf_document;
use crate::skeleton::{BoneTransform, Skeleton};

pub const ANIMATION_CLIP_VERSION: u32 = 1;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BoneTrack {
    pub bone: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub translation: Option<Curve<Vec3>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rotation: Option<Curve<Quat>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<Curve<Vec3>>,
}

/// Curve of a property track, by value type.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum PropertyCurve {
    Float(Curve<f32>),
    Vec2(Curve<Vec2>),
    /// Also drives `[f32; 3]` fields (colors).
    Vec3(Curve<Vec3>),
    /// Also drives `[f32; 4]` fields (colors with alpha).
    Vec4(Curve<Vec4>),
    Quat(Curve<Quat>),
    Bool(Curve<bool>),
}

/// A sampled property value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PropertyValue {
    Float(f32),
    Vec2(Vec2),
    Vec3(Vec3),
    Vec4(Vec4),
    Quat(Quat),
    Bool(bool),
}

impl PropertyCurve {
    pub fn sample(&self, time: f32) -> Option<PropertyValue> {
        Some(match self {
            Self::Float(c) => PropertyValue::Float(c.sample(time)?),
            Self::Vec2(c) => PropertyValue::Vec2(c.sample(time)?),
            Self::Vec3(c) => PropertyValue::Vec3(c.sample(time)?),
            Self::Vec4(c) => PropertyValue::Vec4(c.sample(time)?),
            Self::Quat(c) => PropertyValue::Quat(c.sample(time)?),
            Self::Bool(c) => PropertyValue::Bool(c.sample(time)?),
        })
    }

    pub fn duration(&self) -> f32 {
        match self {
            Self::Float(c) => c.duration(),
            Self::Vec2(c) => c.duration(),
            Self::Vec3(c) => c.duration(),
            Self::Vec4(c) => c.duration(),
            Self::Quat(c) => c.duration(),
            Self::Bool(c) => c.duration(),
        }
    }

    fn validate(&self) -> std::result::Result<(), String> {
        match self {
            Self::Float(c) => c.validate(),
            Self::Vec2(c) => c.validate(),
            Self::Vec3(c) => c.validate(),
            Self::Vec4(c) => c.validate(),
            Self::Quat(c) => c.validate(),
            Self::Bool(c) => c.validate(),
        }
    }
}

impl PropertyValue {
    /// Weighted blend (`bool` switches at 0.5).
    pub fn blend(self, other: Self, t: f32) -> Self {
        match (self, other) {
            (Self::Float(a), Self::Float(b)) => Self::Float(f32::lerp_value(a, b, t)),
            (Self::Vec2(a), Self::Vec2(b)) => Self::Vec2(a.lerp(b, t)),
            (Self::Vec3(a), Self::Vec3(b)) => Self::Vec3(a.lerp(b, t)),
            (Self::Vec4(a), Self::Vec4(b)) => Self::Vec4(a.lerp(b, t)),
            (Self::Quat(a), Self::Quat(b)) => Self::Quat(a.slerp(b, t)),
            (Self::Bool(a), Self::Bool(b)) => Self::Bool(if t < 0.5 { a } else { b }),
            (_, other) => other,
        }
    }
}

/// Animates `component.field` on the entity at `target`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PropertyTrack {
    /// Path of `EntityName`s from the animated entity (`""` = itself,
    /// `"Lid/Glow"` = grandchild).
    #[serde(default)]
    pub target: String,
    /// Component type name (full `engine_render::components::PointLight`
    /// or the short `PointLight`).
    pub component: String,
    /// Reflect path inside the component (`intensity`, `color`,
    /// `translation.y`).
    pub field: String,
    pub curve: PropertyCurve,
}

/// A named timeline event.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClipEvent {
    pub time: f32,
    pub name: String,
    /// Free-form payload (RON), handed to listeners verbatim.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub payload: String,
}

/// Which motion of the root bone becomes entity motion.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RootMotionSettings {
    /// Bone whose motion is extracted (usually the hips/root).
    pub bone: String,
    /// Extract X/Z translation.
    #[serde(default = "yes")]
    pub horizontal: bool,
    /// Extract Y translation (jumps, climbing).
    #[serde(default)]
    pub vertical: bool,
    /// Extract rotation about Y.
    #[serde(default)]
    pub yaw: bool,
}

fn yes() -> bool {
    true
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnimationClip {
    #[serde(default = "clip_version")]
    pub version: u32,
    #[serde(default)]
    pub name: String,
    /// glTF animation providing bone tracks (`.anim.ron` only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<AssetRef>,
    /// Seconds; 0 derives it from the longest track.
    #[serde(default)]
    pub duration: f32,
    #[serde(default = "yes")]
    pub looping: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bone_tracks: Vec<BoneTrack>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub property_tracks: Vec<PropertyTrack>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub events: Vec<ClipEvent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub root_motion: Option<RootMotionSettings>,
}

fn clip_version() -> u32 {
    ANIMATION_CLIP_VERSION
}

impl Default for AnimationClip {
    fn default() -> Self {
        Self {
            version: ANIMATION_CLIP_VERSION,
            name: String::new(),
            source: None,
            duration: 0.0,
            looping: true,
            bone_tracks: Vec::new(),
            property_tracks: Vec::new(),
            events: Vec::new(),
            root_motion: None,
        }
    }
}

impl Asset for AnimationClip {
    const TYPE_NAME: &'static str = "AnimationClip";
}

/// Bone index per bone track for one skeleton.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ClipBinding {
    pub bones: Vec<Option<usize>>,
    /// Bone index of the root-motion bone.
    pub root_bone: Option<usize>,
}

impl AnimationClip {
    /// Effective duration (explicit or longest track).
    pub fn length(&self) -> f32 {
        if self.duration > 0.0 {
            return self.duration;
        }
        let bones = self.bone_tracks.iter().flat_map(|track| {
            [
                track.translation.as_ref().map(Curve::duration),
                track.rotation.as_ref().map(Curve::duration),
                track.scale.as_ref().map(Curve::duration),
            ]
        });
        let properties = self
            .property_tracks
            .iter()
            .map(|t| Some(t.curve.duration()));
        let events = self.events.iter().map(|e| Some(e.time));
        bones
            .chain(properties)
            .chain(events)
            .flatten()
            .fold(0.0, f32::max)
    }

    pub fn bind(&self, skeleton: &Skeleton) -> ClipBinding {
        ClipBinding {
            bones: self
                .bone_tracks
                .iter()
                .map(|track| skeleton.bone_index(&track.bone))
                .collect(),
            root_bone: self
                .root_motion
                .as_ref()
                .and_then(|settings| skeleton.bone_index(&settings.bone)),
        }
    }

    /// Wraps or clamps a playhead time.
    pub fn wrap_time(&self, time: f32) -> f32 {
        let length = self.length();
        if length <= 0.0 {
            return 0.0;
        }
        if self.looping {
            time.rem_euclid(length)
        } else {
            time.clamp(0.0, length)
        }
    }

    /// Samples the bone tracks at `time` into `pose` (untracked channels
    /// keep their current values).
    pub fn sample_bones(&self, binding: &ClipBinding, time: f32, pose: &mut [BoneTransform]) {
        for (track, bone) in self.bone_tracks.iter().zip(&binding.bones) {
            let Some(local) = bone.and_then(|bone| pose.get_mut(bone)) else {
                continue;
            };
            if let Some(t) = track.translation.as_ref().and_then(|c| c.sample(time)) {
                local.translation = t;
            }
            if let Some(r) = track.rotation.as_ref().and_then(|c| c.sample(time)) {
                local.rotation = r;
            }
            if let Some(s) = track.scale.as_ref().and_then(|c| c.sample(time)) {
                local.scale = s;
            }
        }
    }

    /// Local transform of one bone at `time`, starting from `rest`.
    pub fn sample_bone(
        &self,
        binding: &ClipBinding,
        bone: usize,
        time: f32,
        rest: BoneTransform,
    ) -> BoneTransform {
        let mut local = rest;
        for (track, bound) in self.bone_tracks.iter().zip(&binding.bones) {
            if *bound != Some(bone) {
                continue;
            }
            if let Some(t) = track.translation.as_ref().and_then(|c| c.sample(time)) {
                local.translation = t;
            }
            if let Some(r) = track.rotation.as_ref().and_then(|c| c.sample(time)) {
                local.rotation = r;
            }
            if let Some(s) = track.scale.as_ref().and_then(|c| c.sample(time)) {
                local.scale = s;
            }
        }
        local
    }

    /// Events whose time lies in the playhead interval `(from, to]`,
    /// wrapping once around the end for looping clips (`to < from`).
    pub fn events_between(&self, from: f32, to: f32) -> impl Iterator<Item = &ClipEvent> {
        let wrapped = to < from;
        self.events.iter().filter(move |event| {
            if wrapped {
                event.time > from || event.time <= to
            } else {
                event.time > from && event.time <= to
            }
        })
    }

    pub fn validate(&self) -> std::result::Result<(), Vec<String>> {
        let mut errors = Vec::new();
        if self.version > ANIMATION_CLIP_VERSION {
            errors.push(format!(
                "clip version {} is newer than supported {ANIMATION_CLIP_VERSION}",
                self.version
            ));
        }
        for track in &self.bone_tracks {
            for (channel, result) in [
                (
                    "translation",
                    track.translation.as_ref().map(Curve::validate),
                ),
                ("rotation", track.rotation.as_ref().map(Curve::validate)),
                ("scale", track.scale.as_ref().map(Curve::validate)),
            ] {
                if let Some(Err(error)) = result {
                    errors.push(format!("bone '{}' {channel}: {error}", track.bone));
                }
            }
        }
        for track in &self.property_tracks {
            if track.component.is_empty() || track.field.is_empty() {
                errors.push("property track needs a component and a field".to_owned());
            }
            if let Err(error) = track.curve.validate() {
                errors.push(format!("{}.{}: {error}", track.component, track.field));
            }
        }
        if self
            .events
            .iter()
            .any(|event| !event.time.is_finite() || event.time < 0.0)
        {
            errors.push("event times must be finite and non-negative".to_owned());
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Serializes as an `.anim.ron` document.
    pub fn to_ron(&self) -> String {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default().struct_names(false))
            .unwrap_or_default()
    }
}

/// Decodes glTF animation `index` into bone tracks keyed by node name.
pub fn clip_from_gltf(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    index: usize,
) -> std::result::Result<AnimationClip, String> {
    let animation = document
        .animations()
        .nth(index)
        .ok_or_else(|| format!("the file has no animation {index}"))?;
    let mut tracks: Vec<BoneTrack> = Vec::new();
    for channel in animation.channels() {
        let node = channel.target().node();
        let bone = node
            .name()
            .map(str::to_owned)
            .unwrap_or_else(|| format!("node_{}", node.index()));
        let reader = channel.reader(|buffer| buffers.get(buffer.index()).map(|data| &data.0[..]));
        let Some(times) = reader
            .read_inputs()
            .map(|inputs| inputs.collect::<Vec<f32>>())
        else {
            continue;
        };
        let interpolation = match channel.sampler().interpolation() {
            gltf::animation::Interpolation::Step => Interpolation::Step,
            gltf::animation::Interpolation::Linear => Interpolation::Linear,
            gltf::animation::Interpolation::CubicSpline => Interpolation::Cubic,
        };
        let Some(outputs) = reader.read_outputs() else {
            continue;
        };
        let track = match tracks.iter().position(|t| t.bone == bone) {
            Some(i) => &mut tracks[i],
            None => {
                tracks.push(BoneTrack {
                    bone,
                    ..Default::default()
                });
                tracks.last_mut().expect("just pushed")
            }
        };
        use gltf::animation::util::ReadOutputs;
        match outputs {
            ReadOutputs::Translations(values) => {
                track.translation = Some(build_curve(
                    interpolation,
                    times,
                    values.map(Vec3::from).collect(),
                )?)
            }
            ReadOutputs::Scales(values) => {
                track.scale = Some(build_curve(
                    interpolation,
                    times,
                    values.map(Vec3::from).collect(),
                )?)
            }
            ReadOutputs::Rotations(values) => {
                let mut values: Vec<Quat> = values.into_f32().map(Quat::from_array).collect();
                // Tangents are not unit quaternions; only normalize keys.
                let stride = if interpolation == Interpolation::Cubic {
                    3
                } else {
                    1
                };
                for (i, value) in values.iter_mut().enumerate() {
                    if stride == 1 || i % 3 == 1 {
                        *value = value.normalize();
                    }
                }
                track.rotation = Some(build_curve(interpolation, times, values)?)
            }
            ReadOutputs::MorphTargetWeights(_) => {
                log::debug!(target: "engine::animation", "morph target weights are not supported yet");
            }
        }
    }
    let mut clip = AnimationClip {
        name: animation
            .name()
            .map(str::to_owned)
            .unwrap_or_else(|| format!("animation_{index}")),
        bone_tracks: tracks,
        ..Default::default()
    };
    clip.duration = clip.length();
    Ok(clip)
}

fn build_curve<T: Animatable>(
    interpolation: Interpolation,
    times: Vec<f32>,
    values: Vec<T>,
) -> std::result::Result<Curve<T>, String> {
    let mut curve = Curve::new(interpolation);
    if interpolation == Interpolation::Cubic {
        if values.len() != times.len() * 3 {
            return Err("cubic spline output count mismatch".to_owned());
        }
        for (i, chunk) in values.chunks_exact(3).enumerate() {
            curve.times.push(times[i]);
            curve.in_tangents.push(chunk[0]);
            curve.values.push(chunk[1]);
            curve.out_tangents.push(chunk[2]);
        }
    } else {
        if values.len() != times.len() {
            return Err("animation output count mismatch".to_owned());
        }
        curve.times = times;
        curve.values = values;
    }
    // Some exporters emit duplicate times; keep the last key of a run.
    let mut i = 1;
    while i < curve.times.len() {
        if curve.times[i] <= curve.times[i - 1] {
            curve.remove_key(i - 1);
        } else {
            i += 1;
        }
    }
    Ok(curve)
}

/// Loads `*.anim.ron` documents and `file.glb#anim:<i>`.
pub struct AnimationClipLoader;

impl AssetLoader for AnimationClipLoader {
    type Asset = AnimationClip;

    fn extensions(&self) -> &'static [&'static str] {
        &["anim.ron", "glb", "gltf"]
    }

    fn load(&self, bytes: &[u8], ctx: &mut LoadContext<'_>) -> Result<AnimationClip> {
        let is_ron = ctx
            .disk_path
            .to_str()
            .is_some_and(|path| path.ends_with(".ron"));
        if !is_ron {
            let index = match ctx.sub_key {
                None => 0,
                Some(key) => key
                    .strip_prefix("anim:")
                    .and_then(|index| index.parse::<usize>().ok())
                    .ok_or_else(|| {
                        ctx.error(format!("'{key}' is not an animation sub-asset key"))
                    })?,
            };
            let (document, buffers) = gltf_document(bytes, ctx)?;
            return clip_from_gltf(&document, &buffers, index).map_err(|reason| ctx.error(reason));
        }

        let text = std::str::from_utf8(bytes).map_err(|_| ctx.error("clip is not UTF-8"))?;
        let mut clip: AnimationClip =
            ron::from_str(text).map_err(|error| ctx.error(format!("invalid clip: {error}")))?;
        if let Some(source) = clip.source.clone() {
            let (file, sub_key) = source.split_path();
            let file = file.to_owned();
            let sub_key = sub_key.map(str::to_owned);
            let root = assets_root(ctx.disk_path, ctx.relative_path);
            let source_path = root.join(&file);
            let source_bytes = std::fs::read(&source_path).map_err(|error| {
                ctx.error(format!("clip source '{}': {error}", source_path.display()))
            })?;
            ctx.add_dependency(file.clone());
            let index = sub_key
                .as_deref()
                .and_then(|key| key.strip_prefix("anim:"))
                .and_then(|index| index.parse::<usize>().ok())
                .unwrap_or(0);
            let source_ctx = LoadContext::new(&source_path, &file, None, ctx.hardening);
            let (document, buffers) = gltf_document(&source_bytes, &source_ctx)?;
            let imported =
                clip_from_gltf(&document, &buffers, index).map_err(|reason| ctx.error(reason))?;
            let mut tracks = imported.bone_tracks;
            for authored in std::mem::take(&mut clip.bone_tracks) {
                tracks.retain(|track| track.bone != authored.bone);
                tracks.push(authored);
            }
            clip.bone_tracks = tracks;
            if clip.name.is_empty() {
                clip.name = imported.name;
            }
        }
        clip.validate()
            .map_err(|errors| ctx.error(errors.join("; ")))?;
        Ok(clip)
    }
}

/// The assets root, from an asset's absolute and root-relative paths.
pub(crate) fn assets_root(disk_path: &Path, relative_path: &str) -> std::path::PathBuf {
    let depth = relative_path
        .split('/')
        .filter(|part| !part.is_empty())
        .count();
    let mut root = disk_path.to_path_buf();
    for _ in 0..depth {
        root.pop();
    }
    root
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton::{Bone, Skeleton};
    use engine_math::Mat4;

    fn clip() -> AnimationClip {
        AnimationClip {
            name: "wave".into(),
            bone_tracks: vec![BoneTrack {
                bone: "arm".into(),
                rotation: Some(Curve::linear([
                    (0.0, Quat::IDENTITY),
                    (1.0, Quat::from_rotation_z(1.0)),
                ])),
                ..Default::default()
            }],
            events: vec![
                ClipEvent {
                    time: 0.0,
                    name: "start".into(),
                    payload: String::new(),
                },
                ClipEvent {
                    time: 0.5,
                    name: "mid".into(),
                    payload: "(strength: 2)".into(),
                },
            ],
            ..Default::default()
        }
    }

    #[test]
    fn samples_bound_bones_and_derives_duration() {
        let skeleton = Skeleton::new(
            "s",
            vec![
                Bone {
                    name: "root".into(),
                    parent: None,
                    rest: BoneTransform::IDENTITY,
                },
                Bone {
                    name: "arm".into(),
                    parent: Some(0),
                    rest: BoneTransform::IDENTITY,
                },
            ],
            Mat4::IDENTITY,
            vec![],
            vec![],
        );
        let clip = clip();
        assert_eq!(clip.length(), 1.0);
        let binding = clip.bind(&skeleton);
        assert_eq!(binding.bones, vec![Some(1)]);
        let mut pose = skeleton.rest_pose();
        clip.sample_bones(&binding, 0.5, &mut pose.locals);
        assert!(
            pose.locals[1]
                .rotation
                .angle_between(Quat::from_rotation_z(0.5))
                < 1e-4
        );
        assert_eq!(pose.locals[0], BoneTransform::IDENTITY);
    }

    #[test]
    fn events_fire_once_per_crossing_including_wraps() {
        let clip = clip();
        let names = |from, to| {
            clip.events_between(from, to)
                .map(|e| e.name.as_str())
                .collect::<Vec<_>>()
        };
        assert_eq!(names(-0.001, 0.1), vec!["start"]);
        assert_eq!(names(0.1, 0.6), vec!["mid"]);
        assert_eq!(names(0.6, 0.9), Vec::<&str>::new());
        assert_eq!(names(0.9, 0.05), vec!["start"], "wrapped past the end");
        assert_eq!(names(0.4, 0.3), vec!["start", "mid"], "wrapped a full loop");
    }

    #[test]
    fn ron_round_trip_and_validation() {
        let mut clip = clip();
        clip.property_tracks.push(PropertyTrack {
            target: "Light".into(),
            component: "PointLight".into(),
            field: "intensity".into(),
            curve: PropertyCurve::Float(Curve::linear([(0.0, 1.0), (1.0, 5.0)])),
        });
        let text = clip.to_ron();
        let parsed: AnimationClip = ron::from_str(&text).unwrap();
        assert_eq!(parsed, clip);
        parsed.validate().unwrap();
        let mut bad = clip.clone();
        bad.property_tracks[0].field.clear();
        bad.version = 99;
        assert_eq!(bad.validate().unwrap_err().len(), 2);
        assert_eq!(
            assets_root(Path::new("/p/assets/anim/a.anim.ron"), "anim/a.anim.ron"),
            Path::new("/p/assets")
        );
    }
}
