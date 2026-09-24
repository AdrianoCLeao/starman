//! The mixer asset (`*.mixer.ron`): a tree of buses under `master`, send
//! (auxiliary) buses, effects per bus, and named snapshots that override
//! volumes, send levels and effect parameters with a fade and priority.

use std::collections::{BTreeMap, HashSet};

use engine_assets::{Asset, AssetLoader, LoadContext};
use engine_core::Result;
use serde::{Deserialize, Serialize};

pub const MASTER: &str = "master";
pub const AUDIO_MIXER_VERSION: u32 = 1;
/// Volumes at or below this are silent.
pub const SILENCE_DB: f32 = -80.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FilterKind {
    LowPass,
    HighPass,
    BandPass,
    Notch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EqKind {
    Bell,
    LowShelf,
    HighShelf,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum EffectDef {
    Filter {
        kind: FilterKind,
        cutoff: f32,
        #[serde(default)]
        resonance: f32,
        #[serde(default = "one")]
        mix: f32,
    },
    Reverb {
        #[serde(default = "reverb_feedback")]
        feedback: f32,
        #[serde(default = "half")]
        damping: f32,
        #[serde(default = "one")]
        stereo_width: f32,
        #[serde(default = "half")]
        mix: f32,
    },
    Compressor {
        threshold_db: f32,
        ratio: f32,
        #[serde(default = "attack")]
        attack_ms: f32,
        #[serde(default = "release")]
        release_ms: f32,
        #[serde(default)]
        makeup_db: f32,
        #[serde(default = "one")]
        mix: f32,
    },
    Delay {
        time_s: f32,
        #[serde(default = "delay_feedback")]
        feedback_db: f32,
        #[serde(default = "half")]
        mix: f32,
    },
    Eq {
        kind: EqKind,
        frequency: f32,
        gain_db: f32,
        #[serde(default = "one")]
        q: f32,
    },
}

fn one() -> f32 {
    1.0
}
fn half() -> f32 {
    0.5
}
fn reverb_feedback() -> f32 {
    0.9
}
fn attack() -> f32 {
    10.0
}
fn release() -> f32 {
    100.0
}
fn delay_feedback() -> f32 {
    -6.0
}

/// Effect parameters snapshots can animate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EffectParam {
    Cutoff,
    Resonance,
    Mix,
    Feedback,
    Damping,
    GainDb,
    Frequency,
}

impl EffectDef {
    /// Current value of `param` (None when the effect lacks it).
    pub fn param(&self, param: EffectParam) -> Option<f32> {
        Some(match (self, param) {
            (Self::Filter { cutoff, .. }, EffectParam::Cutoff) => *cutoff,
            (Self::Filter { resonance, .. }, EffectParam::Resonance) => *resonance,
            (Self::Filter { mix, .. }, EffectParam::Mix)
            | (Self::Reverb { mix, .. }, EffectParam::Mix)
            | (Self::Compressor { mix, .. }, EffectParam::Mix)
            | (Self::Delay { mix, .. }, EffectParam::Mix) => *mix,
            (Self::Reverb { feedback, .. }, EffectParam::Feedback) => *feedback,
            (Self::Reverb { damping, .. }, EffectParam::Damping) => *damping,
            (Self::Delay { feedback_db, .. }, EffectParam::Feedback) => *feedback_db,
            (Self::Eq { gain_db, .. }, EffectParam::GainDb) => *gain_db,
            (Self::Eq { frequency, .. }, EffectParam::Frequency) => *frequency,
            _ => return None,
        })
    }

    /// Every animatable parameter of this effect.
    pub fn params(&self) -> Vec<EffectParam> {
        [
            EffectParam::Cutoff,
            EffectParam::Resonance,
            EffectParam::Mix,
            EffectParam::Feedback,
            EffectParam::Damping,
            EffectParam::GainDb,
            EffectParam::Frequency,
        ]
        .into_iter()
        .filter(|p| self.param(*p).is_some())
        .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SendLevel {
    pub send: String,
    pub volume_db: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BusDef {
    pub name: String,
    /// Parent bus (`master` when absent).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    #[serde(default)]
    pub volume_db: f32,
    #[serde(default)]
    pub muted: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub effects: Vec<EffectDef>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sends: Vec<SendLevel>,
    /// Concurrent voices on this bus (lowest priority stolen first).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_voices: Option<u32>,
}

impl BusDef {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            parent: None,
            volume_db: 0.0,
            muted: false,
            effects: Vec::new(),
            sends: Vec::new(),
            max_voices: None,
        }
    }
}

/// An auxiliary bus fed by bus sends (reverb, delay).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SendDef {
    pub name: String,
    #[serde(default)]
    pub volume_db: f32,
    #[serde(default)]
    pub effects: Vec<EffectDef>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EffectOverride {
    pub bus: String,
    /// Index into the bus's effects.
    pub effect: usize,
    pub param: EffectParam,
    pub value: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SnapshotDef {
    pub name: String,
    /// Higher priorities apply on top of lower ones.
    #[serde(default)]
    pub priority: i32,
    /// Seconds to blend in/out.
    #[serde(default = "half")]
    pub fade: f32,
    /// Bus volume overrides (dB).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub buses: Vec<(String, f32)>,
    /// `(bus, send, dB)` send level overrides.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sends: Vec<(String, String, f32)>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub effects: Vec<EffectOverride>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AudioMixer {
    #[serde(default = "mixer_version")]
    pub version: u32,
    #[serde(default)]
    pub buses: Vec<BusDef>,
    #[serde(default)]
    pub sends: Vec<SendDef>,
    #[serde(default)]
    pub snapshots: Vec<SnapshotDef>,
    /// Global voice limit.
    #[serde(default = "default_max_voices")]
    pub max_voices: u32,
}

fn mixer_version() -> u32 {
    AUDIO_MIXER_VERSION
}

fn default_max_voices() -> u32 {
    64
}

impl Asset for AudioMixer {
    const TYPE_NAME: &'static str = "AudioMixer";
}

impl Default for AudioMixer {
    /// The engine default: master → music, sfx, ui, ambience, voice.
    fn default() -> Self {
        Self {
            version: AUDIO_MIXER_VERSION,
            buses: ["music", "sfx", "ui", "ambience", "voice"]
                .into_iter()
                .map(BusDef::new)
                .collect(),
            sends: Vec::new(),
            snapshots: Vec::new(),
            max_voices: default_max_voices(),
        }
    }
}

impl AudioMixer {
    pub fn bus(&self, name: &str) -> Option<&BusDef> {
        self.buses.iter().find(|b| b.name == name)
    }

    pub fn snapshot(&self, name: &str) -> Option<&SnapshotDef> {
        self.snapshots.iter().find(|s| s.name == name)
    }

    /// Bus names parent-first (master excluded).
    pub fn bus_order(&self) -> Vec<&BusDef> {
        let mut ordered: Vec<&BusDef> = Vec::new();
        let mut remaining: Vec<&BusDef> = self.buses.iter().filter(|b| b.name != MASTER).collect();
        while !remaining.is_empty() {
            let before = remaining.len();
            remaining.retain(|bus| {
                let parent = bus.parent.as_deref().unwrap_or(MASTER);
                if parent == MASTER || ordered.iter().any(|o| o.name == parent) {
                    ordered.push(bus);
                    false
                } else {
                    true
                }
            });
            if remaining.len() == before {
                break; // cycle; validation reports it
            }
        }
        ordered
    }

    pub fn validate(&self) -> std::result::Result<(), Vec<String>> {
        let mut errors = Vec::new();
        if self.version > AUDIO_MIXER_VERSION {
            errors.push(format!(
                "mixer version {} is newer than supported",
                self.version
            ));
        }
        let mut names: HashSet<&str> = HashSet::from([MASTER]);
        for bus in &self.buses {
            if bus.name != MASTER && !names.insert(bus.name.as_str()) {
                errors.push(format!("bus '{}' is declared twice", bus.name));
            }
        }
        let sends: HashSet<&str> = self.sends.iter().map(|s| s.name.as_str()).collect();
        for bus in &self.buses {
            if let Some(parent) = &bus.parent {
                if !names.contains(parent.as_str()) {
                    errors.push(format!("bus '{}' has unknown parent '{parent}'", bus.name));
                }
            }
            for send in &bus.sends {
                if !sends.contains(send.send.as_str()) {
                    errors.push(format!(
                        "bus '{}' sends to unknown send '{}'",
                        bus.name, send.send
                    ));
                }
            }
        }
        if self.bus_order().len() != self.buses.iter().filter(|b| b.name != MASTER).count() {
            errors.push("bus parents form a cycle".to_owned());
        }
        for snapshot in &self.snapshots {
            for (bus, _) in &snapshot.buses {
                if !names.contains(bus.as_str()) {
                    errors.push(format!(
                        "snapshot '{}' overrides unknown bus '{bus}'",
                        snapshot.name
                    ));
                }
            }
            for (bus, send, _) in &snapshot.sends {
                if !names.contains(bus.as_str()) || !sends.contains(send.as_str()) {
                    errors.push(format!(
                        "snapshot '{}' overrides unknown send '{bus}' -> '{send}'",
                        snapshot.name
                    ));
                }
            }
            for effect in &snapshot.effects {
                let def = self
                    .bus(&effect.bus)
                    .and_then(|b| b.effects.get(effect.effect));
                match def {
                    None => errors.push(format!(
                        "snapshot '{}' targets missing effect {} of bus '{}'",
                        snapshot.name, effect.effect, effect.bus
                    )),
                    Some(def) if def.param(effect.param).is_none() => errors.push(format!(
                        "snapshot '{}': {:?} has no {:?}",
                        snapshot.name, def, effect.param
                    )),
                    _ => {}
                }
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    pub fn to_ron(&self) -> String {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default().struct_names(false))
            .unwrap_or_default()
    }
}

/// Resolved mixer values for one frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MixState {
    pub bus_volume: BTreeMap<String, f32>,
    pub send_levels: BTreeMap<(String, String), f32>,
    pub effect_params: BTreeMap<(String, usize, EffectParam), f32>,
}

/// Linear gain (0..1+) to decibels.
pub fn gain_to_db(gain: f32) -> f32 {
    if gain <= 1e-4 {
        SILENCE_DB
    } else {
        (20.0 * gain.log10()).max(SILENCE_DB)
    }
}

/// Resolves base values, weighted snapshots (applied in priority order)
/// and user bus volumes (linear 0..1) into a mix.
pub fn resolve_mix(
    mixer: &AudioMixer,
    snapshots: &[(&SnapshotDef, f32)],
    user_volumes: &BTreeMap<String, f32>,
) -> MixState {
    let mut state = MixState::default();
    state.bus_volume.insert(MASTER.to_owned(), 0.0);
    for bus in &mixer.buses {
        state.bus_volume.insert(
            bus.name.clone(),
            if bus.muted { SILENCE_DB } else { bus.volume_db },
        );
        for send in &bus.sends {
            state
                .send_levels
                .insert((bus.name.clone(), send.send.clone()), send.volume_db);
        }
        for (index, effect) in bus.effects.iter().enumerate() {
            for param in effect.params() {
                if let Some(value) = effect.param(param) {
                    state
                        .effect_params
                        .insert((bus.name.clone(), index, param), value);
                }
            }
        }
    }
    let mut ordered: Vec<&(&SnapshotDef, f32)> = snapshots.iter().collect();
    ordered.sort_by_key(|(snapshot, _)| snapshot.priority);
    for (snapshot, weight) in ordered {
        let w = weight.clamp(0.0, 1.0);
        if w <= 0.0 {
            continue;
        }
        for (bus, db) in &snapshot.buses {
            let value = state.bus_volume.entry(bus.clone()).or_insert(0.0);
            *value += (db - *value) * w;
        }
        for (bus, send, db) in &snapshot.sends {
            let value = state
                .send_levels
                .entry((bus.clone(), send.clone()))
                .or_insert(SILENCE_DB);
            *value += (db - *value) * w;
        }
        for effect in &snapshot.effects {
            if let Some(value) =
                state
                    .effect_params
                    .get_mut(&(effect.bus.clone(), effect.effect, effect.param))
            {
                *value += (effect.value - *value) * w;
            }
        }
    }
    for (bus, gain) in user_volumes {
        if let Some(value) = state.bus_volume.get_mut(bus) {
            *value = (*value + gain_to_db(*gain)).max(SILENCE_DB);
        }
    }
    state
}

/// Loads and validates `*.mixer.ron`.
pub struct AudioMixerLoader;

impl AssetLoader for AudioMixerLoader {
    type Asset = AudioMixer;

    fn extensions(&self) -> &'static [&'static str] {
        &["mixer.ron"]
    }

    fn load(&self, bytes: &[u8], ctx: &mut LoadContext<'_>) -> Result<AudioMixer> {
        let text = std::str::from_utf8(bytes).map_err(|_| ctx.error("mixer is not UTF-8"))?;
        let mixer: AudioMixer =
            ron::from_str(text).map_err(|error| ctx.error(format!("invalid mixer: {error}")))?;
        mixer
            .validate()
            .map_err(|errors| ctx.error(errors.join("; ")))?;
        Ok(mixer)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) const SAMPLE: &str = r#"(
        buses: [
            (name: "music", volume_db: -6.0),
            (name: "sfx", effects: [Filter(kind: LowPass, cutoff: 20000.0)], sends: [(send: "reverb", volume_db: -80.0)], max_voices: Some(4)),
            (name: "footsteps", parent: Some("sfx")),
            (name: "ui"),
            (name: "ambience"),
        ],
        sends: [(name: "reverb", effects: [Reverb(mix: 1.0)])],
        snapshots: [
            (name: "paused", priority: 10, fade: 0.3, buses: [("sfx", -24.0), ("music", -12.0)], effects: [(bus: "sfx", effect: 0, param: Cutoff, value: 800.0)]),
            (name: "cave", fade: 1.0, sends: [("sfx", "reverb", -6.0)]),
        ],
    )"#;

    pub(crate) fn sample() -> AudioMixer {
        ron::from_str(SAMPLE).unwrap()
    }

    #[test]
    fn sample_validates_orders_buses_and_round_trips() {
        let mixer = sample();
        mixer.validate().unwrap();
        let order: Vec<&str> = mixer.bus_order().iter().map(|b| b.name.as_str()).collect();
        let sfx = order.iter().position(|n| *n == "sfx").unwrap();
        let steps = order.iter().position(|n| *n == "footsteps").unwrap();
        assert!(sfx < steps);
        let parsed: AudioMixer = ron::from_str(&mixer.to_ron()).unwrap();
        assert_eq!(parsed, mixer);
        AudioMixer::default().validate().unwrap();
    }

    #[test]
    fn validation_reports_bad_references() {
        let mut mixer = sample();
        mixer.buses.push(BusDef {
            parent: Some("nowhere".into()),
            ..BusDef::new("orphan")
        });
        mixer.snapshots[0].effects.push(EffectOverride {
            bus: "sfx".into(),
            effect: 0,
            param: EffectParam::Damping,
            value: 1.0,
        });
        let errors = mixer.validate().unwrap_err().join("\n");
        assert!(errors.contains("unknown parent 'nowhere'"), "{errors}");
        assert!(errors.contains("has no Damping"), "{errors}");
    }

    #[test]
    fn snapshots_blend_by_weight_and_priority_and_user_volume_applies() {
        let mixer = sample();
        let paused = mixer.snapshot("paused").unwrap();
        let cave = mixer.snapshot("cave").unwrap();
        let base = resolve_mix(&mixer, &[], &BTreeMap::new());
        assert_eq!(base.bus_volume["music"], -6.0);
        assert_eq!(
            base.effect_params[&("sfx".to_owned(), 0, EffectParam::Cutoff)],
            20000.0
        );
        let mixed = resolve_mix(
            &mixer,
            &[(cave, 0.5), (paused, 1.0)],
            &BTreeMap::from([("music".to_owned(), 0.5)]),
        );
        assert_eq!(mixed.bus_volume["sfx"], -24.0);
        assert!((mixed.bus_volume["music"] - (-12.0 - 6.0206)).abs() < 1e-3);
        assert!(
            (mixed.send_levels[&("sfx".to_owned(), "reverb".to_owned())] - (-43.0)).abs() < 1e-4
        );
        assert_eq!(
            mixed.effect_params[&("sfx".to_owned(), 0, EffectParam::Cutoff)],
            800.0
        );
        assert_eq!(gain_to_db(0.0), SILENCE_DB);
    }
}
