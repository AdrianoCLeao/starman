//! [`AudioEngine`]: the backend plus the engine-side bookkeeping every
//! backend shares — voice records and limits, the snapshot stack and mix
//! resolution. Kept free of ECS queries so it is unit-testable and usable
//! by tools (editor mixer preview).

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_assets::{AssetRef, Assets, Handle};

use crate::backend::{AudioBackend, PlayRequest, VoiceId, VoiceInfo};
use crate::mixer::{resolve_mix, AudioMixer, MixState, SnapshotDef, MASTER};
use crate::null::NullBackend;

/// Fade used when a voice is stolen by a higher-priority one.
pub const STEAL_FADE_SECONDS: f32 = 0.05;
/// Tween applied to per-frame mix changes (weights already move smoothly).
pub const MIX_TWEEN_SECONDS: f32 = 0.05;

/// Bookkeeping for one playing voice.
#[derive(Clone, Debug, PartialEq)]
pub struct VoiceRecord {
    pub bus: String,
    pub priority: i32,
    pub entity: Option<Entity>,
    pub clip_key: String,
    pub tag: Option<String>,
    /// Start order (older voices are stolen first at equal priority).
    pub sequence: u64,
}

/// Which voices a limit counts.
enum LimitScope<'a> {
    All,
    Bus(&'a str),
    Clip(&'a str),
}

impl LimitScope<'_> {
    fn contains(&self, record: &VoiceRecord) -> bool {
        match self {
            Self::All => true,
            Self::Bus(bus) => record.bus == *bus,
            Self::Clip(key) => record.clip_key == *key,
        }
    }
}

/// A voice start request plus its bookkeeping.
#[derive(Clone, Debug)]
pub struct VoiceStart {
    pub request: PlayRequest,
    pub priority: i32,
    pub entity: Option<Entity>,
    pub clip_key: String,
    pub tag: Option<String>,
    /// Concurrent voices of this clip (0 = unlimited).
    pub max_instances: u32,
}

/// One entry of the snapshot stack.
#[derive(Clone, Debug, PartialEq)]
pub struct ActiveSnapshot {
    pub name: String,
    pub weight: f32,
    /// Fading in (`true`) or out.
    pub target_on: bool,
}

/// Where the active mixer comes from (project asset or inline).
#[derive(Resource, Default)]
pub struct AudioMixerSource {
    asset: Option<AssetRef>,
    handle: Option<Handle<AudioMixer>>,
    revision: u64,
    inline: Option<Arc<AudioMixer>>,
}

impl AudioMixerSource {
    pub fn set_asset(&mut self, asset: AssetRef) {
        if self.asset.as_ref() != Some(&asset) {
            *self = Self {
                asset: Some(asset),
                ..Self::default()
            };
        }
    }

    pub fn set_inline(&mut self, mixer: AudioMixer) {
        *self = Self {
            inline: Some(Arc::new(mixer)),
            ..Self::default()
        };
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }

    pub fn asset(&self) -> Option<&AssetRef> {
        self.asset.as_ref()
    }

    /// A newly available mixer (inline value or (re)loaded asset).
    pub(crate) fn poll(&mut self, assets: Option<&Assets>) -> Option<Arc<AudioMixer>> {
        if let Some(inline) = self.inline.take() {
            return Some(inline);
        }
        let (Some(asset), Some(assets)) = (&self.asset, assets) else {
            return None;
        };
        let handle = *self
            .handle
            .get_or_insert_with(|| assets.request::<AudioMixer>(asset));
        let revision = assets.revision(handle);
        if revision == self.revision {
            return None;
        }
        let mixer = assets.get(handle)?;
        self.revision = revision;
        Some(mixer)
    }
}

/// The audio output and its engine-side state.
#[derive(Resource)]
pub struct AudioEngine {
    backend: Box<dyn AudioBackend>,
    mixer: Arc<AudioMixer>,
    voices: BTreeMap<VoiceId, VoiceRecord>,
    snapshots: Vec<ActiveSnapshot>,
    mix: MixState,
    sequence: u64,
    rng: u64,
    finished: Vec<(VoiceId, VoiceRecord)>,
}

impl Default for AudioEngine {
    /// Silent [`NullBackend`]; hosts with a device call
    /// [`AudioEngine::set_backend`].
    fn default() -> Self {
        Self::new(Box::new(NullBackend::new()))
    }
}

impl AudioEngine {
    pub fn new(mut backend: Box<dyn AudioBackend>) -> Self {
        let mixer = Arc::new(AudioMixer::default());
        backend.configure(&mixer);
        Self {
            backend,
            mixer,
            voices: BTreeMap::new(),
            snapshots: Vec::new(),
            mix: MixState::default(),
            sequence: 0,
            rng: 0x853c_49e6_748f_ea9b,
            finished: Vec::new(),
        }
    }

    /// Swaps the output (voices on the previous backend end).
    pub fn set_backend(&mut self, mut backend: Box<dyn AudioBackend>) {
        backend.configure(&self.mixer);
        self.backend = backend;
        let ended: Vec<VoiceId> = self.voices.keys().copied().collect();
        for id in ended {
            self.end_record(id);
        }
        self.mix = MixState::default();
    }

    pub fn backend_name(&self) -> &'static str {
        self.backend.name()
    }

    pub fn backend(&self) -> &dyn AudioBackend {
        self.backend.as_ref()
    }

    pub fn backend_mut(&mut self) -> &mut dyn AudioBackend {
        self.backend.as_mut()
    }

    pub fn mixer(&self) -> &Arc<AudioMixer> {
        &self.mixer
    }

    /// Rebuilds the bus graph; snapshots missing from the new mixer drop.
    pub fn set_mixer(&mut self, mixer: Arc<AudioMixer>) {
        if let Err(errors) = mixer.validate() {
            log::warn!(target: "engine::audio", "mixer has problems: {}", errors.join("; "));
        }
        self.backend.configure(&mixer);
        self.snapshots.retain(|s| mixer.snapshot(&s.name).is_some());
        self.mixer = mixer;
        self.mix = MixState::default();
    }

    /// The last resolved mix.
    pub fn mix(&self) -> &MixState {
        &self.mix
    }

    pub fn voices(&self) -> &BTreeMap<VoiceId, VoiceRecord> {
        &self.voices
    }

    pub fn voice_infos(&self) -> Vec<VoiceInfo> {
        self.backend.voices()
    }

    pub fn is_playing(&self, id: VoiceId) -> bool {
        self.voices.contains_key(&id) && self.backend.is_playing(id)
    }

    pub fn snapshots(&self) -> &[ActiveSnapshot] {
        &self.snapshots
    }

    /// Deterministic uniform random in `[-1, 1]` (pitch/volume variation).
    pub fn random_signed(&mut self) -> f32 {
        self.rng = self
            .rng
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (self.rng >> 40) as u32; // 24 bits
        bits as f32 / (1u32 << 23) as f32 - 1.0
    }

    pub fn push_snapshot(&mut self, name: &str) -> bool {
        if self.mixer.snapshot(name).is_none() {
            log::warn!(target: "engine::audio", "unknown mixer snapshot '{name}'");
            return false;
        }
        match self.snapshots.iter_mut().find(|s| s.name == name) {
            Some(active) => active.target_on = true,
            None => self.snapshots.push(ActiveSnapshot {
                name: name.to_owned(),
                weight: 0.0,
                target_on: true,
            }),
        }
        true
    }

    pub fn pop_snapshot(&mut self, name: &str) {
        if let Some(active) = self.snapshots.iter_mut().find(|s| s.name == name) {
            active.target_on = false;
        }
    }

    /// Buses under (and including) `bus`.
    pub fn bus_subtree(&self, bus: &str) -> HashSet<String> {
        let mut set: HashSet<String> = HashSet::from([bus.to_owned()]);
        if bus == MASTER {
            set.extend(self.mixer.buses.iter().map(|b| b.name.clone()));
            return set;
        }
        loop {
            let before = set.len();
            for def in &self.mixer.buses {
                if def.parent.as_ref().is_some_and(|p| set.contains(p)) {
                    set.insert(def.name.clone());
                }
            }
            if set.len() == before {
                return set;
            }
        }
    }

    /// Starts a voice if the global, bus and clip limits admit it (stealing
    /// the lowest-priority, oldest voice when the newcomer outranks it).
    pub fn start(&mut self, start: VoiceStart) -> Option<VoiceId> {
        let bus = if self.mixer.bus(&start.request.bus).is_some() || start.request.bus == MASTER {
            start.request.bus.clone()
        } else {
            log::warn!(target: "engine::audio", "unknown bus '{}'; using master", start.request.bus);
            MASTER.to_owned()
        };
        let bus_limit = self.mixer.bus(&bus).and_then(|b| b.max_voices).unwrap_or(0);
        let limits = [
            (self.mixer.max_voices, LimitScope::All),
            (bus_limit, LimitScope::Bus(&bus)),
            (start.max_instances, LimitScope::Clip(&start.clip_key)),
        ];
        for (limit, scope) in &limits {
            if *limit == 0 {
                continue;
            }
            let members: Vec<(VoiceId, i32, u64)> = self
                .voices
                .iter()
                .filter(|(_, r)| scope.contains(r))
                .map(|(id, r)| (*id, r.priority, r.sequence))
                .collect();
            if (members.len() as u32) < *limit {
                continue;
            }
            let victim = members
                .iter()
                .min_by_key(|(_, priority, seq)| (*priority, *seq));
            match victim {
                Some((id, priority, _)) if *priority <= start.priority => {
                    self.backend.stop(*id, STEAL_FADE_SECONDS);
                    self.end_record(*id);
                }
                _ => return None,
            }
        }
        let mut request = start.request;
        request.bus = bus.clone();
        let id = self.backend.play(request)?;
        self.sequence += 1;
        self.voices.insert(
            id,
            VoiceRecord {
                bus,
                priority: start.priority,
                entity: start.entity,
                clip_key: start.clip_key,
                tag: start.tag,
                sequence: self.sequence,
            },
        );
        Some(id)
    }

    pub fn stop(&mut self, id: VoiceId, fade: f32) {
        if self.voices.contains_key(&id) {
            self.backend.stop(id, fade);
            if fade <= 0.0 {
                self.end_record(id);
            }
        }
    }

    fn end_record(&mut self, id: VoiceId) {
        if let Some(record) = self.voices.remove(&id) {
            self.finished.push((id, record));
        }
    }

    /// Voices that ended since the last call.
    pub fn take_finished(&mut self) -> Vec<(VoiceId, VoiceRecord)> {
        std::mem::take(&mut self.finished)
    }

    /// Advances fades and the backend, reaps ended voices and pushes the
    /// resolved mix (snapshot stack + `extra` weighted snapshots from
    /// zones + user gains) to the backend.
    pub fn step(
        &mut self,
        dt: f32,
        extra: &[(SnapshotDef, f32)],
        user_gains: &BTreeMap<String, f32>,
    ) {
        for active in &mut self.snapshots {
            let fade = self
                .mixer
                .snapshot(&active.name)
                .map_or(0.0, |s| s.fade.max(0.0));
            let step = if fade <= 0.0 { 1.0 } else { dt / fade };
            active.weight = if active.target_on {
                (active.weight + step).min(1.0)
            } else {
                (active.weight - step).max(0.0)
            };
        }
        self.snapshots.retain(|s| s.target_on || s.weight > 0.0);

        self.backend.update(dt);
        let ended: Vec<VoiceId> = self
            .voices
            .keys()
            .copied()
            .filter(|id| !self.backend.is_playing(*id))
            .collect();
        for id in ended {
            self.end_record(id);
        }

        let mut weighted: Vec<(&SnapshotDef, f32)> = self
            .snapshots
            .iter()
            .filter_map(|a| self.mixer.snapshot(&a.name).map(|def| (def, a.weight)))
            .collect();
        weighted.extend(extra.iter().map(|(def, w)| (def, *w)));
        let mix = resolve_mix(&self.mixer, &weighted, user_gains);
        if mix != self.mix {
            self.backend.apply_mix(&mix, MIX_TWEEN_SECONDS);
            self.mix = mix;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::AudioClip;
    use crate::mixer::tests::sample;

    fn start(clip: &Arc<AudioClip>, bus: &str, priority: i32, key: &str) -> VoiceStart {
        VoiceStart {
            request: PlayRequest::new(clip.clone(), bus),
            priority,
            entity: None,
            clip_key: key.to_owned(),
            tag: None,
            max_instances: 0,
        }
    }

    #[test]
    fn limits_steal_lowest_priority_and_reject_outranked_voices() {
        let mut engine = AudioEngine::default();
        engine.set_mixer(Arc::new(sample()));
        let clip = Arc::new(AudioClip::silence(100, 10.0));
        let low = engine.start(start(&clip, "sfx", 0, "a")).unwrap();
        let mid = engine.start(start(&clip, "sfx", 5, "a")).unwrap();
        engine.start(start(&clip, "sfx", 5, "b")).unwrap();
        engine.start(start(&clip, "sfx", 5, "b")).unwrap();
        // sfx max_voices = 4: a priority-1 voice steals the priority-0 one.
        let stealer = engine.start(start(&clip, "sfx", 1, "c")).unwrap();
        assert!(!engine.voices().contains_key(&low));
        assert!(engine.voices().contains_key(&stealer));
        // Now the lowest is priority 1; a priority-0 voice is rejected.
        assert!(engine.start(start(&clip, "sfx", 0, "d")).is_none());
        // Other buses are unaffected.
        assert!(engine.start(start(&clip, "music", 0, "m")).is_some());
        // Per-clip instance limit.
        let mut limited = start(&clip, "ui", 9, "a");
        limited.max_instances = 1;
        engine.start(limited).unwrap();
        assert!(
            !engine.voices().contains_key(&mid),
            "only 'a' instance stolen"
        );
        let finished: Vec<VoiceId> = engine
            .take_finished()
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert_eq!(finished, vec![low, mid]);
    }

    #[test]
    fn snapshot_stack_fades_and_resolves_into_the_backend_mix() {
        let mut engine = AudioEngine::default();
        engine.set_mixer(Arc::new(sample()));
        assert!(engine.push_snapshot("paused"));
        assert!(!engine.push_snapshot("missing"));
        engine.step(0.15, &[], &BTreeMap::new());
        assert!((engine.snapshots()[0].weight - 0.5).abs() < 1e-5);
        assert!((engine.mix().bus_volume["sfx"] - -12.0).abs() < 1e-4);
        engine.step(0.3, &[], &BTreeMap::new());
        assert_eq!(engine.mix().bus_volume["sfx"], -24.0);
        engine.pop_snapshot("paused");
        engine.step(0.3, &[], &BTreeMap::new());
        assert!(engine.snapshots().is_empty());
        assert_eq!(engine.mix().bus_volume["sfx"], 0.0);
        let subtree = engine.bus_subtree("sfx");
        assert!(subtree.contains("footsteps") && !subtree.contains("music"));
    }

    #[test]
    fn finished_voices_are_reaped_and_variation_is_deterministic() {
        let mut engine = AudioEngine::default();
        let clip = Arc::new(AudioClip::silence(100, 0.1));
        let id = engine.start(start(&clip, "sfx", 0, "a")).unwrap();
        engine.step(0.05, &[], &BTreeMap::new());
        assert!(engine.is_playing(id));
        engine.step(0.06, &[], &BTreeMap::new());
        assert!(!engine.is_playing(id));
        assert_eq!(engine.take_finished().len(), 1);
        let a: Vec<f32> = (0..4).map(|_| engine.random_signed()).collect();
        let mut other = AudioEngine::default();
        let b: Vec<f32> = (0..4).map(|_| other.random_signed()).collect();
        assert_eq!(a, b);
        assert!(a.iter().all(|v| (-1.0..=1.0).contains(v)));
    }
}
