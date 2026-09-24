//! Kira 0.12 backend: one sub-track per bus (parented like the mixer
//! tree, carrying its effects and send routes), one send track per
//! auxiliary bus, and a short-lived spatial sub-track per positional voice.
//!
//! Distance attenuation is computed engine-side (so every backend agrees);
//! the spatial tracks only pan, with `spatialization_strength` taken from
//! the voice's spatial blend.

use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;
use std::time::Duration;

use engine_math::{Quat, Vec3};
use kira::backend::Backend;
use kira::effect::compressor::{CompressorBuilder, CompressorHandle};
use kira::effect::delay::{DelayBuilder, DelayHandle};
use kira::effect::eq_filter::{EqFilterBuilder, EqFilterHandle, EqFilterKind};
use kira::effect::filter::{FilterBuilder, FilterHandle, FilterMode};
use kira::effect::reverb::{ReverbBuilder, ReverbHandle};
use kira::listener::ListenerHandle;
use kira::sound::static_sound::{StaticSoundHandle, StaticSoundSettings};
use kira::sound::PlaybackState;
use kira::track::{
    SendTrackBuilder, SendTrackHandle, SpatialTrackBuilder, SpatialTrackDistances,
    SpatialTrackHandle, TrackBuilder, TrackHandle,
};
use kira::{
    AudioManager, AudioManagerSettings, Decibels, DefaultBackend, Mix, PlaybackRate, Tween,
};

use crate::backend::{AudioBackend, PlayRequest, VoiceId, VoiceInfo, VoiceUpdate};
use crate::mixer::{
    AudioMixer, EffectDef, EffectParam, EqKind, FilterKind, MixState, MASTER, SILENCE_DB,
};

fn tween(seconds: f32) -> Tween {
    Tween {
        duration: Duration::from_secs_f32(seconds.max(0.0)),
        ..Default::default()
    }
}

fn db(value: f32) -> Decibels {
    Decibels(if value <= SILENCE_DB {
        Decibels::SILENCE.0
    } else {
        value
    })
}

enum EffectHandle {
    Filter(FilterHandle),
    Reverb(ReverbHandle),
    Compressor(CompressorHandle),
    Delay(DelayHandle),
    Eq(EqFilterHandle),
}

impl EffectHandle {
    fn set(&mut self, param: EffectParam, value: f32, fade: Tween) {
        let v = value as f64;
        match (self, param) {
            (Self::Filter(h), EffectParam::Cutoff) => h.set_cutoff(v, fade),
            (Self::Filter(h), EffectParam::Resonance) => h.set_resonance(v, fade),
            (Self::Filter(h), EffectParam::Mix) => h.set_mix(Mix(value), fade),
            (Self::Reverb(h), EffectParam::Feedback) => h.set_feedback(v, fade),
            (Self::Reverb(h), EffectParam::Damping) => h.set_damping(v, fade),
            (Self::Reverb(h), EffectParam::Mix) => h.set_mix(Mix(value), fade),
            (Self::Compressor(h), EffectParam::Mix) => h.set_mix(Mix(value), fade),
            (Self::Delay(h), EffectParam::Feedback) => h.set_feedback(db(value), fade),
            (Self::Delay(h), EffectParam::Mix) => h.set_mix(Mix(value), fade),
            (Self::Eq(h), EffectParam::GainDb) => h.set_gain(db(value), fade),
            (Self::Eq(h), EffectParam::Frequency) => h.set_frequency(v, fade),
            _ => {}
        }
    }
}

/// Adds `def` to a track builder (sub or send), returning its handle.
trait EffectHost {
    fn add_filter(&mut self, b: FilterBuilder) -> FilterHandle;
    fn add_reverb(&mut self, b: ReverbBuilder) -> ReverbHandle;
    fn add_compressor(&mut self, b: CompressorBuilder) -> CompressorHandle;
    fn add_delay(&mut self, b: DelayBuilder) -> DelayHandle;
    fn add_eq(&mut self, b: EqFilterBuilder) -> EqFilterHandle;
}

macro_rules! effect_host {
    ($ty:ty) => {
        impl EffectHost for $ty {
            fn add_filter(&mut self, b: FilterBuilder) -> FilterHandle {
                self.add_effect(b)
            }
            fn add_reverb(&mut self, b: ReverbBuilder) -> ReverbHandle {
                self.add_effect(b)
            }
            fn add_compressor(&mut self, b: CompressorBuilder) -> CompressorHandle {
                self.add_effect(b)
            }
            fn add_delay(&mut self, b: DelayBuilder) -> DelayHandle {
                self.add_effect(b)
            }
            fn add_eq(&mut self, b: EqFilterBuilder) -> EqFilterHandle {
                self.add_effect(b)
            }
        }
    };
}

effect_host!(TrackBuilder);
effect_host!(SendTrackBuilder);

fn add_effect(host: &mut impl EffectHost, def: &EffectDef) -> EffectHandle {
    match def {
        EffectDef::Filter {
            kind,
            cutoff,
            resonance,
            mix,
        } => EffectHandle::Filter(
            host.add_filter(
                FilterBuilder::new()
                    .mode(match kind {
                        FilterKind::LowPass => FilterMode::LowPass,
                        FilterKind::HighPass => FilterMode::HighPass,
                        FilterKind::BandPass => FilterMode::BandPass,
                        FilterKind::Notch => FilterMode::Notch,
                    })
                    .cutoff(*cutoff as f64)
                    .resonance(*resonance as f64)
                    .mix(Mix(*mix)),
            ),
        ),
        EffectDef::Reverb {
            feedback,
            damping,
            stereo_width,
            mix,
        } => EffectHandle::Reverb(
            host.add_reverb(
                ReverbBuilder::new()
                    .feedback(*feedback as f64)
                    .damping(*damping as f64)
                    .stereo_width(*stereo_width as f64)
                    .mix(Mix(*mix)),
            ),
        ),
        EffectDef::Compressor {
            threshold_db,
            ratio,
            attack_ms,
            release_ms,
            makeup_db,
            mix,
        } => EffectHandle::Compressor(
            host.add_compressor(
                CompressorBuilder::new()
                    .threshold(*threshold_db as f64)
                    .ratio(*ratio as f64)
                    .attack_duration(Duration::from_secs_f32(attack_ms.max(0.0) / 1000.0))
                    .release_duration(Duration::from_secs_f32(release_ms.max(0.0) / 1000.0))
                    .makeup_gain(Decibels(*makeup_db))
                    .mix(Mix(*mix)),
            ),
        ),
        EffectDef::Delay {
            time_s,
            feedback_db,
            mix,
        } => EffectHandle::Delay(
            host.add_delay(
                DelayBuilder::new()
                    .delay_time(Duration::from_secs_f32(time_s.clamp(0.001, 10.0)))
                    .feedback(db(*feedback_db))
                    .mix(Mix(*mix)),
            ),
        ),
        EffectDef::Eq {
            kind,
            frequency,
            gain_db,
            q,
        } => EffectHandle::Eq(host.add_eq(EqFilterBuilder::new(
            match kind {
                EqKind::Bell => EqFilterKind::Bell,
                EqKind::LowShelf => EqFilterKind::LowShelf,
                EqKind::HighShelf => EqFilterKind::HighShelf,
            },
            *frequency as f64,
            Decibels(*gain_db),
            *q as f64,
        ))),
    }
}

struct BusTrack {
    track: TrackHandle,
    effects: Vec<EffectHandle>,
}

struct SendTrack {
    track: SendTrackHandle,
    /// Held for the track's lifetime (snapshots only target bus effects).
    _effects: Vec<EffectHandle>,
}

struct KiraVoice {
    sound: StaticSoundHandle,
    /// Keeps the spatial track alive for as long as the voice exists.
    spatial: Option<SpatialTrackHandle>,
    info: VoiceInfo,
}

struct Inner<B: Backend> {
    manager: AudioManager<B>,
    listener: ListenerHandle,
    buses: HashMap<String, BusTrack>,
    sends: HashMap<String, SendTrack>,
    voices: BTreeMap<VoiceId, KiraVoice>,
    next_id: VoiceId,
    applied: MixState,
}

/// Plays through kira's `AudioManager` (the default backend opens the
/// system output device).
pub struct KiraBackend<B: Backend = DefaultBackend> {
    inner: Mutex<Inner<B>>,
}

impl KiraBackend<DefaultBackend> {
    /// Opens the default output device.
    pub fn open() -> Result<Self, String> {
        Self::with_settings(AudioManagerSettings::default())
    }
}

impl<B: Backend + Send + 'static> KiraBackend<B>
where
    B::Error: std::fmt::Debug,
{
    pub fn with_settings(settings: AudioManagerSettings<B>) -> Result<Self, String> {
        let mut manager = AudioManager::<B>::new(settings)
            .map_err(|error| format!("cannot open audio output: {error:?}"))?;
        let listener = manager
            .add_listener(Vec3::ZERO, Quat::IDENTITY)
            .map_err(|error| format!("cannot create audio listener: {error:?}"))?;
        let mut backend = Self {
            inner: Mutex::new(Inner {
                manager,
                listener,
                buses: HashMap::new(),
                sends: HashMap::new(),
                voices: BTreeMap::new(),
                next_id: 0,
                applied: MixState::default(),
            }),
        };
        backend.configure(&AudioMixer::default());
        Ok(backend)
    }
}

impl<B: Backend> KiraBackend<B> {
    fn inner(&mut self) -> &mut Inner<B> {
        self.inner.get_mut().unwrap_or_else(|p| p.into_inner())
    }

    /// Direct access to kira's manager (mock-backend tests drive the
    /// renderer through it).
    pub fn with_manager<R>(&mut self, f: impl FnOnce(&mut AudioManager<B>) -> R) -> R {
        f(&mut self.inner().manager)
    }
}

impl<B: Backend> Inner<B> {
    fn build(&mut self, mixer: &AudioMixer) -> Result<(), String> {
        let mut sends = HashMap::new();
        for def in &mixer.sends {
            let mut builder = SendTrackBuilder::new().volume(db(def.volume_db));
            let effects = def
                .effects
                .iter()
                .map(|e| add_effect(&mut builder, e))
                .collect();
            let track = self
                .manager
                .add_send_track(builder)
                .map_err(|e| format!("send '{}': {e}", def.name))?;
            sends.insert(
                def.name.clone(),
                SendTrack {
                    track,
                    _effects: effects,
                },
            );
        }
        let mut buses: HashMap<String, BusTrack> = HashMap::new();
        let master_def = mixer.bus(MASTER);
        let mut builder = TrackBuilder::new();
        let effects = master_def
            .map(|d| {
                d.effects
                    .iter()
                    .map(|e| add_effect(&mut builder, e))
                    .collect()
            })
            .unwrap_or_default();
        let track = self
            .manager
            .add_sub_track(builder)
            .map_err(|e| format!("master bus: {e}"))?;
        buses.insert(MASTER.to_owned(), BusTrack { track, effects });
        for def in mixer.bus_order() {
            let mut builder = TrackBuilder::new();
            for send in &def.sends {
                if let Some(target) = sends.get(&send.send) {
                    builder = builder.with_send(target.track.id(), db(send.volume_db));
                }
            }
            let effects = def
                .effects
                .iter()
                .map(|e| add_effect(&mut builder, e))
                .collect();
            let parent = def.parent.as_deref().unwrap_or(MASTER);
            let parent = buses
                .get_mut(parent)
                .ok_or_else(|| format!("bus '{}' has no parent track", def.name))?;
            let track = parent
                .track
                .add_sub_track(builder)
                .map_err(|e| format!("bus '{}': {e}", def.name))?;
            buses.insert(def.name.clone(), BusTrack { track, effects });
        }
        // Dropping the old handles removes the old tracks once their
        // sounds finish.
        self.buses = buses;
        self.sends = sends;
        self.applied = MixState::default();
        Ok(())
    }
}

impl<B: Backend + Send + 'static> AudioBackend for KiraBackend<B> {
    fn name(&self) -> &'static str {
        "kira"
    }

    fn configure(&mut self, mixer: &AudioMixer) {
        if let Err(error) = self.inner().build(mixer) {
            log::error!(target: "engine::audio", "mixer setup failed: {error}");
        }
    }

    fn apply_mix(&mut self, mix: &MixState, fade: f32) {
        let fade = tween(fade);
        let inner = self.inner();
        for (bus, volume) in &mix.bus_volume {
            if inner.applied.bus_volume.get(bus) == Some(volume) {
                continue;
            }
            if let Some(track) = inner.buses.get_mut(bus) {
                track.track.set_volume(db(*volume), fade);
            }
        }
        for ((bus, send), volume) in &mix.send_levels {
            let key = (bus.clone(), send.clone());
            if inner.applied.send_levels.get(&key) == Some(volume) {
                continue;
            }
            if let (Some(track), Some(target)) = (inner.buses.get_mut(bus), inner.sends.get(send)) {
                let _ = track.track.set_send(target.track.id(), db(*volume), fade);
            }
        }
        for ((bus, index, param), value) in &mix.effect_params {
            let key = (bus.clone(), *index, *param);
            if inner.applied.effect_params.get(&key) == Some(value) {
                continue;
            }
            if let Some(effect) = inner
                .buses
                .get_mut(bus)
                .and_then(|t| t.effects.get_mut(*index))
            {
                effect.set(*param, *value, fade);
            }
        }
        inner.applied = mix.clone();
    }

    fn set_listener(&mut self, position: Vec3, rotation: Quat) {
        let inner = self.inner();
        inner.listener.set_position(position, Tween::default());
        inner.listener.set_orientation(rotation, Tween::default());
    }

    fn play(&mut self, request: PlayRequest) -> Option<VoiceId> {
        let inner = self.inner();
        let mut settings = StaticSoundSettings::new()
            .volume(db(request.volume_db))
            .playback_rate(PlaybackRate(request.pitch.max(0.0) as f64))
            .start_position(request.start_offset.max(0.0) as f64);
        if request.looping {
            settings = settings.loop_region(..);
        }
        if request.fade_in > 0.0 {
            settings = settings.fade_in_tween(tween(request.fade_in));
        }
        let data = request.clip.sound_data().with_settings(settings);
        let listener = inner.listener.id();
        let bus_name = if inner.buses.contains_key(&request.bus) {
            request.bus.clone()
        } else {
            MASTER.to_owned()
        };
        let bus = inner.buses.get_mut(&bus_name)?;
        let (sound, spatial) = match request.spatial {
            Some(spatial) => {
                let builder = SpatialTrackBuilder::new()
                    .distances(SpatialTrackDistances {
                        min_distance: spatial.min_distance.max(0.0),
                        max_distance: spatial.max_distance.max(spatial.min_distance + 1e-3),
                    })
                    .attenuation_function(None)
                    .spatialization_strength(spatial.spatial_blend.clamp(0.0, 1.0))
                    .persist_until_sounds_finish(true);
                let mut track = bus
                    .track
                    .add_spatial_sub_track(listener, spatial.position, builder)
                    .map_err(|e| log::warn!(target: "engine::audio", "voice dropped: {e}"))
                    .ok()?;
                let sound = track
                    .play(data)
                    .map_err(|e| log::warn!(target: "engine::audio", "voice dropped: {e}"))
                    .ok()?;
                (sound, Some(track))
            }
            None => {
                let sound = bus
                    .track
                    .play(data)
                    .map_err(|e| log::warn!(target: "engine::audio", "voice dropped: {e}"))
                    .ok()?;
                (sound, None)
            }
        };
        inner.next_id += 1;
        let id = inner.next_id;
        inner.voices.insert(
            id,
            KiraVoice {
                sound,
                spatial,
                info: VoiceInfo {
                    id,
                    bus: bus_name,
                    volume_db: request.volume_db,
                    pitch: request.pitch,
                    position: request.spatial.map(|s| s.position),
                    paused: false,
                    time: request.start_offset,
                },
            },
        );
        Some(id)
    }

    fn update_voice(&mut self, id: VoiceId, update: VoiceUpdate, fade: f32) {
        let fade = tween(fade);
        let Some(voice) = self.inner().voices.get_mut(&id) else {
            return;
        };
        if let Some(volume) = update.volume_db {
            if volume != voice.info.volume_db {
                voice.sound.set_volume(db(volume), fade);
                voice.info.volume_db = volume;
            }
        }
        if let Some(pitch) = update.pitch {
            if pitch != voice.info.pitch {
                voice
                    .sound
                    .set_playback_rate(PlaybackRate(pitch.max(0.0) as f64), fade);
                voice.info.pitch = pitch;
            }
        }
        if let Some(position) = update.position {
            if let Some(track) = &mut voice.spatial {
                track.set_position(position, fade);
            }
            voice.info.position = Some(position);
        }
    }

    fn set_paused(&mut self, id: VoiceId, paused: bool, fade: f32) {
        if let Some(voice) = self.inner().voices.get_mut(&id) {
            if voice.info.paused != paused {
                if paused {
                    voice.sound.pause(tween(fade));
                } else {
                    voice.sound.resume(tween(fade));
                }
                voice.info.paused = paused;
            }
        }
    }

    fn stop(&mut self, id: VoiceId, fade: f32) {
        if let Some(voice) = self.inner().voices.get_mut(&id) {
            voice.sound.stop(tween(fade));
        }
    }

    fn is_playing(&self, id: VoiceId) -> bool {
        let inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        inner
            .voices
            .get(&id)
            .is_some_and(|v| v.sound.state() != PlaybackState::Stopped)
    }

    fn update(&mut self, _dt: f32) {
        let inner = self.inner();
        inner.voices.retain(|_, voice| {
            voice.info.time = voice.sound.position() as f32;
            voice.sound.state() != PlaybackState::Stopped
        });
    }

    fn voices(&self) -> Vec<VoiceInfo> {
        let inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        inner.voices.values().map(|v| v.info.clone()).collect()
    }
}
