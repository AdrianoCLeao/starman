//! A silent, deterministic backend: voices advance with the frame clock at
//! their pitch, end when their clip ends and honor fades, so gameplay that
//! waits for sounds behaves exactly as with a device.

use std::collections::BTreeMap;

use engine_math::{Quat, Vec3};

use crate::backend::{AudioBackend, PlayRequest, VoiceId, VoiceInfo, VoiceUpdate};
use crate::mixer::{AudioMixer, MixState, MASTER};

#[derive(Clone, Debug)]
struct NullVoice {
    info: VoiceInfo,
    duration: f32,
    looping: bool,
    /// Seconds left until a requested stop completes.
    stopping: Option<f32>,
}

#[derive(Debug, Default)]
pub struct NullBackend {
    voices: BTreeMap<VoiceId, NullVoice>,
    next_id: VoiceId,
    buses: Vec<String>,
    mix: MixState,
    listener: (Vec3, Quat),
}

impl NullBackend {
    pub fn new() -> Self {
        Self::default()
    }

    /// The last mix applied (tests inspect snapshot resolution here).
    pub fn mix(&self) -> &MixState {
        &self.mix
    }

    pub fn listener(&self) -> (Vec3, Quat) {
        self.listener
    }

    pub fn buses(&self) -> &[String] {
        &self.buses
    }
}

impl AudioBackend for NullBackend {
    fn name(&self) -> &'static str {
        "null"
    }

    fn configure(&mut self, mixer: &AudioMixer) {
        self.buses = std::iter::once(MASTER.to_owned())
            .chain(mixer.bus_order().iter().map(|b| b.name.clone()))
            .collect();
    }

    fn apply_mix(&mut self, mix: &MixState, _fade: f32) {
        self.mix = mix.clone();
    }

    fn set_listener(&mut self, position: Vec3, rotation: Quat) {
        self.listener = (position, rotation);
    }

    fn play(&mut self, request: PlayRequest) -> Option<VoiceId> {
        self.next_id += 1;
        let id = self.next_id;
        let bus = if self.buses.contains(&request.bus) {
            request.bus
        } else {
            MASTER.to_owned()
        };
        self.voices.insert(
            id,
            NullVoice {
                info: VoiceInfo {
                    id,
                    bus,
                    volume_db: request.volume_db,
                    pitch: request.pitch.max(0.0),
                    position: request.spatial.map(|s| s.position),
                    paused: false,
                    time: request.start_offset.max(0.0),
                },
                duration: request.clip.duration(),
                looping: request.looping,
                stopping: None,
            },
        );
        Some(id)
    }

    fn update_voice(&mut self, id: VoiceId, update: VoiceUpdate, _fade: f32) {
        if let Some(voice) = self.voices.get_mut(&id) {
            if let Some(volume) = update.volume_db {
                voice.info.volume_db = volume;
            }
            if let Some(pitch) = update.pitch {
                voice.info.pitch = pitch.max(0.0);
            }
            if let Some(position) = update.position {
                voice.info.position = Some(position);
            }
        }
    }

    fn set_paused(&mut self, id: VoiceId, paused: bool, _fade: f32) {
        if let Some(voice) = self.voices.get_mut(&id) {
            voice.info.paused = paused;
        }
    }

    fn stop(&mut self, id: VoiceId, fade: f32) {
        if fade <= 0.0 {
            self.voices.remove(&id);
        } else if let Some(voice) = self.voices.get_mut(&id) {
            voice.stopping = Some(voice.stopping.map_or(fade, |left| left.min(fade)));
        }
    }

    fn is_playing(&self, id: VoiceId) -> bool {
        self.voices.contains_key(&id)
    }

    fn update(&mut self, dt: f32) {
        let dt = dt.max(0.0);
        self.voices.retain(|_, voice| {
            if let Some(left) = &mut voice.stopping {
                *left -= dt;
                if *left <= 0.0 {
                    return false;
                }
            }
            if voice.info.paused {
                return true;
            }
            voice.info.time += dt * voice.info.pitch;
            if voice.info.time < voice.duration {
                return true;
            }
            if voice.looping && voice.duration > 0.0 {
                voice.info.time %= voice.duration;
                true
            } else {
                false
            }
        });
    }

    fn voices(&self) -> Vec<VoiceInfo> {
        self.voices.values().map(|v| v.info.clone()).collect()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::clip::AudioClip;

    #[test]
    fn voices_end_loop_pause_and_fade_out_deterministically() {
        let mut backend = NullBackend::new();
        backend.configure(&AudioMixer::default());
        let clip = Arc::new(AudioClip::silence(1000, 0.5));
        let one_shot = backend
            .play(PlayRequest {
                pitch: 2.0,
                ..PlayRequest::new(clip.clone(), "sfx")
            })
            .unwrap();
        let looped = backend
            .play(PlayRequest {
                looping: true,
                ..PlayRequest::new(clip.clone(), "nope")
            })
            .unwrap();
        assert_eq!(backend.voices()[1].bus, MASTER);
        backend.update(0.2);
        assert!(backend.is_playing(one_shot));
        backend.update(0.06);
        assert!(
            !backend.is_playing(one_shot),
            "0.52s at 2x pitch > 0.5s clip"
        );
        backend.set_paused(looped, true, 0.0);
        backend.update(5.0);
        assert!((backend.voices()[0].time - 0.26).abs() < 1e-5);
        backend.set_paused(looped, false, 0.0);
        backend.update(0.3);
        assert!((backend.voices()[0].time - 0.06).abs() < 1e-5);
        backend.stop(looped, 0.1);
        backend.update(0.05);
        assert!(backend.is_playing(looped));
        backend.update(0.06);
        assert!(!backend.is_playing(looped));
    }
}
