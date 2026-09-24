//! ECS glue: listener, sources, zones, commands, finished events and debug
//! drawing, all in `UpdateSet::Audio`.

use std::collections::BTreeMap;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParam;
use engine_assets::{AssetRef, Assets, LoadState};
use engine_core::{DebugCategory, DebugDraw, FrameTime, GlobalTransform};
use engine_math::{Quat, Vec3};

use crate::backend::{attenuation, PlayRequest, SpatialParams, VoiceUpdate};
use crate::clip::AudioClip;
use crate::components::{
    AudioCommand, AudioFinished, AudioListener, AudioSource, AudioSourceState, AudioZone,
    AudioZoneState, OneShot, ZoneShape,
};
use crate::engine::{AudioEngine, AudioMixerSource, VoiceStart};
use crate::mixer::{gain_to_db, SnapshotDef, SILENCE_DB};
use crate::settings::AudioUserSettingsStore;

/// Seconds a one-shot may wait for its clip before being dropped.
pub const ONE_SHOT_LOAD_TIMEOUT: f32 = 2.0;
const PAUSE_FADE: f32 = 0.05;
const ZONE_AMBIENT_FADE: f32 = 0.5;

/// Where the listener is this frame.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct ListenerPose {
    pub position: Vec3,
    pub rotation: Quat,
}

impl Default for ListenerPose {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
        }
    }
}

/// One-shots waiting for their clip to load.
#[derive(Resource, Default)]
pub struct PendingOneShots(pub(crate) Vec<(OneShot, f32)>);

/// Effective gain of a positional voice for a listener position.
pub fn spatial_gain(
    source_pos: Vec3,
    listener: Vec3,
    min: f32,
    max: f32,
    rolloff: crate::Rolloff,
    blend: f32,
) -> f32 {
    let att = attenuation(source_pos.distance(listener), min, max, rolloff);
    1.0 + (att - 1.0) * blend.clamp(0.0, 1.0)
}

fn spatial_volume_db(base_db: f32, gain: f32) -> f32 {
    if gain <= 0.0 {
        SILENCE_DB
    } else {
        (base_db + gain_to_db(gain)).max(SILENCE_DB)
    }
}

fn clip_key(asset: &AssetRef) -> String {
    asset.request_key()
}

/// The clip for `asset`: `Ok(Some)` when loaded, `Ok(None)` while loading,
/// `Err` when it failed (or there is no asset store).
fn resolve_clip(assets: Option<&Assets>, asset: &AssetRef) -> Result<Option<Arc<AudioClip>>, ()> {
    let Some(assets) = assets else {
        return Err(());
    };
    if asset.is_empty() {
        return Err(());
    }
    let handle = assets.request::<AudioClip>(asset);
    if let Some(clip) = assets.get(handle) {
        return Ok(Some(clip));
    }
    match assets.state(handle) {
        Some(LoadState::Failed(_)) | None => Err(()),
        _ => Ok(None),
    }
}

#[derive(SystemParam)]
pub struct AudioFrame<'w> {
    engine: ResMut<'w, AudioEngine>,
    mixer_source: ResMut<'w, AudioMixerSource>,
    user: ResMut<'w, AudioUserSettingsStore>,
    pending: ResMut<'w, PendingOneShots>,
    listener_pose: ResMut<'w, ListenerPose>,
    time: Option<Res<'w, FrameTime>>,
    assets: Option<Res<'w, Assets>>,
    debug: Option<ResMut<'w, DebugDraw>>,
}

#[allow(clippy::too_many_arguments)]
pub fn update_audio(
    mut frame: AudioFrame,
    mut commands_in: EventReader<AudioCommand>,
    mut finished_out: EventWriter<AudioFinished>,
    mut commands: Commands,
    listeners: Query<(&AudioListener, &GlobalTransform)>,
    mut sources: SourceQuery,
    mut zones: ZoneQuery,
    mut removed_sources: RemovedComponents<AudioSource>,
    mut removed_zones: RemovedComponents<AudioZone>,
) {
    let dt = frame.time.as_ref().map_or(0.0, |t| t.real_delta_seconds);
    let assets = frame.assets.as_deref();
    let engine = &mut *frame.engine;

    if let Some(mixer) = frame.mixer_source.poll(assets) {
        engine.set_mixer(mixer);
    }

    // Listener.
    let pose = listeners
        .iter()
        .find(|(l, _)| l.enabled)
        .map(|(_, t)| ListenerPose {
            position: t.translation(),
            rotation: t.rotation(),
        })
        .unwrap_or_default();
    if *frame.listener_pose != pose {
        *frame.listener_pose = pose;
    }
    engine
        .backend_mut()
        .set_listener(pose.position, pose.rotation);

    // Removed sources / zones stop their voices.
    for entity in removed_sources.read() {
        let ids: Vec<_> = engine
            .voices()
            .iter()
            .filter(|(_, r)| r.entity == Some(entity) && r.tag.as_deref() != Some(ZONE_TAG))
            .map(|(id, _)| *id)
            .collect();
        for id in ids {
            engine.stop(id, PAUSE_FADE);
        }
    }
    for entity in removed_zones.read() {
        let ids: Vec<_> = engine
            .voices()
            .iter()
            .filter(|(_, r)| r.entity == Some(entity) && r.tag.as_deref() == Some(ZONE_TAG))
            .map(|(id, _)| *id)
            .collect();
        for id in ids {
            engine.stop(id, ZONE_AMBIENT_FADE);
        }
    }

    // Commands.
    for command in commands_in.read() {
        match command {
            AudioCommand::Play(entity) => {
                if let Ok((_, _, _, Some(mut state))) = sources.get_mut(*entity) {
                    if let Some(voice) = state.voice.take() {
                        engine.stop(voice, PAUSE_FADE);
                    }
                    state.pending_play = true;
                } else if sources.get(*entity).is_ok() {
                    commands.entity(*entity).insert(AudioSourceState {
                        pending_play: true,
                        pitch_scale: 1.0,
                        ..Default::default()
                    });
                }
            }
            AudioCommand::Stop { entity, fade } => {
                if let Ok((_, _, _, Some(mut state))) = sources.get_mut(*entity) {
                    state.pending_play = false;
                    if let Some(voice) = state.voice.take() {
                        engine.stop(voice, *fade);
                    }
                }
            }
            AudioCommand::SetPaused { entity, paused } => {
                if let Ok((_, _, _, Some(mut state))) = sources.get_mut(*entity) {
                    state.paused = *paused;
                    if let Some(voice) = state.voice {
                        engine.backend_mut().set_paused(voice, *paused, PAUSE_FADE);
                    }
                }
            }
            AudioCommand::OneShot(shot) => frame.pending.0.push((shot.clone(), 0.0)),
            AudioCommand::PushSnapshot(name) => {
                engine.push_snapshot(name);
            }
            AudioCommand::PopSnapshot(name) => engine.pop_snapshot(name),
            AudioCommand::SetUserVolume { bus, volume } => {
                frame.user.set_volume(bus.clone(), *volume)
            }
            AudioCommand::SetMuted(muted) => frame.user.set_muted(*muted),
            AudioCommand::PauseBus { bus, paused } => {
                let buses = engine.bus_subtree(bus);
                let ids: Vec<_> = engine
                    .voices()
                    .iter()
                    .filter(|(_, r)| buses.contains(&r.bus))
                    .map(|(id, _)| *id)
                    .collect();
                for id in ids {
                    engine.backend_mut().set_paused(id, *paused, PAUSE_FADE);
                }
            }
            AudioCommand::StopAll { bus, fade } => {
                let buses = bus.as_ref().map(|b| engine.bus_subtree(b));
                let ids: Vec<_> = engine
                    .voices()
                    .iter()
                    .filter(|(_, r)| buses.as_ref().is_none_or(|set| set.contains(&r.bus)))
                    .map(|(id, _)| *id)
                    .collect();
                for id in ids {
                    engine.stop(id, *fade);
                }
            }
        }
    }

    // One-shots (possibly waiting for their clip).
    let pending = std::mem::take(&mut frame.pending.0);
    for (shot, age) in pending {
        match resolve_clip(assets, &shot.clip) {
            Ok(Some(clip)) => {
                play_one_shot(engine, &shot, clip, pose.position);
            }
            Ok(None) if age + dt < ONE_SHOT_LOAD_TIMEOUT => frame.pending.0.push((shot, age + dt)),
            Ok(None) => {
                log::warn!(target: "engine::audio", "one-shot '{}' timed out loading", shot.clip.request_key())
            }
            Err(()) => {
                log::warn!(target: "engine::audio", "one-shot clip '{}' unavailable", shot.clip.request_key())
            }
        }
    }

    // Sources.
    for (entity, source, transform, state) in &mut sources {
        let Some(mut state) = state else {
            commands.entity(entity).insert(AudioSourceState {
                pending_play: source.autoplay,
                pitch_scale: 1.0,
                clip_key: clip_key(&source.clip),
                ..Default::default()
            });
            continue;
        };
        let key = clip_key(&source.clip);
        if state.clip_key != key {
            // Clip changed: restart if it was playing or autoplays.
            let was_playing = state.voice.is_some();
            if let Some(voice) = state.voice.take() {
                engine.stop(voice, PAUSE_FADE);
            }
            state.clip_key = key;
            state.autoplayed = false;
            state.pending_play = was_playing || source.autoplay;
        }
        if state.voice.is_some_and(|v| !engine.is_playing(v)) {
            state.voice = None;
        }
        let position = transform.map_or(Vec3::ZERO, GlobalTransform::translation);
        if state.pending_play {
            match resolve_clip(assets, &source.clip) {
                Ok(Some(clip)) => {
                    state.pending_play = false;
                    state.autoplayed = true;
                    state.volume_offset_db = engine.random_signed() * source.volume_variation_db;
                    state.pitch_scale = 1.0 + engine.random_signed() * source.pitch_variation;
                    let gain = if source.spatial {
                        spatial_gain(
                            position,
                            pose.position,
                            source.min_distance,
                            source.max_distance,
                            source.rolloff,
                            source.spatial_blend,
                        )
                    } else {
                        1.0
                    };
                    let request = PlayRequest {
                        volume_db: spatial_volume_db(
                            source.volume_db + state.volume_offset_db,
                            gain,
                        ),
                        pitch: (source.pitch * state.pitch_scale).max(0.01),
                        looping: source.looping,
                        fade_in: source.fade_in,
                        spatial: source.spatial.then_some(SpatialParams {
                            position,
                            min_distance: source.min_distance,
                            max_distance: source.max_distance,
                            rolloff: source.rolloff,
                            spatial_blend: source.spatial_blend,
                        }),
                        ..PlayRequest::new(clip, source.bus.clone())
                    };
                    state.voice = engine.start(VoiceStart {
                        request,
                        priority: source.priority,
                        entity: Some(entity),
                        clip_key: state.clip_key.clone(),
                        tag: None,
                        max_instances: source.max_instances,
                    });
                    if state.paused {
                        if let Some(voice) = state.voice {
                            engine.backend_mut().set_paused(voice, true, 0.0);
                        }
                    }
                }
                Ok(None) => {}
                Err(()) => {
                    state.pending_play = false;
                    log::warn!(target: "engine::audio", "audio source clip '{}' unavailable", source.clip.request_key());
                }
            }
        } else if let Some(voice) = state.voice {
            let gain = if source.spatial {
                spatial_gain(
                    position,
                    pose.position,
                    source.min_distance,
                    source.max_distance,
                    source.rolloff,
                    source.spatial_blend,
                )
            } else {
                1.0
            };
            engine.backend_mut().update_voice(
                voice,
                VoiceUpdate {
                    volume_db: Some(spatial_volume_db(
                        source.volume_db + state.volume_offset_db,
                        gain,
                    )),
                    pitch: Some((source.pitch * state.pitch_scale).max(0.01)),
                    position: source.spatial.then_some(position),
                },
                MIX_FRAME_TWEEN,
            );
        }
    }

    // Zones → weighted snapshots, send levels and ambient loops.
    let mut extra: Vec<(SnapshotDef, f32)> = Vec::new();
    let mut ordered: Vec<(i32, Entity)> =
        zones.iter().map(|(e, z, _, _)| (z.priority, e)).collect();
    ordered.sort();
    for (_, entity) in ordered {
        let Ok((entity, zone, transform, state)) = zones.get_mut(entity) else {
            continue;
        };
        let weight = zone.weight_at(transform, pose.position);
        let Some(mut state) = state else {
            commands.entity(entity).insert(AudioZoneState::default());
            continue;
        };
        state.weight = weight;
        if weight > 0.0 {
            if let Some(snapshot) = engine.mixer().snapshot(&zone.snapshot) {
                let mut def = snapshot.clone();
                def.priority = def.priority.max(zone.priority);
                extra.push((def, weight));
            }
            if !zone.reverb_send.is_empty() {
                extra.push((
                    SnapshotDef {
                        name: format!("zone:{entity:?}"),
                        priority: zone.priority,
                        fade: 0.0,
                        buses: Vec::new(),
                        sends: vec![(
                            zone.send_bus.clone(),
                            zone.reverb_send.clone(),
                            zone.send_db,
                        )],
                        effects: Vec::new(),
                    },
                    weight,
                ));
            }
        }
        if state.ambient_voice.is_some_and(|v| !engine.is_playing(v)) {
            state.ambient_voice = None;
        }
        let ambient_db = spatial_volume_db(zone.ambient_volume_db, weight);
        match (
            state.ambient_voice,
            weight > 0.0 && !zone.ambient.is_empty(),
        ) {
            (None, true) => {
                if let Ok(Some(clip)) = resolve_clip(assets, &zone.ambient) {
                    state.ambient_voice = engine.start(VoiceStart {
                        request: PlayRequest {
                            volume_db: ambient_db,
                            looping: true,
                            fade_in: ZONE_AMBIENT_FADE,
                            ..PlayRequest::new(clip, zone.ambient_bus.clone())
                        },
                        priority: i32::MAX / 2,
                        entity: Some(entity),
                        clip_key: clip_key(&zone.ambient),
                        tag: Some(ZONE_TAG.to_owned()),
                        max_instances: 0,
                    });
                }
            }
            (Some(voice), true) => engine.backend_mut().update_voice(
                voice,
                VoiceUpdate {
                    volume_db: Some(ambient_db),
                    ..Default::default()
                },
                MIX_FRAME_TWEEN,
            ),
            (Some(voice), false) => {
                engine.stop(voice, ZONE_AMBIENT_FADE);
                state.ambient_voice = None;
            }
            (None, false) => {}
        }
    }

    let gains: BTreeMap<String, f32> = frame.user.settings.gains();
    engine.step(dt, &extra, &gains);
    frame.user.tick(dt);

    for (voice, record) in engine.take_finished() {
        if record.tag.as_deref() == Some(ZONE_TAG) {
            continue;
        }
        finished_out.send(AudioFinished {
            voice,
            entity: record.entity,
            tag: record.tag,
        });
    }

    if let Some(debug) = frame.debug.as_deref_mut() {
        if debug.is_enabled(DebugCategory::Audio) {
            draw_debug(debug, &pose, &sources, &zones);
        }
    }
}

/// Tag for zone ambient voices (not reported as [`AudioFinished`]).
const ZONE_TAG: &str = "\u{0}zone";
/// Tween for per-frame voice parameter changes.
const MIX_FRAME_TWEEN: f32 = 0.03;

fn play_one_shot(engine: &mut AudioEngine, shot: &OneShot, clip: Arc<AudioClip>, listener: Vec3) {
    let volume = shot.volume_db + engine.random_signed() * shot.volume_variation_db;
    let pitch = (shot.pitch * (1.0 + engine.random_signed() * shot.pitch_variation)).max(0.01);
    let gain = shot.position.map_or(1.0, |p| {
        spatial_gain(
            p,
            listener,
            shot.min_distance,
            shot.max_distance,
            shot.rolloff,
            1.0,
        )
    });
    let request = PlayRequest {
        volume_db: spatial_volume_db(volume, gain),
        pitch,
        spatial: shot.position.map(|position| SpatialParams {
            position,
            min_distance: shot.min_distance,
            max_distance: shot.max_distance,
            rolloff: shot.rolloff,
            spatial_blend: 1.0,
        }),
        ..PlayRequest::new(clip, shot.bus.clone())
    };
    engine.start(VoiceStart {
        request,
        priority: shot.priority,
        entity: None,
        clip_key: clip_key(&shot.clip),
        tag: shot.tag.clone(),
        max_instances: 0,
    });
}

type SourceQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static AudioSource,
        Option<&'static GlobalTransform>,
        Option<&'static mut AudioSourceState>,
    ),
>;
type ZoneQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static AudioZone,
        &'static GlobalTransform,
        Option<&'static mut AudioZoneState>,
    ),
>;

fn draw_debug(
    debug: &mut DebugDraw,
    pose: &ListenerPose,
    sources: &SourceQuery,
    zones: &ZoneQuery,
) {
    const LISTENER: [f32; 4] = [0.3, 0.9, 1.0, 1.0];
    const PLAYING: [f32; 4] = [1.0, 0.85, 0.2, 1.0];
    const IDLE: [f32; 4] = [0.6, 0.6, 0.6, 0.6];
    const ZONE: [f32; 4] = [0.4, 0.6, 1.0, 0.9];
    const FADE: [f32; 4] = [0.4, 0.6, 1.0, 0.35];
    debug.cross(pose.position, 0.4, LISTENER);
    debug.arrow(
        pose.position,
        pose.position + pose.rotation * Vec3::NEG_Z,
        LISTENER,
    );
    for (_, source, transform, state) in sources.iter() {
        let (Some(transform), true) = (transform, source.spatial) else {
            continue;
        };
        let position = transform.translation();
        let playing = state.is_some_and(|s| s.voice.is_some());
        let color = if playing { PLAYING } else { IDLE };
        debug.sphere(position, source.min_distance.max(0.05), color);
        let mut far = color;
        far[3] *= 0.4;
        debug.sphere(position, source.max_distance, far);
    }
    for (_, zone, transform, state) in zones.iter() {
        let t = transform.compute_transform();
        let active = state.is_some_and(|s| s.weight > 0.0);
        let color = if active { PLAYING } else { ZONE };
        match zone.scaled_shape(t.scale) {
            ZoneShape::Box { half_extents: half } => {
                debug.oriented_box(t.translation, t.rotation, half, color);
                debug.oriented_box(
                    t.translation,
                    t.rotation,
                    half + Vec3::splat(zone.fade_distance),
                    FADE,
                );
            }
            ZoneShape::Sphere { radius: r } => {
                debug.sphere(t.translation, r, color);
                debug.sphere(t.translation, r + zone.fade_distance, FADE);
            }
        }
    }
}
