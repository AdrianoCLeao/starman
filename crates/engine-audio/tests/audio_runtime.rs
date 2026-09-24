//! Audio through the runtime with the deterministic null backend, and the
//! kira graph built on kira's mock backend (no device needed).

use std::sync::Arc;

use engine_assets::{AssetRef, Assets};
use engine_audio::*;
use engine_core::{GameRuntime, GlobalTransform, Transform};
use engine_math::Vec3;

const DT: f32 = 1.0 / 30.0;

fn runtime() -> (GameRuntime, Assets) {
    let assets = Assets::new();
    let mut runtime = GameRuntime::new();
    runtime.insert_resource(assets.clone());
    runtime.add_plugin(AudioPlugin);
    (runtime, assets)
}

fn install_clip(assets: &Assets, path: &str, seconds: f32) {
    let handle = assets.request::<AudioClip>(&AssetRef::from_path(path));
    assert!(assets.replace(handle, AudioClip::silence(1000, seconds)));
}

fn mixer() -> AudioMixer {
    ron::from_str(
        r#"(
        buses: [
            (name: "music"),
            (name: "sfx", sends: [(send: "reverb", volume_db: -80.0)], max_voices: Some(2)),
            (name: "ambience"),
        ],
        sends: [(name: "reverb", effects: [Reverb(mix: 1.0)])],
        snapshots: [
            (name: "paused", fade: 0.2, buses: [("sfx", -30.0)]),
            (name: "cave", fade: 0.0, buses: [("music", -10.0)]),
        ],
    )"#,
    )
    .unwrap()
}

fn at(position: Vec3) -> (Transform, GlobalTransform) {
    (
        Transform::from_translation(position),
        GlobalTransform::from_translation(position),
    )
}

fn null_voices(runtime: &GameRuntime) -> Vec<VoiceInfo> {
    runtime.world.resource::<AudioEngine>().voice_infos()
}

#[test]
fn sources_autoplay_attenuate_follow_and_report_finishing() {
    let (mut runtime, assets) = runtime();
    install_clip(&assets, "audio/step.wav", 0.5);
    runtime
        .world
        .resource_mut::<AudioMixerSource>()
        .set_inline(mixer());
    runtime
        .world
        .spawn((AudioListener::default(), at(Vec3::ZERO)));
    let near = runtime
        .world
        .spawn((
            AudioSource {
                clip: AssetRef::from_path("audio/step.wav"),
                autoplay: true,
                min_distance: 1.0,
                max_distance: 11.0,
                rolloff: Rolloff::Linear,
                ..Default::default()
            },
            at(Vec3::new(6.0, 0.0, 0.0)),
        ))
        .id();
    runtime.step(DT); // state inserted
    runtime.step(DT); // voice started
    let voices = null_voices(&runtime);
    assert_eq!(voices.len(), 1);
    assert_eq!(voices[0].bus, "sfx");
    // Halfway between min and max with linear rolloff → gain 0.5 (-6 dB).
    assert!(
        (voices[0].volume_db - gain_to_db(0.5)).abs() < 0.01,
        "{}",
        voices[0].volume_db
    );
    assert_eq!(voices[0].position, Some(Vec3::new(6.0, 0.0, 0.0)));

    // Moving the source updates volume and position.
    *runtime.world.get_mut::<GlobalTransform>(near).unwrap() =
        GlobalTransform::from_translation(Vec3::new(0.5, 0.0, 0.0));
    runtime.step(DT);
    let voices = null_voices(&runtime);
    assert!(voices[0].volume_db.abs() < 1e-4);
    assert_eq!(voices[0].position, Some(Vec3::new(0.5, 0.0, 0.0)));

    // The 0.5 s clip ends and reports the entity.
    let mut finished = Vec::new();
    for _ in 0..20 {
        runtime.step(DT);
        finished.extend(
            runtime
                .world
                .resource_mut::<bevy_ecs::event::Events<AudioFinished>>()
                .drain(),
        );
    }
    assert_eq!(finished.len(), 1);
    assert_eq!(finished[0].entity, Some(near));
    assert!(null_voices(&runtime).is_empty());
    assert!(runtime
        .world
        .get::<AudioSourceState>(near)
        .unwrap()
        .voice
        .is_none());

    // Play again on command; despawn stops it.
    runtime.world.send_event(AudioCommand::Play(near));
    runtime.step(DT);
    assert_eq!(null_voices(&runtime).len(), 1);
    runtime.world.despawn(near);
    runtime.step(DT);
    runtime.step(DT);
    runtime.step(DT);
    assert!(null_voices(&runtime).is_empty());
}

#[test]
fn one_shots_wait_for_clips_respect_bus_limits_and_echo_tags() {
    let (mut runtime, assets) = runtime();
    runtime
        .world
        .resource_mut::<AudioMixerSource>()
        .set_inline(mixer());
    for i in 0..3 {
        let mut shot = OneShot::new(AssetRef::from_path("audio/hit.wav"), "sfx");
        shot.priority = i;
        shot.tag = Some(format!("hit{i}"));
        shot.pitch_variation = 0.1;
        runtime.world.send_event(AudioCommand::OneShot(shot));
    }
    let handle = assets.request::<AudioClip>(&AssetRef::from_path("audio/hit.wav"));
    runtime.step(DT);
    assert!(null_voices(&runtime).is_empty(), "clip not loaded yet");
    assert!(assets.replace(handle, AudioClip::silence(1000, 0.2)));
    runtime.step(DT);
    // sfx allows 2 voices: the third (priority 2) stole priority 0, which
    // is still fading out in the backend.
    assert_eq!(runtime.world.resource::<AudioEngine>().voices().len(), 2);
    let voices = null_voices(&runtime);
    assert_eq!(voices.len(), 3);
    assert!(voices.iter().all(|v| (0.9..=1.1).contains(&v.pitch)));
    let finished: Vec<AudioFinished> = runtime
        .world
        .resource_mut::<bevy_ecs::event::Events<AudioFinished>>()
        .drain()
        .collect();
    assert_eq!(finished.len(), 1);
    assert_eq!(finished[0].tag.as_deref(), Some("hit0"));
}

#[test]
fn snapshots_zones_user_volume_and_bus_pause_shape_the_mix() {
    let (mut runtime, assets) = runtime();
    install_clip(&assets, "audio/wind.ogg", 3.0);
    runtime
        .world
        .resource_mut::<AudioMixerSource>()
        .set_inline(mixer());
    let listener = runtime
        .world
        .spawn((AudioListener::default(), at(Vec3::new(20.0, 0.0, 0.0))))
        .id();
    let zone = runtime
        .world
        .spawn((
            AudioZone {
                shape: ZoneShape::Sphere { radius: 4.0 },
                fade_distance: 2.0,
                snapshot: "cave".into(),
                reverb_send: "reverb".into(),
                send_bus: "sfx".into(),
                send_db: -6.0,
                ambient: AssetRef::from_path("audio/wind.ogg"),
                ambient_bus: "ambience".into(),
                ..Default::default()
            },
            at(Vec3::ZERO),
        ))
        .id();
    runtime.step(DT);
    runtime.step(DT);
    let mix = |runtime: &GameRuntime| runtime.world.resource::<AudioEngine>().mix().clone();
    assert_eq!(mix(&runtime).bus_volume["music"], 0.0);
    assert!(null_voices(&runtime).is_empty());

    // Listener in the fade band (distance 1 outside → weight 0.5).
    *runtime.world.get_mut::<GlobalTransform>(listener).unwrap() =
        GlobalTransform::from_translation(Vec3::new(5.0, 0.0, 0.0));
    runtime.step(DT);
    assert_eq!(
        runtime.world.get::<AudioZoneState>(zone).unwrap().weight,
        0.5
    );
    let m = mix(&runtime);
    assert!((m.bus_volume["music"] - -5.0).abs() < 1e-4);
    assert!((m.send_levels[&("sfx".into(), "reverb".into())] - -43.0).abs() < 1e-3);
    let voices = null_voices(&runtime);
    assert_eq!(voices.len(), 1, "ambient loop");
    assert_eq!(voices[0].bus, "ambience");
    assert!((voices[0].volume_db - gain_to_db(0.5)).abs() < 1e-3);

    // Pushed snapshot fades in over 0.2 s; user volume and mute apply.
    runtime
        .world
        .send_event(AudioCommand::PushSnapshot("paused".into()));
    runtime.world.send_event(AudioCommand::SetUserVolume {
        bus: "music".into(),
        volume: 0.5,
    });
    for _ in 0..8 {
        runtime.step(DT);
    }
    let m = mix(&runtime);
    assert_eq!(m.bus_volume["sfx"], -30.0);
    assert!((m.bus_volume["music"] - (-5.0 + gain_to_db(0.5))).abs() < 1e-3);
    runtime.world.send_event(AudioCommand::SetMuted(true));
    runtime.step(DT);
    assert_eq!(mix(&runtime).bus_volume[MASTER], SILENCE_DB);

    // Pausing the ambience bus pauses the loop.
    runtime.world.send_event(AudioCommand::PauseBus {
        bus: "ambience".into(),
        paused: true,
    });
    runtime.step(DT);
    assert!(null_voices(&runtime)[0].paused);

    // Leaving the zone fades the ambient loop out and drops its effect.
    *runtime.world.get_mut::<GlobalTransform>(listener).unwrap() =
        GlobalTransform::from_translation(Vec3::new(50.0, 0.0, 0.0));
    for _ in 0..30 {
        runtime.step(DT);
    }
    assert!(null_voices(&runtime).is_empty());
    let m = mix(&runtime);
    assert_eq!(m.bus_volume["music"], gain_to_db(0.5));
    assert_eq!(m.send_levels[&("sfx".into(), "reverb".into())], -80.0);
    assert!(
        runtime
            .world
            .resource::<bevy_ecs::event::Events<AudioFinished>>()
            .is_empty(),
        "zone ambience is internal"
    );
}

#[test]
fn kira_backend_builds_the_bus_graph_and_plays_on_the_mock_device() {
    use kira::backend::mock::{MockBackend, MockBackendSettings};
    let settings = kira::AudioManagerSettings::<MockBackend> {
        backend_settings: MockBackendSettings { sample_rate: 1000 },
        ..Default::default()
    };
    let mut backend = KiraBackend::with_settings(settings).unwrap();
    let mixer = mixer();
    backend.configure(&mixer);
    backend.apply_mix(&resolve_mix(&mixer, &[], &Default::default()), 0.0);
    let clip = Arc::new(AudioClip::silence(1000, 0.05));
    let flat = backend.play(PlayRequest::new(clip.clone(), "sfx")).unwrap();
    let spatial = backend
        .play(PlayRequest {
            spatial: Some(SpatialParams {
                position: Vec3::new(3.0, 0.0, 0.0),
                min_distance: 1.0,
                max_distance: 10.0,
                rolloff: Rolloff::Linear,
                spatial_blend: 1.0,
            }),
            looping: true,
            ..PlayRequest::new(clip, "unknown-bus")
        })
        .unwrap();
    assert_eq!(backend.voices()[1].bus, MASTER);
    backend.update_voice(
        spatial,
        VoiceUpdate {
            volume_db: Some(-3.0),
            position: Some(Vec3::ONE),
            ..Default::default()
        },
        0.0,
    );
    // Render ~0.2 s of audio.
    backend.with_manager(|manager| {
        for _ in 0..200 {
            manager.backend_mut().on_start_processing();
            manager.backend_mut().process();
        }
    });
    backend.update(0.2);
    assert!(!backend.is_playing(flat), "one-shot finished");
    assert!(backend.is_playing(spatial), "loop keeps playing");
    backend.stop(spatial, 0.0);
    backend.with_manager(|manager| {
        for _ in 0..10 {
            manager.backend_mut().on_start_processing();
            manager.backend_mut().process();
        }
    });
    backend.update(0.01);
    assert!(backend.voices().is_empty());
}
