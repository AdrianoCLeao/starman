//! Runtime plugin: loaders, resources, events and the audio system.

use bevy_ecs::schedule::IntoSystemConfigs;
use engine_assets::Assets;
use engine_core::{GameRuntime, RuntimePlugin, ScheduleKind, UpdateSet};
use engine_reflect::ReflectRegistration;

use crate::backend::{AudioBackend, Rolloff};
use crate::clip::AudioClipLoader;
use crate::components::{
    AudioCommand, AudioFinished, AudioListener, AudioSource, AudioZone, ZoneShape,
};
use crate::engine::{AudioEngine, AudioMixerSource};
use crate::kira_backend::KiraBackend;
use crate::mixer::AudioMixerLoader;
use crate::null::NullBackend;
use crate::settings::AudioUserSettingsStore;
use crate::systems::{update_audio, ListenerPose, PendingOneShots};

/// Installs audio with the silent [`NullBackend`]; hosts that own an output
/// device swap in [`open_device_backend`] via [`AudioEngine::set_backend`].
#[derive(Default)]
pub struct AudioPlugin;

impl RuntimePlugin for AudioPlugin {
    fn name(&self) -> &'static str {
        "engine::audio"
    }

    fn build(&self, runtime: &mut GameRuntime) {
        if let Some(assets) = runtime.world.get_resource::<Assets>() {
            assets.register_loader(AudioClipLoader);
            assets.register_loader(AudioMixerLoader);
        }
        runtime
            .init_resource::<AudioEngine>()
            .init_resource::<AudioMixerSource>()
            .init_resource::<AudioUserSettingsStore>()
            .init_resource::<PendingOneShots>()
            .init_resource::<ListenerPose>()
            .add_event::<AudioCommand>()
            .add_event::<AudioFinished>()
            .add_systems(ScheduleKind::Update, update_audio.in_set(UpdateSet::Audio));
        engine_reflect::with_reflection_registries(
            &mut runtime.world,
            |types, components, metadata| {
                types.register::<engine_assets::AssetRef>();
                types.register::<Rolloff>();
                types.register::<ZoneShape>();
                AudioSource::register_reflect(types, components, metadata);
                AudioListener::register_reflect(types, components, metadata);
                AudioZone::register_reflect(types, components, metadata);
            },
        );
    }
}

/// The system output device through kira, or the [`NullBackend`] (with a
/// warning) when no device can be opened — a missing sound card never
/// prevents the game from running.
pub fn open_device_backend() -> Box<dyn AudioBackend> {
    match KiraBackend::open() {
        Ok(backend) => Box::new(backend),
        Err(error) => {
            log::warn!(target: "engine::audio", "{error}; continuing without audio output");
            Box::new(NullBackend::new())
        }
    }
}
