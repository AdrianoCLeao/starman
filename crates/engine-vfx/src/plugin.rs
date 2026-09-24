//! Runtime plugin: effect loader, emitter systems and the render node.

use bevy_ecs::schedule::IntoSystemConfigs;
use engine_assets::Assets;
use engine_core::{GameRuntime, RuntimePlugin, ScheduleKind, UpdateSet};
use engine_reflect::ReflectRegistration;
use engine_render::extension::RenderExtensions;

use crate::components::{ParticleCommand, ParticleEmitter};
use crate::effect::ParticleEffectLoader;
use crate::render::ParticleRenderer;
use crate::systems::{update_particles, ParticleSettings};

#[derive(Default)]
pub struct VfxPlugin;

impl RuntimePlugin for VfxPlugin {
    fn name(&self) -> &'static str {
        "engine::vfx"
    }

    fn build(&self, runtime: &mut GameRuntime) {
        if let Some(assets) = runtime.world.get_resource::<Assets>() {
            assets.register_loader(ParticleEffectLoader);
        }
        runtime
            .init_resource::<ParticleSettings>()
            .init_resource::<RenderExtensions>()
            .add_event::<ParticleCommand>()
            .add_systems(
                ScheduleKind::Update,
                update_particles.in_set(UpdateSet::Particles),
            );
        runtime
            .world
            .resource::<RenderExtensions>()
            .register("engine::vfx::particles", || {
                Box::new(ParticleRenderer::default())
            });
        engine_reflect::with_reflection_registries(
            &mut runtime.world,
            |types, components, metadata| {
                types.register::<engine_assets::AssetRef>();
                ParticleEmitter::register_reflect(types, components, metadata);
            },
        );
    }
}
