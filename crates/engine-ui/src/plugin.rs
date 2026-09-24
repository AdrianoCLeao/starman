//! Runtime plugin: UI assets, resources, the update system and the
//! render node.

use bevy_ecs::schedule::IntoSystemConfigs;
use engine_assets::Assets;
use engine_core::{GameRuntime, RuntimePlugin, ScheduleKind, UpdateSet};
use engine_reflect::ReflectRegistration;
use engine_render::extension::RenderExtensions;

use crate::document::UiLayoutLoader;
use crate::draw::UiDrawList;
use crate::model::UiModel;
use crate::render::UiRenderer;
use crate::style::UiStyleSheetLoader;
use crate::systems::{update_ui, UiDocument, UiEvent, UiInputCapture, UiLayoutTrees, UiSettings};
use crate::text::{FontLoader, UiFonts};

#[derive(Default)]
pub struct UiPlugin;

impl RuntimePlugin for UiPlugin {
    fn name(&self) -> &'static str {
        "engine::ui"
    }

    fn build(&self, runtime: &mut GameRuntime) {
        if let Some(assets) = runtime.world.get_resource::<Assets>() {
            assets.register_loader(UiLayoutLoader);
            assets.register_loader(UiStyleSheetLoader);
            assets.register_loader(FontLoader);
        }
        if !runtime.has_plugin("engine::localization") {
            runtime.add_plugin(engine_localization::LocalizationPlugin);
        }
        runtime
            .init_resource::<UiModel>()
            .init_resource::<UiSettings>()
            .init_resource::<UiDrawList>()
            .init_resource::<UiInputCapture>()
            .init_resource::<RenderExtensions>()
            .add_event::<UiEvent>()
            .add_systems(ScheduleKind::Update, update_ui.in_set(UpdateSet::UiLayout));
        runtime.world.insert_resource(UiFonts::new());
        runtime
            .world
            .insert_non_send_resource(UiLayoutTrees::default());
        runtime
            .world
            .resource::<RenderExtensions>()
            .register("engine::ui", || Box::new(UiRenderer::default()));
        engine_reflect::with_reflection_registries(
            &mut runtime.world,
            |types, components, metadata| {
                types.register::<engine_assets::AssetRef>();
                UiDocument::register_reflect(types, components, metadata);
            },
        );
    }
}
