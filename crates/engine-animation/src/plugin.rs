//! Runtime plugin installing the animation pipeline.

use bevy_ecs::schedule::IntoSystemConfigs;
use engine_assets::Assets;
use engine_core::{GameRuntime, PreRenderSet, RuntimePlugin, ScheduleKind, UpdateSet};
use engine_reflect::ReflectRegistration;

use crate::clip::AnimationClipLoader;
use crate::components::{
    AnimationEvent, AnimationPlayer, Animator, BoneAttachment, InverseKinematics, LookAtChain,
    PlaybackLoop, SkinnedMesh, TwoBoneChain,
};
use crate::graph::AnimGraphLoader;
use crate::property::{apply_property_writes, PropertyWrites};
use crate::skeleton::SkeletonLoader;
use crate::systems::*;

#[derive(Default)]
pub struct AnimationPlugin;

impl RuntimePlugin for AnimationPlugin {
    fn name(&self) -> &'static str {
        "engine::animation"
    }

    fn build(&self, runtime: &mut GameRuntime) {
        if let Some(assets) = runtime.world.get_resource::<Assets>() {
            assets.register_loader(SkeletonLoader);
            assets.register_loader(AnimationClipLoader);
            assets.register_loader(AnimGraphLoader);
        }
        runtime
            .init_resource::<PropertyWrites>()
            .add_event::<AnimationEvent>()
            .add_systems(
                ScheduleKind::Update,
                (
                    (
                        attach_runtime_state,
                        resolve_skeletons,
                        resolve_animator_assets,
                        resolve_player_assets,
                        evaluate_animators,
                        evaluate_players,
                    )
                        .chain()
                        .in_set(UpdateSet::AnimationGraph),
                    sample_poses.in_set(UpdateSet::AnimationSample),
                    (
                        apply_inverse_kinematics,
                        apply_root_motion,
                        apply_property_writes,
                        update_bone_attachments,
                    )
                        .chain()
                        .in_set(UpdateSet::AnimationApply),
                ),
            )
            .add_systems(
                ScheduleKind::PreRender,
                compute_skin_palettes.in_set(PreRenderSet::SkinPalette),
            );
        engine_reflect::with_reflection_registries(
            &mut runtime.world,
            |types, components, metadata| {
                types.register::<engine_assets::AssetRef>();
                types.register::<PlaybackLoop>();
                types.register::<TwoBoneChain>();
                types.register::<LookAtChain>();
                types.register::<Vec<TwoBoneChain>>();
                types.register::<Vec<LookAtChain>>();
                SkinnedMesh::register_reflect(types, components, metadata);
                Animator::register_reflect(types, components, metadata);
                AnimationPlayer::register_reflect(types, components, metadata);
                BoneAttachment::register_reflect(types, components, metadata);
                InverseKinematics::register_reflect(types, components, metadata);
            },
        );
    }
}
