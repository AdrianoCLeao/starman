//! Nested scene composition: live instances, override stacks, structural
//! diffs, and cycle detection (M2 / ADR 0007).

mod cycle;
mod diff;
mod graph;
mod ops;
mod path;
mod resolve;

pub use cycle::CycleDetector;
pub use diff::{diff_from_overrides, DiffChange, StructuralDiff};
pub use graph::CompositionGraph;
pub use ops::{
    add_local_entity, apply_overrides_to_source, load_scene_file, local_parent_template,
    mark_removed, new_instance, promote_local_entity, revert_overrides, save_scene_file,
    set_override,
};
pub use path::InstancePath;
pub use resolve::{count_named, expand_all_instances, resync_instance, InstanceResolver};

// Re-export authored format pieces used by consumers of this crate.
pub use engine_assets::{
    InheritedEntity, InstanceLocalEntity, LocalAddedEntity, LocalParent, OverrideEntry, SceneFile,
    SceneInstance, SceneInstanceData, SceneValue,
};

pub fn module_name() -> &'static str {
    "engine-scene"
}
