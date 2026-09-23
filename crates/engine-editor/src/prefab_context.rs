//! Prefab / nested-scene edit context stack (breadcrumb navigation).

use std::path::PathBuf;

use engine_core::SourceAssetId;

#[derive(Clone, Debug)]
pub struct PrefabEditFrame {
    pub scene_path: PathBuf,
    #[allow(dead_code)]
    pub scene_id: Option<SourceAssetId>,
    pub label: String,
    #[allow(dead_code)]
    pub unsaved: bool,
}

#[derive(Default, Debug)]
pub struct PrefabEditStack {
    pub frames: Vec<PrefabEditFrame>,
}

impl PrefabEditStack {
    pub fn push(&mut self, frame: PrefabEditFrame) {
        self.frames.push(frame);
    }

    pub fn pop(&mut self) -> Option<PrefabEditFrame> {
        self.frames.pop()
    }

    pub fn current(&self) -> Option<&PrefabEditFrame> {
        self.frames.last()
    }

    pub fn current_mut(&mut self) -> Option<&mut PrefabEditFrame> {
        self.frames.last_mut()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    pub fn breadcrumbs(&self) -> Vec<&str> {
        self.frames.iter().map(|f| f.label.as_str()).collect()
    }

    pub fn truncate_to(&mut self, index: usize) -> Vec<PrefabEditFrame> {
        if index + 1 >= self.frames.len() {
            return Vec::new();
        }
        self.frames.split_off(index + 1)
    }
}
