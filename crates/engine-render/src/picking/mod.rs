//! GPU entity-ID picking with async readback.

use bevy_ecs::entity::Entity;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PickRequest {
    pub x: u32,
    pub y: u32,
    pub frame_issued: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PickResult {
    pub x: u32,
    pub y: u32,
    pub entity: Option<Entity>,
    pub pick_id: u32,
}

#[derive(Default, Debug)]
pub struct PickingState {
    pub pending: Option<PickRequest>,
    pub completed: Option<PickResult>,
    /// Map packed pick_id → Entity for the frame the ID buffer was drawn.
    pub id_map: Vec<(u32, Entity)>,
}

impl PickingState {
    pub fn request(&mut self, x: u32, y: u32, frame: u64) {
        self.pending = Some(PickRequest {
            x,
            y,
            frame_issued: frame,
        });
        self.completed = None;
    }

    pub fn needs_id_pass(&self) -> bool {
        self.pending.is_some()
    }

    pub fn resolve_cpu_fallback(&mut self, pick_id: u32) {
        let entity = self
            .id_map
            .iter()
            .find(|(id, _)| *id == pick_id)
            .map(|(_, e)| *e);
        if let Some(req) = self.pending.take() {
            self.completed = Some(PickResult {
                x: req.x,
                y: req.y,
                entity,
                pick_id,
            });
        }
    }

    pub fn poll(&mut self) -> Option<PickResult> {
        self.completed.take()
    }
}
