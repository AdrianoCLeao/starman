//! Reflection probes: captures the lit scene into a cubemap from the
//! probe position (one probe per frame at most), prefilters it into a
//! slot of the probe cube array, and exposes up to four blended probes to
//! the lit shader with box projection.

use bevy_ecs::entity::Entity;
use engine_core::PersistentId;

use crate::layouts::{DEPTH_FORMAT, HDR_FORMAT};
use crate::passes::environment::{cube_texture, face_view, PREFILTER_MIPS};

pub const MAX_PROBES: usize = 4;
pub const PROBE_RESOLUTION: u32 = 128;

/// Stable identity of a probe across frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeKey {
    Persistent(PersistentId),
    Entity(Entity),
}

#[derive(Clone, Debug)]
pub struct ProbeSlot {
    pub key: ProbeKey,
    pub position: [f32; 3],
    pub half_extents: [f32; 3],
    pub blend_distance: f32,
    pub intensity: f32,
    pub priority: i32,
}

/// Slot policy: the slot already holding `key`, else a free slot, else
/// the lowest-priority slot when `priority` beats it.
pub fn choose_slot(slots: &[Option<ProbeSlot>], key: ProbeKey, priority: i32) -> Option<usize> {
    if let Some(index) = slots
        .iter()
        .position(|slot| slot.as_ref().is_some_and(|slot| slot.key == key))
    {
        return Some(index);
    }
    if let Some(index) = slots.iter().position(Option::is_none) {
        return Some(index);
    }
    let (index, lowest) = slots
        .iter()
        .enumerate()
        .filter_map(|(i, slot)| slot.as_ref().map(|slot| (i, slot.priority)))
        .min_by_key(|(_, priority)| *priority)?;
    (priority > lowest).then_some(index)
}

pub struct ProbeResources {
    pub array: wgpu::Texture,
    pub array_view: wgpu::TextureView,
    pub capture: wgpu::Texture,
    pub capture_view: wgpu::TextureView,
    pub capture_depth: wgpu::TextureView,
    pub slots: [Option<ProbeSlot>; MAX_PROBES],
    pub bakes: u64,
}

impl ProbeResources {
    pub fn new(device: &wgpu::Device) -> Self {
        let array = cube_texture(
            device,
            "engine-render-probe-array",
            PROBE_RESOLUTION,
            PREFILTER_MIPS,
            MAX_PROBES as u32,
        );
        let array_view = array.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::CubeArray),
            ..Default::default()
        });
        let capture = cube_texture(
            device,
            "engine-render-probe-capture",
            PROBE_RESOLUTION,
            crate::gpu::assets::mip_count(PROBE_RESOLUTION, PROBE_RESOLUTION),
            1,
        );
        let capture_view = crate::passes::environment::cube_view(&capture);
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("engine-render-probe-depth"),
            size: wgpu::Extent3d {
                width: PROBE_RESOLUTION,
                height: PROBE_RESOLUTION,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        Self {
            array,
            array_view,
            capture,
            capture_view,
            capture_depth: depth.create_view(&Default::default()),
            slots: Default::default(),
            bakes: 0,
        }
    }

    /// Slot holding `key`, or a free/lowest-priority slot to reuse.
    pub fn slot_for(&self, key: ProbeKey, priority: i32) -> Option<usize> {
        choose_slot(&self.slots, key, priority)
    }

    /// Drops slots whose probe no longer exists.
    pub fn retain(&mut self, alive: &[ProbeKey]) {
        for slot in &mut self.slots {
            if slot.as_ref().is_some_and(|slot| !alive.contains(&slot.key)) {
                *slot = None;
            }
        }
    }

    pub fn capture_face_view(&self, face: u32) -> wgpu::TextureView {
        face_view(&self.capture, face, 0)
    }

    pub fn active(&self) -> impl Iterator<Item = (usize, &ProbeSlot)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| slot.as_ref().map(|slot| (i, slot)))
    }

    pub fn capture_format() -> wgpu::TextureFormat {
        HDR_FORMAT
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slot(key: u32, priority: i32) -> ProbeSlot {
        ProbeSlot {
            key: ProbeKey::Entity(Entity::from_raw(key)),
            position: [0.0; 3],
            half_extents: [1.0; 3],
            blend_distance: 1.0,
            intensity: 1.0,
            priority,
        }
    }

    #[test]
    fn slots_are_reused_by_key_then_free_then_by_priority() {
        let mut slots: [Option<ProbeSlot>; MAX_PROBES] =
            [Some(slot(1, 5)), None, Some(slot(3, 3)), Some(slot(4, 2))];
        let key = |k| ProbeKey::Entity(Entity::from_raw(k));
        assert_eq!(choose_slot(&slots, key(3), 0), Some(2));
        assert_eq!(choose_slot(&slots, key(9), 0), Some(1));
        slots[1] = Some(slot(2, 1));
        assert_eq!(choose_slot(&slots, key(9), 4), Some(1));
        assert_eq!(choose_slot(&slots, key(9), 0), None);
    }
}
