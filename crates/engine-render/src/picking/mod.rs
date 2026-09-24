//! GPU entity picking: on request, mesh pick ids are rendered into an
//! `R32Uint` target, the texel under the cursor is copied to a readback
//! buffer and mapped asynchronously; the result arrives a frame or two
//! later through [`PickingState::poll`].

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

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

const MAP_PENDING: u8 = 0;
const MAP_READY: u8 = 1;
const MAP_FAILED: u8 = 2;

struct InFlight {
    request: PickRequest,
    id_map: Vec<(u32, Entity)>,
    status: Arc<AtomicU8>,
}

#[derive(Default)]
pub struct PickingState {
    pub pending: Option<PickRequest>,
    pub completed: Option<PickResult>,
    /// Map packed pick_id → Entity for the frame the ID buffer was drawn.
    pub id_map: Vec<(u32, Entity)>,
    readback: Option<wgpu::Buffer>,
    in_flight: Option<InFlight>,
}

impl std::fmt::Debug for PickingState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PickingState")
            .field("pending", &self.pending)
            .field("completed", &self.completed)
            .field("in_flight", &self.in_flight.is_some())
            .finish()
    }
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

    /// Whether this frame must render the id pass.
    pub fn needs_id_pass(&self) -> bool {
        self.pending.is_some() && self.in_flight.is_none()
    }

    /// Resolves a pick without the GPU (tests).
    #[cfg(test)]
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

    pub fn readback_buffer(&mut self, device: &wgpu::Device) -> &wgpu::Buffer {
        self.readback.get_or_insert_with(|| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("engine-render-pick-readback"),
                size: 256,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        })
    }

    /// Records the texel copy of `(x, y)` from `ids` into the readback
    /// buffer. Call after the id pass, before submission.
    pub fn encode_copy(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        ids: &wgpu::Texture,
    ) {
        let Some(request) = self.pending else {
            return;
        };
        let x = request.x.min(ids.width().saturating_sub(1));
        let y = request.y.min(ids.height().saturating_sub(1));
        let buffer = self.readback_buffer(device).clone();
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: ids,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        self.in_flight = Some(InFlight {
            request,
            id_map: std::mem::take(&mut self.id_map),
            status: Arc::new(AtomicU8::new(MAP_PENDING)),
        });
        self.pending = None;
    }

    /// Starts mapping the readback after the copy was submitted.
    pub fn after_submit(&mut self) {
        let (Some(in_flight), Some(buffer)) = (&self.in_flight, &self.readback) else {
            return;
        };
        if in_flight.status.load(Ordering::Acquire) != MAP_PENDING {
            return;
        }
        let status = Arc::clone(&in_flight.status);
        buffer
            .slice(..4)
            .map_async(wgpu::MapMode::Read, move |result| {
                status.store(
                    if result.is_ok() {
                        MAP_READY
                    } else {
                        MAP_FAILED
                    },
                    Ordering::Release,
                );
            });
    }

    /// Advances the readback without blocking; finished picks move to
    /// `completed`.
    pub fn update(&mut self, device: &wgpu::Device) {
        let Some(in_flight) = &self.in_flight else {
            return;
        };
        let _ = device.poll(wgpu::Maintain::Poll);
        let status = in_flight.status.load(Ordering::Acquire);
        if status == MAP_PENDING {
            return;
        }
        let in_flight = self.in_flight.take().expect("checked above");
        let mut pick_id = 0;
        if status == MAP_READY {
            if let Some(buffer) = &self.readback {
                {
                    let data = buffer.slice(..4).get_mapped_range();
                    pick_id = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                }
                buffer.unmap();
            }
        }
        let entity = (pick_id != 0)
            .then(|| {
                in_flight
                    .id_map
                    .iter()
                    .find(|(id, _)| *id == pick_id)
                    .map(|(_, entity)| *entity)
            })
            .flatten();
        self.completed = Some(PickResult {
            x: in_flight.request.x,
            y: in_flight.request.y,
            entity,
            pick_id,
        });
    }

    pub fn poll(&mut self) -> Option<PickResult> {
        self.completed.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_fallback_resolves_through_the_id_map() {
        let mut state = PickingState::default();
        state.request(3, 4, 1);
        assert!(state.needs_id_pass());
        state.id_map = vec![(42, Entity::from_raw(7))];
        state.resolve_cpu_fallback(42);
        let result = state.poll().unwrap();
        assert_eq!(result.entity, Some(Entity::from_raw(7)));
        assert_eq!((result.x, result.y), (3, 4));
        assert!(state.poll().is_none());
    }
}
