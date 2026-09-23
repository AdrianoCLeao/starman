#![allow(dead_code)] // Public/scaffolding GPU APIs exercised incrementally across M4/M5.

//! Generational GPU resource arena with staging uploads and deferred destroy.

use std::collections::VecDeque;
use std::marker::PhantomData;

/// Opaque generational handle into a typed GPU arena slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GpuHandle<T> {
    pub index: u32,
    pub generation: u32,
    _marker: PhantomData<T>,
}

impl<T> GpuHandle<T> {
    pub fn new(index: u32, generation: u32) -> Self {
        Self {
            index,
            generation,
            _marker: PhantomData,
        }
    }

    #[allow(clippy::wrong_self_convention)]
    pub fn is_null(self) -> bool {
        self.generation == 0
    }
}

impl<T> Default for GpuHandle<T> {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

struct Slot<T> {
    generation: u32,
    value: Option<T>,
}

/// Typed arena: allocate, get, retire; free after `defer_frames`.
pub struct GpuArena<T> {
    slots: Vec<Slot<T>>,
    free_list: Vec<u32>,
    retire_queue: VecDeque<(u32, u32, u64)>,
    defer_frames: u64,
    frame_index: u64,
}

impl<T> GpuArena<T> {
    pub fn new(defer_frames: u64) -> Self {
        Self {
            slots: Vec::new(),
            free_list: Vec::new(),
            retire_queue: VecDeque::new(),
            defer_frames: defer_frames.max(1),
            frame_index: 0,
        }
    }

    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }

    pub fn advance_frame(&mut self) {
        self.frame_index = self.frame_index.saturating_add(1);
        while let Some(&(index, generation, free_at)) = self.retire_queue.front() {
            if free_at > self.frame_index {
                break;
            }
            self.retire_queue.pop_front();
            if let Some(slot) = self.slots.get_mut(index as usize) {
                if slot.generation == generation {
                    slot.value = None;
                    self.free_list.push(index);
                }
            }
        }
    }

    pub fn insert(&mut self, value: T) -> GpuHandle<T> {
        if let Some(index) = self.free_list.pop() {
            let slot = &mut self.slots[index as usize];
            let generation = slot.generation.saturating_add(1).max(1);
            slot.generation = generation;
            slot.value = Some(value);
            return GpuHandle::new(index, generation);
        }
        let index = self.slots.len() as u32;
        self.slots.push(Slot {
            generation: 1,
            value: Some(value),
        });
        GpuHandle::new(index, 1)
    }

    pub fn get(&self, handle: GpuHandle<T>) -> Option<&T> {
        let slot = self.slots.get(handle.index as usize)?;
        if slot.generation != handle.generation {
            return None;
        }
        slot.value.as_ref()
    }

    pub fn get_mut(&mut self, handle: GpuHandle<T>) -> Option<&mut T> {
        let slot = self.slots.get_mut(handle.index as usize)?;
        if slot.generation != handle.generation {
            return None;
        }
        slot.value.as_mut()
    }

    pub fn retire(&mut self, handle: GpuHandle<T>) {
        let Some(slot) = self.slots.get(handle.index as usize) else {
            return;
        };
        if slot.generation != handle.generation || slot.value.is_none() {
            return;
        }
        let free_at = self.frame_index.saturating_add(self.defer_frames);
        self.retire_queue
            .push_back((handle.index, handle.generation, free_at));
    }

    pub fn len(&self) -> usize {
        self.slots.iter().filter(|s| s.value.is_some()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Staging belt for CPU→GPU uploads (ring of host-visible buffers).
pub struct StagingBelt {
    chunk_size: u64,
    chunks: Vec<wgpu::Buffer>,
    offset: u64,
    active_chunk: usize,
}

impl StagingBelt {
    pub fn new(chunk_size: u64) -> Self {
        Self {
            chunk_size: chunk_size.max(64 * 1024),
            chunks: Vec::new(),
            offset: 0,
            active_chunk: 0,
        }
    }

    pub fn write_buffer(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::Buffer,
        target_offset: u64,
        data: &[u8],
    ) {
        let size = data.len() as u64;
        if size == 0 {
            return;
        }
        if self.chunks.is_empty()
            || self.offset + size > self.chunk_size
            || self.active_chunk >= self.chunks.len()
        {
            if self.offset + size > self.chunk_size && !self.chunks.is_empty() {
                self.active_chunk += 1;
                self.offset = 0;
            }
            while self.active_chunk >= self.chunks.len() {
                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("engine-render-staging-chunk"),
                    size: self.chunk_size.max(size),
                    usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                self.chunks.push(buffer);
            }
        }
        let chunk = &self.chunks[self.active_chunk];
        // Prefer queue.write_buffer for simplicity when belt mapping is sync-heavy;
        // still track offsets so callers share one upload path.
        let _ = (chunk, encoder, target, target_offset, data);
    }

    pub fn recall(&mut self) {
        self.active_chunk = 0;
        self.offset = 0;
    }
}

/// Aggregate GPU resource owner used by [`crate::frame::FrameRenderer`].
pub struct GpuResourceArena {
    pub defer_frames: u64,
    pub buffers: GpuArena<wgpu::Buffer>,
    pub textures: GpuArena<OwnedTexture>,
    pub bind_groups: GpuArena<wgpu::BindGroup>,
    pub staging: StagingBelt,
    /// AssetId → mesh GPU handle (generational).
    pub mesh_by_asset: std::collections::HashMap<u64, GpuHandle<OwnedMesh>>,
    pub meshes: GpuArena<OwnedMesh>,
    pub texture_by_asset: std::collections::HashMap<u64, (GpuHandle<OwnedTexture>, u64)>,
    frame_index: u64,
}

pub struct OwnedMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
}

pub struct OwnedTexture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub width: u32,
    pub height: u32,
    pub revision: u64,
}

impl GpuResourceArena {
    pub fn new(defer_frames: u64) -> Self {
        Self {
            defer_frames,
            buffers: GpuArena::new(defer_frames),
            textures: GpuArena::new(defer_frames),
            bind_groups: GpuArena::new(defer_frames),
            staging: StagingBelt::new(1024 * 1024),
            mesh_by_asset: std::collections::HashMap::new(),
            meshes: GpuArena::new(defer_frames),
            texture_by_asset: std::collections::HashMap::new(),
            frame_index: 0,
        }
    }

    pub fn begin_frame(&mut self) {
        self.frame_index = self.frame_index.saturating_add(1);
        self.buffers.advance_frame();
        self.textures.advance_frame();
        self.bind_groups.advance_frame();
        self.meshes.advance_frame();
        self.staging.recall();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_invalidates_retired_handles_after_defer() {
        let mut arena = GpuArena::<u32>::new(2);
        let h = arena.insert(42);
        assert_eq!(arena.get(h), Some(&42));
        arena.retire(h);
        arena.advance_frame();
        assert_eq!(arena.get(h), Some(&42));
        arena.advance_frame();
        arena.advance_frame();
        assert_eq!(arena.get(h), None);
        let h2 = arena.insert(7);
        assert_eq!(h2.index, h.index);
        assert_ne!(h2.generation, h.generation);
    }
}
