//! Generational GPU resource arena with staging uploads and deferred destroy.

pub mod assets;

use std::collections::VecDeque;
use std::marker::PhantomData;

/// Opaque generational handle into a typed GPU arena slot.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct GpuHandle<T> {
    pub index: u32,
    pub generation: u32,
    _marker: PhantomData<T>,
}

impl<T> Clone for GpuHandle<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for GpuHandle<T> {}

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

/// Per-frame CPU→GPU uploads through a recycled ring of mapped staging
/// chunks (`wgpu::util::StagingBelt`): writes are recorded into the frame's
/// command encoder instead of allocating inside `Queue::write_buffer`.
pub struct StagingBelt {
    belt: wgpu::util::StagingBelt,
    bytes_this_frame: u64,
}

impl StagingBelt {
    pub fn new(chunk_size: u64) -> Self {
        Self {
            belt: wgpu::util::StagingBelt::new(chunk_size.max(64 * 1024)),
            bytes_this_frame: 0,
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
        let Some(size) = std::num::NonZeroU64::new(data.len() as u64) else {
            return;
        };
        self.belt
            .write_buffer(encoder, target, target_offset, size, device)
            .copy_from_slice(data);
        self.bytes_this_frame += size.get();
    }

    /// Unmaps chunks used this frame; call before submitting the encoder.
    pub fn finish(&mut self) {
        self.belt.finish();
    }

    /// Returns chunks whose GPU copies completed; call after submission.
    pub fn recall(&mut self) {
        self.belt.recall();
        self.bytes_this_frame = 0;
    }

    pub fn bytes_this_frame(&self) -> u64 {
        self.bytes_this_frame
    }
}

/// A GPU buffer that grows (power of two) to fit what is written into it.
/// `generation` changes whenever the buffer is recreated so dependent bind
/// groups know to rebuild.
pub struct GrowableBuffer {
    label: &'static str,
    usage: wgpu::BufferUsages,
    buffer: wgpu::Buffer,
    capacity: u64,
    generation: u64,
}

impl GrowableBuffer {
    pub fn new(
        device: &wgpu::Device,
        label: &'static str,
        usage: wgpu::BufferUsages,
        initial: u64,
    ) -> Self {
        let capacity = initial.max(256).next_power_of_two();
        let usage = usage | wgpu::BufferUsages::COPY_DST;
        Self {
            label,
            usage,
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: capacity,
                usage,
                mapped_at_creation: false,
            }),
            capacity,
            generation: 1,
        }
    }

    /// Ensures capacity for `size` bytes; returns `true` when recreated.
    pub fn reserve(&mut self, device: &wgpu::Device, size: u64) -> bool {
        if size <= self.capacity {
            return false;
        }
        self.capacity = size.next_power_of_two();
        self.buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(self.label),
            size: self.capacity,
            usage: self.usage,
            mapped_at_creation: false,
        });
        self.generation += 1;
        true
    }

    /// Writes `data` at offset 0 through the staging belt (growing first).
    pub fn upload(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        belt: &mut StagingBelt,
        data: &[u8],
    ) -> bool {
        let recreated = self.reserve(device, data.len() as u64);
        belt.write_buffer(device, encoder, &self.buffer, 0, data);
        recreated
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn capacity(&self) -> u64 {
        self.capacity
    }
}

/// Packs many small uniform/storage records into one buffer at
/// `alignment`-aligned offsets for dynamic-offset binding.
pub struct AlignedWriter {
    alignment: u64,
    data: Vec<u8>,
}

impl AlignedWriter {
    pub fn new(alignment: u64) -> Self {
        Self {
            alignment: alignment.max(4),
            data: Vec::new(),
        }
    }

    /// Appends `value`, returning its byte offset.
    pub fn push<T: bytemuck::Pod>(&mut self, value: &T) -> u32 {
        self.push_bytes(bytemuck::bytes_of(value))
    }

    pub fn push_bytes(&mut self, bytes: &[u8]) -> u32 {
        let offset = align_up(self.data.len() as u64, self.alignment);
        self.data.resize(offset as usize, 0);
        self.data.extend_from_slice(bytes);
        offset as u32
    }

    /// Appends a slice of `u32`s, returning its byte offset.
    pub fn push_slice<T: bytemuck::Pod>(&mut self, values: &[T]) -> u32 {
        self.push_bytes(bytemuck::cast_slice(values))
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }

    pub fn bytes(&self) -> &[u8] {
        &self.data
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

pub fn align_up(value: u64, alignment: u64) -> u64 {
    value.div_ceil(alignment) * alignment
}

/// Transient render targets shared across frames, keyed by descriptor.
/// Nodes ask for "a texture like this" each frame; matching textures are
/// reused, unused ones are released after a few frames.
#[derive(Default)]
pub struct TransientPool {
    entries: Vec<TransientEntry>,
    frame: u64,
}

struct TransientEntry {
    key: TransientKey,
    texture: wgpu::Texture,
    last_used: u64,
    in_use: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TransientKey {
    pub width: u32,
    pub height: u32,
    pub format: wgpu::TextureFormat,
    pub mip_levels: u32,
    pub layers: u32,
    pub samples: u32,
    pub usage: wgpu::TextureUsages,
}

impl TransientPool {
    pub fn begin_frame(&mut self) {
        self.frame += 1;
        let frame = self.frame;
        self.entries.retain(|entry| frame - entry.last_used <= 3);
        for entry in &mut self.entries {
            entry.in_use = false;
        }
    }

    /// A texture matching `key` not yet handed out this frame.
    pub fn acquire(
        &mut self,
        device: &wgpu::Device,
        label: &str,
        key: TransientKey,
    ) -> wgpu::Texture {
        if let Some(entry) = self
            .entries
            .iter_mut()
            .find(|entry| !entry.in_use && entry.key == key)
        {
            entry.in_use = true;
            entry.last_used = self.frame;
            return entry.texture.clone();
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: key.width.max(1),
                height: key.height.max(1),
                depth_or_array_layers: key.layers.max(1),
            },
            mip_level_count: key.mip_levels.max(1),
            sample_count: key.samples.max(1),
            dimension: wgpu::TextureDimension::D2,
            format: key.format,
            usage: key.usage,
            view_formats: &[],
        });
        self.entries.push(TransientEntry {
            key,
            texture: texture.clone(),
            last_used: self.frame,
            in_use: true,
        });
        texture
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_writer_aligns_every_record() {
        let mut writer = AlignedWriter::new(256);
        assert_eq!(writer.push(&[1u32; 4]), 0);
        assert_eq!(writer.push(&[2u32; 4]), 256);
        assert_eq!(writer.push_slice(&[3u32; 100]), 512);
        assert_eq!(writer.bytes().len(), 512 + 400);
        assert_eq!(align_up(257, 256), 512);
    }

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
