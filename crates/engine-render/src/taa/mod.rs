#![allow(dead_code)]

//! Temporal anti-aliasing: Halton jitter and history helpers.

pub struct HaltonSequence {
    index: u32,
    base_x: u32,
    base_y: u32,
}

impl HaltonSequence {
    pub fn new() -> Self {
        Self {
            index: 0,
            base_x: 2,
            base_y: 3,
        }
    }

    pub fn next_jitter(&mut self, width: u32, height: u32) -> [f32; 2] {
        self.index = self.index.wrapping_add(1);
        let x = (halton(self.index, self.base_x) * 2.0 - 1.0) / width.max(1) as f32;
        let y = (halton(self.index, self.base_y) * 2.0 - 1.0) / height.max(1) as f32;
        [x, y]
    }

    /// Deterministic jitter for smoke/CI (fixed index).
    pub fn jitter_at(index: u32, width: u32, height: u32) -> [f32; 2] {
        let x = (halton(index.max(1), 2) * 2.0 - 1.0) / width.max(1) as f32;
        let y = (halton(index.max(1), 3) * 2.0 - 1.0) / height.max(1) as f32;
        [x, y]
    }
}

impl Default for HaltonSequence {
    fn default() -> Self {
        Self::new()
    }
}

fn halton(index: u32, base: u32) -> f32 {
    let mut f = 1.0;
    let mut r = 0.0;
    let mut i = index;
    while i > 0 {
        f /= base as f32;
        r += f * (i % base) as f32;
        i /= base;
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jitter_is_subpixel() {
        let j = HaltonSequence::jitter_at(1, 128, 128);
        assert!(j[0].abs() < 1.0 / 64.0);
        assert!(j[1].abs() < 1.0 / 64.0);
    }
}
