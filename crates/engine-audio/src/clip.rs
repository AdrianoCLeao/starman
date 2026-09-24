//! Audio clips: decoded sample frames (WAV/OGG/MP3) shared by every
//! backend, plus the source path for streamed playback.

use std::path::PathBuf;
use std::sync::Arc;

use engine_assets::{Asset, AssetLoader, LoadContext};
use engine_core::Result;
use kira::sound::static_sound::StaticSoundData;

/// A decoded clip: interleaved-free stereo frames at `sample_rate`.
#[derive(Clone, Debug)]
pub struct AudioClip {
    pub sample_rate: u32,
    /// Left/right sample pairs.
    pub frames: Arc<[kira::Frame]>,
    /// The file on disk, for streamed playback of long clips (music).
    pub path: Option<PathBuf>,
}

impl Asset for AudioClip {
    const TYPE_NAME: &'static str = "AudioClip";
}

impl AudioClip {
    /// A clip from raw frames (procedural audio, tests).
    pub fn from_frames(sample_rate: u32, frames: Vec<[f32; 2]>) -> Self {
        Self {
            sample_rate: sample_rate.max(1),
            frames: frames
                .into_iter()
                .map(|[left, right]| kira::Frame { left, right })
                .collect(),
            path: None,
        }
    }

    /// A silent clip of `seconds` (tests, placeholders).
    pub fn silence(sample_rate: u32, seconds: f32) -> Self {
        let count = (sample_rate as f32 * seconds.max(0.0)) as usize;
        Self::from_frames(sample_rate, vec![[0.0, 0.0]; count])
    }

    pub fn duration(&self) -> f32 {
        self.frames.len() as f32 / self.sample_rate.max(1) as f32
    }

    /// Kira sound data sharing these frames (no copy).
    pub fn sound_data(&self) -> StaticSoundData {
        StaticSoundData {
            sample_rate: self.sample_rate,
            frames: self.frames.clone(),
            settings: Default::default(),
            slice: None,
        }
    }
}

/// Decodes `wav`, `ogg` and `mp3` files.
pub struct AudioClipLoader;

impl AssetLoader for AudioClipLoader {
    type Asset = AudioClip;

    fn extensions(&self) -> &'static [&'static str] {
        &["wav", "ogg", "mp3"]
    }

    fn load(&self, bytes: &[u8], ctx: &mut LoadContext<'_>) -> Result<AudioClip> {
        let data = StaticSoundData::from_cursor(std::io::Cursor::new(bytes.to_vec()))
            .map_err(|error| ctx.error(format!("cannot decode audio: {error}")))?;
        Ok(AudioClip {
            sample_rate: data.sample_rate,
            frames: data.frames,
            path: Some(ctx.disk_path.to_path_buf()),
        })
    }
}

/// Encodes 16-bit PCM stereo WAV (content generation, tests).
pub fn encode_wav(sample_rate: u32, frames: &[[f32; 2]]) -> Vec<u8> {
    let data_len = (frames.len() * 4) as u32;
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&2u16.to_le_bytes()); // stereo
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 4).to_le_bytes());
    out.extend_from_slice(&4u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for frame in frames {
        for sample in frame {
            let value = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_round_trips_through_the_decoder() {
        let frames: Vec<[f32; 2]> = (0..4410)
            .map(|i| {
                let v = (i as f32 * 0.05).sin() * 0.5;
                [v, -v]
            })
            .collect();
        let bytes = encode_wav(44_100, &frames);
        let data = StaticSoundData::from_cursor(std::io::Cursor::new(bytes)).unwrap();
        assert_eq!(data.sample_rate, 44_100);
        assert_eq!(data.frames.len(), 4410);
        assert!((data.frames[100].left - frames[100][0]).abs() < 1e-3);
        let clip = AudioClip {
            sample_rate: data.sample_rate,
            frames: data.frames,
            path: None,
        };
        assert!((clip.duration() - 0.1).abs() < 1e-6);
    }
}
