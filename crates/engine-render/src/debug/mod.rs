//! Debug views and frame buffer dumps.

use std::path::Path;

use engine_core::{EngineError, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum DebugView {
    #[default]
    None,
    Depth,
    Normals,
    Clusters,
    Overdraw,
    LightHeat,
    Ssao,
    Velocity,
}

impl DebugView {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Depth => "depth",
            Self::Normals => "normals",
            Self::Clusters => "clusters",
            Self::Overdraw => "overdraw",
            Self::LightHeat => "light_heat",
            Self::Ssao => "ssao",
            Self::Velocity => "velocity",
        }
    }

    pub fn all() -> [Self; 8] {
        [
            Self::None,
            Self::Depth,
            Self::Normals,
            Self::Clusters,
            Self::Overdraw,
            Self::LightHeat,
            Self::Ssao,
            Self::Velocity,
        ]
    }

    pub fn enables_debug_blit(self) -> bool {
        !matches!(self, Self::None)
    }
}

/// Write a raw RGBA8 buffer as a simple PPM (no extra image crate dep required).
pub fn dump_rgba8_ppm(path: &Path, width: u32, height: u32, pixels: &[u8]) -> Result<()> {
    if pixels.len() < (width as usize) * (height as usize) * 4 {
        return Err(EngineError::Render(
            "debug dump pixel buffer too small".to_owned(),
        ));
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| EngineError::Render(e.to_string()))?;
    }
    let mut body = format!("P6\n{width} {height}\n255\n").into_bytes();
    for chunk in pixels.chunks_exact(4) {
        body.extend_from_slice(&chunk[..3]);
    }
    std::fs::write(path, body).map_err(|e| EngineError::Render(e.to_string()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn writes_ppm() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("starman-debug-{nanos}.ppm"));
        dump_rgba8_ppm(
            &path,
            2,
            2,
            &[255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 0, 0, 0, 255],
        )
        .unwrap();
        assert!(path.is_file());
        let _ = std::fs::remove_file(&path);
    }
}
