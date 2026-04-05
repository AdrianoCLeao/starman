use std::borrow::Cow;

use engine_core::{EngineError, Result};

pub(crate) const RGBA8_BYTES_PER_PIXEL: u32 = 4;

#[derive(Debug)]
pub(crate) struct PreparedTextureUpload<'a> {
    pub(crate) bytes_per_row: u32,
    pub(crate) rows_per_image: u32,
    pub(crate) data: Cow<'a, [u8]>,
}

pub(crate) fn prepare_rgba8_upload_data<'a>(
    width: u32,
    height: u32,
    pixels: &'a [u8],
) -> Result<PreparedTextureUpload<'a>> {
    let row_bytes = width
        .checked_mul(RGBA8_BYTES_PER_PIXEL)
        .ok_or_else(|| EngineError::Render("texture row byte size overflow".to_owned()))?;
    let height_usize = usize::try_from(height)
        .map_err(|_| EngineError::Render("texture height conversion overflow".to_owned()))?;
    let row_bytes_usize = usize::try_from(row_bytes)
        .map_err(|_| EngineError::Render("texture row bytes conversion overflow".to_owned()))?;
    let expected_size = row_bytes_usize
        .checked_mul(height_usize)
        .ok_or_else(|| EngineError::Render("texture upload size overflow".to_owned()))?;

    if pixels.len() < expected_size {
        return Err(EngineError::Render(format!(
            "texture payload too small: expected at least {expected_size} bytes, got {}",
            pixels.len()
        )));
    }

    let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let aligned_row_bytes = if row_bytes % alignment == 0 {
        row_bytes
    } else {
        row_bytes
            .checked_add(alignment - (row_bytes % alignment))
            .ok_or_else(|| {
                EngineError::Render("aligned texture row byte size overflow".to_owned())
            })?
    };

    if aligned_row_bytes == row_bytes {
        return Ok(PreparedTextureUpload {
            bytes_per_row: row_bytes,
            rows_per_image: height,
            data: Cow::Borrowed(&pixels[..expected_size]),
        });
    }

    let aligned_row_bytes_usize = usize::try_from(aligned_row_bytes).map_err(|_| {
        EngineError::Render("aligned texture row bytes conversion overflow".to_owned())
    })?;
    let mut padded_pixels = vec![0_u8; aligned_row_bytes_usize * height_usize];

    for row in 0..height_usize {
        let src_start = row * row_bytes_usize;
        let src_end = src_start + row_bytes_usize;
        let dst_start = row * aligned_row_bytes_usize;
        let dst_end = dst_start + row_bytes_usize;
        padded_pixels[dst_start..dst_end].copy_from_slice(&pixels[src_start..src_end]);
    }

    Ok(PreparedTextureUpload {
        bytes_per_row: aligned_row_bytes,
        rows_per_image: height,
        data: Cow::Owned(padded_pixels),
    })
}

pub(crate) fn upload_rgba8_texture(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
    pixels: &[u8],
) -> Result<()> {
    let prepared = prepare_rgba8_upload_data(width, height, pixels)?;

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        prepared.data.as_ref(),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(prepared.bytes_per_row),
            rows_per_image: Some(prepared.rows_per_image),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    Ok(())
}
