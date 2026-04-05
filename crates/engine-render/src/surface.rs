use engine_core::{EngineError, Result};

pub(crate) fn create_depth_target(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> crate::DepthTarget {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("engine-render-depth-texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    crate::DepthTarget {
        _texture: texture,
        view,
    }
}

pub(crate) fn choose_present_mode(
    vsync: bool,
    supported_modes: &[wgpu::PresentMode],
) -> wgpu::PresentMode {
    if vsync {
        return wgpu::PresentMode::Fifo;
    }

    for preferred in [
        wgpu::PresentMode::Immediate,
        wgpu::PresentMode::Mailbox,
        wgpu::PresentMode::FifoRelaxed,
        wgpu::PresentMode::Fifo,
    ] {
        if supported_modes.contains(&preferred) {
            return preferred;
        }
    }

    wgpu::PresentMode::Fifo
}

pub(crate) fn acquire_frame(
    surface: &wgpu::Surface<'_>,
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> Result<Option<wgpu::SurfaceTexture>> {
    match surface.get_current_texture() {
        Ok(frame) => Ok(Some(frame)),
        Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
            log::warn!(
                target: "engine::render",
                "Surface outdated/lost; reconfiguring swapchain"
            );
            surface.configure(device, config);
            Ok(None)
        }
        Err(wgpu::SurfaceError::Timeout) => {
            log::warn!(target: "engine::render", "Surface acquire timeout");
            Ok(None)
        }
        Err(wgpu::SurfaceError::OutOfMemory) => Err(EngineError::Render(
            "surface out of memory while acquiring frame".to_owned(),
        )),
        Err(wgpu::SurfaceError::Other) => {
            log::warn!(target: "engine::render", "Surface acquire returned unknown error");
            Ok(None)
        }
    }
}
