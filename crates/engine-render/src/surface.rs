use engine_core::{EngineError, Result};

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
        Err(wgpu::SurfaceError::Other) => Err(EngineError::Render(
            "surface acquire returned unknown error".to_owned(),
        )),
    }
}
