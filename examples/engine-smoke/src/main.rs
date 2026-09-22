use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Instant;

use serde::Serialize;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SmokeBackend {
    Dx12,
    Vulkan,
    Metal,
}

impl SmokeBackend {
    fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "dx12" | "d3d12" => Some(Self::Dx12),
            "vulkan" | "vk" => Some(Self::Vulkan),
            "metal" => Some(Self::Metal),
            _ => None,
        }
    }

    fn backends(self) -> wgpu::Backends {
        match self {
            Self::Dx12 => wgpu::Backends::DX12,
            Self::Vulkan => wgpu::Backends::VULKAN,
            Self::Metal => wgpu::Backends::METAL,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Dx12 => "dx12",
            Self::Vulkan => "vulkan",
            Self::Metal => "metal",
        }
    }
}

#[derive(Debug)]
struct Args {
    backend: SmokeBackend,
    report: PathBuf,
    fallback_adapter: bool,
}

#[derive(Serialize)]
struct SmokeReport {
    schema: u32,
    backend: String,
    adapter: Option<AdapterReport>,
    checks: Vec<CheckReport>,
    success: bool,
    duration_millis: u128,
}

#[derive(Serialize)]
struct AdapterReport {
    name: String,
    backend: String,
    device_type: String,
}

#[derive(Serialize)]
struct CheckReport {
    name: &'static str,
    success: bool,
    detail: String,
}

fn main() {
    let args = match parse_args() {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(1);
        }
    };

    let _diagnostics = engine_diagnostics::initialize(
        engine_diagnostics::DiagnosticsConfig::for_application("engine-smoke").json_stdout(false),
    );

    let start = Instant::now();
    let mut report = SmokeReport {
        schema: 1,
        backend: args.backend.as_str().to_owned(),
        adapter: None,
        checks: Vec::new(),
        success: false,
        duration_millis: 0,
    };

    let exit_code =
        match pollster::block_on(run_smoke(args.backend, args.fallback_adapter, &mut report)) {
            Ok(()) => {
                report.success = true;
                0
            }
            Err(SmokeError::NoAdapter(message)) => {
                report.checks.push(check("adapter", false, message));
                2
            }
            Err(SmokeError::Gpu(message)) => {
                report.checks.push(check("gpu", false, message));
                3
            }
            Err(SmokeError::Integration(message)) => {
                report.checks.push(check("integration", false, message));
                4
            }
        };

    report.duration_millis = start.elapsed().as_millis();

    if let Some(parent) = args.report.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    if let Err(error) = std::fs::write(
        &args.report,
        serde_json::to_vec_pretty(&report).expect("smoke report should serialize"),
    ) {
        eprintln!(
            "failed to write smoke report '{}': {error}",
            args.report.display()
        );
        std::process::exit(1);
    }

    std::process::exit(exit_code);
}

async fn run_smoke(
    backend: SmokeBackend,
    fallback_adapter: bool,
    report: &mut SmokeReport,
) -> Result<(), SmokeError> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: backend.backends(),
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: fallback_adapter,
        })
        .await
        .ok_or_else(|| SmokeError::NoAdapter(format!("no {:?} adapter found", backend)))?;

    let adapter_info = adapter.get_info();
    report.adapter = Some(AdapterReport {
        name: adapter_info.name.clone(),
        backend: format!("{:?}", adapter_info.backend),
        device_type: format!("{:?}", adapter_info.device_type),
    });
    report
        .checks
        .push(check("adapter", true, adapter_info.name));

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("engine-smoke-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await
        .map_err(|error| SmokeError::Gpu(format!("request_device failed: {error}")))?;
    report.checks.push(check("device", true, "device created"));

    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("engine-smoke-offscreen"),
        size: wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("engine-smoke-encoder"),
    });
    {
        let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("engine-smoke-clear-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.2,
                        g: 0.4,
                        b: 0.6,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
    }
    queue.submit(Some(encoder.finish()));

    match device.pop_error_scope().await {
        Some(error) => return Err(SmokeError::Gpu(format!("validation error: {error}"))),
        None => report
            .checks
            .push(check("offscreen-render", true, "64x64 clear pass")),
    }

    validate_readback(&device, &queue, &texture, report).await?;

    let mut world = engine_core::create_world();
    world.insert_resource(engine_input::InputState::default());
    report
        .checks
        .push(check("world-input", true, "input resource inserted"));

    let _physics = engine_physics::PhysicsWorld3D::default();
    report
        .checks
        .push(check("physics", true, "physics world constructed"));

    let mut assets = engine_assets::AssetServer::new("examples/reference-project/assets");
    let _ = assets
        .load_texture_handle("textures/placeholder.png")
        .map_err(|error| SmokeError::Integration(format!("texture load failed: {error}")))?;
    report
        .checks
        .push(check("assets", true, "placeholder texture loaded"));

    let mut render_module = engine_render::RenderModule::new();
    render_module
        .tick(&mut world, &assets)
        .map_err(|error| SmokeError::Integration(format!("render tick failed: {error}")))?;
    report.checks.push(check(
        "render-module",
        true,
        "headless tick skipped cleanly",
    ));

    Ok(())
}

async fn validate_readback(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    report: &mut SmokeReport,
) -> Result<(), SmokeError> {
    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 64;
    const BYTES_PER_PIXEL: u32 = 4;
    const BYTES_PER_ROW: u32 = WIDTH * BYTES_PER_PIXEL;
    const BUFFER_SIZE: u64 = (BYTES_PER_ROW * HEIGHT) as u64;

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("engine-smoke-readback"),
        size: BUFFER_SIZE,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("engine-smoke-readback-encoder"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(BYTES_PER_ROW),
                rows_per_image: Some(HEIGHT),
            },
        },
        wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    let slice = buffer.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|error| SmokeError::Gpu(format!("readback callback failed: {error}")))?
        .map_err(|error| SmokeError::Gpu(format!("readback map failed: {error}")))?;

    {
        let data = slice.get_mapped_range();
        let corner = &data[0..4];
        let center_index =
            (((HEIGHT / 2) * BYTES_PER_ROW) + ((WIDTH / 2) * BYTES_PER_PIXEL)) as usize;
        let center = &data[center_index..center_index + 4];

        if corner != center {
            return Err(SmokeError::Integration(format!(
                "clear readback mismatch: corner={corner:?} center={center:?}"
            )));
        }

        if center[2] <= center[0] || center[3] == 0 {
            return Err(SmokeError::Integration(format!(
                "unexpected clear color readback: center={center:?}"
            )));
        }
    }

    buffer.unmap();
    report.checks.push(check(
        "offscreen-readback",
        true,
        "center and corner pixels match clear color",
    ));
    Ok(())
}

fn parse_args() -> Result<Args, String> {
    let mut backend = None;
    let mut report = None;
    let mut fallback_adapter = false;
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--backend" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--backend requires a value".to_owned())?;
                backend = SmokeBackend::parse(&value);
                if backend.is_none() {
                    return Err(format!("unsupported backend '{value}'"));
                }
            }
            "--report" => {
                report = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--report requires a path".to_owned())?,
                ));
            }
            "--fallback" => {
                fallback_adapter = true;
            }
            "-h" | "--help" => return Err(usage()),
            other => return Err(format!("unexpected argument '{other}'\n{}", usage())),
        }
    }

    Ok(Args {
        backend: backend.ok_or_else(usage)?,
        report: report.unwrap_or_else(|| PathBuf::from("target/engine-smoke-report.json")),
        fallback_adapter,
    })
}

fn usage() -> String {
    "usage: engine-smoke --backend <dx12|vulkan|metal> [--fallback] [--report <path>]".to_owned()
}

fn check(name: &'static str, success: bool, detail: impl Into<String>) -> CheckReport {
    CheckReport {
        name,
        success,
        detail: detail.into(),
    }
}

#[derive(Debug)]
enum SmokeError {
    NoAdapter(String),
    Gpu(String),
    Integration(String),
}
