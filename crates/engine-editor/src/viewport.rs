use bevy_ecs::world::World;
use eframe::{egui, egui_wgpu, wgpu};
use engine_assets::AssetServer;
use engine_math::glam::{EulerRot, Quat, Vec3};
use engine_math::Mat4;
use engine_render::ViewportRenderModule;

const MAX_CAMERA_PITCH_RADIANS: f32 = 89.0_f32.to_radians();
const MIN_CAMERA_DISTANCE: f32 = 0.25;
const MAX_CAMERA_DISTANCE: f32 = 10_000.0;

#[derive(Clone, Debug)]
pub struct EditorCamera {
    pub target: Vec3,
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y_radians: f32,
    pub near: f32,
    pub far: f32,
    orbit_sensitivity: f32,
    pan_sensitivity: f32,
    zoom_sensitivity: f32,
}

impl Default for EditorCamera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            distance: 12.0,
            yaw: 0.0,
            pitch: -20.0_f32.to_radians(),
            fov_y_radians: std::f32::consts::FRAC_PI_4,
            near: 0.1,
            far: 1000.0,
            orbit_sensitivity: 0.01,
            pan_sensitivity: 1.0,
            zoom_sensitivity: 1.0,
        }
    }
}

impl EditorCamera {
    pub fn restore_orbit(&mut self, target: Vec3, distance: f32, yaw: f32, pitch: f32) {
        self.target = target;
        self.distance = distance.clamp(MIN_CAMERA_DISTANCE, MAX_CAMERA_DISTANCE);
        self.yaw = yaw;
        self.pitch = pitch.clamp(-MAX_CAMERA_PITCH_RADIANS, MAX_CAMERA_PITCH_RADIANS);
    }

    pub fn rotation(&self) -> Quat {
        Quat::from_euler(EulerRot::YXZ, self.yaw, self.pitch, 0.0)
    }

    pub fn eye_position(&self) -> Vec3 {
        self.target - self.forward() * self.distance
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye_position(), self.target, self.up())
    }

    pub fn projection_matrix(&self, viewport_width: f32, viewport_height: f32) -> Mat4 {
        let safe_width = viewport_width.max(1.0);
        let safe_height = viewport_height.max(1.0);
        let aspect_ratio = safe_width / safe_height;
        Mat4::perspective_rh(self.fov_y_radians, aspect_ratio, self.near, self.far)
    }

    pub fn focus_on(&mut self, target: Vec3) {
        self.target = target;
    }

    pub fn orbit(&mut self, pointer_delta: egui::Vec2) {
        if pointer_delta == egui::Vec2::ZERO {
            return;
        }

        self.yaw -= pointer_delta.x * self.orbit_sensitivity;
        self.pitch = (self.pitch - pointer_delta.y * self.orbit_sensitivity)
            .clamp(-MAX_CAMERA_PITCH_RADIANS, MAX_CAMERA_PITCH_RADIANS);
    }

    pub fn pan(&mut self, pointer_delta: egui::Vec2, viewport_size: egui::Vec2) {
        if pointer_delta == egui::Vec2::ZERO || viewport_size.y <= 0.0 {
            return;
        }

        let vertical_world_span =
            2.0 * self.distance.max(MIN_CAMERA_DISTANCE) * (self.fov_y_radians * 0.5).tan();
        let units_per_pixel = vertical_world_span / viewport_size.y.max(1.0);
        let right = self.right();
        let up = self.up();

        self.target += right * (-pointer_delta.x * units_per_pixel * self.pan_sensitivity);
        self.target += up * (pointer_delta.y * units_per_pixel * self.pan_sensitivity);
    }

    pub fn zoom(&mut self, scroll_delta_y: f32) {
        if scroll_delta_y.abs() <= f32::EPSILON {
            return;
        }

        let zoom_step = scroll_delta_y * self.zoom_sensitivity * (self.distance * 0.002).max(0.01);
        self.distance = (self.distance - zoom_step).clamp(MIN_CAMERA_DISTANCE, MAX_CAMERA_DISTANCE);
    }

    fn forward(&self) -> Vec3 {
        self.rotation() * Vec3::NEG_Z
    }

    fn right(&self) -> Vec3 {
        self.rotation() * Vec3::X
    }

    fn up(&self) -> Vec3 {
        self.rotation() * Vec3::Y
    }
}

pub struct ViewportRenderer {
    texture: Option<wgpu::Texture>,
    view: Option<wgpu::TextureView>,
    texture_id: Option<egui::TextureId>,
    size: [u32; 2],
    runtime: Option<ViewportRenderModule>,
    last_error: Option<String>,
}

impl Default for ViewportRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl ViewportRenderer {
    pub fn new() -> Self {
        Self {
            texture: None,
            view: None,
            texture_id: None,
            size: [0, 0],
            runtime: None,
            last_error: None,
        }
    }

    pub fn texture_id(&self) -> Option<egui::TextureId> {
        self.texture_id
    }

    pub fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    pub fn ensure_size(&mut self, render_state: &egui_wgpu::RenderState, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);

        if self.size == [width, height] && self.texture_id.is_some() {
            if let Some(runtime) = self.runtime.as_mut() {
                runtime.resize(width, height);
            }
            return;
        }

        let texture = render_state
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("engine-editor-viewport-texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let texture_id = {
            let mut renderer = render_state.renderer.write();
            if let Some(texture_id) = self.texture_id {
                renderer.update_egui_texture_from_wgpu_texture(
                    &render_state.device,
                    &view,
                    wgpu::FilterMode::Linear,
                    texture_id,
                );
                texture_id
            } else {
                renderer.register_native_texture(
                    &render_state.device,
                    &view,
                    wgpu::FilterMode::Linear,
                )
            }
        };

        self.texture = Some(texture);
        self.view = Some(view);
        self.texture_id = Some(texture_id);
        self.size = [width, height];

        if let Some(runtime) = self.runtime.as_mut() {
            runtime.resize(width, height);
        } else {
            self.runtime = Some(ViewportRenderModule::new(
                render_state.device.clone(),
                render_state.queue.clone(),
                wgpu::TextureFormat::Rgba8UnormSrgb,
                width,
                height,
            ));
        }
    }

    pub fn render(
        &mut self,
        _render_state: &egui_wgpu::RenderState,
        world: &mut World,
        asset_server: &AssetServer,
    ) {
        let Some(view) = self.view.as_ref() else {
            return;
        };

        let Some(runtime) = self.runtime.as_mut() else {
            return;
        };

        runtime.resize(self.size[0], self.size[1]);
        match runtime.render(world, asset_server, view) {
            Ok(()) => self.last_error = None,
            Err(error) => self.last_error = Some(error.to_string()),
        }
    }

    pub fn free(&mut self, render_state: &egui_wgpu::RenderState) {
        if let Some(texture_id) = self.texture_id.take() {
            render_state.renderer.write().free_texture(&texture_id);
        }

        self.texture = None;
        self.view = None;
        self.size = [0, 0];
        self.runtime = None;
        self.last_error = None;
    }
}

#[cfg(test)]
#[path = "viewport_tests.rs"]
mod tests;
