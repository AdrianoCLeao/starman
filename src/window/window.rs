use std::cell::RefCell;
use std::iter::repeat;
use std::path::Path;
use std::rc::Rc;
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

use nalgebra::{Isometry3, Point2, Point3, Translation3, Vector2, Vector3};

use egui::{Context as EguiContext, RawInput};
use egui_glow::Painter as EguiPainter;

use crate::camera::arc_ball::ArcBall;
use crate::camera::camera::Camera;
use crate::context::context::Context;
use crate::context::context::Texture;
use crate::event::event_manager::EventManager;
use crate::event::window_event::{Action, Key, WindowEvent};
use crate::light::Light;
use crate::planar_camera::{FixedView, PlanarCamera};
use crate::planar_line_renderer::PlanarLineRenderer;
use crate::post_processing::post_processing_effect::PostProcessingEffect;
use crate::renderer::line_renderer::LineRenderer;
use crate::renderer::point_renderer::PointRenderer;
use crate::renderer::renderer::Renderer;
use crate::resource::framebuffer_manager::{FramebufferManager, RenderTarget};
use crate::resource::mesh::Mesh;
use crate::resource::planar_mesh::PlanarMesh;
use crate::resource::texture_manager::TextureManager;
use crate::scene::planar_scene_node::PlanarSceneNode;
use crate::scene::scene_node::SceneNode;
use crate::text::font::Font;
use crate::text::renderer::TextRenderer;
use crate::verify;
use crate::window::canvas::Canvas;
use crate::window::canvas::CanvasSetup;
use crate::window::state::State;
use image::imageops;
use image::{GenericImage, Pixel};
use image::{ImageBuffer, Rgb};
use ncollide3d::procedural::TriMesh;

use super::window_cache::WindowCache;

static DEFAULT_WIDTH: u32 = 800u32;
static DEFAULT_HEIGHT: u32 = 600u32;

pub struct Window {
    events: Rc<Receiver<WindowEvent>>,
    unhandled_events: Rc<RefCell<Vec<WindowEvent>>>,
    min_dur_per_frame: Option<Duration>,
    scene: SceneNode,
    scene2: PlanarSceneNode,
    light_mode: Light,
    background: Vector3<f32>,
    line_renderer: LineRenderer,
    planar_line_renderer: PlanarLineRenderer,
    point_renderer: PointRenderer,
    text_renderer: TextRenderer,
    framebuffer_manager: FramebufferManager,
    post_process_render_target: RenderTarget,
    #[cfg(not(target_arch = "wasm32"))]
    curr_time: std::time::Instant,
    planar_camera: Rc<RefCell<FixedView>>,
    camera: Rc<RefCell<ArcBall>>,
    should_close: bool,
    canvas: Canvas,
    egui_ctx: EguiContext,
    ui_painter_left: EguiPainter,
    ui_painter_right: EguiPainter,
}

impl Drop for Window {
    fn drop(&mut self) {
        WindowCache::clear();
    }
}

impl Window {
    #[inline]
    pub fn should_close(&self) -> bool {
        self.should_close
    }

    #[inline]
    pub fn width(&self) -> u32 {
        self.canvas.size().0
    }

    #[inline]
    pub fn height(&self) -> u32 {
        self.canvas.size().1
    }

    #[inline]
    pub fn size(&self) -> Vector2<u32> {
        let (w, h) = self.canvas.size();
        Vector2::new(w, h)
    }

    #[inline]
    pub fn set_framerate_limit(&mut self, fps: Option<u64>) {
        self.min_dur_per_frame = fps.map(|f| {
            assert!(f != 0);
            Duration::from_millis(1000 / f)
        })
    }

    pub fn set_title(&mut self, title: &str) {
        self.canvas.set_title(title)
    }

    pub fn set_icon(&mut self, icon: impl GenericImage<Pixel = impl Pixel<Subpixel = u8>>) {
        self.canvas.set_icon(icon)
    }

    pub fn set_cursor_grab(&self, grab: bool) {
        self.canvas.set_cursor_grab(grab);
    }

    #[inline]
    pub fn set_cursor_position(&self, x: f64, y: f64) {
        self.canvas.set_cursor_position(x, y);
    }

    #[inline]
    pub fn hide_cursor(&self, hide: bool) {
        self.canvas.hide_cursor(hide);
    }

    #[inline]
    pub fn close(&mut self) {
        self.should_close = true;
    }

    #[inline]
    pub fn hide(&mut self) {
        self.canvas.hide()
    }

    #[inline]
    pub fn show(&mut self) {
        self.canvas.show()
    }

    #[inline]
    pub fn set_background_color(&mut self, r: f32, g: f32, b: f32) {
        self.background.x = r;
        self.background.y = g;
        self.background.z = b;
    }

    #[inline]
    pub fn set_point_size(&mut self, pt_size: f32) {
        self.point_renderer.set_point_size(pt_size);
    }

    #[inline]
    pub fn set_line_width(&mut self, line_width: f32) {
        self.line_renderer.set_line_width(line_width);
        self.planar_line_renderer.set_line_width(line_width);
    }

    #[inline]
    pub fn draw_line(&mut self, a: &Point3<f32>, b: &Point3<f32>, color: &Point3<f32>) {
        self.line_renderer.draw_line(*a, *b, *color);
    }

    #[inline]
    pub fn draw_planar_line(&mut self, a: &Point2<f32>, b: &Point2<f32>, color: &Point3<f32>) {
        self.planar_line_renderer.draw_line(*a, *b, *color);
    }

    #[inline]
    pub fn draw_point(&mut self, pt: &Point3<f32>, color: &Point3<f32>) {
        self.point_renderer.draw_point(*pt, *color);
    }

    #[inline]
    pub fn draw_text(
        &mut self,
        text: &str,
        pos: &Point2<f32>,
        scale: f32,
        font: &Rc<Font>,
        color: &Point3<f32>,
    ) {
        self.text_renderer.draw_text(text, pos, scale, font, color);
    }

    #[deprecated(note = "Use `remove_node` instead.")]
    pub fn remove(&mut self, sn: &mut SceneNode) {
        self.remove_node(sn)
    }

    pub fn remove_node(&mut self, sn: &mut SceneNode) {
        sn.unlink()
    }

    pub fn remove_planar_node(&mut self, sn: &mut PlanarSceneNode) {
        sn.unlink()
    }

    pub fn add_group(&mut self) -> SceneNode {
        self.scene.add_group()
    }

    pub fn add_planar_group(&mut self) -> PlanarSceneNode {
        self.scene2.add_group()
    }

    pub fn add_obj(
        &mut self,
        path: &Path,
        mtl_dir: &Path,
        scale: Vector3<f32>,
        position: Vector3<f32>,
    ) -> SceneNode {
        self.scene.add_obj(path, mtl_dir, scale, position)
    }

    pub fn add_glb(&mut self, path: &Path, scale: Vector3<f32>) -> SceneNode {
        self.scene.add_glb(path, scale)
    }

    pub fn add_mesh(&mut self, mesh: Rc<RefCell<Mesh>>, scale: Vector3<f32>) -> SceneNode {
        self.scene.add_mesh(mesh, scale)
    }

    pub fn add_planar_mesh(
        &mut self,
        mesh: Rc<RefCell<PlanarMesh>>,
        scale: Vector2<f32>,
    ) -> PlanarSceneNode {
        self.scene2.add_mesh(mesh, scale)
    }

    pub fn add_trimesh(&mut self, descr: TriMesh<f32>, scale: Vector3<f32>) -> SceneNode {
        self.scene.add_trimesh(descr, scale)
    }

    pub fn add_geom_with_name(
        &mut self,
        geometry_name: &str,
        scale: Vector3<f32>,
    ) -> Option<SceneNode> {
        self.scene.add_geom_with_name(geometry_name, scale)
    }

    pub fn draw_compass(&mut self, size: f32, color: &Point3<f32>) {
        let viewport_width = self.canvas.size().0 as f32;
        let viewport_height = self.canvas.size().1 as f32;

        let compass_exists = self.canvas.has_compass();

        if !compass_exists {
            let mut compass_node = self.add_cube(size, size, size);
            compass_node.set_fixed(true);
            self.canvas.set_compass_node(compass_node);
        }

        if let Some(compass_node) = self.canvas.get_compass_node_mut() {
            let x_position = viewport_width / 20.0 - size;
            let y_position = viewport_height / 20.0 - size;
            let z_position = 0.0;

            let compass_position = Point3::new(x_position, y_position, z_position);

            compass_node.set_color(color.x, color.y, color.z);
            compass_node.set_local_translation(compass_position.into());
        }
    }

    pub fn add_cube(&mut self, wx: f32, wy: f32, wz: f32) -> SceneNode {
        self.scene.add_cube(wx, wy, wz)
    }

    pub fn add_sphere(&mut self, r: f32) -> SceneNode {
        self.scene.add_sphere(r)
    }

    pub fn add_cone(&mut self, r: f32, h: f32) -> SceneNode {
        self.scene.add_cone(r, h)
    }

    pub fn add_cylinder(&mut self, r: f32, h: f32) -> SceneNode {
        self.scene.add_cylinder(r, h)
    }

    pub fn add_capsule(&mut self, r: f32, h: f32) -> SceneNode {
        self.scene.add_capsule(r, h)
    }

    pub fn add_planar_capsule(&mut self, r: f32, h: f32) -> PlanarSceneNode {
        self.scene2.add_capsule(r, h)
    }

    pub fn add_quad(&mut self, w: f32, h: f32, usubdivs: usize, vsubdivs: usize) -> SceneNode {
        self.scene.add_quad(w, h, usubdivs, vsubdivs)
    }

    pub fn add_quad_with_vertices(
        &mut self,
        vertices: &[Point3<f32>],
        nhpoints: usize,
        nvpoints: usize,
    ) -> SceneNode {
        self.scene
            .add_quad_with_vertices(vertices, nhpoints, nvpoints)
    }

    pub fn add_texture(&mut self, path: &Path, name: &str) -> Rc<Texture> {
        TextureManager::get_global_manager(|tm| tm.add(path, name))
    }

    pub fn add_rectangle(&mut self, wx: f32, wy: f32) -> PlanarSceneNode {
        self.scene2.add_rectangle(wx, wy)
    }

    pub fn add_circle(&mut self, r: f32) -> PlanarSceneNode {
        self.scene2.add_circle(r)
    }

    pub fn add_convex_polygon(
        &mut self,
        polygon: Vec<Point2<f32>>,
        scale: Vector2<f32>,
    ) -> PlanarSceneNode {
        self.scene2.add_convex_polygon(polygon, scale)
    }

    pub fn is_closed(&self) -> bool {
        false
    }

    pub fn scale_factor(&self) -> f64 {
        self.canvas.scale_factor()
    }

    pub fn set_light(&mut self, pos: Light) {
        self.light_mode = pos;
    }

    pub fn new_hidden(title: &str) -> Window {
        Window::do_new(title, true, DEFAULT_WIDTH, DEFAULT_HEIGHT, None)
    }

    pub fn new(title: &str) -> Window {
        Window::do_new(title, false, DEFAULT_WIDTH, DEFAULT_HEIGHT, None)
    }

    pub fn new_with_size(title: &str, width: u32, height: u32) -> Window {
        Window::do_new(title, false, width, height, None)
    }

    pub fn new_with_setup(title: &str, width: u32, height: u32, setup: CanvasSetup) -> Window {
        Window::do_new(title, false, width, height, Some(setup))
    }

    fn do_new(
        title: &str,
        hide: bool,
        width: u32,
        height: u32,
        setup: Option<CanvasSetup>,
    ) -> Window {
        let (event_send, event_receive) = mpsc::channel();
        let canvas = Canvas::open(title, hide, width, height, setup, event_send);

        init_gl();
        WindowCache::populate();
        let egui_ctx = egui::Context::default();
        let gl = Context::get().raw_gl();

        let ui_painter_left = egui_glow::Painter::new(gl.clone(), "", None)
            .expect("Falha ao criar o painter do egui para o painel esquerdo.");
        let ui_painter_right = egui_glow::Painter::new(gl, "", None)
            .expect("Falha ao criar o painter do egui para o painel direito.");

        let mut usr_window = Window {
            should_close: false,
            min_dur_per_frame: None,
            canvas,
            events: Rc::new(event_receive),
            unhandled_events: Rc::new(RefCell::new(Vec::new())),
            scene: SceneNode::new_empty(),
            scene2: PlanarSceneNode::new_empty(),
            light_mode: Light::Absolute(Point3::new(0.0, 10.0, 0.0)),
            background: Vector3::new(0.20, 0.20, 0.20),
            line_renderer: LineRenderer::new(),
            planar_line_renderer: PlanarLineRenderer::new(),
            point_renderer: PointRenderer::new(),
            text_renderer: TextRenderer::new(),
            post_process_render_target: FramebufferManager::new_render_target(
                width as usize,
                height as usize,
                true,
            ),
            framebuffer_manager: FramebufferManager::new(),
            #[cfg(not(target_arch = "wasm32"))]
            curr_time: std::time::Instant::now(),
            planar_camera: Rc::new(RefCell::new(FixedView::new())),
            camera: Rc::new(RefCell::new(ArcBall::new(
                Point3::new(0.0f32, 0.0, -1.0),
                Point3::origin(),
            ))),
            egui_ctx,
            ui_painter_left,
            ui_painter_right,
        };

        if hide {
            usr_window.canvas.hide()
        }

        let light = usr_window.light_mode.clone();
        usr_window.set_light(light);

        usr_window
    }

    #[inline]
    pub fn scene(&self) -> &SceneNode {
        &self.scene
    }

    #[inline]
    pub fn scene_mut(&mut self) -> &mut SceneNode {
        &mut self.scene
    }

    pub fn snap(&self, out: &mut Vec<u8>) {
        let (width, height) = self.canvas.size();
        self.snap_rect(out, 0, 0, width as usize, height as usize)
    }

    pub fn snap_rect(&self, out: &mut Vec<u8>, x: usize, y: usize, width: usize, height: usize) {
        let size = (width * height * 3) as usize;

        if out.len() < size {
            let diff = size - out.len();
            out.extend(repeat(0).take(diff));
        } else {
            out.truncate(size)
        }

        let ctxt = Context::get();
        ctxt.pixel_storei(Context::PACK_ALIGNMENT, 1);
        ctxt.read_pixels(
            x as i32,
            y as i32,
            width as i32,
            height as i32,
            Context::RGB,
            Some(out),
        );
    }

    pub fn snap_image(&self) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        let (width, height) = self.canvas.size();
        let mut buf = Vec::new();
        self.snap(&mut buf);
        let img_opt = ImageBuffer::from_vec(width as u32, height as u32, buf);
        let img = img_opt.expect("Buffer created from window was not big enough for image.");
        imageops::flip_vertical(&img)
    }

    pub fn events(&self) -> EventManager {
        EventManager::new(self.events.clone(), self.unhandled_events.clone())
    }

    pub fn get_key(&self, key: Key) -> Action {
        self.canvas.get_key(key)
    }

    pub fn cursor_pos(&self) -> Option<(f64, f64)> {
        self.canvas.cursor_pos()
    }

    #[inline]
    fn handle_events(
        &mut self,
        camera: &mut Option<&mut dyn Camera>,
        planar_camera: &mut Option<&mut dyn PlanarCamera>,
    ) {
        let unhandled_events = self.unhandled_events.clone();
        let events = self.events.clone();

        for event in unhandled_events.borrow().iter() {
            self.handle_event(camera, planar_camera, event)
        }

        for event in events.try_iter() {
            self.handle_event(camera, planar_camera, &event)
        }

        unhandled_events.borrow_mut().clear();
        self.canvas.poll_events();
    }

    fn handle_event(
        &mut self,
        camera: &mut Option<&mut dyn Camera>,
        planar_camera: &mut Option<&mut dyn PlanarCamera>,
        event: &WindowEvent,
    ) {
        match *event {
            WindowEvent::Key(Key::Escape, Action::Release, _) | WindowEvent::Close => {
                self.close();
            }
            WindowEvent::FramebufferSize(w, h) => {
                self.update_viewport(w as f32, h as f32);
            }
            _ => {}
        }

        match *planar_camera {
            Some(ref mut cam) => cam.handle_event(&self.canvas, event),
            None => self.camera.borrow_mut().handle_event(&self.canvas, event),
        }

        match *camera {
            Some(ref mut cam) => cam.handle_event(&self.canvas, event),
            None => self.camera.borrow_mut().handle_event(&self.canvas, event),
        }
    }

    pub fn render_loop<S: State>(self, state: S) {
        struct DropControl<S> {
            state: S,
            window: Window,
        }

        let mut dc = DropControl {
            state,
            window: self,
        };

        Canvas::render_loop(move |_| dc.window.do_render_with_state(&mut dc.state));
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn render_with_state<S: State>(&mut self, state: &mut S) -> bool {
        self.do_render_with_state(state)
    }

    fn do_render_with_state<S: State>(&mut self, state: &mut S) -> bool {
        {
            let (camera, planar_camera, renderer, effect) = state.cameras_and_effect_and_renderer();
            self.should_close = !self.do_render_with(camera, planar_camera, renderer, effect);
        }

        if !self.should_close {
            state.step(self)
        }

        !self.should_close
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn render(&mut self) -> bool {
        self.render_with(None, None, None)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn render_with_effect(&mut self, effect: &mut (dyn PostProcessingEffect)) -> bool {
        self.render_with(None, None, Some(effect))
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn render_with_camera(&mut self, camera: &mut (dyn Camera)) -> bool {
        self.render_with(Some(camera), None, None)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn render_with_cameras(
        &mut self,
        camera: &mut dyn Camera,
        planar_camera: &mut dyn PlanarCamera,
    ) -> bool {
        self.render_with(Some(camera), Some(planar_camera), None)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn render_with_camera_and_effect(
        &mut self,
        camera: &mut dyn Camera,
        effect: &mut dyn PostProcessingEffect,
    ) -> bool {
        self.render_with(Some(camera), None, Some(effect))
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn render_with_cameras_and_effect(
        &mut self,
        camera: &mut dyn Camera,
        planar_camera: &mut dyn PlanarCamera,
        effect: &mut dyn PostProcessingEffect,
    ) -> bool {
        self.render_with(Some(camera), Some(planar_camera), Some(effect))
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn render_with(
        &mut self,
        camera: Option<&mut dyn Camera>,
        planar_camera: Option<&mut dyn PlanarCamera>,
        post_processing: Option<&mut dyn PostProcessingEffect>,
    ) -> bool {
        self.do_render_with(camera, planar_camera, None, post_processing)
    }

    fn do_render_with(
        &mut self,
        camera: Option<&mut dyn Camera>,
        planar_camera: Option<&mut dyn PlanarCamera>,
        renderer: Option<&mut dyn Renderer>,
        post_processing: Option<&mut dyn PostProcessingEffect>,
    ) -> bool {
        let mut camera = camera;
        let mut planar_camera = planar_camera;
        self.handle_events(&mut camera, &mut planar_camera);

        let self_cam2 = self.planar_camera.clone();
        let mut bself_cam2 = self_cam2.borrow_mut();

        let self_cam = self.camera.clone();
        let mut bself_cam = self_cam.borrow_mut();

        match (camera, planar_camera) {
            (Some(cam), Some(cam2)) => {
                self.render_single_frame(cam, cam2, renderer, post_processing)
            }
            (None, Some(cam2)) => {
                self.render_single_frame(&mut *bself_cam, cam2, renderer, post_processing)
            }
            (Some(cam), None) => {
                self.render_single_frame(cam, &mut *bself_cam2, renderer, post_processing)
            }
            (None, None) => self.render_single_frame(
                &mut *bself_cam,
                &mut *bself_cam2,
                renderer,
                post_processing,
            ),
        }
    }

    fn render_ui_left(&mut self, width: i32, height: i32) {
        let raw_input = RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::new(0.0, 0.0),
                egui::vec2(width as f32, height as f32),
            )),
            pixels_per_point: Some(self.canvas.scale_factor() as f32),
            ..Default::default()
        };

        self.egui_ctx.begin_frame(raw_input);

        egui::SidePanel::left("left_side_panel")
            .resizable(false)
            .default_width(width as f32)
            .show(&self.egui_ctx, |ui| {
                ui.heading("Painel Esquerdo");
                if ui.button("Botão 1").clicked() {
                    println!("Botão 1 clicado!");
                }
                if ui.button("Botão 2").clicked() {
                    println!("Botão 2 clicado!");
                }
                ui.separator();
                ui.label("Outros widgets podem ser adicionados aqui.");
            });

        let full_output = self.egui_ctx.end_frame();
        let clipped_primitives = self.egui_ctx.tessellate(full_output.shapes);

        let screen_size_px = [width as u32, height as u32];
        self.ui_painter_left.paint_and_update_textures(
            screen_size_px,
            self.egui_ctx.pixels_per_point(),
            &clipped_primitives,
            &full_output.textures_delta,
        )
    }

    fn render_ui_right(&mut self, width: i32, height: i32) {
        let raw_input = RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::new(0.0, 0.0),
                egui::vec2(width as f32, height as f32),
            )),
            pixels_per_point: Some(self.canvas.scale_factor() as f32),
            ..Default::default()
        };

        self.egui_ctx.begin_frame(raw_input);

        egui::SidePanel::right("right_side_panel")
            .resizable(false)
            .default_width(width as f32)
            .show(&self.egui_ctx, |ui| {
                ui.heading("Painel Direito");
                ui.label("Informações ou controles aqui.");
                if ui.button("Ação").clicked() {
                    println!("Botão do painel direito clicado!");
                }
            });

        let full_output = self.egui_ctx.end_frame();
        let clipped_primitives = self.egui_ctx.tessellate(full_output.shapes);

        let screen_size_px = [width as u32, height as u32];

        self.ui_painter_right.paint_and_update_textures(
            screen_size_px,
            self.egui_ctx.pixels_per_point(),
            &clipped_primitives,
            &full_output.textures_delta,
        )
    }

    fn render_single_frame(
        &mut self,
        camera: &mut dyn Camera,
        planar_camera: &mut dyn PlanarCamera,
        mut renderer: Option<&mut dyn Renderer>,
        mut post_processing: Option<&mut dyn PostProcessingEffect>,
    ) -> bool {
        let window_width = self.width() as i32;
        let window_height = self.height() as i32;
        let sidebar_width: i32 = 200;
        let central_width = window_width - 2 * sidebar_width;

        planar_camera.handle_event(
            &self.canvas,
            &WindowEvent::FramebufferSize(window_width as u32, window_height as u32),
        );
        camera.handle_event(
            &self.canvas,
            &WindowEvent::FramebufferSize(window_width as u32, window_height as u32),
        );
        planar_camera.update(&self.canvas);
        camera.update(&self.canvas);

        if let Light::StickToCamera = self.light_mode {
            self.set_light(Light::StickToCamera)
        }

        if post_processing.is_some() {
            self.framebuffer_manager
                .select(&self.post_process_render_target);
        } else {
            self.framebuffer_manager
                .select(&FramebufferManager::screen());
        }

        {
            let central_x = sidebar_width;
            Context::get().viewport(central_x, 0, central_width, window_height);
            Context::get().scissor(central_x, 0, central_width, window_height);

            for pass in 0usize..camera.num_passes() {
                camera.start_pass(pass, &self.canvas);
                self.render_scene(camera, pass);

                if let Some(ref mut renderer) = renderer {
                    renderer.render(pass, camera);
                }
            }
            camera.render_complete(&self.canvas);
        }

        self.render_planar_scene(planar_camera);

        let (znear, zfar) = camera.clip_planes();
        if let Some(ref mut p) = post_processing {
            self.framebuffer_manager
                .select(&FramebufferManager::screen());
            p.update(
                0.016,
                central_width as f32,
                window_height as f32,
                znear,
                zfar,
            );
            p.draw(&self.post_process_render_target);
        }

        {
            Context::get().viewport(0, 0, sidebar_width, window_height);
            Context::get().scissor(0, 0, sidebar_width, window_height);
            self.render_ui_left(sidebar_width, window_height);

            // Renderize a barra lateral direita
            Context::get().viewport(
                window_width - sidebar_width,
                0,
                sidebar_width,
                window_height,
            );
            Context::get().scissor(
                window_width - sidebar_width,
                0,
                sidebar_width,
                window_height,
            );
            self.render_ui_right(sidebar_width, window_height);
        }

        self.text_renderer
            .render(window_width as f32, window_height as f32);

        Context::get().viewport(0, 0, window_width, window_height);
        Context::get().scissor(0, 0, window_width, window_height);

        self.canvas.swap_buffers();

        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Some(dur) = self.min_dur_per_frame {
                let elapsed = self.curr_time.elapsed();
                if elapsed < dur {
                    std::thread::sleep(dur - elapsed);
                }
            }
            self.curr_time = std::time::Instant::now();
        }

        !self.should_close()
    }

    fn render_scene(&mut self, camera: &mut dyn Camera, pass: usize) {
        let ctxt = Context::get();
        verify!(ctxt.active_texture(Context::TEXTURE0));
        verify!(ctxt.clear_color(self.background.x, self.background.y, self.background.z, 1.0));
        verify!(ctxt.clear(Context::COLOR_BUFFER_BIT));
        verify!(ctxt.clear(Context::DEPTH_BUFFER_BIT));

        self.line_renderer.render(pass, camera);
        self.point_renderer.render(pass, camera);
        self.scene.data_mut().render(pass, camera, &self.light_mode);
    }

    fn render_planar_scene(&mut self, camera: &mut dyn PlanarCamera) {
        let ctxt = Context::get();
        verify!(ctxt.active_texture(Context::TEXTURE0));

        if self.planar_line_renderer.needs_rendering() {
            self.planar_line_renderer.render(camera);
        }

        // if self.point_renderer2.needs_rendering() {
        //     self.point_renderer2.render(camera);
        // }

        self.scene2.data_mut().render(camera);
    }

    fn update_viewport(&mut self, w: f32, h: f32) {
        verify!(Context::get().scissor(0, 0, w as i32, h as i32));
        FramebufferManager::screen().resize(w, h);
        self.post_process_render_target.resize(w, h);
    }
}

fn init_gl() {
    let ctxt = Context::get();
    verify!(ctxt.front_face(Context::CCW));
    verify!(ctxt.enable(Context::DEPTH_TEST));
    verify!(ctxt.enable(Context::SCISSOR_TEST));
    #[cfg(not(target_arch = "wasm32"))]
    {
        verify!(ctxt.enable(Context::PROGRAM_POINT_SIZE));
    }
    verify!(ctxt.depth_func(Context::LEQUAL));
    verify!(ctxt.front_face(Context::CCW));
    verify!(ctxt.enable(Context::CULL_FACE));
    verify!(ctxt.cull_face(Context::BACK));
}
