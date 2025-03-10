use crate::camera::camera::Camera;
use crate::context::context::Context;
use crate::renderer::renderer::Renderer;
use crate::resource::effect::{Effect, ShaderAttribute, ShaderUniform};
use crate::resource::gpu_vector::{AllocationType, BufferType, GPUVec};
use crate::verify;
use nalgebra::{Matrix4, Point3};

pub struct PointRenderer {
    shader: Effect,
    pos: ShaderAttribute<Point3<f32>>,
    color: ShaderAttribute<Point3<f32>>,
    proj: ShaderUniform<Matrix4<f32>>,
    view: ShaderUniform<Matrix4<f32>>,
    points: GPUVec<Point3<f32>>,
    point_size: f32,
}

impl PointRenderer {
    pub fn new() -> PointRenderer {
        let mut shader = Effect::new_from_str(POINTS_VERTEX_SRC, POINTS_FRAGMENT_SRC);

        shader.use_program();

        PointRenderer {
            points: GPUVec::new(Vec::new(), BufferType::Array, AllocationType::StreamDraw),
            pos: shader.get_attrib::<Point3<f32>>("position").unwrap(),
            color: shader.get_attrib::<Point3<f32>>("color").unwrap(),
            proj: shader.get_uniform::<Matrix4<f32>>("proj").unwrap(),
            view: shader.get_uniform::<Matrix4<f32>>("view").unwrap(),
            shader,
            point_size: 1.0,
        }
    }

    pub fn needs_rendering(&self) -> bool {
        self.points.len() != 0
    }

    pub fn set_point_size(&mut self, pt_size: f32) {
        self.point_size = pt_size;
    }

    pub fn draw_point(&mut self, pt: Point3<f32>, color: Point3<f32>) {
        for points in self.points.data_mut().iter_mut() {
            points.push(pt);
            points.push(color);
        }
    }
}

impl Renderer for PointRenderer {
    fn render(&mut self, pass: usize, camera: &mut dyn Camera) {
        if self.points.len() == 0 {
            return;
        }

        self.shader.use_program();
        self.pos.enable();
        self.color.enable();

        camera.upload(pass, &mut self.proj, &mut self.view);

        self.color.bind_sub_buffer(&mut self.points, 1, 1);
        self.pos.bind_sub_buffer(&mut self.points, 1, 0);

        let ctxt = Context::get();
        verify!(ctxt.point_size(self.point_size));
        verify!(ctxt.draw_arrays(Context::POINTS, 0, (self.points.len() / 2) as i32));

        self.pos.disable();
        self.color.disable();

        for points in self.points.data_mut().iter_mut() {
            points.clear()
        }
    }
}

pub static POINTS_VERTEX_SRC: &str = A_VERY_LONG_STRING;
pub static POINTS_FRAGMENT_SRC: &str = ANOTHER_VERY_LONG_STRING;

const A_VERY_LONG_STRING: &str = "#version 100
    attribute vec3 position;
    attribute vec3 color;
    varying   vec3 Color;
    uniform   mat4 proj;
    uniform   mat4 view;
    void main() {
        gl_Position = proj * view * vec4(position, 1.0);
        Color = color;
    }";

const ANOTHER_VERY_LONG_STRING: &str = "#version 100
#ifdef GL_FRAGMENT_PRECISION_HIGH
   precision highp float;
#else
   precision mediump float;
#endif

    varying vec3 Color;
    void main() {
        gl_FragColor = vec4(Color, 1.0);
    }";
