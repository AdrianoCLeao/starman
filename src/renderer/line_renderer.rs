use crate::camera::camera::Camera;
use crate::context::context::Context;
use crate::renderer::renderer::Renderer;
use crate::resource::effect::{Effect, ShaderAttribute, ShaderUniform};
use crate::resource::gpu_vector::{AllocationType, BufferType, GPUVec};
use crate::verify;
use nalgebra::{Matrix4, Point3};

pub struct LineRenderer {
    shader: Effect,
    pos: ShaderAttribute<Point3<f32>>,
    color: ShaderAttribute<Point3<f32>>,
    view: ShaderUniform<Matrix4<f32>>,
    proj: ShaderUniform<Matrix4<f32>>,
    lines: GPUVec<Point3<f32>>,
    line_width: f32,
}

impl LineRenderer {
    pub fn new() -> LineRenderer {
        let mut shader = Effect::new_from_str(LINES_VERTEX_SRC, LINES_FRAGMENT_SRC);

        shader.use_program();

        LineRenderer {
            lines: GPUVec::new(Vec::new(), BufferType::Array, AllocationType::StreamDraw),
            pos: shader
                .get_attrib::<Point3<f32>>("position")
                .expect("Failed to get shader attribute."),
            color: shader
                .get_attrib::<Point3<f32>>("color")
                .expect("Failed to get shader attribute."),
            view: shader
                .get_uniform::<Matrix4<f32>>("view")
                .expect("Failed to get shader uniform."),
            proj: shader
                .get_uniform::<Matrix4<f32>>("proj")
                .expect("Failed to get shader uniform."),
            shader,
            line_width: 1.0,
        }
    }

    pub fn needs_rendering(&self) -> bool {
        self.lines.len() != 0
    }

    pub fn draw_line(&mut self, a: Point3<f32>, b: Point3<f32>, color: Point3<f32>) {
        for lines in self.lines.data_mut().iter_mut() {
            lines.push(a);
            lines.push(color);
            lines.push(b);
            lines.push(color);
        }
    }

    pub fn set_line_width(&mut self, line_width: f32) {
        self.line_width = line_width.max(
            f32::EPSILON, 
        );
    }
}

impl Renderer for LineRenderer {
    fn render(&mut self, pass: usize, camera: &mut dyn Camera) {
        if self.lines.len() == 0 {
            return;
        }

        self.shader.use_program();
        self.pos.enable();
        self.color.enable();

        camera.upload(pass, &mut self.proj, &mut self.view);

        self.color.bind_sub_buffer(&mut self.lines, 1, 1);
        self.pos.bind_sub_buffer(&mut self.lines, 1, 0);

        let ctxt = Context::get();
        verify!(ctxt.line_width(self.line_width));
        verify!(ctxt.draw_arrays(Context::LINES, 0, (self.lines.len() / 2) as i32));

        self.pos.disable();
        self.color.disable();

        for lines in self.lines.data_mut().iter_mut() {
            lines.clear()
        }
    }
}

pub static LINES_VERTEX_SRC: &str = A_VERY_LONG_STRING;
pub static LINES_FRAGMENT_SRC: &str = ANOTHER_VERY_LONG_STRING;

const A_VERY_LONG_STRING: &str = "#version 100
    attribute vec3 position;
    attribute vec3 color;
    varying   vec3 vColor;
    uniform   mat4 proj;
    uniform   mat4 view;
    void main() {
        gl_Position = proj * view * vec4(position, 1.0);
        vColor = color;
    }";

const ANOTHER_VERY_LONG_STRING: &str = "#version 100
#ifdef GL_FRAGMENT_PRECISION_HIGH
   precision highp float;
#else
   precision mediump float;
#endif

    varying vec3 vColor;
    void main() {
        gl_FragColor = vec4(vColor, 1.0);
    }";
