use std::f32;

use nalgebra::Vector2;

use crate::context::context::Context;
use crate::post_processing::post_processing_effect::PostProcessingEffect;
use crate::resource::effect::{Effect, ShaderAttribute, ShaderUniform};
use crate::resource::framebuffer_manager::RenderTarget;
use crate::resource::gpu_vector::{AllocationType, BufferType, GPUVec};
use crate::verify;

pub struct Waves {
    shader: Effect,
    time: f32,
    offset: ShaderUniform<f32>,
    fbo_texture: ShaderUniform<i32>,
    v_coord: ShaderAttribute<Vector2<f32>>,
    fbo_vertices: GPUVec<Vector2<f32>>,
}

impl Waves {
    pub fn new() -> Waves {
        let fbo_vertices: Vec<Vector2<f32>> = vec![
            Vector2::new(-1.0, -1.0),
            Vector2::new(1.0, -1.0),
            Vector2::new(-1.0, 1.0),
            Vector2::new(1.0, 1.0),
        ];

        let mut fbo_vertices =
            GPUVec::new(fbo_vertices, BufferType::Array, AllocationType::StaticDraw);
        fbo_vertices.load_to_gpu();
        fbo_vertices.unload_from_ram();

        let mut shader = Effect::new_from_str(VERTEX_SHADER, FRAGMENT_SHADER);

        shader.use_program();

        Waves {
            time: 0.0,
            offset: shader.get_uniform("offset").unwrap(),
            fbo_texture: shader.get_uniform("fbo_texture").unwrap(),
            v_coord: shader.get_attrib("v_coord").unwrap(),
            fbo_vertices,
            shader,
        }
    }
}

impl PostProcessingEffect for Waves {
    fn update(&mut self, dt: f32, _: f32, _: f32, _: f32, _: f32) {
        self.time += dt;
    }

    fn draw(&mut self, target: &RenderTarget) {
        let ctxt = Context::get();

        self.shader.use_program();

        let move_amount = self.time * 2.0 * f32::consts::PI * 0.75; 

        self.offset.upload(&move_amount);

        verify!(ctxt.clear_color(0.0, 0.0, 0.0, 1.0));
        verify!(ctxt.clear(Context::COLOR_BUFFER_BIT | Context::DEPTH_BUFFER_BIT));
        verify!(ctxt.bind_texture(Context::TEXTURE_2D, target.texture_id()));

        self.fbo_texture.upload(&0);
        self.v_coord.enable();
        self.v_coord.bind(&mut self.fbo_vertices);

        verify!(ctxt.draw_arrays(Context::TRIANGLE_STRIP, 0, 4));

        self.v_coord.disable();
    }
}

static VERTEX_SHADER: &str = "#version 100
    attribute vec2    v_coord;
    uniform sampler2D fbo_texture;
    varying vec2      f_texcoord;

    void main(void) {
      gl_Position = vec4(v_coord, 0.0, 1.0);
      f_texcoord  = (v_coord + 1.0) / 2.0;
    }";

static FRAGMENT_SHADER: &str = "#version 100
#ifdef GL_FRAGMENT_PRECISION_HIGH
   precision highp float;
#else
   precision mediump float;
#endif

    uniform sampler2D fbo_texture;
    uniform float     offset;
    varying vec2      f_texcoord;

    void main(void) {
      vec2 texcoord =  f_texcoord;
      texcoord.x    += sin(texcoord.y * 4.0 * 2.0 * 3.14159 + offset) / 100.0;
      gl_FragColor  =  texture2D(fbo_texture, texcoord);
    }";
