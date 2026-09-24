//! Immediate-mode debug lines drawn over the tonemapped image, either
//! depth-tested against the scene or always on top.

use std::mem::size_of;

use bytemuck::{Pod, Zeroable};
use engine_core::{DebugLine, Result};

use crate::gpu::{GrowableBuffer, StagingBelt};
use crate::layouts::{SharedLayouts, DEPTH_FORMAT};
use crate::passes::post::needs_manual_srgb;
use crate::shader::ShaderLibrary;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct LineVertex {
    position: [f32; 3],
    color: [f32; 4],
}

const ATTRIBUTES: [wgpu::VertexAttribute; 2] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4];

pub struct DebugLinesPass {
    pipeline_layout: wgpu::PipelineLayout,
    module_linear: std::sync::Arc<wgpu::ShaderModule>,
    module_encode: std::sync::Arc<wgpu::ShaderModule>,
    pipelines: std::collections::HashMap<(wgpu::TextureFormat, bool), wgpu::RenderPipeline>,
    vertices: GrowableBuffer,
    depth_tested: u32,
    overlay: u32,
}

impl DebugLinesPass {
    pub fn new(
        device: &wgpu::Device,
        shaders: &mut ShaderLibrary,
        layouts: &SharedLayouts,
    ) -> Result<Self> {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("engine-render-debug-lines-pl"),
            bind_group_layouts: &[&layouts.view_basic],
            push_constant_ranges: &[],
        });
        Ok(Self {
            pipeline_layout,
            module_linear: shaders.module(device, "debug/lines.wgsl", &[])?,
            module_encode: shaders.module(device, "debug/lines.wgsl", &["OUTPUT_SRGB_ENCODE"])?,
            pipelines: std::collections::HashMap::new(),
            vertices: GrowableBuffer::new(
                device,
                "engine-render-debug-lines",
                wgpu::BufferUsages::VERTEX,
                size_of::<LineVertex>() as u64 * 1024,
            ),
            depth_tested: 0,
            overlay: 0,
        })
    }

    pub fn line_count(&self) -> u32 {
        (self.depth_tested + self.overlay) / 2
    }

    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        belt: &mut StagingBelt,
        lines: &[DebugLine],
    ) {
        let mut vertices = Vec::with_capacity(lines.len() * 2);
        for pass in [true, false] {
            for line in lines.iter().filter(|line| line.depth_test == pass) {
                vertices.push(LineVertex {
                    position: line.start.to_array(),
                    color: line.color,
                });
                vertices.push(LineVertex {
                    position: line.end.to_array(),
                    color: line.color,
                });
            }
        }
        self.depth_tested = 2 * lines.iter().filter(|line| line.depth_test).count() as u32;
        self.overlay = vertices.len() as u32 - self.depth_tested;
        if !vertices.is_empty() {
            self.vertices
                .upload(device, encoder, belt, bytemuck::cast_slice(&vertices));
        }
    }

    fn pipeline(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat, depth: bool) {
        let module = if needs_manual_srgb(format) {
            // Colors are sRGB already; a linear target stores them as-is.
            &self.module_encode
        } else {
            &self.module_linear
        };
        let layout = &self.pipeline_layout;
        self.pipelines.entry((format, depth)).or_insert_with(|| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("engine-render-debug-lines"),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some("vs_main"),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: size_of::<LineVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &ATTRIBUTES,
                    }],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineList,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: false,
                    depth_compare: if depth {
                        wgpu::CompareFunction::LessEqual
                    } else {
                        wgpu::CompareFunction::Always
                    },
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            })
        });
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        format: wgpu::TextureFormat,
        depth: &wgpu::TextureView,
        view_bind_group: &wgpu::BindGroup,
        view_offset: u32,
    ) {
        if self.depth_tested + self.overlay == 0 {
            return;
        }
        self.pipeline(device, format, true);
        self.pipeline(device, format, false);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("engine-render-debug-lines"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: None,
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        pass.set_bind_group(0, view_bind_group, &[view_offset]);
        pass.set_vertex_buffer(0, self.vertices.buffer().slice(..));
        if self.depth_tested > 0 {
            pass.set_pipeline(&self.pipelines[&(format, true)]);
            pass.draw(0..self.depth_tested, 0..1);
        }
        if self.overlay > 0 {
            pass.set_pipeline(&self.pipelines[&(format, false)]);
            pass.draw(self.depth_tested..self.depth_tested + self.overlay, 0..1);
        }
    }
}
