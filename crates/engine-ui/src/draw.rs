//! Draw lists: the quads (rounded/bordered boxes, images, nine-slices,
//! glyphs) of every document, clipped and in paint order, handed to the
//! UI render extension.

use std::sync::Arc;

use bevy_ecs::prelude::Resource;
use engine_assets::{Assets, Handle, TextureData};
use engine_math::Vec2;

use crate::document::NodeKind;
use crate::instance::{Rect, UiInstance};
use crate::text::{glyph_quads, UiFonts};

/// Which texture a quad samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QuadTexture {
    /// Solid color with an SDF rounded box and border.
    None,
    /// Coverage from the glyph atlas.
    Glyphs,
    Image(Handle<TextureData>),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UiQuad {
    /// x, y, width, height in physical pixels.
    pub rect: [f32; 4],
    /// u0, v0, u1, v1 (normalized; atlas pixels for glyphs).
    pub uv: [f32; 4],
    /// sRGB-authored RGBA (alpha already includes opacity).
    pub color: [f32; 4],
    pub border_color: [f32; 4],
    pub corner_radius: f32,
    pub border_width: f32,
    pub texture: QuadTexture,
    pub clip: Rect,
}

/// The glyph atlas image as last published.
#[derive(Clone, Debug)]
pub struct AtlasImage {
    pub size: u32,
    pub pixels: Arc<Vec<u8>>,
    pub revision: u64,
}

/// Every quad of the frame, all documents in order.
#[derive(Resource, Clone, Debug, Default)]
pub struct UiDrawList {
    pub quads: Vec<UiQuad>,
    pub atlas: Option<AtlasImage>,
}

impl UiDrawList {
    /// Publishes the atlas when it changed since the last frame.
    pub fn sync_atlas(&mut self, fonts: &UiFonts) {
        let atlas = &fonts.atlas;
        if self
            .atlas
            .as_ref()
            .is_none_or(|a| a.revision != atlas.revision)
        {
            self.atlas = Some(AtlasImage {
                size: atlas.size,
                pixels: Arc::new(atlas.pixels.clone()),
                revision: atlas.revision,
            });
        }
    }
}

fn with_alpha(mut color: [f32; 4], opacity: f32) -> [f32; 4] {
    color[3] *= opacity;
    color
}

fn solid(rect: Rect, color: [f32; 4], clip: Rect) -> UiQuad {
    UiQuad {
        rect: [rect.x, rect.y, rect.width, rect.height],
        uv: [0.0, 0.0, 1.0, 1.0],
        color,
        border_color: [0.0; 4],
        corner_radius: 0.0,
        border_width: 0.0,
        texture: QuadTexture::None,
        clip,
    }
}

/// Appends the quads of `instance`, offset by `origin` (world anchors).
#[allow(clippy::too_many_arguments)]
pub fn build_quads(
    instance: &mut UiInstance,
    fonts: &mut UiFonts,
    assets: Option<&Assets>,
    focus: Option<usize>,
    caret_visible: bool,
    scale: f32,
    origin: Vec2,
    out: &mut Vec<UiQuad>,
) {
    let order = paint_order(instance);
    for (index, opacity) in order {
        let node = &instance.nodes[index];
        let offset = |r: Rect| Rect {
            x: r.x + origin.x,
            y: r.y + origin.y,
            ..r
        };
        let rect = offset(node.rect);
        let clip = if node.clip == Rect::INFINITE {
            Rect::INFINITE
        } else {
            offset(node.clip)
        };
        if rect.width <= 0.0
            || rect.height <= 0.0
            || clip.intersect(&rect).width <= 0.0 && clip != Rect::INFINITE
        {
            continue;
        }
        let visual = node.visual;
        let border_width = node.style.border_width.unwrap_or(0.0) * scale;
        let radius = (node.style.corner_radius.unwrap_or(0.0) * scale)
            .min(rect.width.min(rect.height) * 0.5);
        if visual.background[3] > 0.0 || (border_width > 0.0 && visual.border_color[3] > 0.0) {
            out.push(UiQuad {
                color: with_alpha(visual.background, opacity),
                border_color: with_alpha(visual.border_color, opacity),
                corner_radius: radius,
                border_width,
                ..solid(rect, [0.0; 4], clip)
            });
        }

        // Images (optionally nine-sliced).
        if let (NodeKind::Image { nine_slice, .. }, Some(image), Some(assets)) =
            (&node.kind, &node.image, assets)
        {
            let handle = assets.request::<TextureData>(image);
            if let Some(texture) = assets.get(handle) {
                let tint = with_alpha(node.style.image_tint.unwrap_or([1.0; 4]), opacity);
                let quad = |r: Rect, uv: [f32; 4]| UiQuad {
                    uv,
                    texture: QuadTexture::Image(handle),
                    ..solid(r, tint, clip)
                };
                match nine_slice {
                    None => out.push(quad(rect, [0.0, 0.0, 1.0, 1.0])),
                    Some([left, right, top, bottom]) => {
                        let (tw, th) = (texture.width.max(1) as f32, texture.height.max(1) as f32);
                        let xs = [0.0, left * scale, rect.width - right * scale, rect.width];
                        let ys = [0.0, top * scale, rect.height - bottom * scale, rect.height];
                        let us = [0.0, left / tw, 1.0 - right / tw, 1.0];
                        let vs = [0.0, top / th, 1.0 - bottom / th, 1.0];
                        for row in 0..3 {
                            for column in 0..3 {
                                let r = Rect {
                                    x: rect.x + xs[column],
                                    y: rect.y + ys[row],
                                    width: (xs[column + 1] - xs[column]).max(0.0),
                                    height: (ys[row + 1] - ys[row]).max(0.0),
                                };
                                if r.width > 0.0 && r.height > 0.0 {
                                    out.push(quad(
                                        r,
                                        [us[column], vs[row], us[column + 1], vs[row + 1]],
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Text.
        if let Some(slot) = node.text {
            let content = offset(node.content);
            let color = with_alpha(visual.text_color, opacity);
            let block = &mut instance.texts[slot];
            let (_, height) = block.measure(fonts, Some(content.width.max(1.0)));
            // Vertically centre single blocks in taller boxes.
            let top = content.y + ((content.height - height) * 0.5).max(0.0);
            let generation = fonts.atlas.generation;
            let quads = glyph_quads(block, fonts);
            if fonts.atlas.generation != generation {
                // The atlas was cleared while rasterizing: skip this frame's
                // glyphs; they are complete next frame.
                continue;
            }
            for glyph in quads {
                let e = glyph.entry;
                out.push(UiQuad {
                    rect: [
                        content.x + glyph.x,
                        top + glyph.y,
                        e.width as f32,
                        e.height as f32,
                    ],
                    // Atlas pixel coordinates: the atlas may grow during the
                    // frame, the renderer normalizes by its final size.
                    uv: [
                        e.x as f32,
                        e.y as f32,
                        (e.x + e.width) as f32,
                        (e.y + e.height) as f32,
                    ],
                    texture: QuadTexture::Glyphs,
                    ..solid(Rect::default(), color, clip)
                });
            }
            // Caret at the end of a focused text input.
            let owner = node.parent.filter(|p| Some(*p) == focus);
            if caret_visible
                && owner.is_some()
                && matches!(
                    instance.nodes[owner.unwrap_or(0)].kind,
                    NodeKind::TextInput { .. }
                )
            {
                let has_text = !instance.nodes[owner.unwrap_or(0)].widget.input.is_empty();
                let block = &instance.texts[slot];
                let end = if has_text {
                    block
                        .buffer
                        .layout_runs()
                        .last()
                        .map_or(0.0, |run| run.line_w)
                } else {
                    0.0
                };
                out.push(solid(
                    Rect {
                        x: content.x + end + 1.0,
                        y: top,
                        width: (1.5 * scale).max(1.0),
                        height: height.max(block.params.line_height),
                    },
                    color,
                    clip,
                ));
            }
        }

        // Scroll indicator.
        if let NodeKind::ScrollView { .. } = node.kind {
            let content = node.content_size;
            if content.y > node.rect.height + 1.0 {
                let fraction = node.rect.height / content.y;
                let travel = node.rect.height * (1.0 - fraction);
                let position = node.widget.scroll.y / (content.y - node.rect.height).max(1.0);
                let bar = Rect {
                    x: rect.x + rect.width - 4.0 * scale,
                    y: rect.y + travel * position,
                    width: 3.0 * scale,
                    height: node.rect.height * fraction,
                };
                out.push(UiQuad {
                    corner_radius: 1.5 * scale,
                    ..solid(
                        bar,
                        with_alpha([1.0, 1.0, 1.0, 0.35], opacity),
                        clip.intersect(&rect),
                    )
                });
            }
        }
    }
}

/// Visible nodes in paint order with their accumulated opacity.
fn paint_order(instance: &UiInstance) -> Vec<(usize, f32)> {
    let mut order = Vec::new();
    fn visit(instance: &UiInstance, index: usize, opacity: f32, order: &mut Vec<(usize, f32)>) {
        let node = &instance.nodes[index];
        if !node.visible || node.style.display == Some(crate::style::Display::None) {
            return;
        }
        let opacity = opacity * node.visual.opacity;
        if opacity <= 0.001 {
            return;
        }
        order.push((index, opacity));
        for child in instance.draw_children(index) {
            visit(instance, child, opacity, order);
        }
    }
    visit(instance, instance.root, 1.0, &mut order);
    order
}
