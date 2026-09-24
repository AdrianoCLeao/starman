//! Text: a deterministic font system (bundled Inter plus project fonts,
//! never system fonts), shaping through cosmic-text and a CPU glyph atlas
//! (R8 coverage) the UI renderer uploads when it changes.

use std::collections::{HashMap, HashSet};

use bevy_ecs::prelude::Resource;
use cosmic_text::{
    fontdb, Attrs, Buffer, CacheKey, Family, FontSystem, Metrics, Shaping, SwashCache,
    SwashContent, Weight,
};
use engine_assets::{Asset, AssetId, AssetLoader, LoadContext};
use engine_core::Result;

use crate::style::TextAlign;

pub const INTER_REGULAR: &[u8] = include_bytes!("../fonts/Inter-Regular.ttf");
pub const INTER_BOLD: &[u8] = include_bytes!("../fonts/Inter-Bold.ttf");
pub const DEFAULT_FAMILY: &str = "Inter";

/// A font file asset (`.ttf`/`.otf`).
#[derive(Clone, Debug)]
pub struct FontData {
    pub bytes: Vec<u8>,
}

impl Asset for FontData {
    const TYPE_NAME: &'static str = "Font";
}

pub struct FontLoader;

impl AssetLoader for FontLoader {
    type Asset = FontData;

    fn extensions(&self) -> &'static [&'static str] {
        &["ttf", "otf"]
    }

    fn load(&self, bytes: &[u8], ctx: &mut LoadContext<'_>) -> Result<FontData> {
        // Validate early so broken fonts fail at load, not at draw.
        let mut db = fontdb::Database::new();
        db.load_font_data(bytes.to_vec());
        if db.is_empty() {
            return Err(ctx.error("not a TrueType/OpenType font"));
        }
        Ok(FontData {
            bytes: bytes.to_vec(),
        })
    }
}

/// A packed glyph in the atlas.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlyphEntry {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    /// Offset from the pen position to the bitmap's top-left.
    pub left: i32,
    pub top: i32,
}

#[derive(Clone, Copy, Debug)]
struct Shelf {
    y: u32,
    height: u32,
    x: u32,
}

pub const ATLAS_INITIAL: u32 = 512;
pub const ATLAS_MAX: u32 = 4096;
const PADDING: u32 = 1;

/// R8 glyph coverage atlas with shelf packing. Grows up to
/// [`ATLAS_MAX`]; when full at the maximum it is cleared and refilled.
pub struct GlyphAtlas {
    pub size: u32,
    pub pixels: Vec<u8>,
    shelves: Vec<Shelf>,
    entries: HashMap<CacheKey, Option<GlyphEntry>>,
    /// Bumped on every change (upload trigger).
    pub revision: u64,
    /// Bumped when entries are invalidated (clear); cached layouts must
    /// re-request their glyphs.
    pub generation: u64,
}

impl Default for GlyphAtlas {
    fn default() -> Self {
        Self::with_size(ATLAS_INITIAL)
    }
}

impl GlyphAtlas {
    pub fn with_size(size: u32) -> Self {
        Self {
            size,
            pixels: vec![0; (size * size) as usize],
            shelves: Vec::new(),
            entries: HashMap::new(),
            revision: 1,
            generation: 1,
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn allocate(&mut self, width: u32, height: u32) -> Option<(u32, u32)> {
        let (w, h) = (width + PADDING, height + PADDING);
        if w > self.size || h > self.size {
            return None;
        }
        for shelf in &mut self.shelves {
            if h <= shelf.height && shelf.x + w <= self.size && shelf.height <= h * 2 {
                let position = (shelf.x, shelf.y);
                shelf.x += w;
                return Some(position);
            }
        }
        let y = self.shelves.last().map_or(0, |s| s.y + s.height);
        if y + h > self.size {
            return None;
        }
        self.shelves.push(Shelf { y, height: h, x: w });
        Some((0, y))
    }

    fn grow(&mut self) -> bool {
        if self.size >= ATLAS_MAX {
            return false;
        }
        let new_size = self.size * 2;
        let mut pixels = vec![0; (new_size * new_size) as usize];
        for row in 0..self.size {
            let from = (row * self.size) as usize;
            let to = (row * new_size) as usize;
            pixels[to..to + self.size as usize]
                .copy_from_slice(&self.pixels[from..from + self.size as usize]);
        }
        self.pixels = pixels;
        self.size = new_size;
        self.revision += 1;
        true
    }

    fn clear(&mut self) {
        self.pixels.fill(0);
        self.shelves.clear();
        self.entries.clear();
        self.revision += 1;
        self.generation += 1;
    }

    /// The atlas entry for `key`, rasterizing on first use (`None` for
    /// empty glyphs such as spaces).
    pub fn glyph(
        &mut self,
        key: CacheKey,
        fonts: &mut FontSystem,
        swash: &mut SwashCache,
    ) -> Option<GlyphEntry> {
        if let Some(entry) = self.entries.get(&key) {
            return *entry;
        }
        let image = swash.get_image_uncached(fonts, key);
        let Some(image) = image.filter(|i| i.placement.width > 0 && i.placement.height > 0) else {
            self.entries.insert(key, None);
            return None;
        };
        let (width, height) = (image.placement.width, image.placement.height);
        let coverage: Vec<u8> = match image.content {
            SwashContent::Mask => image.data.clone(),
            SwashContent::Color => image.data.chunks_exact(4).map(|p| p[3]).collect(),
            SwashContent::SubpixelMask => image
                .data
                .chunks_exact(4)
                .map(|p| ((p[0] as u16 + p[1] as u16 + p[2] as u16) / 3) as u8)
                .collect(),
        };
        let position = match self.allocate(width, height) {
            Some(position) => position,
            None => {
                if !self.grow() {
                    self.clear();
                }
                self.allocate(width, height)?
            }
        };
        for row in 0..height {
            let src = (row * width) as usize;
            let dst = ((position.1 + row) * self.size + position.0) as usize;
            self.pixels[dst..dst + width as usize]
                .copy_from_slice(&coverage[src..src + width as usize]);
        }
        let entry = GlyphEntry {
            x: position.0,
            y: position.1,
            width,
            height,
            left: image.placement.left,
            top: image.placement.top,
        };
        self.entries.insert(key, Some(entry));
        self.revision += 1;
        Some(entry)
    }
}

/// Fonts, rasterizer and glyph atlas shared by every UI document.
#[derive(Resource)]
pub struct UiFonts {
    pub system: FontSystem,
    pub swash: SwashCache,
    pub atlas: GlyphAtlas,
    loaded: HashSet<AssetId>,
    families: Vec<String>,
    /// Bumped when fonts are added (text must be re-shaped).
    pub revision: u64,
}

impl Default for UiFonts {
    fn default() -> Self {
        Self::new()
    }
}

impl UiFonts {
    pub fn new() -> Self {
        let mut db = fontdb::Database::new();
        db.load_font_data(INTER_REGULAR.to_vec());
        db.load_font_data(INTER_BOLD.to_vec());
        db.set_sans_serif_family(DEFAULT_FAMILY);
        let families = family_names(&db);
        Self {
            system: FontSystem::new_with_locale_and_db("en-US".to_owned(), db),
            swash: SwashCache::new(),
            atlas: GlyphAtlas::default(),
            loaded: HashSet::new(),
            families,
            revision: 1,
        }
    }

    /// Adds a project font once.
    pub fn add_font(&mut self, id: AssetId, data: &FontData) {
        if self.loaded.insert(id) {
            self.system.db_mut().load_font_data(data.bytes.clone());
            self.families = family_names(self.system.db());
            self.revision += 1;
        }
    }

    pub fn families(&self) -> &[String] {
        &self.families
    }

    pub fn has_family(&self, family: &str) -> bool {
        self.families.iter().any(|f| f == family)
    }
}

fn family_names(db: &fontdb::Database) -> Vec<String> {
    let mut names: Vec<String> = db
        .faces()
        .flat_map(|face| face.families.iter().map(|(name, _)| name.clone()))
        .collect();
    names.sort();
    names.dedup();
    names
}

/// Parameters of one text block.
#[derive(Clone, Debug, PartialEq)]
pub struct TextParams {
    pub text: String,
    /// Physical pixels.
    pub font_size: f32,
    pub line_height: f32,
    pub family: String,
    pub bold: bool,
    pub align: TextAlign,
}

/// A shaped text block owned by a UI node.
pub struct TextBlock {
    pub params: TextParams,
    pub buffer: Buffer,
    /// Width the buffer was last laid out for.
    width: Option<f32>,
    fonts_revision: u64,
}

impl TextBlock {
    pub fn new(fonts: &mut UiFonts, params: TextParams) -> Self {
        let buffer = Buffer::new(
            &mut fonts.system,
            Metrics::new(params.font_size.max(1.0), params.line_height.max(1.0)),
        );
        let mut block = Self {
            params,
            buffer,
            width: Some(-1.0),
            fonts_revision: 0,
        };
        block.reshape(fonts);
        block
    }

    fn reshape(&mut self, fonts: &mut UiFonts) {
        self.buffer.set_metrics(Metrics::new(
            self.params.font_size.max(1.0),
            self.params.line_height.max(1.0),
        ));
        let align = match self.params.align {
            TextAlign::Left => cosmic_text::Align::Left,
            TextAlign::Center => cosmic_text::Align::Center,
            TextAlign::Right => cosmic_text::Align::Right,
            TextAlign::Justified => cosmic_text::Align::Justified,
        };
        let params = self.params.clone();
        let family = if fonts.has_family(&params.family) {
            Family::Name(&params.family)
        } else {
            Family::SansSerif
        };
        let attrs = Attrs::new().family(family).weight(if params.bold {
            Weight::BOLD
        } else {
            Weight::NORMAL
        });
        self.buffer
            .set_text(&params.text, &attrs, Shaping::Advanced, Some(align));
        self.fonts_revision = fonts.revision;
        self.width = Some(-1.0);
    }

    /// Updates the text/style; returns whether anything changed.
    pub fn update(&mut self, fonts: &mut UiFonts, params: TextParams) -> bool {
        if params == self.params && self.fonts_revision == fonts.revision {
            return false;
        }
        self.params = params;
        self.reshape(fonts);
        true
    }

    /// Lays out for `width` (None = unbounded) and returns the size.
    pub fn measure(&mut self, fonts: &mut UiFonts, width: Option<f32>) -> (f32, f32) {
        if self.width != width {
            self.buffer.set_size(width, None);
            self.width = width;
        }
        self.buffer.shape_until_scroll(&mut fonts.system, false);
        let mut w: f32 = 0.0;
        let mut h: f32 = 0.0;
        for run in self.buffer.layout_runs() {
            w = w.max(run.line_w);
            h = h.max(run.line_top + run.line_height);
        }
        (w.ceil(), h.ceil())
    }
}

/// A positioned glyph quad produced from a laid-out block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlyphQuad {
    /// Top-left in the block's space (physical px).
    pub x: f32,
    pub y: f32,
    pub entry: GlyphEntry,
}

/// Glyph quads of `block` (after [`TextBlock::measure`]).
pub fn glyph_quads(block: &TextBlock, fonts: &mut UiFonts) -> Vec<GlyphQuad> {
    let mut quads = Vec::new();
    for run in block.buffer.layout_runs() {
        for glyph in run.glyphs {
            let physical = glyph.physical((0.0, 0.0), 1.0);
            let Some(entry) =
                fonts
                    .atlas
                    .glyph(physical.cache_key, &mut fonts.system, &mut fonts.swash)
            else {
                continue;
            };
            quads.push(GlyphQuad {
                x: (physical.x + entry.left) as f32,
                y: (run.line_y as i32 + physical.y - entry.top) as f32,
                entry,
            });
        }
    }
    quads
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(text: &str) -> TextParams {
        TextParams {
            text: text.to_owned(),
            font_size: 16.0,
            line_height: 20.0,
            family: DEFAULT_FAMILY.to_owned(),
            bold: false,
            align: TextAlign::Left,
        }
    }

    #[test]
    fn bundled_font_shapes_measures_and_wraps() {
        let mut fonts = UiFonts::new();
        assert!(fonts.has_family("Inter"), "{:?}", fonts.families());
        let mut block = TextBlock::new(&mut fonts, params("Salvar jogo — ação"));
        let (w, h) = block.measure(&mut fonts, None);
        assert!(w > 80.0 && w < 200.0, "{w}");
        assert_eq!(h, 20.0);
        let (narrow_w, narrow_h) = block.measure(&mut fonts, Some(60.0));
        assert!(narrow_w <= 60.0 + 1.0, "{narrow_w}");
        assert!(narrow_h >= 40.0, "wrapped onto more lines: {narrow_h}");
        let quads = glyph_quads(&block, &mut fonts);
        assert!(quads.len() >= 14, "{}", quads.len());
        assert!(
            fonts.atlas.pixels.iter().any(|p| *p > 128),
            "coverage rasterized"
        );
        let bold = TextBlock::new(
            &mut fonts,
            TextParams {
                bold: true,
                ..params("Salvar jogo — ação")
            },
        );
        let _ = bold;
    }

    #[test]
    fn atlas_packs_grows_and_clears() {
        let mut atlas = GlyphAtlas::with_size(32);
        let mut placed = Vec::new();
        for _ in 0..3 {
            placed.push(atlas.allocate(10, 10).unwrap());
        }
        assert_eq!(placed[0], (0, 0));
        assert_eq!(placed[1], (11, 0));
        assert!(atlas.allocate(40, 4).is_none());
        assert!(atlas.grow());
        assert_eq!(atlas.size, 64);
        assert!(atlas.allocate(40, 4).is_some());
        let generation = atlas.generation;
        atlas.clear();
        assert_eq!(atlas.generation, generation + 1);
    }
}
