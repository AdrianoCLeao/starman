use nalgebra::Vector2;

pub struct Glyph {
    pub tex: Vector2<f32>,
    pub advance: Vector2<f32>,
    pub dimensions: Vector2<f32>,
    pub offset: Vector2<f32>,
    pub buffer: Vec<u8>,
}

impl Glyph {
    pub fn new(
        tex: Vector2<f32>,
        advance: Vector2<f32>,
        dimensions: Vector2<f32>,
        offset: Vector2<f32>,
        buffer: Vec<u8>,
    ) -> Glyph {
        Glyph {
            tex,
            advance,
            dimensions,
            offset,
            buffer,
        }
    }
}
