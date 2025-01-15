pub use inner::*;

const UNSIGNED_SHORT: u32 = 0x1403; 
const UNSIGNED_INT: u32 = 0x1405;   

#[cfg(not(feature = "vertex_index_u32"))]
mod inner {
    pub type VertexIndex = u16;
    pub const VERTEX_INDEX_TYPE: u32 = super::UNSIGNED_SHORT;
}

#[cfg(feature = "vertex_index_u32")]
mod inner {
    pub type VertexIndex = u32;
    pub const VERTEX_INDEX_TYPE: u32 = super::UNSIGNED_INT;
}
