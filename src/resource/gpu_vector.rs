use super::gl_primitive::GLPrimitive;

pub struct GPUVec<T> {
    trash: bool,
    len: usize,
    buf_type: BufferType,
    alloc_type: AllocationType,
    data: Option<Vec<T>>,
}

#[derive(Clone, Copy)]
pub enum BufferType {
    Array,
    ElementArray,
}

#[derive(Clone, Copy)]
pub enum AllocationType {
    StaticDraw,
    DynamicDraw,
    StreamDraw,
}

impl<T: GLPrimitive> GPUVec<T> {
    pub fn new(data: Vec<T>, buf_type: BufferType, alloc_type: AllocationType) -> GPUVec<T> {
        GPUVec {
            trash: true,
            len: data.len(),
            buf_type,
            alloc_type,
            data: Some(data),
        }
    }
}