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