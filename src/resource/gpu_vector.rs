use crate::context::context::{Buffer, Context};
use crate::resource::gl_primitive::GLPrimitive;
use crate::verify;

pub struct GPUVec<T> {
    trash: bool,
    len: usize,
    buf_type: BufferType,
    alloc_type: AllocationType,
    buffer: Option<(usize, Buffer)>,
    data: Option<Vec<T>>,
}

impl<T: GLPrimitive> GPUVec<T> {
    pub fn new(data: Vec<T>, buf_type: BufferType, alloc_type: AllocationType) -> GPUVec<T> {
        GPUVec {
            trash: true,
            len: data.len(),
            buf_type,
            alloc_type,
            buffer: None,
            data: Some(data),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        if self.trash {
            match self.data {
                Some(ref d) => d.len(),
                None => panic!("This should never happend."),
            }
        } else {
            self.len
        }
    }

    #[inline]
    pub fn data_mut(&mut self) -> &mut Option<Vec<T>> {
        self.trash = true;

        &mut self.data
    }

    #[inline]
    pub fn data(&self) -> &Option<Vec<T>> {
        &self.data
    }

    #[inline]
    pub fn is_on_gpu(&self) -> bool {
        self.buffer.is_some()
    }

    #[inline]
    pub fn trash(&self) -> bool {
        self.trash
    }

    #[inline]
    pub fn is_on_ram(&self) -> bool {
        self.data.is_some()
    }

    #[inline]
    pub fn load_to_gpu(&mut self) {
        if !self.is_on_gpu() {
            let buf_type = self.buf_type;
            let alloc_type = self.alloc_type;
            let len = &mut self.len;

            self.buffer = self.data.as_ref().map(|d| {
                *len = d.len();
                (d.len(), upload_array(&d[..], buf_type, alloc_type))
            });
        } else if self.trash() {
            for d in self.data.iter() {
                self.len = d.len();

                if let Some((ref mut len, ref buffer)) = self.buffer {
                    *len = update_buffer(&d[..], *len, buffer, self.buf_type, self.alloc_type)
                }
            }
        }

        self.trash = false;
    }

    #[inline]
    pub fn bind(&mut self) {
        self.load_to_gpu();

        let buffer = self.buffer.as_ref().map(|e| &e.1);
        verify!(Context::get().bind_buffer(self.buf_type.to_gl(), buffer));
    }

    #[inline]
    pub fn unbind(&mut self) {
        if self.is_on_gpu() {
            verify!(Context::get().bind_buffer(self.buf_type.to_gl(), None));
        }
    }

    #[inline]
    pub fn unload_from_gpu(&mut self) {
        let _ = self
            .buffer
            .as_ref()
            .map(|&(_, ref h)| unsafe { verify!(Context::get().delete_buffer(Some(h))) });
        self.len = self.len();
        self.buffer = None;
        self.trash = false;
    }

    #[inline]
    pub fn unload_from_ram(&mut self) {
        if self.trash && self.is_on_gpu() {
            self.load_to_gpu();
        }

        self.data = None;
    }
}

impl<T: Clone + GLPrimitive> GPUVec<T> {
    #[inline]
    pub fn to_owned(&self) -> Option<Vec<T>> {
        self.data.as_ref().cloned()
    }
}

#[derive(Clone, Copy)]
pub enum BufferType {
    Array,
    ElementArray,
}

impl BufferType {
    #[inline]
    fn to_gl(&self) -> u32 {
        match *self {
            BufferType::Array => Context::ARRAY_BUFFER,
            BufferType::ElementArray => Context::ELEMENT_ARRAY_BUFFER,
        }
    }
}

#[derive(Clone, Copy)]
pub enum AllocationType {
    StaticDraw,
    DynamicDraw,
    StreamDraw,
}

impl AllocationType {
    #[inline]
    fn to_gl(&self) -> u32 {
        match *self {
            AllocationType::StaticDraw => Context::STATIC_DRAW,
            AllocationType::DynamicDraw => Context::DYNAMIC_DRAW,
            AllocationType::StreamDraw => Context::STREAM_DRAW,
        }
    }
}

#[inline]
pub fn upload_array<T: GLPrimitive>(
    arr: &[T],
    buf_type: BufferType,
    allocation_type: AllocationType,
) -> Buffer {
    // Upload values of vertices
    let buf = verify!(Context::get()
        .create_buffer()
        .expect("Could not create GPU buffer."));
    let _ = update_buffer(arr, 0, &buf, buf_type, allocation_type);
    buf
}

#[inline]
pub fn update_buffer<T: GLPrimitive>(
    arr: &[T],
    gpu_buf_len: usize,
    gpu_buf: &Buffer,
    gpu_buf_type: BufferType,
    gpu_allocation_type: AllocationType,
) -> usize {
    unsafe {
        let ctxt = Context::get();

        verify!(ctxt.bind_buffer(gpu_buf_type.to_gl(), Some(gpu_buf)));

        if arr.len() < gpu_buf_len {
            verify!(ctxt.buffer_sub_data(gpu_buf_type.to_gl(), 0, arr));
            gpu_buf_len
        } else {
            verify!(ctxt.buffer_data(gpu_buf_type.to_gl(), arr, gpu_allocation_type.to_gl()));
            arr.len()
        }
    }
}
