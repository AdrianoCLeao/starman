use std::{cell::RefCell, mem::take};

use crate::resource::texture_manager::TextureManager;

#[derive(Default)]
pub(crate) struct WindowCache {
    pub(crate) mesh_manager: Option<MeshManager>,
    pub(crate) texture_manager: Option<TextureManager>,
    pub(crate) material_manager: Option<MaterialManager>,
}

thread_local!(pub(crate) static WINDOW_CACHE: RefCell<WindowCache>  = RefCell::new(WindowCache::default()));

impl WindowCache {
    pub fn populate() {
        WINDOW_CACHE.with(|cache| {
            cache.borrow_mut().mesh_manager = Some(MeshManager::new());
            cache.borrow_mut().texture_manager = Some(TextureManager::new());
            cache.borrow_mut().material_manager = Some(MaterialManager::new());
        });
    }

    #[allow(unused_results)]
    pub fn clear() {
        WINDOW_CACHE.with(|cache| take(&mut *cache.borrow_mut()));
    }
}
