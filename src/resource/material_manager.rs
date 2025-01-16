//! A resource manager to load materials.

use crate::builtin::{normals_material::NormalsMaterial, object_material::ObjectMaterial, uvs_material::UvsMaterial};
use crate::resource::material::Material;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

pub struct MaterialManager {
    default_material: Rc<RefCell<Box<dyn Material + 'static>>>,
    materials: HashMap<String, Rc<RefCell<Box<dyn Material + 'static>>>>,
}

impl MaterialManager {
    pub fn new() -> MaterialManager {
        let mut materials = HashMap::new();

        let om = Rc::new(RefCell::new(
            Box::new(ObjectMaterial::new()) as Box<dyn Material + 'static>
        ));
        let _ = materials.insert("object".to_string(), om.clone());

        let nm = Rc::new(RefCell::new(
            Box::new(NormalsMaterial::new()) as Box<dyn Material + 'static>
        ));
        let _ = materials.insert("normals".to_string(), nm.clone());

        let um = Rc::new(RefCell::new(
            Box::new(UvsMaterial::new()) as Box<dyn Material + 'static>
        ));
        let _ = materials.insert("uvs".to_string(), um.clone());

        MaterialManager {
            default_material: om,
            materials,
        }
    }

    pub fn get_global_manager<T, F: FnMut(&mut MaterialManager) -> T>(mut f: F) -> T {
        crate::window::window_cache::WINDOW_CACHE
            .with(|manager| f(&mut *manager.borrow_mut().material_manager.as_mut().unwrap()))
    }

    pub fn get_default(&self) -> Rc<RefCell<Box<dyn Material + 'static>>> {
        self.default_material.clone()
    }

    pub fn get(&mut self, name: &str) -> Option<Rc<RefCell<Box<dyn Material + 'static>>>> {
        self.materials.get(&name.to_string()).cloned()
    }

    pub fn add(&mut self, material: Rc<RefCell<Box<dyn Material + 'static>>>, name: &str) {
        let _ = self.materials.insert(name.to_string(), material);
    }

    pub fn remove(&mut self, name: &str) {
        let _ = self.materials.remove(&name.to_string());
    }
}
