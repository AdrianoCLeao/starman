use crate::builtin::planar_object_material::PlanarObjectMaterial;
use crate::resource::material::PlanarMaterial;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

thread_local!(static KEY_MATERIAL_MANAGER: RefCell<PlanarMaterialManager> = RefCell::new(PlanarMaterialManager::new()));

pub struct PlanarMaterialManager {
    default_material: Rc<RefCell<Box<dyn PlanarMaterial + 'static>>>,
    materials: HashMap<String, Rc<RefCell<Box<dyn PlanarMaterial + 'static>>>>,
}

impl PlanarMaterialManager {
    pub fn new() -> PlanarMaterialManager {
        let mut materials = HashMap::new();

        let om = Rc::new(RefCell::new(
            Box::new(PlanarObjectMaterial::new()) as Box<dyn PlanarMaterial + 'static>
        ));
        let _ = materials.insert("object".to_string(), om.clone());

        PlanarMaterialManager {
            default_material: om,
            materials,
        }
    }

    pub fn get_global_manager<T, F: FnMut(&mut PlanarMaterialManager) -> T>(mut f: F) -> T {
        KEY_MATERIAL_MANAGER.with(|manager| f(&mut *manager.borrow_mut()))
    }

    pub fn get_default(&self) -> Rc<RefCell<Box<dyn PlanarMaterial + 'static>>> {
        self.default_material.clone()
    }

    pub fn get(&mut self, name: &str) -> Option<Rc<RefCell<Box<dyn PlanarMaterial + 'static>>>> {
        self.materials.get(&name.to_string()).cloned()
    }

    pub fn add(&mut self, material: Rc<RefCell<Box<dyn PlanarMaterial + 'static>>>, name: &str) {
        let _ = self.materials.insert(name.to_string(), material);
    }

    pub fn remove(&mut self, name: &str) {
        let _ = self.materials.remove(&name.to_string());
    }
}
