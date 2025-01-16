use std::borrow::Borrow;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::rc::Rc;
use std::sync::Once;

use rusttype;

pub struct Font {
    font: rusttype::Font<'static>,
}

impl Font {
    pub fn new(path: &Path) -> Option<Rc<Font>> {
        let mut memory = Vec::new();
        let mut file = File::open(path).unwrap();
        let _ = file.read_to_end(&mut memory).unwrap();
        Font::from_bytes(&memory)
    }

    pub fn from_bytes(memory: &[u8]) -> Option<Rc<Font>> {
        let font = rusttype::Font::from_bytes(memory.to_vec()).unwrap();
        Some(Rc::new(Font { font }))
    }

    pub fn default() -> Rc<Font> {
        const DATA: &[u8] = include_bytes!("../../font/WorkSans-Regular.ttf");
        static mut DEFAULT_FONT_SINGLETON: Option<Rc<Font>> = None;
        static INIT: Once = Once::new();

        unsafe {
            INIT.call_once(|| {
                DEFAULT_FONT_SINGLETON =
                    Some(Font::from_bytes(DATA).expect("Default font creation failed."));
            });

            DEFAULT_FONT_SINGLETON.clone().unwrap()
        }
    }

    #[inline]
    pub fn font(&self) -> &rusttype::Font<'static> {
        &self.font
    }

    #[inline]
    pub fn uid(font: &Rc<Font>) -> usize {
        (*font).borrow() as *const Font as usize
    }
}
