use std::cell::RefCell;
use std::rc::Rc;
use std::sync::mpsc::{Receiver, TryIter};

use crate::event::window_event::WindowEvent;

/// An event.
pub struct Event<'a> {
    pub value: WindowEvent,
    pub inhibited: bool,
    inhibitor: &'a RefCell<Vec<WindowEvent>>,
}

impl<'a> Drop for Event<'a> {
    #[inline]
    fn drop(&mut self) {
        if !self.inhibited {
            self.inhibitor.borrow_mut().push(self.value)
        }
    }
}

impl<'a> Event<'a> {
    #[inline]
    fn new(value: WindowEvent, inhibitor: &RefCell<Vec<WindowEvent>>) -> Event {
        Event {
            value,
            inhibited: false,
            inhibitor,
        }
    }
}

pub struct Events<'a> {
    stream: TryIter<'a, WindowEvent>,
    inhibitor: &'a RefCell<Vec<WindowEvent>>,
}

impl<'a> Events<'a> {
    #[inline]
    fn new(
        stream: TryIter<'a, WindowEvent>,
        inhibitor: &'a RefCell<Vec<WindowEvent>>,
    ) -> Events<'a> {
        Events { stream, inhibitor }
    }
}

impl<'a> Iterator for Events<'a> {
    type Item = Event<'a>;

    #[inline]
    fn next(&mut self) -> Option<Event<'a>> {
        match self.stream.next() {
            None => None,
            Some(e) => Some(Event::new(e, self.inhibitor)),
        }
    }
}

pub struct EventManager {
    events: Rc<Receiver<WindowEvent>>,
    inhibitor: Rc<RefCell<Vec<WindowEvent>>>,
}

impl EventManager {
    #[inline]
    pub fn new(
        events: Rc<Receiver<WindowEvent>>,
        inhibitor: Rc<RefCell<Vec<WindowEvent>>>,
    ) -> EventManager {
        EventManager { events, inhibitor }
    }

    #[inline]
    pub fn iter(&mut self) -> Events {
        Events::new(self.events.try_iter(), &*self.inhibitor)
    }
}
