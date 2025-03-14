use bitflags::bitflags;

#[derive(Copy, Clone, PartialEq, PartialOrd, Debug, Serialize, Deserialize)]
pub enum WindowEvent {
    Pos(i32, i32),
    Size(u32, u32),
    Close,
    Refresh,
    Focus(bool),
    Iconify(bool),
    FramebufferSize(u32, u32),
    MouseButton(MouseButton, Action, Modifiers),
    CursorPos(f64, f64, Modifiers),
    CursorEnter(bool),
    Scroll(f64, f64, Modifiers),
    Key(Key, Action, Modifiers),
    Char(char),
    CharModifiers(char, Modifiers),
    Touch(u64, f64, f64, TouchAction, Modifiers),
}

use WindowEvent::*;
impl WindowEvent {
    pub fn is_keyboard_event(&self) -> bool {
        matches!(self, Key(..) | Char(..) | CharModifiers(..))
    }

    pub fn is_mouse_event(&self) -> bool {
        matches!(
            self,
            MouseButton(..) | CursorPos(..) | CursorEnter(..) | Scroll(..)
        )
    }

    pub fn is_touch_event(&self) -> bool {
        matches!(self, Touch(..))
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Serialize, Deserialize)]
pub enum Key {
    Key1,
    Key2,
    Key3,
    Key4,
    Key5,
    Key6,
    Key7,
    Key8,
    Key9,
    Key0,
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
    I,
    J,
    K,
    L,
    M,
    N,
    O,
    P,
    Q,
    R,
    S,
    T,
    U,
    V,
    W,
    X,
    Y,
    Z,
    Escape,
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,
    F13,
    F14,
    F15,
    F16,
    F17,
    F18,
    F19,
    F20,
    F21,
    F22,
    F23,
    F24,
    Snapshot,
    Scroll,
    Pause,
    Insert,
    Home,
    Delete,
    End,
    PageDown,
    PageUp,
    Left,
    Up,
    Right,
    Down,
    Back,
    Return,
    Space,
    Compose,
    Caret,
    Numlock,
    Numpad0,
    Numpad1,
    Numpad2,
    Numpad3,
    Numpad4,
    Numpad5,
    Numpad6,
    Numpad7,
    Numpad8,
    Numpad9,
    AbntC1,
    AbntC2,
    Add,
    Apostrophe,
    Apps,
    At,
    Ax,
    Backslash,
    Calculator,
    Capital,
    Colon,
    Comma,
    Convert,
    Decimal,
    Divide,
    Equals,
    Grave,
    Kana,
    Kanji,
    LAlt,
    LBracket,
    LControl,
    LShift,
    LWin,
    Mail,
    MediaSelect,
    MediaStop,
    Minus,
    Multiply,
    Mute,
    MyComputer,
    NavigateForward,
    NavigateBackward,
    NextTrack,
    NoConvert,
    NumpadComma,
    NumpadEnter,
    NumpadEquals,
    OEM102,
    Period,
    PlayPause,
    Power,
    PrevTrack,
    RAlt,
    RBracket,
    RControl,
    RShift,
    RWin,
    Semicolon,
    Slash,
    Sleep,
    Stop,
    Subtract,
    Sysrq,
    Tab,
    Underline,
    Unlabeled,
    VolumeDown,
    VolumeUp,
    Wake,
    WebBack,
    WebFavorites,
    WebForward,
    WebHome,
    WebRefresh,
    WebSearch,
    WebStop,
    Yen,
    Copy,
    Paste,
    Cut,
    Unknown,
}
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Serialize, Deserialize)]
pub enum MouseButton {
    Button1,
    Button2,
    Button3,
    Button4,
    Button5,
    Button6,
    Button7,
    Button8,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Serialize, Deserialize)]
pub enum Action {
    Release,
    Press,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Serialize, Deserialize)]
pub enum TouchAction {
    Start,
    End,
    Move,
    Cancel,
}

bitflags! {
    #[doc = "Key modifiers"]
    #[derive(Serialize, Deserialize)]
    pub struct Modifiers: i32 {
        const Shift       = 0b0001;
        const Control     = 0b0010;
        const Alt         = 0b0100;
        const Super       = 0b1000;
    }
}
