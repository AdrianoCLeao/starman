//! The view-model store: gameplay (Lua, Rust plugins) writes values under
//! dotted keys (`hud.health`), layouts bind to them.

use std::collections::{BTreeMap, HashMap};

use bevy_ecs::prelude::Resource;

#[derive(Clone, Debug, PartialEq)]
pub enum UiValue {
    Bool(bool),
    Number(f64),
    Text(String),
    /// Items for `List` nodes; each item's fields bind as `item.<field>`.
    List(Vec<BTreeMap<String, UiValue>>),
}

impl UiValue {
    pub fn as_bool(&self) -> bool {
        match self {
            Self::Bool(v) => *v,
            Self::Number(v) => *v != 0.0,
            Self::Text(v) => !v.is_empty(),
            Self::List(v) => !v.is_empty(),
        }
    }

    pub fn as_number(&self) -> f64 {
        match self {
            Self::Bool(v) => f64::from(u8::from(*v)),
            Self::Number(v) => *v,
            Self::Text(v) => v.trim().parse().unwrap_or(0.0),
            Self::List(v) => v.len() as f64,
        }
    }

    /// Display text: integers without decimals, others with up to two.
    pub fn to_text(&self) -> String {
        match self {
            Self::Bool(v) => v.to_string(),
            Self::Number(v) => {
                if v.fract() == 0.0 && v.abs() < 1e15 {
                    format!("{}", *v as i64)
                } else {
                    let text = format!("{v:.2}");
                    text.trim_end_matches('0').trim_end_matches('.').to_owned()
                }
            }
            Self::Text(v) => v.clone(),
            Self::List(v) => v.len().to_string(),
        }
    }
}

impl From<bool> for UiValue {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl From<f64> for UiValue {
    fn from(value: f64) -> Self {
        Self::Number(value)
    }
}

impl From<f32> for UiValue {
    fn from(value: f32) -> Self {
        Self::Number(value as f64)
    }
}

impl From<i32> for UiValue {
    fn from(value: i32) -> Self {
        Self::Number(value as f64)
    }
}

impl From<&str> for UiValue {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

impl From<String> for UiValue {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

/// Global view-model values shared by every UI document.
#[derive(Resource, Clone, Debug, Default)]
pub struct UiModel {
    values: HashMap<String, UiValue>,
    revision: u64,
}

impl UiModel {
    /// Sets `key` (no-op, and no revision bump, when unchanged).
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<UiValue>) {
        let key = key.into();
        let value = value.into();
        if self.values.get(&key) != Some(&value) {
            self.values.insert(key, value);
            self.revision += 1;
        }
    }

    pub fn remove(&mut self, key: &str) {
        if self.values.remove(key).is_some() {
            self.revision += 1;
        }
    }

    pub fn get(&self, key: &str) -> Option<&UiValue> {
        self.values.get(key)
    }

    pub fn number(&self, key: &str) -> Option<f64> {
        self.get(key).map(UiValue::as_number)
    }

    pub fn text(&self, key: &str) -> Option<String> {
        self.get(key).map(UiValue::to_text)
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &UiValue)> {
        self.values.iter()
    }
}

/// Resolves a key against the model and an optional list item scope.
pub fn lookup<'a>(
    model: &'a UiModel,
    item: Option<&'a BTreeMap<String, UiValue>>,
    key: &str,
) -> Option<&'a UiValue> {
    match (key.strip_prefix("item."), item) {
        (Some(field), Some(item)) => item.get(field),
        _ => model.get(key),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_bumps_revision_only_on_change_and_formats_values() {
        let mut model = UiModel::default();
        model.set("hud.keys", 2);
        let revision = model.revision();
        model.set("hud.keys", 2);
        assert_eq!(model.revision(), revision);
        model.set("hud.health", 0.756);
        assert_eq!(model.text("hud.keys").as_deref(), Some("2"));
        assert_eq!(model.text("hud.health").as_deref(), Some("0.76"));
        assert_eq!(UiValue::Number(1.5).to_text(), "1.5");
        let item: BTreeMap<String, UiValue> =
            [("label".to_owned(), UiValue::from("Slot 1"))].into();
        assert_eq!(
            lookup(&model, Some(&item), "item.label"),
            Some(&UiValue::from("Slot 1"))
        );
        assert_eq!(
            lookup(&model, Some(&item), "hud.keys"),
            Some(&UiValue::Number(2.0))
        );
    }
}
