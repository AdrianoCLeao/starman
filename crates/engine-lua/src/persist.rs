//! Persist table for Lua hot reload (`STARMAN_PERSIST`).

use std::fs;
use std::path::Path;

use engine_core::{EngineError, Result};
use mlua::{Lua, Table, Value};
use serde_json::Value as JsonValue;

pub const PERSIST_GLOBAL: &str = "STARMAN_PERSIST";

/// Serializes the global `STARMAN_PERSIST` table to JSON (best-effort).
pub fn capture_persist(lua: &Lua) -> Result<JsonValue> {
    let globals = lua.globals();
    let value: Value = globals
        .get(PERSIST_GLOBAL)
        .map_err(|error| EngineError::Config(format!("lua persist get: {error}")))?;
    lua_value_to_json(lua, value)
}

/// Restores `STARMAN_PERSIST` from JSON after a reload.
pub fn restore_persist(lua: &Lua, data: &JsonValue) -> Result<()> {
    let value = json_to_lua(lua, data)?;
    lua.globals()
        .set(PERSIST_GLOBAL, value)
        .map_err(|error| EngineError::Config(format!("lua persist set: {error}")))?;
    Ok(())
}

pub fn save_persist(path: &Path, data: &JsonValue) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| EngineError::AssetLoad {
            path: parent.display().to_string(),
            reason: error.to_string(),
        })?;
    }
    let text = serde_json::to_string_pretty(data).map_err(|error| EngineError::Config(error.to_string()))?;
    fs::write(path, text).map_err(|error| EngineError::AssetLoad {
        path: path.display().to_string(),
        reason: error.to_string(),
    })
}

pub fn load_persist(path: &Path) -> Result<Option<JsonValue>> {
    if !path.is_file() {
        return Ok(None);
    }
    let text = fs::read_to_string(path).map_err(|error| EngineError::AssetLoad {
        path: path.display().to_string(),
        reason: error.to_string(),
    })?;
    let value: JsonValue =
        serde_json::from_str(&text).map_err(|error| EngineError::Config(error.to_string()))?;
    Ok(Some(value))
}

fn lua_value_to_json(lua: &Lua, value: Value) -> Result<JsonValue> {
    match value {
        Value::Nil => Ok(JsonValue::Null),
        Value::Boolean(b) => Ok(JsonValue::Bool(b)),
        Value::Integer(i) => Ok(JsonValue::from(i)),
        Value::Number(n) => Ok(JsonValue::from(n)),
        Value::String(s) => {
            let s = s
                .to_str()
                .map_err(|error| EngineError::Config(error.to_string()))?
                .to_owned();
            Ok(JsonValue::String(s))
        }
        Value::Table(table) => table_to_json(lua, table),
        _ => Ok(JsonValue::Null),
    }
}

fn table_to_json(lua: &Lua, table: Table) -> Result<JsonValue> {
    // Prefer array if consecutive integer keys starting at 1.
    let len = table
        .raw_len();
    let mut is_array = len > 0;
    if is_array {
        for i in 1..=len {
            let v: Value = table
                .raw_get(i as i64)
                .map_err(|error| EngineError::Config(error.to_string()))?;
            if matches!(v, Value::Nil) {
                is_array = false;
                break;
            }
        }
    }

    if is_array {
        let mut arr = Vec::new();
        for i in 1..=len {
            let v: Value = table
                .raw_get(i as i64)
                .map_err(|error| EngineError::Config(error.to_string()))?;
            arr.push(lua_value_to_json(lua, v)?);
        }
        return Ok(JsonValue::Array(arr));
    }

    let mut map = serde_json::Map::new();
    for pair in table.pairs::<Value, Value>() {
        let (key, value) = pair.map_err(|error| EngineError::Config(error.to_string()))?;
        let key = match key {
            Value::String(s) => s
                .to_str()
                .map_err(|error| EngineError::Config(error.to_string()))?
                .to_owned(),
            Value::Integer(i) => i.to_string(),
            _ => continue,
        };
        map.insert(key, lua_value_to_json(lua, value)?);
    }
    Ok(JsonValue::Object(map))
}

fn json_to_lua(lua: &Lua, value: &JsonValue) -> Result<Value> {
    match value {
        JsonValue::Null => Ok(Value::Nil),
        JsonValue::Bool(b) => Ok(Value::Boolean(*b)),
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(Value::Integer(i))
            } else if let Some(f) = n.as_f64() {
                Ok(Value::Number(f))
            } else {
                Ok(Value::Nil)
            }
        }
        JsonValue::String(s) => Ok(Value::String(
            lua.create_string(s)
                .map_err(|error| EngineError::Config(error.to_string()))?,
        )),
        JsonValue::Array(arr) => {
            let table = lua
                .create_table()
                .map_err(|error| EngineError::Config(error.to_string()))?;
            for (i, item) in arr.iter().enumerate() {
                table
                    .raw_set((i + 1) as i64, json_to_lua(lua, item)?)
                    .map_err(|error| EngineError::Config(error.to_string()))?;
            }
            Ok(Value::Table(table))
        }
        JsonValue::Object(map) => {
            let table = lua
                .create_table()
                .map_err(|error| EngineError::Config(error.to_string()))?;
            for (key, item) in map {
                table
                    .raw_set(key.as_str(), json_to_lua(lua, item)?)
                    .map_err(|error| EngineError::Config(error.to_string()))?;
            }
            Ok(Value::Table(table))
        }
    }
}
