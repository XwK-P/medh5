//! Coercions of JSON values, as the 1.x reference implementation applied them.
//!
//! The document readers of 1.x were written against Python's built-ins:
//! `str(doc["sample_id"])`, `int(doc["index"])`, `float(doc["value"])`.  Those
//! coercions decide what a slightly-off document parses to --- an integer
//! `sample_id` becomes the string `"123"`, a float class id is truncated ---
//! and two implementations must agree on that as much as on the valid case.

use serde_json::{Map, Number, Value};

use crate::json::{py_str, repr};
use crate::{Error, Result};

/// `str(value)`.
pub fn to_str(value: &Value) -> String {
    py_str(value)
}

/// `int(value)`: truncates floats, parses decimal strings, maps bools to 0/1.
pub fn to_int(value: &Value) -> Result<i64> {
    match value {
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i)
            } else if let Some(u) = n.as_u64() {
                i64::try_from(u).map_err(|_| Error::Value(format!("integer {u} is out of range")))
            } else {
                let f = n.as_f64().unwrap_or(f64::NAN);
                if f.is_finite() {
                    Ok(f.trunc() as i64)
                } else {
                    Err(Error::Value(format!("cannot convert float {} to integer", repr(value))))
                }
            }
        }
        Value::Bool(b) => Ok(i64::from(*b)),
        Value::String(s) => s
            .trim()
            .parse::<i64>()
            .map_err(|_| Error::Value(format!("invalid literal for int() with base 10: {}", repr(value)))),
        other => Err(Error::Type(format!(
            "int() argument must be a string, a bytes-like object or a real number, not '{}'",
            type_name(other)
        ))),
    }
}

/// `float(value)`.
pub fn to_float(value: &Value) -> Result<f64> {
    match value {
        Value::Number(n) => Ok(n.as_f64().unwrap_or(f64::NAN)),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::String(s) => {
            let t = s.trim();
            match t.to_ascii_lowercase().as_str() {
                "nan" | "+nan" | "-nan" => Ok(f64::NAN),
                "inf" | "+inf" | "infinity" | "+infinity" => Ok(f64::INFINITY),
                "-inf" | "-infinity" => Ok(f64::NEG_INFINITY),
                _ => t
                    .parse::<f64>()
                    .map_err(|_| Error::Value(format!("could not convert string to float: {}", repr(value)))),
            }
        }
        other => {
            Err(Error::Type(format!("float() argument must be a string or a real number, not '{}'", type_name(other))))
        }
    }
}

/// Python's name for the type a JSON value parses to.
pub fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(n) if n.is_f64() => "float",
        Value::Number(_) => "int",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

/// `doc[key]`, or a `KeyError` naming the key as Python would.
pub fn require<'a>(doc: &'a Map<String, Value>, key: &str) -> Result<&'a Value> {
    doc.get(key).ok_or_else(|| Error::Key(crate::json::repr_str(key)))
}

/// `doc.get(key)`, treating JSON `null` as absent.
pub fn get<'a>(doc: &'a Map<String, Value>, key: &str) -> Option<&'a Value> {
    match doc.get(key) {
        Some(Value::Null) | None => None,
        Some(v) => Some(v),
    }
}

/// `doc.get(key)` as an optional string (`str()` of a non-string).
pub fn get_str(doc: &Map<String, Value>, key: &str) -> Option<String> {
    get(doc, key).map(to_str)
}

/// `doc.get(key)` as an optional number, keeping whether it was an integer.
pub fn get_number(doc: &Map<String, Value>, key: &str) -> Result<Option<Number>> {
    match get(doc, key) {
        None => Ok(None),
        Some(Value::Number(n)) => Ok(Some(n.clone())),
        Some(other) => Ok(Some(float_number(to_float(other)?))),
    }
}

/// `doc.get(key) or ()` as a list of values.
pub fn get_list<'a>(doc: &'a Map<String, Value>, key: &str) -> &'a [Value] {
    match doc.get(key) {
        Some(Value::Array(items)) => items,
        _ => &[],
    }
}

/// A JSON object view of a value, or a type error naming what was expected.
pub fn as_object<'a>(value: &'a Value, what: &str) -> Result<&'a Map<String, Value>> {
    value.as_object().ok_or_else(|| Error::Type(format!("{what} must be a JSON object, not {}", type_name(value))))
}

/// A JSON number from an `f64` (finite), as Python's `float` would be stored.
pub fn float_number(value: f64) -> Number {
    Number::from_f64(value).unwrap_or_else(|| Number::from(0))
}

/// Python's truthiness of a JSON value.
pub fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Python's `str.title()`.
pub fn title_case(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut previous_cased = false;
    for ch in text.chars() {
        if ch.is_alphabetic() {
            if previous_cased {
                out.extend(ch.to_lowercase());
            } else {
                out.extend(ch.to_uppercase());
            }
            previous_cased = true;
        } else {
            out.push(ch);
            previous_cased = false;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn coercions_follow_python() {
        assert_eq!(to_str(&json!(123)), "123");
        assert_eq!(to_str(&json!(1.5)), "1.5");
        assert_eq!(to_int(&json!(5.9)).unwrap(), 5);
        assert_eq!(to_int(&json!("7")).unwrap(), 7);
        assert_eq!(to_float(&json!(2)).unwrap(), 2.0);
        assert_eq!(title_case("left_kidney"), "Left_Kidney");
        assert_eq!(title_case("left kidney"), "Left Kidney");
        assert_eq!(title_case("a1b"), "A1B");
    }
}
