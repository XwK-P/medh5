//! What every value class shares: dataclass-style `repr`, hashing over the
//! canonical JSON, and pickling through `from_json(to_json())`.

use std::hash::{Hash, Hasher};

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::IntoPyObjectExt;
use serde_json::Value;

/// `Name(field=repr(value), ...)`, the repr a dataclass prints.
pub fn dataclass_repr(name: &str, fields: &[(&str, Bound<'_, PyAny>)]) -> PyResult<String> {
    let mut parts = Vec::with_capacity(fields.len());
    for (key, value) in fields {
        parts.push(format!("{key}={}", value.repr()?));
    }
    Ok(format!("{name}({})", parts.join(", ")))
}

/// A stable hash of a JSON value (its canonical serialization).
pub fn json_hash(value: &Value) -> isize {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    medh5::json::canonical(value).hash(&mut hasher);
    hasher.finish() as isize
}

/// `(type(obj).from_json, (obj.to_json(),))`: pickle and `copy` support for
/// a class whose JSON form round-trips.
pub fn reduce_via_json<'py>(slf: &Bound<'py, PyAny>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
    let constructor = slf.get_type().getattr("from_json")?;
    let state = slf.call_method0("to_json")?;
    Ok((constructor, PyTuple::new(slf.py(), [state])?))
}

/// An optional string as a Python value.
pub fn opt<'py, T: IntoPyObject<'py>>(py: Python<'py>, value: Option<T>) -> PyResult<Bound<'py, PyAny>> {
    match value {
        Some(v) => v.into_bound_py_any(py),
        None => Ok(py.None().into_bound(py)),
    }
}
