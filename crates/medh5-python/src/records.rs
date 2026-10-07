//! Record classes: engine value types whose Python face is a frozen
//! dataclass.
//!
//! A record holds the engine value and its JSON form.  Construction builds
//! the JSON object from the arguments (positional in field order, or by
//! keyword --- the dataclass call convention) and parses it with the engine's
//! `from_json`, which is where every rule lives.  Fields read back out of the
//! JSON with the Python types 1.x returned (tuples for sequences, floats for
//! floats, nested records for nested values).

use pyo3::exceptions::{PyAttributeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use serde_json::{Map, Value};

use crate::convert::{json_to_py, py_to_json};

/// How one field reads back.
#[derive(Clone, Copy)]
pub enum Kind {
    /// The JSON value as is (`None` when absent).
    Plain,
    /// A JSON number as `float`.
    Float,
    /// A JSON array as a `tuple` (`()` when absent).
    Tuple,
    /// A JSON array of pairs as a tuple of tuples.
    Pairs,
    /// A JSON object as a `dict` (`{}` when absent).
    Dict,
    /// A string with a default when absent.
    Default(&'static str),
    /// `False` when absent.
    FalseDefault,
    /// A nested record, built by the given constructor (`None` when absent).
    Record(fn(Python<'_>, &Value) -> PyResult<Py<PyAny>>),
    /// A tuple of nested records.
    Records(fn(Python<'_>, &Value) -> PyResult<Py<PyAny>>),
    /// Every top-level member not named by another field (`{}` when none).
    Extra,
}

pub struct Field {
    pub name: &'static str,
    pub key: &'static str,
    pub kind: Kind,
    /// A dataclass field without a default: the constructor needs it.
    pub required: bool,
}

pub const fn field(name: &'static str, kind: Kind) -> Field {
    Field { name, key: name, kind, required: false }
}

/// A field the constructor cannot do without.
pub const fn required(name: &'static str, kind: Kind) -> Field {
    Field { name, key: name, kind, required: true }
}

/// `missing 2 required positional arguments: 'a' and 'b'`, as a dataclass says it.
fn missing_arguments(class: &str, missing: &[&str]) -> PyErr {
    let quoted: Vec<String> = missing.iter().map(|n| medh5::json::repr_str(n)).collect();
    let names = match quoted.as_slice() {
        [one] => one.clone(),
        [a, b] => format!("{a} and {b}"),
        [rest @ .., last] => format!("{}, and {last}", rest.join(", ")),
        [] => String::new(),
    };
    let plural = if missing.len() == 1 { "argument" } else { "arguments" };
    PyTypeError::new_err(format!("{class}.__init__() missing {} required positional {plural}: {names}", missing.len()))
}

/// Read one field from a record's JSON form.
pub fn read_field<'py>(py: Python<'py>, json: &Value, fields: &[Field], f: &Field) -> PyResult<Bound<'py, PyAny>> {
    let value = json.get(f.key);
    let none = || py.None().into_bound(py);
    Ok(match f.kind {
        Kind::Plain => match value {
            Some(v) => json_to_py(py, v)?,
            None => none(),
        },
        Kind::Float => match value.and_then(Value::as_f64) {
            Some(v) => v.into_pyobject(py)?.into_any(),
            None => match value {
                Some(Value::Null) | None => none(),
                Some(other) => json_to_py(py, other)?,
            },
        },
        Kind::Tuple => match value {
            Some(Value::Array(items)) => {
                PyTuple::new(py, items.iter().map(|i| json_to_py(py, i)).collect::<PyResult<Vec<_>>>()?)?.into_any()
            }
            Some(Value::Null) | None => PyTuple::empty(py).into_any(),
            Some(other) => json_to_py(py, other)?,
        },
        Kind::Pairs => match value {
            Some(Value::Array(items)) => {
                let pairs = items
                    .iter()
                    .map(|i| match i {
                        Value::Array(inner) => Ok(PyTuple::new(
                            py,
                            inner.iter().map(|v| json_to_py(py, v)).collect::<PyResult<Vec<_>>>()?,
                        )?
                        .into_any()),
                        other => json_to_py(py, other),
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                PyTuple::new(py, pairs)?.into_any()
            }
            _ => PyTuple::empty(py).into_any(),
        },
        Kind::Dict => match value {
            Some(v @ Value::Object(_)) => json_to_py(py, v)?,
            _ => PyDict::new(py).into_any(),
        },
        Kind::Default(text) => match value {
            Some(v) if !v.is_null() => json_to_py(py, v)?,
            _ => text.into_pyobject(py)?.into_any(),
        },
        Kind::FalseDefault => match value {
            Some(v) if !v.is_null() => json_to_py(py, v)?,
            _ => false.into_pyobject(py)?.to_owned().into_any(),
        },
        Kind::Record(make) => match value {
            Some(v) if !v.is_null() => make(py, v)?.into_bound(py),
            _ => none(),
        },
        Kind::Records(make) => match value {
            Some(Value::Array(items)) => {
                PyTuple::new(py, items.iter().map(|i| make(py, i)).collect::<PyResult<Vec<_>>>()?)?.into_any()
            }
            _ => PyTuple::empty(py).into_any(),
        },
        Kind::Extra => {
            let out = PyDict::new(py);
            if let Value::Object(map) = json {
                for (k, v) in map {
                    if !fields.iter().any(|f| f.key == k) {
                        out.set_item(k, json_to_py(py, v)?)?;
                    }
                }
            }
            out.into_any()
        }
    })
}

/// The JSON object a constructor call describes: arguments matched to
/// fields positionally, then by keyword; `None` and empty values omitted.
pub fn build_json(
    class: &str,
    fields: &[Field],
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Value> {
    if args.len() > fields.len() {
        return Err(PyTypeError::new_err(format!(
            "{class}.__init__() takes from 1 to {} positional arguments but {} were given",
            fields.len() + 1,
            args.len() + 1
        )));
    }
    let mut given: Vec<Option<Bound<'_, PyAny>>> = vec![None; fields.len()];
    for (i, arg) in args.iter().enumerate() {
        given[i] = Some(arg);
    }
    if let Some(kw) = kwargs {
        for (k, v) in kw.iter() {
            let name: String = k.extract()?;
            let Some(i) = fields.iter().position(|f| f.name == name) else {
                return Err(PyTypeError::new_err(format!(
                    "{class}.__init__() got an unexpected keyword argument {}",
                    medh5::json::repr_str(&name)
                )));
            };
            if given[i].is_some() {
                return Err(PyTypeError::new_err(format!(
                    "{class}.__init__() got multiple values for argument {}",
                    medh5::json::repr_str(&name)
                )));
            }
            given[i] = Some(v);
        }
    }
    let missing: Vec<&str> =
        fields.iter().zip(&given).filter(|(f, v)| f.required && v.is_none()).map(|(f, _)| f.name).collect();
    if !missing.is_empty() {
        return Err(missing_arguments(class, &missing));
    }
    let mut out = Map::new();
    for (f, value) in fields.iter().zip(given) {
        let Some(value) = value else { continue };
        if value.is_none() {
            continue;
        }
        let json = py_to_json(&value)?;
        match f.kind {
            Kind::Extra => {
                if let Value::Object(extra) = json {
                    for (k, v) in extra {
                        out.entry(k).or_insert(v);
                    }
                }
            }
            Kind::Tuple | Kind::Pairs | Kind::Records(_) if json.as_array().is_some_and(Vec::is_empty) => {}
            Kind::Dict if json.as_object().is_some_and(Map::is_empty) => {}
            _ => {
                out.insert(f.key.to_string(), json);
            }
        }
    }
    Ok(Value::Object(out))
}

pub fn missing_attribute(class: &str, name: &str) -> PyErr {
    PyAttributeError::new_err(format!(
        "{} object has no attribute {}",
        medh5::json::repr_str(class),
        medh5::json::repr_str(name)
    ))
}

/// Define a record class over an engine value type.
///
/// `parse` turns the JSON form into the engine value (validating it);
/// `dump` turns the engine value back into JSON.
#[macro_export]
macro_rules! record_class {
    (@nullable) => { false };
    (@nullable $flag:expr) => { $flag };
    (
        $rust:ident, $py:literal, $module:literal, $engine:ty,
        parse = $parse:expr,
        dump = $dump:expr,
        $(nullable = $nullable:expr,)?
        fields = [$($field:expr),* $(,)?]
        $(, methods = { $($methods:tt)* })?
    ) => {
        #[pyclass(module = $module, name = $py, skip_from_py_object, frozen)]
        pub struct $rust {
            pub inner: $engine,
            pub json: serde_json::Value,
        }

        impl $rust {
            pub const FIELDS: &'static [$crate::records::Field] = &[$($field),*];
            /// `from_json` of an absent or empty document is `None`.
            pub const NULLABLE: bool = $crate::record_class!(@nullable $($nullable)?);

            pub fn wrap(inner: $engine) -> Self {
                let dump: fn(&$engine) -> serde_json::Value = $dump;
                let json = dump(&inner);
                $rust { inner, json }
            }

            pub fn from_value(value: &serde_json::Value) -> $crate::errors::R<Self> {
                let parse: fn(&serde_json::Value) -> medh5::Result<$engine> = $parse;
                Ok(Self::wrap(parse(value)?))
            }

            /// A Python object for a nested JSON value (used by `Kind::Record`).
            pub fn py_from_json(py: Python<'_>, value: &serde_json::Value) -> PyResult<Py<PyAny>> {
                let made = Self::from_value(value).map_err(PyErr::from)?;
                Ok(Py::new(py, made)?.into_any())
            }

            /// The engine value of a Python argument: this class, or its JSON form.
            pub fn arg(obj: &Bound<'_, PyAny>) -> $crate::errors::R<$engine> {
                if let Ok(rec) = obj.cast::<$rust>() {
                    return Ok(rec.get().inner.clone());
                }
                let parse: fn(&serde_json::Value) -> medh5::Result<$engine> = $parse;
                Ok(parse(&$crate::convert::py_to_json(obj)?)?)
            }
        }

        #[pymethods]
        impl $rust {
            #[new]
            #[pyo3(signature = (*args, **kwargs))]
            fn __new__(args: &Bound<'_, pyo3::types::PyTuple>, kwargs: Option<&Bound<'_, pyo3::types::PyDict>>) -> $crate::errors::R<Self> {
                let json = $crate::records::build_json($py, Self::FIELDS, args, kwargs)?;
                Self::from_value(&json)
            }

            #[classattr]
            fn __dataclass_fields__(py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
                let names: Vec<&str> = Self::FIELDS.iter().map(|f| f.name).collect();
                $crate::geometry::dataclass_fields(py, &names)
            }

            #[classattr]
            fn __match_args__(py: Python<'_>) -> PyResult<Py<pyo3::types::PyTuple>> {
                Ok(pyo3::types::PyTuple::new(py, Self::FIELDS.iter().map(|f| f.name))?.unbind())
            }

            fn __getattr__<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
                match Self::FIELDS.iter().find(|f| f.name == name) {
                    Some(f) => $crate::records::read_field(py, &self.json, Self::FIELDS, f),
                    None => Err($crate::records::missing_attribute($py, name)),
                }
            }

            fn to_json<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                $crate::convert::json_to_py(py, &self.json)
            }

            /// `copy.replace(record, **changes)`.
            #[pyo3(signature = (**changes))]
            fn __replace__(
                slf: &Bound<'_, Self>,
                changes: Option<&Bound<'_, pyo3::types::PyDict>>,
            ) -> PyResult<Py<PyAny>> {
                $crate::values::dataclass_replace(slf.as_any(), changes)
            }

            #[classmethod]
            fn from_json(_cls: &Bound<'_, pyo3::types::PyType>, doc: &Bound<'_, PyAny>) -> $crate::errors::R<Option<Self>> {
                if Self::NULLABLE && (doc.is_none() || !doc.is_truthy()?) {
                    return Ok(None);
                }
                Self::from_value(&$crate::convert::py_to_json(doc)?).map(Some)
            }

            fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
                other.cast::<$rust>().map(|o| o.get().json == self.json).unwrap_or(false)
            }

            fn __hash__(&self) -> isize {
                $crate::values::json_hash(&self.json)
            }

            fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
                let py = slf.py();
                let me = slf.get();
                let fields = Self::FIELDS
                    .iter()
                    .map(|f| Ok((f.name, $crate::records::read_field(py, &me.json, Self::FIELDS, f)?)))
                    .collect::<PyResult<Vec<_>>>()?;
                $crate::values::dataclass_repr($py, &fields)
            }

            fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, pyo3::types::PyTuple>)> {
                $crate::values::reduce_via_json(slf.as_any())
            }

            $($($methods)*)?
        }
    };
}
