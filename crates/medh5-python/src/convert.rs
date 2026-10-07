//! Values across the boundary: JSON documents, NumPy arrays, class keys,
//! regions of interest.

use std::path::PathBuf;

use ndarray::{ArrayD, IxDyn};
use numpy::{PyArrayDescrMethods, PyArrayDyn, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PySlice, PyString, PyTuple};
use serde_json::{Map, Value};

use medh5::array::{DType, Index, NdArray, Slice};
use medh5::labels::ClassKey;

use crate::errors::R;

// -- modules --------------------------------------------------------------------------

pub fn numpy(py: Python<'_>) -> PyResult<&Bound<'_, PyModule>> {
    static NUMPY: PyOnceLock<Py<PyModule>> = PyOnceLock::new();
    Ok(NUMPY.get_or_try_init(py, || py.import("numpy").map(Bound::unbind))?.bind(py))
}

// -- JSON -----------------------------------------------------------------------------

/// A JSON document as Python objects: dicts keep their order, integers stay
/// `int`, every float is a `float`.
pub fn json_to_py<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
    Ok(match value {
        Value::Null => py.None().into_bound(py),
        Value::Bool(b) => PyBool::new(py, *b).to_owned().into_any(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_pyobject(py)?.into_any()
            } else if let Some(u) = n.as_u64() {
                u.into_pyobject(py)?.into_any()
            } else {
                n.as_f64().unwrap_or(f64::NAN).into_pyobject(py)?.into_any()
            }
        }
        Value::String(s) => PyString::new(py, s).into_any(),
        Value::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(json_to_py(py, item)?)?;
            }
            list.into_any()
        }
        Value::Object(map) => {
            let dict = PyDict::new(py);
            for (k, v) in map {
                dict.set_item(k, json_to_py(py, v)?)?;
            }
            dict.into_any()
        }
    })
}

fn key_text(key: &Bound<'_, PyAny>) -> PyResult<String> {
    // `json.dumps` keys: str as is; int, float, bool and None by their JSON spelling.
    if let Ok(s) = key.cast::<PyString>() {
        return Ok(s.to_string());
    }
    if key.is_none() {
        return Ok("null".into());
    }
    if let Ok(b) = key.cast::<PyBool>() {
        return Ok(if b.is_true() { "true" } else { "false" }.into());
    }
    if key.is_instance_of::<PyInt>() || key.is_instance_of::<PyFloat>() {
        return Ok(key.str()?.to_string());
    }
    if let Ok(i) = key.extract::<i64>() {
        return Ok(i.to_string());
    }
    Err(PyTypeError::new_err(format!("keys must be str, int, float, bool or None, not {}", key.get_type().name()?)))
}

/// A Python value as JSON, the way `json.dumps` would see it --- plus NumPy
/// scalars and arrays, which 1.x callers pass freely.
/// A float as a JSON number --- which NaN and infinity are not.
///
/// 1.x let `json.dumps` write them as `NaN`/`Infinity`, which made `/meta`
/// something other than the JSON §2.4 requires; turning them into `null`
/// would change the data without a word.  So they are refused, and the caller
/// says what a missing value is (`None`).
fn finite(value: f64) -> PyResult<Value> {
    if value.is_finite() {
        return Ok(medh5::json::num(value));
    }
    Err(crate::errors::BindError::from(medh5::Error::invalid(format!(
        "{} is not a JSON number: JSON has no NaN or infinity (use None for a missing value)",
        medh5::json::py_float(value)
    )))
    .into())
}

pub fn py_to_json(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(b) = obj.cast::<PyBool>() {
        return Ok(Value::Bool(b.is_true()));
    }
    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(Value::String(s.to_string()));
    }
    if obj.is_instance_of::<PyInt>() {
        if let Ok(i) = obj.extract::<i64>() {
            return Ok(Value::from(i));
        }
        if let Ok(u) = obj.extract::<u64>() {
            return Ok(Value::from(u));
        }
        return finite(obj.extract::<f64>()?);
    }
    if let Ok(f) = obj.cast::<PyFloat>() {
        return finite(f.value());
    }
    if let Ok(dict) = obj.cast::<PyDict>() {
        let mut map = Map::new();
        for (k, v) in dict.iter() {
            map.insert(key_text(&k)?, py_to_json(&v)?);
        }
        return Ok(Value::Object(map));
    }
    if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() {
        let mut items = Vec::new();
        for item in obj.try_iter()? {
            items.push(py_to_json(&item?)?);
        }
        return Ok(Value::Array(items));
    }
    // NumPy scalars and arrays: through their Python equivalents.
    if obj.hasattr("dtype")? && obj.hasattr("tolist")? {
        return py_to_json(&obj.call_method0("tolist")?);
    }
    if obj.hasattr("keys")? && obj.hasattr("__getitem__")? {
        let mut map = Map::new();
        for key in obj.call_method0("keys")?.try_iter()? {
            let key = key?;
            let value = obj.get_item(&key)?;
            map.insert(key_text(&key)?, py_to_json(&value)?);
        }
        return Ok(Value::Object(map));
    }
    if obj.hasattr("to_json")? {
        return py_to_json(&obj.call_method0("to_json")?);
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(Value::from(i));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(medh5::json::num(f));
    }
    if obj.hasattr("__iter__")? && !obj.is_instance_of::<pyo3::types::PyBytes>() {
        let mut items = Vec::new();
        for item in obj.try_iter()? {
            items.push(py_to_json(&item?)?);
        }
        return Ok(Value::Array(items));
    }
    Err(PyTypeError::new_err(format!("Object of type {} is not JSON serializable", obj.get_type().name()?)))
}

/// [`py_to_json`], with `str()` for a value JSON has no form for: for records
/// whose details are free-form and must always serialise.
pub fn py_to_json_lenient(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if let Ok(dict) = obj.cast::<PyDict>() {
        let mut map = Map::new();
        for (k, v) in dict.iter() {
            map.insert(key_text(&k).unwrap_or(k.str()?.to_string()), py_to_json_lenient(&v)?);
        }
        return Ok(Value::Object(map));
    }
    if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() {
        let mut items = Vec::new();
        for item in obj.try_iter()? {
            items.push(py_to_json_lenient(&item?)?);
        }
        return Ok(Value::Array(items));
    }
    match py_to_json(obj) {
        Ok(v) => Ok(v),
        Err(_) => Ok(Value::String(obj.str()?.to_string())),
    }
}

/// A mapping of keyword fields as a JSON object.
pub fn kwargs_to_map(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Map<String, Value>> {
    let mut map = Map::new();
    if let Some(kw) = kwargs {
        for (k, v) in kw.iter() {
            map.insert(k.extract::<String>()?, py_to_json(&v)?);
        }
    }
    Ok(map)
}

pub fn map_to_py<'py>(py: Python<'py>, map: &Map<String, Value>) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (k, v) in map {
        dict.set_item(k, json_to_py(py, v)?)?;
    }
    Ok(dict)
}

// -- arrays ---------------------------------------------------------------------------

/// An engine array as a NumPy array (moved, not copied).
pub fn nd_to_py(py: Python<'_>, array: NdArray) -> Bound<'_, PyAny> {
    match array {
        NdArray::Bool(a) => numpy::PyArray::from_owned_array(py, a).into_any(),
        NdArray::I8(a) => numpy::PyArray::from_owned_array(py, a).into_any(),
        NdArray::U8(a) => numpy::PyArray::from_owned_array(py, a).into_any(),
        NdArray::I16(a) => numpy::PyArray::from_owned_array(py, a).into_any(),
        NdArray::U16(a) => numpy::PyArray::from_owned_array(py, a).into_any(),
        NdArray::I32(a) => numpy::PyArray::from_owned_array(py, a).into_any(),
        NdArray::U32(a) => numpy::PyArray::from_owned_array(py, a).into_any(),
        NdArray::I64(a) => numpy::PyArray::from_owned_array(py, a).into_any(),
        NdArray::U64(a) => numpy::PyArray::from_owned_array(py, a).into_any(),
        NdArray::F16(a) => numpy::PyArray::from_owned_array(py, a).into_any(),
        NdArray::F32(a) => numpy::PyArray::from_owned_array(py, a).into_any(),
        NdArray::F64(a) => numpy::PyArray::from_owned_array(py, a).into_any(),
    }
}

pub fn array_to_py<T: numpy::Element>(py: Python<'_>, array: ArrayD<T>) -> Bound<'_, PyAny> {
    numpy::PyArray::from_owned_array(py, array).into_any()
}

/// `numpy.asarray(obj[, dtype])`, native byte order.
pub fn asarray<'py>(obj: &Bound<'py, PyAny>, dtype: Option<&str>) -> PyResult<Bound<'py, PyUntypedArray>> {
    let py = obj.py();
    let np = numpy(py)?;
    let kwargs = PyDict::new(py);
    if let Some(d) = dtype {
        kwargs.set_item("dtype", d)?;
    }
    let mut array = np.call_method("asarray", (obj,), Some(&kwargs))?;
    let byteorder: String = array.getattr("dtype")?.getattr("byteorder")?.extract()?;
    if byteorder == ">" || (byteorder == "<" && cfg!(target_endian = "big")) {
        let native = array.getattr("dtype")?.call_method1("newbyteorder", ("=",))?;
        array = array.call_method1("astype", (native,))?;
    }
    Ok(array.cast_into::<PyUntypedArray>()?)
}

fn typed<T: numpy::Element + Clone>(array: &Bound<'_, PyUntypedArray>) -> PyResult<ArrayD<T>> {
    let typed = array.cast::<PyArrayDyn<T>>()?;
    Ok(typed.readonly().as_array().to_owned())
}

/// A NumPy array (or anything `numpy.asarray` accepts) as an engine array.
pub fn py_to_nd(obj: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    let array = asarray(obj, None)?;
    untyped_to_nd(&array)
}

pub fn untyped_to_nd(array: &Bound<'_, PyUntypedArray>) -> PyResult<NdArray> {
    let dtype = array.dtype();
    let kind = dtype.kind() as char;
    let size = dtype.itemsize();
    Ok(match (kind, size) {
        ('b', 1) => NdArray::Bool(typed(array)?),
        ('i', 1) => NdArray::I8(typed(array)?),
        ('u', 1) => NdArray::U8(typed(array)?),
        ('i', 2) => NdArray::I16(typed(array)?),
        ('u', 2) => NdArray::U16(typed(array)?),
        ('i', 4) => NdArray::I32(typed(array)?),
        ('u', 4) => NdArray::U32(typed(array)?),
        ('i', 8) => NdArray::I64(typed(array)?),
        ('u', 8) => NdArray::U64(typed(array)?),
        ('f', 2) => NdArray::F16(typed(array)?),
        ('f', 4) => NdArray::F32(typed(array)?),
        ('f', 8) => NdArray::F64(typed(array)?),
        _ => {
            return Err(PyTypeError::new_err(format!(
                "arrays of dtype {} cannot be stored; use a boolean, integer or floating dtype",
                dtype.str()?
            )))
        }
    })
}

/// `numpy.asarray(obj, dtype=bool)`.
pub fn bool_array(obj: &Bound<'_, PyAny>) -> PyResult<ArrayD<bool>> {
    typed(&asarray(obj, Some("bool"))?)
}

/// `numpy.asarray(obj, dtype=float64)`.
pub fn f64_array(obj: &Bound<'_, PyAny>) -> PyResult<ArrayD<f64>> {
    typed(&asarray(obj, Some("float64"))?)
}

/// `numpy.asarray(obj, dtype=int64)`.
pub fn i64_array(obj: &Bound<'_, PyAny>) -> PyResult<ArrayD<i64>> {
    typed(&asarray(obj, Some("int64"))?)
}

pub fn f64_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    Ok(f64_array(obj)?.iter().copied().collect())
}

pub fn i64_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    Ok(i64_array(obj)?.iter().copied().collect())
}

/// A 2-D float matrix.
pub fn matrix(obj: &Bound<'_, PyAny>) -> PyResult<ndarray::Array2<f64>> {
    let a = f64_array(obj)?;
    if a.ndim() != 2 {
        return Err(PyValueError::new_err(format!("expected a 2-D matrix, got shape {:?}", a.shape())));
    }
    a.into_dimensionality::<ndarray::Ix2>().map_err(|e| PyValueError::new_err(e.to_string()))
}

/// `numpy.dtype(obj)` as an engine dtype.
pub fn dtype_arg(obj: &Bound<'_, PyAny>) -> R<DType> {
    let np = numpy(obj.py())?;
    let text: String = np.call_method1("dtype", (obj,))?.getattr("str")?.extract()?;
    Ok(DType::parse(&text)?)
}

pub fn dtype_to_py(py: Python<'_>, dtype: DType) -> PyResult<Bound<'_, PyAny>> {
    numpy(py)?.call_method1("dtype", (dtype.name(),))
}

pub fn zeros_like_shape(shape: &[usize]) -> ArrayD<bool> {
    ArrayD::from_elem(IxDyn(shape), false)
}

// -- keys, regions, paths ----------------------------------------------------------------

/// A class given by id (`int`, NumPy integer) or by key (`str`).
pub fn class_key(obj: &Bound<'_, PyAny>) -> PyResult<ClassKey> {
    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(ClassKey::Key(s.to_string()));
    }
    if obj.is_instance_of::<PyBool>() {
        return Ok(ClassKey::Id(i64::from(obj.extract::<bool>()?)));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(ClassKey::Id(i));
    }
    if let Ok(id) = obj.getattr("id") {
        if let Ok(i) = id.extract::<i64>() {
            return Ok(ClassKey::Id(i));
        }
    }
    Err(PyTypeError::new_err(format!("a class is an int id or a str key, not {}", obj.get_type().name()?)))
}

pub fn class_keys(obj: &Bound<'_, PyAny>) -> PyResult<Vec<ClassKey>> {
    if obj.is_instance_of::<PyString>() {
        return Ok(vec![class_key(obj)?]);
    }
    let mut out = Vec::new();
    for item in obj.try_iter()? {
        out.push(class_key(&item?)?);
    }
    Ok(out)
}

pub fn class_ids(obj: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    let mut out = Vec::new();
    for item in obj.try_iter()? {
        out.push(item?.extract::<i64>()?);
    }
    Ok(out)
}

pub fn strings(obj: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(vec![s.to_string()]);
    }
    let mut out = Vec::new();
    for item in obj.try_iter()? {
        out.push(item?.str()?.to_string());
    }
    Ok(out)
}

pub fn opt_strings(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<String>>> {
    match obj {
        None => Ok(None),
        Some(o) if o.is_none() => Ok(None),
        Some(o) => strings(o).map(Some),
    }
}

fn slice_bound(value: Bound<'_, PyAny>) -> PyResult<Option<i64>> {
    if value.is_none() {
        Ok(None)
    } else {
        Ok(Some(value.extract::<i64>()?))
    }
}

/// One element of a region: `slice` or an integer position.
pub fn index_item(obj: &Bound<'_, PyAny>) -> PyResult<Index> {
    if let Ok(s) = obj.cast::<PySlice>() {
        return Ok(Index::Slice(Slice {
            start: slice_bound(s.getattr("start")?)?,
            stop: slice_bound(s.getattr("stop")?)?,
            step: slice_bound(s.getattr("step")?)?,
        }));
    }
    if obj.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err("a region is slices or integer positions, not bools"));
    }
    Ok(Index::At(obj.extract::<i64>()?))
}

/// A region of interest as a list of index items; `None` stays `None`.
pub fn region(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<Index>>> {
    let Some(obj) = obj else { return Ok(None) };
    if obj.is_none() {
        return Ok(None);
    }
    if obj.is_instance_of::<PySlice>() {
        return Ok(Some(vec![index_item(obj)?]));
    }
    let mut out = Vec::new();
    for item in obj.try_iter()? {
        out.push(index_item(&item?)?);
    }
    Ok(Some(out))
}

/// A region made of slices only (the engine APIs that crop).
pub fn slices(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<Slice>>> {
    let Some(items) = region(obj)? else { return Ok(None) };
    items
        .into_iter()
        .map(|i| match i {
            Index::Slice(s) => Ok(s),
            Index::At(v) => Ok(Slice { start: Some(v), stop: Some(v + 1), step: None }),
        })
        .collect::<PyResult<Vec<_>>>()
        .map(Some)
}

pub fn path(obj: &Bound<'_, PyAny>) -> PyResult<PathBuf> {
    obj.extract::<PathBuf>()
}

/// A tuple of Python values.
pub fn tuple<'py, T: IntoPyObject<'py>>(
    py: Python<'py>,
    items: impl IntoIterator<Item = T, IntoIter: ExactSizeIterator>,
) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(py, items)
}

// -- HDF5 attribute values ------------------------------------------------------------------

use medh5::h5::attrs::AttrValue;

/// An attribute value as h5py would hand it back (scalars as Python
/// scalars, arrays as NumPy arrays, strings as `str`).
pub fn attr_to_py<'py>(py: Python<'py>, value: &AttrValue) -> PyResult<Bound<'py, PyAny>> {
    Ok(match value {
        AttrValue::Str(s) => PyString::new(py, s).into_any(),
        AttrValue::Strs(items) => PyList::new(py, items)?.into_any(),
        AttrValue::Bool(b) => PyBool::new(py, *b).to_owned().into_any(),
        AttrValue::Int(i) => i.into_pyobject(py)?.into_any(),
        AttrValue::Float(f) => f.into_pyobject(py)?.into_any(),
        AttrValue::Array(a) => nd_to_py(py, a.clone()),
        AttrValue::Unsupported(d) => PyString::new(py, d).into_any(),
    })
}

/// A Python value as an attribute (spec §2.5): strings as UTF-8 text, string
/// sequences as string arrays (never a JSON blob), scalars with an explicit
/// width (`int64`, `float64`, `bool`), numeric sequences as `int64`, `float64`
/// or `bool` arrays, arrays as they are.  Anything else is refused.
pub fn py_to_attr(obj: &Bound<'_, PyAny>) -> PyResult<AttrValue> {
    let np = numpy(obj.py())?;
    let is = |name: &str| -> PyResult<bool> { obj.is_instance(&np.getattr(name)?) };
    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(AttrValue::Str(s.to_string()));
    }
    if obj.is_instance_of::<PyBool>() || is("bool_")? {
        return Ok(AttrValue::Bool(obj.is_truthy()?));
    }
    if obj.is_instance_of::<PyInt>() || is("integer")? {
        return Ok(AttrValue::Int(obj.extract()?));
    }
    if obj.is_instance_of::<PyFloat>() || is("floating")? {
        return Ok(AttrValue::Float(obj.extract()?));
    }
    // A fixed-length string as h5py reads one back: text, not code points.
    if obj.is_instance_of::<pyo3::types::PyBytes>() || is("bytes_")? {
        let bytes: Vec<u8> = obj.extract()?;
        return Ok(AttrValue::Str(String::from_utf8_lossy(&bytes).into_owned()));
    }
    if is("ndarray")? {
        let kind: String = obj.getattr("dtype")?.getattr("kind")?.extract()?;
        if matches!(kind.as_str(), "U" | "S" | "O") && obj.getattr("ndim")?.extract::<usize>()? == 1 {
            return py_to_attr(&obj.call_method0("tolist")?);
        }
        return Ok(AttrValue::Array(py_to_nd(obj)?));
    }
    if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() {
        let items: Vec<Bound<'_, PyAny>> = obj.try_iter()?.collect::<PyResult<_>>()?;
        if items.is_empty() {
            return Ok(AttrValue::ints(&[]));
        }
        let all = |test: &dyn Fn(&Bound<'_, PyAny>) -> PyResult<bool>| -> PyResult<bool> {
            for item in &items {
                if !test(item)? {
                    return Ok(false);
                }
            }
            Ok(true)
        };
        if all(&|i| Ok(i.is_instance_of::<PyString>()))? {
            return Ok(AttrValue::Strs(items.iter().map(|i| i.to_string()).collect()));
        }
        let boolean = |i: &Bound<'_, PyAny>| -> PyResult<bool> {
            Ok(i.is_instance_of::<PyBool>() || i.is_instance(&np.getattr("bool_")?)?)
        };
        let integer = |i: &Bound<'_, PyAny>| -> PyResult<bool> {
            Ok(!boolean(i)? && (i.is_instance_of::<PyInt>() || i.is_instance(&np.getattr("integer")?)?))
        };
        let real = |i: &Bound<'_, PyAny>| -> PyResult<bool> {
            Ok(integer(i)? || i.is_instance_of::<PyFloat>() || i.is_instance(&np.getattr("floating")?)?)
        };
        if all(&boolean)? {
            return Ok(AttrValue::Array(NdArray::from(bool_array(obj)?)));
        }
        if all(&integer)? {
            return Ok(AttrValue::Array(NdArray::from(i64_array(obj)?)));
        }
        if all(&real)? {
            return Ok(AttrValue::Array(NdArray::from(f64_array(obj)?)));
        }
        return Ok(AttrValue::Array(py_to_nd(obj)?));
    }
    Err(crate::errors::BindError::from(medh5::Error::invalid(format!(
        "cannot encode attribute value of type {}",
        obj.get_type().repr()?
    )))
    .into())
}
