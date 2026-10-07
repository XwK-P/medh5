//! Attribute encoding (spec §2.5).
//!
//! | Logical type | HDF5 encoding |
//! |---|---|
//! | string | variable-length UTF-8 scalar |
//! | string list | 1-D array of variable-length UTF-8, never a JSON string |
//! | boolean | `int8` enum `FALSE=0, TRUE=1` (NumPy `bool_`) |
//! | integer / list | `int64` scalar / 1-D `int64` |
//! | float / list | `float64` scalar / 1-D `float64` |
//! | matrix | 2-D array, stored 2-D |
//!
//! Readers accept fixed-length as well as variable-length strings, ASCII as
//! well as UTF-8, and normalise to `String`.

use hdf5::types::{TypeDescriptor as TD, VarLenUnicode};
use ndarray::{ArrayD, IxDyn};
use serde_json::Value;

use crate::array::{DType, NdArray};
use crate::json::{num, repr_str};
use crate::{with_array, with_dtype, Error, Result};

/// One attribute value.
#[derive(Debug, Clone, PartialEq)]
pub enum AttrValue {
    /// A scalar string.
    Str(String),
    /// A 1-D (or flattened n-D) array of strings, with its shape.
    Strs(Vec<String>),
    /// A scalar boolean.
    Bool(bool),
    /// A scalar integer (written `int64`).
    Int(i64),
    /// A scalar float (written `float64`).
    Float(f64),
    /// A numeric or boolean array of any dtype and shape.
    Array(NdArray),
    /// A type this codec does not decode (compound, reference, ...).
    Unsupported(String),
}

impl AttrValue {
    /// A 1-D `int64` array.
    pub fn ints(values: &[i64]) -> AttrValue {
        AttrValue::Array(NdArray::from(ArrayD::from_shape_vec(IxDyn(&[values.len()]), values.to_vec()).unwrap()))
    }

    /// A 1-D `float64` array.
    pub fn floats(values: &[f64]) -> AttrValue {
        AttrValue::Array(NdArray::from(ArrayD::from_shape_vec(IxDyn(&[values.len()]), values.to_vec()).unwrap()))
    }

    /// A 2-D `float64` matrix.
    pub fn matrix(rows: usize, cols: usize, values: &[f64]) -> AttrValue {
        AttrValue::Array(NdArray::from(ArrayD::from_shape_vec(IxDyn(&[rows, cols]), values.to_vec()).unwrap()))
    }

    /// A string list.
    pub fn strs<S: AsRef<str>>(values: &[S]) -> AttrValue {
        AttrValue::Strs(values.iter().map(|s| s.as_ref().to_string()).collect())
    }

    /// The value as a string (spec §2.5: readers accept bytes and str).
    pub fn as_str(&self) -> Option<String> {
        match self {
            AttrValue::Str(s) => Some(s.clone()),
            AttrValue::Strs(v) if v.len() == 1 => Some(v[0].clone()),
            _ => None,
        }
    }

    /// The value as a string list; a scalar string is a list of one.
    pub fn as_str_list(&self) -> Option<Vec<String>> {
        match self {
            AttrValue::Str(s) => Some(vec![s.clone()]),
            AttrValue::Strs(v) => Some(v.clone()),
            _ => None,
        }
    }

    /// Every element as `f64` (scalars are lists of one).
    pub fn as_f64_vec(&self) -> Option<Vec<f64>> {
        match self {
            AttrValue::Bool(b) => Some(vec![if *b { 1.0 } else { 0.0 }]),
            AttrValue::Int(i) => Some(vec![*i as f64]),
            AttrValue::Float(f) => Some(vec![*f]),
            AttrValue::Array(a) => Some(a.to_f64().iter().copied().collect()),
            _ => None,
        }
    }

    /// Every element as `i64` (floats truncated; scalars are lists of one).
    pub fn as_i64_vec(&self) -> Option<Vec<i64>> {
        match self {
            AttrValue::Bool(b) => Some(vec![i64::from(*b)]),
            AttrValue::Int(i) => Some(vec![*i]),
            AttrValue::Float(f) => Some(vec![*f as i64]),
            AttrValue::Array(a) => Some(a.cast::<i64>().iter().copied().collect()),
            _ => None,
        }
    }

    /// The value as one integer.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            AttrValue::Array(a) if a.len() == 1 => self.as_i64_vec().map(|v| v[0]),
            AttrValue::Array(_) => None,
            _ => self.as_i64_vec().and_then(|v| v.first().copied()),
        }
    }

    /// The value as one float.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            AttrValue::Array(a) if a.len() == 1 => self.as_f64_vec().map(|v| v[0]),
            AttrValue::Array(_) => None,
            _ => self.as_f64_vec().and_then(|v| v.first().copied()),
        }
    }

    /// The value as one boolean (`bool(np.asarray(v).reshape(()).item())`).
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            AttrValue::Bool(b) => Some(*b),
            AttrValue::Int(i) => Some(*i != 0),
            AttrValue::Float(f) => Some(*f != 0.0),
            AttrValue::Array(a) if a.len() == 1 => Some(a.to_f64().iter().next().copied().unwrap_or(0.0) != 0.0),
            AttrValue::Str(s) => Some(!s.is_empty()),
            _ => None,
        }
    }

    /// The value as a 2-D float matrix `(rows, cols, row-major values)`.
    pub fn as_matrix(&self) -> Option<(usize, usize, Vec<f64>)> {
        match self {
            AttrValue::Array(a) if a.ndim() == 2 => {
                let shape = a.shape();
                Some((shape[0], shape[1], a.to_f64().iter().copied().collect()))
            }
            _ => None,
        }
    }

    /// The shape of the stored value (`[]` for scalars).
    pub fn shape(&self) -> Vec<usize> {
        match self {
            AttrValue::Strs(v) => vec![v.len()],
            AttrValue::Array(a) => a.shape(),
            _ => Vec::new(),
        }
    }

    /// The value as JSON, as the 1.x reference serialized attributes for
    /// `canonical_attrs` (§13.2): arrays as nested lists, scalars as items.
    pub fn to_json(&self) -> Value {
        match self {
            AttrValue::Str(s) => Value::String(s.clone()),
            AttrValue::Strs(v) => Value::Array(v.iter().map(|s| Value::String(s.clone())).collect()),
            AttrValue::Bool(b) => Value::Bool(*b),
            AttrValue::Int(i) => Value::from(*i),
            AttrValue::Float(f) => num(*f),
            AttrValue::Array(a) => array_to_json(a),
            AttrValue::Unsupported(d) => Value::String(d.clone()),
        }
    }

    /// A short type description, for diagnostics.
    pub fn describe(&self) -> String {
        match self {
            AttrValue::Str(_) => "str".into(),
            AttrValue::Strs(v) => format!("str[{}]", v.len()),
            AttrValue::Bool(_) => "bool".into(),
            AttrValue::Int(_) => "int64".into(),
            AttrValue::Float(_) => "float64".into(),
            AttrValue::Array(a) => format!("{}{:?}", a.dtype().name(), a.shape()),
            AttrValue::Unsupported(d) => d.clone(),
        }
    }
}

/// Nested JSON lists of an array (NumPy `tolist()`).
pub fn array_to_json(array: &NdArray) -> Value {
    fn nest(values: &[Value], shape: &[usize]) -> Value {
        if shape.is_empty() {
            return values.first().cloned().unwrap_or(Value::Null);
        }
        if shape.len() == 1 {
            return Value::Array(values.to_vec());
        }
        let stride: usize = shape[1..].iter().product();
        let mut out = Vec::with_capacity(shape[0]);
        for i in 0..shape[0] {
            out.push(nest(&values[i * stride..(i + 1) * stride], &shape[1..]));
        }
        Value::Array(out)
    }
    let flat: Vec<Value> = match array {
        NdArray::Bool(a) => a.iter().map(|v| Value::Bool(*v)).collect(),
        NdArray::F16(_) | NdArray::F32(_) | NdArray::F64(_) => array.to_f64().iter().map(|v| num(*v)).collect(),
        NdArray::U64(a) => a.iter().map(|v| Value::from(*v)).collect(),
        other => other.cast::<i64>().iter().map(|v| Value::from(*v)).collect(),
    };
    nest(&flat, &array.shape())
}

fn descriptor(attr: &hdf5::Attribute) -> Result<TD> {
    Ok(attr.dtype()?.to_descriptor()?)
}

/// Read one attribute, or `None` when it is absent.
pub fn read(obj: &hdf5::Location, name: &str) -> Result<Option<AttrValue>> {
    if !has(obj, name) {
        return Ok(None);
    }
    let attr = obj.attr(name)?;
    Ok(Some(read_attribute(&attr)?))
}

/// Decode an open attribute.
pub fn read_attribute(attr: &hdf5::Attribute) -> Result<AttrValue> {
    let td = match descriptor(attr) {
        Ok(td) => td,
        Err(e) => return Ok(AttrValue::Unsupported(e.to_string())),
    };
    let shape = attr.shape();
    let scalar = shape.is_empty();
    let value = match td {
        TD::VarLenUnicode | TD::VarLenAscii | TD::FixedAscii(_) | TD::FixedUnicode(_) => {
            let strings = read_strings_attr(attr, &td)?;
            if scalar {
                AttrValue::Str(strings.into_iter().next().unwrap_or_default())
            } else {
                AttrValue::Strs(strings)
            }
        }
        TD::Boolean => {
            let a = attr.read_dyn::<bool>()?;
            if scalar {
                AttrValue::Bool(a.iter().next().copied().unwrap_or(false))
            } else {
                AttrValue::Array(NdArray::Bool(a))
            }
        }
        TD::Integer(_) | TD::Unsigned(_) | TD::Float(_) => {
            let dtype = numeric_dtype(&td).expect("numeric descriptor");
            let array = with_dtype!(dtype, T => NdArray::from(attr.read_dyn::<T>()?));
            if scalar {
                match &array {
                    NdArray::F16(_) | NdArray::F32(_) | NdArray::F64(_) => {
                        AttrValue::Float(array.to_f64().iter().next().copied().unwrap_or(0.0))
                    }
                    NdArray::U64(a) if a.iter().next().copied().unwrap_or(0) > i64::MAX as u64 => {
                        AttrValue::Array(array)
                    }
                    _ => AttrValue::Int(array.cast::<i64>().iter().next().copied().unwrap_or(0)),
                }
            } else {
                AttrValue::Array(array)
            }
        }
        TD::Enum(e) => {
            // A non-boolean enum reads as its base integer, as NumPy sees it.
            let dtype = enum_base(&e);
            let raw = read_enum_raw(attr, dtype)?;
            if scalar {
                AttrValue::Int(raw.cast::<i64>().iter().next().copied().unwrap_or(0))
            } else {
                AttrValue::Array(raw)
            }
        }
        other => AttrValue::Unsupported(format!("{other}")),
    };
    Ok(value)
}

fn read_strings_attr(attr: &hdf5::Attribute, td: &TD) -> Result<Vec<String>> {
    Ok(match td {
        TD::VarLenUnicode => attr.read_raw::<VarLenUnicode>()?.into_iter().map(|s| s.as_str().to_string()).collect(),
        TD::VarLenAscii => {
            attr.read_raw::<hdf5::types::VarLenAscii>()?.into_iter().map(|s| s.as_str().to_string()).collect()
        }
        TD::FixedAscii(_) | TD::FixedUnicode(_) => {
            let bytes = read_fixed_strings(attr.id(), attr.size(), td, true)?;
            bytes
        }
        _ => Vec::new(),
    })
}

/// Read fixed-length strings by asking HDF5 to convert them to variable length.
pub(crate) fn read_fixed_strings(
    id: crate::h5sys::h5i::hid_t,
    n: usize,
    _td: &TD,
    is_attr: bool,
) -> Result<Vec<String>> {
    use crate::h5sys::{h5a, h5d, h5p, h5s, h5t};
    // Read the stored bytes in the stored type --- no character-set
    // conversion, which HDF5 refuses between ASCII and UTF-8 --- and decode
    // them as h5py and NumPy do: trailing NULs dropped, UTF-8 with
    // replacement.
    super::locked(|| unsafe {
        let ftype = if is_attr { h5a::H5Aget_type(id) } else { h5d::H5Dget_type(id) };
        if ftype < 0 {
            return Err(Error::Io("could not read a fixed-length string".into()));
        }
        let size = h5t::H5Tget_size(ftype).max(1);
        let mut buf = vec![0u8; size * n.max(1)];
        let status = if is_attr {
            h5a::H5Aread(id, ftype, buf.as_mut_ptr().cast())
        } else {
            h5d::H5Dread(id, ftype, h5s::H5S_ALL, h5s::H5S_ALL, h5p::H5P_DEFAULT, buf.as_mut_ptr().cast())
        };
        h5t::H5Tclose(ftype);
        if status < 0 {
            return Err(Error::Io("could not read a fixed-length string".into()));
        }
        Ok(buf
            .chunks(size)
            .take(n)
            .map(|item| {
                let end = item.iter().rposition(|b| *b != 0).map(|i| i + 1).unwrap_or(0);
                String::from_utf8_lossy(&item[..end]).into_owned()
            })
            .collect())
    })
}

fn enum_base(e: &hdf5::types::EnumType) -> DType {
    use hdf5::types::IntSize;
    match (e.size, e.signed) {
        (IntSize::U1, true) => DType::I8,
        (IntSize::U1, false) => DType::U8,
        (IntSize::U2, true) => DType::I16,
        (IntSize::U2, false) => DType::U16,
        (IntSize::U4, true) => DType::I32,
        (IntSize::U4, false) => DType::U32,
        (IntSize::U8, true) => DType::I64,
        (IntSize::U8, false) => DType::U64,
    }
}

fn read_enum_raw(attr: &hdf5::Attribute, dtype: DType) -> Result<NdArray> {
    // Read the bytes in the file's own enum type and reinterpret them as the
    // base integer: HDF5 has no enum -> integer conversion path.
    use crate::h5sys::{h5a, h5t};
    let shape = attr.shape();
    let n: usize = shape.iter().product::<usize>().max(1);
    let mut buf = vec![0u8; n * dtype.itemsize()];
    super::locked(|| unsafe {
        let ftype = h5a::H5Aget_type(attr.id());
        let mtype = h5t::H5Tget_native_type(ftype, h5t::H5T_direction_t::H5T_DIR_ASCEND);
        let status = h5a::H5Aread(attr.id(), mtype, buf.as_mut_ptr().cast());
        h5t::H5Tclose(mtype);
        h5t::H5Tclose(ftype);
        if status < 0 {
            return Err(Error::Io("could not read an enum attribute".into()));
        }
        Ok(())
    })?;
    bytes_to_array(&buf, dtype, if shape.is_empty() { &[] } else { &shape })
}

/// Native-endian bytes to an array of `dtype` and `shape`.
pub(crate) fn bytes_to_array(buf: &[u8], dtype: DType, shape: &[usize]) -> Result<NdArray> {
    let n: usize = shape.iter().product::<usize>();
    let n = if shape.is_empty() { 1 } else { n };
    let size = dtype.itemsize();
    if buf.len() < n * size {
        return Err(Error::Io("short read".into()));
    }
    macro_rules! decode {
        ($t:ty) => {{
            let values: Vec<$t> =
                (0..n).map(|i| <$t>::from_ne_bytes(buf[i * size..(i + 1) * size].try_into().unwrap())).collect();
            NdArray::from(ArrayD::from_shape_vec(IxDyn(shape), values)?)
        }};
    }
    Ok(match dtype {
        DType::I8 => decode!(i8),
        DType::U8 => decode!(u8),
        DType::I16 => decode!(i16),
        DType::U16 => decode!(u16),
        DType::I32 => decode!(i32),
        DType::U32 => decode!(u32),
        DType::I64 => decode!(i64),
        DType::U64 => decode!(u64),
        DType::Bool => NdArray::from(ArrayD::from_shape_vec(IxDyn(shape), buf[..n].iter().map(|b| *b != 0).collect())?),
        DType::F16 => NdArray::from(ArrayD::from_shape_vec(
            IxDyn(shape),
            (0..n).map(|i| half::f16::from_ne_bytes(buf[i * 2..i * 2 + 2].try_into().unwrap())).collect(),
        )?),
        DType::F32 => decode!(f32),
        DType::F64 => decode!(f64),
    })
}

/// The array dtype of a numeric descriptor.
pub fn numeric_dtype(td: &TD) -> Option<DType> {
    use hdf5::types::{FloatSize, IntSize};
    Some(match td {
        TD::Boolean => DType::Bool,
        TD::Integer(IntSize::U1) => DType::I8,
        TD::Integer(IntSize::U2) => DType::I16,
        TD::Integer(IntSize::U4) => DType::I32,
        TD::Integer(IntSize::U8) => DType::I64,
        TD::Unsigned(IntSize::U1) => DType::U8,
        TD::Unsigned(IntSize::U2) => DType::U16,
        TD::Unsigned(IntSize::U4) => DType::U32,
        TD::Unsigned(IntSize::U8) => DType::U64,
        TD::Float(FloatSize::U2) => DType::F16,
        TD::Float(FloatSize::U4) => DType::F32,
        TD::Float(FloatSize::U8) => DType::F64,
        TD::Enum(e) => enum_base(e),
        _ => return None,
    })
}

/// Whether an attribute exists.
pub fn has(obj: &hdf5::Location, name: &str) -> bool {
    obj.attr_names().map(|names| names.iter().any(|n| n == name)).unwrap_or(false)
}

/// Every attribute name, sorted (HDF5's name order).
pub fn names(obj: &hdf5::Location) -> Result<Vec<String>> {
    let mut names = obj.attr_names()?;
    names.sort();
    Ok(names)
}

/// Delete an attribute if present.
pub fn delete(obj: &hdf5::Location, name: &str) -> Result<()> {
    if has(obj, name) {
        obj.delete_attr(name)?;
    }
    Ok(())
}

/// Write (or overwrite) one attribute.
pub fn write(obj: &hdf5::Location, name: &str, value: &AttrValue) -> Result<()> {
    delete(obj, name)?;
    match value {
        AttrValue::Str(s) => {
            let v: VarLenUnicode = s.parse().map_err(|e| Error::Value(format!("{e}")))?;
            obj.new_attr::<VarLenUnicode>().shape(()).create(name)?.write_scalar(&v)?;
        }
        AttrValue::Strs(values) => {
            let parsed: Vec<VarLenUnicode> = values
                .iter()
                .map(|s| s.parse())
                .collect::<std::result::Result<_, _>>()
                .map_err(|e| Error::Value(format!("{e}")))?;
            let arr = ndarray::Array1::from(parsed);
            obj.new_attr::<VarLenUnicode>().shape(values.len()).create(name)?.write(&arr)?;
        }
        AttrValue::Bool(b) => {
            obj.new_attr::<bool>().shape(()).create(name)?.write_scalar(b)?;
        }
        AttrValue::Int(i) => {
            obj.new_attr::<i64>().shape(()).create(name)?.write_scalar(i)?;
        }
        AttrValue::Float(f) => {
            obj.new_attr::<f64>().shape(()).create(name)?.write_scalar(f)?;
        }
        AttrValue::Array(array) => {
            with_array!(array, a => {
                obj.new_attr_builder().with_data(a.view()).create(name)?;
            });
        }
        AttrValue::Unsupported(d) => {
            return Err(Error::Value(format!("cannot write attribute {} of unsupported type {d}", repr_str(name))));
        }
    }
    Ok(())
}

/// Write a mapping of attributes, skipping `None` values.
pub fn write_all(obj: &hdf5::Location, values: &[(&str, Option<AttrValue>)]) -> Result<()> {
    for (name, value) in values {
        if let Some(v) = value {
            write(obj, name, v)?;
        }
    }
    Ok(())
}

// -- typed getters -----------------------------------------------------------

/// A string attribute, or `None` when absent.
pub fn get_str(obj: &hdf5::Location, name: &str) -> Result<Option<String>> {
    Ok(read(obj, name)?.map(|v| v.as_str().unwrap_or_else(|| stringify_value(&v))))
}

/// A string-list attribute, or `None` when absent.
pub fn get_strs(obj: &hdf5::Location, name: &str) -> Result<Option<Vec<String>>> {
    Ok(read(obj, name)?.map(|v| v.as_str_list().unwrap_or_default()))
}

/// An integer-list attribute, or `None` when absent.
pub fn get_i64s(obj: &hdf5::Location, name: &str) -> Result<Option<Vec<i64>>> {
    Ok(read(obj, name)?.and_then(|v| v.as_i64_vec()))
}

/// A float-list attribute, or `None` when absent.
pub fn get_f64s(obj: &hdf5::Location, name: &str) -> Result<Option<Vec<f64>>> {
    Ok(read(obj, name)?.and_then(|v| v.as_f64_vec()))
}

/// A scalar integer attribute, or `None` when absent.
pub fn get_i64(obj: &hdf5::Location, name: &str) -> Result<Option<i64>> {
    Ok(read(obj, name)?.and_then(|v| v.as_i64()))
}

/// A scalar float attribute, or `None` when absent.
pub fn get_f64(obj: &hdf5::Location, name: &str) -> Result<Option<f64>> {
    Ok(read(obj, name)?.and_then(|v| v.as_f64()))
}

/// A scalar boolean attribute, or `None` when absent.
pub fn get_bool(obj: &hdf5::Location, name: &str) -> Result<Option<bool>> {
    Ok(read(obj, name)?.and_then(|v| v.as_bool()))
}

/// Python's `str()` of an attribute value that is not a string.
pub fn stringify_value(value: &AttrValue) -> String {
    match value {
        AttrValue::Str(s) => s.clone(),
        AttrValue::Int(i) => i.to_string(),
        AttrValue::Float(f) => crate::json::py_float(*f),
        AttrValue::Bool(b) => {
            if *b {
                "True".into()
            } else {
                "False".into()
            }
        }
        other => crate::json::py_str(&other.to_json()),
    }
}

/// The name of the attribute's object, for messages.
pub fn object_name(obj: &hdf5::Location) -> String {
    obj.name()
}

/// Fetch a required attribute, raising a coded validation error when absent.
pub fn require(obj: &hdf5::Location, name: &str, code: &str) -> Result<AttrValue> {
    read(obj, name)?
        .ok_or_else(|| Error::coded(code, format!("{}: required attribute {} is missing", obj.name(), repr_str(name))))
}

/// Copy one attribute byte-for-byte, keeping its stored type and shape.
///
/// An amend copies the root attributes it does not manage this way, so a
/// value a later minor version wrote (§16) survives exactly as written ---
/// an `int32` stays `int32`, a fixed-length string stays fixed-length.
pub fn copy_raw(src: &hdf5::Location, dst: &hdf5::Location, name: &str) -> Result<()> {
    use crate::h5sys::{h5a, h5p, h5s, h5t};
    let cname = std::ffi::CString::new(name).map_err(|e| Error::Value(e.to_string()))?;
    super::locked(|| unsafe {
        let attr = h5a::H5Aopen(src.id(), cname.as_ptr(), h5p::H5P_DEFAULT);
        if attr < 0 {
            return Err(Error::Io(format!("could not open attribute {}", repr_str(name))));
        }
        let ftype = h5a::H5Aget_type(attr);
        let space = h5a::H5Aget_space(attr);
        let size = h5t::H5Tget_size(ftype);
        let npoints = h5s::H5Sget_simple_extent_npoints(space).max(1) as usize;
        let mut buf = vec![0u8; size * npoints];
        let mut status = h5a::H5Aread(attr, ftype, buf.as_mut_ptr().cast());
        if status >= 0 {
            if h5a::H5Aexists(dst.id(), cname.as_ptr()) > 0 {
                h5a::H5Adelete(dst.id(), cname.as_ptr());
            }
            let out = h5a::H5Acreate2(dst.id(), cname.as_ptr(), ftype, space, h5p::H5P_DEFAULT, h5p::H5P_DEFAULT);
            status = if out < 0 { -1 } else { h5a::H5Awrite(out, ftype, buf.as_ptr().cast()) };
            if out >= 0 {
                h5a::H5Aclose(out);
            }
            if h5t::H5Tdetect_class(ftype, h5t::H5T_class_t::H5T_VLEN) > 0 || h5t::H5Tis_variable_str(ftype) > 0 {
                h5t::H5Treclaim(ftype, space, h5p::H5P_DEFAULT, buf.as_mut_ptr().cast());
            }
        }
        h5s::H5Sclose(space);
        h5t::H5Tclose(ftype);
        h5a::H5Aclose(attr);
        if status < 0 {
            return Err(Error::Io(format!("could not copy attribute {}", repr_str(name))));
        }
        Ok(())
    })
}
