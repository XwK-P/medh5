//! Small rule helpers the Python modules re-export: identifiers (§2.3),
//! image value types (§4.2) and the checks converters ask before writing.

use pyo3::prelude::*;

use crate::convert::py_to_nd;
use crate::errors::R;

/// An identifier matching `[A-Za-z0-9_.-]{1,128}`, or E003.
#[pyfunction]
#[pyo3(signature = (name, *, what="identifier"))]
fn validate_id(name: &str, what: &str) -> R<String> {
    Ok(medh5::ids::validate_id(name, what)?.to_string())
}

#[pyfunction]
fn is_valid_id(name: &str) -> bool {
    medh5::ids::is_valid_id(name)
}

/// A collection member key (§2.2), or E003.
#[pyfunction]
fn validate_sample_key(name: &str) -> R<String> {
    Ok(medh5::ids::validate_sample_key(name)?.to_string())
}

#[pyfunction]
fn check_value_type(value_type: &str) -> R<String> {
    medh5::sample::image::check_value_type(value_type)?;
    Ok(value_type.to_string())
}

/// Whether a float array would survive `int16` storage unchanged (W907).
#[pyfunction]
fn lossless_as_int16(array: &Bound<'_, PyAny>) -> R<bool> {
    Ok(medh5::sample::image::lossless_as_int16(&py_to_nd(array)?))
}

/// Whether every value lies in `[0, 1]` (and there is at least one).
#[pyfunction]
fn is_probability(array: &Bound<'_, PyAny>) -> R<bool> {
    Ok(medh5::sample::image::is_probability(&py_to_nd(array)?))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(validate_id, m)?)?;
    m.add_function(wrap_pyfunction!(is_valid_id, m)?)?;
    m.add_function(wrap_pyfunction!(validate_sample_key, m)?)?;
    m.add_function(wrap_pyfunction!(check_value_type, m)?)?;
    m.add_function(wrap_pyfunction!(lossless_as_int16, m)?)?;
    m.add_function(wrap_pyfunction!(is_probability, m)?)?;
    m.add("RESERVED_IDS", pyo3::types::PyTuple::new(m.py(), medh5::ids::RESERVED_IDS)?)?;
    Ok(())
}
