//! Small rule helpers the Python modules use: collection member keys (§2.2),
//! image value types (§4.2) and the checks converters ask before writing.

use pyo3::prelude::*;

use crate::convert::py_to_nd;
use crate::errors::R;

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
    m.add_function(wrap_pyfunction!(validate_sample_key, m)?)?;
    m.add_function(wrap_pyfunction!(check_value_type, m)?)?;
    m.add_function(wrap_pyfunction!(lossless_as_int16, m)?)?;
    m.add_function(wrap_pyfunction!(is_probability, m)?)?;
    Ok(())
}
