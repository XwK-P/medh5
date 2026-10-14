//! Deprecation warnings raised from the bindings: the classes and functions
//! `_core` binds directly (`SampleWriter`, `create`, `amend`) have no Python
//! body to warn from.  The same rules as `medh5/_deprecation.py`: every message
//! starts with [`PREFIX`] and says what to use instead; `DeprecationWarning`
//! for a name or parameter 3.0 removes or renames, `FutureWarning` for a
//! default whose result 3.0 changes.

use std::ffi::CString;

use pyo3::exceptions::{PyDeprecationWarning, PyFutureWarning, PyValueError};
use pyo3::prelude::*;

/// What every deprecation message starts with; `medh5/_deprecation.py`'s
/// `PREFIX` (a test compares the two).
pub(crate) const PREFIX: &str = "medh5 3.0 ";

/// Warn about something 3.0 removes (`future: false`) or changes the result
/// of (`future: true`), pointing at the Python code that called the binding.
/// Under `-W error` the warning is the returned error.
// The first deprecation of something `_core` binds directly is its first
// caller; until then nothing does.
#[allow(dead_code)]
pub(crate) fn warn(py: Python<'_>, message: &str, future: bool) -> PyResult<()> {
    if !message.starts_with(PREFIX) {
        return Err(PyValueError::new_err(format!("a deprecation message starts with {PREFIX:?}: {message:?}")));
    }
    let message = CString::new(message).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let category = if future { py.get_type::<PyFutureWarning>() } else { py.get_type::<PyDeprecationWarning>() };
    PyErr::warn(py, category.as_any(), &message, 1)
}
