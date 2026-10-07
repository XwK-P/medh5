//! Engine errors as the Python exceptions 1.x raised.
//!
//! The `MEDH5Error` family lives in `medh5/errors.py` --- ordinary Python
//! classes, so they pickle, subclass and print like any exception --- and is
//! looked up here once.  Every other variant maps onto the built-in of the
//! same meaning (`KeyError`, `ValueError`, ...).

use pyo3::exceptions::{
    PyIndexError, PyKeyError, PyNotImplementedError, PyOSError, PyRuntimeError, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyType;

/// Why a binding call failed: the engine refused, or Python did.
pub enum BindError {
    Engine(medh5::Error),
    Py(PyErr),
}

/// The binding layer's result type; converts into a raised exception.
pub type R<T> = Result<T, BindError>;

impl From<medh5::Error> for BindError {
    fn from(e: medh5::Error) -> Self {
        BindError::Engine(e)
    }
}

impl From<PyErr> for BindError {
    fn from(e: PyErr) -> Self {
        BindError::Py(e)
    }
}

impl From<std::io::Error> for BindError {
    fn from(e: std::io::Error) -> Self {
        BindError::Engine(e.into())
    }
}

impl From<medh5::hdf5::Error> for BindError {
    fn from(e: medh5::hdf5::Error) -> Self {
        BindError::Engine(e.into())
    }
}

impl From<std::convert::Infallible> for BindError {
    fn from(e: std::convert::Infallible) -> Self {
        match e {}
    }
}

impl From<ndarray::ShapeError> for BindError {
    fn from(e: ndarray::ShapeError) -> Self {
        BindError::Engine(e.into())
    }
}

impl From<BindError> for PyErr {
    fn from(e: BindError) -> Self {
        match e {
            BindError::Py(err) => err,
            BindError::Engine(err) => engine_error(err),
        }
    }
}

fn medh5_class(py: Python<'_>, name: &str) -> PyResult<Py<PyType>> {
    static ERRORS: PyOnceLock<Py<PyModule>> = PyOnceLock::new();
    let module = ERRORS.get_or_try_init(py, || py.import("medh5.errors").map(|m| m.unbind()))?;
    Ok(module.bind(py).getattr(name)?.cast_into::<PyType>()?.unbind())
}

/// The exception a 1.x caller saw for this engine error.
pub fn engine_error(err: medh5::Error) -> PyErr {
    use medh5::Error as E;
    match err {
        E::Key(m) => PyKeyError::new_err(m),
        E::Value(m) => PyValueError::new_err(m),
        E::Type(m) => PyTypeError::new_err(m),
        E::Index(m) => PyIndexError::new_err(m),
        E::Io(m) => PyOSError::new_err(m),
        E::NotImplemented(m) => PyNotImplementedError::new_err(m),
        E::Runtime(m) => PyRuntimeError::new_err(m),
        family => Python::attach(|py| {
            let (name, args): (&str, (String, Option<String>)) = match family {
                E::File(m) => ("MEDH5FileError", (m, None)),
                E::Version(m) => ("MEDH5VersionError", (m, None)),
                E::Schema(m) => ("MEDH5SchemaError", (m, None)),
                E::Integrity(m) => ("MEDH5IntegrityError", (m, None)),
                E::Validation { message, code } => ("MEDH5ValidationError", (message, code)),
                _ => unreachable!("handled above"),
            };
            let class = match medh5_class(py, name) {
                Ok(c) => c,
                Err(e) => return e,
            };
            let made = if name == "MEDH5ValidationError" {
                class.bind(py).call1(args)
            } else {
                class.bind(py).call1((args.0,))
            };
            match made {
                Ok(instance) => PyErr::from_value(instance),
                Err(e) => e,
            }
        }),
    }
}

/// A Python exception as an engine error, for code that hands Python
/// failures back through engine callbacks (`Host`, `Rng`).
pub fn to_engine(py: Python<'_>, err: PyErr) -> medh5::Error {
    let value = err.value(py);
    let text = value.str().map(|s| s.to_string()).unwrap_or_default();
    let is =
        |name: &str| medh5_class(py, name).map(|c| value.is_instance(c.bind(py)).unwrap_or(false)).unwrap_or(false);
    if is("MEDH5ValidationError") {
        let code: Option<String> = value.getattr("code").ok().and_then(|c| c.extract().ok());
        let message: String = value.getattr("message").ok().and_then(|m| m.extract().ok()).unwrap_or(text);
        return medh5::Error::Validation { message, code };
    }
    if is("MEDH5FileError") {
        return medh5::Error::File(text);
    }
    if err.is_instance_of::<PyKeyError>(py) {
        let key = value.getattr("args").ok().and_then(|a| a.get_item(0).ok()).and_then(|k| k.extract().ok());
        return medh5::Error::Key(key.unwrap_or(text));
    }
    if err.is_instance_of::<PyTypeError>(py) {
        return medh5::Error::Type(text);
    }
    if err.is_instance_of::<PyValueError>(py) {
        return medh5::Error::Value(text);
    }
    if err.is_instance_of::<PyOSError>(py) {
        return medh5::Error::Io(text);
    }
    medh5::Error::Runtime(text)
}
