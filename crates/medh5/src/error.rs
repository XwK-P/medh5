//! Errors raised by the engine.
//!
//! The variants mirror the exception hierarchy every frontend exposes, so a
//! failure means the same thing to a Rust caller, a Python caller and a shell
//! script: the Python bindings map each variant onto the exception class of the
//! same name, and the CLI prints the message and exits 1.
//!
//! [`Error::Validation`] carries the stable diagnostic code (spec §15.2) when
//! one applies, so callers can branch on an identifier rather than on text.

use std::fmt;

/// The engine's error type.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// A file cannot be opened, is not HDF5, or is structurally unreadable.
    File(String),
    /// The file declares a `medh5_version` major this reader does not implement.
    Version(String),
    /// The sample document is absent, is not JSON, or violates its schema.
    Schema(String),
    /// Input rejected by a writer, or a validation failure raised as an error.
    Validation {
        /// The message without the `[CODE]` prefix.
        message: String,
        /// The diagnostic code (spec §15.2), when one applies.
        code: Option<String>,
    },
    /// A digest or `content_id` does not match the data it covers.
    Integrity(String),
    /// A name the file or a collection does not have (Python `KeyError`).
    Key(String),
    /// A value outside what an operation accepts (Python `ValueError`).
    Value(String),
    /// An argument of the wrong kind (Python `TypeError`).
    Type(String),
    /// A position outside a sequence (Python `IndexError`).
    Index(String),
    /// An operating-system or HDF5 I/O failure (Python `OSError`).
    Io(String),
    /// An operation the engine does not implement (Python `NotImplementedError`).
    NotImplemented(String),
    /// A failure with no better home (Python `RuntimeError`).
    Runtime(String),
}

/// Shorthand for results in this crate.
pub type Result<T, E = Error> = std::result::Result<T, E>;

impl Error {
    /// A coded validation error: `[code] message`.
    pub fn coded(code: &str, message: impl Into<String>) -> Self {
        Error::Validation { message: message.into(), code: Some(code.to_string()) }
    }

    /// An uncoded validation error.
    pub fn invalid(message: impl Into<String>) -> Self {
        Error::Validation { message: message.into(), code: None }
    }

    /// The diagnostic code, for a coded validation error.
    pub fn code(&self) -> Option<&str> {
        match self {
            Error::Validation { code, .. } => code.as_deref(),
            _ => None,
        }
    }

    /// The message alone, without a `[CODE]` prefix.
    pub fn message(&self) -> &str {
        match self {
            Error::File(m)
            | Error::Version(m)
            | Error::Schema(m)
            | Error::Integrity(m)
            | Error::Key(m)
            | Error::Value(m)
            | Error::Type(m)
            | Error::Index(m)
            | Error::Io(m)
            | Error::NotImplemented(m)
            | Error::Runtime(m) => m,
            Error::Validation { message, .. } => message,
        }
    }

    /// The name of the matching exception class in the Python bindings.
    pub fn kind_name(&self) -> &'static str {
        match self {
            Error::File(_) => "MEDH5FileError",
            Error::Version(_) => "MEDH5VersionError",
            Error::Schema(_) => "MEDH5SchemaError",
            Error::Validation { .. } => "MEDH5ValidationError",
            Error::Integrity(_) => "MEDH5IntegrityError",
            Error::Key(_) => "KeyError",
            Error::Value(_) => "ValueError",
            Error::Type(_) => "TypeError",
            Error::Index(_) => "IndexError",
            Error::Io(_) => "OSError",
            Error::NotImplemented(_) => "NotImplementedError",
            Error::Runtime(_) => "RuntimeError",
        }
    }

    /// What Python's `str(exc)` prints for the matching exception: a
    /// `KeyError` quotes its message, a coded validation error leads with its
    /// code, everything else is the message.
    pub fn python_str(&self) -> String {
        match self {
            Error::Key(m) => crate::json::repr_str(m),
            other => other.to_string(),
        }
    }

    /// `ExceptionName: message`, as Python reports a caught exception.
    pub fn python_line(&self) -> String {
        format!("{}: {}", self.kind_name(), self.python_str())
    }

    /// Whether this is one of the package's own `MEDH5Error` family.
    pub fn is_medh5(&self) -> bool {
        matches!(
            self,
            Error::File(_) | Error::Version(_) | Error::Schema(_) | Error::Validation { .. } | Error::Integrity(_)
        )
    }

    /// Whether this is a lookup failure (`KeyError`/`IndexError`).
    pub fn is_lookup(&self) -> bool {
        matches!(self, Error::Key(_) | Error::Index(_))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Validation { message, code: Some(code) } => write!(f, "[{code}] {message}"),
            other => f.write_str(other.message()),
        }
    }
}

impl std::error::Error for Error {}

impl From<hdf5::Error> for Error {
    fn from(err: hdf5::Error) -> Self {
        Error::Io(err.to_string())
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Io(err.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Error::Value(err.to_string())
    }
}

impl From<ndarray::ShapeError> for Error {
    fn from(err: ndarray::ShapeError) -> Self {
        Error::Value(err.to_string())
    }
}

/// Return early with a coded validation error.
#[macro_export]
macro_rules! bail_coded {
    ($code:expr, $($arg:tt)*) => {
        return Err($crate::Error::coded($code, format!($($arg)*)))
    };
}

/// Return early with an uncoded validation error.
#[macro_export]
macro_rules! bail_invalid {
    ($($arg:tt)*) => {
        return Err($crate::Error::invalid(format!($($arg)*)))
    };
}
