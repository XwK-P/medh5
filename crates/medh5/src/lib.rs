#![doc = include_str!("../README.md")]

pub mod annotations;
pub mod array;
pub mod bench;
pub mod codes;
pub mod collection;
pub mod conformance;
pub mod convert;
pub mod curation;
pub mod dataset;
pub mod digest;
pub mod document;
pub mod error;
pub mod geometry;
pub mod h5;
pub mod ids;
pub mod integrity;
pub mod json;
pub mod labels;
pub mod numeric;
pub mod pyval;
pub mod rng;
pub mod sample;
pub mod sampling;
pub mod storage;
pub mod transforms;
pub mod validate;

pub use error::{Error, Result};
/// The HDF5 bindings the engine is built on, for frontends that need a handle.
pub use hdf5;
/// The array crate the engine's arrays convert to and from ([`array::NdArray`]).
pub use ndarray;

/// Raw HDF5 C bindings, for the operations the high-level API lacks.
pub(crate) use medh5_sys::hdf5_sys as h5sys;

/// The package version, stamped into every file's `generator` attribute.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
/// The MEDH5 format version this engine reads and writes.
pub const FORMAT_VERSION: &str = "1.0";
