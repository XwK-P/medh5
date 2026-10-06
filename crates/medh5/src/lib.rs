//! The MEDH5 format engine.
//!
//! A `.medh5` file is **one subject at one or more timepoints**, with every
//! image, annotation, transform and curation record about them.  This crate is
//! the canonical implementation of the format (`docs/spec/medh5-1.0.md`): the
//! data model, HDF5 I/O, validation, chunked access, compression, geometry and
//! transforms, annotations, provenance and integrity.  The Python package and
//! the `medh5` command line are frontends over it.

pub mod annotations;
pub mod codes;
pub mod curation;
pub mod digest;
pub mod document;
pub mod error;
pub mod ids;
pub mod json;
pub mod labels;
pub mod array;
pub mod h5;
pub mod integrity;
pub mod geometry;
pub mod storage;
pub mod transforms;
pub mod validate;
pub mod pyval;
pub mod rng;
pub mod sample;

pub use error::{Error, Result};

/// Raw HDF5 C bindings, for the operations the high-level API lacks.
pub(crate) use medh5_sys::hdf5_sys as h5sys;

/// The package version, stamped into every file's `generator` attribute.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
/// The MEDH5 format version this engine reads and writes.
pub const FORMAT_VERSION: &str = "1.0";
