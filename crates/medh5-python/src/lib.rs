//! `medh5._core`: the format engine as a Python extension module.
//!
//! Everything format-native --- the data model and its rules, HDF5 I/O,
//! encodings, geometry, transforms, validation, integrity, the dataset and
//! curation tools, the conformance suite and the command line --- is the Rust
//! engine (`medh5` crate).  This module only translates: Python values in,
//! engine calls, Python values out.  The `medh5` Python package re-exports
//! these names from the module each belongs to (`medh5.geometry`,
//! `medh5.labels`, ...), which is also what every class's `__module__` names.

use pyo3::prelude::*;

pub mod annotations;
pub mod cli;
pub mod conformance;
pub mod convert;
pub mod curation;
pub mod curation_tools;
pub mod dataset;
pub mod document;
pub mod errors;
pub mod geometry;
pub mod integrity;
pub mod io;
pub mod labels;
pub mod misc;
pub mod nodes;
pub mod reader;
pub mod records;
pub mod rng;
pub mod sampling;
pub mod storage;
pub mod tracking;
pub mod transforms;
pub mod validate;
pub mod values;
pub mod writer;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", medh5::VERSION)?;
    m.add("__format_version__", medh5::FORMAT_VERSION)?;
    labels::register(m)?;
    geometry::register(m)?;
    curation::register(m)?;
    document::register(m)?;
    nodes::register(m)?;
    reader::register(m)?;
    writer::register(m)?;
    annotations::register(m)?;
    integrity::register(m)?;
    tracking::register(m)?;
    validate::register(m)?;
    misc::register(m)?;
    transforms::register(m)?;
    storage::register(m)?;
    cli::register(m)?;
    conformance::register(m)?;
    dataset::register(m)?;
    curation_tools::register(m)?;
    sampling::register(m)?;
    io::register(m)?;
    Ok(())
}
