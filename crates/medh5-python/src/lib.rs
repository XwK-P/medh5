//! `medh5._core`: the format engine as a Python extension module.
//!
//! Everything format-native --- the data model and its rules, HDF5 I/O,
//! encodings, geometry, transforms, validation, integrity, the dataset and
//! curation tools, the conformance suite and the command line --- is the Rust
//! engine (`medh5` crate).  This module only translates: Python values in,
//! engine calls, Python values out.  The `medh5` Python package is built on
//! these names: each module re-exports what it serves as is (`medh5.geometry`,
//! `medh5.labels`, ...) and wraps the rest --- the `*Handle` classes behind
//! `Sample`, `Image`, `Annotation` and `Transform`.  A class's `__module__`
//! names the module that exports it; a handle, which none does, says
//! `medh5._core`.

use pyo3::prelude::*;

pub mod annotations;
pub mod cli;
pub mod clinical;
pub mod conformance;
pub mod convert;
pub mod curation;
pub mod curation_tools;
pub mod dataset;
mod deprecation;
pub mod document;
pub mod errors;
pub mod exclusive;
pub mod exclusive;
pub mod files;
pub mod geometry;
pub mod integrity;
pub mod io;
pub mod labels;
pub mod misc;
pub mod nodes;
pub mod preflight;
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

// `gil_used = true` is said out loud because PyO3 defaults to the opposite:
// without it, a build for free-threaded CPython (3.14t) declares the module
// `Py_MOD_GIL_NOT_USED` and runs it with no GIL at all.  Nothing here has been
// audited for that --- the mutable handles' borrow checks and the Python
// handle cache's check-then-act both assume one thread at a time --- so a
// free-threaded interpreter must turn the GIL back on when it imports us.
#[pymodule(gil_used = true)]
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
    clinical::register(m)?;
    preflight::register(m)?;
    files::register(m)?;
    Ok(())
}
