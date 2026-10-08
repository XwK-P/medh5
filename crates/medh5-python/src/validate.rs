//! `medh5.validate`: the validator (spec §15) and the code table.
//!
//! Reports cross as their JSON form; `medh5/validate.py` turns them into
//! the `Report` and `Diagnostic` dataclasses 1.x returned.

use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3::types::PyList;

use medh5::validate as engine;

use crate::convert::{json_to_py, opt_strings};
use crate::errors::R;
use crate::reader::SampleHandle;

/// The normative diagnostic code table (§15.2), as JSON text.
#[pyfunction]
fn codes_table() -> &'static str {
    medh5::codes::table_json()
}

#[pyfunction]
#[pyo3(signature = (path, *, level="semantic", profiles=None))]
fn validate_file<'py>(
    py: Python<'py>,
    path: PathBuf,
    level: &str,
    profiles: Option<&Bound<'py, PyAny>>,
) -> R<Bound<'py, PyAny>> {
    let profiles = opt_strings(profiles)?;
    let level = level.to_string();
    let report = py.detach(move || engine::validate_file(&path, &level, profiles.as_deref()))?;
    Ok(json_to_py(py, &report.to_json())?)
}

#[pyfunction]
#[pyo3(signature = (paths, *, level="semantic", profiles=None))]
fn validate_paths<'py>(
    py: Python<'py>,
    paths: Vec<PathBuf>,
    level: &str,
    profiles: Option<&Bound<'py, PyAny>>,
) -> R<Bound<'py, PyList>> {
    let profiles = opt_strings(profiles)?;
    let level = level.to_string();
    let reports = py.detach(move || {
        let refs: Vec<&std::path::Path> = paths.iter().map(PathBuf::as_path).collect();
        engine::validate_paths(&refs, &level, profiles.as_deref())
    })?;
    let out = PyList::empty(py);
    for r in reports {
        out.append(json_to_py(py, &r.to_json())?)?;
    }
    Ok(out)
}

/// Validate an open sample (`Sample`'s handle), with `errors_only` skipping
/// the warning-only checks that read bulk data.
#[pyfunction]
#[pyo3(signature = (sample, *, path=None, level="semantic", profiles=None, errors_only=false))]
fn validate_root<'py>(
    py: Python<'py>,
    sample: &Bound<'py, PyAny>,
    path: Option<String>,
    level: &str,
    profiles: Option<&Bound<'py, PyAny>>,
    errors_only: bool,
) -> R<Bound<'py, PyAny>> {
    if !engine::LEVELS.contains(&level) {
        return Err(medh5::Error::Value(format!(
            "unknown validation level {}; expected one of {}",
            medh5::json::repr_str(level),
            medh5::json::repr_list(&engine::LEVELS).replacen('[', "(", 1).replacen(']', ")", 1)
        ))
        .into());
    }
    let handle = match sample.cast::<SampleHandle>() {
        Ok(h) => h.clone(),
        Err(_) => sample.getattr("_handle")?.cast_into::<SampleHandle>().map_err(PyErr::from)?,
    };
    let engine_sample = handle.get().sample()?;
    let profiles = opt_strings(profiles)?;
    let shown = path.or_else(|| engine_sample.path.as_ref().map(|p| p.to_string_lossy().into_owned()));
    let shown = shown.unwrap_or_else(|| "<memory>".into());
    let level = level.to_string();
    let report = py.detach(move || {
        let names = if level == "integrity" || level == "strict" {
            medh5::sample::attr_name_map_of(&engine_sample.root).ok()
        } else {
            None
        };
        engine::validate_root_with(
            &engine_sample.root,
            &shown,
            &level,
            profiles.as_deref(),
            names.as_ref(),
            errors_only,
        )
    })?;
    Ok(json_to_py(py, &report.to_json())?)
}

/// Every rule run at `level`, in order.
#[pyfunction]
fn rules_for(level: &str) -> Vec<&'static str> {
    engine::rules::rules_for(level).into_iter().map(|(n, _)| n).collect()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(codes_table, m)?)?;
    m.add_function(wrap_pyfunction!(validate_file, m)?)?;
    m.add_function(wrap_pyfunction!(validate_paths, m)?)?;
    m.add_function(wrap_pyfunction!(validate_root, m)?)?;
    m.add_function(wrap_pyfunction!(rules_for, m)?)?;
    m.add("LEVELS", pyo3::types::PyTuple::new(m.py(), engine::LEVELS)?)?;
    Ok(())
}
