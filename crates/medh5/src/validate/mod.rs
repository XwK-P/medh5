//! Validation (spec §15).  (Being ported.)

use std::path::Path;

/// One finding.
#[derive(Debug, Clone, PartialEq)]
pub struct Diagnostic {
    pub code: String,
    pub severity: String,
    pub location: String,
    pub message: String,
}

/// A validation report.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Report {
    pub diagnostics: Vec<Diagnostic>,
}

/// Validate a sample root.
pub fn validate_root(_root: &hdf5::Group, _path: Option<&Path>, _level: &str, _errors_only: bool) -> crate::Result<Report> {
    Ok(Report::default())
}
