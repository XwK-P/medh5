//! The conformance corpus and the suite that publishes it (spec §15).
//!
//! Every case is a file plus the exact set of diagnostic codes a conforming
//! validator must emit for it.  Valid cases prove the format is writable;
//! invalid cases --- one per error code, built by mutating a valid file ---
//! prove the validator catches what the spec says it must.  The corpus is a
//! **shipped artifact**: [`build_corpus`] writes the files and an
//! `expected.json`, [`run_corpus`] checks this validator against it,
//! [`publish`] writes the distributable suite and [`score`] measures an
//! implementation that is not this one, in any language, from the codes it
//! reports back.
//!
//! Voxel data is drawn from [`SeededRng`](crate::rng::SeededRng), which is
//! NumPy's generator, so the corpus holds the same values the 1.x corpus did.

mod build;
pub mod suite;

use std::collections::BTreeSet;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use serde_json::{json, Map, Value};

use crate::json::repr_str;
use crate::validate::validate_file;
use crate::{Error, Result};

pub use suite::{check_checksums, load_manifest, publish, score, summarize, CHECKSUMS, SCHEMA};

/// How a case's file is made.
pub type Build = Arc<dyn Fn(&Path) -> Result<()> + Send + Sync>;

/// One corpus entry.
#[derive(Clone)]
pub struct Case {
    pub name: String,
    pub description: String,
    pub clause: String,
    pub build: Option<Build>,
    /// The validation level the case is checked at.
    pub level: String,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    /// `.medh5c` for a collection case (§2.1).
    pub suffix: String,
    /// Built by editing a committed file, so its digests are deliberately
    /// stale: "no expected errors" does not mean "this file also verifies".
    pub mutated: bool,
}

impl fmt::Debug for Case {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Case")
            .field("name", &self.name)
            .field("clause", &self.clause)
            .field("level", &self.level)
            .field("errors", &self.errors)
            .field("warnings", &self.warnings)
            .field("suffix", &self.suffix)
            .field("mutated", &self.mutated)
            .finish()
    }
}

impl Case {
    pub fn valid(&self) -> bool {
        self.errors.is_empty()
    }

    pub fn to_json(&self) -> Value {
        let sorted = |v: &[String]| -> Vec<String> {
            let mut s = v.to_vec();
            s.sort();
            s
        };
        json!({
            "name": self.name,
            "description": self.description,
            "clause": self.clause,
            "level": self.level,
            "file_suffix": self.suffix,
            "valid": self.valid(),
            "mutated": self.mutated,
            "expect_errors": sorted(&self.errors),
            "expect_warnings": sorted(&self.warnings),
        })
    }

    /// A case standing in for a manifest entry this build does not know, so
    /// scoring a suite published by a newer medh5 reports on its cases.
    pub fn from_record(record: &Value) -> Case {
        let text = |k: &str| record.get(k).map(crate::json::py_str).unwrap_or_default();
        let list = |k: &str| -> Vec<String> {
            record
                .get(k)
                .and_then(Value::as_array)
                .map(|a| a.iter().map(crate::json::py_str).collect())
                .unwrap_or_default()
        };
        Case {
            name: text("name"),
            description: text("description"),
            clause: text("clause"),
            build: None,
            level: record.get("level").and_then(Value::as_str).unwrap_or("semantic").to_string(),
            errors: list("expect_errors"),
            warnings: list("expect_warnings"),
            suffix: record.get("file_suffix").and_then(Value::as_str).unwrap_or(".medh5").to_string(),
            mutated: record.get("mutated").and_then(Value::as_bool).unwrap_or(false),
        }
    }
}

/// What running one case produced.
#[derive(Debug, Clone)]
pub struct CaseResult {
    pub case: Case,
    pub path: String,
    pub got_errors: Vec<String>,
    pub got_warnings: Vec<String>,
    pub missing: Vec<String>,
    pub unexpected: Vec<String>,
    pub error: Option<String>,
    pub details: Vec<String>,
}

impl CaseResult {
    pub fn new(case: Case, path: String) -> CaseResult {
        CaseResult {
            case,
            path,
            got_errors: Vec::new(),
            got_warnings: Vec::new(),
            missing: Vec::new(),
            unexpected: Vec::new(),
            error: None,
            details: Vec::new(),
        }
    }

    pub fn ok(&self) -> bool {
        self.missing.is_empty() && self.unexpected.is_empty() && self.error.is_none()
    }

    pub fn to_json(&self) -> Value {
        let sorted = |v: &[String]| -> Vec<String> {
            let mut s = v.to_vec();
            s.sort();
            s
        };
        json!({
            "name": self.case.name,
            "ok": self.ok(),
            "expect_errors": sorted(&self.case.errors),
            "got_errors": sorted(&self.got_errors),
            "expect_warnings": sorted(&self.case.warnings),
            "got_warnings": sorted(&self.got_warnings),
            "missing": sorted(&self.missing),
            "unexpected": sorted(&self.unexpected),
            "error": self.error,
        })
    }

    /// Fill in what a validator reported against what the case expects.
    pub fn compare(&mut self, errors: BTreeSet<String>, warnings: BTreeSet<String>) {
        let expected: BTreeSet<String> = self.case.errors.iter().chain(&self.case.warnings).cloned().collect();
        let got: BTreeSet<String> = errors.union(&warnings).cloned().collect();
        self.got_errors = errors.into_iter().collect();
        self.got_warnings = warnings.into_iter().collect();
        self.missing = expected.difference(&got).cloned().collect();
        self.unexpected = got.difference(&expected).cloned().collect();
    }
}

/// Every case, in the order the corpus lists them.
pub fn cases() -> &'static [Case] {
    static CASES: OnceLock<Vec<Case>> = OnceLock::new();
    CASES.get_or_init(build::registry)
}

pub fn case_by_name(name: &str) -> Result<&'static Case> {
    cases()
        .iter()
        .find(|c| c.name == name)
        .ok_or_else(|| Error::Key(format!("unknown conformance case {}", repr_str(name))))
}

fn selected(names: Option<&[String]>) -> Vec<&'static Case> {
    cases().iter().filter(|c| names.is_none_or(|n| n.contains(&c.name))).collect()
}

/// Write every case and an `expected.json` manifest beside them.
pub fn build_corpus(outdir: &Path, names: Option<&[String]>) -> Result<PathBuf> {
    std::fs::create_dir_all(outdir)?;
    let mut records = Vec::new();
    for entry in selected(names) {
        let file = format!("{}{}", entry.name, entry.suffix);
        let path = outdir.join(&file);
        if path.exists() {
            std::fs::remove_file(&path)?;
        }
        if let Some(build) = &entry.build {
            build(&path)?;
        }
        let mut record = match entry.to_json() {
            Value::Object(m) => m,
            _ => Map::new(),
        };
        record.insert("file".into(), json!(file));
        records.push(Value::Object(record));
    }
    let manifest = json!({
        "format": crate::FORMAT_VERSION,
        "generator": format!("medh5 {}", crate::VERSION),
        "cases": records,
    });
    let target = outdir.join("expected.json");
    std::fs::write(&target, crate::json::pretty(&manifest) + "\n")?;
    Ok(target)
}

/// Build every case, validate it, and compare against its expected codes.
pub fn run_corpus(outdir: &Path, names: Option<&[String]>) -> Result<Vec<CaseResult>> {
    build_corpus(outdir, names)?;
    let mut results = Vec::new();
    for entry in selected(names) {
        let path = outdir.join(format!("{}{}", entry.name, entry.suffix));
        let mut result = CaseResult::new(entry.clone(), crate::pyval::py_path(&path));
        match validate_file(&path, &entry.level, None) {
            Err(e) => result.error = Some(e.python_line()),
            Ok(report) => {
                let errors = report.errors().iter().map(|d| d.code.clone()).collect();
                let warnings = report.warnings().iter().map(|d| d.code.clone()).collect();
                result.compare(errors, warnings);
                result.details = report.diagnostics.iter().map(|d| d.line()).collect();
            }
        }
        results.push(result);
    }
    Ok(results)
}
