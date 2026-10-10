//! Validation (spec §15).
//!
//! Four levels, each a superset of the last:
//!
//! - `structural` --- layout, required attributes, dtypes, shapes,
//!   identifier syntax, JSON Schema;
//! - `semantic` --- cross-references resolve, geometry consistency, class ids
//!   in the label set, encoding invariants, profile requirements;
//! - `integrity` --- per-object digests, `content_id`, index currency;
//! - `strict` --- all of the above, with warnings promoted to failures.
//!
//! A validation pass never fails on a bad file --- it reports.  Curation needs
//! to see everything wrong with a file at once.

pub mod rules;

use std::path::Path;

use serde_json::{json, Value};

use crate::h5::attrs;
use crate::h5::ops;
use crate::integrity::AttrNameMap;
use crate::json::repr_str;
use crate::{Error, Result};

/// The validation levels, in order.
pub const LEVELS: [&str; 4] = ["structural", "semantic", "integrity", "strict"];

/// The group a collection keeps its sample roots in (§2.2).
pub const SAMPLES_GROUP: &str = "samples";

/// One finding.
#[derive(Debug, Clone, PartialEq)]
pub struct Diagnostic {
    pub code: String,
    pub location: String,
    pub message: String,
    pub severity: String,
    pub level: String,
}

impl Diagnostic {
    /// The code table's one-line summary.
    pub fn summary(&self) -> &'static str {
        crate::codes::get(&self.code).map(|c| c.summary.as_str()).unwrap_or("")
    }

    pub fn to_json(&self) -> Value {
        json!({
            "code": self.code,
            "severity": self.severity,
            "location": self.location,
            "message": self.message,
            "level": self.level,
        })
    }

    /// `SEVERITY CODE location: message`, severity padded to seven columns.
    pub fn line(&self) -> String {
        format!("{:7} {} {}: {}", self.severity.to_uppercase(), self.code, self.location, self.message)
    }
}

/// Everything a validation pass found.
#[derive(Debug, Clone, PartialEq)]
pub struct Report {
    pub path: String,
    pub level: String,
    pub profiles: Vec<String>,
    pub diagnostics: Vec<Diagnostic>,
    pub checked: Value,
}

impl Report {
    pub fn new(path: &str, level: &str) -> Report {
        Report {
            path: path.into(),
            level: level.into(),
            profiles: Vec::new(),
            diagnostics: Vec::new(),
            checked: json!({}),
        }
    }

    /// Severity at this level: `strict` promotes every warning to an error.
    fn effective(&self, d: &Diagnostic) -> String {
        if self.level == "strict" {
            "error".into()
        } else {
            d.severity.clone()
        }
    }

    pub fn errors(&self) -> Vec<&Diagnostic> {
        self.diagnostics.iter().filter(|d| self.effective(d) == "error").collect()
    }

    pub fn warnings(&self) -> Vec<&Diagnostic> {
        self.diagnostics.iter().filter(|d| self.effective(d) == "warning").collect()
    }

    /// Diagnostics the table calls warnings and this level treats as errors.
    pub fn promoted(&self) -> Vec<&Diagnostic> {
        if self.level != "strict" {
            return Vec::new();
        }
        self.diagnostics.iter().filter(|d| d.severity == "warning").collect()
    }

    /// Every code emitted, sorted and deduplicated --- the conformance key.
    pub fn codes(&self) -> Vec<String> {
        let mut out: Vec<String> = self.diagnostics.iter().map(|d| d.code.clone()).collect();
        out.sort();
        out.dedup();
        out
    }

    /// Whether the file conforms at this level.
    pub fn ok(&self) -> bool {
        self.errors().is_empty()
    }

    pub fn to_json(&self) -> Value {
        json!({
            "path": self.path,
            "level": self.level,
            "profiles": self.profiles,
            "ok": self.ok(),
            "errors": self.errors().len(),
            "warnings": self.warnings().len(),
            "promoted": self.promoted().len(),
            "diagnostics": self.diagnostics.iter().map(Diagnostic::to_json).collect::<Vec<_>>(),
            "checked": self.checked,
        })
    }

    /// Human-readable text, one diagnostic per line.
    pub fn format(&self, verbose: bool) -> String {
        let profiles = if self.profiles.is_empty() { "-".to_string() } else { self.profiles.join(",") };
        let mut lines = vec![format!(
            "{}: {} [{}] profiles={profiles} ({} errors, {} warnings)",
            self.path,
            if self.ok() { "OK" } else { "FAILED" },
            self.level,
            self.errors().len(),
            self.warnings().len()
        )];
        for d in &self.diagnostics {
            lines.push(format!("  {}", d.line()));
            let summary = d.summary();
            if verbose && !summary.is_empty() {
                lines.push(format!("          -> {summary}"));
            }
        }
        lines.join("\n")
    }

    /// Python's `repr()`.
    pub fn repr(&self) -> String {
        format!(
            "Report({}, ok={}, {} errors, {} warnings)",
            repr_str(&self.path),
            if self.ok() { "True" } else { "False" },
            self.errors().len(),
            self.warnings().len()
        )
    }

    fn sort(&mut self) {
        self.diagnostics.sort_by(|a, b| {
            ((a.severity != "error"), &a.code, &a.location).cmp(&((b.severity != "error"), &b.code, &b.location))
        });
    }
}

/// Combine several reports, prefixing each location with its file.
pub fn merge(reports: &[Report], path: &str) -> Report {
    let mut out = Report::new(path, reports.first().map(|r| r.level.as_str()).unwrap_or("structural"));
    for report in reports {
        for d in &report.diagnostics {
            out.diagnostics.push(Diagnostic { location: format!("{}:{}", report.path, d.location), ..d.clone() });
        }
    }
    out
}

fn check_level(level: &str) -> Result<()> {
    if !LEVELS.contains(&level) {
        return Err(Error::Value(format!(
            "unknown validation level {}; expected one of {}",
            repr_str(level),
            crate::json::repr_list(&LEVELS).replacen('[', "(", 1).replacen(']', ")", 1)
        )));
    }
    Ok(())
}

/// Validate an open sample root.
///
/// `errors_only` skips the warning-only checks that read bulk data --- what
/// `SampleWriter::commit` asks for.
pub fn validate_root_with(
    root: &hdf5::Group,
    path: &str,
    level: &str,
    profiles: Option<&[String]>,
    attr_names: Option<&AttrNameMap>,
    errors_only: bool,
) -> Result<Report> {
    check_level(level)?;
    let declared: Vec<String> = match profiles {
        Some(p) => p.to_vec(),
        None => attrs::get_strs(root, "medh5_profiles").ok().flatten().unwrap_or_default(),
    };
    let mut ctx = rules::Context::new(root.clone(), path, level, declared.clone(), errors_only);
    ctx.attr_names = attr_names.cloned();
    let mut report = Report::new(path, level);
    report.profiles = declared;
    let rules = rules::rules_for(level);
    for (name, rule) in &rules {
        match rule(&mut ctx) {
            Ok(found) => report.diagnostics.extend(found),
            Err(e) => report.diagnostics.push(Diagnostic {
                code: "E001".into(),
                location: "/".into(),
                message: format!("{name} could not read the file: {}", e.python_line()),
                severity: "error".into(),
                level: level.into(),
            }),
        }
    }
    report.checked = json!({
        "rules": rules.iter().map(|(n, _)| *n).collect::<Vec<_>>(),
        "schema_checked": ctx.schema_checked,
    });
    report.sort();
    Ok(report)
}

/// Validate an open sample root at `level` (what the writer's commit gate
/// calls with `errors_only`).
pub fn validate_root(root: &hdf5::Group, path: Option<&Path>, level: &str, errors_only: bool) -> Result<Report> {
    let shown = path.map(|p| p.to_string_lossy().into_owned()).unwrap_or_else(|| "<memory>".into());
    validate_root_with(root, &shown, level, None, None, errors_only)
}

/// The digest attribute map, needed only where integrity is checked.
fn attr_names_for(root: &hdf5::Group, level: &str) -> Option<AttrNameMap> {
    if level != "integrity" && level != "strict" {
        return None;
    }
    crate::sample::attr_name_map_of(root).ok()
}

/// Validate a `collection` root and every sample in it (§2.2).
pub fn validate_collection(root: &hdf5::Group, path: &str, level: &str, profiles: Option<&[String]>) -> Result<Report> {
    check_level(level)?;
    let mut ctx = rules::Context::new(root.clone(), path, level, Vec::new(), false);
    let header = rules::check_collection(&mut ctx)?;
    let mut members = Vec::new();
    if let Some(node) = ops::child_group(root, SAMPLES_GROUP) {
        for key in ops::members(&node)? {
            if let Some(member) = ops::child_group(&node, &key) {
                let names = attr_names_for(&member, level);
                members.push(validate_root_with(
                    &member,
                    &format!("/{SAMPLES_GROUP}/{key}"),
                    level,
                    profiles,
                    names.as_ref(),
                    false,
                )?);
            }
        }
    }
    let mut combined = if members.is_empty() { Report::new(path, level) } else { merge(&members, path) };
    combined.level = level.into();
    let mut diagnostics = header;
    diagnostics.extend(combined.diagnostics);
    combined.diagnostics = diagnostics;
    combined.checked = json!({"samples": members.iter().map(|r| r.path.clone()).collect::<Vec<_>>()});
    combined.sort();
    Ok(combined)
}

fn read_failure(path: &str, level: &str, message: String) -> Report {
    let mut report = Report::new(path, level);
    report.diagnostics.push(Diagnostic {
        code: "E001".into(),
        location: "/".into(),
        message,
        severity: "error".into(),
        level: level.into(),
    });
    report
}

/// Validate one `.medh5` sample or `.medh5c` collection file.
pub fn validate_file(path: &Path, level: &str, profiles: Option<&[String]>) -> Result<Report> {
    check_level(level)?;
    let text = path.to_string_lossy().into_owned();
    let handle = match crate::h5::file::open_read(path) {
        Ok(h) => h,
        Err(e) => return Ok(read_failure(&text, level, e.to_string())),
    };
    let result = (|| -> Result<Report> {
        let root = handle.as_group()?;
        let kind = attrs::get_str(&root, "medh5_kind")?.unwrap_or_else(|| "sample".into());
        if kind == "collection" {
            return validate_collection(&root, &text, level, profiles);
        }
        let names = attr_names_for(&root, level);
        validate_root_with(&root, &text, level, profiles, names.as_ref(), false)
    })();
    match result {
        Ok(r) => Ok(r),
        Err(e @ Error::Value(_)) if e.message().starts_with("unknown validation level") => Err(e),
        Err(e) => Ok(read_failure(&text, level, format!("the file could not be read: {}", e.python_line()))),
    }
}

/// The clinical profile's errors in one sample root (E8xx, and the document
/// and declaration rules they rest on), without the bulk-data rules: what a
/// task's preflight asks of each source it will select from.
pub fn clinical_errors(root: &hdf5::Group, path: &str) -> Result<Vec<Diagnostic>> {
    Ok(clinical_checked(root, path)?.0)
}

/// [`clinical_errors`], and the tables the rules read --- document text
/// deferred --- so a caller that goes on to use the profile reads it once
/// ([`Clinical::from_tables`](crate::clinical::Clinical::from_tables)).
pub fn clinical_checked(root: &hdf5::Group, path: &str) -> Result<(Vec<Diagnostic>, Option<rules::ClinicalTables>)> {
    let declared = attrs::get_strs(root, "medh5_profiles")?.unwrap_or_default();
    let mut ctx = rules::Context::new(root.clone(), path, "semantic", declared, true);
    let mut out = Vec::new();
    for rule in [rules::check_document as rules::Rule, rules::check_clinical, rules::check_clinical_records] {
        out.extend(rule(&mut ctx)?);
    }
    out.retain(|d| d.severity == "error" && (d.code.starts_with("E8") || d.code == "E009" || d.code == "E004"));
    Ok((out, ctx.clinical.take()))
}

/// Validate many files, one report each.
pub fn validate_paths(paths: &[&Path], level: &str, profiles: Option<&[String]>) -> Result<Vec<Report>> {
    paths.iter().map(|p| validate_file(p, level, profiles)).collect()
}
