//! Cohort-level checks: what is wrong *between* files, not inside one.
//!
//! `medh5 validate` answers "is this file legal"; every question that matters
//! for training is about the cohort --- do these files mean the same thing by
//! class 3, was this split computed before the cohort last changed, is one
//! subject in train and test.  Findings have the validator's shape, but the
//! codes are cohort codes (`C1xx`) and belong to this tool, not to the format.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use indexmap::IndexMap;
use serde_json::{json, Map, Value};

use crate::collection::{open_any, AnyFile};
use crate::curation::splits::anatomy_units;
use crate::dataset::manifest::{Entry, Manifest};
use crate::json::repr_int_list;

pub const SEVERITIES: [&str; 3] = ["error", "warning", "info"];

/// Cohort codes.  Distinct from the format's E/W table on purpose (§15.2).
pub const CHECK_CODES: [(&str, &str); 12] = [
    ("C101", "the cohort uses more than one label set"),
    ("C102", "a class id means different things in different label sets"),
    ("C103", "a sample declares no label set"),
    ("C201", "a split claim's manifest digest is not this manifest's"),
    ("C202", "one subject's or grouping key's samples claim different partitions of one split"),
    ("C203", "a sample carries no split claim"),
    ("C204", "a group holds part of a subject, so the split is not subject-safe"),
    ("C301", "a class is examined in only part of the cohort"),
    ("C302", "a class appears in no sample"),
    ("C401", "a file changed after the manifest was written"),
    ("C402", "a file in the manifest no longer exists"),
    ("C501", "the cohort mixes de-identified and non-de-identified samples"),
];

/// The summary of a cohort code, or `""`.
pub fn code_summary(code: &str) -> &'static str {
    CHECK_CODES.iter().find(|(c, _)| *c == code).map(|(_, s)| *s).unwrap_or("")
}

/// One cohort finding.
#[derive(Debug, Clone, PartialEq)]
pub struct Finding {
    pub code: String,
    pub severity: String,
    pub message: String,
    pub r#where: Vec<String>,
}

impl Finding {
    /// A finding from its JSON form (what [`Finding::to_json`] writes).
    pub fn from_json(doc: &Value) -> Finding {
        let text = |k: &str| doc.get(k).map(crate::json::py_str).unwrap_or_default();
        Finding {
            code: text("code"),
            severity: text("severity"),
            message: text("message"),
            r#where: doc
                .get("where")
                .and_then(Value::as_array)
                .map(|a| a.iter().map(crate::json::py_str).collect())
                .unwrap_or_default(),
        }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "where": self.r#where,
            "summary": code_summary(&self.code),
        })
    }
}

impl std::fmt::Display for Finding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:<7} {} {}", self.severity.to_uppercase(), self.code, self.message)?;
        if self.r#where.is_empty() {
            return Ok(());
        }
        let shown = self.r#where.iter().take(3).cloned().collect::<Vec<_>>().join(", ");
        let more = if self.r#where.len() <= 3 { String::new() } else { format!(" (+{} more)", self.r#where.len() - 3) };
        write!(f, "\n          {shown}{more}")
    }
}

/// Coverage of one class across the cohort.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Coverage {
    pub examined_in: usize,
    pub present_in: usize,
    pub of: usize,
}

/// Everything [`check`] found.
#[derive(Debug, Clone, PartialEq)]
pub struct CohortReport {
    pub manifest_sha256: String,
    pub samples: usize,
    pub findings: Vec<Finding>,
    pub coverage: BTreeMap<i64, Coverage>,
}

impl CohortReport {
    pub fn add(&mut self, code: &str, severity: &str, message: impl Into<String>, r#where: Vec<String>) -> &Finding {
        self.findings.push(Finding { code: code.into(), severity: severity.into(), message: message.into(), r#where });
        self.findings.last().expect("just pushed")
    }

    pub fn errors(&self) -> Vec<&Finding> {
        self.findings.iter().filter(|f| f.severity == "error").collect()
    }

    pub fn warnings(&self) -> Vec<&Finding> {
        self.findings.iter().filter(|f| f.severity == "warning").collect()
    }

    pub fn ok(&self) -> bool {
        self.errors().is_empty()
    }

    pub fn to_json(&self) -> Value {
        let coverage: Map<String, Value> = self
            .coverage
            .iter()
            .map(|(k, v)| {
                (k.to_string(), json!({"examined_in": v.examined_in, "present_in": v.present_in, "of": v.of}))
            })
            .collect();
        json!({
            "manifest_sha256": self.manifest_sha256,
            "samples": self.samples,
            "ok": self.ok(),
            "errors": self.errors().len(),
            "warnings": self.warnings().len(),
            "findings": self.findings.iter().map(Finding::to_json).collect::<Vec<_>>(),
            "coverage": coverage,
        })
    }

    /// A report from its JSON form (what [`CohortReport::to_json`] writes).
    pub fn from_json(doc: &Value) -> CohortReport {
        let count = |v: Option<&Value>, k: &str| v.and_then(|v| v.get(k)).and_then(Value::as_u64).unwrap_or(0) as usize;
        let coverage = doc
            .get("coverage")
            .and_then(Value::as_object)
            .map(|m| {
                m.iter()
                    .filter_map(|(k, v)| {
                        let id = k.parse::<i64>().ok()?;
                        let v = Some(v);
                        Some((
                            id,
                            Coverage {
                                examined_in: count(v, "examined_in"),
                                present_in: count(v, "present_in"),
                                of: count(v, "of"),
                            },
                        ))
                    })
                    .collect()
            })
            .unwrap_or_default();
        CohortReport {
            manifest_sha256: doc.get("manifest_sha256").map(crate::json::py_str).unwrap_or_default(),
            samples: doc.get("samples").and_then(Value::as_u64).unwrap_or(0) as usize,
            findings: doc
                .get("findings")
                .and_then(Value::as_array)
                .map(|a| a.iter().map(Finding::from_json).collect())
                .unwrap_or_default(),
            coverage,
        }
    }

    pub fn format(&self) -> String {
        let mut lines = vec![format!(
            "{} samples: {} ({} errors, {} warnings)",
            self.samples,
            if self.ok() { "OK" } else { "FAILED" },
            self.errors().len(),
            self.warnings().len()
        )];
        lines.extend(self.findings.iter().map(|f| format!("  {f}")));
        lines.join("\n")
    }
}

/// Every cohort-level check, over a manifest.
///
/// `deep` re-reads each file's `content_id` rather than trusting size and
/// mtime --- slower, and the only way to be sure a file is the one scanned.
pub fn check(manifest: &Manifest, set_id: Option<&str>, deep: bool) -> CohortReport {
    let mut report = CohortReport {
        manifest_sha256: manifest.sha256(),
        samples: manifest.len(),
        findings: Vec::new(),
        coverage: BTreeMap::new(),
    };
    vocabulary(manifest, &mut report);
    splits(manifest, &mut report, set_id);
    coverage(manifest, &mut report);
    freshness(manifest, &mut report, deep);
    deidentification(manifest, &mut report);
    report
}

/// One cohort, one meaning per class id.
fn vocabulary(manifest: &Manifest, report: &mut CohortReport) {
    let mut by_digest: IndexMap<&str, Vec<&Entry>> = IndexMap::new();
    let mut missing = Vec::new();
    for entry in &manifest.entries {
        match &entry.label_set_digest {
            None => missing.push(entry.path.clone()),
            Some(d) => by_digest.entry(d.as_str()).or_default().push(entry),
        }
    }
    if !missing.is_empty() {
        report.add(
            "C103",
            "warning",
            format!(
                "{} sample(s) declare no label set, so their class ids mean whatever the reader assumes",
                missing.len()
            ),
            missing,
        );
    }
    if by_digest.len() > 1 {
        let names: BTreeSet<String> = by_digest
            .values()
            .flatten()
            .map(|e| e.label_set_id.clone().filter(|s| !s.is_empty()).unwrap_or_else(|| "?".into()))
            .collect();
        report.add(
            "C101",
            "warning",
            format!(
                "{} distinct label sets in one cohort ({})",
                by_digest.len(),
                names.into_iter().collect::<Vec<_>>().join(", ")
            ),
            by_digest.values().map(|es| es[0].path.clone()).collect(),
        );
        // Same id, different key, silently mislabels a whole training run.
        let mut meanings: BTreeMap<i64, BTreeSet<String>> = BTreeMap::new();
        for entries in by_digest.values() {
            for entry in entries {
                for class_id in &entry.class_ids {
                    meanings.entry(*class_id).or_default().insert(format!(
                        "{}@{}",
                        entry.label_set_id.as_deref().unwrap_or("None"),
                        entry.label_set_version.as_deref().unwrap_or("None")
                    ));
                }
            }
        }
        let clashing: Vec<i64> = meanings.iter().filter(|(_, sets)| sets.len() > 1).map(|(c, _)| *c).collect();
        if !clashing.is_empty() {
            report.add(
                "C102",
                "error",
                format!(
                    "class id(s) {} appear in more than one label set; check they mean the same thing before training \
                     on the union",
                    repr_int_list(&clashing)
                ),
                by_digest.values().map(|es| es[0].path.clone()).collect(),
            );
        }
    }
}

/// A claim is only worth something if it is about this cohort (§12.3).
///
/// Grouped by the §12.2 grouping key, joined through shared subjects
/// (`anatomy_units`), which is what `medh5 splits` audits and what the
/// splitter assigns by.
fn splits(manifest: &Manifest, report: &mut CohortReport, set_id: Option<&str>) {
    let digest = manifest.sha256();
    let mut stale = Vec::new();
    let mut unclaimed = Vec::new();
    let units = anatomy_units(manifest.entries.iter().map(|e| (e.subject_id.as_str(), e.group_id.as_str())));
    let mut by_unit: IndexMap<Vec<String>, IndexMap<String, BTreeSet<String>>> = IndexMap::new();
    let mut subjects: IndexMap<Vec<String>, BTreeSet<String>> = IndexMap::new();
    for entry in &manifest.entries {
        let claims: Vec<&Map<String, Value>> = entry
            .splits
            .iter()
            .filter(|c| set_id.is_none_or(|s| c.get("set_id").and_then(Value::as_str) == Some(s)))
            .collect();
        if claims.is_empty() {
            unclaimed.push(entry.path.clone());
            continue;
        }
        let unit = units.get(&entry.group_id).cloned().unwrap_or_else(|| vec![entry.group_id.clone()]);
        for claim in claims {
            if let Some(recorded) = claim.get("manifest_sha256").filter(|v| !v.is_null()) {
                if recorded.as_str() != Some(digest.as_str()) {
                    stale.push(entry.path.clone());
                }
            }
            let text = |k: &str| claim.get(k).map(crate::json::py_str).unwrap_or_else(|| "None".into());
            by_unit.entry(unit.clone()).or_default().entry(text("set_id")).or_default().insert(text("partition"));
            subjects.entry(unit.clone()).or_default().insert(entry.subject_id.clone());
        }
    }
    if !stale.is_empty() {
        report.add(
            "C201",
            "error",
            format!(
                "{} sample(s) claim a split computed against a different manifest --- re-split, or the partitions are \
                 not the ones in use",
                stale.len()
            ),
            stale,
        );
    }
    if !unclaimed.is_empty() && unclaimed.len() != manifest.len() {
        report.add(
            "C203",
            "warning",
            format!("{} of {} sample(s) carry no split claim", unclaimed.len(), manifest.len()),
            unclaimed,
        );
    }
    let leaking: BTreeSet<Vec<String>> = by_unit
        .iter()
        .filter(|(_, sets)| sets.values().any(|partitions| partitions.len() > 1))
        .map(|(unit, _)| unit.clone())
        .collect();
    if !leaking.is_empty() {
        let involved: Vec<String> = leaking
            .iter()
            .take(3)
            .map(|unit| {
                let who =
                    subjects.get(unit).map(|s| s.iter().cloned().collect::<Vec<_>>().join(", ")).unwrap_or_default();
                format!("{} ({who})", unit.join(" + "))
            })
            .collect();
        report.add(
            "C202",
            "error",
            format!(
                "{} grouping key(s), or keys sharing a subject, appear in more than one partition of the same split \
                 --- this leaks anatomy between train and test. Subjects involved: {}",
                leaking.len(),
                involved.join("; ")
            ),
            leaking.iter().flatten().cloned().collect(),
        );
    }
}

/// Which classes were examined where (§11.3), and where they were not.
fn coverage(manifest: &Manifest, report: &mut CohortReport) {
    let every: BTreeSet<i64> =
        manifest.entries.iter().flat_map(|e| e.class_ids.iter().chain(&e.annotated_class_ids)).copied().collect();
    let n = manifest.len();
    for class_id in every {
        let examined = manifest.entries.iter().filter(|e| e.examined(class_id)).count();
        let present = manifest.entries.iter().filter(|e| e.has_class(class_id)).count();
        report.coverage.insert(class_id, Coverage { examined_in: examined, present_in: present, of: n });
    }
    let partial: Vec<i64> =
        report.coverage.iter().filter(|(_, v)| v.examined_in > 0 && v.examined_in < n).map(|(c, _)| *c).collect();
    if !partial.is_empty() {
        report.add(
            "C301",
            "warning",
            format!(
                "class(es) {} are examined in only part of the cohort; the unexamined samples are not negative \
                 examples of them (§11.3)",
                repr_int_list(&partial)
            ),
            Vec::new(),
        );
    }
    let absent: Vec<i64> = report.coverage.iter().filter(|(_, v)| v.present_in == 0).map(|(c, _)| *c).collect();
    if !absent.is_empty() {
        report.add(
            "C302",
            "info",
            format!("class(es) {} are named but occur in no sample", repr_int_list(&absent)),
            Vec::new(),
        );
    }
}

fn freshness(manifest: &Manifest, report: &mut CohortReport, deep: bool) {
    let missing: Vec<String> =
        manifest.entries.iter().filter(|e| !Path::new(&e.path).exists()).map(|e| e.path.clone()).collect();
    if !missing.is_empty() {
        report.add("C402", "error", format!("{} manifest file(s) no longer exist", missing.len()), missing.clone());
    }
    let mut changed: Vec<String> = manifest.stale().into_iter().filter(|p| !missing.contains(p)).collect();
    if deep {
        let mut all: BTreeSet<String> = changed.into_iter().collect();
        all.extend(changed_content(manifest));
        changed = all.into_iter().collect();
    }
    if !changed.is_empty() {
        report.add(
            "C401",
            if deep { "error" } else { "warning" },
            format!(
                "{} file(s) changed after the manifest was written; re-scan before trusting a split made from it",
                changed.len()
            ),
            changed,
        );
    }
}

/// Paths whose `content_id` no longer matches --- the real check.
fn changed_content(manifest: &Manifest) -> Vec<String> {
    let mut out = Vec::new();
    for entry in &manifest.entries {
        let Some(recorded) = &entry.content_id else { continue };
        if !Path::new(&entry.path).exists() {
            continue;
        }
        match open_any(Path::new(&entry.path), entry.key.as_deref()) {
            // No key: the entry names a shard, not a sample in it.
            Ok(AnyFile::Collection(_)) => out.push(entry.path.clone()),
            Ok(AnyFile::Sample(s)) => match s.content_id() {
                Ok(current) if current.as_deref() == Some(recorded.as_str()) => {}
                _ => out.push(entry.path.clone()),
            },
            Err(_) => out.push(entry.path.clone()),
        }
    }
    out
}

/// A cohort that is only partly de-identified is not de-identified (§11.4).
fn deidentification(manifest: &Manifest, report: &mut CohortReport) {
    let identified: Vec<String> = manifest.entries.iter().filter(|e| !e.deidentified).map(|e| e.path.clone()).collect();
    if !identified.is_empty() && identified.len() != manifest.len() {
        report.add(
            "C501",
            "warning",
            format!(
                "{} of {} sample(s) carry no de-identification record; absence is not evidence of it",
                identified.len(),
                manifest.len()
            ),
            identified,
        );
    }
}
