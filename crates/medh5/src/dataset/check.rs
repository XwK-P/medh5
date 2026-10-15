//! Cohort-level checks: what is wrong *between* files, not inside one.
//!
//! `medh5 validate` answers "is this file legal"; every question that matters
//! for training is about the cohort --- do these files mean the same thing by
//! class 3, was this split computed before the cohort last changed, is one
//! subject in train and test.  Findings have the validator's shape, but the
//! codes are cohort codes (`C1xx`--`C5xx`) and belong to this tool, not to the
//! format.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::Path;

use indexmap::IndexMap;
use serde_json::{json, Map, Value};

use crate::collection::{open_any, AnyFile};
use crate::curation::splits::anatomy_units;
use crate::dataset::manifest::{Entry, Manifest};
use crate::dataset::split::locator;
use crate::json::{repr_int_list, repr_str};

pub const SEVERITIES: [&str; 3] = ["error", "warning", "info"];

/// Cohort codes.  Distinct from the format's E/W table on purpose (§15.2).
pub const CHECK_CODES: [(&str, &str); 12] = [
    ("C101", "the cohort uses more than one label set"),
    ("C102", "a class id means different things in different label sets"),
    ("C103", "a sample declares no label set"),
    ("C201", "a split claim's manifest digest is not this manifest's"),
    ("C202", "one subject's or grouping key's samples claim different partitions, or folds, of one split"),
    ("C203", "a sample carries no split claim"),
    ("C204", "a group holds part of a subject, so the split is not subject-safe (raised by dataset split)"),
    ("C301", "a class is examined in only part of the cohort"),
    ("C302", "a class is named but examined in no sample"),
    ("C401", "a file changed after the manifest was written"),
    ("C402", "a file in the manifest no longer exists"),
    ("C501", "the cohort mixes de-identified and non-de-identified samples"),
];

/// Each cohort code's severity, in the order of [`CHECK_CODES`].
///
/// A code's base severity: `C401` is a warning, and an error under
/// `dataset check --deep`, which re-reads every `content_id` and so knows the
/// file changed rather than suspecting it. `C204` is a refusal: `dataset
/// split` raises it rather than reporting it.
pub const CHECK_SEVERITY: &[(&str, &str)] = &[
    ("C101", "warning"),
    ("C102", "error"),
    ("C103", "warning"),
    ("C201", "error"),
    ("C202", "error"),
    ("C203", "warning"),
    ("C204", "error"),
    ("C301", "warning"),
    ("C302", "info"),
    ("C401", "warning"),
    ("C402", "error"),
    ("C501", "warning"),
];

/// The base severity of a cohort code, or `None` for a code that is not one.
pub fn code_severity(code: &str) -> Option<&'static str> {
    CHECK_SEVERITY.iter().find(|(c, _)| *c == code).map(|(_, s)| *s)
}

/// The base severity of a code this module reports.
fn severity(code: &str) -> &'static str {
    code_severity(code).expect("a cohort code")
}

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

/// Coverage of one class across the cohort, as the manifest knows it.
///
/// A manifest is metadata, so it says which samples *name* a class (an
/// annotation can express it) and which *examined* it (§11.3), never which
/// hold any of it: that is a voxel question, which `dataset stats` answers
/// (`present_in`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Coverage {
    pub examined_in: usize,
    pub named_in: usize,
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
            .map(|(k, v)| (k.to_string(), json!({"examined_in": v.examined_in, "named_in": v.named_in, "of": v.of})))
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
                                named_in: count(v, "named_in"),
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
            None => missing.push(locator(entry)),
            Some(d) => by_digest.entry(d.as_str()).or_default().push(entry),
        }
    }
    if !missing.is_empty() {
        report.add(
            "C103",
            severity("C103"),
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
        let first: Vec<String> = by_digest.values().map(|es| locator(es[0])).collect();
        report.add(
            "C101",
            severity("C101"),
            format!(
                "{} distinct label sets in one cohort ({})",
                by_digest.len(),
                names.into_iter().collect::<Vec<_>>().join(", ")
            ),
            first.clone(),
        );
        meanings(&by_digest, first, report);
    }
}

/// Same id, different meaning, silently mislabels a whole training run.
///
/// A class id's meaning is its key in the label set, so two versions of one
/// vocabulary that agree on a class do not clash, and one id and version
/// carrying two vocabularies does --- where keying on `id@version` reported
/// the first and missed the second.  A label set whose classes the manifest
/// does not record (a vocabulary by reference, or a manifest indexed before
/// 2.0) is its own meaning, so an id it shares with another is reported, as
/// nothing can confirm it means the same.
fn meanings(by_digest: &IndexMap<&str, Vec<&Entry>>, first: Vec<String>, report: &mut CohortReport) {
    let mut keys_of: HashMap<&str, BTreeMap<i64, &str>> = HashMap::new();
    for (digest, entries) in by_digest {
        for entry in entries {
            let known = keys_of.entry(digest).or_default();
            known.extend(entry.class_keys.iter().map(|(id, key)| (*id, key.as_str())));
        }
    }
    let mut meant: BTreeMap<i64, BTreeSet<Option<&str>>> = BTreeMap::new();
    let mut unknown: BTreeMap<i64, BTreeSet<&str>> = BTreeMap::new();
    for (digest, entries) in by_digest {
        for entry in entries {
            for class_id in &entry.class_ids {
                match keys_of.get(digest).and_then(|keys| keys.get(class_id)) {
                    Some(key) => {
                        meant.entry(*class_id).or_default().insert(Some(*key));
                    }
                    None => {
                        meant.entry(*class_id).or_default().insert(None);
                        unknown.entry(*class_id).or_default().insert(digest);
                    }
                }
            }
        }
    }
    let clashing: Vec<i64> = meant
        .iter()
        .filter(|(id, keys)| keys.len() > 1 || unknown.get(*id).is_some_and(|digests| digests.len() > 1))
        .map(|(id, _)| *id)
        .collect();
    if clashing.is_empty() {
        return;
    }
    let what: Vec<String> = clashing
        .iter()
        .take(5)
        .map(|id| {
            let keys: Vec<String> =
                meant[id].iter().map(|key| key.map(repr_str).unwrap_or_else(|| "unrecorded".into())).collect();
            format!("{id} is {}", keys.join(" or "))
        })
        .collect();
    let more = if clashing.len() > 5 { format!("; and {} more", clashing.len() - 5) } else { String::new() };
    let unrecorded = if clashing.iter().any(|id| meant[id].contains(&None)) {
        ". An unrecorded meaning is a label set whose classes this manifest does not hold (one by reference, or \
         a manifest indexed before 2.0 --- re-index it): check it means the same"
    } else {
        ""
    };
    report.add(
        "C102",
        severity("C102"),
        format!(
            "class id(s) {} mean different things in different label sets ({}{more}); training on the union \
             labels one class with another's examples{unrecorded}",
            repr_int_list(&clashing),
            what.join("; ")
        ),
        first,
    );
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
            unclaimed.push(locator(entry));
            continue;
        }
        let unit = units.get(&entry.group_id).cloned().unwrap_or_else(|| vec![entry.group_id.clone()]);
        // A sample is stale once, however many of its claims are: counted per
        // claim, two stale sets made one sample two.
        let mut is_stale = false;
        for claim in claims {
            if let Some(recorded) = claim.get("manifest_sha256").filter(|v| !v.is_null()) {
                is_stale |= recorded.as_str() != Some(digest.as_str());
            }
            let text = |k: &str| claim.get(k).map(crate::json::py_str).unwrap_or_else(|| "None".into());
            // A fold is a place of its own: k-fold claims are `train` in every
            // fold but the validation one, so a unit in folds 0 and 3 read
            // `train` twice and was no leak (`medh5 splits` and
            // `Split::leaks` hold the same rule).
            let fold = claim.get("fold").and_then(|f| f.as_i64().or_else(|| f.as_f64().map(|f| f as i64)));
            let placed = match fold {
                Some(fold) => format!("{} (fold {fold})", text("partition")),
                None => text("partition"),
            };
            by_unit.entry(unit.clone()).or_default().entry(text("set_id")).or_default().insert(placed);
            subjects.entry(unit.clone()).or_default().insert(entry.subject_id.clone());
        }
        if is_stale {
            stale.push(locator(entry));
        }
    }
    if !stale.is_empty() {
        report.add(
            "C201",
            severity("C201"),
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
            severity("C203"),
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
            severity("C202"),
            format!(
                "{} grouping key(s), or keys sharing a subject, appear in more than one partition, or fold, of the \
                 same split --- this leaks anatomy between train and test. Subjects involved: {}",
                leaking.len(),
                involved.join("; ")
            ),
            leaking.iter().flatten().cloned().collect(),
        );
    }
}

/// Which classes were examined where (§11.3), and where they were not.
///
/// What a manifest can say about a class: where it is named and where it
/// was looked for.  Counting the samples that name a class as the ones it
/// occurs in made C302 unreachable --- every sample that examines a class
/// names it (`annotated_class_ids ⊆ class_ids`) --- so occurrence is left to
/// `dataset stats`, which reads the voxels.
fn coverage(manifest: &Manifest, report: &mut CohortReport) {
    let every: BTreeSet<i64> =
        manifest.entries.iter().flat_map(|e| e.class_ids.iter().chain(&e.annotated_class_ids)).copied().collect();
    let n = manifest.len();
    for class_id in every {
        let examined = manifest.entries.iter().filter(|e| e.examined(class_id)).count();
        let named = manifest.entries.iter().filter(|e| e.has_class(class_id)).count();
        report.coverage.insert(class_id, Coverage { examined_in: examined, named_in: named, of: n });
    }
    let partial: Vec<i64> =
        report.coverage.iter().filter(|(_, v)| v.examined_in > 0 && v.examined_in < n).map(|(c, _)| *c).collect();
    if !partial.is_empty() {
        report.add(
            "C301",
            severity("C301"),
            format!(
                "class(es) {} are examined in only part of the cohort; the unexamined samples are not negative \
                 examples of them (§11.3)",
                repr_int_list(&partial)
            ),
            Vec::new(),
        );
    }
    let unexamined: Vec<i64> = report.coverage.iter().filter(|(_, v)| v.examined_in == 0).map(|(c, _)| *c).collect();
    if !unexamined.is_empty() {
        report.add(
            "C302",
            severity("C302"),
            format!(
                "class(es) {} are named but examined in no sample, so no sample is a negative example of them \
                 (§11.3); `medh5 dataset stats` counts where they occur",
                repr_int_list(&unexamined)
            ),
            Vec::new(),
        );
    }
}

fn freshness(manifest: &Manifest, report: &mut CohortReport, deep: bool) {
    // A collection's members are an entry each and one file: counted once.
    let mut seen = HashSet::new();
    let missing: Vec<String> = manifest
        .entries
        .iter()
        .map(|e| e.path.as_str())
        .filter(|p| seen.insert(*p) && !Path::new(p).exists())
        .map(str::to_string)
        .collect();
    if !missing.is_empty() {
        report.add(
            "C402",
            severity("C402"),
            format!("{} manifest file(s) no longer exist", missing.len()),
            missing.clone(),
        );
    }
    let gone: HashSet<&str> = missing.iter().map(String::as_str).collect();
    let mut changed: Vec<String> = manifest.stale().into_iter().filter(|p| !gone.contains(p.as_str())).collect();
    if deep {
        let mut all: BTreeSet<String> = changed.into_iter().collect();
        all.extend(changed_content(manifest));
        changed = all.into_iter().collect();
    }
    if !changed.is_empty() {
        report.add(
            "C401",
            if deep { "error" } else { severity("C401") },
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
    let identified: Vec<String> = manifest.entries.iter().filter(|e| !e.deidentified).map(locator).collect();
    if !identified.is_empty() && identified.len() != manifest.len() {
        report.add(
            "C501",
            severity("C501"),
            format!(
                "{} of {} sample(s) carry no de-identification record; absence is not evidence of it",
                identified.len(),
                manifest.len()
            ),
            identified,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::split::{make_splits_with, SplitRequest};

    fn claim(partition: &str, digest: &str) -> Map<String, Value> {
        json!({"set_id": "s", "partition": partition, "manifest_sha256": digest}).as_object().unwrap().clone()
    }

    /// A cohort that raises every code `check` reports: two label sets that
    /// disagree on class 1 and a sample with none, a stale claim, one subject
    /// in two partitions and a sample with no claim, class 1 examined in one
    /// of three samples and class 2 in none, a missing file and a changed
    /// one, and one sample without a de-identification record.
    fn every_code(dir: &Path) -> Manifest {
        let changed = dir.join("changed.medh5");
        std::fs::write(&changed, b"not the bytes scanned").unwrap();
        let gone = Entry {
            path: dir.join("gone.medh5").to_string_lossy().into_owned(),
            sample_id: "a".into(),
            subject_id: "p1".into(),
            group_id: "p1".into(),
            class_ids: vec![1],
            annotated_class_ids: vec![1],
            class_keys: BTreeMap::from([(1, "liver".to_string())]),
            label_set_id: Some("one".into()),
            label_set_digest: Some("d1".into()),
            splits: vec![claim("train", "another manifest")],
            deidentified: true,
            ..Default::default()
        };
        let changed = Entry {
            path: changed.to_string_lossy().into_owned(),
            sample_id: "b".into(),
            subject_id: "p1".into(),
            group_id: "p1".into(),
            class_ids: vec![1, 2],
            class_keys: BTreeMap::from([(1, "kidney".to_string()), (2, "spleen".to_string())]),
            label_set_id: Some("two".into()),
            label_set_digest: Some("d2".into()),
            splits: vec![claim("test", "another manifest")],
            ..Default::default()
        };
        let bare = Entry {
            path: dir.join("bare.medh5").to_string_lossy().into_owned(),
            sample_id: "c".into(),
            subject_id: "p2".into(),
            group_id: "p2".into(),
            deidentified: true,
            ..Default::default()
        };
        Manifest { entries: vec![gone, changed, bare], ..Default::default() }
    }

    #[test]
    fn every_code_has_one_severity_from_the_vocabulary() {
        let codes: Vec<&str> = CHECK_CODES.iter().map(|(c, _)| *c).collect();
        let severities: Vec<&str> = CHECK_SEVERITY.iter().map(|(c, _)| *c).collect();
        assert_eq!(codes, severities);
        for (code, severity) in CHECK_SEVERITY {
            assert!(SEVERITIES.contains(severity), "{code}: {severity}");
            assert_eq!(code_severity(code), Some(*severity));
        }
        assert_eq!(code_severity("C999"), None);
    }

    #[test]
    fn every_finding_carries_its_codes_severity() {
        let dir = tempfile::tempdir().unwrap();
        let report = check(&every_code(dir.path()), None, false);
        let found: BTreeSet<&str> = report.findings.iter().map(|f| f.code.as_str()).collect();
        let reported: BTreeSet<&str> = CHECK_CODES.iter().map(|(c, _)| *c).filter(|c| *c != "C204").collect();
        assert_eq!(found, reported, "{}", report.format());
        for finding in &report.findings {
            assert_eq!(Some(finding.severity.as_str()), code_severity(&finding.code), "{finding}");
        }
        // C204 is the splitter's refusal, not a finding.
        let mut cohort = every_code(dir.path());
        cohort.entries[1].group_id = "p1-again".into();
        let request = SplitRequest { group_by: "group_id".into(), ..Default::default() };
        let refused = make_splits_with(&cohort, &request).unwrap_err();
        assert_eq!(refused.code(), Some("C204"));
        assert_eq!(code_severity("C204"), Some("error"));
    }

    #[test]
    fn c401_is_an_error_only_with_deep() {
        let dir = tempfile::tempdir().unwrap();
        let cohort = every_code(dir.path());
        let c401 = |deep: bool| {
            check(&cohort, None, deep).findings.into_iter().find(|f| f.code == "C401").expect("C401").severity
        };
        assert_eq!(c401(false), "warning");
        assert_eq!(c401(true), "error");
    }
}
