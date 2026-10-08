//! Fixtures for the task-and-cache contract (`docs/spec/task-cache-1.md`).
//!
//! The format corpus scores a validator; these score an implementation of the
//! contract.  Each fixture is a task manifest over the corpus's own clinical
//! samples --- pinned to the content they have when the corpus is built ---
//! and what checking it must find: at level `validate`, the findings of the
//! manifest alone (T1xx, T2xx; no file opened); at level `preflight`, the
//! findings of opening its sources too (T3xx) and, for a valid task, each
//! row's status, selected event versions, slot images and target label.
//!
//! [`write_fixtures`] writes them beside a built corpus, under `companion/`,
//! with an `expected.json`; [`publish`](super::publish) includes them, and
//! [`run_fixtures`] checks this implementation against them.

use std::collections::BTreeSet;
use std::path::Path;

use serde_json::{json, Map, Value};

use crate::clinical::model::HOUR;
use crate::companion::task::TaskManifest;
use crate::companion::{preflight, SourceRef};
use crate::{Error, Result};

/// Where the fixtures live inside a published suite.
pub const DIR: &str = "companion";
/// The corpus cases the fixtures read.
pub const SOURCES: [&str; 3] =
    ["clinical-worked-example.medh5", "clinical-one-visit-history.medh5", "W913-higher-minor-projection.medh5"];

const WORKED: &str = "clinical-worked-example.medh5";
const VISIT: &str = "clinical-one-visit-history.medh5";
const FUTURE: &str = "W913-higher-minor-projection.medh5";
const H: i64 = HOUR;

/// One fixture: a manifest and what checking it must find.
#[derive(Debug, Clone, PartialEq)]
pub struct Fixture {
    pub name: String,
    pub description: String,
    /// The contract clause it holds.
    pub clause: String,
    /// `validate` or `preflight`.
    pub level: String,
    pub manifest: Value,
    /// The exact set of finding codes.
    pub findings: BTreeSet<String>,
    /// For a valid preflight: `{row_id: {"status", "events", "slots", "target"}}`.
    pub rows: Value,
}

impl Fixture {
    /// The fixture's file name inside `companion/`.
    pub fn file(&self) -> String {
        format!("{}.task.json", self.name)
    }

    pub fn to_json(&self) -> Value {
        json!({
            "name": self.name,
            "file": self.file(),
            "description": self.description,
            "clause": self.clause,
            "level": self.level,
            "findings": self.findings,
            "rows": self.rows,
        })
    }
}

/// A reference to a corpus file from `companion/`, pinned to what it is now.
fn source(corpus: &Path, file: &str, source_id: &str) -> Result<Value> {
    let sample = SourceRef::new(file, None, "").open(Some(corpus))?;
    let mut pinned = SourceRef::pin(format!("../{file}"), None, &sample)?;
    pinned.source_id = source_id.into();
    Ok(pinned.to_json())
}

/// The task every fixture varies: a CT slot and a lesion-assessment target.
fn task(subjects: Vec<Value>, rows: Vec<Value>) -> Value {
    json!({
        "schema": crate::companion::task::SCHEMA,
        "task": {"id": "lesion-follow-up", "version": "1",
                 "description": "Lesion presence at the next assessment"},
        "identity_namespace": "conformance",
        "slots": [{"name": "ct", "modality": "CT", "required": true}],
        "target": {
            "id": "lesion-present", "version": "1",
            "event": {"kind": "assessment", "code_system": crate::clinical::model::ASSESSMENT_SYSTEM,
                      "code": crate::clinical::model::LESION_PRESENCE},
            "positive": ["present"], "negative": ["absent", "resolved"],
            "horizon_us": 2400 * H, "min_follow_up_us": 1000 * H
        },
        "split": {"set_id": "fold-0", "partitions": ["train", "test"]},
        "subjects": subjects,
        "rows": rows,
    })
}

fn subject(id: &str, partition: Option<&str>, sources: Vec<Value>) -> Value {
    let mut out = json!({"subject_id": id, "sources": sources});
    if let Some(p) = partition {
        out["partition"] = json!(p);
    }
    out
}

fn row(id: &str, subject: &str, cutoff_hours: i64) -> Value {
    json!({"row_id": id, "subject_id": subject, "cutoff_us": cutoff_hours * H})
}

fn expect(status: &str, events: &[&str], ct: &str, target: &str) -> Value {
    json!({"status": status, "events": events, "slots": {"ct": ct}, "target": target})
}

fn fixture(
    name: &str,
    description: &str,
    clause: &str,
    level: &str,
    manifest: Value,
    findings: &[&str],
    rows: Value,
) -> Fixture {
    Fixture {
        name: name.into(),
        description: description.into(),
        clause: clause.into(),
        level: level.into(),
        manifest,
        findings: findings.iter().map(|c| c.to_string()).collect(),
        rows,
    }
}

/// Every fixture, pinned to the corpus files in `corpus`.
pub fn fixtures(corpus: &Path) -> Result<Vec<Fixture>> {
    let worked = || source(corpus, WORKED, "worked");
    let visit = || source(corpus, VISIT, "visit");
    let valid_subjects = || -> Result<Vec<Value>> {
        Ok(vec![
            subject("subj-clinical-01", Some("train"), vec![worked()?]),
            subject("subj-clinical-02", Some("test"), vec![visit()?]),
        ])
    };
    let valid_rows = || {
        vec![
            row("worked@h24", "subj-clinical-01", 24),
            row("worked@h2190", "subj-clinical-01", 2190),
            row("visit@h24", "subj-clinical-02", 24),
            row("visit@d11", "subj-clinical-02", 264),
        ]
    };
    let mut out = Vec::new();

    out.push(fixture(
        "valid-two-subjects",
        "Two subjects in two partitions. At hour 24 the preliminary report is the version the record had; by hour \
         2190 the amendment, the follow-up CT and the lesion assessment are known. A planned order is never a completed \
         treatment, a static observation is unordered, and the annotations attested by events of unknown time are never \
         inputs. The worked subject's assessment labels its early row; nothing labels the others, so they are censored.",
        "§3, §4, §5; 1.1 §9",
        "preflight",
        task(valid_subjects()?, valid_rows()),
        &[],
        json!({
            "worked@h24": expect("eligible", &["lab0", "ct0", "report0_v1"], "CT_tp0", "negative"),
            "worked@h2190": expect("eligible", &["lab0", "ct0", "report0_v2", "ct1", "response1"], "CT_tp1", "censored"),
            "visit@h24": expect(
                "eligible",
                &["hba1c_0", "hba1c_1", "hba1c_2", "hba1c_3", "hba1c_4", "ct", "sex"],
                "CT",
                "censored"
            ),
            "visit@d11": expect(
                "eligible",
                &["hba1c_0", "hba1c_1", "hba1c_2", "hba1c_3", "hba1c_4", "ct", "metformin_given", "sex"],
                "CT",
                "censored"
            ),
        }),
    ));

    let mut no_namespace = task(valid_subjects()?, valid_rows());
    if let Some(m) = no_namespace.as_object_mut() {
        m.remove("identity_namespace");
    }
    out.push(fixture(
        "T101-no-identity-namespace",
        "A manifest that names no identity namespace fails its schema.",
        "§3.1",
        "validate",
        no_namespace,
        &["T101"],
        json!({}),
    ));

    let mut overlap = task(valid_subjects()?, valid_rows());
    overlap["target"]["negative"] = json!(["absent", "present"]);
    out.push(fixture(
        "T102-positive-and-negative-overlap",
        "A target value that is both the outcome and its absence.",
        "§3.4, §5",
        "validate",
        overlap,
        &["T102"],
        json!({}),
    ));

    let mut fingerprint = task(valid_subjects()?, valid_rows());
    fingerprint["fingerprint"] = json!(format!("sha256:{}", "0".repeat(64)));
    out.push(fixture(
        "T103-wrong-fingerprint",
        "A declared manifest fingerprint the manifest does not have.",
        "§3.2",
        "validate",
        fingerprint,
        &["T103"],
        json!({}),
    ));

    let mut undeclared = task(valid_subjects()?, valid_rows());
    if let Some(rows) = undeclared["rows"].as_array_mut() {
        rows.push(row("nobody@h24", "nobody", 24));
    }
    out.push(fixture(
        "T201-row-for-an-undeclared-subject",
        "A row naming a subject the manifest does not declare.",
        "§3.3",
        "validate",
        undeclared,
        &["T201"],
        json!({}),
    ));

    let unsplit = task(
        vec![
            subject("subj-clinical-01", Some("train"), vec![worked()?]),
            subject("subj-clinical-02", None, vec![visit()?]),
        ],
        valid_rows(),
    );
    out.push(fixture(
        "T202-subject-without-a-partition",
        "A subject with no partition in a split task: subjects are split before rows are built.",
        "§3.3",
        "validate",
        unsplit,
        &["T202"],
        json!({}),
    ));

    let mut twice = worked()?;
    twice["source_id"] = json!("worked-again");
    let shared = task(
        vec![
            subject("subj-clinical-01", Some("train"), vec![worked()?]),
            subject("subj-clinical-02", Some("test"), vec![twice]),
        ],
        valid_rows(),
    );
    out.push(fixture(
        "T203-two-subjects-one-sample",
        "Two subjects, in two partitions, pinning one sample: the split would leak it.",
        "§3.3",
        "validate",
        shared,
        &["T203"],
        json!({}),
    ));

    let mut duplicate_row = task(valid_subjects()?, valid_rows());
    if let Some(rows) = duplicate_row["rows"].as_array_mut() {
        rows.push(row("worked@h24", "subj-clinical-01", 48));
    }
    out.push(fixture(
        "T204-duplicate-row-id",
        "Two rows sharing an id.",
        "§3.3",
        "validate",
        duplicate_row,
        &["T204"],
        json!({}),
    ));

    let mut missing = worked()?;
    missing["uri"] = json!("../no-such-sample.medh5");
    out.push(fixture(
        "T301-source-does-not-open",
        "A source whose locator names no file.",
        "§2",
        "preflight",
        task(
            vec![subject("subj-clinical-01", Some("train"), vec![missing])],
            vec![row("worked@h24", "subj-clinical-01", 24)],
        ),
        &["T301"],
        json!({}),
    ));

    let mut stale = worked()?;
    stale["content_id"] = visit()?["content_id"].clone();
    out.push(fixture(
        "T302-pin-is-not-the-sample",
        "A source pinned to another sample's content: the locator finds bytes the pin does not name.",
        "§2",
        "preflight",
        task(
            vec![subject("subj-clinical-01", Some("train"), vec![stale])],
            vec![row("worked@h24", "subj-clinical-01", 24)],
        ),
        &["T302"],
        json!({}),
    ));

    let mut stranger = worked()?;
    stranger["local_subject_id"] = json!("someone-else");
    out.push(fixture(
        "T303-identity-is-not-the-recorded-one",
        "A source whose own subject id is not the one the manifest records for it.",
        "§3.3",
        "preflight",
        task(
            vec![subject("subj-clinical-01", Some("train"), vec![stranger])],
            vec![row("worked@h24", "subj-clinical-01", 24)],
        ),
        &["T303"],
        json!({}),
    ));

    out.push(fixture(
        "T304-fragments-on-two-clocks",
        "One subject's fragments declaring two different clocks: a temporal join needs one.",
        "§3.3; 1.1 §3",
        "preflight",
        task(
            vec![subject("subj-clinical-01", Some("train"), vec![worked()?, visit()?])],
            vec![row("worked@h24", "subj-clinical-01", 24)],
        ),
        &["T304"],
        json!({}),
    ));

    let mut again = worked()?;
    again["source_id"] = json!("worked-again");
    out.push(fixture(
        "T305-duplicates-not-reconciled",
        "One subject's history exported twice, with no reconciliation recorded for the events both copies hold.",
        "§3.3",
        "preflight",
        task(
            vec![subject("subj-clinical-01", Some("train"), vec![worked()?, again])],
            vec![row("worked@h24", "subj-clinical-01", 24)],
        ),
        &["T305"],
        json!({}),
    ));

    let future = source(corpus, FUTURE, "future")?;
    out.push(fixture(
        "T306-a-projection-cannot-certify",
        "A source of a later minor version, read as a projection: it cannot supply certified inputs.",
        "§4; 1.1 §2.2",
        "preflight",
        task(
            vec![subject("subj-clinical-01", Some("train"), vec![future])],
            vec![row("worked@h24", "subj-clinical-01", 24)],
        ),
        &["T306"],
        json!({}),
    ));
    Ok(out)
}

/// Write the fixtures, and their `expected.json`, under `<corpus>/companion/`.
/// The corpus must already hold [`SOURCES`].
pub fn write_fixtures(corpus: &Path) -> Result<Vec<Fixture>> {
    for file in SOURCES {
        if !corpus.join(file).exists() {
            return Err(Error::Value(format!("the corpus at {} has no {file}", corpus.display())));
        }
    }
    let dir = corpus.join(DIR);
    std::fs::create_dir_all(&dir)?;
    let found = fixtures(corpus)?;
    for f in &found {
        std::fs::write(dir.join(f.file()), crate::json::pretty(&f.manifest) + "\n")?;
    }
    let expected = json!({
        "contract": crate::companion::task::SCHEMA,
        "fixtures": found.iter().map(Fixture::to_json).collect::<Vec<_>>(),
    });
    std::fs::write(dir.join("expected.json"), crate::json::pretty(&expected) + "\n")?;
    Ok(found)
}

/// What checking one fixture found, against what it should.
#[derive(Debug, Clone, PartialEq)]
pub struct FixtureResult {
    pub name: String,
    pub missing: Vec<String>,
    pub unexpected: Vec<String>,
    /// Row differences, `row: what` --- a status, a selection, a slot or a label.
    pub rows: Vec<String>,
}

impl FixtureResult {
    pub fn ok(&self) -> bool {
        self.missing.is_empty() && self.unexpected.is_empty() && self.rows.is_empty()
    }
}

/// Check this implementation against fixtures written by [`write_fixtures`].
pub fn run_fixtures(corpus: &Path) -> Result<Vec<FixtureResult>> {
    let dir = corpus.join(DIR);
    let text = std::fs::read_to_string(dir.join("expected.json"))?;
    let expected = crate::json::loads(&text).map_err(|e| Error::Value(format!("expected.json: {e}")))?;
    let mut out = Vec::new();
    for f in expected["fixtures"].as_array().into_iter().flatten() {
        let name = f["name"].as_str().unwrap_or_default().to_string();
        let file = dir.join(f["file"].as_str().unwrap_or_default());
        let want: BTreeSet<String> =
            f["findings"].as_array().into_iter().flatten().filter_map(Value::as_str).map(str::to_string).collect();
        let (found, rows) = check(&file, &dir, f["level"].as_str().unwrap_or("validate"), &f["rows"])?;
        out.push(FixtureResult {
            name,
            missing: want.difference(&found).cloned().collect(),
            unexpected: found.difference(&want).cloned().collect(),
            rows,
        });
    }
    Ok(out)
}

fn check(file: &Path, base: &Path, level: &str, rows: &Value) -> Result<(BTreeSet<String>, Vec<String>)> {
    let manifest = match TaskManifest::load(file) {
        Ok(m) => m,
        Err(e) => return Ok((BTreeSet::from([e.code().unwrap_or("T101").to_string()]), Vec::new())),
    };
    if level == "validate" {
        return Ok((manifest.validate().into_iter().map(|f| f.code).collect(), Vec::new()));
    }
    let report = preflight(&manifest, Some(base), false)?;
    let found = report.findings.iter().map(|f| f.code.clone()).collect();
    let mut differences = Vec::new();
    for (row_id, want) in rows.as_object().unwrap_or(&Map::new()) {
        let Some(view) = report.row(row_id) else {
            differences.push(format!("{row_id}: not a row"));
            continue;
        };
        if want["status"] != json!(view.status) {
            differences.push(format!("{row_id}: status {} not {}", view.status, want["status"]));
        }
        let selected: BTreeSet<&str> = view.events.iter().map(|e| e.event_id.as_str()).collect();
        let wanted: BTreeSet<&str> =
            want["events"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
        if selected != wanted {
            differences.push(format!("{row_id}: selects {selected:?}, not {wanted:?}"));
        }
        for fill in &view.slots {
            if want["slots"][&fill.slot] != json!(fill.image_id) {
                differences.push(format!(
                    "{row_id}: slot {} holds {:?}, not {}",
                    fill.slot, fill.image_id, want["slots"][&fill.slot]
                ));
            }
        }
        if want["target"] != json!(view.target.status) {
            differences.push(format!("{row_id}: target {} not {}", view.target.status, want["target"]));
        }
    }
    Ok((found, differences))
}
