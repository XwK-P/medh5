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
//! Beside them, under `companion/caches/`, feature caches (`medh5.cache/1`)
//! over the same samples: one valid cache of each level, and one for each of
//! T401--T407, with what validating it must find --- given its task, where the
//! finding is about the task.
//!
//! [`write_fixtures`] writes them beside a built corpus, under `companion/`,
//! with an `expected.json`; [`publish`](super::publish) includes them,
//! [`run_fixtures`] checks this implementation against them, and [`score`]
//! scores another from what it reported.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

use serde_json::{json, Value};

use super::{Case, CaseResult};
use crate::array::NdArray;
use crate::clinical::model::HOUR;
use crate::companion::cache::{event_entry_id, CacheEntry, CacheHeader, CacheWriter};
use crate::companion::task::TaskManifest;
use crate::companion::{preflight, validate_cache, Admitted, SourceRef};
use crate::h5::attrs::{self, AttrValue};
use crate::pyval::py_path;
use crate::{Error, Result};

/// Where the fixtures live inside a published suite.
pub const DIR: &str = "companion";
/// Where the cache fixtures live inside [`DIR`].
pub const CACHE_DIR: &str = "caches";
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
    /// For a valid preflight: `{row_id: {"status", "events", "slots", "target"}}`,
    /// and optionally `"inputs": {slot: {"annotations"}}`, the eligible voxel
    /// annotations the slot may read as inputs.
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
    json!({"status": status, "reasons": [], "events": events, "slots": {"ct": ct}, "target": target})
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
         treatment, a static observation is unordered, and no annotation is an input before the procedure that drew it is \
         known (hour 2200). The worked subject's assessment labels its early row; nothing labels the others, so they are censored.",
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

    let mut after_seg0 = task(valid_subjects()?, vec![row("worked@h2250", "subj-clinical-01", 2250)]);
    after_seg0["subjects"] = json!([subject("subj-clinical-01", Some("train"), vec![worked()?])]);
    let mut attested = expect("eligible", &["lab0", "ct0", "report0_v2", "ct1", "response1"], "CT_tp1", "censored");
    attested["inputs"] = json!({"ct": {"annotations": ["lesions_tp1"]}});
    out.push(fixture(
        "valid-self-asserted-mask",
        "The segmentation procedure, of unknown time and so never selected under effective ordering, asserts that it \
         describes the masks it drew. Once it is known (hour 2200), its assertion attests them: the follow-up CT's \
         mask is an input, though the procedure is not; an implementation that requires the asserter to be selected \
         when it is also the link's endpoint finds none.",
        "§3.5; 1.1 §7.3, §9.1 step 5",
        "preflight",
        after_seg0,
        &[],
        json!({"worked@h2250": attested}),
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

/// One cache fixture: a feature cache and what validating it must find.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct CacheFixture {
    pub name: String,
    pub description: String,
    /// The contract clause it holds.
    pub clause: String,
    /// The task fixture it is validated against, when a finding is about the
    /// task (T404--T406) or the cache is patient-level.
    pub task: Option<String>,
    /// The exact set of finding codes.
    pub findings: BTreeSet<String>,
}

impl CacheFixture {
    /// The fixture's file, relative to `companion/`.
    pub fn file(&self) -> String {
        format!("{CACHE_DIR}/{}.medh5cache", self.name)
    }

    pub fn to_json(&self) -> Value {
        let mut out = json!({
            "name": self.name,
            "file": self.file(),
            "description": self.description,
            "clause": self.clause,
            "level": "cache",
            "findings": self.findings,
        });
        if let Some(task) = &self.task {
            out["task"] = json!(task);
        }
        out
    }
}

/// The task fixture the patient-level caches are built for.
const CACHE_TASK: &str = "valid-two-subjects";

/// A four-float feature.
fn feature(value: f32) -> Result<NdArray> {
    NdArray::from_vec(&[4], vec![value; 4])
}

fn cache_header(level: &str, task: Option<&TaskManifest>) -> CacheHeader {
    CacheHeader {
        level: level.into(),
        encoder: json!({"name": "conformance-constant", "revision": format!("sha256:{}", "0".repeat(64)),
                        "tokenizer": null}),
        preprocessing: json!({"lowercase": true}),
        output: json!({"dtype": "float32", "shape": [4], "pooling": "mean", "chunking": null}),
        task_fingerprint: task.map(TaskManifest::task_fingerprint),
        selection: Some("strict_prospective".into()),
        fitted_on: None,
    }
}

/// A feature of one event version of `source` and the document it owns.
fn event_entry(source: &SourceRef, event: &str, document: Option<&str>) -> CacheEntry {
    CacheEntry {
        entry_id: event_entry_id(&source.content_id, event),
        sources: vec![source.clone()],
        event_id: Some(event.into()),
        document_id: document.map(str::to_string),
        row_id: None,
        cutoff_us: None,
        event_versions: None,
        row_fingerprint: None,
        digest: String::new(),
    }
}

/// Write a cache of `entries` at `path`.
fn write_cache(path: &Path, header: CacheHeader, entries: Vec<CacheEntry>) -> Result<()> {
    let mut writer = CacheWriter::create(path, header)?;
    for (i, entry) in entries.into_iter().enumerate() {
        writer.add(entry, &feature(i as f32 + 0.5)?)?;
    }
    writer.commit()?;
    Ok(())
}

/// Every cache fixture, written under `<corpus>/companion/caches/`.  The task
/// fixtures must already be written: the patient-level caches are built
/// from [`CACHE_TASK`]'s preflight.
fn write_caches(corpus: &Path) -> Result<Vec<CacheFixture>> {
    let dir = corpus.join(DIR);
    let caches = dir.join(CACHE_DIR);
    std::fs::create_dir_all(&caches)?;
    let task_file = dir.join(format!("{CACHE_TASK}.task.json"));
    let task = TaskManifest::load(&task_file)?;
    let report = preflight(&task, Some(&dir), false)?;
    // An event-level cache validated alone resolves its sources against its
    // own directory (task-cache-1 §7.1); a patient-level one copies its
    // task's references, relative to the task's.
    let mut worked = task.subjects[0].sources[0].clone();
    worked.uri = format!("../{}", worked.uri);
    let path = |name: &str| caches.join(format!("{name}.medh5cache"));
    let reports = [("report0_v1", "report0_text_v1"), ("report0_v2", "report0_text_v2")];
    let event_cache =
        |name: &str, entries: Vec<CacheEntry>| write_cache(&path(name), cache_header("event", None), entries);
    let documents = || reports.iter().map(|(e, d)| event_entry(&worked, e, Some(d))).collect::<Vec<_>>();
    // A row's feature, as preflight admits it.
    let row_entry = |row_id: &str| -> Result<CacheEntry> {
        let view = report.row(row_id).ok_or_else(|| Error::Value(format!("{CACHE_TASK} has no row {row_id}")))?;
        let subject = task.subject(&view.subject_id).ok_or_else(|| Error::Value(view.subject_id.clone()))?;
        Ok(CacheEntry {
            entry_id: row_id.into(),
            sources: subject.sources.clone(),
            event_id: None,
            document_id: None,
            row_id: Some(row_id.into()),
            cutoff_us: Some(view.cutoff_us),
            event_versions: Some(report.events_of(view).iter().map(|e| e.event_id.clone()).collect()),
            row_fingerprint: Some(view.fingerprint.clone()),
            digest: String::new(),
        })
    };
    let rows = || -> Result<Vec<CacheEntry>> { ["worked@h24", "visit@h24"].iter().map(|r| row_entry(r)).collect() };
    let patient = || cache_header("patient", Some(&task));
    let mut out = Vec::new();
    let mut fixture = |name: &str, description: &str, clause: &str, with_task: bool, findings: &[&str]| {
        out.push(CacheFixture {
            name: name.into(),
            description: description.into(),
            clause: clause.into(),
            task: with_task.then(|| format!("{CACHE_TASK}.task.json")),
            findings: findings.iter().map(|c| c.to_string()).collect(),
        });
    };

    event_cache("valid-event", documents())?;
    fixture(
        "valid-event",
        "An event-level cache of the worked example's report, both versions, each joined to the text it owns.",
        "§7.1, §7.2, §8",
        false,
        &[],
    );

    write_cache(&path("valid-patient"), patient(), rows()?)?;
    fixture(
        "valid-patient",
        "A patient-level cache of two rows of the task it names, each pinned to its row's fingerprint, cutoff and the \
         versions its cutoff admits.",
        "§7.2, §8",
        true,
        &[],
    );

    event_cache("T401-manifest-checksum-mismatch", documents())?;
    let file = crate::h5::file::open_rw(&path("T401-manifest-checksum-mismatch"))?;
    let wrong = AttrValue::Str(format!("sha256:{}", "0".repeat(64)));
    attrs::write(&file, crate::companion::cache::DIGEST_ATTR, &wrong)?;
    file.close()?;
    fixture(
        "T401-manifest-checksum-mismatch",
        "A cache whose manifest is not the one its checksum names: the cache is wrong about itself.",
        "§7.1, §8",
        false,
        &["T401"],
    );

    event_cache("T402-payload-changed", documents())?;
    let file = crate::h5::file::open_rw(&path("T402-payload-changed"))?;
    let changed = format!("entries/{}", event_entry_id(&worked.content_id, "report0_v2"));
    file.dataset(&changed)?.write_raw(&[9.0f32; 4])?;
    file.close()?;
    fixture(
        "T402-payload-changed",
        "A cache one of whose features no longer has the bytes its digest covers.",
        "§8",
        false,
        &["T402"],
    );

    let mut elsewhere = worked.clone();
    elsewhere.content_id = source(corpus, VISIT, "visit")?["content_id"].as_str().unwrap_or_default().to_string();
    event_cache("T403-source-changed", vec![event_entry(&elsewhere, "report0_v1", Some("report0_text_v1"))])?;
    fixture(
        "T403-source-changed",
        "A feature pinned to a version of its source the file no longer has: the cache is right about a sample that \
         changed, and the entry is stale.",
        "§7.1, §8",
        false,
        &["T403"],
    );

    let mut late = row_entry("worked@h24")?;
    late.cutoff_us = Some(48 * H);
    write_cache(&path("T404-another-cutoff"), patient(), vec![late])?;
    fixture(
        "T404-another-cutoff",
        "A row's feature built at a cutoff other than the row's: it belongs to another row of the task.",
        "§8",
        true,
        &["T404"],
    );

    let mut header = patient();
    header.fitted_on = Some(CacheHeader::fitted_on(&task, "test"));
    write_cache(&path("T405-fitted-on-the-test-partition"), header, rows()?)?;
    fixture(
        "T405-fitted-on-the-test-partition",
        "Preprocessing fitted on the task's test partition, not its training partition.",
        "§7.3, §8",
        true,
        &["T405"],
    );

    let mut whole = row_entry("worked@h24")?;
    whole.event_versions = Some(["lab0", "ct0", "report0_v2", "ct1", "response1"].map(str::to_string).to_vec());
    write_cache(&path("T406-whole-history"), patient(), vec![whole])?;
    fixture(
        "T406-whole-history",
        "A row's feature encoding the subject's whole history, versions its cutoff does not admit among them.",
        "§8",
        true,
        &["T406"],
    );

    event_cache("T407-another-versions-text", vec![event_entry(&worked, "report0_v1", Some("report0_text_v2"))])?;
    fixture(
        "T407-another-versions-text",
        "A feature of the preliminary report's version joined to the amendment's text: every row admitting the \
         preliminary version would read text written after it.",
        "§7.2, §8",
        false,
        &["T407"],
    );
    Ok(out)
}

/// Write the fixtures, and their `expected.json`, under `<corpus>/companion/`.
/// The corpus must already hold [`SOURCES`].  Returns the task fixtures; the
/// cache fixtures are listed in `expected.json` beside them.
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
    let caches = write_caches(corpus)?;
    let listed: Vec<Value> =
        found.iter().map(Fixture::to_json).chain(caches.iter().map(CacheFixture::to_json)).collect();
    let expected = json!({
        "contract": crate::companion::task::SCHEMA,
        "cache_contract": crate::companion::cache::SCHEMA,
        "fixtures": listed,
    });
    std::fs::write(dir.join("expected.json"), crate::json::pretty(&expected) + "\n")?;
    Ok(found)
}

/// The fixtures' `expected.json`.
pub fn load_expected(corpus: &Path) -> Result<Value> {
    let path = corpus.join(DIR).join("expected.json");
    let text = std::fs::read_to_string(&path).map_err(|e| crate::error::os_error(&e, py_path(&path)))?;
    crate::json::loads(&text).map_err(|e| Error::Value(format!("expected.json: {e}")))
}

/// What checking one fixture found, against what it should.
#[derive(Debug, Clone, PartialEq)]
pub struct FixtureResult {
    pub name: String,
    pub missing: Vec<String>,
    pub unexpected: Vec<String>,
    /// Every other difference: `ok` disagreeing with the findings, and per
    /// row, `row: what` --- a row missing or extra, its status or reasons, a
    /// selection, a slot, an input or a label.
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
    let expected = load_expected(corpus)?;
    let mut out = Vec::new();
    for f in expected["fixtures"].as_array().into_iter().flatten() {
        let name = f["name"].as_str().unwrap_or_default().to_string();
        let file = dir.join(f["file"].as_str().unwrap_or_default());
        let report = check(&file, &dir, f)?;
        let want = codes_of(&f["findings"]);
        let found = codes_of(&report["findings"]);
        out.push(FixtureResult {
            name,
            missing: want.difference(&found).cloned().collect(),
            unexpected: found.difference(&want).cloned().collect(),
            rows: differences(f, &report),
        });
    }
    Ok(out)
}

/// This implementation's report on one fixture, as its command line prints
/// it: `task validate`, `task preflight` or `cache validate` with `--json`.
fn check(file: &Path, base: &Path, fixture: &Value) -> Result<Value> {
    let level = fixture["level"].as_str().unwrap_or("validate");
    if level == "cache" {
        let (task, admitted) = match fixture["task"].as_str() {
            Some(task) => {
                let manifest = TaskManifest::load(&base.join(task))?;
                let admitted = Admitted::preflight(&manifest, Some(base), false)?;
                (Some(manifest), Some(admitted))
            }
            None => (None, None),
        };
        let base = task.as_ref().map(|_| base);
        return Ok(validate_cache(file, base, task.as_ref(), admitted.as_ref())?.to_json());
    }
    let manifest = match TaskManifest::load(file) {
        Ok(m) => m,
        Err(e) => {
            let code = e.code().unwrap_or("T101").to_string();
            return Ok(json!({"ok": false, "findings": [{"code": code}]}));
        }
    };
    if level == "validate" {
        let findings = manifest.validate();
        return Ok(json!({
            "ok": findings.is_empty(),
            "findings": findings.iter().map(crate::companion::Finding::to_json).collect::<Vec<_>>(),
        }));
    }
    Ok(preflight(&manifest, Some(base), false)?.to_json())
}

/// The codes of a findings list: finding objects with a `code`, or codes.
fn codes_of(findings: &Value) -> BTreeSet<String> {
    findings
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|f| f.get("code").unwrap_or(f).as_str())
        .map(str::to_string)
        .collect()
}

/// A row reason's code: `missing_required_slot:ct` is `missing_required_slot`,
/// a finding's line `T302 worked: ...` is `T302`.
fn reason_code(reason: &Value) -> String {
    let text = reason.as_str().unwrap_or_default();
    text.split([':', ' ']).next().unwrap_or_default().to_string()
}

/// How a report differs from a fixture beyond its codes: `ok`, and --- for a
/// fixture that states its rows, which states every row --- each row, both
/// ways: a row stated and not reported, or reported and not stated, fails.
fn differences(fixture: &Value, report: &Value) -> Vec<String> {
    let mut out = Vec::new();
    let clean = codes_of(&fixture["findings"]).is_empty();
    if let Some(ok) = report.get("ok").and_then(Value::as_bool) {
        if ok != clean {
            out.push(format!(
                "ok: reported {ok}, but the fixture {}",
                if clean { "has no findings" } else { "has findings" }
            ));
        }
    }
    let Some(stated) = fixture["rows"].as_object().filter(|rows| !rows.is_empty()) else {
        return out;
    };
    let reported: BTreeMap<String, &Value> = report["rows"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|row| (row["row_id"].as_str().unwrap_or_default().to_string(), row))
        .collect();
    for (row_id, want) in stated {
        let Some(got) = reported.get(row_id) else {
            out.push(format!("{row_id}: not reported"));
            continue;
        };
        if want["status"] != got["status"] {
            out.push(format!("{row_id}: status {} not {}", got["status"], want["status"]));
        }
        let strings = |v: &Value| -> BTreeSet<String> {
            v.as_array().into_iter().flatten().filter_map(Value::as_str).map(str::to_string).collect()
        };
        let reasons: BTreeSet<String> = got["reasons"].as_array().into_iter().flatten().map(reason_code).collect();
        let wanted_reasons: BTreeSet<String> =
            want["reasons"].as_array().into_iter().flatten().map(reason_code).collect();
        if reasons != wanted_reasons {
            out.push(format!("{row_id}: reasons {reasons:?}, not {wanted_reasons:?}"));
        }
        let selected: BTreeSet<String> = got["selection"]["events"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|e| e["event_id"].as_str())
            .map(str::to_string)
            .collect();
        let wanted = strings(&want["events"]);
        if selected != wanted {
            out.push(format!("{row_id}: selects {selected:?}, not {wanted:?}"));
        }
        let fills: HashMap<String, &Value> = got["slots"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|fill| (fill["slot"].as_str().unwrap_or_default().to_string(), fill))
            .collect();
        let slots: BTreeMap<String, Value> =
            fills.iter().map(|(slot, fill)| (slot.clone(), fill["image_id"].clone())).collect();
        let wanted_slots: BTreeMap<String, Value> =
            want["slots"].as_object().into_iter().flatten().map(|(k, v)| (k.clone(), v.clone())).collect();
        if slots != wanted_slots {
            out.push(format!("{row_id}: slots {slots:?}, not {wanted_slots:?}"));
        }
        for (slot, inputs) in want["inputs"].as_object().into_iter().flatten() {
            let found = fills.get(slot).map(|fill| &fill["annotations"]);
            if found != Some(&inputs["annotations"]) {
                out.push(format!("{row_id}: slot {slot} reads {found:?}, not {}", inputs["annotations"]));
            }
        }
        if want["target"] != got["target"]["status"] {
            out.push(format!("{row_id}: target {} not {}", got["target"]["status"], want["target"]));
        }
    }
    for row_id in reported.keys().filter(|r| !stated.contains_key(*r)) {
        out.push(format!("{row_id}: not a row the fixture states"));
    }
    out
}

/// Score another implementation's reports on the fixtures of a published
/// suite (`<root>/companion/`).
///
/// A report is what checking a fixture printed --- `task validate`, `task
/// preflight` or `cache validate` with `--json` --- naming its fixture by
/// `file` (or `path`, as `cache validate` does): its `findings` (objects with
/// a `code`, or codes), and its `ok` and `rows` where it has them.  Each
/// fixture's result is a [`CaseResult`] whose expected errors are its
/// findings; a difference beyond them (a row, `ok`) is its `error`, each one
/// a line of its `details`.  A fixture with no report fails, as a case does.
pub fn score(root: &Path, submitted: &[Value]) -> Result<Vec<CaseResult>> {
    let expected = load_expected(root)?;
    let by_file: HashMap<String, &Value> = submitted.iter().map(|e| (super::suite::submitted_name(e), e)).collect();
    let mut results = Vec::new();
    for fixture in expected["fixtures"].as_array().into_iter().flatten() {
        let file = fixture["file"].as_str().unwrap_or_default();
        let base = Path::new(file).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let findings: Vec<String> = codes_of(&fixture["findings"]).into_iter().collect();
        let case = Case {
            name: fixture["name"].as_str().unwrap_or_default().to_string(),
            description: fixture["description"].as_str().unwrap_or_default().to_string(),
            clause: fixture["clause"].as_str().unwrap_or_default().to_string(),
            build: None,
            level: fixture["level"].as_str().unwrap_or_default().to_string(),
            errors: findings.clone(),
            warnings: Vec::new(),
            suffix: if base.ends_with(".medh5cache") { ".medh5cache" } else { ".task.json" }.to_string(),
            mutated: false,
        };
        let mut result = CaseResult::new(case, py_path(&root.join(DIR).join(file)));
        let Some(entry) = by_file.get(&base) else {
            result.error = Some("no result submitted for this fixture".into());
            results.push(result);
            continue;
        };
        let reported = codes_of(&entry["findings"]);
        let (missing, unexpected) = super::differences(&findings, &[], &reported, &BTreeSet::new());
        result.got_errors = reported.into_iter().collect();
        result.missing = missing;
        result.unexpected = unexpected;
        let other = differences(fixture, entry);
        if !other.is_empty() {
            result.error = Some(other.join("; "));
            result.details = other;
        }
        results.push(result);
    }
    Ok(results)
}
