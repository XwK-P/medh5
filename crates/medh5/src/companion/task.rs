//! The task manifest (`medh5.task/1`): one JSON document per task.
//!
//! A task is identified by its **definition** --- schema, task id and version,
//! identity namespace, selection policy, modality slots and target --- not by
//! its rows: [`TaskManifest::task_fingerprint`] hashes the canonical,
//! defaults-filled definition, so two spellings of one task share it.  A row's
//! identity is the task's plus the subject, the pinned source versions and
//! the cutoff ([`TaskManifest::row_fingerprint`]): a cohort-membership digest
//! or a URI alone pins no data.
//!
//! **Patient splits precede window construction**: a partition belongs to a
//! *subject*, and every row of the subject inherits it, so no window can put
//! one patient on both sides.  [`TaskManifest::validate`] checks the manifest
//! without opening a file; [`preflight`](super::view::preflight) checks the
//! sources it pins.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::OnceLock;

use serde_json::{json, Map, Value};

use super::source::SourceRef;
use super::{fingerprint, Finding};
use crate::clinical::select::SelectionPolicy;
use crate::json::repr_str;
use crate::{Error, Result};

/// The manifest's own version.
pub const SCHEMA: &str = "medh5.task/1";
/// Its JSON Schema's file name, beside the specification.
pub const SCHEMA_FILE: &str = "medh5-task-1.schema.json";
const SCHEMA_TEXT: &str = include_str!("../../data/medh5-task-1.schema.json");

/// The bundled task schema, as text.
pub fn schema_text() -> &'static str {
    SCHEMA_TEXT
}

fn validator() -> &'static jsonschema::Validator {
    static V: OnceLock<jsonschema::Validator> = OnceLock::new();
    V.get_or_init(|| {
        crate::document::compile(&serde_json::from_str(SCHEMA_TEXT).expect("bundled task schema is valid JSON"))
    })
}

/// How a slot's region is centred.
pub const ROI: [&str; 2] = ["center", "eligible_instances"];
/// What a target does with a row it cannot label.
pub const CENSORING: [&str; 2] = ["censor", "exclude"];

/// A stable modality slot (task-and-cache contract §3.5): filled per row by
/// the newest eligible image of its modality, never by a filename.
#[derive(Debug, Clone, PartialEq)]
pub struct Slot {
    pub name: String,
    pub modality: String,
    /// A row without an eligible image for a required slot is excluded.
    pub required: bool,
    /// The region read around the centre, in voxels of the image's own grid;
    /// `None` reads the whole volume.
    pub patch: Option<Vec<usize>>,
    /// `center` or `eligible_instances`.
    pub roi: String,
    /// Classes to read labels for, from eligible annotations on the grid.
    pub classes: Vec<i64>,
}

impl Slot {
    pub fn to_json(&self) -> Value {
        json!({
            "name": self.name,
            "modality": self.modality,
            "required": self.required,
            "patch": self.patch,
            "roi": self.roi,
            "classes": self.classes,
        })
    }

    fn from_json(v: &Value) -> Result<Slot> {
        Ok(Slot {
            name: v["name"].as_str().unwrap_or_default().to_string(),
            modality: v["modality"].as_str().unwrap_or_default().to_string(),
            required: v.get("required").and_then(Value::as_bool).unwrap_or(false),
            patch: v
                .get("patch")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_u64).map(|n| n as usize).collect()),
            roi: v.get("roi").and_then(Value::as_str).unwrap_or("center").to_string(),
            classes: v
                .get("classes")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_i64).collect())
                .unwrap_or_default(),
        })
    }
}

/// What a row is labelled with (task-and-cache contract §5).
#[derive(Debug, Clone, PartialEq)]
pub struct TargetSpec {
    pub id: String,
    pub version: String,
    /// The event kind the target reads, when it narrows the concept.
    pub kind: Option<String>,
    pub code_system: String,
    pub code: String,
    /// `value_text`s that are a positive outcome.
    pub positive: Vec<String>,
    /// `value_text`s that are a verified negative outcome.
    pub negative: Vec<String>,
    /// The window after the cutoff an outcome must fall in: `(c, c + h]`.
    pub horizon_us: i64,
    /// A negative needs an observation at least this long after the cutoff.
    pub min_follow_up_us: i64,
    /// `censor` (keep the row, target unobserved) or `exclude` (drop it).
    pub censoring: String,
    /// Exclude rows whose outcome had already occurred by the cutoff.
    pub exclude_prevalent: bool,
}

impl TargetSpec {
    pub fn to_json(&self) -> Value {
        let mut event = Map::new();
        if let Some(k) = &self.kind {
            event.insert("kind".into(), json!(k));
        }
        event.insert("code_system".into(), json!(self.code_system));
        event.insert("code".into(), json!(self.code));
        json!({
            "id": self.id,
            "version": self.version,
            "event": event,
            "positive": self.positive,
            "negative": self.negative,
            "horizon_us": self.horizon_us,
            "min_follow_up_us": self.min_follow_up_us,
            "censoring": self.censoring,
            "exclude_prevalent": self.exclude_prevalent,
        })
    }

    fn from_json(v: &Value) -> Result<TargetSpec> {
        let strings = |k: &str| -> Vec<String> {
            v.get(k)
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect())
                .unwrap_or_default()
        };
        let horizon = micros(&v["horizon_us"], "target horizon_us")?;
        let min_follow_up_us = match v.get("min_follow_up_us") {
            None | Some(Value::Null) => horizon,
            Some(m) => micros(m, "target min_follow_up_us")?,
        };
        Ok(TargetSpec {
            id: v["id"].as_str().unwrap_or_default().to_string(),
            version: v["version"].as_str().unwrap_or_default().to_string(),
            kind: v["event"].get("kind").and_then(Value::as_str).map(str::to_string),
            code_system: v["event"]["code_system"].as_str().unwrap_or_default().to_string(),
            code: v["event"]["code"].as_str().unwrap_or_default().to_string(),
            positive: strings("positive"),
            negative: strings("negative"),
            horizon_us: horizon,
            min_follow_up_us,
            censoring: v.get("censoring").and_then(Value::as_str).unwrap_or("censor").to_string(),
            exclude_prevalent: v.get("exclude_prevalent").and_then(Value::as_bool).unwrap_or(true),
        })
    }
}

/// A time in microseconds, as the manifest states it: a 64-bit integer, and
/// refused otherwise (T101).  JSON Schema's `integer` admits `3600000000.0`
/// and values past `i64`, and reading either as 0 moved a row's cutoff, and
/// with it what the row may read, without a finding.
fn micros(value: &Value, what: &str) -> Result<i64> {
    value
        .as_i64()
        .ok_or_else(|| Error::coded("T101", format!("{what} is {value}, not a 64-bit integer number of microseconds")))
}

/// An event version present in several fragments, and the digest they share.
#[derive(Debug, Clone, PartialEq)]
pub struct Reconciled {
    pub event_id: String,
    pub digest: String,
    pub sources: Vec<String>,
}

/// One subject: its fragments, its clock and its partition.
#[derive(Debug, Clone, PartialEq)]
pub struct Subject {
    pub subject_id: String,
    pub clock_id: Option<String>,
    pub partition: Option<String>,
    pub sources: Vec<SourceRef>,
    pub reconciled: Vec<Reconciled>,
}

impl Subject {
    pub fn to_json(&self) -> Value {
        json!({
            "subject_id": self.subject_id,
            "clock_id": self.clock_id,
            "partition": self.partition,
            "sources": self.sources.iter().map(SourceRef::to_json).collect::<Vec<_>>(),
            "reconciled": self.reconciled.iter().map(|r| json!({
                "event_id": r.event_id, "digest": r.digest, "sources": r.sources,
            })).collect::<Vec<_>>(),
        })
    }
}

/// One example: a subject at a cutoff.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Row {
    pub row_id: String,
    pub subject_id: String,
    pub cutoff_us: i64,
}

/// A parsed task manifest.
#[derive(Debug, Clone, PartialEq)]
pub struct TaskManifest {
    pub task_id: String,
    pub task_version: String,
    pub description: Option<String>,
    pub identity_namespace: String,
    pub policy: SelectionPolicy,
    pub slots: Vec<Slot>,
    pub target: Option<TargetSpec>,
    /// `(set_id, partitions)`.
    pub split: Option<(String, Vec<String>)>,
    pub subjects: Vec<Subject>,
    pub rows: Vec<Row>,
    /// The fingerprint the manifest declared, if it declared one.
    pub fingerprint: Option<String>,
}

impl TaskManifest {
    /// An empty manifest for a task.
    pub fn new(task_id: &str, task_version: &str, identity_namespace: &str) -> TaskManifest {
        TaskManifest {
            task_id: task_id.into(),
            task_version: task_version.into(),
            description: None,
            identity_namespace: identity_namespace.into(),
            policy: SelectionPolicy::strict(),
            slots: Vec::new(),
            target: None,
            split: None,
            subjects: Vec::new(),
            rows: Vec::new(),
            fingerprint: None,
        }
    }

    /// Schema messages for a manifest document (empty when it conforms).
    pub fn schema_messages(doc: &Value) -> Vec<String> {
        crate::document::schema_messages(validator(), doc)
    }

    /// Parse a manifest; a document its schema rejects is a T101 error.
    pub fn from_json(doc: &Value) -> Result<TaskManifest> {
        let messages = TaskManifest::schema_messages(doc);
        if !messages.is_empty() {
            return Err(Error::coded("T101", format!("the task manifest fails its schema: {}", messages.join("; "))));
        }
        let policy = match doc.get("policy") {
            Some(p) => SelectionPolicy::from_json(p).map_err(|e| Error::coded("T102", e.message().to_string()))?,
            None => SelectionPolicy::strict(),
        };
        let subjects = doc["subjects"]
            .as_array()
            .map(|items| {
                items
                    .iter()
                    .map(|s| {
                        Ok(Subject {
                            subject_id: s["subject_id"].as_str().unwrap_or_default().to_string(),
                            clock_id: s.get("clock_id").and_then(Value::as_str).map(str::to_string),
                            partition: s.get("partition").and_then(Value::as_str).map(str::to_string),
                            sources: s["sources"]
                                .as_array()
                                .into_iter()
                                .flatten()
                                .map(SourceRef::from_json)
                                .collect::<Result<_>>()?,
                            reconciled: s
                                .get("reconciled")
                                .and_then(Value::as_array)
                                .into_iter()
                                .flatten()
                                .map(|r| Reconciled {
                                    event_id: r["event_id"].as_str().unwrap_or_default().to_string(),
                                    digest: r["digest"].as_str().unwrap_or_default().to_string(),
                                    sources: r["sources"]
                                        .as_array()
                                        .into_iter()
                                        .flatten()
                                        .filter_map(Value::as_str)
                                        .map(str::to_string)
                                        .collect(),
                                })
                                .collect(),
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default();
        Ok(TaskManifest {
            task_id: doc["task"]["id"].as_str().unwrap_or_default().to_string(),
            task_version: doc["task"]["version"].as_str().unwrap_or_default().to_string(),
            description: doc["task"].get("description").and_then(Value::as_str).map(str::to_string),
            identity_namespace: doc["identity_namespace"].as_str().unwrap_or_default().to_string(),
            policy,
            slots: doc
                .get("slots")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .map(Slot::from_json)
                .collect::<Result<_>>()?,
            target: match doc.get("target") {
                None | Some(Value::Null) => None,
                Some(t) => Some(TargetSpec::from_json(t)?),
            },
            split: doc.get("split").filter(|s| !s.is_null()).map(|s| {
                (
                    s["set_id"].as_str().unwrap_or_default().to_string(),
                    s["partitions"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect(),
                )
            }),
            subjects,
            rows: doc["rows"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|r| {
                    Ok(Row {
                        row_id: r["row_id"].as_str().unwrap_or_default().to_string(),
                        subject_id: r["subject_id"].as_str().unwrap_or_default().to_string(),
                        cutoff_us: micros(&r["cutoff_us"], "a row's cutoff_us")?,
                    })
                })
                .collect::<Result<_>>()?,
            fingerprint: doc.get("fingerprint").and_then(Value::as_str).map(str::to_string),
        })
    }

    /// Read a manifest file.
    pub fn load(path: &Path) -> Result<TaskManifest> {
        let text = std::fs::read_to_string(path).map_err(|e| crate::error::os_error(&e, path))?;
        let doc = crate::json::loads(&text)
            .map_err(|e| Error::coded("T101", format!("the task manifest is not JSON: {e}")))?;
        TaskManifest::from_json(&doc)
    }

    /// The task's definition: what identifies the task, defaults filled in.
    pub fn definition(&self) -> Value {
        json!({
            "schema": SCHEMA,
            "task": {"id": self.task_id, "version": self.task_version},
            "identity_namespace": self.identity_namespace,
            "policy": self.policy.to_json(),
            "slots": self.slots.iter().map(Slot::to_json).collect::<Vec<_>>(),
            "target": self.target.as_ref().map(TargetSpec::to_json),
        })
    }

    /// The whole manifest, normalised; `fingerprint` is the declared one.
    pub fn to_json(&self) -> Value {
        let mut out = match self.definition() {
            Value::Object(m) => m,
            _ => Map::new(),
        };
        if let Some(d) = &self.description {
            out["task"]["description"] = json!(d);
        }
        if let Some((set_id, partitions)) = &self.split {
            out.insert("split".into(), json!({"set_id": set_id, "partitions": partitions}));
        }
        out.insert("subjects".into(), Value::Array(self.subjects.iter().map(Subject::to_json).collect()));
        out.insert(
            "rows".into(),
            Value::Array(
                self.rows
                    .iter()
                    .map(|r| json!({"row_id": r.row_id, "subject_id": r.subject_id, "cutoff_us": r.cutoff_us}))
                    .collect(),
            ),
        );
        if let Some(f) = &self.fingerprint {
            out.insert("fingerprint".into(), json!(f));
        }
        Value::Object(out)
    }

    /// Write the manifest with its fingerprint, pretty-printed.
    pub fn save(&self, path: &Path) -> Result<()> {
        let mut doc = self.to_json();
        doc["fingerprint"] = json!(self.manifest_fingerprint());
        std::fs::write(path, crate::json::pretty(&doc) + "\n")?;
        Ok(())
    }

    /// Identifies the task: its canonical definition's digest.
    pub fn task_fingerprint(&self) -> String {
        fingerprint(&self.definition())
    }

    /// Identifies the whole manifest, rows and pins included.
    pub fn manifest_fingerprint(&self) -> String {
        let mut doc = self.to_json();
        if let Value::Object(m) = &mut doc {
            m.remove("fingerprint");
        }
        fingerprint(&doc)
    }

    pub fn subject(&self, subject_id: &str) -> Option<&Subject> {
        self.subjects.iter().find(|s| s.subject_id == subject_id)
    }

    pub fn partition_of(&self, subject_id: &str) -> Option<&str> {
        self.subject(subject_id).and_then(|s| s.partition.as_deref())
    }

    /// Identifies one example: the task, the subject, every pinned source
    /// version and the cutoff.
    pub fn row_fingerprint(&self, row: &Row) -> String {
        self.row_fingerprint_with(&self.task_fingerprint(), self.subject(&row.subject_id), row)
    }

    /// [`row_fingerprint`](Self::row_fingerprint) with the task fingerprint
    /// and the row's subject already in hand: what a preflight of many rows
    /// computes once.
    pub fn row_fingerprint_with(&self, task_fingerprint: &str, subject: Option<&Subject>, row: &Row) -> String {
        let mut pins: Vec<&str> =
            subject.map(|s| s.sources.iter().map(|r| r.content_id.as_str()).collect()).unwrap_or_default();
        pins.sort_unstable();
        fingerprint(&json!({
            "task": task_fingerprint,
            "namespace": self.identity_namespace,
            "subject_id": row.subject_id,
            "sources": pins,
            "cutoff_us": row.cutoff_us,
        }))
    }

    /// The digest of who is in a partition: what learned preprocessing
    /// records as the split it was fitted on.
    pub fn subjects_digest(&self, partition: &str) -> String {
        let mut members: Vec<(&str, &str)> = self
            .subjects
            .iter()
            .filter(|s| s.partition.as_deref() == Some(partition))
            .map(|s| (self.identity_namespace.as_str(), s.subject_id.as_str()))
            .collect();
        members.sort_unstable();
        fingerprint(&json!(members))
    }

    /// The rows of one partition, in manifest order.
    pub fn rows_in(&self, partition: &str) -> Vec<&Row> {
        self.rows.iter().filter(|r| self.partition_of(&r.subject_id) == Some(partition)).collect()
    }

    /// Everything wrong with the manifest that opening no file can find.
    pub fn validate(&self) -> Vec<Finding> {
        let mut out = Vec::new();
        let doc = self.to_json();
        for m in TaskManifest::schema_messages(&doc) {
            out.push(Finding::new("T101", "/", m));
        }
        if let Err(e) = self.policy.check() {
            out.push(Finding::new("T102", "/policy", e.message()));
        }
        let mut slots = BTreeSet::new();
        for (i, slot) in self.slots.iter().enumerate() {
            if !slots.insert(slot.name.as_str()) {
                out.push(Finding::new(
                    "T102",
                    format!("/slots/{i}"),
                    format!("slot {} is declared twice", repr_str(&slot.name)),
                ));
            }
            if !ROI.contains(&slot.roi.as_str()) {
                out.push(Finding::new(
                    "T102",
                    format!("/slots/{i}"),
                    format!("roi {} is not one of {ROI:?}", repr_str(&slot.roi)),
                ));
            }
        }
        if let Some(t) = &self.target {
            let overlap: Vec<&String> = t.positive.iter().filter(|p| t.negative.contains(p)).collect();
            if !overlap.is_empty() {
                out.push(Finding::new("T102", "/target", format!("{overlap:?} are both positive and negative")));
            }
            if t.min_follow_up_us > t.horizon_us {
                out.push(Finding::new(
                    "T102",
                    "/target",
                    "min_follow_up_us exceeds horizon_us: no observation inside the window could ever be a negative",
                ));
            }
            if !CENSORING.contains(&t.censoring.as_str()) {
                out.push(Finding::new(
                    "T102",
                    "/target",
                    format!("censoring {} is not one of {CENSORING:?}", repr_str(&t.censoring)),
                ));
            }
        }
        if let Some(declared) = &self.fingerprint {
            let computed = self.manifest_fingerprint();
            if declared != &computed {
                out.push(Finding::new(
                    "T103",
                    "/fingerprint",
                    format!("declares {declared}; the manifest is {computed}"),
                ));
            }
        }
        // T201, T202: subjects, rows, partitions.
        let mut seen: BTreeSet<&str> = BTreeSet::new();
        for (i, s) in self.subjects.iter().enumerate() {
            if !seen.insert(s.subject_id.as_str()) {
                out.push(Finding::new(
                    "T201",
                    format!("/subjects/{i}"),
                    format!("subject {} is declared twice", repr_str(&s.subject_id)),
                ));
            }
            match (&self.split, &s.partition) {
                (Some((set_id, partitions)), Some(p)) if !partitions.contains(p) => out.push(Finding::new(
                    "T202",
                    format!("/subjects/{i}"),
                    format!("partition {} is not one of split {}'s {partitions:?}", repr_str(p), repr_str(set_id)),
                )),
                (Some((set_id, _)), None) => out.push(Finding::new(
                    "T202",
                    format!("/subjects/{i}"),
                    format!(
                        "subject {} has no partition in split {}: subjects are split before rows are built",
                        repr_str(&s.subject_id),
                        repr_str(set_id)
                    ),
                )),
                (None, Some(p)) => out.push(Finding::new(
                    "T202",
                    format!("/subjects/{i}"),
                    format!("partition {} names no declared split", repr_str(p)),
                )),
                _ => {}
            }
        }
        let mut rows = BTreeSet::new();
        for (i, r) in self.rows.iter().enumerate() {
            if !seen.contains(r.subject_id.as_str()) {
                out.push(Finding::new(
                    "T201",
                    format!("/rows/{i}"),
                    format!("row {} names undeclared subject {}", repr_str(&r.row_id), repr_str(&r.subject_id)),
                ));
            }
            if !rows.insert(r.row_id.as_str()) {
                out.push(Finding::new(
                    "T204",
                    format!("/rows/{i}"),
                    format!("row id {} is not unique", repr_str(&r.row_id)),
                ));
            }
        }
        // T203, T204: sources.
        let mut by_locator: BTreeMap<(String, Option<String>), &str> = BTreeMap::new();
        let mut by_pin: BTreeMap<&str, &str> = BTreeMap::new();
        let mut ids: BTreeSet<&str> = BTreeSet::new();
        for (i, s) in self.subjects.iter().enumerate() {
            for (j, src) in s.sources.iter().enumerate() {
                let at = format!("/subjects/{i}/sources/{j}");
                if !ids.insert(src.source_id.as_str()) {
                    out.push(Finding::new(
                        "T204",
                        &at,
                        format!("source id {} is not unique", repr_str(&src.source_id)),
                    ));
                }
                let key = (src.uri.clone(), src.sample_key.clone());
                if let Some(other) = by_locator.insert(key, s.subject_id.as_str()) {
                    if other != s.subject_id {
                        out.push(Finding::new(
                            "T203",
                            &at,
                            format!(
                                "{} is a source of both {} and {}",
                                src.locator(),
                                repr_str(other),
                                repr_str(&s.subject_id)
                            ),
                        ));
                    }
                }
                if let Some(other) = by_pin.insert(src.content_id.as_str(), s.subject_id.as_str()) {
                    if other != s.subject_id {
                        out.push(Finding::new(
                            "T203",
                            &at,
                            format!(
                                "subjects {} and {} pin one sample ({}): an overlapping snapshot would cross the split",
                                repr_str(other),
                                repr_str(&s.subject_id),
                                src.content_id
                            ),
                        ));
                    }
                }
            }
        }
        out
    }
}
