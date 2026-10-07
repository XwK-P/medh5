//! Cohort manifests: what a metadata-only scan of a directory tree knows.
//!
//! One metadata-only pass writes a JSON file, and splitting, stratification,
//! filtering and cohort checks all run against it without touching a voxel.
//! The manifest is also the **authority for splits** (§12.3): its `sha256` is
//! what lets a reader notice that a file's claim predates the current split.

use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use serde_json::{json, Map, Value};

use crate::collection::{open_any, AnyFile};
use crate::json::{canonical, num, pretty, repr, repr_str};
use crate::sample::Sample;
use crate::{Error, Result};

pub const SUFFIXES: [&str; 2] = [".medh5", ".medh5c"];

/// Fields worth grouping or stratifying on --- single-valued and scannable.
pub const GROUPABLE: [&str; 10] = [
    "subject_id",
    "group_id",
    "site_id",
    "scanner_id",
    "dataset_id",
    "acquisition_protocol",
    "sex",
    "laterality",
    "bodypart",
    "label_set_id",
];

/// Every field of an [`Entry`], as `--group-by` and `--stratify-by` name them.
pub const ENTRY_FIELDS: [&str; 29] = [
    "path",
    "sample_id",
    "subject_id",
    "group_id",
    "content_id",
    "profiles",
    "sex",
    "laterality",
    "bodypart",
    "dataset_id",
    "site_id",
    "scanner_id",
    "acquisition_protocol",
    "timepoints",
    "days_from_baseline",
    "images",
    "modalities",
    "annotations",
    "class_ids",
    "annotated_class_ids",
    "label_set_id",
    "label_set_version",
    "label_set_digest",
    "quality",
    "splits",
    "deidentified",
    "key",
    "size",
    "mtime",
];

/// One sample's metadata, as far as a cohort needs it.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Entry {
    pub path: String,
    pub sample_id: String,
    pub subject_id: String,
    pub group_id: String,
    pub content_id: Option<String>,
    pub profiles: Vec<String>,
    pub sex: Option<String>,
    pub laterality: Option<String>,
    pub bodypart: Option<String>,
    pub dataset_id: Option<String>,
    pub site_id: Option<String>,
    pub scanner_id: Option<String>,
    pub acquisition_protocol: Option<String>,
    pub timepoints: Vec<String>,
    /// As each timepoint stores it: an integer, a float, or absent.
    pub days_from_baseline: Vec<Value>,
    pub images: Vec<String>,
    pub modalities: Vec<String>,
    pub annotations: Map<String, Value>,
    pub class_ids: Vec<i64>,
    pub annotated_class_ids: Vec<i64>,
    pub label_set_id: Option<String>,
    pub label_set_version: Option<String>,
    pub label_set_digest: Option<String>,
    pub quality: IndexMap<String, String>,
    pub splits: Vec<Map<String, Value>>,
    pub deidentified: bool,
    /// The sample's key inside a collection, or `None` for a plain file.
    pub key: Option<String>,
    pub size: u64,
    pub mtime: f64,
}

fn py_tuple(items: &[Value]) -> String {
    let inner: Vec<String> = items.iter().map(repr).collect();
    if inner.len() == 1 {
        format!("({},)", inner[0])
    } else {
        format!("({})", inner.join(", "))
    }
}

fn opt(value: &Option<String>) -> Value {
    value.as_ref().map(|v| Value::String(v.clone())).unwrap_or(Value::Null)
}

impl Entry {
    pub fn is_longitudinal(&self) -> bool {
        self.timepoints.len() > 1
    }

    /// Whether any annotation *names* the class --- see [`Entry::examined`].
    pub fn has_class(&self, class_id: i64) -> bool {
        self.class_ids.contains(&class_id)
    }

    /// Whether the class was looked for (§11.3), found or not: a sample never
    /// examined for a class is not a negative example of it.
    pub fn examined(&self, class_id: i64) -> bool {
        self.annotated_class_ids.contains(&class_id)
    }

    pub fn to_json(&self) -> Value {
        let mut out = Map::new();
        out.insert("path".into(), json!(self.path));
        out.insert("sample_id".into(), json!(self.sample_id));
        out.insert("subject_id".into(), json!(self.subject_id));
        out.insert("group_id".into(), json!(self.group_id));
        out.insert("content_id".into(), opt(&self.content_id));
        out.insert("profiles".into(), json!(self.profiles));
        out.insert("timepoints".into(), json!(self.timepoints));
        out.insert("days_from_baseline".into(), Value::Array(self.days_from_baseline.clone()));
        out.insert("images".into(), json!(self.images));
        out.insert("modalities".into(), json!(self.modalities));
        out.insert("annotations".into(), Value::Object(self.annotations.clone()));
        out.insert("class_ids".into(), json!(self.class_ids));
        out.insert("annotated_class_ids".into(), json!(self.annotated_class_ids));
        out.insert("quality".into(), json!(self.quality));
        out.insert("splits".into(), Value::Array(self.splits.iter().cloned().map(Value::Object).collect()));
        out.insert("deidentified".into(), json!(self.deidentified));
        out.insert("size".into(), json!(self.size));
        out.insert("mtime".into(), num(self.mtime));
        for (name, value) in [
            ("sex", &self.sex),
            ("laterality", &self.laterality),
            ("bodypart", &self.bodypart),
            ("dataset_id", &self.dataset_id),
            ("site_id", &self.site_id),
            ("scanner_id", &self.scanner_id),
            ("acquisition_protocol", &self.acquisition_protocol),
            ("label_set_id", &self.label_set_id),
            ("label_set_version", &self.label_set_version),
            ("label_set_digest", &self.label_set_digest),
            ("key", &self.key),
        ] {
            if let Some(v) = value {
                out.insert(name.into(), json!(v));
            }
        }
        Value::Object(out)
    }

    pub fn from_json(doc: &Value) -> Result<Entry> {
        let get = |k: &str| doc.get(k).cloned().unwrap_or(Value::Null);
        let string = |k: &str| -> Result<String> {
            match doc.get(k) {
                Some(Value::String(s)) => Ok(s.clone()),
                Some(Value::Null) | None => Err(Error::Key(repr_str(k))),
                Some(other) => Ok(crate::json::py_str(other)),
            }
        };
        let opt_str = |k: &str| doc.get(k).and_then(Value::as_str).map(str::to_string);
        let list_str = |k: &str| -> Vec<String> {
            get(k).as_array().map(|a| a.iter().map(crate::json::py_str).collect()).unwrap_or_default()
        };
        let list_int = |k: &str| -> Vec<i64> {
            get(k).as_array().map(|a| a.iter().filter_map(Value::as_i64).collect()).unwrap_or_default()
        };
        Ok(Entry {
            path: string("path")?,
            sample_id: string("sample_id")?,
            subject_id: string("subject_id")?,
            group_id: string("group_id")?,
            content_id: opt_str("content_id"),
            profiles: list_str("profiles"),
            sex: opt_str("sex"),
            laterality: opt_str("laterality"),
            bodypart: opt_str("bodypart"),
            dataset_id: opt_str("dataset_id"),
            site_id: opt_str("site_id"),
            scanner_id: opt_str("scanner_id"),
            acquisition_protocol: opt_str("acquisition_protocol"),
            timepoints: list_str("timepoints"),
            days_from_baseline: get("days_from_baseline").as_array().cloned().unwrap_or_default(),
            images: list_str("images"),
            modalities: list_str("modalities"),
            annotations: get("annotations").as_object().cloned().unwrap_or_default(),
            class_ids: list_int("class_ids"),
            annotated_class_ids: list_int("annotated_class_ids"),
            label_set_id: opt_str("label_set_id"),
            label_set_version: opt_str("label_set_version"),
            label_set_digest: opt_str("label_set_digest"),
            quality: get("quality")
                .as_object()
                .map(|m| m.iter().map(|(k, v)| (k.clone(), crate::json::py_str(v))).collect())
                .unwrap_or_default(),
            splits: get("splits")
                .as_array()
                .map(|a| a.iter().filter_map(|s| s.as_object().cloned()).collect())
                .unwrap_or_default(),
            deidentified: get("deidentified").as_bool().unwrap_or(false),
            key: opt_str("key"),
            size: get("size").as_u64().unwrap_or(0),
            mtime: get("mtime").as_f64().unwrap_or(0.0),
        })
    }

    /// A field by the dotted name the CLI accepts: `cohort.site_id`,
    /// `identity.sex` and bare `site_id` all reach the same value.
    pub fn field(&self, dotted: &str) -> Result<Value> {
        let name = dotted.rsplit('.').next().unwrap_or(dotted);
        if !ENTRY_FIELDS.contains(&name) {
            let mut sorted = GROUPABLE.to_vec();
            sorted.sort_unstable();
            return Err(Error::invalid(format!(
                "{} is not a manifest field; try one of {}",
                repr_str(dotted),
                sorted.join(", ")
            )));
        }
        Ok(match self.to_json() {
            Value::Object(mut m) => m.remove(name).unwrap_or(Value::Null),
            _ => Value::Null,
        })
    }

    /// `str(entry.field(name))`, as the 1.x grouping and stratification keyed
    /// on it: tuple-valued fields print as Python tuples.
    pub fn field_str(&self, dotted: &str) -> Result<String> {
        let name = dotted.rsplit('.').next().unwrap_or(dotted);
        let value = self.field(dotted)?;
        let tuple_valued = matches!(
            name,
            "profiles"
                | "timepoints"
                | "days_from_baseline"
                | "images"
                | "modalities"
                | "class_ids"
                | "annotated_class_ids"
                | "splits"
        );
        Ok(match (&value, tuple_valued) {
            (Value::Array(items), true) => py_tuple(items),
            (Value::String(s), _) => s.clone(),
            (other, _) => repr(other),
        })
    }
}

/// A cohort as metadata, with the digest that makes a split checkable.
#[derive(Debug, Clone, PartialEq)]
pub struct Manifest {
    pub entries: Vec<Entry>,
    pub root: Option<String>,
    pub generator: String,
    pub format: String,
}

impl Default for Manifest {
    fn default() -> Self {
        Manifest {
            entries: Vec::new(),
            root: None,
            generator: format!("medh5 {}", crate::VERSION),
            format: crate::FORMAT_VERSION.into(),
        }
    }
}

impl Manifest {
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn by_path(&self, path: &str) -> Option<&Entry> {
        self.entries.iter().find(|e| e.path == path)
    }

    /// A manifest over the entries that satisfy `predicate`; its digest
    /// changes with its contents.
    pub fn filter(&self, predicate: impl Fn(&Entry) -> bool) -> Manifest {
        Manifest {
            entries: self.entries.iter().filter(|e| predicate(e)).cloned().collect(),
            root: self.root.clone(),
            generator: self.generator.clone(),
            format: self.format.clone(),
        }
    }

    /// Entries by the string value of a field, in first-seen order.
    pub fn groups(&self, by: &str) -> Result<IndexMap<String, Vec<&Entry>>> {
        let mut out: IndexMap<String, Vec<&Entry>> = IndexMap::new();
        for entry in &self.entries {
            out.entry(entry.field_str(by)?).or_default().push(entry);
        }
        Ok(out)
    }

    pub fn subjects(&self) -> Vec<String> {
        let mut out: Vec<String> = self.entries.iter().map(|e| e.subject_id.clone()).collect();
        out.sort();
        out.dedup();
        out
    }

    pub fn to_json(&self) -> Value {
        json!({
            "format": self.format,
            "generator": self.generator,
            "root": self.root,
            "sha256": self.sha256(),
            "samples": self.entries.len(),
            "entries": self.entries.iter().map(Entry::to_json).collect::<Vec<_>>(),
        })
    }

    /// Digest of the cohort's *membership*: which samples, grouped how.
    ///
    /// Only `sample_id`, `subject_id` and `group_id` are hashed --- not paths,
    /// sizes, mtimes, and deliberately not `content_id`: writing a claim into
    /// a file changes its content, so every claim would be stale the instant
    /// it was written.  Content drift is `dataset check`'s question (C401).
    pub fn sha256(&self) -> String {
        use sha2::{Digest, Sha256};
        let mut sorted: Vec<&Entry> = self.entries.iter().collect();
        sorted.sort_by(|a, b| (&a.sample_id, &a.path).cmp(&(&b.sample_id, &b.path)));
        let payload: Vec<Value> = sorted
            .iter()
            .map(|e| json!({"sample_id": e.sample_id, "subject_id": e.subject_id, "group_id": e.group_id}))
            .collect();
        hex::encode(Sha256::digest(canonical(&json!({"entries": payload})).as_bytes()))
    }

    pub fn save(&self, path: &Path) -> Result<PathBuf> {
        std::fs::write(path, pretty(&self.to_json()) + "\n")?;
        Ok(path.to_path_buf())
    }

    pub fn load(path: &Path) -> Result<Manifest> {
        let text = std::fs::read_to_string(path)?;
        let doc = crate::json::loads(&text).map_err(|e| Error::Value(e.to_string()))?;
        Manifest::from_json(&doc)
    }

    pub fn from_json(doc: &Value) -> Result<Manifest> {
        let entries = match doc.get("entries") {
            Some(Value::Array(items)) => items.iter().map(Entry::from_json).collect::<Result<_>>()?,
            _ => Vec::new(),
        };
        Ok(Manifest {
            entries,
            root: doc.get("root").and_then(Value::as_str).map(str::to_string),
            generator: doc.get("generator").map(crate::json::py_str).unwrap_or_default(),
            format: doc.get("format").map(crate::json::py_str).unwrap_or_else(|| crate::FORMAT_VERSION.into()),
        })
    }

    /// Paths whose file no longer matches what the manifest recorded.
    ///
    /// Size and mtime are the cheap check; `content_id` is the proof, and
    /// `dataset check --deep` uses it.
    pub fn stale(&self) -> Vec<String> {
        let mut out = Vec::new();
        for entry in &self.entries {
            match std::fs::metadata(&entry.path) {
                Err(_) => out.push(entry.path.clone()),
                Ok(meta) => {
                    if meta.len() != entry.size || (st_mtime(&meta) - entry.mtime).abs() > 1e-6 {
                        out.push(entry.path.clone());
                    }
                }
            }
        }
        out
    }
}

/// `os.stat(path).st_mtime`: seconds plus nanoseconds, as CPython adds them.
pub fn st_mtime(meta: &std::fs::Metadata) -> f64 {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        meta.mtime() as f64 + meta.mtime_nsec() as f64 * 1e-9
    }
    #[cfg(not(unix))]
    {
        match meta.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()) {
            Some(d) => d.as_secs() as f64 + f64::from(d.subsec_nanos()) * 1e-9,
            None => 0.0,
        }
    }
}

/// Every sample and collection under `root`, in a stable order.
pub fn find(root: &Path) -> Result<Vec<PathBuf>> {
    find_with(root, &SUFFIXES)
}

/// [`find`] for other file suffixes (each with its leading dot).
pub fn find_with(root: &Path, suffixes: &[&str]) -> Result<Vec<PathBuf>> {
    if root.is_file() {
        return Ok(vec![root.to_path_buf()]);
    }
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(listing) = std::fs::read_dir(&dir) else { continue };
        for item in listing.flatten() {
            let path = item.path();
            let is_dir = item.file_type().map(|t| t.is_dir()).unwrap_or(false);
            if is_dir {
                stack.push(path);
            } else if path.is_file() {
                let suffix = path.extension().map(|e| format!(".{}", e.to_string_lossy())).unwrap_or_default();
                if suffixes.contains(&suffix.as_str()) {
                    found.push(path);
                }
            }
        }
    }
    found.sort();
    Ok(found)
}

/// Build a manifest from a directory tree, and report what would not open.
///
/// A cohort scan that dies on one broken file has told you nothing about the
/// other 9 999, so failures are collected unless `strict`.
pub fn scan(root: &Path, strict: bool) -> Result<(Manifest, Vec<String>)> {
    scan_with(root, &SUFFIXES, strict)
}

/// [`scan`] over the files [`find_with`] finds.
pub fn scan_with(root: &Path, suffixes: &[&str], strict: bool) -> Result<(Manifest, Vec<String>)> {
    let mut manifest = Manifest { root: Some(root.to_string_lossy().into_owned()), ..Default::default() };
    let mut failures = Vec::new();
    for path in find_with(root, suffixes)? {
        match entries_for(&path) {
            Ok(entries) => manifest.entries.extend(entries),
            Err(e) if e.is_medh5() || matches!(e, Error::Io(_)) => {
                if strict {
                    return Err(e);
                }
                failures.push(format!("{}: {}", path.display(), e.python_str()));
            }
            Err(e) => return Err(e),
        }
    }
    Ok((manifest, failures))
}

/// The manifest entries in one file: one per sample, so a collection fans out.
pub fn entries_for(path: &Path) -> Result<Vec<Entry>> {
    let meta = std::fs::metadata(path)?;
    let text = path.to_string_lossy().into_owned();
    match open_any(path, None)? {
        AnyFile::Collection(c) => {
            let mut out = Vec::new();
            for key in c.keys()? {
                let sample = c.get(&key)?;
                out.push(entry(&sample, &text, &meta, Some(key))?);
            }
            Ok(out)
        }
        AnyFile::Sample(s) => Ok(vec![entry(&s, &text, &meta, None)?]),
    }
}

fn entry(sample: &Sample, path: &str, meta: &std::fs::Metadata, key: Option<String>) -> Result<Entry> {
    let document = sample.document()?;
    let identity = &document.identity;
    let cohort = &document.cohort;
    let label_set = document.label_set.as_ref();
    let mut annotations = Map::new();
    let mut classes = std::collections::BTreeSet::new();
    let mut annotated = std::collections::BTreeSet::new();
    for (name, annotation) in sample.annotations()? {
        annotations.insert(
            name.clone(),
            json!({
                "kind": annotation.kind(),
                "task": annotation.task(),
                "classes": annotation.class_ids(),
                "annotated_classes": annotation.annotated_class_ids(),
                "timepoints": annotation.timepoints(),
            }),
        );
        classes.extend(annotation.class_ids().iter().copied());
        annotated.extend(annotation.annotated_class_ids().iter().copied());
    }
    let images = sample.images()?;
    let mut modalities: Vec<String> = Vec::new();
    for image in images.values() {
        let modality = image.modality()?;
        if !modality.is_empty() && !modalities.contains(&modality) {
            modalities.push(modality);
        }
    }
    modalities.sort();
    let mut profiles: Vec<String> = sample.profiles()?.into_iter().collect();
    profiles.sort();
    Ok(Entry {
        path: path.to_string(),
        sample_id: identity.sample_id.clone(),
        subject_id: identity.subject_id.clone(),
        group_id: document.group_id().to_string(),
        content_id: sample.content_id()?,
        profiles,
        sex: identity.sex.clone(),
        laterality: identity.laterality.clone(),
        bodypart: identity.bodypart.clone(),
        dataset_id: cohort.dataset_id.clone(),
        site_id: cohort.site_id.clone(),
        scanner_id: cohort.scanner_id.clone(),
        acquisition_protocol: cohort.acquisition_protocol.clone(),
        timepoints: document.timepoints.iter().map(|t| t.id.clone()).collect(),
        days_from_baseline: document
            .timepoints
            .iter()
            .map(|t| t.days_from_baseline.clone().map(Value::Number).unwrap_or(Value::Null))
            .collect(),
        images: images.keys().cloned().collect(),
        modalities,
        annotations,
        class_ids: classes.into_iter().collect(),
        annotated_class_ids: annotated.into_iter().collect(),
        label_set_id: label_set.map(|l| l.id.clone()),
        label_set_version: label_set.map(|l| l.version.clone()),
        label_set_digest: label_set.map(|l| l.sha256()),
        quality: document.quality.iter().map(|(k, v)| (k.clone(), v.status.clone())).collect(),
        splits: document
            .splits
            .iter()
            .filter_map(|s| match s.to_json() {
                Value::Object(m) => Some(m),
                _ => None,
            })
            .collect(),
        deidentified: document.deidentification.is_some(),
        key,
        size: meta.len(),
        mtime: st_mtime(meta),
    })
}

/// How many entries carry each value of `by` --- the stratification tally.
pub fn counts(entries: &[Entry], by: &str) -> Result<std::collections::BTreeMap<String, usize>> {
    let mut out = std::collections::BTreeMap::new();
    for entry in entries {
        *out.entry(entry.field_str(by)?).or_insert(0) += 1;
    }
    Ok(out)
}
