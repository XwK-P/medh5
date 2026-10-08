//! Feature caches (`medh5.cache/1`, task-and-cache contract §7--§8).
//!
//! One HDF5 file:
//!
//! ```text
//! /                    medh5_companion = "medh5.cache/1", manifest_digest
//! ├── manifest         scalar UTF-8, canonical JSON (the dependency manifest)
//! └── entries/<id>     one feature array per entry, with its §13.1 digest
//! ```
//!
//! Two different failures, told apart (§8): a **stale** entry is one whose
//! source no longer has the content it pins --- the cache is right about a
//! sample that changed; a **corrupt** entry is one whose own bytes no longer
//! match their checksum --- the cache is wrong about itself.  Either is
//! rejected and rebuilt; neither says anything about the source, which a
//! derived cache never redefines.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde_json::{json, Value};

use super::source::SourceRef;
use super::task::TaskManifest;
use super::view::Preflight;
use super::{sha256, Finding};
use crate::array::{DType, NdArray};
use crate::h5::attrs::{self, AttrValue};
use crate::h5::data::{self, Layout};
use crate::h5::file::{open_read, AtomicFile};
use crate::h5::ops;
use crate::integrity::array_digest;
use crate::json::repr_str;
use crate::{Error, Result};

/// The cache manifest's own version.
pub const SCHEMA: &str = "medh5.cache/1";
/// The root attribute naming the companion format.
pub const COMPANION_ATTR: &str = "medh5_companion";
/// The root attribute holding the manifest's checksum.
pub const DIGEST_ATTR: &str = "manifest_digest";
/// Conventional file suffix.
pub const SUFFIX: &str = ".medh5cache";
/// Its JSON Schema's file name.
pub const SCHEMA_FILE: &str = "medh5-cache-1.schema.json";
const SCHEMA_TEXT: &str = include_str!("../../data/medh5-cache-1.schema.json");
/// Feature levels.
pub const LEVELS: [&str; 2] = ["event", "patient"];

/// The bundled cache schema, as text.
pub fn schema_text() -> &'static str {
    SCHEMA_TEXT
}

fn validator() -> &'static jsonschema::Validator {
    static V: OnceLock<jsonschema::Validator> = OnceLock::new();
    V.get_or_init(|| {
        crate::document::compile(&serde_json::from_str(SCHEMA_TEXT).expect("bundled cache schema is valid JSON"))
    })
}

/// One cached feature and everything it depends on.
#[derive(Debug, Clone, PartialEq)]
pub struct CacheEntry {
    pub entry_id: String,
    /// Every source version the feature read.
    pub sources: Vec<SourceRef>,
    /// Event level: the event version (and document) the feature encodes.
    pub event_id: Option<String>,
    pub document_id: Option<String>,
    /// Patient level: the row, its cutoff and the versions it selected.
    pub row_id: Option<String>,
    pub cutoff_us: Option<i64>,
    pub event_versions: Option<Vec<String>>,
    /// The payload's §13.1 digest.
    pub digest: String,
}

impl CacheEntry {
    pub fn to_json(&self) -> Value {
        json!({
            "entry_id": self.entry_id,
            "sources": self.sources.iter().map(SourceRef::to_pin_json).collect::<Vec<_>>(),
            "event_id": self.event_id,
            "document_id": self.document_id,
            "row_id": self.row_id,
            "cutoff_us": self.cutoff_us,
            "event_versions": self.event_versions,
            "digest": self.digest,
        })
    }

    fn from_json(v: &Value) -> Result<CacheEntry> {
        let text = |k: &str| v.get(k).and_then(Value::as_str).map(str::to_string);
        Ok(CacheEntry {
            entry_id: text("entry_id").unwrap_or_default(),
            sources: v["sources"].as_array().into_iter().flatten().map(SourceRef::from_json).collect::<Result<_>>()?,
            event_id: text("event_id"),
            document_id: text("document_id"),
            row_id: text("row_id"),
            cutoff_us: v.get("cutoff_us").and_then(Value::as_i64),
            event_versions: v
                .get("event_versions")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect()),
            digest: text("digest").unwrap_or_default(),
        })
    }
}

/// What determines every output of a cache.
#[derive(Debug, Clone, PartialEq)]
pub struct CacheHeader {
    /// `event` or `patient`.
    pub level: String,
    /// `{"name", "revision", "tokenizer"}`: an immutable encoder revision.
    pub encoder: Value,
    pub preprocessing: Value,
    /// `{"dtype", "shape", "pooling", "chunking"}`.
    pub output: Value,
    pub task_fingerprint: Option<String>,
    pub selection: Option<String>,
    /// `{"task_fingerprint", "set_id", "partition", "subjects_digest"}` for
    /// learned preprocessing.
    pub fitted_on: Option<Value>,
}

impl CacheHeader {
    fn manifest(&self, entries: &[CacheEntry]) -> Value {
        json!({
            "schema": SCHEMA,
            "level": self.level,
            "encoder": self.encoder,
            "preprocessing": self.preprocessing,
            "output": self.output,
            "task_fingerprint": self.task_fingerprint,
            "selection": self.selection,
            "fitted_on": self.fitted_on,
            "entries": entries.iter().map(CacheEntry::to_json).collect::<Vec<_>>(),
        })
    }

    fn from_manifest(doc: &Value) -> CacheHeader {
        let opt = |k: &str| doc.get(k).and_then(Value::as_str).map(str::to_string);
        CacheHeader {
            level: opt("level").unwrap_or_default(),
            encoder: doc.get("encoder").cloned().unwrap_or(Value::Null),
            preprocessing: doc.get("preprocessing").cloned().unwrap_or(Value::Null),
            output: doc.get("output").cloned().unwrap_or(Value::Null),
            task_fingerprint: opt("task_fingerprint"),
            selection: opt("selection"),
            fitted_on: doc.get("fitted_on").filter(|v| !v.is_null()).cloned(),
        }
    }

    /// The declared output dtype and shape.
    pub fn output_layout(&self) -> Result<(DType, Vec<usize>)> {
        let dtype = DType::parse(self.output["dtype"].as_str().unwrap_or_default())?;
        let shape = self.output["shape"]
            .as_array()
            .ok_or_else(|| Error::invalid("a cache's output declares its `shape`"))?
            .iter()
            .map(|v| v.as_u64().map(|n| n as usize).ok_or_else(|| Error::invalid("output shape lists sizes")))
            .collect::<Result<_>>()?;
        Ok((dtype, shape))
    }

    /// The fitted-on record for learned preprocessing over `partition` of a
    /// task.
    pub fn fitted_on(task: &TaskManifest, partition: &str) -> Value {
        json!({
            "task_fingerprint": task.task_fingerprint(),
            "set_id": task.split.as_ref().map(|s| s.0.clone()),
            "partition": partition,
            "subjects_digest": task.subjects_digest(partition),
        })
    }
}

/// An entry id: the task id syntax (1.0 id characters, `@` and `:`), so a
/// patient-level entry may be named by its row.
pub fn is_entry_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && id.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-' | '@' | ':'))
}

/// The entry id a feature of `(content_id, event_id)` gets by default.
pub fn event_entry_id(content_id: &str, event_id: &str) -> String {
    let digest = sha256(format!("{content_id}\n{event_id}").as_bytes());
    format!("e{}", &digest["sha256:".len().."sha256:".len() + 24])
}

/// Writes a cache file atomically.
pub struct CacheWriter {
    pub path: PathBuf,
    header: CacheHeader,
    layout: (DType, Vec<usize>),
    entries: Vec<CacheEntry>,
    ids: BTreeSet<String>,
    file: Option<AtomicFile>,
}

impl std::fmt::Debug for CacheWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CacheWriter").field("path", &self.path).field("entries", &self.entries.len()).finish()
    }
}

impl CacheWriter {
    pub fn create(path: &Path, header: CacheHeader) -> Result<CacheWriter> {
        let messages = crate::document::schema_messages(validator(), &header.manifest(&[]));
        if let Some(first) = messages.first() {
            return Err(Error::coded("T401", format!("the cache header fails its schema: {first}")));
        }
        if header.level == "patient" && header.task_fingerprint.is_none() {
            return Err(Error::coded(
                "T404",
                "a patient-level cache names the task (`task_fingerprint`) it was built for",
            ));
        }
        let layout = header.output_layout()?;
        let file = AtomicFile::create(path)?;
        file.handle().create_group("entries")?;
        Ok(CacheWriter {
            path: path.to_path_buf(),
            header,
            layout,
            entries: Vec::new(),
            ids: BTreeSet::new(),
            file: Some(file),
        })
    }

    fn handle(&self) -> Result<&hdf5::File> {
        self.file.as_ref().map(AtomicFile::handle).ok_or_else(|| Error::invalid("this cache writer has finished"))
    }

    /// Add one feature; its dtype and shape must be the header's.
    pub fn add(&mut self, mut entry: CacheEntry, values: &NdArray) -> Result<CacheEntry> {
        let (dtype, shape) = &self.layout;
        if values.dtype() != *dtype || &values.shape() != shape {
            return Err(Error::coded(
                "T402",
                format!(
                    "entry {} is {} {:?}; the cache declares {} {:?}",
                    repr_str(&entry.entry_id),
                    values.dtype().name(),
                    values.shape(),
                    dtype.name(),
                    shape
                ),
            ));
        }
        if self.header.level == "patient"
            && (entry.row_id.is_none() || entry.cutoff_us.is_none() || entry.event_versions.is_none())
        {
            return Err(Error::coded(
                "T404",
                "a patient-level feature pins its row, cutoff and selected event versions",
            ));
        }
        if entry.sources.is_empty() {
            return Err(Error::coded("T403", "a feature names every source version it read"));
        }
        if !is_entry_id(&entry.entry_id) || !self.ids.insert(entry.entry_id.clone()) {
            return Err(Error::invalid(format!("entry id {} is malformed or not unique", repr_str(&entry.entry_id))));
        }
        let path = format!("entries/{}", entry.entry_id);
        let ds = data::create(&self.handle()?.group("entries")?, &entry.entry_id, values, &Layout::contiguous())?;
        entry.digest = array_digest(&path, values, "sha256")?;
        attrs::write(&ds, "digest", &AttrValue::Str(entry.digest.clone()))?;
        self.entries.push(entry.clone());
        Ok(entry)
    }

    /// Write the manifest and its checksum, and move the file into place.
    pub fn commit(mut self) -> Result<String> {
        let text = crate::json::canonical(&self.header.manifest(&self.entries));
        let digest = sha256(text.as_bytes());
        let file = self.file.take().ok_or_else(|| Error::invalid("this cache writer has finished"))?;
        let result = (|| -> Result<()> {
            let handle = file.handle();
            data::create_scalar_string(handle, "manifest", &text)?;
            attrs::write(handle, COMPANION_ATTR, &AttrValue::Str(SCHEMA.into()))?;
            attrs::write(handle, DIGEST_ATTR, &AttrValue::Str(digest.clone()))?;
            Ok(())
        })();
        match result {
            Ok(()) => {
                file.commit()?;
                Ok(digest)
            }
            Err(e) => {
                file.abort();
                Err(e)
            }
        }
    }
}

impl Drop for CacheWriter {
    fn drop(&mut self) {
        if let Some(f) = self.file.take() {
            f.abort();
        }
    }
}

/// An open cache.  Opening checks the manifest's checksum; reading an entry
/// checks the entry's.
#[derive(Debug)]
pub struct FeatureCache {
    pub path: PathBuf,
    pub header: CacheHeader,
    pub entries: Vec<CacheEntry>,
    pub manifest_digest: String,
    file: hdf5::File,
    by_id: HashMap<String, usize>,
    by_event: HashMap<(String, String), usize>,
    by_row: HashMap<String, usize>,
}

impl FeatureCache {
    pub fn open(path: &Path) -> Result<FeatureCache> {
        let file = open_read(path)?;
        let companion = attrs::get_str(&file, COMPANION_ATTR)?;
        if companion.as_deref() != Some(SCHEMA) {
            return Err(Error::coded(
                "T401",
                format!("{} is not a {SCHEMA} cache (it declares {:?})", repr_str(&path.to_string_lossy()), companion),
            ));
        }
        let text = match ops::child_dataset(&file, "manifest") {
            Some(ds) => data::read_scalar_string(&ds)?,
            None => return Err(Error::coded("T401", "the cache has no `manifest`")),
        };
        let declared = attrs::get_str(&file, DIGEST_ATTR)?.unwrap_or_default();
        let digest = sha256(text.as_bytes());
        if declared != digest {
            return Err(Error::coded(
                "T401",
                format!("the cache manifest's bytes do not match its checksum ({declared} declared, {digest} found)"),
            ));
        }
        let doc = crate::json::loads(&text)
            .map_err(|e| Error::coded("T401", format!("the cache manifest is not JSON: {e}")))?;
        let messages = crate::document::schema_messages(validator(), &doc);
        if let Some(first) = messages.first() {
            return Err(Error::coded("T401", format!("the cache manifest fails its schema: {first}")));
        }
        let entries: Vec<CacheEntry> =
            doc["entries"].as_array().into_iter().flatten().map(CacheEntry::from_json).collect::<Result<_>>()?;
        let mut by_id = HashMap::new();
        let mut by_event = HashMap::new();
        let mut by_row = HashMap::new();
        for (i, e) in entries.iter().enumerate() {
            by_id.insert(e.entry_id.clone(), i);
            if let (Some(event), Some(source)) = (&e.event_id, e.sources.first()) {
                by_event.insert((source.content_id.clone(), event.clone()), i);
            }
            if let Some(row) = &e.row_id {
                by_row.insert(row.clone(), i);
            }
        }
        Ok(FeatureCache {
            path: path.to_path_buf(),
            header: CacheHeader::from_manifest(&doc),
            entries,
            manifest_digest: digest,
            file,
            by_id,
            by_event,
            by_row,
        })
    }

    pub fn entry(&self, entry_id: &str) -> Option<&CacheEntry> {
        self.by_id.get(entry_id).map(|i| &self.entries[*i])
    }

    /// The event-level entry for an event version of a pinned source.
    pub fn event_entry(&self, content_id: &str, event_id: &str) -> Option<&CacheEntry> {
        self.by_event.get(&(content_id.to_string(), event_id.to_string())).map(|i| &self.entries[*i])
    }

    /// The patient-level entry of a row.
    pub fn row_entry(&self, row_id: &str) -> Option<&CacheEntry> {
        self.by_row.get(row_id).map(|i| &self.entries[*i])
    }

    /// An entry's payload, its checksum verified (T402 when it fails).
    pub fn get(&self, entry_id: &str) -> Result<NdArray> {
        let entry =
            self.entry(entry_id).ok_or_else(|| Error::Key(format!("the cache has no entry {}", repr_str(entry_id))))?;
        let path = format!("entries/{entry_id}");
        let ds = self
            .file
            .dataset(&path)
            .map_err(|_| Error::coded("T402", format!("entry {} has no payload", repr_str(entry_id))))?;
        let values = data::read(&ds)?;
        let digest = array_digest(&path, &values, "sha256")?;
        if digest != entry.digest || attrs::get_str(&ds, "digest")?.as_deref() != Some(entry.digest.as_str()) {
            return Err(Error::coded(
                "T402",
                format!("entry {}'s bytes do not match its checksum", repr_str(entry_id)),
            ));
        }
        Ok(values)
    }
}

/// What validating a cache found.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CacheReport {
    pub path: String,
    pub entries: usize,
    pub findings: Vec<Finding>,
}

impl CacheReport {
    pub fn ok(&self) -> bool {
        self.findings.is_empty()
    }

    fn with(&self, codes: &[&str]) -> Vec<String> {
        let mut out: Vec<String> =
            self.findings.iter().filter(|f| codes.contains(&f.code.as_str())).map(|f| f.location.clone()).collect();
        out.sort();
        out.dedup();
        out
    }

    /// Entries whose sources changed or vanished.
    pub fn stale(&self) -> Vec<String> {
        self.with(&["T403"])
    }

    /// Entries (or the manifest) whose own bytes are wrong.
    pub fn corrupt(&self) -> Vec<String> {
        self.with(&["T401", "T402"])
    }

    pub fn to_json(&self) -> Value {
        json!({
            "path": self.path,
            "ok": self.ok(),
            "entries": self.entries,
            "stale": self.stale(),
            "corrupt": self.corrupt(),
            "findings": self.findings.iter().map(Finding::to_json).collect::<Vec<_>>(),
        })
    }
}

/// Validate a cache: its manifest's and every payload's checksum, every
/// source pin, and --- given the task and its preflight --- that it was built
/// for this task, at these cutoffs, from these selections, and fitted on this
/// task's training partition.  `base` resolves relative source URIs (by
/// default, the cache's directory).
pub fn validate_cache(
    path: &Path,
    base: Option<&Path>,
    task: Option<&TaskManifest>,
    preflight: Option<&Preflight>,
) -> Result<CacheReport> {
    let mut report = CacheReport { path: path.to_string_lossy().into_owned(), ..Default::default() };
    let cache = match FeatureCache::open(path) {
        Ok(c) => c,
        Err(e) => {
            report.findings.push(Finding::new(e.code().unwrap_or("T401"), "manifest", e.message()));
            return Ok(report);
        }
    };
    report.entries = cache.entries.len();
    for entry in &cache.entries {
        if let Err(e) = cache.get(&entry.entry_id) {
            report.findings.push(Finding::new(e.code().unwrap_or("T402"), &entry.entry_id, e.message()));
        }
    }
    let base = base.map(Path::to_path_buf).or_else(|| path.parent().map(Path::to_path_buf));
    let mut checked: HashMap<(String, Option<String>, String), Option<String>> = HashMap::new();
    for entry in &cache.entries {
        for source in &entry.sources {
            let key = (source.uri.clone(), source.sample_key.clone(), source.content_id.clone());
            let problem = checked
                .entry(key)
                .or_insert_with(|| match source.open(base.as_deref()) {
                    Err(e) => Some(format!("{} cannot be reached: {e}", source.locator())),
                    Ok(sample) => match source.check(&sample, false) {
                        Ok(f) if f.is_empty() => None,
                        Ok(f) => Some(f[0].message.clone()),
                        Err(e) => Some(e.to_string()),
                    },
                })
                .clone();
            if let Some(why) = problem {
                report.findings.push(Finding::new("T403", &entry.entry_id, why));
            }
        }
    }
    if let Some(task) = task {
        let expected = task.task_fingerprint();
        if let Some(found) = &cache.header.task_fingerprint {
            if found != &expected {
                report.findings.push(Finding::new(
                    "T404",
                    "manifest",
                    format!("built for task {found}; this task is {expected}"),
                ));
            }
        }
        if let Some(fitted) = &cache.header.fitted_on {
            let partition = fitted["partition"].as_str().unwrap_or_default();
            let wanted = CacheHeader::fitted_on(task, partition);
            for key in ["task_fingerprint", "subjects_digest"] {
                if fitted.get(key) != wanted.get(key) {
                    report.findings.push(Finding::new(
                        "T405",
                        "manifest",
                        format!(
                            "fitted on {key} {}, but this task's {} partition is {}",
                            fitted.get(key).cloned().unwrap_or(Value::Null),
                            repr_str(partition),
                            wanted[key]
                        ),
                    ));
                }
            }
            // The split lists its training partition first (contract §3.3).
            let trains = task.split.as_ref().and_then(|(_, p)| p.first().cloned());
            if trains.as_deref().is_some_and(|t| t != partition) {
                report.findings.push(Finding::new(
                    "T405",
                    "manifest",
                    format!("fitted on partition {}, not the task's training partition", repr_str(partition)),
                ));
            }
        }
        if let Some(pre) = preflight {
            for entry in cache.entries.iter().filter(|e| e.row_id.is_some()) {
                let row_id = entry.row_id.as_deref().unwrap_or_default();
                let Some(view) = pre.row(row_id) else {
                    report.findings.push(Finding::new(
                        "T404",
                        &entry.entry_id,
                        format!("row {} is not a row of this task", repr_str(row_id)),
                    ));
                    continue;
                };
                if entry.cutoff_us != Some(view.cutoff_us) {
                    report.findings.push(Finding::new(
                        "T404",
                        &entry.entry_id,
                        format!("built at cutoff {:?}; the row's cutoff is {}", entry.cutoff_us, view.cutoff_us),
                    ));
                }
                let selected: Vec<String> = view.events.iter().map(|e| e.event_id.clone()).collect();
                let pinned = entry.event_versions.clone().unwrap_or_default();
                if pinned != selected {
                    let extra: Vec<&String> = pinned.iter().filter(|v| !selected.contains(v)).collect();
                    report.findings.push(Finding::new(
                        "T406",
                        &entry.entry_id,
                        format!(
                            "encodes event versions {pinned:?}, but row {} admits {selected:?}{}",
                            repr_str(row_id),
                            if extra.is_empty() {
                                String::new()
                            } else {
                                format!("; {extra:?} are not admissible at its cutoff")
                            }
                        ),
                    ));
                }
            }
        }
    }
    Ok(report)
}
