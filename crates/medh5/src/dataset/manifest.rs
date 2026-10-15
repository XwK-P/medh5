//! Cohort manifests: what a metadata-only scan of a directory tree knows.
//!
//! One metadata-only pass writes a JSON file, and splitting, stratification,
//! filtering and cohort checks all run against it without touching a voxel.
//! The manifest is also the **authority for splits** (§12.3): its `sha256` is
//! what lets a reader notice that a file's claim predates the current split.

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};

use indexmap::IndexMap;
use serde_json::{json, Map, Value};

use crate::collection::{open_any, AnyFile};
use crate::json::{canonical, num, pretty, repr, repr_str};
use crate::sample::Sample;
use crate::{Error, Result};

pub const SUFFIXES: [&str; 2] = [".medh5", ".medh5c"];

/// What every manifest [`Manifest::save`] writes names under `schema`: paths
/// are relative to the manifest's directory, or absolute.  A manifest with no
/// `schema` was written before 2.1, with paths as its scan named them.
pub const MANIFEST_SCHEMA: &str = "medh5.dataset/1";

/// How [`Manifest::save_with`] writes `root` and each entry's path.
///
/// A path is written relative --- to the manifest's directory --- only when it
/// lies beneath that directory; otherwise it is written absolute.  Never with
/// `..`: a lexical `..` and the system's disagree across a symbolic link.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PathStyle {
    /// A path the caller named absolute stays absolute; a relative one is
    /// re-based onto the manifest's directory.
    #[default]
    Keep,
    /// Every path beneath the manifest's directory relative, absolute paths
    /// included.
    Relative,
    /// Every path absolute.
    Absolute,
}

impl PathStyle {
    /// `"keep"`, `"relative"` or `"absolute"`, as the frontends spell it.
    pub fn parse(text: &str) -> Result<PathStyle> {
        match text {
            "keep" => Ok(PathStyle::Keep),
            "relative" => Ok(PathStyle::Relative),
            "absolute" => Ok(PathStyle::Absolute),
            other => {
                Err(Error::Value(format!("paths must be 'keep', 'relative' or 'absolute', not {}", repr_str(other))))
            }
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            PathStyle::Keep => "keep",
            PathStyle::Relative => "relative",
            PathStyle::Absolute => "absolute",
        }
    }
}

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
pub const ENTRY_FIELDS: [&str; 30] = [
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
    "class_keys",
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
    /// The label set's key for each class id an annotation names: what the
    /// id *means*, which `dataset check` compares between label sets (C102).
    /// Empty where the sample declares no label set, or one by reference,
    /// whose classes the file does not carry.
    pub class_keys: BTreeMap<i64, String>,
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

/// `class_keys` as a manifest stores it, `{"3": "liver"}`; a manifest
/// written before 2.0 has none, and an id that is not one is refused.
fn class_keys(doc: &Value) -> Result<BTreeMap<i64, String>> {
    let Some(Value::Object(map)) = doc.get("class_keys") else { return Ok(BTreeMap::new()) };
    map.iter()
        .map(|(id, key)| match id.parse::<i64>() {
            Ok(id) => Ok((id, crate::json::py_str(key))),
            Err(_) => Err(Error::Value(format!(
                "manifest entry {}: `class_keys` names {}, which is not a class id",
                repr(&doc.get("path").cloned().unwrap_or(Value::Null)),
                repr_str(id)
            ))),
        })
        .collect()
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
        out.insert(
            "class_keys".into(),
            Value::Object(self.class_keys.iter().map(|(id, key)| (id.to_string(), json!(key))).collect()),
        );
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
        // Refused, never dropped: a class id that did not read as an integer
        // --- `3.0` --- left the entry without the class.
        let list_int = |k: &str| -> Result<Vec<i64>> {
            match get(k).as_array() {
                None => Ok(Vec::new()),
                Some(items) => crate::json::integers(items)
                    .map_err(|why| Error::Value(format!("manifest entry {}: `{k}` holds {why}", repr(&get("path"))))),
            }
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
            class_ids: list_int("class_ids")?,
            annotated_class_ids: list_int("annotated_class_ids")?,
            class_keys: class_keys(doc)?,
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
            "schema": MANIFEST_SCHEMA,
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
    /// A split grouped or stratified on another field depends on values this
    /// does not cover, so re-curating them changes the split, not the digest.
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

    /// Write the manifest to `path`; an error names the file.
    ///
    /// Atomically: a write cut short --- a full disk, a crash --- left the
    /// manifest a cohort's splits are drawn from truncated.  Paths are written
    /// as [`PathStyle::Keep`] says.
    pub fn save(&self, path: &Path) -> Result<PathBuf> {
        self.save_with(path, PathStyle::Keep)
    }

    /// [`save`](Manifest::save), writing each path as `paths` says.
    ///
    /// In memory a relative path is relative to the working directory, as the
    /// caller named it; in the file it is relative to the manifest's
    /// directory, so the manifest opens from anywhere and moves with its
    /// cohort.  The digest does not change: it covers no path.
    pub fn save_with(&self, path: &Path, paths: PathStyle) -> Result<PathBuf> {
        let dir = absolute(path.parent().unwrap_or(Path::new("")))?;
        let mut doc = self.to_json();
        if let Some(root) = &self.root {
            doc["root"] = json!(rebase(root, &dir, paths)?);
        }
        let mut rebased: std::collections::HashMap<&str, String> = std::collections::HashMap::new();
        if let Some(Value::Array(entries)) = doc.get_mut("entries") {
            for (entry, item) in self.entries.iter().zip(entries.iter_mut()) {
                let written = match rebased.get(entry.path.as_str()) {
                    Some(written) => written.clone(),
                    None => {
                        let written = rebase(&entry.path, &dir, paths)?;
                        rebased.insert(&entry.path, written.clone());
                        written
                    }
                };
                item["path"] = json!(written);
            }
        }
        crate::h5::file::write_atomic(path, (pretty(&doc) + "\n").as_bytes())?;
        Ok(path.to_path_buf())
    }

    /// Read a manifest [`save`](Manifest::save) wrote; an error names the file.
    ///
    /// Each relative path is joined onto the manifest's directory as `path`
    /// names it, so in memory it is relative to the working directory again.
    /// A manifest written before 2.1 (no `schema`) is read against the working
    /// directory, as it was written; [`load_with_base`](Manifest::load_with_base)
    /// names the directory instead.
    pub fn load(path: &Path) -> Result<Manifest> {
        Manifest::load_with_base(path, None)
    }

    /// [`load`](Manifest::load), resolving relative paths against `base`
    /// when it is given.
    pub fn load_with_base(path: &Path, base: Option<&Path>) -> Result<Manifest> {
        Ok(Manifest::load_noting(path, base)?.0)
    }

    /// [`load_with_base`](Manifest::load_with_base), and whether the manifest
    /// is legacy and read differently than 3.0 will read it: some path is
    /// relative, no `base` was given, and the manifest is not in the working
    /// directory.  The frontends warn on it.
    #[doc(hidden)]
    pub fn load_noting(path: &Path, base: Option<&Path>) -> Result<(Manifest, bool)> {
        let text = std::fs::read_to_string(path).map_err(|e| crate::error::os_error(&e, path))?;
        let doc = crate::json::loads(&text)
            .map_err(|e| Error::Value(format!("{} is not JSON: {e}", repr_str(&path.to_string_lossy()))))?;
        let legacy = check_schema(&doc).map_err(|e| match e {
            Error::Value(why) => Error::Value(format!("{}: {why}", repr_str(&path.to_string_lossy()))),
            other => other,
        })?;
        let mut manifest = Manifest::from_json(&doc)?;
        let dir = path.parent().unwrap_or(Path::new(""));
        let base = match base {
            Some(base) => Some(base),
            None if legacy => None,
            None => Some(dir),
        };
        let mut differs = false;
        match base {
            Some(base) => {
                if let Some(root) = &manifest.root {
                    manifest.root = Some(resolve(root, base));
                }
                for entry in &mut manifest.entries {
                    entry.path = resolve(&entry.path, base);
                }
            }
            None => {
                let relative = manifest
                    .root
                    .iter()
                    .chain(manifest.entries.iter().map(|e| &e.path))
                    .any(|p| Path::new(p).is_relative());
                if relative {
                    let here = std::env::current_dir().ok();
                    let there = if dir.as_os_str().is_empty() { here.clone() } else { Some(dir.to_path_buf()) };
                    differs = match (here, there) {
                        (Some(here), Some(there)) => identity(&here).is_none() || identity(&here) != identity(&there),
                        _ => true,
                    };
                }
            }
        }
        Ok((manifest, differs))
    }

    /// A manifest from its JSON, its paths as the document holds them; an
    /// unknown `schema` is refused.
    pub fn from_json(doc: &Value) -> Result<Manifest> {
        check_schema(doc)?;
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

    /// Paths whose file no longer matches what the manifest recorded, each
    /// once: a collection's members are an entry each, and one file.
    ///
    /// Size and mtime are the cheap check; `content_id` is the proof, and
    /// `dataset check --deep` uses it.
    pub fn stale(&self) -> Vec<String> {
        let mut out = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for entry in &self.entries {
            if !seen.insert(entry.path.as_str()) {
                continue;
            }
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

/// Whether a manifest document is legacy --- written before 2.1, with no
/// `schema` --- or names [`MANIFEST_SCHEMA`]; any other schema is refused.
pub fn check_schema(doc: &Value) -> Result<bool> {
    match doc.get("schema") {
        None | Some(Value::Null) => Ok(true),
        Some(Value::String(s)) if s == MANIFEST_SCHEMA => Ok(false),
        Some(other) => Err(Error::Value(format!(
            "the manifest's schema is {}, which this medh5 does not read (it reads {})",
            repr(other),
            repr_str(MANIFEST_SCHEMA)
        ))),
    }
}

/// `path` without its `.` components; `.` when nothing is left.
fn lexical(path: &Path) -> PathBuf {
    let out: PathBuf = path.components().filter(|c| !matches!(c, Component::CurDir)).collect();
    if out.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        out
    }
}

/// `path` made absolute against the working directory, lexically: nothing
/// is resolved, `..` included.
pub(crate) fn absolute(path: &Path) -> Result<PathBuf> {
    let path = if path.as_os_str().is_empty() { Path::new(".") } else { path };
    Ok(lexical(&std::path::absolute(path).map_err(|e| crate::error::os_error(&e, path))?))
}

/// A stored path, read against `base`: joined onto it when relative.
fn resolve(stored: &str, base: &Path) -> String {
    let path = Path::new(stored);
    if path.is_absolute() || base.as_os_str().is_empty() {
        return stored.to_string();
    }
    lexical(&base.join(path)).to_string_lossy().into_owned()
}

/// A path as a manifest in `dir` (absolute) writes it, as `style` says.
pub(crate) fn rebase(path: &str, dir: &Path, style: PathStyle) -> Result<String> {
    let named = Path::new(path);
    let full = absolute(named)?;
    let keep_absolute = style == PathStyle::Keep && named.is_absolute();
    if style != PathStyle::Absolute && !keep_absolute {
        if let Ok(rest) = full.strip_prefix(dir) {
            let mut parts = Vec::new();
            for part in rest.components() {
                match part {
                    Component::Normal(name) => parts.push(name.to_string_lossy().into_owned()),
                    // Beneath only lexically: `dir/x/../..` is not.
                    _ => return Ok(full.to_string_lossy().into_owned()),
                }
            }
            // Forward slashes, which every platform reads, so the manifest
            // moves between them with its cohort.
            return Ok(if parts.is_empty() { ".".into() } else { parts.join("/") });
        }
    }
    Ok(full.to_string_lossy().into_owned())
}

/// Every sample and collection under `root`, in a stable order.
pub fn find(root: &Path) -> Result<Vec<PathBuf>> {
    find_with(root, &SUFFIXES)
}

/// [`find`] for other file suffixes (each with its leading dot).  A directory
/// that cannot be listed is passed over; [`scan_with`] reports it.
pub fn find_with(root: &Path, suffixes: &[&str]) -> Result<Vec<PathBuf>> {
    Ok(walk(root, suffixes).0)
}

/// The files under `root` with one of `suffixes`, in a stable order, and the
/// directories under it that could not be listed, with why.
///
/// A symlinked directory is part of the tree and is walked, once: every real
/// directory is listed before any link is followed, links in name order, and
/// a directory already listed --- a link back up the tree, two links to one
/// place --- is not listed again.  Suffixes match without regard to case
/// (`CASE.MEDH5`), and AppleDouble files (`._case.medh5`, the metadata a Mac
/// leaves on a foreign volume) are not samples.
fn walk(root: &Path, suffixes: &[&str]) -> (Vec<PathBuf>, Vec<(PathBuf, std::io::Error)>) {
    if root.is_file() {
        return (vec![root.to_path_buf()], Vec::new());
    }
    let suffixes: Vec<String> = suffixes.iter().map(|s| s.to_lowercase()).collect();
    let mut found = Vec::new();
    let mut unlisted = Vec::new();
    let mut listed = std::collections::HashSet::new();
    let mut stack = vec![root.to_path_buf()];
    let mut linked = std::collections::BTreeSet::new();
    loop {
        let dir = match stack.pop() {
            Some(dir) => dir,
            None => match linked.pop_first() {
                Some(dir) => dir,
                None => break,
            },
        };
        let listing = match std::fs::read_dir(&dir) {
            Ok(listing) => listing,
            Err(e) => {
                unlisted.push((dir, e));
                continue;
            }
        };
        if let Some(id) = identity(&dir) {
            if !listed.insert(id) {
                continue;
            }
        }
        for item in listing.flatten() {
            let path = item.path();
            let name = item.file_name().to_string_lossy().to_lowercase();
            if name.starts_with("._") {
                continue;
            }
            let link = item.file_type().map(|t| t.is_symlink()).unwrap_or(false);
            let Ok(meta) = std::fs::metadata(&path) else { continue };
            if meta.is_dir() {
                if link {
                    linked.insert(path);
                } else {
                    stack.push(path);
                }
            } else if meta.is_file() && suffixes.iter().any(|s| name.len() > s.len() && name.ends_with(s.as_str())) {
                found.push(path);
            }
        }
    }
    found.sort();
    unlisted.sort_by(|a, b| a.0.cmp(&b.0));
    (found, unlisted)
}

/// What makes two paths one directory: its device and inode, where there are
/// such things, else its canonical path.
fn identity(dir: &Path) -> Option<(u64, u64, PathBuf)> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        std::fs::metadata(dir).ok().map(|m| (m.dev(), m.ino(), PathBuf::new()))
    }
    #[cfg(not(unix))]
    {
        std::fs::canonicalize(dir).ok().map(|p| (0, 0, p))
    }
}

/// Build a manifest from a directory tree, and report what would not open.
///
/// A cohort scan that dies on one broken file has told you nothing about the
/// other 9 999, so failures are collected unless `strict`.
pub fn scan(root: &Path, strict: bool) -> Result<(Manifest, Vec<String>)> {
    scan_with(root, &SUFFIXES, strict)
}

/// [`scan`] over the files [`find_with`] finds.  A directory that cannot be
/// listed is a failure as a file that will not open is: passed over in
/// silence, it left the manifest short of every sample under it, even when
/// `strict`.
pub fn scan_with(root: &Path, suffixes: &[&str], strict: bool) -> Result<(Manifest, Vec<String>)> {
    let mut manifest = Manifest { root: Some(root.to_string_lossy().into_owned()), ..Default::default() };
    let mut failures = Vec::new();
    let (paths, unlisted) = walk(root, suffixes);
    for (dir, error) in unlisted {
        let error = crate::error::os_error(&error, &dir);
        if strict {
            return Err(error);
        }
        failures.push(format!("{}: {}", dir.display(), error.python_str()));
    }
    for path in paths {
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
    let class_keys = classes.iter().filter_map(|id| Some((*id, label_set?.by_id(*id)?.key.clone()))).collect();
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
        class_keys,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn cohort(paths: &[&Path]) -> Manifest {
        let entries = paths
            .iter()
            .enumerate()
            .map(|(i, p)| Entry {
                path: p.to_string_lossy().into_owned(),
                sample_id: format!("s{i}"),
                subject_id: format!("s{i}"),
                group_id: format!("s{i}"),
                ..Default::default()
            })
            .collect();
        Manifest { entries, ..Default::default() }
    }

    fn stored(path: &Path) -> Value {
        crate::json::loads(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    #[test]
    fn s12_3_paths_beneath_the_manifest_are_written_relative_with_forward_slashes() {
        let dir = tempfile::tempdir().unwrap();
        let inside = dir.path().join("out").join("site-a").join("a.medh5");
        let outside = dir.path().join("elsewhere").join("b.medh5");
        let mut manifest = cohort(&[&inside, &outside]);
        manifest.root = Some(dir.path().join("out").to_string_lossy().into_owned());
        let target = dir.path().join("out").join("m.json");
        std::fs::create_dir_all(dir.path().join("out")).unwrap();
        manifest.save_with(&target, PathStyle::Relative).unwrap();
        let doc = stored(&target);
        assert_eq!(doc["schema"], MANIFEST_SCHEMA);
        assert_eq!(doc["root"], ".");
        assert_eq!(doc["entries"][0]["path"], "site-a/a.medh5");
        assert_eq!(doc["entries"][1]["path"], outside.to_string_lossy().as_ref());
        // Kept absolute unless asked; every path absolute on request.
        manifest.save(&target).unwrap();
        assert_eq!(stored(&target)["entries"][0]["path"], inside.to_string_lossy().as_ref());
        manifest.save_with(&target, PathStyle::Absolute).unwrap();
        assert_eq!(stored(&target)["root"], dir.path().join("out").to_string_lossy().as_ref());
        // Read back, a relative path is joined onto the manifest's directory.
        manifest.save_with(&target, PathStyle::Relative).unwrap();
        let back = Manifest::load(&target).unwrap();
        assert_eq!(back.entries[0].path, inside.to_string_lossy());
        assert_eq!(back.root.as_deref(), Some(dir.path().join("out").to_string_lossy().as_ref()));
        assert_eq!(back.sha256(), manifest.sha256());
    }

    #[test]
    fn s12_3_a_path_beneath_only_lexically_is_written_absolute() {
        let dir = tempfile::tempdir().unwrap();
        let sneaky = dir.path().join("out").join("x").join("..").join("..").join("b.medh5");
        let target = dir.path().join("out").join("m.json");
        std::fs::create_dir_all(dir.path().join("out")).unwrap();
        cohort(&[&sneaky]).save_with(&target, PathStyle::Relative).unwrap();
        let written = stored(&target)["entries"][0]["path"].as_str().unwrap().to_string();
        assert!(Path::new(&written).is_absolute(), "{written}");
    }

    #[test]
    fn s12_3_a_legacy_manifest_is_read_as_written_unless_given_a_base() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("m.json");
        let mut doc = cohort(&[Path::new("studies/a.medh5")]).to_json();
        doc.as_object_mut().unwrap().remove("schema");
        std::fs::write(&target, pretty(&doc)).unwrap();
        let (legacy, differs) = Manifest::load_noting(&target, None).unwrap();
        assert_eq!(legacy.entries[0].path, "studies/a.medh5");
        assert!(differs, "the temporary directory is not the working directory");
        let based = Manifest::load_with_base(&target, Some(dir.path())).unwrap();
        assert_eq!(based.entries[0].path, dir.path().join("studies").join("a.medh5").to_string_lossy());
        // Upgraded, the same manifest reads against its own directory.
        doc["schema"] = json!(MANIFEST_SCHEMA);
        std::fs::write(&target, pretty(&doc)).unwrap();
        let (current, differs) = Manifest::load_noting(&target, None).unwrap();
        assert!(!differs);
        assert_eq!(current.entries[0].path, based.entries[0].path);
    }

    #[test]
    fn s12_3_an_unknown_schema_is_refused() {
        let mut doc = cohort(&[]).to_json();
        doc["schema"] = json!("medh5.dataset/2");
        let err = Manifest::from_json(&doc).unwrap_err().to_string();
        assert!(err.contains("medh5.dataset/2"), "{err}");
        assert!(PathStyle::parse("loose").is_err());
    }

    /// A symlinked directory was passed over, and `CASE.MEDH5` with it, while
    /// `._case.medh5` --- a Mac's AppleDouble file --- was taken for a sample.
    #[cfg(unix)]
    #[test]
    fn a_scan_follows_linked_directories_once_ignoring_case_and_appledouble() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("root");
        let elsewhere = dir.path().join("elsewhere");
        std::fs::create_dir_all(root.join("real")).unwrap();
        std::fs::create_dir_all(&elsewhere).unwrap();
        for file in ["real/a.medh5", "B.MEDH5", "._a.medh5", "real/._b.medh5c", "notes.txt"] {
            std::fs::write(root.join(file), b"").unwrap();
        }
        std::fs::write(elsewhere.join("c.medh5c"), b"").unwrap();
        std::os::unix::fs::symlink(&elsewhere, root.join("linked")).unwrap();
        // A cycle, and a second way into a directory already walked.
        std::os::unix::fs::symlink(&root, root.join("real/up")).unwrap();
        std::os::unix::fs::symlink(root.join("real"), root.join("again")).unwrap();
        let (found, unlisted) = walk(&root, &SUFFIXES);
        assert!(unlisted.is_empty());
        let names: Vec<String> =
            found.iter().map(|p| p.strip_prefix(&root).unwrap().to_string_lossy().into_owned()).collect();
        assert_eq!(names, ["B.MEDH5", "linked/c.medh5c", "real/a.medh5"]);
    }
}
