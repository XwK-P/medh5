//! The clinical profile (format 1.1) and the task-and-cache contract.
//!
//! Records cross this boundary as JSON-shaped dicts --- the logical-record
//! form of `medh5-clinical-1.schema.json` --- and `medh5/clinical.py` turns
//! them into frozen dataclasses.  The companion functions are stateless: a
//! task manifest is a dict, re-parsed per call, so it pickles into DataLoader
//! workers with nothing to rebuild.

use std::path::PathBuf;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json::{json, Value};

use medh5::clinical::model::{ClinicalRecords, Event, Link};
use medh5::clinical::select::SelectionPolicy;
use medh5::clinical::Clinical as EngineClinical;
use medh5::companion::cache::{CacheEntry, CacheHeader, CacheWriter, FeatureCache};
use medh5::companion::task::TaskManifest;
use medh5::companion::SourceRef;
use medh5::sample::Sample as EngineSample;

use crate::convert::{json_to_py, nd_to_py, py_to_json, py_to_nd};
use crate::errors::R;

/// A Python value as JSON: a dict, or anything with `to_json()`.
pub fn record(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.hasattr("to_json")? && !obj.is_instance_of::<PyDict>() {
        return py_to_json(&obj.call_method0("to_json")?);
    }
    py_to_json(obj)
}

/// An exact instant given as one integer reads as `[t, t]`.
fn widen_instants(mut value: Value) -> Value {
    if let Value::Object(map) = &mut value {
        for key in ["effective_start_us", "effective_end_us", "available_us"] {
            if let Some(Value::Number(n)) = map.get(key) {
                if let Some(t) = n.as_i64() {
                    map.insert(key.into(), json!([t, t]));
                }
            }
        }
    }
    value
}

/// A record from a positional value and keyword fields (the fields win).
pub fn record_with(value: Option<&Bound<'_, PyAny>>, fields: Option<&Bound<'_, PyDict>>) -> PyResult<Value> {
    let mut out = match value {
        Some(v) if !v.is_none() => record(v)?,
        _ => json!({}),
    };
    if let (Value::Object(map), Some(f)) = (&mut out, fields) {
        if let Value::Object(extra) = py_to_json(f.as_any())? {
            map.extend(extra);
        }
    }
    Ok(widen_instants(out))
}

pub fn event_arg(value: Option<&Bound<'_, PyAny>>, fields: Option<&Bound<'_, PyDict>>) -> R<Event> {
    Ok(Event::from_json(&record_with(value, fields)?)?)
}

pub fn policy_arg(policy: Option<&Bound<'_, PyAny>>) -> R<SelectionPolicy> {
    match policy {
        None => Ok(SelectionPolicy::strict()),
        Some(p) if p.is_none() => Ok(SelectionPolicy::strict()),
        Some(p) if p.is_instance_of::<pyo3::types::PyString>() => {
            let mut out = SelectionPolicy::strict();
            out.selection = p.extract()?;
            out.check()?;
            Ok(out)
        }
        Some(p) => Ok(SelectionPolicy::from_json(&record(p)?)?),
    }
}

fn to_list<'py>(py: Python<'py>, values: impl IntoIterator<Item = Value>) -> PyResult<Bound<'py, PyList>> {
    let items: Vec<Bound<'py, PyAny>> = values.into_iter().map(|v| json_to_py(py, &v)).collect::<PyResult<_>>()?;
    PyList::new(py, items)
}

/// One sample's clinical profile, read.
#[pyclass(module = "medh5._core", name = "ClinicalHandle", frozen)]
pub struct ClinicalHandle {
    pub inner: Arc<EngineClinical>,
    /// The sample it was read from, kept open while this is alive.
    pub _sample: Arc<EngineSample>,
}

#[pymethods]
impl ClinicalHandle {
    #[getter]
    fn projection(&self) -> bool {
        self.inner.projection
    }
    fn descriptor<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &self.inner.descriptor.to_json())
    }
    fn events<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        to_list(py, self.inner.events.iter().map(Event::to_json))
    }
    fn links<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        to_list(py, self.inner.links.iter().map(Link::to_json))
    }
    /// Document metadata (no text): id, media type, language, source type,
    /// and the text's length in UTF-8 bytes.
    fn documents<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        to_list(
            py,
            self.inner.documents().iter().map(|d| {
                json!({
                    "document_id": d.document_id,
                    "media_type": d.media_type,
                    "language": d.language,
                    "source_type": d.source_type,
                    "n_bytes": d.n_bytes,
                })
            }),
        )
    }
    /// One document's text, read from the file now.
    fn text(&self, py: Python<'_>, document_id: &str) -> R<String> {
        let inner = self.inner.clone();
        let id = document_id.to_string();
        Ok(py.detach(move || inner.text(&id))?)
    }
    fn document<'py>(&self, py: Python<'py>, document_id: &str) -> R<Bound<'py, PyAny>> {
        Ok(json_to_py(py, &self.inner.document(document_id)?.to_json())?)
    }
    fn records<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        Ok(json_to_py(py, &self.inner.records()?.to_json())?)
    }
    #[pyo3(signature = (cutoff_us, policy=None))]
    fn select<'py>(&self, py: Python<'py>, cutoff_us: i64, policy: Option<&Bound<'py, PyAny>>) -> R<Bound<'py, PyAny>> {
        let policy = policy_arg(policy)?;
        let links: Vec<(usize, &Link)> = self.inner.links.iter().map(|l| (0, l)).collect();
        let selection = medh5::clinical::select(&self.inner.events, &links, cutoff_us, &policy)?;
        Ok(json_to_py(py, &selection.to_json(&self.inner.events))?)
    }
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &self.inner.summary())
    }
}

/// Select from records not read from a file: `events` and `links` as dicts.
#[pyfunction]
#[pyo3(signature = (events, links, cutoff_us, policy=None))]
fn clinical_select<'py>(
    py: Python<'py>,
    events: Vec<Bound<'py, PyAny>>,
    links: Vec<Bound<'py, PyAny>>,
    cutoff_us: i64,
    policy: Option<&Bound<'py, PyAny>>,
) -> R<Bound<'py, PyAny>> {
    let events: Vec<Event> = events.iter().map(|e| event_arg(Some(e), None)).collect::<R<_>>()?;
    let links: Vec<Link> = links.iter().map(|l| Ok(Link::from_json(&record(l)?)?)).collect::<R<_>>()?;
    let refs: Vec<(usize, &Link)> = links.iter().map(|l| (0, l)).collect();
    let selection = medh5::clinical::select(&events, &refs, cutoff_us, &policy_arg(policy)?)?;
    Ok(json_to_py(py, &selection.to_json(&events))?)
}

/// Check a logical-record bundle against its schema and parse it back.
#[pyfunction]
fn clinical_records<'py>(py: Python<'py>, records: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
    let parsed = ClinicalRecords::from_json(&record(records)?)?;
    Ok(json_to_py(py, &parsed.to_json())?)
}

#[pyfunction]
#[pyo3(signature = (path, records, out=None))]
fn clinical_augment<'py>(
    py: Python<'py>,
    path: PathBuf,
    records: &Bound<'py, PyAny>,
    out: Option<PathBuf>,
) -> R<Bound<'py, PyAny>> {
    let records = ClinicalRecords::from_json(&record(records)?)?;
    let report = py.detach(move || medh5::clinical::augment::augment(&path, records, out.as_deref()))?;
    Ok(json_to_py(py, &report.to_json())?)
}

#[pyfunction]
fn clinical_strip<'py>(py: Python<'py>, path: PathBuf, out: PathBuf) -> R<Bound<'py, PyAny>> {
    let report = py.detach(move || medh5::clinical::augment::strip(&path, &out))?;
    Ok(json_to_py(py, &report.to_json())?)
}

/// `(events, links, notes)`: imaging events from `days_from_baseline`.
#[pyfunction]
fn imaging_events_from_timepoints<'py>(py: Python<'py>, path: PathBuf) -> R<Bound<'py, PyAny>> {
    let (events, links, notes) = py.detach(move || medh5::clinical::augment::imaging_events_from_timepoints(&path))?;
    let value = json!([
        events.iter().map(Event::to_json).collect::<Vec<_>>(),
        links.iter().map(Link::to_json).collect::<Vec<_>>(),
        notes,
    ]);
    Ok(json_to_py(py, &value)?)
}

#[pyfunction]
fn baseline_day_clock<'py>(py: Python<'py>, clock_id: &str) -> PyResult<Bound<'py, PyAny>> {
    json_to_py(py, &medh5::clinical::augment::baseline_day_clock(clock_id).to_json())
}

#[pyfunction]
fn clinical_schema_text() -> &'static str {
    medh5::clinical::schema::schema_text()
}

// -- the task-and-cache contract -----------------------------------------------------------------

fn manifest(doc: &Bound<'_, PyAny>) -> R<TaskManifest> {
    Ok(TaskManifest::from_json(&record(doc)?)?)
}

fn findings<'py>(py: Python<'py>, found: &[medh5::companion::Finding]) -> PyResult<Bound<'py, PyList>> {
    to_list(py, found.iter().map(medh5::companion::Finding::to_json))
}

/// Parse, check the schema, and return the normalised manifest.
#[pyfunction]
fn task_normalize<'py>(py: Python<'py>, doc: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
    Ok(json_to_py(py, &manifest(doc)?.to_json())?)
}

/// Everything wrong with a manifest that opening no file can find.
#[pyfunction]
fn task_validate<'py>(py: Python<'py>, doc: &Bound<'py, PyAny>) -> R<Bound<'py, PyList>> {
    Ok(findings(py, &manifest(doc)?.validate())?)
}

/// `{"task": ..., "manifest": ...}`.
#[pyfunction]
fn task_fingerprints<'py>(py: Python<'py>, doc: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
    let m = manifest(doc)?;
    Ok(json_to_py(py, &json!({"task": m.task_fingerprint(), "manifest": m.manifest_fingerprint()}))?)
}

#[pyfunction]
fn task_row_fingerprint(doc: &Bound<'_, PyAny>, row_id: &str) -> R<String> {
    let m = manifest(doc)?;
    let row =
        m.rows.iter().find(|r| r.row_id == row_id).ok_or_else(|| medh5::Error::Key(format!("no row {row_id:?}")))?;
    Ok(m.row_fingerprint(row))
}

#[pyfunction]
fn task_subjects_digest(doc: &Bound<'_, PyAny>, partition: &str) -> R<String> {
    Ok(manifest(doc)?.subjects_digest(partition))
}

/// The manifest with every subject's duplicated events recorded.
#[pyfunction]
#[pyo3(signature = (doc, base=None))]
fn task_reconcile<'py>(py: Python<'py>, doc: &Bound<'py, PyAny>, base: Option<PathBuf>) -> R<Bound<'py, PyAny>> {
    let mut m = manifest(doc)?;
    let m = py.detach(move || -> medh5::Result<TaskManifest> {
        for i in 0..m.subjects.len() {
            m.subjects[i].reconciled = medh5::companion::view::reconcile(&m.subjects[i], base.as_deref())?;
        }
        Ok(m)
    })?;
    Ok(json_to_py(py, &m.to_json())?)
}

/// The preflight of a task, as columns (`crate::preflight`).
#[pyfunction]
#[pyo3(signature = (doc, base=None, deep=false))]
fn task_preflight<'py>(
    py: Python<'py>,
    doc: &Bound<'py, PyAny>,
    base: Option<PathBuf>,
    deep: bool,
) -> R<Bound<'py, PyAny>> {
    let m = manifest(doc)?;
    Ok(crate::preflight::preflight_to_py(py, m, base.as_deref(), deep)?.into_any())
}

/// A source reference pinned to what the sample is now.
#[pyfunction]
#[pyo3(signature = (path, sample_key=None, source_id=None, uri=None))]
fn source_pin<'py>(
    py: Python<'py>,
    path: PathBuf,
    sample_key: Option<String>,
    source_id: Option<String>,
    uri: Option<String>,
) -> R<Bound<'py, PyAny>> {
    let locator = uri.unwrap_or_else(|| path.to_string_lossy().into_owned());
    let found = py.detach(move || -> medh5::Result<SourceRef> {
        let mut src = SourceRef::new(path.to_string_lossy(), sample_key.clone(), "");
        let sample = src.open(None)?;
        src = SourceRef::pin(locator, sample_key, &sample)?;
        src.source_id = source_id.unwrap_or_default();
        Ok(src)
    })?;
    Ok(json_to_py(py, &found.to_json())?)
}

/// The findings of checking a reference against its sample now.
#[pyfunction]
#[pyo3(signature = (source, base=None, deep=false))]
fn source_check<'py>(
    py: Python<'py>,
    source: &Bound<'py, PyAny>,
    base: Option<PathBuf>,
    deep: bool,
) -> R<Bound<'py, PyList>> {
    let src = SourceRef::from_json(&record(source)?)?;
    let found = py.detach(move || -> medh5::Result<Vec<medh5::companion::Finding>> {
        match src.open(base.as_deref()) {
            Err(e) => Ok(vec![medh5::companion::Finding::new("T301", src.locator(), e.to_string())]),
            Ok(sample) => src.check(&sample, deep),
        }
    })?;
    Ok(findings(py, &found)?)
}

fn header(value: &Value) -> CacheHeader {
    let opt = |k: &str| value.get(k).and_then(Value::as_str).map(str::to_string);
    CacheHeader {
        level: opt("level").unwrap_or_else(|| "event".into()),
        encoder: value.get("encoder").cloned().unwrap_or(Value::Null),
        preprocessing: value.get("preprocessing").cloned().unwrap_or_else(|| json!({})),
        output: value.get("output").cloned().unwrap_or(Value::Null),
        task_fingerprint: opt("task_fingerprint"),
        selection: opt("selection"),
        fitted_on: value.get("fitted_on").filter(|v| !v.is_null()).cloned(),
    }
}

/// Writes a feature cache; `commit()` moves it into place.
#[pyclass(module = "medh5._core", name = "CacheWriterHandle")]
pub struct CacheWriterHandle {
    inner: Option<CacheWriter>,
}

#[pymethods]
impl CacheWriterHandle {
    /// Add one feature: `entry` as a dict (its `digest` is computed).
    fn add<'py>(
        &mut self,
        py: Python<'py>,
        entry: &Bound<'py, PyAny>,
        values: &Bound<'py, PyAny>,
    ) -> R<Bound<'py, PyAny>> {
        let doc = record(entry)?;
        let text = |k: &str| doc.get(k).and_then(Value::as_str).map(str::to_string);
        let entry = CacheEntry {
            entry_id: text("entry_id").unwrap_or_default(),
            sources: doc
                .get("sources")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .map(SourceRef::from_json)
                .collect::<medh5::Result<_>>()?,
            event_id: text("event_id"),
            document_id: text("document_id"),
            row_id: text("row_id"),
            cutoff_us: doc.get("cutoff_us").and_then(Value::as_i64),
            event_versions: doc
                .get("event_versions")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect()),
            digest: String::new(),
        };
        let values = py_to_nd(values)?;
        let writer = self.inner.as_mut().ok_or_else(|| medh5::Error::invalid("this cache writer has finished"))?;
        Ok(json_to_py(py, &writer.add(entry, &values)?.to_json())?)
    }
    /// Write the manifest and its checksum; returns the checksum.
    fn commit(&mut self) -> R<String> {
        let writer = self.inner.take().ok_or_else(|| medh5::Error::invalid("this cache writer has finished"))?;
        Ok(writer.commit()?)
    }
    /// Discard the half-written cache.
    fn abort(&mut self) {
        self.inner.take();
    }
}

#[pyfunction]
fn cache_create(path: PathBuf, header_doc: &Bound<'_, PyAny>) -> R<CacheWriterHandle> {
    let h = header(&record(header_doc)?);
    Ok(CacheWriterHandle { inner: Some(CacheWriter::create(&path, h)?) })
}

/// An open feature cache; `close()` releases the file.
#[pyclass(module = "medh5._core", name = "FeatureCacheHandle", frozen)]
pub struct FeatureCacheHandle {
    inner: std::sync::Mutex<Option<Arc<FeatureCache>>>,
}

impl FeatureCacheHandle {
    fn cache(&self) -> R<Arc<FeatureCache>> {
        let guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        Ok(guard.clone().ok_or_else(|| medh5::Error::File("the feature cache is closed".into()))?)
    }
}

#[pymethods]
impl FeatureCacheHandle {
    #[getter]
    fn path(&self) -> R<String> {
        Ok(self.cache()?.path.to_string_lossy().into_owned())
    }
    #[getter]
    fn manifest_digest(&self) -> R<String> {
        Ok(self.cache()?.manifest_digest.clone())
    }
    #[getter]
    fn is_open(&self) -> bool {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).is_some()
    }
    fn close(&self) {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).take();
    }
    /// Forget the handle without closing it: what a forked child does with
    /// its parent's (§14.4) --- the descriptor is the parent's to close.
    fn abandon(&self) {
        if let Some(cache) = self.inner.lock().unwrap_or_else(|e| e.into_inner()).take() {
            std::mem::forget(cache);
        }
    }
    fn header<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        let cache = self.cache()?;
        let h = &cache.header;
        Ok(json_to_py(
            py,
            &json!({
                "level": h.level, "encoder": h.encoder, "preprocessing": h.preprocessing, "output": h.output,
                "task_fingerprint": h.task_fingerprint, "selection": h.selection, "fitted_on": h.fitted_on,
            }),
        )?)
    }
    fn entries<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyList>> {
        Ok(to_list(py, self.cache()?.entries.iter().map(CacheEntry::to_json))?)
    }
    /// An entry's payload, its checksum verified.
    fn get<'py>(&self, py: Python<'py>, entry_id: &str) -> R<Bound<'py, PyAny>> {
        let cache = self.cache()?;
        let id = entry_id.to_string();
        Ok(nd_to_py(py, py.detach(move || cache.get(&id))?))
    }
    fn event_entry<'py>(&self, py: Python<'py>, content_id: &str, event_id: &str) -> R<Option<Bound<'py, PyAny>>> {
        Ok(self.cache()?.event_entry(content_id, event_id).map(|e| json_to_py(py, &e.to_json())).transpose()?)
    }
    fn row_entry<'py>(&self, py: Python<'py>, row_id: &str) -> R<Option<Bound<'py, PyAny>>> {
        Ok(self.cache()?.row_entry(row_id).map(|e| json_to_py(py, &e.to_json())).transpose()?)
    }
}

#[pyfunction]
fn cache_open(py: Python<'_>, path: PathBuf) -> R<FeatureCacheHandle> {
    let cache = py.detach(move || FeatureCache::open(&path))?;
    Ok(FeatureCacheHandle { inner: std::sync::Mutex::new(Some(Arc::new(cache))) })
}

/// Validate a cache; with a task, also against that task's preflight.
#[pyfunction]
#[pyo3(signature = (path, base=None, task=None, task_base=None, check_rows=true))]
fn cache_validate<'py>(
    py: Python<'py>,
    path: PathBuf,
    base: Option<PathBuf>,
    task: Option<&Bound<'py, PyAny>>,
    task_base: Option<PathBuf>,
    check_rows: bool,
) -> R<Bound<'py, PyAny>> {
    let task = match task {
        Some(t) if !t.is_none() => Some(manifest(t)?),
        _ => None,
    };
    let report = py.detach(move || -> medh5::Result<medh5::companion::CacheReport> {
        let admitted = match &task {
            Some(t) if check_rows => Some(medh5::companion::Admitted::preflight(t, task_base.as_deref(), false)?),
            _ => None,
        };
        medh5::companion::validate_cache(&path, base.as_deref(), task.as_ref(), admitted.as_ref())
    })?;
    Ok(json_to_py(py, &report.to_json())?)
}

#[pyfunction]
fn cache_fitted_on<'py>(py: Python<'py>, doc: &Bound<'py, PyAny>, partition: &str) -> R<Bound<'py, PyAny>> {
    Ok(json_to_py(py, &CacheHeader::fitted_on(&manifest(doc)?, partition))?)
}

#[pyfunction]
fn cache_event_entry_id(content_id: &str, event_id: &str) -> String {
    medh5::companion::cache::event_entry_id(content_id, event_id)
}

#[pyfunction]
fn task_schema_text() -> &'static str {
    medh5::companion::task::schema_text()
}

#[pyfunction]
fn cache_schema_text() -> &'static str {
    medh5::companion::cache::schema_text()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ClinicalHandle>()?;
    m.add_class::<CacheWriterHandle>()?;
    m.add_class::<FeatureCacheHandle>()?;
    for f in [
        wrap_pyfunction!(clinical_select, m)?,
        wrap_pyfunction!(clinical_records, m)?,
        wrap_pyfunction!(clinical_augment, m)?,
        wrap_pyfunction!(clinical_strip, m)?,
        wrap_pyfunction!(imaging_events_from_timepoints, m)?,
        wrap_pyfunction!(baseline_day_clock, m)?,
        wrap_pyfunction!(clinical_schema_text, m)?,
        wrap_pyfunction!(task_normalize, m)?,
        wrap_pyfunction!(task_validate, m)?,
        wrap_pyfunction!(task_fingerprints, m)?,
        wrap_pyfunction!(task_row_fingerprint, m)?,
        wrap_pyfunction!(task_subjects_digest, m)?,
        wrap_pyfunction!(task_reconcile, m)?,
        wrap_pyfunction!(task_preflight, m)?,
        wrap_pyfunction!(source_pin, m)?,
        wrap_pyfunction!(source_check, m)?,
        wrap_pyfunction!(cache_create, m)?,
        wrap_pyfunction!(cache_open, m)?,
        wrap_pyfunction!(cache_validate, m)?,
        wrap_pyfunction!(cache_fitted_on, m)?,
        wrap_pyfunction!(cache_event_entry_id, m)?,
        wrap_pyfunction!(task_schema_text, m)?,
        wrap_pyfunction!(cache_schema_text, m)?,
    ] {
        m.add_function(f)?;
    }
    let py = m.py();
    let codes = PyDict::new(py);
    for (code, summary) in medh5::companion::CODES {
        codes.set_item(code, summary)?;
    }
    m.add("COMPANION_CODES", codes)?;
    m.add("CLINICAL_PROFILE", medh5::clinical::PROFILE)?;
    m.add("CLINICAL_SCHEMA", medh5::clinical::SCHEMA)?;
    m.add("TASK_SCHEMA", medh5::companion::task::SCHEMA)?;
    m.add("CACHE_SCHEMA", medh5::companion::cache::SCHEMA)?;
    m.add("FORMAT_VERSIONS", pyo3::types::PyTuple::new(py, medh5::version::FORMAT_VERSIONS)?)?;
    for (name, values) in [
        ("EVENT_KINDS", &medh5::clinical::EVENT_KINDS[..]),
        ("TEMPORAL_TYPES", &medh5::clinical::TEMPORAL_TYPES[..]),
        ("STATUSES", &medh5::clinical::STATUSES[..]),
        ("COMPARATORS", &medh5::clinical::COMPARATORS[..]),
        ("ENDPOINT_TYPES", &medh5::clinical::ENDPOINT_TYPES[..]),
        ("RELATIONS", &medh5::clinical::RELATIONS[..]),
        ("CLOCK_REFERENCES", &medh5::clinical::CLOCK_REFERENCES[..]),
        ("LESION_VALUES", &medh5::clinical::LESION_VALUES[..]),
        ("SELECTION_POLICIES", &medh5::clinical::POLICIES[..]),
    ] {
        m.add(name, pyo3::types::PyTuple::new(py, values.iter().copied())?)?;
    }
    Ok(())
}
