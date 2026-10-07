//! `medh5.document`: the `/meta` sample document (spec §2.4).

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use serde_json::{Map, Value};

use medh5::document as engine;

use crate::convert::{json_to_py, map_to_py, py_to_json};
use crate::curation::{timeline_arg, Cohort, Deidentification, Identity, QualityRecord, SplitClaim, Timeline};
use crate::errors::R;
use crate::labels::{label_set_arg, LabelSet};

/// Where a document object's fields live.
enum Source {
    /// Its own copy: a document read from a file or built by hand.
    Owned(engine::SampleDocument),
    /// A writer's document, read and edited in place: `SampleWriter.document`,
    /// as 1.x had it, so `w.document.label_set = ...` reaches the file.
    Writer(Py<crate::writer::SampleWriter>),
}

/// The whole document, typed.  Fields read and write the engine's types.
#[pyclass(module = "medh5.document", name = "SampleDocument", skip_from_py_object)]
pub struct SampleDocument {
    source: Source,
}

fn object(value: Value) -> Map<String, Value> {
    match value {
        Value::Object(m) => m,
        _ => Map::new(),
    }
}

impl SampleDocument {
    pub fn owned(inner: engine::SampleDocument) -> Self {
        SampleDocument { source: Source::Owned(inner) }
    }

    /// The live document of `writer`.
    pub fn of_writer(writer: Py<crate::writer::SampleWriter>) -> Self {
        SampleDocument { source: Source::Writer(writer) }
    }

    /// Whether this is the live document of the writer at `writer`.
    pub fn is_view_of(&self, writer: *mut pyo3::ffi::PyObject) -> bool {
        matches!(&self.source, Source::Writer(w) if w.as_ptr() == writer)
    }

    fn read<T>(&self, py: Python<'_>, f: impl FnOnce(&engine::SampleDocument) -> T) -> PyResult<T> {
        match &self.source {
            Source::Owned(doc) => Ok(f(doc)),
            Source::Writer(w) => Ok(f(w.bind(py).try_borrow()?.inner.document())),
        }
    }

    fn edit<T>(&mut self, py: Python<'_>, f: impl FnOnce(&mut engine::SampleDocument) -> T) -> R<T> {
        match &mut self.source {
            Source::Owned(doc) => Ok(f(doc)),
            Source::Writer(w) => {
                let mut writer = w.bind(py).try_borrow_mut()?;
                Ok(f(writer.engine()?.document_mut()))
            }
        }
    }

    /// A copy of the document as it stands.
    pub fn current(&self, py: Python<'_>) -> PyResult<engine::SampleDocument> {
        self.read(py, |doc| doc.clone())
    }
}

#[pymethods]
impl SampleDocument {
    #[new]
    #[pyo3(signature = (identity, timepoints, cohort=None, label_set=None, provenance=None, quality=None, splits=None, acquisition=None, deidentification=None, extra=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        identity: &Bound<'_, PyAny>,
        timepoints: &Bound<'_, PyAny>,
        cohort: Option<&Bound<'_, PyAny>>,
        label_set: Option<&Bound<'_, PyAny>>,
        provenance: Option<&Bound<'_, PyAny>>,
        quality: Option<&Bound<'_, PyAny>>,
        splits: Option<&Bound<'_, PyAny>>,
        acquisition: Option<&Bound<'_, PyAny>>,
        deidentification: Option<&Bound<'_, PyAny>>,
        extra: Option<&Bound<'_, PyAny>>,
    ) -> R<Self> {
        let mut doc = engine::SampleDocument::new(Identity::arg(identity)?, timeline_arg(timepoints)?);
        let me = &mut doc;
        if let Some(c) = cohort.filter(|c| !c.is_none()) {
            me.cohort = Cohort::arg(c)?;
        }
        if let Some(ls) = label_set.filter(|c| !c.is_none()) {
            me.label_set = Some(label_set_arg(ls)?);
        }
        if let Some(p) = provenance.filter(|c| !c.is_none()) {
            me.provenance = provenance_arg(p)?;
        }
        if let Some(q) = quality.filter(|c| !c.is_none()) {
            me.quality = quality_arg(q)?;
        }
        if let Some(s) = splits.filter(|c| !c.is_none()) {
            me.splits = s.try_iter()?.map(|i| SplitClaim::arg(&i?)).collect::<R<Vec<_>>>()?;
        }
        if let Some(a) = acquisition.filter(|c| !c.is_none()) {
            me.acquisition = object(py_to_json(a)?);
        }
        if let Some(d) = deidentification.filter(|c| !c.is_none()) {
            me.deidentification = Some(Deidentification::arg(d)?);
        }
        if let Some(e) = extra.filter(|c| !c.is_none()) {
            me.extra = object(py_to_json(e)?);
        }
        Ok(SampleDocument::owned(doc))
    }
    #[getter]
    fn identity(&self, py: Python<'_>) -> PyResult<Identity> {
        self.read(py, |d| Identity::wrap(d.identity.clone()))
    }
    #[setter]
    fn set_identity(&mut self, py: Python<'_>, value: &Bound<'_, PyAny>) -> R<()> {
        let value = Identity::arg(value)?;
        self.edit(py, |d| d.identity = value)
    }
    #[getter]
    fn timepoints(&self, py: Python<'_>) -> PyResult<Timeline> {
        self.read(py, |d| Timeline::wrap(d.timepoints.clone()))
    }
    #[setter]
    fn set_timepoints(&mut self, py: Python<'_>, value: &Bound<'_, PyAny>) -> R<()> {
        let value = timeline_arg(value)?;
        self.edit(py, |d| d.timepoints = value)
    }
    #[getter]
    fn cohort(&self, py: Python<'_>) -> PyResult<Cohort> {
        self.read(py, |d| Cohort::wrap(d.cohort.clone()))
    }
    #[setter]
    fn set_cohort(&mut self, py: Python<'_>, value: &Bound<'_, PyAny>) -> R<()> {
        let value = Cohort::arg(value)?;
        self.edit(py, |d| d.cohort = value)
    }
    #[getter]
    fn label_set(&self, py: Python<'_>) -> PyResult<Option<LabelSet>> {
        self.read(py, |d| d.label_set.clone().map(LabelSet::wrap))
    }
    #[setter]
    fn set_label_set(&mut self, py: Python<'_>, value: &Bound<'_, PyAny>) -> R<()> {
        let value = if value.is_none() { None } else { Some(label_set_arg(value)?) };
        self.edit(py, |d| d.label_set = value)
    }
    #[getter]
    fn provenance(&self, py: Python<'_>) -> PyResult<crate::curation::Provenance> {
        self.read(py, |d| crate::curation::Provenance { inner: d.provenance.clone() })
    }
    #[setter]
    fn set_provenance(&mut self, py: Python<'_>, value: &Bound<'_, PyAny>) -> R<()> {
        let value = provenance_arg(value)?;
        self.edit(py, |d| d.provenance = value)
    }
    #[getter]
    fn quality<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let records = self.read(py, |d| d.quality.clone())?;
        let out = PyDict::new(py);
        for (k, v) in records {
            out.set_item(k, QualityRecord::wrap(v))?;
        }
        Ok(out)
    }
    #[setter]
    fn set_quality(&mut self, py: Python<'_>, value: &Bound<'_, PyAny>) -> R<()> {
        let value = quality_arg(value)?;
        self.edit(py, |d| d.quality = value)
    }
    #[getter]
    fn splits<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let claims = self.read(py, |d| d.splits.clone())?;
        PyTuple::new(py, claims.into_iter().map(SplitClaim::wrap))
    }
    #[setter]
    fn set_splits(&mut self, py: Python<'_>, value: &Bound<'_, PyAny>) -> R<()> {
        let value = value.try_iter()?.map(|i| SplitClaim::arg(&i?)).collect::<R<Vec<_>>>()?;
        self.edit(py, |d| d.splits = value)
    }
    #[getter]
    fn acquisition<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        map_to_py(py, &self.read(py, |d| d.acquisition.clone())?)
    }
    #[setter]
    fn set_acquisition(&mut self, py: Python<'_>, value: &Bound<'_, PyAny>) -> R<()> {
        let value = object(py_to_json(value)?);
        self.edit(py, |d| d.acquisition = value)
    }
    #[getter]
    fn deidentification(&self, py: Python<'_>) -> PyResult<Option<Deidentification>> {
        self.read(py, |d| d.deidentification.clone().map(Deidentification::wrap))
    }
    #[setter]
    fn set_deidentification(&mut self, py: Python<'_>, value: &Bound<'_, PyAny>) -> R<()> {
        let value = if value.is_none() { None } else { Some(Deidentification::arg(value)?) };
        self.edit(py, |d| d.deidentification = value)
    }
    #[getter]
    fn extra<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        map_to_py(py, &self.read(py, |d| d.extra.clone())?)
    }
    #[setter]
    fn set_extra(&mut self, py: Python<'_>, value: &Bound<'_, PyAny>) -> R<()> {
        let value = object(py_to_json(value)?);
        self.edit(py, |d| d.extra = value)
    }
    fn to_json<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &self.read(py, |d| d.to_json())?)
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, pyo3::types::PyType>, doc: &Bound<'_, PyAny>) -> R<Self> {
        Ok(SampleDocument::owned(engine::SampleDocument::from_json(&py_to_json(doc)?)?))
    }
    #[pyo3(signature = (*, indent=None))]
    fn dumps(&self, py: Python<'_>, indent: Option<usize>) -> PyResult<String> {
        self.read(py, |d| match indent {
            Some(n) => d.dumps_indented(n),
            None => d.dumps(),
        })
    }
    #[classmethod]
    fn loads(_cls: &Bound<'_, pyo3::types::PyType>, payload: &Bound<'_, PyAny>) -> R<Self> {
        let text: String = match payload.extract::<String>() {
            Ok(s) => s,
            Err(_) => {
                let bytes: Vec<u8> = payload.extract()?;
                String::from_utf8_lossy(&bytes).into_owned()
            }
        };
        Ok(SampleDocument::owned(engine::SampleDocument::loads(&text)?))
    }
    fn check_schema(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        self.read(py, |d| d.check_schema())
    }
    #[getter]
    fn subject_id(&self, py: Python<'_>) -> PyResult<String> {
        self.read(py, |d| d.subject_id().to_string())
    }
    #[getter]
    fn group_id(&self, py: Python<'_>) -> PyResult<String> {
        self.read(py, |d| d.group_id().to_string())
    }
    fn quality_of(&self, py: Python<'_>, key: Option<&str>) -> PyResult<Option<QualityRecord>> {
        self.read(py, |d| d.quality_of(key).cloned().map(QualityRecord::wrap))
    }
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &self.read(py, |d| d.summary())?)
    }
    fn __eq__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let Ok(other) = other.cast::<SampleDocument>() else { return Ok(false) };
        let theirs = other.try_borrow()?.current(py)?;
        self.read(py, |d| *d == theirs)
    }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        self.read(py, |d| {
            format!(
                "SampleDocument(sample_id={}, subject_id={}, timepoints={})",
                medh5::json::repr_str(&d.identity.sample_id),
                medh5::json::repr_str(&d.identity.subject_id),
                medh5::json::repr_list(&d.timepoints.ids())
            )
        })
    }
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        crate::values::reduce_via_json(slf.as_any())
    }
}

pub fn provenance_arg(obj: &Bound<'_, PyAny>) -> R<medh5::curation::provenance::Provenance> {
    if let Ok(p) = obj.cast::<crate::curation::Provenance>() {
        return Ok(p.borrow().inner.clone());
    }
    Ok(medh5::curation::provenance::Provenance::from_json(Some(&py_to_json(obj)?))?)
}

fn quality_arg(obj: &Bound<'_, PyAny>) -> R<indexmap::IndexMap<String, medh5::curation::quality::QualityRecord>> {
    let mut out = indexmap::IndexMap::new();
    for item in obj.call_method0("items")?.try_iter()? {
        let (k, v): (String, Bound<'_, PyAny>) = item?.extract()?;
        out.insert(k, QualityRecord::arg(&v)?);
    }
    Ok(out)
}

#[pyfunction]
fn schema<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    json_to_py(py, engine::schema())
}

#[pyfunction]
fn schema_text() -> &'static str {
    engine::schema_text()
}

#[pyfunction]
fn validate_against_schema(doc: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    Ok(engine::validate_against_schema(&py_to_json(doc)?))
}

#[pyfunction]
#[pyo3(signature = (sample_id, subject_id=None, *, timepoints=None, **identity_fields))]
fn new_document(
    sample_id: &str,
    subject_id: Option<&str>,
    timepoints: Option<&Bound<'_, PyAny>>,
    identity_fields: Option<&Bound<'_, PyDict>>,
) -> R<SampleDocument> {
    let mut doc = engine::new_document(sample_id, subject_id, None)?;
    if let Some(t) = timepoints.filter(|t| !t.is_none()) {
        doc.timepoints = match t.cast::<Timeline>() {
            Ok(tl) => tl.get().inner.clone(),
            Err(_) => {
                let ids = crate::convert::strings(t)?;
                engine::new_document(sample_id, subject_id, Some(&ids))?.timepoints
            }
        };
    }
    if let Some(fields) = identity_fields {
        let mut identity = object(doc.identity.to_json());
        identity.extend(crate::convert::kwargs_to_map(Some(fields))?);
        doc.identity = medh5::curation::identity::Identity::from_json(&Value::Object(identity))?;
    }
    Ok(SampleDocument::owned(doc))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SampleDocument>()?;
    m.add_function(wrap_pyfunction!(schema, m)?)?;
    m.add_function(wrap_pyfunction!(schema_text, m)?)?;
    m.add_function(wrap_pyfunction!(validate_against_schema, m)?)?;
    m.add_function(wrap_pyfunction!(new_document, m)?)?;
    m.add("META_DATASET", engine::META_DATASET)?;
    Ok(())
}
