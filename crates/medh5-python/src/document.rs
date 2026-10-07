//! `medh5.document`: the `/meta` sample document (spec §2.4).

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use serde_json::{Map, Value};

use medh5::document as engine;

use crate::convert::{json_to_py, map_to_py, py_to_json};
use crate::curation::{timeline_arg, Cohort, Deidentification, Identity, QualityRecord, SplitClaim, Timeline};
use crate::errors::R;
use crate::labels::{label_set_arg, LabelSet};

/// The whole document, typed.  Fields read and write the engine's types.
#[pyclass(module = "medh5.document", name = "SampleDocument", skip_from_py_object)]
pub struct SampleDocument {
    pub inner: engine::SampleDocument,
}

fn object(value: Value) -> Map<String, Value> {
    match value {
        Value::Object(m) => m,
        _ => Map::new(),
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
        Ok(SampleDocument { inner: doc })
    }
    #[getter]
    fn identity(&self) -> Identity {
        Identity::wrap(self.inner.identity.clone())
    }
    #[setter]
    fn set_identity(&mut self, value: &Bound<'_, PyAny>) -> R<()> {
        self.inner.identity = Identity::arg(value)?;
        Ok(())
    }
    #[getter]
    fn timepoints(&self) -> Timeline {
        Timeline::wrap(self.inner.timepoints.clone())
    }
    #[setter]
    fn set_timepoints(&mut self, value: &Bound<'_, PyAny>) -> R<()> {
        self.inner.timepoints = timeline_arg(value)?;
        Ok(())
    }
    #[getter]
    fn cohort(&self) -> Cohort {
        Cohort::wrap(self.inner.cohort.clone())
    }
    #[setter]
    fn set_cohort(&mut self, value: &Bound<'_, PyAny>) -> R<()> {
        self.inner.cohort = Cohort::arg(value)?;
        Ok(())
    }
    #[getter]
    fn label_set(&self) -> Option<LabelSet> {
        self.inner.label_set.clone().map(LabelSet::wrap)
    }
    #[setter]
    fn set_label_set(&mut self, value: &Bound<'_, PyAny>) -> R<()> {
        self.inner.label_set = if value.is_none() { None } else { Some(label_set_arg(value)?) };
        Ok(())
    }
    #[getter]
    fn provenance(&self) -> crate::curation::Provenance {
        crate::curation::Provenance { inner: self.inner.provenance.clone() }
    }
    #[setter]
    fn set_provenance(&mut self, value: &Bound<'_, PyAny>) -> R<()> {
        self.inner.provenance = provenance_arg(value)?;
        Ok(())
    }
    #[getter]
    fn quality<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (k, v) in &self.inner.quality {
            out.set_item(k, QualityRecord::wrap(v.clone()))?;
        }
        Ok(out)
    }
    #[setter]
    fn set_quality(&mut self, value: &Bound<'_, PyAny>) -> R<()> {
        self.inner.quality = quality_arg(value)?;
        Ok(())
    }
    #[getter]
    fn splits<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.splits.iter().cloned().map(SplitClaim::wrap))
    }
    #[setter]
    fn set_splits(&mut self, value: &Bound<'_, PyAny>) -> R<()> {
        self.inner.splits = value.try_iter()?.map(|i| SplitClaim::arg(&i?)).collect::<R<Vec<_>>>()?;
        Ok(())
    }
    #[getter]
    fn acquisition<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        map_to_py(py, &self.inner.acquisition)
    }
    #[setter]
    fn set_acquisition(&mut self, value: &Bound<'_, PyAny>) -> R<()> {
        self.inner.acquisition = object(py_to_json(value)?);
        Ok(())
    }
    #[getter]
    fn deidentification(&self) -> Option<Deidentification> {
        self.inner.deidentification.clone().map(Deidentification::wrap)
    }
    #[setter]
    fn set_deidentification(&mut self, value: &Bound<'_, PyAny>) -> R<()> {
        self.inner.deidentification = if value.is_none() { None } else { Some(Deidentification::arg(value)?) };
        Ok(())
    }
    #[getter]
    fn extra<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        map_to_py(py, &self.inner.extra)
    }
    #[setter]
    fn set_extra(&mut self, value: &Bound<'_, PyAny>) -> R<()> {
        self.inner.extra = object(py_to_json(value)?);
        Ok(())
    }
    fn to_json<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &self.inner.to_json())
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, pyo3::types::PyType>, doc: &Bound<'_, PyAny>) -> R<Self> {
        Ok(SampleDocument { inner: engine::SampleDocument::from_json(&py_to_json(doc)?)? })
    }
    #[pyo3(signature = (*, indent=None))]
    fn dumps(&self, indent: Option<usize>) -> String {
        match indent {
            Some(n) => self.inner.dumps_indented(n),
            None => self.inner.dumps(),
        }
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
        Ok(SampleDocument { inner: engine::SampleDocument::loads(&text)? })
    }
    fn check_schema(&self) -> Vec<String> {
        self.inner.check_schema()
    }
    #[getter]
    fn subject_id(&self) -> &str {
        self.inner.subject_id()
    }
    #[getter]
    fn group_id(&self) -> &str {
        self.inner.group_id()
    }
    fn quality_of(&self, key: Option<&str>) -> Option<QualityRecord> {
        self.inner.quality_of(key).cloned().map(QualityRecord::wrap)
    }
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &self.inner.summary())
    }
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other.cast::<SampleDocument>().map(|o| o.borrow().inner == self.inner).unwrap_or(false)
    }
    fn __repr__(&self) -> String {
        format!(
            "SampleDocument(sample_id={}, subject_id={}, timepoints={})",
            medh5::json::repr_str(&self.inner.identity.sample_id),
            medh5::json::repr_str(&self.inner.identity.subject_id),
            medh5::json::repr_list(&self.inner.timepoints.ids())
        )
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
    Ok(SampleDocument { inner: doc })
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
