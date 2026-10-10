//! `medh5.curation` value types: identity, provenance, quality, timeline
//! (spec §3.7, §11, §12).

use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyFrozenSet, PySlice, PyString, PyTuple};
use serde_json::Value;

use medh5::curation::identity as id_engine;
use medh5::curation::provenance as prov_engine;
use medh5::curation::quality as q_engine;
use medh5::curation::timeline as tl_engine;

use crate::convert::{json_to_py, map_to_py, py_to_json};
use crate::errors::R;
use crate::record_class;
use crate::records::{field, required, Kind};

// -- identity ------------------------------------------------------------------------------

record_class!(
    Identity,
    "Identity",
    "medh5.curation",
    id_engine::Identity,
    parse = id_engine::Identity::from_json,
    dump = |v| v.to_json(),
    fields = [
        required("sample_id", Kind::Plain),
        required("subject_id", Kind::Plain),
        field("sex", Kind::Plain),
        field("laterality", Kind::Plain),
        field("bodypart", Kind::Plain),
        field("extra", Kind::Extra),
    ]
);

record_class!(
    Cohort,
    "Cohort",
    "medh5.curation",
    id_engine::Cohort,
    parse = |v| id_engine::Cohort::from_json(Some(v)),
    dump = |v| v.to_json(),
    fields = [
        field("dataset_id", Kind::Plain),
        field("site_id", Kind::Plain),
        field("scanner_id", Kind::Plain),
        field("group_id", Kind::Plain),
        field("acquisition_protocol", Kind::Plain),
        field("extra", Kind::Extra),
    ],
    methods = {
        /// The key subjects are grouped by when splitting (§12.2).
        fn grouping_key(&self, subject_id: &str) -> String {
            self.inner.grouping_key(subject_id).to_string()
        }
    }
);

record_class!(
    SplitClaim,
    "SplitClaim",
    "medh5.curation",
    id_engine::SplitClaim,
    parse = id_engine::SplitClaim::from_json,
    dump = |v| v.to_json(),
    fields = [
        required("set_id", Kind::Plain),
        required("partition", Kind::Plain),
        field("fold", Kind::Plain),
        field("assigned_by", Kind::Plain),
        field("assigned_at", Kind::Plain),
        field("manifest_sha256", Kind::Plain),
    ]
);

fn parse_deidentification(v: &Value) -> medh5::Result<id_engine::Deidentification> {
    id_engine::Deidentification::from_json(Some(v))?
        .ok_or_else(|| medh5::Error::invalid("a de-identification record needs at least its `method` (§11.4)"))
}

record_class!(
    Deidentification,
    "Deidentification",
    "medh5.curation",
    id_engine::Deidentification,
    parse = parse_deidentification,
    dump = |v| v.to_json(),
    nullable = true,
    fields = [
        required("method", Kind::Plain),
        field("profile", Kind::Plain),
        field("date_shift_days", Kind::Plain),
        field("id_mapping", Kind::Plain),
        field("performed_by", Kind::Plain),
        field("date", Kind::Plain),
        field("burned_in_annotation_checked", Kind::Plain),
        field("extra", Kind::Extra),
    ]
);

#[pyfunction]
fn splits_from_json<'py>(py: Python<'py>, docs: &Bound<'py, PyAny>) -> R<Bound<'py, PyTuple>> {
    let doc = if docs.is_none() { None } else { Some(py_to_json(docs)?) };
    let claims = id_engine::splits_from_json(doc.as_ref())?;
    Ok(PyTuple::new(py, claims.into_iter().map(SplitClaim::wrap))?)
}

// -- provenance ------------------------------------------------------------------------------

record_class!(
    Agent,
    "Agent",
    "medh5.curation",
    prov_engine::Agent,
    parse = prov_engine::Agent::from_json,
    dump = |v| v.to_json(),
    fields = [
        required("id", Kind::Plain),
        required("type", Kind::Plain),
        required("name", Kind::Plain),
        field("role", Kind::Plain),
        field("version", Kind::Plain),
        field("qualification", Kind::Plain),
        field("organization", Kind::Plain),
    ]
);

record_class!(
    Activity,
    "Activity",
    "medh5.curation",
    prov_engine::Activity,
    parse = prov_engine::Activity::from_json,
    dump = |v| v.to_json(),
    fields = [
        required("id", Kind::Plain),
        required("type", Kind::Plain),
        field("agent", Kind::Plain),
        field("started", Kind::Plain),
        field("ended", Kind::Plain),
        field("tool", Kind::Plain),
        field("inputs", Kind::Tuple),
        field("outputs", Kind::Tuple),
        field("params", Kind::Dict),
    ]
);

/// Who did what (§11.1): agents and the activities they performed.
#[pyclass(module = "medh5.curation", name = "Provenance", skip_from_py_object)]
pub struct Provenance {
    pub inner: prov_engine::Provenance,
}

fn agent_arg(obj: &Bound<'_, PyAny>) -> R<prov_engine::Agent> {
    Agent::arg(obj)
}

fn activity_arg(obj: &Bound<'_, PyAny>) -> R<prov_engine::Activity> {
    Activity::arg(obj)
}

#[pymethods]
impl Provenance {
    #[new]
    #[pyo3(signature = (agents=None, activities=None))]
    fn new(agents: Option<&Bound<'_, PyAny>>, activities: Option<&Bound<'_, PyAny>>) -> R<Self> {
        let mut a = Vec::new();
        if let Some(items) = agents {
            for item in items.try_iter()? {
                a.push(agent_arg(&item?)?);
            }
        }
        let mut b = Vec::new();
        if let Some(items) = activities {
            for item in items.try_iter()? {
                b.push(activity_arg(&item?)?);
            }
        }
        Ok(Provenance { inner: prov_engine::Provenance::new(a, b)? })
    }
    fn __repr__(&self) -> String {
        self.inner.repr()
    }
    #[getter]
    fn agents<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.agents().cloned().map(Agent::wrap).collect::<Vec<_>>())
    }
    #[getter]
    fn activities<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.activities().cloned().map(Activity::wrap).collect::<Vec<_>>())
    }
    fn agent(&self, agent_id: &str) -> R<Agent> {
        Ok(Agent::wrap(self.inner.agent(agent_id)?.clone()))
    }
    fn activity(&self, activity_id: &str) -> R<Activity> {
        Ok(Activity::wrap(self.inner.activity(activity_id)?.clone()))
    }
    fn has_activity(&self, activity_id: &str) -> bool {
        self.inner.has_activity(activity_id)
    }
    fn has_agent(&self, agent_id: &str) -> bool {
        self.inner.has_agent(agent_id)
    }
    #[pyo3(signature = (agent, *, replace=false))]
    fn add_agent(&mut self, agent: &Bound<'_, PyAny>, replace: bool) -> R<Agent> {
        Ok(Agent::wrap(self.inner.add_agent(agent_arg(agent)?, replace)?))
    }
    #[pyo3(signature = (activity, *, replace=false))]
    fn add_activity(&mut self, activity: &Bound<'_, PyAny>, replace: bool) -> R<Activity> {
        Ok(Activity::wrap(self.inner.add_activity(activity_arg(activity)?, replace)?))
    }
    fn activities_by_type<'py>(&self, py: Python<'py>, activity_type: &str) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.activities_by_type(activity_type).into_iter().cloned().map(Activity::wrap))
    }
    fn produced_by<'py>(&self, py: Python<'py>, object_path: &str) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.produced_by(object_path).into_iter().cloned().map(Activity::wrap))
    }
    fn dangling_agent_refs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.dangling_agent_refs())
    }
    fn __iter__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        Ok(slf.getattr("activities")?.try_iter()?.into_any())
    }
    fn __len__(&self) -> usize {
        self.inner.n_activities()
    }
    fn to_json<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &self.inner.to_json())
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, pyo3::types::PyType>, doc: &Bound<'_, PyAny>) -> R<Self> {
        let value = if doc.is_none() { None } else { Some(py_to_json(doc)?) };
        Ok(Provenance { inner: prov_engine::Provenance::from_json(value.as_ref())? })
    }
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other.cast::<Provenance>().map(|o| o.borrow().inner == self.inner).unwrap_or(false)
    }
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        crate::values::reduce_via_json(slf.as_any())
    }
}

#[pyfunction]
#[pyo3(signature = (value, *, r#where))]
fn check_timestamp(value: &str, r#where: &str) -> R<String> {
    prov_engine::check_timestamp(value, r#where)?;
    Ok(value.to_string())
}

// -- quality --------------------------------------------------------------------------------

record_class!(
    Agreement,
    "Agreement",
    "medh5.curation",
    q_engine::Agreement,
    parse = q_engine::Agreement::from_json,
    dump = |v| v.to_json(),
    fields = [
        required("metric", Kind::Plain),
        required("value", Kind::Float),
        field("against", Kind::Plain),
        field("per_class", Kind::Dict),
    ]
);

record_class!(
    Issue,
    "Issue",
    "medh5.curation",
    q_engine::Issue,
    parse = q_engine::Issue::from_json,
    dump = |v| v.to_json(),
    fields = [
        required("code", Kind::Plain),
        field("severity", Kind::Default("info")),
        field("class_ids", Kind::Tuple),
        field("note", Kind::Plain),
    ]
);

record_class!(
    QualityRecord,
    "QualityRecord",
    "medh5.curation",
    q_engine::QualityRecord,
    parse = q_engine::QualityRecord::from_json,
    dump = |v| v.to_json(),
    fields = [
        required("status", Kind::Plain),
        field("confidence", Kind::Float),
        field("reviewed_by", Kind::Tuple),
        field("agreement", Kind::Records(Agreement::py_from_json)),
        field("issues", Kind::Records(Issue::py_from_json)),
        field("edit_effort_s", Kind::Float),
    ],
    methods = {
        /// Whether the record may be trained on: not rejected or deprecated.
        #[getter]
        fn is_usable(&self) -> bool {
            self.inner.is_usable()
        }
    }
);

#[pyfunction]
fn quality_from_json<'py>(py: Python<'py>, doc: &Bound<'py, PyAny>) -> R<Bound<'py, PyDict>> {
    let value = if doc.is_none() { None } else { Some(py_to_json(doc)?) };
    let records = q_engine::quality_from_json(value.as_ref())?;
    let out = PyDict::new(py);
    for (k, v) in records {
        out.set_item(k, QualityRecord::wrap(v))?;
    }
    Ok(out)
}

#[pyfunction]
fn quality_to_json<'py>(py: Python<'py>, records: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
    let mut map = indexmap::IndexMap::new();
    for item in records.call_method0("items")?.try_iter()? {
        let (k, v): (String, Bound<'_, PyAny>) = item?.extract()?;
        map.insert(k, QualityRecord::arg(&v)?);
    }
    Ok(json_to_py(py, &q_engine::quality_to_json(&map))?)
}

#[pyfunction]
#[pyo3(signature = (per_class, against=None))]
fn dice_agreement(per_class: &Bound<'_, PyAny>, against: Option<String>) -> R<Agreement> {
    let mut pairs = Vec::new();
    for item in per_class.call_method0("items")?.try_iter()? {
        let (k, v): (i64, f64) = item?.extract()?;
        pairs.push((k, v));
    }
    Ok(Agreement::wrap(q_engine::dice_agreement(&pairs, against)))
}

// -- timeline --------------------------------------------------------------------------------

record_class!(
    Timepoint,
    "Timepoint",
    "medh5.curation",
    tl_engine::Timepoint,
    parse = tl_engine::Timepoint::from_json,
    dump = |v| v.to_json(),
    fields = [
        required("id", Kind::Plain),
        required("index", Kind::Plain),
        field("label", Kind::Plain),
        field("date", Kind::Plain),
        field("days_from_baseline", Kind::Plain),
        field("study_uid", Kind::Plain),
        field("series_uids", Kind::Dict),
        field("subject_age_years", Kind::Plain),
        field("description", Kind::Plain),
    ]
);

/// The sample's timepoints, in acquisition order; indexable by position or id.
#[pyclass(module = "medh5.curation", name = "Timeline", skip_from_py_object, frozen)]
pub struct Timeline {
    pub inner: tl_engine::Timeline,
    points: PyOnceLock<Py<PyTuple>>,
}

impl Timeline {
    pub fn wrap(inner: tl_engine::Timeline) -> Self {
        Timeline { inner, points: PyOnceLock::new() }
    }

    fn objects<'py>(&self, py: Python<'py>) -> PyResult<&Bound<'py, PyTuple>> {
        Ok(self
            .points
            .get_or_try_init(py, || {
                PyTuple::new(py, self.inner.iter().cloned().map(Timepoint::wrap)).map(Bound::unbind)
            })?
            .bind(py))
    }

    fn position(&self, id: &str) -> Option<usize> {
        self.inner.iter().position(|t| t.id == id)
    }
}

/// A timeline argument: the binding class, or a list of timepoints.
pub fn timeline_arg(obj: &Bound<'_, PyAny>) -> R<tl_engine::Timeline> {
    if let Ok(t) = obj.cast::<Timeline>() {
        return Ok(t.get().inner.clone());
    }
    let mut points = Vec::new();
    for item in obj.try_iter()? {
        points.push(Timepoint::arg(&item?)?);
    }
    Ok(tl_engine::Timeline::new(points)?)
}

#[pymethods]
impl Timeline {
    #[new]
    fn new(timepoints: &Bound<'_, PyAny>) -> R<Self> {
        Ok(Timeline::wrap(timeline_arg(timepoints)?))
    }
    fn check(&self) -> R<()> {
        Ok(self.inner.check()?)
    }
    /// The position of the first timepoint equal to `value` (`Sequence.index`).
    #[pyo3(signature = (value, start=0, stop=None))]
    fn index(&self, py: Python<'_>, value: &Bound<'_, PyAny>, start: isize, stop: Option<isize>) -> PyResult<usize> {
        let items = self.objects(py)?;
        let n = items.len() as isize;
        let clamp = |i: isize| (if i < 0 { (i + n).max(0) } else { i.min(n) }) as usize;
        let (lo, hi) = (clamp(start), clamp(stop.unwrap_or(n)));
        for i in lo..hi.max(lo) {
            if items.get_item(i)?.eq(value)? {
                return Ok(i);
            }
        }
        Err(pyo3::exceptions::PyValueError::new_err("timepoint is not in the timeline"))
    }
    /// How many timepoints equal `value` (`Sequence.count`).
    fn count(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        let mut found = 0;
        for item in self.objects(py)?.iter() {
            if item.eq(value)? {
                found += 1;
            }
        }
        Ok(found)
    }
    fn __len__(&self) -> usize {
        self.inner.len()
    }
    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(self.objects(py)?.try_iter()?.into_any())
    }
    fn __getitem__<'py>(&self, py: Python<'py>, key: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let objects = self.objects(py)?;
        if let Ok(id) = key.cast::<PyString>() {
            let id = id.to_string();
            return match self.position(&id) {
                Some(i) => objects.get_item(i),
                None => Err(PyKeyError::new_err(format!(
                    "undeclared timepoint {}; declared: {}",
                    medh5::json::repr_str(&id),
                    medh5::json::repr_list(&self.inner.ids())
                ))),
            };
        }
        if key.is_instance_of::<PySlice>() {
            return objects.as_any().get_item(key);
        }
        objects.as_any().get_item(key)
    }
    fn __contains__(&self, py: Python<'_>, item: &Bound<'_, PyAny>) -> PyResult<bool> {
        if let Ok(id) = item.cast::<PyString>() {
            return Ok(self.inner.contains(&id.to_string()));
        }
        self.objects(py)?.contains(item)
    }
    fn __repr__(&self) -> String {
        self.inner.repr()
    }
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other.cast::<Timeline>().map(|o| o.get().inner == self.inner).unwrap_or(false)
    }
    #[getter]
    fn ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.ids())
    }
    #[getter]
    fn baseline<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.objects(py)?.get_item(0)
    }
    #[getter]
    fn is_longitudinal(&self) -> bool {
        self.inner.is_longitudinal()
    }
    fn interval_days(&self, a: &str, b: &str) -> R<Option<f64>> {
        Ok(self.inner.interval_days(a, b)?)
    }
    #[pyo3(signature = (timepoint_id, *, r#where=""))]
    fn require<'py>(&self, py: Python<'py>, timepoint_id: &str, r#where: &str) -> R<Bound<'py, PyAny>> {
        let found = self.inner.require(timepoint_id, r#where)?;
        let index = self.position(&found.id).unwrap_or(0);
        Ok(self.objects(py)?.get_item(index)?)
    }
    fn to_json<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &self.inner.to_json())
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, pyo3::types::PyType>, docs: &Bound<'_, PyAny>) -> R<Self> {
        Ok(Timeline::wrap(tl_engine::Timeline::from_json(&py_to_json(docs)?)?))
    }
    #[classmethod]
    #[pyo3(signature = (timepoint_id="tp0", **kwargs))]
    fn single(
        _cls: &Bound<'_, pyo3::types::PyType>,
        timepoint_id: &str,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> R<Self> {
        let mut doc = crate::convert::kwargs_to_map(kwargs)?;
        doc.insert("id".into(), Value::String(timepoint_id.into()));
        doc.insert("index".into(), Value::from(0));
        let point = tl_engine::Timepoint::from_json(&Value::Object(doc))?;
        Ok(Timeline::wrap(tl_engine::Timeline::new(vec![point])?))
    }
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        crate::values::reduce_via_json(slf.as_any())
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add_class::<Identity>()?;
    m.add_class::<Cohort>()?;
    m.add_class::<SplitClaim>()?;
    m.add_class::<Deidentification>()?;
    m.add_class::<Agent>()?;
    m.add_class::<Activity>()?;
    m.add_class::<Provenance>()?;
    m.add_class::<Agreement>()?;
    m.add_class::<Issue>()?;
    m.add_class::<QualityRecord>()?;
    m.add_class::<Timepoint>()?;
    m.add_class::<Timeline>()?;
    for f in [
        wrap_pyfunction!(splits_from_json, m)?,
        wrap_pyfunction!(check_timestamp, m)?,
        wrap_pyfunction!(quality_from_json, m)?,
        wrap_pyfunction!(quality_to_json, m)?,
        wrap_pyfunction!(dice_agreement, m)?,
    ] {
        m.add_function(f)?;
    }
    m.add("SEX_VALUES", PyTuple::new(py, id_engine::SEX_VALUES)?)?;
    m.add("LATERALITY_VALUES", PyTuple::new(py, id_engine::LATERALITY_VALUES)?)?;
    m.add("PARTITIONS", PyTuple::new(py, id_engine::PARTITIONS)?)?;
    m.add("ID_SOURCE", id_engine::ID_SOURCE)?;
    m.add("PSEUDONYM_SOURCE", id_engine::PSEUDONYM_SOURCE)?;
    m.add("AGENT_TYPES", PyTuple::new(py, prov_engine::AGENT_TYPES)?)?;
    m.add("AGENT_FIELDS", PyFrozenSet::new(py, prov_engine::AGENT_FIELDS)?)?;
    m.add("ACTIVITY_FIELDS", PyFrozenSet::new(py, prov_engine::ACTIVITY_FIELDS)?)?;
    m.add("ACTIVITY_TYPES", PyTuple::new(py, prov_engine::ACTIVITY_TYPES)?)?;
    m.add("QUALITY_FIELDS", PyFrozenSet::new(py, q_engine::QUALITY_FIELDS)?)?;
    m.add("QUALITY_STATUS", PyTuple::new(py, q_engine::QUALITY_STATUS)?)?;
    m.add("ISSUE_SEVERITY", PyTuple::new(py, q_engine::ISSUE_SEVERITY)?)?;
    m.add("TIMEPOINT_FIELDS", PyFrozenSet::new(py, tl_engine::TIMEPOINT_FIELDS)?)?;
    let _ = map_to_py;
    Ok(())
}
