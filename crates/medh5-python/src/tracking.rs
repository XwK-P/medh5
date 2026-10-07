//! `medh5.curation.tracking`: joining `instance_id` across timepoints (§7.4).

use ndarray::{ArrayD, IxDyn};
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use medh5::curation::tracking as engine;

use crate::convert::{array_to_py, json_to_py};
use crate::values::{dataclass_repr, opt};

#[pyclass(module = "medh5.curation.tracking", name = "Observation", skip_from_py_object, frozen)]
#[derive(Clone)]
pub struct Observation(pub engine::Observation);

#[pymethods]
impl Observation {
    #[getter]
    fn timepoint(&self) -> &str {
        &self.0.timepoint
    }
    #[getter]
    fn annotation(&self) -> &str {
        &self.0.annotation
    }
    #[getter]
    fn index(&self) -> usize {
        self.0.index
    }
    #[getter]
    fn instance_id(&self) -> u64 {
        self.0.instance_id
    }
    #[getter]
    fn class_id(&self) -> i64 {
        self.0.class_id
    }
    #[getter(r#box)]
    fn bbox<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        array_to_py(py, self.0.bbox.clone().into_dyn())
    }
    #[getter]
    fn voxel_count(&self) -> Option<u64> {
        self.0.voxel_count
    }
    #[getter]
    fn volume(&self) -> Option<f64> {
        self.0.volume
    }
    #[getter]
    fn units(&self) -> Option<&str> {
        self.0.units.as_deref()
    }
    #[getter]
    fn score(&self) -> Option<f64> {
        self.0.score
    }
    #[getter]
    fn grid(&self) -> Option<&str> {
        self.0.grid.as_deref()
    }
    #[getter]
    fn centroid<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let c = self.0.centroid();
        Ok(array_to_py(py, ArrayD::from_shape_vec(IxDyn(&[c.len()]), c).expect("1-D")))
    }
    #[getter]
    fn extent<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let e = self.0.extent();
        Ok(array_to_py(py, ArrayD::from_shape_vec(IxDyn(&[e.len()]), e).expect("1-D")))
    }
    fn to_json<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &self.0.to_json())
    }
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other.cast::<Observation>().map(|o| o.get().0 == self.0).unwrap_or(false)
    }
    fn __repr__(&self) -> String {
        self.0.repr()
    }
}

#[pyclass(module = "medh5.curation.tracking", name = "Track", skip_from_py_object, frozen)]
#[derive(Clone)]
pub struct Track(pub engine::Track);

#[pymethods]
impl Track {
    #[getter]
    fn instance_id(&self) -> u64 {
        self.0.instance_id
    }
    #[getter]
    fn class_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.0.class_ids)
    }
    #[getter]
    fn observations<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.0.observations.iter().cloned().map(Observation))
    }
    #[getter]
    fn class_key(&self) -> Option<&str> {
        self.0.class_key.as_deref()
    }
    #[getter]
    fn class_id(&self) -> i64 {
        self.0.class_id()
    }
    #[getter]
    fn has_class_conflict(&self) -> bool {
        self.0.has_class_conflict()
    }
    #[getter]
    fn timepoints<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.0.timepoints())
    }
    fn at(&self, timepoint: &str) -> Option<Observation> {
        self.0.at(timepoint).cloned().map(Observation)
    }
    fn volume(&self, timepoint: &str) -> Option<f64> {
        self.0.volume(timepoint)
    }
    #[getter]
    fn volumes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (k, v) in self.0.volumes() {
            out.set_item(k, v)?;
        }
        Ok(out)
    }
    fn relative_change(&self, first: &str, second: &str) -> Option<f64> {
        self.0.relative_change(first, second)
    }
    fn to_json<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &self.0.to_json())
    }
    fn __len__(&self) -> usize {
        self.0.observations.len()
    }
    fn __iter__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        Ok(slf.getattr("observations")?.try_iter()?.into_any())
    }
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other.cast::<Track>().map(|o| o.get().0 == self.0).unwrap_or(false)
    }
    fn __repr__(&self) -> String {
        self.0.repr()
    }
}

/// `{instance_id: Track}` with the coverage needed to tell *resolved* from
/// *unexamined*.
#[pyclass(module = "medh5.curation.tracking", name = "Tracking", frozen)]
pub struct Tracking(pub engine::Tracking);

impl Tracking {
    pub fn wrap(t: engine::Tracking) -> Self {
        Tracking(t)
    }

    fn require(&self, instance_id: u64) -> PyResult<&engine::Track> {
        self.0.tracks.get(&instance_id).ok_or_else(|| PyKeyError::new_err(instance_id))
    }
}

#[pymethods]
impl Tracking {
    fn __getitem__(&self, instance_id: u64) -> PyResult<Track> {
        Ok(Track(self.require(instance_id)?.clone()))
    }
    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(PyTuple::new(py, self.0.tracks.keys())?.try_iter()?.into_any())
    }
    fn __len__(&self) -> usize {
        self.0.tracks.len()
    }
    fn __contains__(&self, instance_id: &Bound<'_, PyAny>) -> bool {
        instance_id.extract::<u64>().map(|i| self.0.tracks.contains_key(&i)).unwrap_or(false)
    }
    fn keys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.0.tracks.keys())
    }
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.0.tracks.values().cloned().map(Track))
    }
    fn items<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let pairs = self
            .0
            .tracks
            .iter()
            .map(|(k, v)| {
                PyTuple::new(py, [k.into_pyobject(py)?.into_any(), Bound::new(py, Track(v.clone()))?.into_any()])
            })
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, pairs)
    }
    #[pyo3(signature = (instance_id, default=None))]
    fn get<'py>(
        &self,
        py: Python<'py>,
        instance_id: &Bound<'py, PyAny>,
        default: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        match instance_id.extract::<u64>().ok().and_then(|i| self.0.tracks.get(&i)) {
            Some(t) => Ok(Bound::new(py, Track(t.clone()))?.into_any()),
            None => Ok(default.unwrap_or_else(|| py.None().into_bound(py))),
        }
    }
    #[getter]
    fn timepoints<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.0.timepoints)
    }
    #[getter]
    fn coverage<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (k, v) in &self.0.coverage {
            out.set_item(k, pyo3::types::PyFrozenSet::new(py, v)?)?;
        }
        Ok(out)
    }
    fn state_at(&self, instance_id: u64, timepoint: &str) -> PyResult<&'static str> {
        self.require(instance_id)?;
        Ok(self.0.state_at(instance_id, timepoint).unwrap_or(engine::UNEXAMINED))
    }
    fn states<'py>(&self, py: Python<'py>, instance_id: u64) -> PyResult<Bound<'py, PyDict>> {
        self.require(instance_id)?;
        let out = PyDict::new(py);
        for (k, v) in self.0.states(instance_id) {
            out.set_item(k, v)?;
        }
        Ok(out)
    }
    fn is_new(&self, instance_id: u64) -> PyResult<bool> {
        self.require(instance_id)?;
        Ok(self.0.is_new(instance_id))
    }
    fn is_resolved(&self, instance_id: u64) -> PyResult<bool> {
        self.require(instance_id)?;
        Ok(self.0.is_resolved(instance_id))
    }
    fn is_persistent(&self, instance_id: u64) -> PyResult<bool> {
        self.require(instance_id)?;
        Ok(self.0.is_persistent(instance_id))
    }
    fn class_conflicts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (k, v) in self.0.class_conflicts() {
            out.set_item(k, PyTuple::new(py, v)?)?;
        }
        Ok(out)
    }
    fn unexamined<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (k, v) in self.0.unexamined() {
            out.set_item(k, PyTuple::new(py, v)?)?;
        }
        Ok(out)
    }
    fn to_json<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &self.0.to_json())
    }
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &self.0.summary())
    }
    fn __repr__(&self) -> String {
        self.0.repr()
    }
}

/// Whether an annotation carries object identity (`instance_ids`) to join on.
#[pyfunction]
fn carries_instance_ids(annotation: &Bound<'_, PyAny>) -> PyResult<bool> {
    let handle = match annotation.cast::<crate::reader::AnnotationHandle>() {
        Ok(h) => h.clone(),
        Err(_) => annotation.getattr("_handle")?.cast_into::<crate::reader::AnnotationHandle>()?,
    };
    Ok(engine::carries_instance_ids(&handle.get().inner))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(carries_instance_ids, m)?)?;
    m.add_class::<Observation>()?;
    m.add_class::<Track>()?;
    m.add_class::<Tracking>()?;
    m.add("PRESENT", engine::PRESENT)?;
    m.add("RESOLVED", engine::RESOLVED)?;
    m.add("UNEXAMINED", engine::UNEXAMINED)?;
    m.add("STATES", PyTuple::new(m.py(), engine::STATES)?)?;
    let _ = (dataclass_repr, opt::<i64>);
    Ok(())
}
