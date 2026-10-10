//! Read handles: a sample, its images, annotations, transforms and indices.
//!
//! The Python facades in `medh5/sample.py`, `medh5/image.py`,
//! `medh5/annotations/` and `medh5/transforms.py` wrap these; everything the
//! facades answer is computed here, by the engine.

use std::sync::{Arc, Mutex};

use ndarray::{ArrayD, IxDyn};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySlice, PyString, PyTuple};

use medh5::annotations::{Annotation as EngineAnnotation, GridRef};
use medh5::array::{Index, Slice};
use medh5::sample::{Image as EngineImage, Sample as EngineSample};
use medh5::storage::index::SamplingIndex as EngineIndex;
use medh5::transforms::model::Transform as EngineTransform;

use crate::convert::{
    array_to_py, attr_to_py, class_key, class_keys, dtype_arg, dtype_to_py, f64_array, json_to_py, nd_to_py, slices,
};
use crate::errors::R;
use crate::geometry::Grid;
use crate::nodes::Dataset;

fn closed() -> medh5::Error {
    medh5::Error::File("this sample has been closed".into())
}

fn opt_keys(classes: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<medh5::labels::ClassKey>>> {
    match classes {
        Some(c) if !c.is_none() => class_keys(c).map(Some),
        _ => Ok(None),
    }
}

fn tuple_of<'py, T: IntoPyObject<'py> + Clone>(py: Python<'py>, items: &[T]) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(py, items.iter().cloned())
}

// -- Image ----------------------------------------------------------------------------------

#[pyclass(module = "medh5._core", name = "ImageHandle", frozen)]
pub struct ImageHandle {
    pub inner: Arc<EngineImage>,
}

#[pymethods]
impl ImageHandle {
    #[getter]
    fn image_id(&self) -> &str {
        &self.inner.image_id
    }
    #[getter]
    fn is_multiscale(&self) -> bool {
        self.inner.is_multiscale()
    }
    #[getter]
    fn levels(&self) -> R<usize> {
        Ok(self.inner.levels()?)
    }
    #[getter]
    fn level_index(&self) -> usize {
        self.inner.level_index()
    }
    fn level(&self, index: usize) -> R<ImageHandle> {
        Ok(ImageHandle { inner: Arc::new(self.inner.level(index)?) })
    }
    #[getter]
    fn pyramid(&self) -> R<Option<crate::geometry::Pyramid>> {
        Ok(self.inner.pyramid()?.map(crate::geometry::Pyramid))
    }
    #[getter]
    fn dataset(&self) -> R<Dataset> {
        Ok(Dataset { ds: self.inner.dataset()? })
    }
    #[getter]
    fn attrs<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for name in self.inner.attr_names()? {
            if let Some(value) = self.inner.attr(&name)? {
                out.set_item(&name, attr_to_py(py, &value)?)?;
            }
        }
        Ok(out)
    }
    #[getter]
    fn grid_id(&self) -> R<String> {
        Ok(self.inner.grid_id()?)
    }
    #[getter]
    fn grid(&self) -> R<Grid> {
        Ok(Grid::wrap(self.inner.grid()?.clone()))
    }
    #[getter]
    fn timepoint(&self) -> R<Option<String>> {
        Ok(self.inner.timepoint()?)
    }
    #[getter]
    fn modality(&self) -> R<String> {
        Ok(self.inner.modality()?)
    }
    #[getter]
    fn value_type(&self) -> R<String> {
        Ok(self.inner.value_type()?)
    }
    #[getter]
    fn value_units(&self) -> R<Option<String>> {
        Ok(self.inner.value_units()?)
    }
    #[getter]
    fn channel_names<'py>(&self, py: Python<'py>) -> R<Option<Bound<'py, PyTuple>>> {
        Ok(self.inner.channel_names()?.map(|n| PyTuple::new(py, n)).transpose()?)
    }
    #[getter]
    fn rescale(&self) -> R<(f64, f64)> {
        Ok(self.inner.rescale()?)
    }
    #[getter]
    fn is_rescaled(&self) -> R<bool> {
        Ok(self.inner.is_rescaled()?)
    }
    #[getter]
    fn window<'py>(&self, py: Python<'py>) -> R<Option<Bound<'py, PyTuple>>> {
        Ok(match self.inner.window()? {
            Some((c, w)) => Some(PyTuple::new(py, [PyTuple::new(py, c)?, PyTuple::new(py, w)?])?),
            None => None,
        })
    }
    #[getter]
    fn valid_mask(&self) -> R<Option<String>> {
        Ok(self.inner.valid_mask()?)
    }
    #[getter]
    fn prov(&self) -> R<Option<String>> {
        Ok(self.inner.prov()?)
    }
    #[getter]
    fn digest(&self) -> R<Option<String>> {
        Ok(self.inner.digest()?)
    }
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyTuple>> {
        Ok(PyTuple::new(py, self.inner.shape()?)?)
    }
    #[getter]
    fn dtype<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        Ok(dtype_to_py(py, self.inner.dtype()?)?)
    }
    #[getter]
    fn nbytes(&self) -> R<usize> {
        Ok(self.inner.nbytes()?)
    }
    #[getter]
    fn chunks<'py>(&self, py: Python<'py>) -> R<Option<Bound<'py, PyTuple>>> {
        Ok(self.inner.chunks()?.map(|c| PyTuple::new(py, c)).transpose()?)
    }
    #[pyo3(signature = (roi=None, *, physical=false, dtype=None))]
    fn read<'py>(
        &self,
        py: Python<'py>,
        roi: Option<&Bound<'py, PyAny>>,
        physical: bool,
        dtype: Option<&Bound<'py, PyAny>>,
    ) -> R<Bound<'py, PyAny>> {
        let dtype = match dtype {
            Some(d) if !d.is_none() => Some(dtype_arg(d)?),
            _ => None,
        };
        let grid = self.inner.grid()?;
        let index = match crate::convert::region(roi)? {
            None => vec![Index::Slice(Slice::full()); grid.ndim()],
            Some(items) if items.len() == grid.ndim() => items,
            Some(items) if items.len() == grid.n_spatial() => {
                let mut out = vec![Index::Slice(Slice::full()); grid.ndim() - grid.n_spatial()];
                out.extend(items);
                out
            }
            Some(items) => {
                return Err(medh5::Error::invalid(format!(
                    "roi has {} axes; image {} has {} ({} spatial)",
                    items.len(),
                    medh5::json::repr_str(&self.inner.image_id),
                    grid.ndim(),
                    grid.n_spatial()
                ))
                .into())
            }
        };
        let image = self.inner.clone();
        let block = py.detach(move || image.read_index(&index, physical, dtype))?;
        Ok(nd_to_py(py, block))
    }
    fn summary<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        Ok(json_to_py(py, &self.inner.summary()?)?)
    }
    fn __repr__(&self) -> R<String> {
        Ok(self.inner.repr()?)
    }
}

// -- Annotation --------------------------------------------------------------------------------

#[pyclass(module = "medh5._core", name = "AnnotationHandle", frozen)]
pub struct AnnotationHandle {
    pub inner: Arc<EngineAnnotation>,
    /// Object columns as handed out: read once, the same read-only array
    /// every time, so `ann.boxes[i]` in a loop does not copy the column.
    columns: Mutex<std::collections::HashMap<&'static str, Py<PyAny>>>,
}

impl AnnotationHandle {
    pub fn wrap(inner: Arc<EngineAnnotation>) -> Self {
        AnnotationHandle { inner, columns: Mutex::new(std::collections::HashMap::new()) }
    }

    /// A column, made once and frozen: a caller writing into it would
    /// otherwise change what every later read returns.
    fn column<'py>(
        &self,
        py: Python<'py>,
        name: &'static str,
        make: impl FnOnce() -> R<Bound<'py, PyAny>>,
    ) -> R<Bound<'py, PyAny>> {
        if let Some(found) = self.columns.lock().unwrap().get(name) {
            return Ok(found.bind(py).clone());
        }
        let value = make()?;
        if !value.is_none() {
            value.getattr("flags")?.setattr("writeable", false)?;
        }
        self.columns.lock().unwrap().entry(name).or_insert_with(|| value.clone().unbind());
        Ok(value)
    }
}

/// A `grid=` argument: `None` (the annotation's own), a grid id, or a Grid.
fn grid_ref_owned(grid: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Result<String, medh5::geometry::Grid>>> {
    match grid {
        None => Ok(None),
        Some(g) if g.is_none() => Ok(None),
        Some(g) => {
            if let Ok(s) = g.cast::<PyString>() {
                return Ok(Some(Ok(s.to_string())));
            }
            Ok(Some(Err(crate::geometry::grid_arg(g)?)))
        }
    }
}

fn with_grid<T>(
    grid: &Option<Result<String, medh5::geometry::Grid>>,
    f: impl FnOnce(GridRef<'_>) -> medh5::Result<T>,
) -> medh5::Result<T> {
    match grid {
        None => f(GridRef::Own),
        Some(Ok(id)) => f(GridRef::Id(id)),
        Some(Err(g)) => f(GridRef::Grid(g)),
    }
}

fn instance_tuple<'py>(py: Python<'py>, i: medh5::annotations::Instance) -> PyResult<Bound<'py, PyTuple>> {
    let mask = match i.mask {
        Some(m) => array_to_py(py, m),
        None => py.None().into_bound(py),
    };
    PyTuple::new(
        py,
        [
            i.index.into_pyobject(py)?.into_any(),
            i.instance_id.into_pyobject(py)?.into_any(),
            i.class_id.into_pyobject(py)?.into_any(),
            array_to_py(py, i.bbox.into_dyn()),
            mask,
            crate::values::opt(py, i.score)?,
        ],
    )
}

fn assertion_tuple<'py>(py: Python<'py>, a: &medh5::annotations::Assertion) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(
        py,
        [
            a.class_id.into_pyobject(py)?.into_any(),
            a.value.into_pyobject(py)?.into_any(),
            crate::values::opt(py, a.scope_id)?,
            crate::values::opt(py, a.scheme.clone())?,
            crate::values::opt(py, a.scheme_value.clone())?,
        ],
    )
}

#[pymethods]
impl AnnotationHandle {
    #[getter]
    fn ann_id(&self) -> &str {
        &self.inner.ann_id
    }
    /// The annotation's stored group, for inspection.
    #[getter]
    fn group(&self) -> crate::nodes::Group {
        crate::nodes::Group::wrap(self.inner.group.clone())
    }
    #[getter]
    fn header(&self) -> crate::annotations::AnnotationHeader {
        crate::annotations::AnnotationHeader { inner: self.inner.header.clone() }
    }
    #[getter]
    fn kind(&self) -> &str {
        self.inner.kind()
    }
    #[getter]
    fn class_name(&self) -> &'static str {
        self.inner.class_name()
    }
    #[getter]
    fn task(&self) -> &str {
        self.inner.task()
    }
    #[getter]
    fn class_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        tuple_of(py, self.inner.class_ids())
    }
    #[getter]
    fn annotated_class_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        tuple_of(py, self.inner.annotated_class_ids())
    }
    #[getter]
    fn closure(&self) -> &str {
        self.inner.closure()
    }
    #[getter]
    fn ignore_id(&self) -> i64 {
        self.inner.ignore_id()
    }
    #[getter]
    fn prov(&self) -> Option<&str> {
        self.inner.prov()
    }
    #[getter]
    fn quality_key(&self) -> Option<&str> {
        self.inner.quality_key()
    }
    #[getter]
    fn label_set(&self) -> Option<crate::labels::LabelSet> {
        self.inner.label_set().map(|ls| crate::labels::LabelSet::wrap(ls.clone()))
    }
    #[getter]
    fn grid_id(&self) -> Option<&str> {
        self.inner.grid_id()
    }
    #[getter]
    fn grid(&self) -> R<Grid> {
        Ok(Grid::wrap(self.inner.grid()?.clone()))
    }
    #[getter]
    fn timepoints<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.timepoints())
    }
    fn is_annotated(&self, class_key_: &Bound<'_, PyAny>) -> R<bool> {
        Ok(self.inner.is_annotated(&class_key(class_key_)?)?)
    }
    #[getter]
    fn is_fully_covered(&self) -> bool {
        self.inner.is_fully_covered()
    }
    #[getter]
    fn has_ignore_region(&self) -> R<bool> {
        Ok(self.inner.has_ignore_region()?)
    }
    fn resolve_class(&self, key: &Bound<'_, PyAny>) -> R<i64> {
        Ok(self.inner.resolve_class(&class_key(key)?)?)
    }
    #[pyo3(signature = (keys=None))]
    fn resolve_classes<'py>(&self, py: Python<'py>, keys: Option<&Bound<'py, PyAny>>) -> R<Bound<'py, PyTuple>> {
        let keys = opt_keys(keys)?;
        Ok(PyTuple::new(py, self.inner.resolve_classes(keys.as_deref())?)?)
    }
    #[getter]
    fn classes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.classes().into_iter().map(crate::labels::LabelClass))
    }
    #[getter]
    fn annotated_classes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.annotated_classes().into_iter().map(crate::labels::LabelClass))
    }
    fn class_key(&self, class_id: i64) -> String {
        self.inner.class_key(class_id)
    }
    fn summary<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        Ok(json_to_py(py, &self.inner.summary()?)?)
    }
    fn __repr__(&self) -> String {
        self.inner.repr()
    }
    /// A stored dataset of the annotation group.
    fn dataset(&self, name: &str) -> R<Dataset> {
        Ok(Dataset { ds: self.inner.dataset(name)? })
    }
    fn has_dataset(&self, name: &str) -> bool {
        self.inner.optional_dataset(name).is_some()
    }

    // -- voxel ------------------------------------------------------------------
    #[getter]
    fn spatial_shape<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyTuple>> {
        Ok(PyTuple::new(py, self.inner.spatial_shape()?)?)
    }
    #[pyo3(signature = (classes=None, roi=None))]
    fn dense<'py>(
        &self,
        py: Python<'py>,
        classes: Option<&Bound<'py, PyAny>>,
        roi: Option<&Bound<'py, PyAny>>,
    ) -> R<Bound<'py, PyAny>> {
        let classes = opt_keys(classes)?;
        let roi = slices(roi)?;
        let ann = self.inner.clone();
        let out = py.detach(move || ann.dense(classes.as_deref(), roi.as_deref()))?;
        Ok(array_to_py(py, out))
    }
    fn contains(&self, class_key_: &Bound<'_, PyAny>, voxel: Vec<i64>) -> R<bool> {
        Ok(self.inner.contains(&class_key(class_key_)?, &voxel)?)
    }
    /// `(volume, overwritten, order, warning)`.
    #[pyo3(signature = (roi=None, priority=None, dtype=None))]
    fn labelmap<'py>(
        &self,
        py: Python<'py>,
        roi: Option<&Bound<'py, PyAny>>,
        priority: Option<&Bound<'py, PyAny>>,
        dtype: Option<&Bound<'py, PyAny>>,
    ) -> R<Bound<'py, PyTuple>> {
        let roi = slices(roi)?;
        let priority = opt_keys(priority)?;
        let dtype = match dtype {
            Some(d) if !d.is_none() => dtype_arg(d)?,
            _ => medh5::array::DType::U16,
        };
        let ann = self.inner.clone();
        let (volume, overwritten, order) =
            py.detach(move || ann.labelmap(roi.as_deref(), priority.as_deref(), dtype))?;
        let warning = (overwritten > 0).then(|| self.inner.flatten_warning(overwritten, &order));
        Ok(PyTuple::new(
            py,
            [
                nd_to_py(py, volume),
                overwritten.into_pyobject(py)?.into_any(),
                PyTuple::new(py, order)?.into_any(),
                crate::values::opt(py, warning)?,
            ],
        )?)
    }
    #[pyo3(signature = (classes=None))]
    fn voxel_counts<'py>(&self, py: Python<'py>, classes: Option<&Bound<'py, PyAny>>) -> R<Bound<'py, PyDict>> {
        let classes = opt_keys(classes)?;
        let ann = self.inner.clone();
        let counts = py.detach(move || ann.voxel_counts(classes.as_deref()))?;
        let out = PyDict::new(py);
        for (k, v) in counts {
            out.set_item(k, v)?;
        }
        Ok(out)
    }
    #[pyo3(signature = (classes=None))]
    fn class_bboxes<'py>(&self, py: Python<'py>, classes: Option<&Bound<'py, PyAny>>) -> R<Bound<'py, PyDict>> {
        let classes = opt_keys(classes)?;
        let ann = self.inner.clone();
        let boxes = py.detach(move || ann.class_bboxes(classes.as_deref()))?;
        let out = PyDict::new(py);
        for (k, v) in boxes {
            match v {
                Some(b) => out.set_item(k, array_to_py(py, b.into_dyn()))?,
                None => out.set_item(k, py.None())?,
            }
        }
        Ok(out)
    }
    #[pyo3(signature = (roi=None))]
    fn ignore_mask<'py>(&self, py: Python<'py>, roi: Option<&Bound<'py, PyAny>>) -> R<Bound<'py, PyAny>> {
        let roi = slices(roi)?;
        Ok(array_to_py(py, self.inner.ignore_mask(roi.as_deref())?))
    }
    #[pyo3(signature = (roi=None))]
    fn read_mask<'py>(&self, py: Python<'py>, roi: Option<&Bound<'py, PyAny>>) -> R<Bound<'py, PyAny>> {
        let roi = slices(roi)?;
        Ok(array_to_py(py, self.inner.read_mask(roi.as_deref())?))
    }
    #[getter]
    fn layer_class_ids<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        Ok(array_to_py(py, self.inner.layer_class_ids()?.into_dyn()))
    }
    #[getter]
    fn n_layers(&self) -> R<usize> {
        Ok(self.inner.n_layers()?)
    }
    #[getter]
    fn layer_of<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (k, v) in self.inner.layer_of()? {
            out.set_item(k, v)?;
        }
        Ok(out)
    }
    fn layer_classes<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyTuple>> {
        let layers = self.inner.layer_classes()?;
        Ok(PyTuple::new(py, layers.into_iter().map(|l| PyTuple::new(py, l)).collect::<PyResult<Vec<_>>>()?)?)
    }
    #[pyo3(signature = (layer, roi=None))]
    fn read_layer<'py>(&self, py: Python<'py>, layer: usize, roi: Option<&Bound<'py, PyAny>>) -> R<Bound<'py, PyAny>> {
        let roi = slices(roi)?;
        Ok(nd_to_py(py, self.inner.read_layer(layer, roi.as_deref())?))
    }
    #[getter]
    fn bit_class_ids<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        let ids = self.inner.bit_class_ids()?;
        Ok(array_to_py(py, ArrayD::from_shape_vec(IxDyn(&[ids.len()]), ids)?))
    }
    #[getter]
    fn n_planes(&self) -> R<usize> {
        Ok(self.inner.n_planes()?)
    }
    #[getter]
    fn position_of<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        let mut found: Vec<(i64, usize)> = self.inner.position_of()?.into_iter().collect();
        found.sort_by_key(|(_, p)| *p);
        for (k, v) in found {
            out.set_item(k, v)?;
        }
        Ok(out)
    }
    fn classes_at<'py>(&self, py: Python<'py>, voxel: Vec<i64>) -> R<Bound<'py, PyTuple>> {
        Ok(PyTuple::new(py, self.inner.classes_at(&voxel)?)?)
    }
    #[getter]
    fn normalized(&self) -> R<bool> {
        Ok(self.inner.normalized()?)
    }
    #[getter]
    fn threshold(&self) -> R<f64> {
        Ok(self.inner.threshold()?)
    }
    #[pyo3(signature = (classes=None, roi=None))]
    fn probabilities<'py>(
        &self,
        py: Python<'py>,
        classes: Option<&Bound<'py, PyAny>>,
        roi: Option<&Bound<'py, PyAny>>,
    ) -> R<Bound<'py, PyAny>> {
        let classes = opt_keys(classes)?;
        let roi = slices(roi)?;
        Ok(array_to_py(py, self.inner.probabilities(classes.as_deref(), roi.as_deref())?))
    }
    #[getter]
    fn has_masks(&self) -> bool {
        self.inner.has_masks()
    }
    #[getter]
    fn boxes<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        self.column(py, "boxes", || Ok(array_to_py(py, self.inner.boxes()?)))
    }
    #[getter]
    fn object_class_ids<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        self.column(py, "object_class_ids", || {
            let ids = self.inner.object_class_ids()?;
            Ok(array_to_py(py, ArrayD::from_shape_vec(IxDyn(&[ids.len()]), ids)?))
        })
    }
    #[getter]
    fn instance_ids<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        self.column(py, "instance_ids", || {
            Ok(match self.inner.instance_ids()? {
                Some(ids) => array_to_py(py, ArrayD::from_shape_vec(IxDyn(&[ids.len()]), ids)?),
                None => py.None().into_bound(py),
            })
        })
    }
    #[getter]
    fn scores<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        self.column(py, "scores", || {
            Ok(match self.inner.scores()? {
                Some(s) => array_to_py(py, ArrayD::from_shape_vec(IxDyn(&[s.len()]), s)?),
                None => py.None().into_bound(py),
            })
        })
    }
    #[getter]
    fn n_objects(&self) -> R<usize> {
        Ok(self.inner.n_objects()?)
    }
    fn crop<'py>(&self, py: Python<'py>, index: usize) -> R<Bound<'py, PyAny>> {
        Ok(match self.inner.crop(index)? {
            Some(c) => array_to_py(py, c),
            None => py.None().into_bound(py),
        })
    }
    /// Every instance as `(index, instance_id, class_id, box, mask, score)`.
    fn instances<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyList>> {
        let ann = self.inner.clone();
        let found = py.detach(move || ann.instances())?;
        Ok(PyList::new(py, found.into_iter().map(|i| instance_tuple(py, i)).collect::<PyResult<Vec<_>>>()?)?)
    }
    fn instance<'py>(&self, py: Python<'py>, instance_id: u64) -> R<Bound<'py, PyTuple>> {
        Ok(instance_tuple(py, self.inner.instance(instance_id)?)?)
    }
    fn tracking<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (k, v) in self.inner.tracking()? {
            out.set_item(k, v)?;
        }
        Ok(out)
    }

    // -- geometric --------------------------------------------------------------------
    #[getter]
    fn space(&self) -> R<String> {
        Ok(self.inner.space()?)
    }
    #[getter]
    fn frame_uid(&self) -> Option<String> {
        self.inner.frame_uid()
    }
    #[getter]
    fn n_spatial(&self) -> R<usize> {
        Ok(self.inner.n_spatial()?)
    }
    #[pyo3(signature = (coords, *, grid=None))]
    fn to_world<'py>(
        &self,
        py: Python<'py>,
        coords: &Bound<'py, PyAny>,
        grid: Option<&Bound<'py, PyAny>>,
    ) -> R<Bound<'py, PyAny>> {
        let grid = grid_ref_owned(grid)?;
        let coords = f64_array(coords)?;
        Ok(array_to_py(py, with_grid(&grid, |g| self.inner.to_world(&coords, g))?))
    }
    #[pyo3(signature = (coords, *, grid=None))]
    fn to_index<'py>(
        &self,
        py: Python<'py>,
        coords: &Bound<'py, PyAny>,
        grid: Option<&Bound<'py, PyAny>>,
    ) -> R<Bound<'py, PyAny>> {
        let grid = grid_ref_owned(grid)?;
        let coords = f64_array(coords)?;
        Ok(array_to_py(py, with_grid(&grid, |g| self.inner.to_index(&coords, g))?))
    }
    #[getter]
    fn attributes<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        Ok(match self.inner.attributes()? {
            Some(items) => {
                PyTuple::new(py, items.iter().map(|v| json_to_py(py, v)).collect::<PyResult<Vec<_>>>()?)?.into_any()
            }
            None => py.None().into_bound(py),
        })
    }
    #[getter]
    fn n_items(&self) -> R<usize> {
        Ok(self.inner.n_items()?)
    }
    #[getter]
    fn box_ndim(&self) -> R<usize> {
        Ok(self.inner.box_ndim()?)
    }
    #[getter]
    fn slice_index<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        Ok(match self.inner.slice_index()? {
            Some(s) => array_to_py(py, ArrayD::from_shape_vec(IxDyn(&[s.len()]), s)?),
            None => py.None().into_bound(py),
        })
    }
    #[pyo3(signature = (grid=None))]
    fn as_slices<'py>(&self, py: Python<'py>, grid: Option<&Bound<'py, PyAny>>) -> R<Bound<'py, PyList>> {
        let grid = grid_ref_owned(grid)?;
        let found = with_grid(&grid, |g| self.inner.as_slices(g))?;
        let builtins = py.import("builtins")?;
        let slice = builtins.getattr("slice")?;
        let out = PyList::empty(py);
        for item in found {
            let parts = item.iter().map(|(a, b)| slice.call1((*a, *b))).collect::<PyResult<Vec<_>>>()?;
            out.append(PyTuple::new(py, parts)?)?;
        }
        Ok(out)
    }
    #[pyo3(signature = (grid=None))]
    fn world_corners<'py>(&self, py: Python<'py>, grid: Option<&Bound<'py, PyAny>>) -> R<Bound<'py, PyAny>> {
        let grid = grid_ref_owned(grid)?;
        Ok(array_to_py(py, with_grid(&grid, |g| self.inner.world_corners(g))?.into_dyn()))
    }
    #[pyo3(signature = (grid=None))]
    fn as_world<'py>(&self, py: Python<'py>, grid: Option<&Bound<'py, PyAny>>) -> R<Bound<'py, PyAny>> {
        let grid = grid_ref_owned(grid)?;
        Ok(array_to_py(py, with_grid(&grid, |g| self.inner.as_world(g))?.into_dyn()))
    }
    fn obb_corners<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        Ok(array_to_py(py, self.inner.obb_corners()?.into_dyn()))
    }
    fn obb_as_aabb<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        Ok(array_to_py(py, self.inner.obb_as_aabb()?.into_dyn()))
    }
    fn obb_volumes<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        let v = self.inner.obb_volumes()?;
        Ok(array_to_py(py, ArrayD::from_shape_vec(IxDyn(&[v.len()]), v)?))
    }
    #[getter]
    fn visibility<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        Ok(array_to_py(py, self.inner.visibility()?))
    }
    #[getter]
    fn skeleton_id(&self) -> R<Option<String>> {
        Ok(self.inner.skeleton_id()?)
    }
    fn skeleton(&self) -> R<Option<crate::labels::Skeleton>> {
        Ok(self.inner.skeleton()?.map(crate::labels::Skeleton))
    }
    fn labelled<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        Ok(array_to_py(py, self.inner.labelled()?))
    }
    #[getter]
    fn point_names<'py>(&self, py: Python<'py>) -> R<Option<Bound<'py, PyTuple>>> {
        Ok(self.inner.point_names()?.map(|n| PyTuple::new(py, n)).transpose()?)
    }
    #[getter]
    fn correspondence(&self) -> R<Option<String>> {
        Ok(self.inner.correspondence()?)
    }
    fn named_points<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (k, v) in self.inner.named_points()? {
            out.set_item(k, array_to_py(py, ArrayD::from_shape_vec(IxDyn(&[v.len()]), v)?))?;
        }
        Ok(out)
    }
    #[getter]
    fn contour_offsets<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        let v = self.inner.contour_offsets()?;
        Ok(array_to_py(py, ArrayD::from_shape_vec(IxDyn(&[v.len()]), v)?))
    }
    #[getter]
    fn contour_planes<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        Ok(array_to_py(py, self.inner.contour_planes()?.into_dyn()))
    }
    #[getter]
    fn contour_roles<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyTuple>> {
        Ok(PyTuple::new(py, self.inner.contour_roles()?)?)
    }
    fn polygon<'py>(&self, py: Python<'py>, index: usize) -> R<Bound<'py, PyAny>> {
        Ok(array_to_py(py, self.inner.polygon(index)?))
    }
    /// Every polygon as `(vertices, class_id, (axis, index), role)`.
    fn polygons<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        for p in self.inner.polygons()? {
            out.append((array_to_py(py, p.vertices), p.class_id, p.plane, p.role))?;
        }
        Ok(out)
    }
    fn by_plane<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for ((a, b), v) in self.inner.by_plane()? {
            out.set_item((a, b), PyList::new(py, v)?)?;
        }
        Ok(out)
    }
    #[getter]
    fn n_submeshes(&self) -> R<usize> {
        Ok(self.inner.n_submeshes()?)
    }
    fn mesh_bounds<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        Ok(array_to_py(py, self.inner.mesh_bounds()?.into_dyn()))
    }

    // -- classification ------------------------------------------------------------------
    #[getter]
    fn scope(&self) -> R<String> {
        Ok(self.inner.scope()?)
    }
    #[getter]
    fn multilabel(&self) -> R<bool> {
        Ok(self.inner.multilabel()?)
    }
    #[getter]
    fn asserted_class_ids<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        let v = self.inner.asserted_class_ids()?;
        Ok(array_to_py(py, ArrayD::from_shape_vec(IxDyn(&[v.len()]), v)?))
    }
    #[getter]
    fn assertion_values<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        let v = self.inner.assertion_values()?;
        Ok(array_to_py(py, ArrayD::from_shape_vec(IxDyn(&[v.len()]), v)?))
    }
    #[getter]
    fn scope_ids<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        Ok(match self.inner.scope_ids()? {
            Some(v) => array_to_py(py, ArrayD::from_shape_vec(IxDyn(&[v.len()]), v)?),
            None => py.None().into_bound(py),
        })
    }
    #[getter]
    fn schemes<'py>(&self, py: Python<'py>) -> R<Option<Bound<'py, PyTuple>>> {
        Ok(self.inner.schemes()?.map(|s| PyTuple::new(py, s)).transpose()?)
    }
    #[getter]
    fn scheme_values<'py>(&self, py: Python<'py>) -> R<Option<Bound<'py, PyTuple>>> {
        Ok(self.inner.scheme_values()?.map(|s| PyTuple::new(py, s)).transpose()?)
    }
    /// Every assertion as `(class_id, value, scope_id, scheme, scheme_value)`.
    fn assertions<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyList>> {
        let found = self.inner.assertions()?;
        Ok(PyList::new(py, found.iter().map(|a| assertion_tuple(py, a)).collect::<PyResult<Vec<_>>>()?)?)
    }
    #[getter]
    fn labels<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (k, v) in self.inner.labels()? {
            out.set_item(k, v)?;
        }
        Ok(out)
    }
    #[getter]
    fn positives<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyTuple>> {
        Ok(PyTuple::new(py, self.inner.positives()?)?)
    }
    #[pyo3(signature = (class_key_, *, scope_id=None))]
    fn value(&self, class_key_: &Bound<'_, PyAny>, scope_id: Option<i64>) -> R<Option<f64>> {
        Ok(self.inner.value(&class_key(class_key_)?, scope_id)?)
    }
    #[pyo3(signature = (class_key_, *, scope_id=None))]
    fn state(&self, class_key_: &Bound<'_, PyAny>, scope_id: Option<i64>) -> R<&'static str> {
        Ok(self.inner.state(&class_key(class_key_)?, scope_id)?)
    }
    fn scheme(&self, name: &str) -> R<Option<String>> {
        Ok(self.inner.scheme(name)?)
    }
    /// `{scope_id: [assertion tuples]}`.
    fn by_scope_id<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (k, v) in self.inner.by_scope_id()? {
            out.set_item(k, PyList::new(py, v.iter().map(|a| assertion_tuple(py, a)).collect::<PyResult<Vec<_>>>()?)?)?;
        }
        Ok(out)
    }
    #[getter]
    fn is_change_label(&self) -> R<bool> {
        Ok(self.inner.is_change_label()?)
    }
    #[getter]
    fn compared_timepoints<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyTuple>> {
        Ok(PyTuple::new(py, self.inner.compared_timepoints()?)?)
    }
}

// -- Transform -------------------------------------------------------------------------------

#[pyclass(module = "medh5._core", name = "TransformHandle", frozen)]
pub struct TransformHandle {
    pub inner: Arc<EngineTransform>,
}

impl TransformHandle {
    pub fn wrap(t: EngineTransform) -> Self {
        TransformHandle { inner: Arc::new(t) }
    }
}

#[pymethods]
impl TransformHandle {
    #[getter]
    fn transform_id(&self) -> &str {
        &self.inner.transform_id
    }
    /// The transform's stored group, for inspection.
    #[getter]
    fn group(&self) -> crate::nodes::Group {
        crate::nodes::Group::wrap(self.inner.group.clone())
    }
    #[getter]
    fn kind(&self) -> &str {
        self.inner.kind()
    }
    #[getter]
    fn class_name(&self) -> &'static str {
        self.inner.class_name()
    }
    #[getter(from_frame)]
    fn source_frame(&self) -> &str {
        self.inner.from_frame()
    }
    #[getter]
    fn to_frame(&self) -> &str {
        self.inner.to_frame()
    }
    #[getter]
    fn units(&self) -> &str {
        self.inner.units()
    }
    #[getter]
    fn prov(&self) -> Option<&str> {
        self.inner.prov()
    }
    #[getter]
    fn metrics_key(&self) -> Option<&str> {
        self.inner.metrics_key()
    }
    #[getter]
    fn header(&self) -> crate::transforms::TransformHeader {
        crate::transforms::TransformHeader { inner: self.inner.header.clone() }
    }
    #[getter]
    fn is_invertible(&self) -> bool {
        self.inner.is_invertible()
    }
    #[getter]
    fn timepoints<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.timepoints())
    }
    fn grid_in(&self, frame: &str) -> Option<Grid> {
        self.inner.grid_in(frame).cloned().map(Grid::wrap)
    }
    fn transform_points<'py>(&self, py: Python<'py>, points: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
        let points = f64_array(points)?;
        let t = self.inner.clone();
        Ok(array_to_py(py, py.detach(move || t.transform_points(&points))?))
    }
    fn inverse(&self) -> R<Option<TransformHandle>> {
        Ok(self.inner.inverse()?.map(TransformHandle::wrap))
    }
    #[getter]
    fn steps(&self) -> Option<Vec<TransformHandle>> {
        self.inner.steps().map(|s| s.iter().cloned().map(TransformHandle::wrap).collect())
    }
    #[getter]
    fn matrix<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        Ok(array_to_py(py, self.inner.matrix()?.into_dyn()))
    }
    #[getter]
    fn n_spatial(&self) -> R<usize> {
        Ok(self.inner.n_spatial()?)
    }
    fn inverse_matrix<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        Ok(array_to_py(py, self.inner.inverse_matrix()?.into_dyn()))
    }
    fn inverse_points<'py>(&self, py: Python<'py>, points: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
        Ok(array_to_py(py, self.inner.inverse_points(&f64_array(points)?)?))
    }
    #[getter]
    fn jacobian_determinant_value(&self) -> R<f64> {
        Ok(self.inner.jacobian_determinant_value()?)
    }
    #[getter]
    fn field(&self) -> R<Dataset> {
        Ok(Dataset { ds: self.inner.field()? })
    }
    #[getter]
    fn field_grid_id(&self) -> R<String> {
        Ok(self.inner.field_grid_id()?)
    }
    #[getter]
    fn field_grid(&self) -> R<Grid> {
        Ok(Grid::wrap(self.inner.field_grid()?.clone()))
    }
    #[getter]
    fn vector_space(&self) -> R<String> {
        Ok(self.inner.vector_space()?)
    }
    #[getter]
    fn interpolation(&self) -> R<String> {
        Ok(self.inner.interpolation()?)
    }
    #[getter]
    fn extrapolation(&self) -> R<String> {
        Ok(self.inner.extrapolation()?)
    }
    #[pyo3(signature = (roi=None, component=None))]
    fn read_field<'py>(
        &self,
        py: Python<'py>,
        roi: Option<&Bound<'py, PyAny>>,
        component: Option<usize>,
    ) -> R<Bound<'py, PyAny>> {
        let roi = slices(roi)?;
        Ok(nd_to_py(py, self.inner.read_field(roi.as_deref(), component)?))
    }
    fn displacement_at<'py>(&self, py: Python<'py>, points: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
        Ok(array_to_py(py, self.inner.displacement_at(&f64_array(points)?)?))
    }
    /// The stored field interpolated at continuous field indices, `(N, S)`.
    fn sample_indices<'py>(&self, py: Python<'py>, indices: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
        let found = crate::convert::matrix(indices)?;
        Ok(array_to_py(py, self.inner.sample_indices(&found)?.into_dyn()))
    }
    #[pyo3(signature = (roi=None))]
    fn jacobian_determinant<'py>(&self, py: Python<'py>, roi: Option<&Bound<'py, PyAny>>) -> R<Bound<'py, PyAny>> {
        let roi = slices(roi)?;
        Ok(array_to_py(py, self.inner.jacobian_determinant(roi.as_deref())?))
    }
    #[pyo3(signature = (roi=None))]
    fn folding_fraction(&self, roi: Option<&Bound<'_, PyAny>>) -> R<f64> {
        let roi = slices(roi)?;
        Ok(self.inner.folding_fraction(roi.as_deref())?)
    }
    #[getter]
    fn max_magnitude(&self) -> R<f64> {
        Ok(self.inner.max_magnitude()?)
    }
    #[getter]
    fn control_points<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        Ok(array_to_py(py, self.inner.control_points()?))
    }
    #[getter]
    fn cp_grid_id(&self) -> R<String> {
        Ok(self.inner.cp_grid_id()?)
    }
    #[getter]
    fn cp_grid(&self) -> R<Grid> {
        Ok(Grid::wrap(self.inner.cp_grid()?.clone()))
    }
    #[getter]
    fn order(&self) -> R<i64> {
        Ok(self.inner.order()?)
    }
    fn to_displacement_field<'py>(&self, py: Python<'py>, grid: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
        let grid = crate::geometry::grid_arg(grid)?;
        Ok(array_to_py(py, self.inner.to_displacement_field(&grid)?))
    }
    #[getter]
    fn component_ids<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyTuple>> {
        Ok(PyTuple::new(py, self.inner.component_ids()?)?)
    }
    fn components(&self) -> R<Vec<TransformHandle>> {
        Ok(self.inner.components()?.into_iter().map(TransformHandle::wrap).collect())
    }
    fn check_chain(&self) -> Vec<String> {
        self.inner.check_chain()
    }
    fn summary<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        Ok(json_to_py(py, &self.inner.summary()?)?)
    }
    fn __repr__(&self) -> String {
        self.inner.repr()
    }
    #[staticmethod]
    fn inverse_of(inner: &Bound<'_, TransformHandle>) -> TransformHandle {
        TransformHandle::wrap(EngineTransform::inverse_of((*inner.get().inner).clone()))
    }
    #[staticmethod]
    fn can_invert(inner: &Bound<'_, TransformHandle>) -> R<bool> {
        Ok(medh5::transforms::model::can_invert(&inner.get().inner)?)
    }
    #[staticmethod]
    fn chain(steps: Vec<Bound<'_, TransformHandle>>) -> R<TransformHandle> {
        let steps = steps.iter().map(|s| (*s.get().inner).clone()).collect();
        Ok(TransformHandle::wrap(EngineTransform::chain(steps)?))
    }
}

// -- SamplingIndex -----------------------------------------------------------------------------

#[pyclass(module = "medh5._core", name = "SamplingIndex", frozen)]
pub struct IndexHandle {
    pub inner: Arc<EngineIndex>,
}

#[pymethods]
impl IndexHandle {
    #[getter]
    fn ann_id(&self) -> &str {
        &self.inner.ann_id
    }
    /// The `index/<ann_id>` group, as 1.x exposed it.
    #[getter]
    fn group(&self) -> crate::nodes::Group {
        crate::nodes::Group::wrap(self.inner.group.clone())
    }
    #[getter]
    fn class_ids<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyTuple>> {
        Ok(PyTuple::new(py, self.inner.class_ids()?)?)
    }
    fn has_class(&self, class_id: i64) -> R<bool> {
        Ok(self.inner.has_class(class_id)?)
    }
    #[getter]
    fn source_digest(&self) -> R<Option<String>> {
        Ok(self.inner.source_digest()?)
    }
    #[getter]
    fn max_coords(&self) -> R<i64> {
        Ok(self.inner.max_coords()?)
    }
    #[getter]
    fn voxel_counts<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (k, v) in self.inner.voxel_counts()? {
            out.set_item(k, v)?;
        }
        Ok(out)
    }
    fn bbox<'py>(&self, py: Python<'py>, class_id: i64) -> R<Bound<'py, PyAny>> {
        Ok(match self.inner.bbox(class_id)? {
            Some(b) => array_to_py(py, b.into_dyn()),
            None => py.None().into_bound(py),
        })
    }
    fn coords<'py>(&self, py: Python<'py>, class_id: i64) -> R<Bound<'py, PyAny>> {
        Ok(array_to_py(py, self.inner.coords(class_id)?.into_dyn()))
    }
    /// Draw `n` foreground voxel coordinates of a class; `rng` seeds the draw
    /// (an int, ints, or a `numpy.random.Generator`; fresh entropy when `None`).
    #[pyo3(signature = (class_id, n=1, rng=None))]
    fn sample_foreground<'py>(
        &self,
        py: Python<'py>,
        class_id: i64,
        n: usize,
        rng: Option<Bound<'py, PyAny>>,
    ) -> R<Bound<'py, PyAny>> {
        let mut rng = crate::rng::rng_arg(rng.as_ref())?;
        Ok(array_to_py(py, self.inner.sample_foreground(class_id, n, &mut rng)?.into_dyn()))
    }
    #[pyo3(signature = (mode="inverse_frequency"))]
    fn class_weights<'py>(&self, py: Python<'py>, mode: &str) -> R<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (k, v) in self.inner.class_weights(mode)? {
            out.set_item(k, v)?;
        }
        Ok(out)
    }
    fn is_current(&self, annotation_digest: &str) -> R<bool> {
        Ok(self.inner.is_current(annotation_digest)?)
    }
    #[getter]
    fn has_occupancy(&self) -> bool {
        self.inner.has_occupancy()
    }
    fn occupancy_plane<'py>(&self, py: Python<'py>, position: usize) -> R<Bound<'py, PyAny>> {
        Ok(match self.inner.occupancy_plane(position)? {
            Some(p) => array_to_py(py, p),
            None => py.None().into_bound(py),
        })
    }
    fn summary<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        Ok(json_to_py(py, &self.inner.summary()?)?)
    }
    fn __repr__(&self) -> String {
        format!("SamplingIndex({})", medh5::json::repr_str(&self.inner.ann_id))
    }
}

// -- Sample ---------------------------------------------------------------------------------------

/// One open sample root.  `close()` releases the file handle.
#[pyclass(module = "medh5._core", name = "SampleHandle", frozen)]
pub struct SampleHandle {
    inner: Mutex<Option<Arc<EngineSample>>>,
    path: Option<String>,
    /// Inherited across `fork`: never closed by this process.
    abandoned: std::sync::atomic::AtomicBool,
}

impl SampleHandle {
    pub fn wrap(sample: EngineSample) -> Self {
        let path = sample.path.as_ref().map(|p| p.to_string_lossy().into_owned());
        SampleHandle {
            inner: Mutex::new(Some(Arc::new(sample))),
            path,
            abandoned: std::sync::atomic::AtomicBool::new(false),
        }
    }

    pub fn sample(&self) -> R<Arc<EngineSample>> {
        Ok(self.inner.lock().unwrap().clone().ok_or_else(closed)?)
    }
}

/// The engine sample of a `SampleHandle`, or of a `Sample` holding one.
pub fn sample_arg(obj: &Bound<'_, PyAny>) -> R<Arc<EngineSample>> {
    if let Ok(handle) = obj.cast::<SampleHandle>() {
        return handle.get().sample();
    }
    let inner = obj.getattr("_handle")?;
    inner.cast::<SampleHandle>().map_err(PyErr::from)?.get().sample()
}

#[pymethods]
impl SampleHandle {
    #[getter]
    fn path(&self) -> Option<String> {
        self.path.clone()
    }
    /// The sample's root group (read-only use).
    #[getter]
    fn root(&self) -> R<crate::nodes::Group> {
        Ok(crate::nodes::Group::wrap(self.sample()?.root.clone()))
    }
    /// Close the file and everything read from it, as 1.x's sample did: an
    /// image, annotation or group still held becomes invalid rather than
    /// keeping the file open and locked.  A collection member's file is its
    /// collection's, and stays open.
    fn close(&self) {
        if let Some(sample) = self.inner.lock().unwrap().take() {
            if !self.abandoned.load(std::sync::atomic::Ordering::Acquire) {
                let _ = sample.close_file();
            }
            drop(sample);
        }
    }
    /// Make sure this process never closes the file: for a process that
    /// inherited the handle across `fork`, where closing would call into HDF5
    /// on descriptors that belong to the parent.  One reference to the engine
    /// sample is leaked, so neither `close()` nor collection releases it.
    /// Returns whether there was an open handle to pin.
    fn abandon(&self) -> bool {
        self.abandoned.store(true, std::sync::atomic::Ordering::Release);
        // `try_lock`: a lock another of the parent's threads held at the fork
        // is held forever here; the caller keeps the object alive regardless.
        match self.inner.try_lock() {
            Ok(slot) => match slot.as_ref() {
                Some(sample) => {
                    std::mem::forget(Arc::clone(sample));
                    true
                }
                None => false,
            },
            Err(_) => false,
        }
    }
    #[getter]
    fn is_open(&self) -> bool {
        self.inner.lock().unwrap().is_some()
    }
    fn repr(&self) -> R<String> {
        Ok(self.sample()?.repr()?)
    }
    fn document(&self) -> R<crate::document::SampleDocument> {
        Ok(crate::document::SampleDocument::owned(self.sample()?.document()?.clone()))
    }
    #[getter]
    fn version(&self) -> R<String> {
        Ok(self.sample()?.version()?)
    }
    #[getter]
    fn kind(&self) -> R<String> {
        Ok(self.sample()?.kind()?)
    }
    #[getter]
    fn profiles(&self) -> R<Vec<String>> {
        Ok(self.sample()?.profiles()?.into_iter().collect())
    }
    #[getter]
    fn content_id(&self) -> R<Option<String>> {
        Ok(self.sample()?.content_id()?)
    }
    /// `full` or `projection` (a higher minor, read as what this engine knows).
    #[getter]
    fn support(&self) -> R<&'static str> {
        Ok(match self.sample()?.support()? {
            medh5::version::Support::Full => "full",
            medh5::version::Support::Projection => "projection",
            medh5::version::Support::Unsupported => "unsupported",
        })
    }
    /// The clinical profile's records, when the sample declares it (1.1).
    fn clinical(&self, py: Python<'_>) -> R<Option<crate::clinical::ClinicalHandle>> {
        let sample = self.sample()?;
        let reader = sample.clone();
        let found = py.detach(move || reader.clinical().map(|c| c.cloned()))?;
        Ok(found.map(|inner| crate::clinical::ClinicalHandle { inner, _sample: sample }))
    }
    /// One clinical document's text, read on its own: the document table's
    /// offsets and that document's bytes --- not the events, the links or any
    /// other document's text (1.1 §6).
    fn document_text(&self, py: Python<'_>, document_id: &str) -> R<String> {
        let sample = self.sample()?;
        let id = document_id.to_string();
        Ok(py.detach(move || -> medh5::Result<String> {
            match sample.documents()? {
                Some(documents) => documents.text(&id),
                None => Err(medh5::Error::Key(format!(
                    "{}: the sample does not declare the clinical profile",
                    medh5::json::repr_str(&id)
                ))),
            }
        })?)
    }
    /// `{grid_id: Grid}`, with §3.7's implicit timepoint resolved.
    fn grids<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyDict>> {
        let sample = self.sample()?;
        let out = PyDict::new(py);
        for (k, g) in sample.grids()?.iter() {
            out.set_item(k, Grid::wrap(g.clone()))?;
        }
        Ok(out)
    }
    fn reference_grid(&self) -> R<Grid> {
        Ok(Grid::wrap(self.sample()?.reference_grid()?))
    }
    fn image_ids(&self) -> R<Vec<String>> {
        Ok(self.sample()?.images()?.keys().cloned().collect())
    }
    fn image(&self, image_id: &str) -> R<ImageHandle> {
        Ok(ImageHandle { inner: Arc::new(self.sample()?.image(image_id)?.clone()) })
    }
    fn annotation_ids(&self) -> R<Vec<String>> {
        Ok(self.sample()?.annotations()?.keys().cloned().collect())
    }
    fn annotation(&self, ann_id: &str) -> R<AnnotationHandle> {
        Ok(AnnotationHandle::wrap(self.sample()?.annotation(ann_id)?.clone()))
    }
    #[pyo3(signature = (ann_id, roi=None))]
    fn ignore_region<'py>(
        &self,
        py: Python<'py>,
        ann_id: &str,
        roi: Option<&Bound<'py, PyAny>>,
    ) -> R<Bound<'py, PyAny>> {
        let roi = slices(roi)?;
        Ok(array_to_py(py, self.sample()?.ignore_region(ann_id, roi.as_deref())?))
    }
    #[pyo3(signature = (image_id, roi=None))]
    fn valid_region<'py>(
        &self,
        py: Python<'py>,
        image_id: &str,
        roi: Option<&Bound<'py, PyAny>>,
    ) -> R<Bound<'py, PyAny>> {
        let roi = slices(roi)?;
        Ok(array_to_py(py, self.sample()?.valid_region(image_id, roi.as_deref())?))
    }
    fn index_ids(&self) -> R<Vec<String>> {
        Ok(self.sample()?.index()?.keys().cloned().collect())
    }
    fn index(&self, ann_id: &str) -> R<IndexHandle> {
        let sample = self.sample()?;
        let found = sample
            .index()?
            .get(ann_id)
            .ok_or_else(|| medh5::Error::Key(format!("no index entry {}", medh5::json::repr_str(ann_id))))?;
        Ok(IndexHandle { inner: Arc::new(EngineIndex::new(&found.ann_id, found.group.clone())) })
    }
    fn fresh_indices(&self) -> R<Vec<String>> {
        Ok(self.sample()?.fresh_indices()?.iter().cloned().collect())
    }
    fn transform_ids(&self) -> R<Vec<String>> {
        Ok(self.sample()?.transforms()?.keys().cloned().collect())
    }
    fn transform(&self, transform_id: &str) -> R<TransformHandle> {
        Ok(TransformHandle::wrap(self.sample()?.transform(transform_id)?.clone()))
    }
    fn transform_between(&self, source: &str, target: &str) -> R<Option<TransformHandle>> {
        Ok(self.sample()?.transform_between(source, target)?.map(TransformHandle::wrap))
    }
    fn frames_for(&self, key: &str) -> R<Vec<String>> {
        Ok(self.sample()?.frames_for(key)?)
    }
    fn resolve_frames(&self, from_frame: &str, to_frame: &str) -> R<Option<TransformHandle>> {
        Ok(self.sample()?.resolve_frames(from_frame, to_frame)?.map(TransformHandle::wrap))
    }
    fn images_at(&self, timepoint: &str) -> R<Vec<String>> {
        Ok(self.sample()?.images_at(timepoint)?)
    }
    fn annotations_at(&self, timepoint: &str) -> R<Vec<String>> {
        Ok(self.sample()?.annotations_at(timepoint)?)
    }
    fn attr_name_map<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (k, v) in self.sample()?.attr_name_map()? {
            out.set_item(k, PyTuple::new(py, v)?)?;
        }
        Ok(out)
    }
    #[pyo3(signature = (partial=None))]
    fn verify<'py>(&self, py: Python<'py>, partial: Option<Vec<String>>) -> R<Bound<'py, PyAny>> {
        let sample = self.sample()?;
        let result = py.detach(move || sample.verify(partial.as_deref()))?;
        Ok(crate::integrity::verify_result_to_py(py, &result)?)
    }
    fn compute_content_id(&self, py: Python<'_>) -> R<String> {
        let sample = self.sample()?;
        Ok(py.detach(move || sample.compute_content_id())?)
    }
    fn summary<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        Ok(json_to_py(py, &self.sample()?.summary()?)?)
    }
    #[pyo3(signature = (class_key=None, *, measure=true))]
    fn tracks(&self, class_key: Option<&Bound<'_, PyAny>>, measure: bool) -> R<crate::tracking::Tracking> {
        let key = match class_key {
            Some(k) if !k.is_none() => Some(crate::convert::class_key(k)?),
            _ => None,
        };
        let sample = self.sample()?;
        Ok(crate::tracking::Tracking::wrap(medh5::curation::tracking::build_tracks(&sample, key.as_ref(), measure)?))
    }
}

/// Open a `.medh5` sample read-only.
#[pyfunction]
pub fn open_sample(py: Python<'_>, path: std::path::PathBuf) -> R<SampleHandle> {
    let sample = py.detach(move || medh5::sample::open_sample(&path))?;
    Ok(SampleHandle::wrap(sample))
}

#[pyfunction]
fn annotation_id(reference: &str) -> String {
    medh5::sample::annotation_id(reference).to_string()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ImageHandle>()?;
    m.add_class::<AnnotationHandle>()?;
    m.add_class::<TransformHandle>()?;
    m.add_class::<IndexHandle>()?;
    m.add_class::<SampleHandle>()?;
    m.add_function(wrap_pyfunction!(open_sample, m)?)?;
    m.add_function(wrap_pyfunction!(annotation_id, m)?)?;
    let py = m.py();
    m.add("FORMAT_VERSION", medh5::FORMAT_VERSION)?;
    m.add("PROFILES", PyTuple::new(py, medh5::sample::PROFILES)?)?;
    m.add("ROOT_DIGEST_ATTRS", PyTuple::new(py, medh5::sample::ROOT_DIGEST_ATTRS)?)?;
    m.add("SPEC_IMAGE_ATTRS", PyTuple::new(py, medh5::sample::SPEC_IMAGE_ATTRS)?)?;
    m.add("VALUE_TYPES", PyTuple::new(py, medh5::sample::VALUE_TYPES)?)?;
    let _ = PySlice::full(py);
    Ok(())
}
