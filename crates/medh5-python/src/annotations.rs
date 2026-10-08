//! `medh5.annotations`: payloads, the encoders, encoding selection and
//! transcoding (spec §6--§9).
//!
//! The encoders are exported for third-party converters: they validate
//! exactly as the writer does, because the writer calls the same engine
//! functions.

use std::collections::{BTreeMap, BTreeSet};

use ndarray::ArrayD;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFrozenSet, PyList, PyString, PyTuple};
use serde_json::Value;

use medh5::annotations::encode as enc;
use medh5::annotations::encode_geometric as geo;
use medh5::annotations::payload::{Masks, Payload, PayloadData};
use medh5::annotations::select;
use medh5::array::{DType, NdArray};

use crate::convert::{
    array_to_py, attr_to_py, bool_array, dtype_arg, dtype_to_py, f64_array, f64_vec, i64_array, i64_vec, json_to_py,
    nd_to_py, opt_strings, py_to_attr, py_to_json, py_to_nd, untyped_to_nd,
};
use crate::errors::R;

// -- argument helpers -------------------------------------------------------------------

fn given<'py>(obj: Option<&Bound<'py, PyAny>>) -> Option<Bound<'py, PyAny>> {
    obj.filter(|o| !o.is_none()).cloned()
}

/// `int(obj)`, the conversion 1.x applied to every class id.
pub fn int_of(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(i);
    }
    obj.py().import("builtins")?.getattr("int")?.call1((obj,))?.extract::<i64>()
}

pub fn ints_of(obj: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    let items = if obj.hasattr("tolist")? { obj.call_method0("tolist")? } else { obj.clone() };
    if items.is_instance_of::<PyString>() {
        return Err(PyTypeError::new_err("class ids are integers, not a string"));
    }
    items.try_iter()?.map(|i| int_of(&i?)).collect()
}

fn u64s(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u64>> {
    let items = if obj.hasattr("tolist")? { obj.call_method0("tolist")? } else { obj.clone() };
    items.extract::<Vec<u64>>()
}

fn items_of<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)>> {
    let mut out = Vec::new();
    for item in obj.call_method0("items")?.try_iter()? {
        out.push(item?.extract()?);
    }
    Ok(out)
}

/// `{class_id: mask}` as boolean arrays (any dtype is coerced, as 1.x did).
pub fn masks_arg(obj: &Bound<'_, PyAny>) -> PyResult<Masks> {
    let mut out = Masks::new();
    for (k, v) in items_of(obj)? {
        out.insert(int_of(&k)?, bool_array(&v)?);
    }
    Ok(out)
}

pub fn masks_to_py<'py>(py: Python<'py>, masks: Masks) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    for (k, v) in masks {
        out.set_item(k, array_to_py(py, v))?;
    }
    Ok(out)
}

fn shape_arg(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<usize>>> {
    match given(obj) {
        None => Ok(None),
        Some(o) => Ok(Some(i64_vec(&o)?.into_iter().map(|v| v.max(0) as usize).collect())),
    }
}

fn colouring_arg(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<BTreeMap<i64, usize>>> {
    let Some(o) = given(obj) else { return Ok(None) };
    let mut out = BTreeMap::new();
    for (k, v) in items_of(&o)? {
        out.insert(int_of(&k)?, v.extract::<usize>()?);
    }
    Ok(Some(out))
}

/// The keyword options `encode_masks` and `transcode_payload` forward.
fn encode_options(kwargs: Option<&Bound<'_, PyDict>>) -> R<enc::EncodeOptions> {
    let mut options = enc::EncodeOptions::default();
    let Some(kw) = kwargs else { return Ok(options) };
    for (k, v) in kw.iter() {
        let name: String = k.extract()?;
        if v.is_none() {
            continue;
        }
        match name.as_str() {
            "ignore" => options.ignore = Some(bool_array(&v)?),
            "ignore_id" => options.ignore_id = Some(int_of(&v)?),
            "colouring" => options.colouring = colouring_arg(Some(&v))?,
            "dtype" => options.probmap_dtype = Some(dtype_arg(&v)?),
            "normalized" => options.normalized = v.is_truthy()?,
            "threshold" => options.threshold = Some(v.extract()?),
            "start_id" => options.start_id = Some(v.extract()?),
            "store_masks" => options.store_masks = Some(v.is_truthy()?),
            other => {
                return Err(PyTypeError::new_err(format!(
                    "unexpected keyword argument {}",
                    medh5::json::repr_str(other)
                ))
                .into())
            }
        }
    }
    Ok(options)
}

// -- AnnotationPayload ------------------------------------------------------------------------

/// Datasets and kind-specific attributes for one encoded annotation.
#[pyclass(module = "medh5.annotations.base", name = "AnnotationPayload", skip_from_py_object)]
#[derive(Clone)]
pub struct AnnotationPayload {
    pub inner: Payload,
}

impl AnnotationPayload {
    pub fn wrap(inner: Payload) -> Self {
        AnnotationPayload { inner }
    }
}

fn data_to_py<'py>(py: Python<'py>, data: &PayloadData) -> PyResult<Bound<'py, PyAny>> {
    Ok(match data {
        PayloadData::Array(a) => nd_to_py(py, a.clone()),
        PayloadData::Strings(values) => {
            let list = PyList::new(py, values)?;
            crate::convert::numpy(py)?.call_method1("array", (list, "O"))?
        }
    })
}

fn data_arg(obj: &Bound<'_, PyAny>) -> PyResult<PayloadData> {
    let np = crate::convert::numpy(obj.py())?;
    let array = np.call_method1("asarray", (obj,))?;
    let kind: String = array.getattr("dtype")?.getattr("kind")?.extract()?;
    if kind == "O" || kind == "U" || kind == "S" {
        let mut values = Vec::new();
        for item in array.call_method0("tolist")?.try_iter()? {
            let item = item?;
            values.push(match item.extract::<Vec<u8>>() {
                Ok(bytes) if item.is_instance_of::<pyo3::types::PyBytes>() => {
                    String::from_utf8_lossy(&bytes).into_owned()
                }
                _ => item.str()?.to_string(),
            });
        }
        return Ok(PayloadData::Strings(values));
    }
    Ok(PayloadData::Array(py_to_nd(&array)?))
}

/// A payload argument: this class, or anything with its attributes.
pub fn payload_arg(obj: &Bound<'_, PyAny>) -> R<Payload> {
    if let Ok(p) = obj.cast::<AnnotationPayload>() {
        return Ok(p.borrow().inner.clone());
    }
    let mut p = Payload::new(&obj.getattr("kind")?.extract::<String>()?);
    if let Ok(datasets) = obj.getattr("datasets") {
        for (k, v) in items_of(&datasets)? {
            p.datasets.insert(k.extract()?, data_arg(&v)?);
        }
    }
    if let Ok(attrs) = obj.getattr("attrs") {
        for (k, v) in items_of(&attrs)? {
            p.attrs.push((k.extract()?, py_to_attr(&v)?));
        }
    }
    if let Ok(s) = obj.getattr("stacked_axes") {
        p.stacked_axes = s.extract()?;
    }
    if let Ok(c) = obj.getattr("class_ids") {
        p.class_ids = ints_of(&c)?;
    }
    Ok(p)
}

impl AnnotationPayload {
    /// The fields of the 1.x dataclass, in order.
    const FIELDS: &'static [&'static str] = &["kind", "datasets", "attrs", "stacked_axes", "class_ids"];
}

#[pymethods]
impl AnnotationPayload {
    #[classattr]
    fn __dataclass_fields__(py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
        crate::geometry::dataclass_fields(py, Self::FIELDS)
    }

    #[classattr]
    fn __match_args__(py: Python<'_>) -> PyResult<Py<pyo3::types::PyTuple>> {
        crate::values::match_args(py, Self::FIELDS)
    }

    /// `copy.replace(obj, **changes)`.
    #[pyo3(signature = (**changes))]
    fn __replace__(slf: &Bound<'_, Self>, changes: Option<&Bound<'_, pyo3::types::PyDict>>) -> PyResult<Py<PyAny>> {
        crate::values::dataclass_replace(slf.as_any(), changes)
    }

    #[new]
    #[pyo3(signature = (kind, datasets=None, attrs=None, stacked_axes=0, class_ids=None))]
    fn new(
        kind: &str,
        datasets: Option<&Bound<'_, PyAny>>,
        attrs: Option<&Bound<'_, PyAny>>,
        stacked_axes: usize,
        class_ids: Option<&Bound<'_, PyAny>>,
    ) -> R<Self> {
        let mut p = Payload::new(kind);
        if let Some(d) = given(datasets) {
            for (k, v) in items_of(&d)? {
                p.datasets.insert(k.extract()?, data_arg(&v)?);
            }
        }
        if let Some(a) = given(attrs) {
            for (k, v) in items_of(&a)? {
                p.attrs.push((k.extract()?, py_to_attr(&v)?));
            }
        }
        p.stacked_axes = stacked_axes;
        if let Some(c) = given(class_ids) {
            p.class_ids = ints_of(&c)?;
        }
        Ok(AnnotationPayload { inner: p })
    }
    #[getter]
    fn kind(&self) -> &str {
        &self.inner.kind
    }
    #[getter]
    fn datasets<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (k, v) in &self.inner.datasets {
            out.set_item(k, data_to_py(py, v)?)?;
        }
        Ok(out)
    }
    #[getter]
    fn attrs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (k, v) in &self.inner.attrs {
            out.set_item(k, attr_to_py(py, v)?)?;
        }
        Ok(out)
    }
    #[getter]
    fn stacked_axes(&self) -> usize {
        self.inner.stacked_axes
    }
    #[getter]
    fn class_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.class_ids)
    }
    /// The `data` dataset.
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        Ok(nd_to_py(py, self.inner.data()?.clone()))
    }
    #[getter]
    fn nbytes(&self) -> usize {
        self.inner.nbytes()
    }
    fn describe<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &self.inner.describe())
    }
    fn __repr__(&self) -> String {
        let names: Vec<&str> = self.inner.datasets.keys().map(String::as_str).collect();
        format!(
            "AnnotationPayload(kind={}, datasets={}, class_ids={})",
            medh5::json::repr_str(&self.inner.kind),
            medh5::json::repr_list(&names),
            medh5::json::repr_int_tuple(&self.inner.class_ids)
        )
    }
}

// -- AnnotationHeader -------------------------------------------------------------------------

/// The fixed attribute header every annotation carries (spec §6.2).
#[pyclass(module = "medh5.annotations.base", name = "AnnotationHeader", skip_from_py_object, frozen)]
#[derive(Clone)]
pub struct AnnotationHeader {
    pub inner: medh5::annotations::header::AnnotationHeader,
}

impl AnnotationHeader {
    /// The fields of the 1.x dataclass, in order.
    const FIELDS: &'static [&'static str] = &[
        "kind",
        "task",
        "grid",
        "timepoints",
        "space",
        "frame_uid",
        "class_ids",
        "annotated_class_ids",
        "closure",
        "ignore_id",
        "ignore_mask",
        "prov",
        "quality",
        "derived_from",
        "extra",
    ];
}

#[pymethods]
impl AnnotationHeader {
    /// The header an annotation group carries (`sample.root["annotations/x"]`).
    #[classmethod]
    fn read(_cls: &Bound<'_, pyo3::types::PyType>, group: &Bound<'_, PyAny>) -> R<Self> {
        let group = crate::integrity::group_of(group)?;
        Ok(AnnotationHeader { inner: medh5::annotations::header::AnnotationHeader::read(&group)? })
    }

    #[classattr]
    fn __dataclass_fields__(py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
        crate::geometry::dataclass_fields(py, Self::FIELDS)
    }

    #[classattr]
    fn __match_args__(py: Python<'_>) -> PyResult<Py<pyo3::types::PyTuple>> {
        crate::values::match_args(py, Self::FIELDS)
    }

    /// `copy.replace(obj, **changes)`.
    #[pyo3(signature = (**changes))]
    fn __replace__(slf: &Bound<'_, Self>, changes: Option<&Bound<'_, pyo3::types::PyDict>>) -> PyResult<Py<PyAny>> {
        crate::values::dataclass_replace(slf.as_any(), changes)
    }

    #[new]
    #[pyo3(signature = (kind, task, grid=None, timepoints=None, space=None, frame_uid=None, class_ids=None,
        annotated_class_ids=None, closure="explicit", ignore_id=medh5::labels::IGNORE_ID, ignore_mask=None,
        prov=None, quality=None, derived_from=None, extra=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        kind: &str,
        task: &str,
        grid: Option<String>,
        timepoints: Option<&Bound<'_, PyAny>>,
        space: Option<String>,
        frame_uid: Option<String>,
        class_ids: Option<&Bound<'_, PyAny>>,
        annotated_class_ids: Option<&Bound<'_, PyAny>>,
        closure: &str,
        ignore_id: i64,
        ignore_mask: Option<String>,
        prov: Option<String>,
        quality: Option<String>,
        derived_from: Option<&Bound<'_, PyAny>>,
        extra: Option<&Bound<'_, PyAny>>,
    ) -> R<Self> {
        let mut h = medh5::annotations::header::AnnotationHeader::new(kind, task);
        h.grid = grid;
        h.timepoints = opt_strings(timepoints)?;
        h.space = space;
        h.frame_uid = frame_uid;
        h.class_ids = given(class_ids).map(|c| ints_of(&c)).transpose()?.unwrap_or_default();
        h.annotated_class_ids = given(annotated_class_ids).map(|c| ints_of(&c)).transpose()?.unwrap_or_default();
        h.closure = closure.to_string();
        h.ignore_id = ignore_id;
        h.ignore_mask = ignore_mask;
        h.prov = prov;
        h.quality = quality;
        h.derived_from = opt_strings(derived_from)?.unwrap_or_default();
        if let Some(e) = given(extra) {
            for (k, v) in items_of(&e)? {
                h.extra.push((k.extract()?, py_to_attr(&v)?));
            }
        }
        h.check()?;
        Ok(AnnotationHeader { inner: h })
    }
    #[getter]
    fn kind(&self) -> &str {
        &self.inner.kind
    }
    #[getter]
    fn task(&self) -> &str {
        &self.inner.task
    }
    #[getter]
    fn grid(&self) -> Option<&str> {
        self.inner.grid.as_deref()
    }
    #[getter]
    fn timepoints<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyTuple>>> {
        self.inner.timepoints.as_ref().map(|t| PyTuple::new(py, t)).transpose()
    }
    #[getter]
    fn space(&self) -> Option<&str> {
        self.inner.space.as_deref()
    }
    #[getter]
    fn frame_uid(&self) -> Option<&str> {
        self.inner.frame_uid.as_deref()
    }
    #[getter]
    fn class_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.class_ids)
    }
    #[getter]
    fn annotated_class_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.annotated_class_ids)
    }
    #[getter]
    fn closure(&self) -> &str {
        &self.inner.closure
    }
    #[getter]
    fn ignore_id(&self) -> i64 {
        self.inner.ignore_id
    }
    #[getter]
    fn ignore_mask(&self) -> Option<&str> {
        self.inner.ignore_mask.as_deref()
    }
    #[getter]
    fn prov(&self) -> Option<&str> {
        self.inner.prov.as_deref()
    }
    #[getter]
    fn quality(&self) -> Option<&str> {
        self.inner.quality.as_deref()
    }
    #[getter]
    fn derived_from<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.derived_from)
    }
    #[getter]
    fn extra<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (k, v) in &self.inner.extra {
            out.set_item(k, attr_to_py(py, v)?)?;
        }
        Ok(out)
    }
    /// The attributes this header writes.
    fn attrs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (k, v) in self.inner.attrs() {
            out.set_item(k, attr_to_py(py, &v)?)?;
        }
        Ok(out)
    }
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other.cast::<AnnotationHeader>().map(|o| o.get().inner == self.inner).unwrap_or(false)
    }
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let names = [
            "kind",
            "task",
            "grid",
            "timepoints",
            "space",
            "frame_uid",
            "class_ids",
            "annotated_class_ids",
            "closure",
            "ignore_id",
            "ignore_mask",
            "prov",
            "quality",
            "derived_from",
            "extra",
        ];
        let fields = names.iter().map(|n| Ok((*n, slf.getattr(*n)?))).collect::<PyResult<Vec<_>>>()?;
        crate::values::dataclass_repr("AnnotationHeader", &fields)
    }
}

// -- OverlapStats and CostModel ---------------------------------------------------------------

/// Measured properties of a set of class masks, and what they imply (§7.6).
#[pyclass(module = "medh5.annotations.voxel", name = "OverlapStats", skip_from_py_object, frozen)]
#[derive(Clone)]
pub struct OverlapStats {
    pub inner: select::OverlapStats,
}

impl OverlapStats {
    pub fn wrap(inner: select::OverlapStats) -> Self {
        OverlapStats { inner }
    }
}

fn int_dict<'py, K: IntoPyObject<'py> + Copy, V: IntoPyObject<'py> + Copy>(
    py: Python<'py>,
    items: impl IntoIterator<Item = (K, V)>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    for (k, v) in items {
        out.set_item(k, v)?;
    }
    Ok(out)
}

impl OverlapStats {
    /// The fields of the 1.x dataclass, in order.
    const FIELDS: &'static [&'static str] =
        &["class_ids", "spatial_shape", "counts", "edges", "colouring", "localized", "n_labelled_voxels"];
}

#[pymethods]
impl OverlapStats {
    #[classattr]
    fn __dataclass_fields__(py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
        crate::geometry::dataclass_fields(py, Self::FIELDS)
    }

    #[classattr]
    fn __match_args__(py: Python<'_>) -> PyResult<Py<pyo3::types::PyTuple>> {
        crate::values::match_args(py, Self::FIELDS)
    }

    #[getter]
    fn class_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.class_ids)
    }
    #[getter]
    fn spatial_shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.spatial_shape)
    }
    #[getter]
    fn counts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        int_dict(py, self.inner.counts.iter().map(|(k, v)| (*k, *v)))
    }
    #[getter]
    fn edges<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyFrozenSet>> {
        PyFrozenSet::new(py, self.inner.edges.iter().copied())
    }
    #[getter]
    fn colouring<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        int_dict(py, self.inner.colouring.iter().map(|(k, v)| (*k, *v)))
    }
    #[getter]
    fn localized<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyFrozenSet>> {
        PyFrozenSet::new(py, self.inner.localized.iter().copied())
    }
    #[getter]
    fn n_labelled_voxels(&self) -> u64 {
        self.inner.n_labelled_voxels
    }
    #[getter]
    fn n_classes(&self) -> usize {
        self.inner.n_classes()
    }
    #[getter]
    fn n_voxels(&self) -> u64 {
        self.inner.n_voxels()
    }
    #[getter]
    fn total_foreground(&self) -> u64 {
        self.inner.total_foreground()
    }
    #[getter]
    fn fill(&self) -> f64 {
        self.inner.fill()
    }
    #[getter]
    fn depth(&self) -> f64 {
        self.inner.depth()
    }
    #[getter]
    fn n_layers(&self) -> usize {
        self.inner.n_layers()
    }
    #[getter]
    fn mean_degree(&self) -> f64 {
        self.inner.mean_degree()
    }
    #[getter]
    fn is_edgeless(&self) -> bool {
        self.inner.is_edgeless()
    }
    #[getter]
    fn n_planes(&self) -> usize {
        self.inner.n_planes()
    }
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &self.inner.summary())
    }
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other.cast::<OverlapStats>().map(|o| o.get().inner == self.inner).unwrap_or(false)
    }
    fn __repr__(&self) -> String {
        format!(
            "OverlapStats(classes={}, voxels={}, fill={}, layers={}, planes={}, edges={})",
            self.inner.n_classes(),
            self.inner.n_voxels(),
            medh5::json::py_float(self.inner.fill()),
            self.inner.n_layers(),
            self.inner.n_planes(),
            self.inner.edges.len()
        )
    }
}

/// Raw (pre-compression) bytes per encoding.
#[pyclass(module = "medh5.annotations.voxel", name = "CostModel", skip_from_py_object, frozen)]
pub struct CostModel {
    pub inner: select::CostModel,
}

impl CostModel {
    /// The fields of the 1.x dataclass, in order.
    const FIELDS: &'static [&'static str] = &["labelmap", "layers", "bitmask", "instances", "probmap", "detail"];
}

#[pymethods]
impl CostModel {
    #[classattr]
    fn __dataclass_fields__(py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
        crate::geometry::dataclass_fields(py, Self::FIELDS)
    }

    #[classattr]
    fn __match_args__(py: Python<'_>) -> PyResult<Py<pyo3::types::PyTuple>> {
        crate::values::match_args(py, Self::FIELDS)
    }

    #[getter]
    fn labelmap(&self) -> Option<u64> {
        self.inner.labelmap
    }
    #[getter]
    fn layers(&self) -> u64 {
        self.inner.layers
    }
    #[getter]
    fn bitmask(&self) -> u64 {
        self.inner.bitmask
    }
    #[getter]
    fn instances(&self) -> u64 {
        self.inner.instances
    }
    #[getter]
    fn probmap(&self) -> u64 {
        self.inner.probmap
    }
    #[getter]
    fn detail<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &self.inner.detail)
    }
    /// The cheapest encoding (`labelmap` only when it can hold the masks).
    fn best(&self) -> &'static str {
        self.inner.best()
    }
    fn to_json<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &self.inner.to_json())
    }
    fn __repr__(&self) -> String {
        format!(
            "CostModel(labelmap={}, layers={}, bitmask={}, instances={}, probmap={})",
            self.inner.labelmap.map(|v| v.to_string()).unwrap_or_else(|| "None".into()),
            self.inner.layers,
            self.inner.bitmask,
            self.inner.instances,
            self.inner.probmap
        )
    }
}

// -- selection ------------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (masks, spatial_shape=None))]
fn analyse(masks: &Bound<'_, PyAny>, spatial_shape: Option<&Bound<'_, PyAny>>) -> R<OverlapStats> {
    let masks = masks_arg(masks)?;
    let shape = shape_arg(spatial_shape)?;
    Ok(OverlapStats::wrap(select::analyse(&masks, shape.as_deref())?))
}

#[pyfunction]
fn greedy_colour<'py>(
    py: Python<'py>,
    class_ids: &Bound<'py, PyAny>,
    edges: &Bound<'py, PyAny>,
) -> R<Bound<'py, PyDict>> {
    let ids = ints_of(class_ids)?;
    let mut set = BTreeSet::new();
    for e in edges.try_iter()? {
        let (a, b): (i64, i64) = e?.extract()?;
        set.insert((a, b));
    }
    Ok(int_dict(py, select::greedy_colour(&ids, &set))?)
}

#[pyfunction]
fn layers_from_colouring<'py>(py: Python<'py>, colouring: &Bound<'py, PyAny>) -> R<Bound<'py, PyTuple>> {
    let map = colouring_arg(Some(colouring))?.unwrap_or_default();
    let layers = select::layers_from_colouring(&map);
    Ok(PyTuple::new(py, layers.into_iter().map(|l| PyTuple::new(py, l)).collect::<PyResult<Vec<_>>>()?)?)
}

#[pyfunction]
#[pyo3(signature = (class_ids, *, ignore=false))]
fn label_dtype_size(class_ids: &Bound<'_, PyAny>, ignore: bool) -> R<usize> {
    Ok(select::label_dtype_size(&ints_of(class_ids)?, ignore))
}

#[pyfunction]
#[pyo3(signature = (stats, *, ignore=false))]
fn cost_model(stats: &Bound<'_, OverlapStats>, ignore: bool) -> CostModel {
    CostModel { inner: select::cost_model(&stats.get().inner, ignore) }
}

/// Choose an encoding by measurement: `(kind, stats)`.
#[pyfunction]
#[pyo3(signature = (masks=None, spatial_shape=None, *, stats=None, soft=false, prefer=None, ignore=false))]
fn select_encoding<'py>(
    py: Python<'py>,
    masks: Option<&Bound<'py, PyAny>>,
    spatial_shape: Option<&Bound<'py, PyAny>>,
    stats: Option<&Bound<'py, OverlapStats>>,
    soft: bool,
    prefer: Option<&str>,
    ignore: bool,
) -> R<Bound<'py, PyTuple>> {
    let stats = match (stats, given(masks)) {
        (Some(s), _) => s.get().inner.clone(),
        (None, Some(m)) => {
            let masks = masks_arg(&m)?;
            let shape = shape_arg(spatial_shape)?;
            select::analyse(&masks, shape.as_deref())?
        }
        (None, None) => return Err(medh5::Error::invalid("select_encoding needs masks= or stats=").into()),
    };
    let kind = select::select_encoding(&stats, soft, prefer, ignore);
    Ok(PyTuple::new(py, [PyString::new(py, &kind).into_any(), Bound::new(py, OverlapStats::wrap(stats))?.into_any()])?)
}

// -- voxel encoders ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (masks, spatial_shape=None, *, ignore=None, ignore_id=medh5::labels::IGNORE_ID))]
fn encode_labelmap(
    masks: &Bound<'_, PyAny>,
    spatial_shape: Option<&Bound<'_, PyAny>>,
    ignore: Option<&Bound<'_, PyAny>>,
    ignore_id: i64,
) -> R<AnnotationPayload> {
    let masks = masks_arg(masks)?;
    let shape = shape_arg(spatial_shape)?;
    let ignore = given(ignore).map(|i| bool_array(&i)).transpose()?;
    Ok(AnnotationPayload::wrap(enc::encode_labelmap(&masks, shape.as_deref(), ignore.as_ref(), ignore_id)?))
}

#[pyfunction]
#[pyo3(signature = (masks, spatial_shape=None, *, colouring=None, ignore=None, ignore_id=medh5::labels::IGNORE_ID))]
fn encode_layers(
    masks: &Bound<'_, PyAny>,
    spatial_shape: Option<&Bound<'_, PyAny>>,
    colouring: Option<&Bound<'_, PyAny>>,
    ignore: Option<&Bound<'_, PyAny>>,
    ignore_id: i64,
) -> R<AnnotationPayload> {
    let masks = masks_arg(masks)?;
    let shape = shape_arg(spatial_shape)?;
    let colouring = colouring_arg(colouring)?;
    let ignore = given(ignore).map(|i| bool_array(&i)).transpose()?;
    Ok(AnnotationPayload::wrap(enc::encode_layers(
        &masks,
        shape.as_deref(),
        colouring.as_ref(),
        ignore.as_ref(),
        ignore_id,
    )?))
}

#[pyfunction]
#[pyo3(signature = (masks, spatial_shape=None))]
fn encode_bitmask(masks: &Bound<'_, PyAny>, spatial_shape: Option<&Bound<'_, PyAny>>) -> R<AnnotationPayload> {
    let masks = masks_arg(masks)?;
    let shape = shape_arg(spatial_shape)?;
    Ok(AnnotationPayload::wrap(enc::encode_bitmask(&masks, shape.as_deref())?))
}

#[pyfunction]
fn encode_mask(mask: &Bound<'_, PyAny>) -> R<AnnotationPayload> {
    Ok(AnnotationPayload::wrap(enc::encode_mask(bool_array(mask)?)))
}

fn planes_arg(obj: &Bound<'_, PyAny>) -> PyResult<BTreeMap<i64, ArrayD<f64>>> {
    let mut out = BTreeMap::new();
    for (k, v) in items_of(obj)? {
        out.insert(int_of(&k)?, f64_array(&v)?);
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (probabilities, spatial_shape=None, *, dtype=None, normalized=false, threshold=None))]
fn encode_probmap(
    probabilities: &Bound<'_, PyAny>,
    spatial_shape: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    normalized: bool,
    threshold: Option<f64>,
) -> R<AnnotationPayload> {
    let planes = planes_arg(probabilities)?;
    let shape = shape_arg(spatial_shape)?;
    let dtype = match given(dtype) {
        Some(d) => dtype_arg(&d)?,
        None => DType::F16,
    };
    Ok(AnnotationPayload::wrap(enc::encode_probmap(&planes, shape.as_deref(), dtype, normalized, threshold)?))
}

/// `values >= threshold`, decided in the stored precision (§7.5).
#[pyfunction]
fn contains_at<'py>(py: Python<'py>, values: &Bound<'py, PyAny>, threshold: f64) -> R<Bound<'py, PyAny>> {
    Ok(array_to_py(py, enc::contains_at(&py_to_nd(values)?, threshold)))
}

#[pyfunction]
#[pyo3(signature = (planes, threshold, requested=None))]
fn storage_dtype<'py>(
    py: Python<'py>,
    planes: &Bound<'py, PyAny>,
    threshold: f64,
    requested: Option<&Bound<'py, PyAny>>,
) -> R<Bound<'py, PyAny>> {
    let arrays: Vec<ArrayD<f64>> = if planes.hasattr("values")? && planes.hasattr("items")? {
        planes.call_method0("values")?.try_iter()?.map(|p| f64_array(&p?)).collect::<PyResult<_>>()?
    } else {
        planes.try_iter()?.map(|p| f64_array(&p?)).collect::<PyResult<_>>()?
    };
    let requested = match given(requested) {
        Some(d) => dtype_arg(&d)?,
        None => DType::F16,
    };
    Ok(dtype_to_py(py, enc::storage_dtype(&arrays, threshold, requested))?)
}

/// `uint32` unless an id needs the wider form (§7.4, §8.2).
#[pyfunction]
fn instance_id_dtype<'py>(py: Python<'py>, ids: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
    Ok(dtype_to_py(py, enc::instance_id_dtype(&u64s(ids)?))?)
}

/// An `InstanceInput`-shaped object (attributes, as 1.x took them).
pub fn instance_input(item: &Bound<'_, PyAny>) -> PyResult<enc::InstanceInput> {
    let get = |name: &str| -> Option<Bound<'_, PyAny>> {
        match item.getattr(name) {
            Ok(v) if !v.is_none() => Some(v),
            _ => None,
        }
    };
    let class = get("class_id").ok_or_else(|| PyTypeError::new_err("an instance needs class_id"))?;
    let instance_id = get("instance_id").ok_or_else(|| PyTypeError::new_err("an instance needs instance_id"))?;
    Ok(enc::InstanceInput {
        class_id: int_of(&class)?,
        instance_id: instance_id.extract::<u64>()?,
        mask: get("mask").map(|m| bool_array(&m)).transpose()?,
        bbox: get("box").map(|b| f64_vec(&b)).transpose()?,
        crop: get("crop").map(|c| bool_array(&c)).transpose()?,
        score: get("score").map(|s| s.extract::<f64>()).transpose()?,
    })
}

#[pyfunction]
#[pyo3(signature = (objects, spatial_shape=None, *, store_masks=true, class_ids=None))]
fn encode_instances(
    objects: &Bound<'_, PyAny>,
    spatial_shape: Option<&Bound<'_, PyAny>>,
    store_masks: bool,
    class_ids: Option<&Bound<'_, PyAny>>,
) -> R<AnnotationPayload> {
    let objects = objects.try_iter()?.map(|o| instance_input(&o?)).collect::<PyResult<Vec<_>>>()?;
    let shape = shape_arg(spatial_shape)?;
    let ids = given(class_ids).map(|c| ints_of(&c)).transpose()?;
    Ok(AnnotationPayload::wrap(enc::encode_instances(&objects, shape.as_deref(), store_masks, ids.as_deref())?))
}

/// One object per class mask, ids from `start_id`: the field values of
/// `InstanceInput`s, as `(class_id, instance_id, mask)` triples.
#[pyfunction]
#[pyo3(signature = (masks, *, start_id=1))]
fn instances_from_masks<'py>(py: Python<'py>, masks: &Bound<'py, PyAny>, start_id: u64) -> R<Bound<'py, PyList>> {
    let masks = masks_arg(masks)?;
    let out = PyList::empty(py);
    for i in enc::instances_from_masks(&masks, start_id) {
        let mask = match i.mask {
            Some(m) => array_to_py(py, m),
            None => py.None().into_bound(py),
        };
        out.append((i.class_id, i.instance_id, mask))?;
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (payload, *, spatial_shape=None, threshold=None))]
fn payload_to_masks<'py>(
    py: Python<'py>,
    payload: &Bound<'py, PyAny>,
    spatial_shape: Option<&Bound<'py, PyAny>>,
    threshold: Option<f64>,
) -> R<Bound<'py, PyDict>> {
    let payload = payload_arg(payload)?;
    let shape = shape_arg(spatial_shape)?;
    Ok(masks_to_py(py, enc::payload_to_masks(&payload, shape.as_deref(), threshold)?)?)
}

#[pyfunction]
#[pyo3(signature = (masks, kind, spatial_shape=None, **kwargs))]
fn encode_masks(
    masks: &Bound<'_, PyAny>,
    kind: &str,
    spatial_shape: Option<&Bound<'_, PyAny>>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> R<AnnotationPayload> {
    let masks = masks_arg(masks)?;
    let shape = shape_arg(spatial_shape)?;
    let options = encode_options(kwargs)?;
    Ok(AnnotationPayload::wrap(enc::encode_masks(&masks, kind, shape.as_deref(), &options)?))
}

/// Encode class masks, choosing the encoding by measurement when asked to:
/// `(payload, stats)`.  An ignore region rides in band only under
/// `labelmap`/`layers`; any other choice is refused (E404), because one
/// payload cannot carry the §7.7 sibling mask.
#[pyfunction]
#[pyo3(signature = (masks, spatial_shape=None, *, encoding="auto", ignore=None, **kwargs))]
fn encode_voxels<'py>(
    py: Python<'py>,
    masks: &Bound<'py, PyAny>,
    spatial_shape: Option<&Bound<'py, PyAny>>,
    encoding: &str,
    ignore: Option<&Bound<'py, PyAny>>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> R<Bound<'py, PyTuple>> {
    let masks = masks_arg(masks)?;
    let shape = shape_arg(spatial_shape)?;
    let shape = medh5::annotations::payload::normalize_masks(&masks, shape.as_deref())?;
    let ignore = given(ignore).map(|i| bool_array(&i)).transpose()?;
    let stats = select::analyse(&masks, Some(&shape))?;
    let prefer = if encoding == "auto" { None } else { Some(encoding) };
    let kind = select::select_encoding(&stats, false, prefer, ignore.is_some());
    let mut options = encode_options(kwargs)?;
    if let Some(region) = ignore {
        if !enc::IN_BAND_IGNORE_KINDS.contains(&kind.as_str()) {
            return Err(medh5::Error::coded(
                "E404",
                format!(
                    "encode_voxels: {} cannot hold an ignore region in band; §7.7 puts it in a separate `mask` \
                     annotation, which a single payload cannot carry. Use SampleWriter.add_segmentation, which writes \
                     the sibling mask, or choose 'labelmap' or 'layers'.",
                    medh5::json::repr_str(&kind)
                ),
            )
            .into());
        }
        if options.ignore.is_none() {
            options.ignore = Some(region);
        }
    }
    let payload = enc::encode_masks(&masks, &kind, Some(&shape), &options)?;
    Ok(PyTuple::new(
        py,
        [
            Bound::new(py, AnnotationPayload::wrap(payload))?.into_any(),
            Bound::new(py, OverlapStats::wrap(stats))?.into_any(),
        ],
    )?)
}

/// Re-encode a payload; the same payload comes back when it already has
/// `to_kind`.
#[pyfunction]
#[pyo3(signature = (payload, to_kind, *, spatial_shape=None, threshold=None, drop_identity=false, **kwargs))]
fn transcode_payload<'py>(
    payload: &Bound<'py, PyAny>,
    to_kind: &str,
    spatial_shape: Option<&Bound<'py, PyAny>>,
    threshold: Option<f64>,
    drop_identity: bool,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> R<Bound<'py, PyAny>> {
    let py = payload.py();
    let source = payload_arg(payload)?;
    if source.kind == to_kind {
        enc::check_target(to_kind)?;
        return Ok(payload.clone());
    }
    let shape = shape_arg(spatial_shape)?;
    let options = encode_options(kwargs)?;
    let out = enc::transcode_payload(&source, to_kind, shape.as_deref(), threshold, drop_identity, &options)?;
    Ok(Bound::new(py, AnnotationPayload::wrap(out))?.into_any())
}

/// Convert an open voxel annotation to another encoding (§7.6).
#[pyfunction]
#[pyo3(signature = (annotation, to_kind, *, drop_identity=false))]
fn transcode(annotation: &Bound<'_, PyAny>, to_kind: &str, drop_identity: bool) -> R<AnnotationPayload> {
    let handle = annotation_handle(annotation)?;
    let payload = medh5::sample::writer_annotations::transcode(&handle, to_kind, drop_identity)?;
    Ok(AnnotationPayload::wrap(payload))
}

/// Every class of a voxel annotation as a full mask.
#[pyfunction]
#[pyo3(signature = (annotation, classes=None))]
fn annotation_to_masks<'py>(
    py: Python<'py>,
    annotation: &Bound<'py, PyAny>,
    classes: Option<&Bound<'py, PyAny>>,
) -> R<Bound<'py, PyDict>> {
    let handle = annotation_handle(annotation)?;
    let keys = given(classes).map(|c| crate::convert::class_keys(&c)).transpose()?;
    Ok(masks_to_py(py, medh5::sample::writer_annotations::annotation_to_masks(&handle, keys.as_deref())?)?)
}

/// The engine annotation of an `AnnotationHandle` or a facade holding one.
pub fn annotation_handle(obj: &Bound<'_, PyAny>) -> PyResult<std::sync::Arc<medh5::annotations::Annotation>> {
    if let Ok(h) = obj.cast::<crate::reader::AnnotationHandle>() {
        return Ok(h.get().inner.clone());
    }
    let inner = obj.getattr("_handle")?;
    Ok(inner.cast::<crate::reader::AnnotationHandle>()?.get().inner.clone())
}

#[pyfunction]
fn masks_equal(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> R<bool> {
    Ok(enc::masks_equal(&masks_arg(a)?, &masks_arg(b)?))
}

#[pyfunction]
#[pyo3(signature = (payload, to_kind, *, spatial_shape=None))]
fn check_roundtrip(payload: &Bound<'_, PyAny>, to_kind: &str, spatial_shape: Option<&Bound<'_, PyAny>>) -> R<bool> {
    let payload = payload_arg(payload)?;
    let shape = shape_arg(spatial_shape)?;
    Ok(enc::check_roundtrip(&payload, to_kind, shape.as_deref())?)
}

#[pyfunction]
#[pyo3(signature = (masks, spatial_shape=None))]
fn normalize_masks<'py>(
    py: Python<'py>,
    masks: &Bound<'py, PyAny>,
    spatial_shape: Option<&Bound<'py, PyAny>>,
) -> R<Bound<'py, PyTuple>> {
    let masks = masks_arg(masks)?;
    let shape = shape_arg(spatial_shape)?;
    let found = medh5::annotations::payload::normalize_masks(&masks, shape.as_deref())?;
    Ok(PyTuple::new(py, [masks_to_py(py, masks)?.into_any(), PyTuple::new(py, found)?.into_any()])?)
}

#[pyfunction]
fn check_class_id(class_id: &Bound<'_, PyAny>) -> R<i64> {
    Ok(i64::from(medh5::labels::check_class_id(int_of(class_id)?)?))
}

// -- geometric encoders -----------------------------------------------------------------------

struct Columns {
    class_ids: Vec<i64>,
    instance_ids: Option<Vec<u64>>,
    scores: Option<Vec<f64>>,
    attributes: Option<Vec<serde_json::Map<String, Value>>>,
}

impl Columns {
    fn of(
        class_ids: &Bound<'_, PyAny>,
        instance_ids: Option<&Bound<'_, PyAny>>,
        scores: Option<&Bound<'_, PyAny>>,
        attributes: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Columns> {
        let attributes = match given(attributes) {
            None => None,
            Some(items) => Some(
                items
                    .try_iter()?
                    .map(|i| match py_to_json(&i?)? {
                        Value::Object(m) => Ok(m),
                        _ => Err(PyTypeError::new_err("each entry of attributes is a mapping")),
                    })
                    .collect::<PyResult<Vec<_>>>()?,
            ),
        };
        Ok(Columns {
            class_ids: ints_of(class_ids)?,
            instance_ids: given(instance_ids).map(|i| u64s(&i)).transpose()?,
            scores: given(scores).map(|s| f64_vec(&s)).transpose()?,
            attributes,
        })
    }

    fn view(&self) -> geo::ObjectColumns<'_> {
        geo::ObjectColumns {
            class_ids: &self.class_ids,
            instance_ids: self.instance_ids.as_deref(),
            scores: self.scores.as_deref(),
            attributes: self.attributes.as_deref(),
        }
    }
}

#[pyfunction]
#[pyo3(signature = (boxes, class_ids, *, instance_ids=None, scores=None, attributes=None, slice_index=None))]
fn encode_boxes(
    boxes: &Bound<'_, PyAny>,
    class_ids: &Bound<'_, PyAny>,
    instance_ids: Option<&Bound<'_, PyAny>>,
    scores: Option<&Bound<'_, PyAny>>,
    attributes: Option<&Bound<'_, PyAny>>,
    slice_index: Option<&Bound<'_, PyAny>>,
) -> R<AnnotationPayload> {
    let boxes = f64_array(boxes)?;
    let cols = Columns::of(class_ids, instance_ids, scores, attributes)?;
    let n_boxes = boxes.shape().first().copied().unwrap_or(0);
    let planes = given(slice_index).map(|s| slice_index_arg(&s, n_boxes)).transpose()?;
    Ok(AnnotationPayload::wrap(geo::encode_boxes(&boxes, &cols.view(), planes.as_deref())?))
}

#[pyfunction]
#[pyo3(signature = (centers, sizes, rotations, class_ids, *, instance_ids=None, scores=None, attributes=None))]
fn encode_obb(
    centers: &Bound<'_, PyAny>,
    sizes: &Bound<'_, PyAny>,
    rotations: &Bound<'_, PyAny>,
    class_ids: &Bound<'_, PyAny>,
    instance_ids: Option<&Bound<'_, PyAny>>,
    scores: Option<&Bound<'_, PyAny>>,
    attributes: Option<&Bound<'_, PyAny>>,
) -> R<AnnotationPayload> {
    let cols = Columns::of(class_ids, instance_ids, scores, attributes)?;
    Ok(AnnotationPayload::wrap(geo::encode_obb(
        &f64_array(centers)?,
        &f64_array(sizes)?,
        &f64_array(rotations)?,
        &cols.view(),
    )?))
}

#[pyfunction]
#[pyo3(signature = (points, keypoint_class_ids, class_ids, *, visibility=None, instance_ids=None, scores=None, skeleton=None))]
#[allow(clippy::too_many_arguments)]
fn encode_keypoints(
    points: &Bound<'_, PyAny>,
    keypoint_class_ids: &Bound<'_, PyAny>,
    class_ids: &Bound<'_, PyAny>,
    visibility: Option<&Bound<'_, PyAny>>,
    instance_ids: Option<&Bound<'_, PyAny>>,
    scores: Option<&Bound<'_, PyAny>>,
    skeleton: Option<&str>,
) -> R<AnnotationPayload> {
    let cols = Columns::of(class_ids, instance_ids, scores, None)?;
    let slots = ints_of(keypoint_class_ids)?;
    let visibility = given(visibility).map(|v| i64_array(&v)).transpose()?;
    Ok(AnnotationPayload::wrap(geo::encode_keypoints(
        &f64_array(points)?,
        &slots,
        &cols.view(),
        visibility.as_ref(),
        skeleton,
    )?))
}

#[pyfunction]
#[pyo3(signature = (points, *, class_ids=None, names=None, weights=None, correspondence=None))]
fn encode_points(
    points: &Bound<'_, PyAny>,
    class_ids: Option<&Bound<'_, PyAny>>,
    names: Option<&Bound<'_, PyAny>>,
    weights: Option<&Bound<'_, PyAny>>,
    correspondence: Option<&str>,
) -> R<AnnotationPayload> {
    let ids = given(class_ids).map(|c| ints_of(&c)).transpose()?;
    let names = opt_strings(names)?;
    let weights = given(weights).map(|w| f64_vec(&w)).transpose()?;
    Ok(AnnotationPayload::wrap(geo::encode_points(
        &f64_array(points)?,
        ids.as_deref(),
        names.as_deref(),
        weights.as_deref(),
        correspondence,
    )?))
}

pub fn polygon_arg(item: &Bound<'_, PyAny>) -> R<geo::Polygon> {
    let vertices = f64_array(&item.getattr("vertices")?)?;
    let class_id = int_of(&item.getattr("class_id")?)?;
    let plane = match item.getattr("plane") {
        Ok(p) if !p.is_none() => {
            let v = i64_vec(&p)?;
            if v.len() != 2 {
                return Err(medh5::Error::coded("E405", "a polygon's plane is (axis, index)").into());
            }
            (v[0], v[1])
        }
        _ => (-1, 0),
    };
    let role: String = match item.getattr("role") {
        Ok(r) if !r.is_none() => r.extract()?,
        _ => "outer".into(),
    };
    Ok(geo::Polygon::new(vertices, class_id, plane, &role)?)
}

#[pyfunction]
#[pyo3(signature = (polygons, *, ndim=None))]
fn encode_contours(polygons: &Bound<'_, PyAny>, ndim: Option<usize>) -> R<AnnotationPayload> {
    let found = polygons.try_iter()?.map(|p| polygon_arg(&p?)).collect::<R<Vec<_>>>()?;
    Ok(AnnotationPayload::wrap(geo::encode_contours(&found, ndim)?))
}

/// A contour role, or E411.
#[pyfunction]
fn check_contour_role(role: &str) -> R<String> {
    if !geo::CONTOUR_ROLES.contains(&role) {
        return Err(medh5::Error::coded(
            "E411",
            format!(
                "contour role {} must be one of {}",
                medh5::json::repr_str(role),
                medh5::json::repr_list(&geo::CONTOUR_ROLES)
            ),
        )
        .into());
    }
    Ok(role.to_string())
}

#[pyfunction]
#[pyo3(signature = (vertices, faces, *, normals=None, vertex_class_ids=None, mesh_offsets=None, mesh_class_ids=None))]
fn encode_mesh(
    vertices: &Bound<'_, PyAny>,
    faces: &Bound<'_, PyAny>,
    normals: Option<&Bound<'_, PyAny>>,
    vertex_class_ids: Option<&Bound<'_, PyAny>>,
    mesh_offsets: Option<&Bound<'_, PyAny>>,
    mesh_class_ids: Option<&Bound<'_, PyAny>>,
) -> R<AnnotationPayload> {
    let normals = given(normals).map(|n| f64_array(&n)).transpose()?;
    let vertex_ids = given(vertex_class_ids).map(|v| ints_of(&v)).transpose()?;
    let offsets = given(mesh_offsets).map(|o| i64_vec(&o)).transpose()?;
    let mesh_ids = given(mesh_class_ids).map(|m| ints_of(&m)).transpose()?;
    Ok(AnnotationPayload::wrap(geo::encode_mesh(
        &f64_array(vertices)?,
        &i64_array(faces)?,
        normals.as_ref(),
        vertex_ids.as_deref(),
        offsets.as_deref(),
        mesh_ids.as_deref(),
    )?))
}

/// A `slice_index` argument as planes: an array of exactly one plane per box
/// (§8.2), refused with E405 otherwise --- before flattening, because a column
/// of the right length can be the wrong shape.
pub fn slice_index_arg(obj: &Bound<'_, PyAny>, n_boxes: usize) -> R<Vec<i64>> {
    let array = crate::convert::i64_array(obj)?;
    if let Some(problem) = geo::check_slice_index_shape(array.shape(), n_boxes) {
        return Err(medh5::Error::coded("E405", problem).into());
    }
    Ok(array.iter().copied().collect())
}

#[pyfunction]
#[pyo3(signature = (planes, n_boxes, *, boxes=None, shape=None))]
fn check_slice_index(
    planes: &Bound<'_, PyAny>,
    n_boxes: usize,
    boxes: Option<&Bound<'_, PyAny>>,
    shape: Option<&Bound<'_, PyAny>>,
) -> R<Option<String>> {
    let array = crate::convert::i64_array(planes)?;
    if let Some(problem) = geo::check_slice_index_shape(array.shape(), n_boxes) {
        return Ok(Some(problem));
    }
    let planes: Vec<i64> = array.iter().copied().collect();
    let rows: Option<Vec<Vec<f64>>> = match given(boxes) {
        None => None,
        Some(b) => {
            let array = f64_array(&b)?;
            let n = array.shape().first().copied().unwrap_or(0);
            Some((0..n).map(|i| array.index_axis(ndarray::Axis(0), i).iter().copied().collect()).collect())
        }
    };
    let shape = shape_arg(shape)?;
    Ok(geo::check_slice_index(&planes, n_boxes, rows.as_deref(), shape.as_deref()))
}

#[pyfunction]
fn check_space(space: &str) -> R<String> {
    geo::check_space(space)?;
    Ok(space.to_string())
}

#[pyfunction]
fn check_scope(scope: &str) -> R<String> {
    geo::check_scope(scope)?;
    Ok(scope.to_string())
}

// -- classification -------------------------------------------------------------------------

/// `(classes, values, scope_ids, schemes, scheme_values)` from a mapping or
/// assertion rows (§9).
#[pyfunction]
fn assertion_rows<'py>(py: Python<'py>, labels: &Bound<'py, PyAny>) -> R<Bound<'py, PyTuple>> {
    let rows = crate::writer::assertion_rows(labels)?;
    let opt = |v: Option<Bound<'py, PyAny>>| v.unwrap_or_else(|| py.None().into_bound(py));
    Ok(PyTuple::new(
        py,
        [
            PyList::new(py, rows.classes)?.into_any(),
            PyList::new(py, rows.values)?.into_any(),
            opt(rows.scope_ids.map(|s| PyList::new(py, s)).transpose()?.map(Bound::into_any)),
            opt(rows.schemes.map(|s| PyList::new(py, s)).transpose()?.map(Bound::into_any)),
            opt(rows.scheme_values.map(|s| PyList::new(py, s)).transpose()?.map(Bound::into_any)),
        ],
    )?)
}

#[pyfunction]
#[pyo3(signature = (labels, *, scope="sample", multilabel=true, scope_ids=None, schemes=None, scheme_values=None))]
fn encode_classification(
    labels: &Bound<'_, PyAny>,
    scope: &str,
    multilabel: bool,
    scope_ids: Option<&Bound<'_, PyAny>>,
    schemes: Option<&Bound<'_, PyAny>>,
    scheme_values: Option<&Bound<'_, PyAny>>,
) -> R<AnnotationPayload> {
    geo::check_scope(scope)?;
    let rows = crate::writer::assertion_rows(labels)?;
    let class_ids = rows.classes.iter().map(int_of).collect::<PyResult<Vec<_>>>()?;
    let assertions = crate::writer::assertion_columns(rows, class_ids, scope_ids, schemes, scheme_values)?;
    Ok(AnnotationPayload::wrap(geo::encode_classification(&assertions, scope, multilabel)?))
}

// -- header ---------------------------------------------------------------------------------

#[pyfunction]
fn default_task_for_kind(kind: &str) -> Option<&'static str> {
    medh5::annotations::header::default_task_for_kind(kind)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AnnotationPayload>()?;
    m.add_class::<AnnotationHeader>()?;
    m.add_class::<OverlapStats>()?;
    m.add_class::<CostModel>()?;
    for f in [
        wrap_pyfunction!(analyse, m)?,
        wrap_pyfunction!(greedy_colour, m)?,
        wrap_pyfunction!(layers_from_colouring, m)?,
        wrap_pyfunction!(label_dtype_size, m)?,
        wrap_pyfunction!(cost_model, m)?,
        wrap_pyfunction!(select_encoding, m)?,
        wrap_pyfunction!(encode_labelmap, m)?,
        wrap_pyfunction!(encode_layers, m)?,
        wrap_pyfunction!(encode_bitmask, m)?,
        wrap_pyfunction!(encode_mask, m)?,
        wrap_pyfunction!(encode_probmap, m)?,
        wrap_pyfunction!(contains_at, m)?,
        wrap_pyfunction!(storage_dtype, m)?,
        wrap_pyfunction!(instance_id_dtype, m)?,
        wrap_pyfunction!(encode_instances, m)?,
        wrap_pyfunction!(instances_from_masks, m)?,
        wrap_pyfunction!(payload_to_masks, m)?,
        wrap_pyfunction!(encode_masks, m)?,
        wrap_pyfunction!(encode_voxels, m)?,
        wrap_pyfunction!(transcode_payload, m)?,
        wrap_pyfunction!(transcode, m)?,
        wrap_pyfunction!(annotation_to_masks, m)?,
        wrap_pyfunction!(masks_equal, m)?,
        wrap_pyfunction!(check_roundtrip, m)?,
        wrap_pyfunction!(normalize_masks, m)?,
        wrap_pyfunction!(check_class_id, m)?,
        wrap_pyfunction!(encode_boxes, m)?,
        wrap_pyfunction!(encode_obb, m)?,
        wrap_pyfunction!(encode_keypoints, m)?,
        wrap_pyfunction!(encode_points, m)?,
        wrap_pyfunction!(encode_contours, m)?,
        wrap_pyfunction!(check_contour_role, m)?,
        wrap_pyfunction!(encode_mesh, m)?,
        wrap_pyfunction!(check_slice_index, m)?,
        wrap_pyfunction!(check_space, m)?,
        wrap_pyfunction!(check_scope, m)?,
        wrap_pyfunction!(assertion_rows, m)?,
        wrap_pyfunction!(encode_classification, m)?,
        wrap_pyfunction!(default_task_for_kind, m)?,
    ] {
        m.add_function(f)?;
    }
    let py = m.py();
    use medh5::annotations::header as h;
    m.add("VOXEL_KINDS", PyTuple::new(py, h::VOXEL_KINDS)?)?;
    m.add("GEOMETRIC_KINDS", PyTuple::new(py, h::GEOMETRIC_KINDS)?)?;
    m.add("ANNOTATION_KINDS", PyTuple::new(py, h::ANNOTATION_KINDS)?)?;
    m.add("RESERVED_KINDS", PyTuple::new(py, h::RESERVED_KINDS)?)?;
    m.add("TASKS", PyTuple::new(py, h::TASKS)?)?;
    m.add("SPEC_ANNOTATION_ATTRS", PyTuple::new(py, h::SPEC_ANNOTATION_ATTRS)?)?;
    m.add("BITS_PER_PLANE", enc::BITS_PER_PLANE)?;
    m.add("DEFAULT_THRESHOLD", enc::DEFAULT_THRESHOLD)?;
    m.add("TRANSCODABLE", PyTuple::new(py, enc::TRANSCODABLE)?)?;
    m.add("IN_BAND_IGNORE_KINDS", PyTuple::new(py, enc::IN_BAND_IGNORE_KINDS)?)?;
    m.add("SPACES", PyTuple::new(py, geo::SPACES)?)?;
    m.add("CONTOUR_ROLES", PyTuple::new(py, geo::CONTOUR_ROLES)?)?;
    m.add("SCOPES", PyTuple::new(py, geo::SCOPES)?)?;
    m.add("ROTATION_TOL", geo::ROTATION_TOL)?;
    let visibility = PyDict::new(py);
    for (k, v) in geo::VISIBILITY {
        visibility.set_item(k, v)?;
    }
    m.add("VISIBILITY", visibility)?;
    m.add("LOCALIZED_BBOX_FRACTION", select::LOCALIZED_BBOX_FRACTION)?;
    m.add("SPARSE_FILL", select::SPARSE_FILL)?;
    m.add("SLAB_BYTES", medh5::annotations::payload::SLAB_BYTES)?;
    let _ = (untyped_to_nd, NdArray::Bool);
    Ok(())
}
