//! `medh5.transforms`: encoders, field sampling and the frame resolver
//! (spec §10).
//!
//! Every transform maps points from `from_frame` to `to_frame` ---
//! `x_M = T(x_F)`, the ITK convention --- and the resolver returns `None`
//! rather than inventing a path between frames no transform relates.

use indexmap::IndexMap;
use ndarray::{Array2, ArrayD, IxDyn};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use medh5::array::DType;
use medh5::transforms::apply;
use medh5::transforms::encode as enc;
use medh5::transforms::model;
use medh5::transforms::resolve;

use crate::annotations::AnnotationPayload;
use crate::convert::{array_to_py, attr_to_py, dtype_arg, f64_array, py_to_attr, py_to_nd, strings};
use crate::errors::R;
use crate::geometry::{grid_arg, Grid};
use crate::reader::TransformHandle;

fn points2(obj: &Bound<'_, PyAny>) -> PyResult<Array2<f64>> {
    let a = f64_array(obj)?;
    let dim = a.shape().last().copied().unwrap_or(0);
    apply::as_points(&a, dim).map_err(|e| crate::errors::BindError::from(e).into())
}

// -- encoders ---------------------------------------------------------------------------

#[pyfunction]
fn encode_identity() -> AnnotationPayload {
    AnnotationPayload::wrap(enc::encode_identity())
}

#[pyfunction]
fn encode_affine(matrix: &Bound<'_, PyAny>) -> R<AnnotationPayload> {
    Ok(AnnotationPayload::wrap(enc::encode_affine(&f64_array(matrix)?)?))
}

#[pyfunction]
#[pyo3(signature = (field, *, field_grid, vector_space="world", interpolation="linear", extrapolation="zero", dtype=None))]
fn encode_displacement(
    field: &Bound<'_, PyAny>,
    field_grid: &str,
    vector_space: &str,
    interpolation: &str,
    extrapolation: &str,
    dtype: Option<&Bound<'_, PyAny>>,
) -> R<AnnotationPayload> {
    let dtype = match dtype.filter(|d| !d.is_none()) {
        Some(d) => dtype_arg(d)?,
        None => DType::F32,
    };
    Ok(AnnotationPayload::wrap(enc::encode_displacement(
        &py_to_nd(field)?,
        field_grid,
        vector_space,
        interpolation,
        extrapolation,
        dtype,
    )?))
}

#[pyfunction]
#[pyo3(signature = (control_points, *, cp_grid, order=model::DEFAULT_ORDER, vector_space="world"))]
fn encode_bspline(
    control_points: &Bound<'_, PyAny>,
    cp_grid: &str,
    order: i64,
    vector_space: &str,
) -> R<AnnotationPayload> {
    Ok(AnnotationPayload::wrap(enc::encode_bspline(&f64_array(control_points)?, cp_grid, order, vector_space)?))
}

#[pyfunction]
fn encode_composite(components: &Bound<'_, PyAny>) -> R<AnnotationPayload> {
    Ok(AnnotationPayload::wrap(enc::encode_composite(&strings(components)?)?))
}

/// The B-spline basis weights at `t`: `(order + 1, *t.shape)`.
#[pyfunction]
fn basis<'py>(py: Python<'py>, order: i64, t: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
    let ts = f64_array(t)?;
    let n = (order + 1).max(0) as usize;
    let mut shape = vec![n];
    shape.extend_from_slice(ts.shape());
    let mut out = ArrayD::<f64>::zeros(IxDyn(&shape));
    for (i, value) in ts.iter().enumerate() {
        let w = model::basis(order, *value)?;
        for (k, weight) in w.iter().enumerate() {
            out.as_slice_mut().expect("standard layout")[k * ts.len() + i] = *weight;
        }
    }
    Ok(array_to_py(py, out))
}

#[pyfunction]
fn check_transform_id(transform_id: &str) -> R<String> {
    Ok(model::check_transform_id(transform_id)?.to_string())
}

// -- sampling -----------------------------------------------------------------------------

#[pyfunction]
fn inside_extent<'py>(py: Python<'py>, spatial: Vec<usize>, points: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
    let inside = apply::inside_extent(&spatial, &points2(points)?);
    Ok(array_to_py(py, ArrayD::from_shape_vec(IxDyn(&[inside.len()]), inside)?))
}

#[pyfunction]
fn refuse_outside(inside: &Bound<'_, PyAny>) -> R<()> {
    let flags: Vec<bool> = crate::convert::bool_array(inside)?.iter().copied().collect();
    Ok(apply::refuse_outside(&flags)?)
}

fn field_of(obj: &Bound<'_, PyAny>) -> PyResult<ArrayD<f64>> {
    f64_array(obj)
}

#[pyfunction]
#[pyo3(signature = (field, coords, *, extrapolation="zero"))]
fn linear_sample<'py>(
    py: Python<'py>,
    field: &Bound<'py, PyAny>,
    coords: &Bound<'py, PyAny>,
    extrapolation: &str,
) -> R<Bound<'py, PyAny>> {
    Ok(array_to_py(py, apply::linear_sample(&field_of(field)?, &points2(coords)?, extrapolation)?.into_dyn()))
}

#[pyfunction]
#[pyo3(signature = (field, coords, *, extrapolation="zero"))]
fn cubic_sample<'py>(
    py: Python<'py>,
    field: &Bound<'py, PyAny>,
    coords: &Bound<'py, PyAny>,
    extrapolation: &str,
) -> R<Bound<'py, PyAny>> {
    Ok(array_to_py(py, apply::cubic_sample(&field_of(field)?, &points2(coords)?, extrapolation)?.into_dyn()))
}

#[pyfunction]
#[pyo3(signature = (field, coords, *, interpolation="linear", extrapolation="zero"))]
fn sample_field<'py>(
    py: Python<'py>,
    field: &Bound<'py, PyAny>,
    coords: &Bound<'py, PyAny>,
    interpolation: &str,
    extrapolation: &str,
) -> R<Bound<'py, PyAny>> {
    Ok(array_to_py(
        py,
        apply::sample_field(&field_of(field)?, &points2(coords)?, interpolation, extrapolation)?.into_dyn(),
    ))
}

#[pyfunction]
fn linear_part<'py>(py: Python<'py>, grid: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
    Ok(array_to_py(py, apply::linear_part(&grid_arg(grid)?).into_dyn()))
}

#[pyfunction]
fn to_world_vectors<'py>(
    py: Python<'py>,
    vectors: &Bound<'py, PyAny>,
    grid: &Bound<'py, PyAny>,
    vector_space: &str,
) -> R<Bound<'py, PyAny>> {
    Ok(array_to_py(py, apply::to_world_vectors(&points2(vectors)?, &grid_arg(grid)?, vector_space)?.into_dyn()))
}

#[pyfunction]
#[pyo3(signature = (field, grid, *, vector_space="world"))]
fn jacobian_determinant<'py>(
    py: Python<'py>,
    field: &Bound<'py, PyAny>,
    grid: &Bound<'py, PyAny>,
    vector_space: &str,
) -> R<Bound<'py, PyAny>> {
    Ok(array_to_py(py, apply::jacobian_determinant(&field_of(field)?, &grid_arg(grid)?, vector_space)?))
}

#[pyfunction]
fn folding_fraction(determinants: &Bound<'_, PyAny>) -> R<f64> {
    Ok(apply::folding_fraction(&f64_array(determinants)?))
}

/// TRE `‖T(p_F) − p_M‖` over world landmarks with matching row order
/// (§10.6): `{mean, median, max, n}`.
#[pyfunction]
#[pyo3(signature = (transform, fixed_points, moving_points, *, weights=None))]
fn target_registration_error<'py>(
    py: Python<'py>,
    transform: &Bound<'py, PyAny>,
    fixed_points: &Bound<'py, PyAny>,
    moving_points: &Bound<'py, PyAny>,
    weights: Option<Vec<f64>>,
) -> R<Bound<'py, PyDict>> {
    let fixed = f64_array(fixed_points)?;
    let moving = f64_array(moving_points)?;
    if fixed.shape() != moving.shape() {
        return Err(medh5::Error::invalid(format!(
            "landmark sets disagree: {} vs {}; §10.6 requires equal N and matching row order",
            medh5::json::repr_int_tuple(fixed.shape()),
            medh5::json::repr_int_tuple(moving.shape())
        ))
        .into());
    }
    let warped = transform.call_method1("transform_points", (array_to_py(py, fixed.clone()),))?;
    let warped = points2(&warped)?;
    let dim = moving.shape().last().copied().unwrap_or(0);
    let moving = apply::as_points(&moving, dim)?;
    let tre = apply::tre_from_warped(&warped, &moving, weights.as_deref())?;
    let out = PyDict::new(py);
    out.set_item("mean", tre.mean)?;
    out.set_item("median", tre.median)?;
    out.set_item("max", tre.max)?;
    out.set_item("n", tre.n)?;
    Ok(out)
}

// -- header and graph ------------------------------------------------------------------------

/// The attribute header every transform carries (spec §10.1).
#[pyclass(module = "medh5.transforms.base", name = "TransformHeader", skip_from_py_object, frozen)]
pub struct TransformHeader {
    pub inner: model::TransformHeader,
}

#[pymethods]
impl TransformHeader {
    #[new]
    #[pyo3(signature = (kind, from_frame, to_frame, units="mm", from_grid=None, to_grid=None, invertible=None,
        inverse_id=None, prov=None, metrics=None, extra=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        kind: &str,
        from_frame: &str,
        to_frame: &str,
        units: &str,
        from_grid: Option<String>,
        to_grid: Option<String>,
        invertible: Option<bool>,
        inverse_id: Option<String>,
        prov: Option<String>,
        metrics: Option<String>,
        extra: Option<&Bound<'_, PyAny>>,
    ) -> R<Self> {
        let mut h = model::TransformHeader::new(kind, from_frame, to_frame)?;
        h.units = units.to_string();
        h.from_grid = from_grid;
        h.to_grid = to_grid;
        h.invertible = invertible;
        h.inverse_id = inverse_id;
        h.prov = prov;
        h.metrics = metrics;
        if let Some(e) = extra.filter(|e| !e.is_none()) {
            for item in e.call_method0("items")?.try_iter()? {
                let (k, v): (String, Bound<'_, PyAny>) = item?.extract()?;
                h.extra.push((k, py_to_attr(&v)?));
            }
        }
        Ok(TransformHeader { inner: h })
    }
    #[getter]
    fn kind(&self) -> &str {
        &self.inner.kind
    }
    #[getter]
    fn from_frame(&self) -> &str {
        &self.inner.from_frame
    }
    #[getter]
    fn to_frame(&self) -> &str {
        &self.inner.to_frame
    }
    #[getter]
    fn units(&self) -> &str {
        &self.inner.units
    }
    #[getter]
    fn from_grid(&self) -> Option<&str> {
        self.inner.from_grid.as_deref()
    }
    #[getter]
    fn to_grid(&self) -> Option<&str> {
        self.inner.to_grid.as_deref()
    }
    #[getter]
    fn invertible(&self) -> Option<bool> {
        self.inner.invertible
    }
    #[getter]
    fn inverse_id(&self) -> Option<&str> {
        self.inner.inverse_id.as_deref()
    }
    #[getter]
    fn prov(&self) -> Option<&str> {
        self.inner.prov.as_deref()
    }
    #[getter]
    fn metrics(&self) -> Option<&str> {
        self.inner.metrics.as_deref()
    }
    #[getter]
    fn extra<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (k, v) in &self.inner.extra {
            out.set_item(k, attr_to_py(py, v)?)?;
        }
        Ok(out)
    }
    fn attrs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (k, v) in self.inner.attrs() {
            out.set_item(k, attr_to_py(py, &v)?)?;
        }
        Ok(out)
    }
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other.cast::<TransformHeader>().map(|o| o.get().inner == self.inner).unwrap_or(false)
    }
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let names = [
            "kind",
            "from_frame",
            "to_frame",
            "units",
            "from_grid",
            "to_grid",
            "invertible",
            "inverse_id",
            "prov",
            "metrics",
            "extra",
        ];
        let fields = names.iter().map(|n| Ok((*n, slf.getattr(*n)?))).collect::<PyResult<Vec<_>>>()?;
        crate::values::dataclass_repr("TransformHeader", &fields)
    }
}

/// `{transform_id: engine transform}` from a mapping of facades or handles.
fn transforms_arg(obj: &Bound<'_, PyAny>) -> PyResult<IndexMap<String, model::Transform>> {
    let mut out = IndexMap::new();
    for item in obj.call_method0("items")?.try_iter()? {
        let (k, v): (String, Bound<'_, PyAny>) = item?.extract()?;
        out.insert(k, handle_of(&v)?.get().inner.as_ref().clone());
    }
    Ok(out)
}

pub fn handle_of<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, TransformHandle>> {
    match obj.cast::<TransformHandle>() {
        Ok(h) => Ok(h.clone()),
        Err(_) => Ok(obj.getattr("_handle")?.cast_into::<TransformHandle>()?),
    }
}

/// Frame -> frames one hop away, as the resolver would walk them.
#[pyfunction]
fn frame_graph<'py>(py: Python<'py>, transforms: &Bound<'py, PyAny>) -> R<Bound<'py, PyDict>> {
    let graph = resolve::frame_graph(&transforms_arg(transforms)?)?;
    let out = PyDict::new(py);
    for (k, v) in graph {
        out.set_item(k, PyList::new(py, v)?)?;
    }
    Ok(out)
}

/// The transform relating two frames, or `None` when no path exists.
#[pyfunction]
fn resolve_between(transforms: &Bound<'_, PyAny>, from_frame: &str, to_frame: &str) -> R<Option<TransformHandle>> {
    Ok(resolve::resolve_between(&transforms_arg(transforms)?, from_frame, to_frame)?.map(TransformHandle::wrap))
}

/// The frames of every grid of one timepoint, in grid order.
#[pyfunction]
fn frames_of_timepoint<'py>(py: Python<'py>, grids: &Bound<'py, PyAny>, timepoint: &str) -> R<Bound<'py, PyTuple>> {
    let mut map = IndexMap::new();
    for item in grids.call_method0("items")?.try_iter()? {
        let (k, v): (String, Bound<'_, PyAny>) = item?.extract()?;
        map.insert(k, grid_arg(&v)?);
    }
    Ok(PyTuple::new(py, resolve::frames_of_timepoint(&map, timepoint))?)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TransformHeader>()?;
    for f in [
        wrap_pyfunction!(encode_identity, m)?,
        wrap_pyfunction!(encode_affine, m)?,
        wrap_pyfunction!(encode_displacement, m)?,
        wrap_pyfunction!(encode_bspline, m)?,
        wrap_pyfunction!(encode_composite, m)?,
        wrap_pyfunction!(basis, m)?,
        wrap_pyfunction!(check_transform_id, m)?,
        wrap_pyfunction!(inside_extent, m)?,
        wrap_pyfunction!(refuse_outside, m)?,
        wrap_pyfunction!(linear_sample, m)?,
        wrap_pyfunction!(cubic_sample, m)?,
        wrap_pyfunction!(sample_field, m)?,
        wrap_pyfunction!(linear_part, m)?,
        wrap_pyfunction!(to_world_vectors, m)?,
        wrap_pyfunction!(jacobian_determinant, m)?,
        wrap_pyfunction!(folding_fraction, m)?,
        wrap_pyfunction!(target_registration_error, m)?,
        wrap_pyfunction!(frame_graph, m)?,
        wrap_pyfunction!(resolve_between, m)?,
        wrap_pyfunction!(frames_of_timepoint, m)?,
    ] {
        m.add_function(f)?;
    }
    let py = m.py();
    m.add("TRANSFORM_KINDS", PyTuple::new(py, model::TRANSFORM_KINDS)?)?;
    m.add("VECTOR_SPACES", PyTuple::new(py, model::VECTOR_SPACES)?)?;
    m.add("INTERPOLATIONS", PyTuple::new(py, model::INTERPOLATIONS)?)?;
    m.add("EXTRAPOLATIONS", PyTuple::new(py, apply::EXTRAPOLATIONS)?)?;
    m.add("SPEC_TRANSFORM_ATTRS", PyTuple::new(py, model::SPEC_TRANSFORM_ATTRS)?)?;
    m.add("LAST_ROW_TOL", model::LAST_ROW_TOL)?;
    m.add("SUPPORTED_ORDERS", PyTuple::new(py, model::SUPPORTED_ORDERS)?)?;
    m.add("DEFAULT_ORDER", model::DEFAULT_ORDER)?;
    m.add("FLOAT16_SAFE_VOXELS", model::FLOAT16_SAFE_VOXELS)?;
    let _ = Grid::wrap;
    Ok(())
}
