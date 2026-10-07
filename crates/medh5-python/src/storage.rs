//! `medh5.storage`: codec profiles, chunk sizing, sampling indices and
//! recompression (spec §14).

use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use medh5::storage::{chunking, codecs, index, recompress as rc};

use crate::convert::{array_to_py, bool_array, class_keys, json_to_py, opt_strings};
use crate::errors::R;
use crate::geometry::grid_arg;

// -- codecs ------------------------------------------------------------------------------

fn codec_json(c: &codecs::Codec) -> serde_json::Value {
    serde_json::json!({
        "name": c.name,
        "blosc2": c.blosc2.map(|(cname, level, shuffle)| serde_json::json!([cname, level, shuffle])),
        "gzip_level": c.gzip_level,
        "shuffle": c.shuffle,
    })
}

/// The codec profiles, as plain data (`medh5.storage.codecs` builds its
/// `CodecProfile` values from this).
#[pyfunction]
fn codec_profiles<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for p in codecs::profiles() {
        let doc = serde_json::json!({
            "name": p.name,
            "image": codec_json(&p.image),
            "label": codec_json(&p.label),
            "description": p.description,
        });
        out.append(json_to_py(py, &doc)?)?;
    }
    Ok(out)
}

/// A profile name, checked (`None` is the default, `balanced`).
#[pyfunction]
#[pyo3(signature = (name=None))]
fn resolve_profile_name(name: Option<&str>) -> R<String> {
    Ok(codecs::resolve_profile(name)?.name.to_string())
}

fn role_arg(role: &str) -> R<codecs::Role> {
    Ok(match role {
        "image" => codecs::Role::Image,
        "label" => codecs::Role::Label,
        "aux" => codecs::Role::Aux,
        other => {
            return Err(medh5::Error::invalid(format!(
                "unknown storage role {}; expected 'image', 'label' or 'aux'",
                medh5::json::repr_str(other)
            ))
            .into())
        }
    })
}

/// The layout one dataset gets under a profile: `{}` when stored contiguous,
/// else `{"chunks": ..., "codec": ...}`.
#[pyfunction]
#[pyo3(signature = (shape, itemsize, *, profile=None, role="image", chunks=None))]
fn dataset_layout<'py>(
    py: Python<'py>,
    shape: Vec<usize>,
    itemsize: usize,
    profile: Option<&str>,
    role: &str,
    chunks: Option<Vec<usize>>,
) -> R<Bound<'py, PyDict>> {
    let profile = codecs::resolve_profile(profile)?;
    let role = role_arg(role)?;
    let layout = codecs::dataset_layout(&shape, itemsize, &profile, role, chunks);
    let out = PyDict::new(py);
    if let Some(c) = layout.chunks {
        out.set_item("chunks", PyTuple::new(py, c)?)?;
        out.set_item("codec", profile.codec(role).name)?;
    }
    Ok(out)
}

/// A dataset's actual HDF5 filter pipeline, e.g. `blosc2:zstd:3+shuffle`.
#[pyfunction]
fn describe_filters(dataset: &Bound<'_, PyAny>) -> R<String> {
    Ok(codecs::describe_filters(&crate::nodes::dataset_arg(dataset)?.get().ds)?)
}

/// Whether a dataset is large enough for the W902 warning.
#[pyfunction]
fn is_bulk(dataset: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(codecs::is_bulk(&crate::nodes::dataset_arg(dataset)?.get().ds))
}

/// `portable` when every dataset of the file needs only HDF5's own filters,
/// else `balanced` --- what an amend of the file defaults to.
#[pyfunction]
fn profile_family(path: PathBuf) -> R<&'static str> {
    let file = medh5::h5::file::open_read(&path)?;
    Ok(codecs::profile_family(&file.as_group()?)?)
}

// -- chunking ------------------------------------------------------------------------------

fn patch_arg(patch: Option<&Bound<'_, PyAny>>) -> PyResult<chunking::Patch> {
    match patch.filter(|p| !p.is_none()) {
        None => Ok(chunking::Patch::Default),
        Some(p) => match p.extract::<usize>() {
            Ok(n) => Ok(chunking::Patch::Uniform(n)),
            Err(_) => Ok(chunking::Patch::PerAxis(p.extract::<Vec<i64>>()?)),
        },
    }
}

#[pyfunction]
fn detect_l3_bytes() -> u64 {
    chunking::detect_l3_bytes()
}

#[pyfunction]
#[pyo3(signature = (spatial_shape, patch=None, *, itemsize=4, l3_bytes=None))]
fn spatial_chunk_for<'py>(
    py: Python<'py>,
    spatial_shape: Vec<usize>,
    patch: Option<&Bound<'py, PyAny>>,
    itemsize: usize,
    l3_bytes: Option<u64>,
) -> R<Bound<'py, PyTuple>> {
    Ok(PyTuple::new(py, chunking::spatial_chunk_for(&spatial_shape, &patch_arg(patch)?, itemsize, l3_bytes)?)?)
}

#[pyfunction]
#[pyo3(signature = (shape, axis_kinds, patch=None, *, itemsize=4, l3_bytes=None, leading=0))]
fn optimize_chunks<'py>(
    py: Python<'py>,
    shape: Vec<usize>,
    axis_kinds: &Bound<'py, PyAny>,
    patch: Option<&Bound<'py, PyAny>>,
    itemsize: usize,
    l3_bytes: Option<u64>,
    leading: usize,
) -> R<Bound<'py, PyTuple>> {
    let kinds = crate::convert::strings(axis_kinds)?;
    Ok(PyTuple::new(py, chunking::optimize_chunks(&shape, &kinds, &patch_arg(patch)?, itemsize, l3_bytes, leading)?)?)
}

#[pyfunction]
#[pyo3(signature = (grid, itemsize, *, leading=0))]
fn grid_chunks<'py>(
    py: Python<'py>,
    grid: &Bound<'py, PyAny>,
    itemsize: usize,
    leading: usize,
) -> R<Bound<'py, PyTuple>> {
    Ok(PyTuple::new(py, chunking::grid_chunks(&grid_arg(grid)?, itemsize, leading)?)?)
}

#[pyfunction]
fn fit_chunks<'py>(py: Python<'py>, proposed: Vec<usize>, shape: Vec<usize>) -> PyResult<Option<Bound<'py, PyTuple>>> {
    chunking::fit_chunks(&proposed, &shape).map(|c| PyTuple::new(py, c)).transpose()
}

#[pyfunction]
fn field_chunks<'py>(
    py: Python<'py>,
    grid: &Bound<'py, PyAny>,
    shape: Vec<usize>,
    itemsize: usize,
) -> R<Option<Bound<'py, PyTuple>>> {
    Ok(chunking::field_chunks(&grid_arg(grid)?, &shape, itemsize)?.map(|c| PyTuple::new(py, c)).transpose()?)
}

#[pyfunction]
fn chunk_report<'py>(
    py: Python<'py>,
    shape: Vec<usize>,
    chunks: Vec<usize>,
    itemsize: usize,
) -> PyResult<Bound<'py, PyAny>> {
    json_to_py(py, &chunking::chunk_report(&shape, &chunks, itemsize))
}

// -- indices ---------------------------------------------------------------------------------

/// The datasets of one annotation's index entry (§14.3).
#[pyclass(module = "medh5.storage.index", name = "IndexPayload", skip_from_py_object, frozen)]
pub struct IndexPayload {
    pub inner: index::IndexPayload,
}

#[pymethods]
impl IndexPayload {
    #[getter]
    fn ann_id(&self) -> &str {
        &self.inner.ann_id
    }
    #[getter]
    fn class_ids<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        let v = self.inner.class_ids.clone();
        Ok(array_to_py(py, ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[v.len()]), v)?))
    }
    #[getter]
    fn voxel_counts<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        let v = self.inner.voxel_counts.clone();
        Ok(array_to_py(py, ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[v.len()]), v)?))
    }
    #[getter]
    fn class_bboxes<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        array_to_py(py, self.inner.class_bboxes.clone().into_dyn())
    }
    #[getter]
    fn fg_coords<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (k, v) in &self.inner.fg_coords {
            out.set_item(k, array_to_py(py, v.clone().into_dyn()))?;
        }
        Ok(out)
    }
    #[getter]
    fn occupancy<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyAny>> {
        self.inner.occupancy.clone().map(|o| array_to_py(py, o))
    }
    #[getter]
    fn source_digest(&self) -> Option<&str> {
        self.inner.source_digest.as_deref()
    }
    #[getter]
    fn max_coords(&self) -> usize {
        self.inner.max_coords
    }
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }
    #[getter]
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        out.set_item("total_foreground", self.inner.total_foreground)?;
        Ok(out)
    }
    fn __repr__(&self) -> String {
        format!(
            "IndexPayload({}, classes={}, occupancy={})",
            medh5::json::repr_str(&self.inner.ann_id),
            self.inner.class_ids.len(),
            if self.inner.occupancy.is_some() { "True" } else { "False" }
        )
    }
}

/// Compute the sampling index of one voxel annotation (not written).
#[pyfunction]
#[pyo3(signature = (annotation, *, classes=None, max_coords=index::DEFAULT_MAX_COORDS,
    occupancy=Some(index::DEFAULT_OCCUPANCY_FACTOR), seed=0, source_digest=None))]
fn build_index(
    annotation: &Bound<'_, PyAny>,
    classes: Option<&Bound<'_, PyAny>>,
    max_coords: usize,
    occupancy: Option<usize>,
    seed: u64,
    source_digest: Option<String>,
) -> R<IndexPayload> {
    let handle = match annotation.cast::<crate::reader::AnnotationHandle>() {
        Ok(h) => h.clone(),
        Err(_) => annotation.getattr("_handle")?.cast_into::<crate::reader::AnnotationHandle>().map_err(PyErr::from)?,
    };
    let keys = classes.filter(|c| !c.is_none()).map(class_keys).transpose()?;
    let payload = index::build_index(&handle.get().inner, keys.as_deref(), max_coords, occupancy, seed, source_digest)?;
    Ok(IndexPayload { inner: payload })
}

/// Every stored index entry under a sample root (`Sample.root`, or the
/// `Sample`), by annotation id.
#[pyfunction]
fn read_indices<'py>(py: Python<'py>, root: &Bound<'py, PyAny>) -> R<Bound<'py, PyDict>> {
    let group = crate::integrity::group_of(root)?;
    let out = PyDict::new(py);
    for (name, found) in index::read_indices(&group)? {
        out.set_item(name, crate::reader::IndexHandle { inner: std::sync::Arc::new(found) })?;
    }
    Ok(out)
}

/// The occupancy map of a mask: one bit per `factor`-cube of voxels.
#[pyfunction]
fn occupancy<'py>(py: Python<'py>, mask: &Bound<'py, PyAny>, factor: usize) -> R<Bound<'py, PyAny>> {
    Ok(array_to_py(py, index::occupancy(&bool_array(mask)?, factor)))
}

// -- recompression -------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (path, profile, *, out=None, rechunk=false))]
fn recompress<'py>(
    py: Python<'py>,
    path: PathBuf,
    profile: String,
    out: Option<PathBuf>,
    rechunk: bool,
) -> R<Bound<'py, PyAny>> {
    let result = py.detach(move || rc::recompress(&path, &profile, out.as_deref(), rechunk))?;
    Ok(json_to_py(py, &result.to_json())?)
}

#[pyfunction]
#[pyo3(signature = (paths, profile, *, rechunk=false))]
fn recompress_paths<'py>(
    py: Python<'py>,
    paths: Vec<PathBuf>,
    profile: String,
    rechunk: bool,
) -> R<Bound<'py, PyList>> {
    let results = py.detach(move || {
        let refs: Vec<&std::path::Path> = paths.iter().map(PathBuf::as_path).collect();
        rc::recompress_paths(&refs, &profile, rechunk)
    })?;
    let out = PyList::empty(py);
    for r in results {
        out.append(json_to_py(py, &r.to_json())?)?;
    }
    Ok(out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<IndexPayload>()?;
    for f in [
        wrap_pyfunction!(codec_profiles, m)?,
        wrap_pyfunction!(resolve_profile_name, m)?,
        wrap_pyfunction!(dataset_layout, m)?,
        wrap_pyfunction!(describe_filters, m)?,
        wrap_pyfunction!(is_bulk, m)?,
        wrap_pyfunction!(profile_family, m)?,
        wrap_pyfunction!(detect_l3_bytes, m)?,
        wrap_pyfunction!(spatial_chunk_for, m)?,
        wrap_pyfunction!(optimize_chunks, m)?,
        wrap_pyfunction!(grid_chunks, m)?,
        wrap_pyfunction!(fit_chunks, m)?,
        wrap_pyfunction!(field_chunks, m)?,
        wrap_pyfunction!(chunk_report, m)?,
        wrap_pyfunction!(build_index, m)?,
        wrap_pyfunction!(occupancy, m)?,
        wrap_pyfunction!(read_indices, m)?,
        wrap_pyfunction!(recompress, m)?,
        wrap_pyfunction!(recompress_paths, m)?,
    ] {
        m.add_function(f)?;
    }
    m.add("COMPRESS_MIN_BYTES", codecs::COMPRESS_MIN_BYTES)?;
    m.add("BULK_MIN_BYTES", codecs::BULK_MIN_BYTES)?;
    m.add("BLOSC2_FILTER_ID", codecs::BLOSC2_FILTER_ID)?;
    m.add("BLOSC_FILTER_ID", codecs::BLOSC_FILTER_ID)?;
    m.add("BUILTIN_FILTER_IDS", PyTuple::new(m.py(), codecs::BUILTIN_FILTER_IDS)?)?;
    m.add("DEFAULT_PROFILE", codecs::DEFAULT_PROFILE)?;
    m.add("DEFAULT_L3_BYTES", chunking::DEFAULT_L3_BYTES)?;
    m.add("MIN_CHUNK_BYTES", chunking::MIN_CHUNK_BYTES)?;
    m.add("MAX_CHUNK_BYTES", chunking::MAX_CHUNK_BYTES)?;
    m.add("CACHE_SAFETY", chunking::CACHE_SAFETY)?;
    m.add("OVERSHOOT_LIMIT", chunking::OVERSHOOT_LIMIT)?;
    m.add("DEFAULT_PATCH", chunking::DEFAULT_PATCH)?;
    m.add("DEFAULT_MAX_COORDS", index::DEFAULT_MAX_COORDS)?;
    m.add("DEFAULT_OCCUPANCY_FACTOR", index::DEFAULT_OCCUPANCY_FACTOR)?;
    let _ = opt_strings;
    Ok(())
}
