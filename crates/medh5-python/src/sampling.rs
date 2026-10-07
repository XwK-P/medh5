//! `medh5.sampling`: where to read --- patch windows and visit pairs (spec
//! §14.3).
//!
//! The decisions are the engine's: which annotation and grid a draw is
//! measured in, foreground from the index or a scan, class weighting, window
//! placement.  A generator passed in is a `numpy.random.Generator`, drawn from
//! in the order 1.x drew, so a seeded draw is the 1.x draw.  The result types
//! (`Patch`, `TimepointPair`) stay the 1.x dataclasses; this module hands back
//! their fields.

use indexmap::IndexMap;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySlice, PyString, PyTuple};

use medh5::sampling::{self as engine, ClassWeights, PatchSize};

use crate::convert::class_keys;
use crate::errors::R;
use crate::reader::sample_arg;
use crate::rng::AnyRng;

/// A patch size: one length for every axis, or one per axis.
fn patch_size_arg(obj: &Bound<'_, PyAny>) -> PyResult<PatchSize> {
    if let Ok(v) = obj.extract::<i64>() {
        return Ok(PatchSize::Scalar(v));
    }
    let mut axes = Vec::new();
    for item in obj.try_iter()? {
        axes.push(item?.extract::<i64>()?);
    }
    Ok(PatchSize::Axes(axes))
}

fn class_weights_arg(obj: Option<&Bound<'_, PyAny>>) -> PyResult<ClassWeights> {
    let Some(obj) = obj.filter(|o| !o.is_none()) else {
        return Ok(ClassWeights::Named("uniform".into()));
    };
    if let Ok(name) = obj.cast::<PyString>() {
        return Ok(ClassWeights::Named(name.to_string()));
    }
    let mut map = IndexMap::new();
    for item in obj.call_method0("items")?.try_iter()? {
        let (k, v): (i64, f64) = item?.extract()?;
        map.insert(k, v);
    }
    Ok(ClassWeights::Explicit(map))
}

/// `slice(start, stop)`, step `None` as 1.x built them.
fn slice_of<'py>(py: Python<'py>, (start, stop): (i64, i64)) -> PyResult<Bound<'py, PyAny>> {
    py.get_type::<PySlice>().call1((start, stop))
}

fn slices_of<'py>(py: Python<'py>, slices: &[(i64, i64)]) -> PyResult<Bound<'py, PyTuple>> {
    let items = slices.iter().map(|s| slice_of(py, *s)).collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(py, items)
}

/// A `Patch`'s fields, as its constructor takes them.
fn patch_fields<'py>(py: Python<'py>, p: &engine::Patch) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("slices", slices_of(py, &p.slices)?)?;
    out.set_item("pad", PyTuple::new(py, p.pad.iter().copied())?)?;
    out.set_item("center", PyTuple::new(py, p.center.iter().copied())?)?;
    out.set_item("strategy", &p.strategy)?;
    out.set_item("class_id", p.class_id)?;
    out.set_item("used_index", p.used_index)?;
    out.set_item("grid_id", &p.grid_id)?;
    Ok(out)
}

/// The engine half of `PatchSampler`: its configuration, validated, and the
/// draws.
#[pyclass(module = "medh5._core", name = "PatchSamplerHandle", frozen)]
pub struct PatchSamplerHandle {
    inner: engine::PatchSampler,
}

#[pymethods]
impl PatchSamplerHandle {
    #[new]
    #[pyo3(signature = (patch_size, *, strategy="balanced".to_string(), foreground_prob=0.5, foreground_classes=None,
        class_weights=None))]
    fn new(
        patch_size: &Bound<'_, PyAny>,
        strategy: String,
        foreground_prob: f64,
        foreground_classes: Option<&Bound<'_, PyAny>>,
        class_weights: Option<&Bound<'_, PyAny>>,
    ) -> R<Self> {
        let classes = foreground_classes.filter(|c| !c.is_none()).map(class_keys).transpose()?;
        let inner = engine::PatchSampler::new(
            patch_size_arg(patch_size)?,
            &strategy,
            foreground_prob,
            classes,
            class_weights_arg(class_weights)?,
        )?;
        Ok(PatchSamplerHandle { inner })
    }

    fn repr(&self) -> String {
        self.inner.repr()
    }

    /// One draw: the fields of a `Patch`.
    #[pyo3(signature = (sample, annotation=None, rng=None, *, grid=None))]
    fn draw<'py>(
        &self,
        py: Python<'py>,
        sample: &Bound<'py, PyAny>,
        annotation: Option<String>,
        rng: Option<Bound<'py, PyAny>>,
        grid: Option<String>,
    ) -> R<Bound<'py, PyDict>> {
        let sample = sample_arg(sample)?;
        let mut rng = AnyRng::from_arg(rng);
        let patch = self.inner.draw(&sample, annotation.as_deref(), rng.as_dyn(), grid.as_deref())?;
        Ok(patch_fields(py, &patch)?)
    }

    /// The annotation a draw takes foreground from, auto-selected if `None`.
    #[pyo3(signature = (sample, annotation=None, grid=None))]
    fn annotation(
        &self,
        sample: &Bound<'_, PyAny>,
        annotation: Option<String>,
        grid: Option<String>,
    ) -> R<Option<String>> {
        let sample = sample_arg(sample)?;
        Ok(self.inner.annotation(&sample, annotation.as_deref(), grid.as_deref())?)
    }

    /// The grid a draw is measured in.
    #[pyo3(signature = (sample, annotation=None, grid=None))]
    fn window_grid(&self, sample: &Bound<'_, PyAny>, annotation: Option<String>, grid: Option<String>) -> R<String> {
        let sample = sample_arg(sample)?;
        Ok(self.inner.window_grid(&sample, annotation.as_deref(), grid.as_deref())?)
    }

    /// A class to sample from, weighted as configured; `None` when no class
    /// has foreground.
    #[pyo3(signature = (counts, rng=None))]
    fn pick_class<'py>(&self, counts: &Bound<'py, PyAny>, rng: Option<Bound<'py, PyAny>>) -> R<Option<i64>> {
        let mut map = IndexMap::new();
        for item in counts.call_method0("items")?.try_iter()? {
            let (k, v): (i64, i64) = item?.extract()?;
            map.insert(k, v);
        }
        let mut rng = AnyRng::from_arg(rng);
        Ok(self.inner.pick_class(&map, rng.as_dyn())?)
    }
}

/// Broadcast a patch size across `ndim` spatial axes.
#[pyfunction]
fn sampling_coerce_patch_size<'py>(
    py: Python<'py>,
    patch_size: &Bound<'py, PyAny>,
    ndim: usize,
) -> R<Bound<'py, PyTuple>> {
    let size = engine::coerce_patch_size(&patch_size_arg(patch_size)?, ndim)?;
    Ok(PyTuple::new(py, size)?)
}

/// `(slices, pad)` covering `patch` voxels around `center`.
#[pyfunction]
fn sampling_window_around<'py>(
    py: Python<'py>,
    center: Vec<i64>,
    patch: Vec<i64>,
    shape: Vec<i64>,
) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyTuple>)> {
    let (slices, pad) = engine::window_around(&center, &patch, &shape);
    Ok((slices_of(py, &slices)?, PyTuple::new(py, pad)?))
}

/// The sliding-window cover of a volume: a list of `Patch` field dicts.
#[pyfunction]
#[pyo3(signature = (shape, patch_size, *, overlap=0, grid_id=None))]
fn sampling_grid_patches<'py>(
    py: Python<'py>,
    shape: Vec<i64>,
    patch_size: &Bound<'py, PyAny>,
    overlap: i64,
    grid_id: Option<String>,
) -> R<Bound<'py, PyList>> {
    let patches = engine::grid_patches(&shape, &patch_size_arg(patch_size)?, overlap, grid_id.as_deref())?;
    let out = PyList::empty(py);
    for p in &patches {
        out.append(patch_fields(py, p)?)?;
    }
    Ok(out)
}

/// Refuse an unknown pair mode.
#[pyfunction]
fn sampling_check_pair_mode(mode: &str) -> R<()> {
    engine::TimepointPairSampler::new(mode)?;
    Ok(())
}

/// The visit pairs of a sample in `mode`: `(first, second, interval_days,
/// label)` each.
#[pyfunction]
fn sampling_pairs<'py>(py: Python<'py>, mode: &str, sample: &Bound<'py, PyAny>) -> R<Bound<'py, PyList>> {
    let sample = sample_arg(sample)?;
    let pairs = engine::TimepointPairSampler::new(mode)?.pairs(&sample)?;
    let out = PyList::empty(py);
    for p in pairs {
        out.append((p.first, p.second, p.interval_days, p.label))?;
    }
    Ok(out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PatchSamplerHandle>()?;
    for f in [
        wrap_pyfunction!(sampling_coerce_patch_size, m)?,
        wrap_pyfunction!(sampling_window_around, m)?,
        wrap_pyfunction!(sampling_grid_patches, m)?,
        wrap_pyfunction!(sampling_check_pair_mode, m)?,
        wrap_pyfunction!(sampling_pairs, m)?,
    ] {
        m.add_function(f)?;
    }
    let py = m.py();
    m.add("SAMPLING_STRATEGIES", PyTuple::new(py, engine::STRATEGIES)?)?;
    m.add("SAMPLING_PAIR_MODES", PyTuple::new(py, engine::PAIR_MODES)?)?;
    Ok(())
}
