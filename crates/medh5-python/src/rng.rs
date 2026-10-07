//! A NumPy `Generator` as the engine's [`Rng`](medh5::rng::Rng).
//!
//! The engine makes the same draws, in the same order, as the 1.x Python
//! implementation; handing it the caller's generator through these calls
//! keeps a seeded pipeline bit-for-bit reproducible across the port.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use medh5::rng::{Rng, SeededRng};

fn engine_error(e: PyErr) -> medh5::Error {
    medh5::Error::Runtime(format!("the random generator failed: {e}"))
}

/// Calls a `numpy.random.Generator`.
pub struct PyRng<'py> {
    generator: Bound<'py, PyAny>,
}

impl<'py> PyRng<'py> {
    pub fn new(generator: Bound<'py, PyAny>) -> Self {
        PyRng { generator }
    }

    fn call<A: pyo3::call::PyCallArgs<'py>>(
        &self,
        name: &str,
        args: A,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.generator.call_method(name, args, kwargs)
    }
}

impl Rng for PyRng<'_> {
    fn integer(&mut self, low: i64, high: i64) -> medh5::Result<i64> {
        self.call("integers", (low, high), None).and_then(|v| v.extract::<i64>()).map_err(engine_error)
    }

    fn integers(&mut self, low: i64, high: i64, n: usize) -> medh5::Result<Vec<i64>> {
        let py = self.generator.py();
        (|| -> PyResult<Vec<i64>> {
            let kwargs = PyDict::new(py);
            kwargs.set_item("size", n)?;
            self.call("integers", (low, high), Some(&kwargs))?.call_method0("tolist")?.extract()
        })()
        .map_err(engine_error)
    }

    fn random(&mut self) -> medh5::Result<f64> {
        self.generator.call_method0("random").and_then(|v| v.extract::<f64>()).map_err(engine_error)
    }

    fn choice_weighted(&mut self, keys: &[i64], p: &[f64]) -> medh5::Result<i64> {
        let py = self.generator.py();
        (|| -> PyResult<i64> {
            let kwargs = PyDict::new(py);
            kwargs.set_item("p", PyList::new(py, p)?)?;
            self.call("choice", (PyList::new(py, keys)?,), Some(&kwargs))?.extract()
        })()
        .map_err(engine_error)
    }

    fn shuffle(&mut self, values: &mut Vec<i64>) -> medh5::Result<()> {
        let py = self.generator.py();
        let shuffled = (|| -> PyResult<Vec<i64>> {
            let array = crate::convert::numpy(py)?.call_method1("array", (PyList::new(py, values.iter())?,))?;
            self.generator.call_method1("shuffle", (&array,))?;
            array.call_method0("tolist")?.extract()
        })()
        .map_err(engine_error)?;
        *values = shuffled;
        Ok(())
    }

    fn sample_without_replacement(&mut self, total: usize, k: usize) -> medh5::Result<Vec<usize>> {
        let py = self.generator.py();
        (|| -> PyResult<Vec<usize>> {
            let kwargs = PyDict::new(py);
            kwargs.set_item("size", k)?;
            kwargs.set_item("replace", false)?;
            self.call("choice", (total,), Some(&kwargs))?.call_method0("tolist")?.extract()
        })()
        .map_err(engine_error)
    }
}

/// A generator for an optional `rng=` argument: the caller's, or fresh entropy.
pub enum AnyRng<'py> {
    Python(PyRng<'py>),
    Seeded(SeededRng),
}

impl<'py> AnyRng<'py> {
    pub fn from_arg(rng: Option<Bound<'py, PyAny>>) -> AnyRng<'py> {
        match rng.filter(|r| !r.is_none()) {
            Some(g) => AnyRng::Python(PyRng::new(g)),
            None => AnyRng::Seeded(SeededRng::from_entropy()),
        }
    }

    pub fn as_dyn(&mut self) -> &mut dyn Rng {
        match self {
            AnyRng::Python(p) => p,
            AnyRng::Seeded(s) => s,
        }
    }
}
