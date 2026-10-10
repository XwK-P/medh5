//! The `rng=` argument: what seeds the engine's generator.
//!
//! The engine draws from its own generator ([`medh5::rng::Rng`]); a caller
//! names the seed.  A `numpy.random.Generator` supplies one seed and advances,
//! so a seeded pipeline stays reproducible and a shared generator still gives
//! every call different draws --- without a call back into Python per draw.

use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;

use medh5::rng::Rng;

/// The generator an `rng=` argument names: `None` for fresh entropy, an
/// integer or a sequence of integers as the seed, or a
/// `numpy.random.Generator`, which supplies the seed.
pub fn rng_arg(rng: Option<&Bound<'_, PyAny>>) -> PyResult<Rng> {
    let Some(obj) = rng.filter(|r| !r.is_none()) else {
        return Ok(Rng::from_entropy());
    };
    if let Some(seed) = seed(obj)? {
        return Ok(Rng::new(seed));
    }
    if obj.hasattr("integers")? {
        let seed: i64 = obj.call_method1("integers", (0i64, i64::MAX))?.extract()?;
        return Ok(Rng::new(seed as u64));
    }
    if let Ok(items) = obj.extract::<Vec<Bound<'_, PyAny>>>() {
        let seeds = items.iter().map(|s| seed(s)?.ok_or_else(|| not_a_seed(obj))).collect::<PyResult<Vec<_>>>()?;
        return Ok(Rng::from_seeds(&seeds));
    }
    Err(not_a_seed(obj))
}

/// One seed, an integer from 0 to 2**64 - 1, or `None` for what is not an
/// integer.  The engine seeds from 64-bit words: a larger integer would wrap
/// onto a smaller one and repeat its draws, and NumPy refuses a negative
/// seed too, so both are refused rather than folded.
fn seed(obj: &Bound<'_, PyAny>) -> PyResult<Option<u64>> {
    let refused = || {
        let shown = obj.repr().map(|r| r.to_string()).unwrap_or_else(|_| "this".into());
        PyValueError::new_err(format!("a seed is an integer from 0 to 2**64 - 1, not {shown}"))
    };
    match obj.extract::<i128>() {
        Ok(value) => u64::try_from(value).map(Some).map_err(|_| refused()),
        Err(e) if e.is_instance_of::<PyOverflowError>(obj.py()) => Err(refused()),
        Err(_) => Ok(None),
    }
}

fn not_a_seed(obj: &Bound<'_, PyAny>) -> PyErr {
    let kind = obj.get_type().name().map(|n| n.to_string()).unwrap_or_else(|_| "this".into());
    PyTypeError::new_err(format!(
        "rng must be None, an int, a sequence of ints or a numpy.random.Generator, not {kind}"
    ))
}
