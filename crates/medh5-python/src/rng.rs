//! The `rng=` argument: what seeds the engine's generator.
//!
//! The engine draws from its own generator ([`medh5::rng::Rng`]); a caller
//! names the seed.  A `numpy.random.Generator` supplies one seed and advances,
//! so a seeded pipeline stays reproducible and a shared generator still gives
//! every call different draws --- without a call back into Python per draw.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use medh5::rng::Rng;

/// The generator an `rng=` argument names: `None` for fresh entropy, an
/// integer or a sequence of integers as the seed, or a
/// `numpy.random.Generator`, which supplies the seed.
pub fn rng_arg(rng: Option<&Bound<'_, PyAny>>) -> PyResult<Rng> {
    let Some(obj) = rng.filter(|r| !r.is_none()) else {
        return Ok(Rng::from_entropy());
    };
    if let Ok(seed) = obj.extract::<i128>() {
        return Ok(Rng::new(seed as u64));
    }
    if obj.hasattr("integers")? {
        let seed: i64 = obj.call_method1("integers", (0i64, i64::MAX))?.extract()?;
        return Ok(Rng::new(seed as u64));
    }
    if let Ok(seeds) = obj.extract::<Vec<i128>>() {
        return Ok(Rng::from_seeds(&seeds.into_iter().map(|s| s as u64).collect::<Vec<_>>()));
    }
    Err(PyTypeError::new_err(format!(
        "rng must be None, an int, a sequence of ints or a numpy.random.Generator, not {}",
        obj.get_type().name()?
    )))
}
