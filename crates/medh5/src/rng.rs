//! Randomness behind patch sampling and index subsampling.
//!
//! Every draw goes through [`Rng`], so a frontend can supply its own
//! generator: the Python bindings hand the user's `numpy.random.Generator`
//! through, making the same calls in the same order as the 1.x Python
//! implementation, so a seeded draw is reproducible across the rewrite.  The
//! Rust SDK and the CLI use [`SeededRng`].

use rand::seq::SliceRandom;
use rand::{Rng as _, SeedableRng};
use rand_pcg::Pcg64;

use crate::Result;

/// The draws sampling needs, named after the NumPy `Generator` methods.
pub trait Rng {
    /// One integer in `[low, high)` (`Generator.integers(low, high)`).
    fn integer(&mut self, low: i64, high: i64) -> Result<i64>;
    /// `n` integers in `[low, high)` (`Generator.integers(low, high, size=n)`).
    fn integers(&mut self, low: i64, high: i64, n: usize) -> Result<Vec<i64>> {
        (0..n).map(|_| self.integer(low, high)).collect()
    }
    /// One float in `[0, 1)` (`Generator.random()`).
    fn random(&mut self) -> Result<f64>;
    /// One of `keys` with probabilities `p` (`Generator.choice(keys, p=p)`).
    fn choice_weighted(&mut self, keys: &[i64], p: &[f64]) -> Result<i64>;
    /// Permute in place (`Generator.shuffle(values)`).
    fn shuffle(&mut self, values: &mut Vec<i64>) -> Result<()>;
    /// `k` distinct values from `0..total`, in any order
    /// (`Generator.choice(total, size=k, replace=False)`).
    fn sample_without_replacement(&mut self, total: usize, k: usize) -> Result<Vec<usize>>;
}

/// A seeded PCG-64 generator.
pub struct SeededRng(Pcg64);

impl SeededRng {
    /// A generator from a 64-bit seed.
    pub fn new(seed: u64) -> SeededRng {
        SeededRng(Pcg64::seed_from_u64(seed))
    }

    /// A generator seeded from the operating system.
    pub fn from_entropy() -> SeededRng {
        SeededRng(Pcg64::from_os_rng())
    }
}

impl Rng for SeededRng {
    fn integer(&mut self, low: i64, high: i64) -> Result<i64> {
        if high <= low {
            return Err(crate::Error::Value("high <= 0".into()));
        }
        Ok(self.0.random_range(low..high))
    }

    fn random(&mut self) -> Result<f64> {
        Ok(self.0.random::<f64>())
    }

    fn choice_weighted(&mut self, keys: &[i64], p: &[f64]) -> Result<i64> {
        let total: f64 = p.iter().sum();
        let mut x = self.0.random::<f64>() * total;
        for (k, w) in keys.iter().zip(p) {
            if x < *w {
                return Ok(*k);
            }
            x -= w;
        }
        keys.last().copied().ok_or_else(|| crate::Error::Value("a cannot be empty unless no samples are taken".into()))
    }

    fn shuffle(&mut self, values: &mut Vec<i64>) -> Result<()> {
        values.shuffle(&mut self.0);
        Ok(())
    }

    fn sample_without_replacement(&mut self, total: usize, k: usize) -> Result<Vec<usize>> {
        Ok(rand::seq::index::sample(&mut self.0, total, k.min(total)).into_vec())
    }
}
