//! Randomness behind patch sampling, index subsampling and synthetic data.
//!
//! One small generator, [`Rng`]: PCG-64 (XSL-RR 128/64), seeded from one or
//! more integers through SplitMix64, with the few draws the engine needs ---
//! unbiased bounded integers (Lemire's method), 53-bit floats, weighted
//! choice, Fisher--Yates shuffles and sampling without replacement.  Every
//! frontend draws from it, so a seed means the same draws from Rust, from the
//! command line and from Python; a Python caller's `numpy.random.Generator`
//! only supplies the seed.
//!
//! The streams are not NumPy's.  1.x drew from NumPy; nothing in a file
//! depends on which generator chose a patch, and no draw is part of the
//! format --- the sampling index records the seed it was built with, not the
//! generator.

use crate::{Error, Result};

const MULTIPLIER: u128 = (2549297995355413924u128 << 64) | 4865540595714422341u128;

/// SplitMix64: advance `state` and return a well-mixed 64-bit value.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A seeded pseudo-random generator.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u128,
    inc: u128,
}

impl Rng {
    /// A generator from one seed.
    pub fn new(seed: u64) -> Rng {
        Rng::from_seeds(&[seed])
    }

    /// A generator from several integers --- `(seed, epoch, index)` --- each of
    /// which changes every draw; their order and number matter too.
    pub fn from_seeds(seeds: &[u64]) -> Rng {
        let mut h = seeds.len() as u64;
        for &s in seeds {
            let mut x = h ^ s;
            h = splitmix64(&mut x);
        }
        let mut words = || splitmix64(&mut h);
        let state = (u128::from(words()) << 64) | u128::from(words());
        let inc = (u128::from(words()) << 64) | u128::from(words()) | 1;
        let mut rng = Rng { state: 0, inc };
        rng.step();
        rng.state = rng.state.wrapping_add(state);
        rng.step();
        rng
    }

    /// A generator seeded from the operating system's randomness.
    pub fn from_entropy() -> Rng {
        use std::collections::hash_map::RandomState;
        use std::hash::{BuildHasher, Hasher};
        // `RandomState` keys SipHash from the operating system's generator:
        // the one portable source the standard library has.
        let word = || {
            let mut h = RandomState::new().build_hasher();
            h.write_u128(
                std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos()),
            );
            h.finish()
        };
        Rng::from_seeds(&[word(), word()])
    }

    fn step(&mut self) {
        self.state = self.state.wrapping_mul(MULTIPLIER).wrapping_add(self.inc);
    }

    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let state = self.state;
        let xored = ((state >> 64) as u64) ^ (state as u64);
        xored.rotate_right((state >> 122) as u32)
    }

    /// The next 32 random bits.
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// A float in `[0, 1)` with 53 random bits.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// A float in `[0, 1)`.
    pub fn random(&mut self) -> f64 {
        self.next_f64()
    }

    /// A value in `[0, n)`, unbiased (Lemire's multiply-and-reject); `n > 0`.
    fn below(&mut self, n: u64) -> u64 {
        let mut m = u128::from(self.next_u64()) * u128::from(n);
        if (m as u64) < n {
            let threshold = n.wrapping_neg() % n;
            while (m as u64) < threshold {
                m = u128::from(self.next_u64()) * u128::from(n);
            }
        }
        (m >> 64) as u64
    }

    /// One integer in `[low, high)`.
    pub fn integer(&mut self, low: i64, high: i64) -> Result<i64> {
        if high <= low {
            return Err(Error::Value(format!("an empty range: [{low}, {high})")));
        }
        let span = (i128::from(high) - i128::from(low)) as u64;
        Ok((i128::from(low) + i128::from(self.below(span))) as i64)
    }

    /// `n` integers in `[low, high)`.
    pub fn integers(&mut self, low: i64, high: i64, n: usize) -> Result<Vec<i64>> {
        (0..n).map(|_| self.integer(low, high)).collect()
    }

    /// One of `keys`, chosen with the relative weights `p`.
    pub fn choice_weighted(&mut self, keys: &[i64], p: &[f64]) -> Result<i64> {
        if keys.is_empty() || keys.len() != p.len() {
            return Err(Error::Value(format!("{} keys for {} weights", keys.len(), p.len())));
        }
        if p.iter().any(|w| !w.is_finite() || *w < 0.0) {
            return Err(Error::Value("weights must be finite and non-negative".into()));
        }
        let total: f64 = p.iter().sum();
        if total <= 0.0 {
            return Err(Error::Value("weights must not all be zero".into()));
        }
        let target = self.next_f64() * total;
        let mut acc = 0.0;
        for (key, w) in keys.iter().zip(p) {
            acc += w;
            if target < acc {
                return Ok(*key);
            }
        }
        // Rounding left `target` at the very top: the last key with weight.
        Ok(*keys.iter().zip(p).rev().find(|(_, w)| **w > 0.0).map(|(k, _)| k).expect("a positive weight"))
    }

    /// Permute `values` in place (Fisher--Yates).
    pub fn shuffle<T>(&mut self, values: &mut [T]) {
        for i in (1..values.len()).rev() {
            let j = self.below(i as u64 + 1) as usize;
            values.swap(i, j);
        }
    }

    /// `k` distinct values from `0..total`, in random order.
    ///
    /// Floyd's algorithm when `k` is small next to `total`, so the cost is
    /// `O(k)` however large the population; a partial shuffle otherwise.
    pub fn sample_without_replacement(&mut self, total: usize, k: usize) -> Result<Vec<usize>> {
        if k > total {
            return Err(Error::Value(format!("cannot take {k} distinct values from {total}")));
        }
        if k.saturating_mul(4) >= total {
            let mut all: Vec<usize> = (0..total).collect();
            for i in 0..k {
                let j = i + self.below((total - i) as u64) as usize;
                all.swap(i, j);
            }
            all.truncate(k);
            return Ok(all);
        }
        let mut seen = std::collections::HashSet::with_capacity(k);
        let mut out = Vec::with_capacity(k);
        for j in (total - k)..total {
            let t = self.below(j as u64 + 1) as usize;
            let pick = if seen.insert(t) { t } else { j };
            seen.insert(pick);
            out.push(pick);
        }
        self.shuffle(&mut out);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_seed_is_a_stream_and_seeds_differ() {
        let draws = |seeds: &[u64]| {
            let mut g = Rng::from_seeds(seeds);
            (0..8).map(|_| g.next_u64()).collect::<Vec<_>>()
        };
        assert_eq!(draws(&[7]), draws(&[7]));
        assert_eq!(Rng::new(7).next_u64(), Rng::from_seeds(&[7]).next_u64());
        for (a, b) in [(&[7u64][..], &[8u64][..]), (&[1, 2], &[2, 1]), (&[0], &[0, 0]), (&[7, 0, 1], &[7, 0, 2])] {
            assert_ne!(draws(a), draws(b), "{a:?} vs {b:?}");
        }
        assert_ne!(Rng::from_entropy().next_u64(), Rng::from_entropy().next_u64());
    }

    #[test]
    fn integers_are_in_range_and_unbiased() {
        let mut g = Rng::new(1);
        let mut counts = [0usize; 6];
        for _ in 0..60_000 {
            let v = g.integer(-3, 3).unwrap();
            assert!((-3..3).contains(&v));
            counts[(v + 3) as usize] += 1;
        }
        // Each face within 5% of 10 000 (about 10 standard deviations).
        assert!(counts.iter().all(|c| (9_500..10_500).contains(c)), "{counts:?}");
        assert!(g.integer(i64::MIN, i64::MAX).is_ok());
        assert!(g.integer(5, 5).is_err());
        let f = g.random();
        assert!((0.0..1.0).contains(&f));
    }

    #[test]
    fn weighted_choice_follows_the_weights() {
        let mut g = Rng::new(2);
        let mut counts = [0usize; 3];
        for _ in 0..30_000 {
            let k = g.choice_weighted(&[10, 20, 30], &[1.0, 0.0, 3.0]).unwrap();
            counts[(k / 10 - 1) as usize] += 1;
        }
        assert_eq!(counts[1], 0);
        assert!((7_000..8_000).contains(&counts[0]) && (22_000..23_000).contains(&counts[2]), "{counts:?}");
        assert!(g.choice_weighted(&[1], &[0.0]).is_err());
        assert!(g.choice_weighted(&[1, 2], &[1.0]).is_err());
        assert!(g.choice_weighted(&[1], &[f64::NAN]).is_err());
    }

    #[test]
    fn sampling_without_replacement_is_distinct_in_both_regimes() {
        let mut g = Rng::new(3);
        for (total, k) in [(50, 10), (50, 50), (100_000, 3000), (10, 0)] {
            let picks = g.sample_without_replacement(total, k).unwrap();
            let distinct: std::collections::HashSet<_> = picks.iter().collect();
            assert_eq!((picks.len(), distinct.len()), (k, k));
            assert!(picks.iter().all(|p| *p < total));
        }
        assert!(g.sample_without_replacement(3, 4).is_err());
        // Every value equally likely: 20 000 draws of 2 from 4.
        let mut counts = [0usize; 4];
        for _ in 0..20_000 {
            for p in g.sample_without_replacement(4, 2).unwrap() {
                counts[p] += 1;
            }
        }
        assert!(counts.iter().all(|c| (9_500..10_500).contains(c)), "{counts:?}");
        let mut order: Vec<i64> = (0..10).collect();
        g.shuffle(&mut order);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    }
}
