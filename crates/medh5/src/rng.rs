//! Randomness behind patch sampling and index subsampling.
//!
//! Every draw goes through [`Rng`], so a frontend can supply its own
//! generator: the Python bindings hand the user's `numpy.random.Generator`
//! through, making the same calls in the same order as the 1.x Python
//! implementation.
//!
//! [`SeededRng`] is NumPy's `default_rng(seed)` reimplemented bit for bit ---
//! `SeedSequence` seeding, the PCG-64 (XSL-RR) generator, and the bounded
//! integer, float, shuffle and choice algorithms of `numpy.random.Generator`.
//! So `medh5 index build --seed 7`, `SampleWriter::build_index(.., 7)` from
//! Rust and `writer.build_index(seed=7)` from Python write the same index, and
//! the one 1.x wrote.

use crate::{Error, Result};

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
    /// `k` distinct values from `0..total`, in the generator's order
    /// (`Generator.choice(total, size=k, replace=False)`).
    fn sample_without_replacement(&mut self, total: usize, k: usize) -> Result<Vec<usize>>;
}

const MULTIPLIER: u128 = (2549297995355413924u128 << 64) | 4865540595714422341u128;

const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const XSHIFT: u32 = 16;
const POOL_SIZE: usize = 4;

fn hashmix(value: u32, hash_const: &mut u32) -> u32 {
    let mut value = value ^ *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    value = value.wrapping_mul(*hash_const);
    value ^ (value >> XSHIFT)
}

fn mix(x: u32, y: u32) -> u32 {
    let result = MIX_MULT_L.wrapping_mul(x).wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    result ^ (result >> XSHIFT)
}

/// NumPy's `SeedSequence(entropy).generate_state(n_words)` (`uint32`).
fn seed_sequence_state(entropy: &[u32], n_words: usize) -> Vec<u32> {
    let mut pool = [0u32; POOL_SIZE];
    let mut hash_const = INIT_A;
    for (i, slot) in pool.iter_mut().enumerate() {
        *slot = hashmix(entropy.get(i).copied().unwrap_or(0), &mut hash_const);
    }
    for i_src in 0..POOL_SIZE {
        for i_dst in 0..POOL_SIZE {
            if i_src != i_dst {
                pool[i_dst] = mix(pool[i_dst], hashmix(pool[i_src], &mut hash_const));
            }
        }
    }
    for word in entropy.iter().skip(POOL_SIZE) {
        for slot in pool.iter_mut() {
            *slot = mix(*slot, hashmix(*word, &mut hash_const));
        }
    }
    let mut hash_const = INIT_B;
    (0..n_words)
        .map(|i| {
            let mut value = pool[i % POOL_SIZE] ^ hash_const;
            hash_const = hash_const.wrapping_mul(MULT_B);
            value = value.wrapping_mul(hash_const);
            value ^ (value >> XSHIFT)
        })
        .collect()
}

/// The 32-bit words of a non-negative integer, least significant first.
fn entropy_words(seed: u128) -> Vec<u32> {
    if seed == 0 {
        return vec![0];
    }
    let mut words = Vec::new();
    let mut n = seed;
    while n > 0 {
        words.push((n & 0xFFFF_FFFF) as u32);
        n >>= 32;
    }
    words
}

/// NumPy's `default_rng(seed)`: PCG-64 seeded through `SeedSequence`.
#[derive(Debug, Clone)]
pub struct SeededRng {
    state: u128,
    inc: u128,
    has_uint32: bool,
    uinteger: u32,
}

impl SeededRng {
    /// `numpy.random.default_rng(seed)`.
    pub fn new(seed: u64) -> SeededRng {
        SeededRng::from_entropy_value(u128::from(seed))
    }

    /// A generator from a seed of up to 128 bits.
    pub fn from_entropy_value(seed: u128) -> SeededRng {
        let words = seed_sequence_state(&entropy_words(seed), 8);
        let word = |i: usize| u64::from(words[2 * i]) | (u64::from(words[2 * i + 1]) << 32);
        let initstate = (u128::from(word(0)) << 64) | u128::from(word(1));
        let initseq = (u128::from(word(2)) << 64) | u128::from(word(3));
        let mut rng = SeededRng { state: 0, inc: (initseq << 1) | 1, has_uint32: false, uinteger: 0 };
        rng.step();
        rng.state = rng.state.wrapping_add(initstate);
        rng.step();
        rng
    }

    /// A generator seeded from the operating system, as `default_rng()` is.
    pub fn from_entropy() -> SeededRng {
        let mut bytes = [0u8; 16];
        if getrandom_fill(&mut bytes).is_err() {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default();
            bytes = (nanos ^ (u128::from(std::process::id()) << 64)).to_le_bytes();
        }
        SeededRng::from_entropy_value(u128::from_le_bytes(bytes))
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

    /// The next 32 random bits; halves of one 64-bit draw, low half first.
    pub fn next_u32(&mut self) -> u32 {
        if self.has_uint32 {
            self.has_uint32 = false;
            return self.uinteger;
        }
        let next = self.next_u64();
        self.has_uint32 = true;
        self.uinteger = (next >> 32) as u32;
        (next & 0xFFFF_FFFF) as u32
    }

    /// A float in `[0, 1)` with 53 random bits.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }

    fn lemire32(&mut self, rng: u32) -> u32 {
        let rng_excl = rng.wrapping_add(1);
        let mut m = u64::from(self.next_u32()) * u64::from(rng_excl);
        let mut leftover = (m & 0xFFFF_FFFF) as u32;
        if leftover < rng_excl {
            let threshold = (u32::MAX - rng) % rng_excl;
            while leftover < threshold {
                m = u64::from(self.next_u32()) * u64::from(rng_excl);
                leftover = (m & 0xFFFF_FFFF) as u32;
            }
        }
        (m >> 32) as u32
    }

    fn lemire64(&mut self, rng: u64) -> u64 {
        let rng_excl = rng.wrapping_add(1);
        let mut m = u128::from(self.next_u64()) * u128::from(rng_excl);
        let mut leftover = m as u64;
        if leftover < rng_excl {
            let threshold = (u64::MAX - rng) % rng_excl;
            while leftover < threshold {
                m = u128::from(self.next_u64()) * u128::from(rng_excl);
                leftover = m as u64;
            }
        }
        (m >> 64) as u64
    }

    /// `random_bounded_uint64(off, rng)`: a value in `[off, off + rng]`.
    fn bounded(&mut self, off: u64, rng: u64) -> u64 {
        if rng == 0 {
            off
        } else if rng <= 0xFFFF_FFFF {
            if rng == 0xFFFF_FFFF {
                off.wrapping_add(u64::from(self.next_u32()))
            } else {
                off.wrapping_add(u64::from(self.lemire32(rng as u32)))
            }
        } else if rng == u64::MAX {
            off.wrapping_add(self.next_u64())
        } else {
            off.wrapping_add(self.lemire64(rng))
        }
    }

    /// `random_interval(max)`: a value in `[0, max]` by masked rejection.
    fn interval(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        let mut mask = max;
        for shift in [1, 2, 4, 8, 16, 32] {
            mask |= mask >> shift;
        }
        if max <= 0xFFFF_FFFF {
            loop {
                let value = u64::from(self.next_u32()) & mask;
                if value <= max {
                    return value;
                }
            }
        }
        loop {
            let value = self.next_u64() & mask;
            if value <= max {
                return value;
            }
        }
    }
}

fn getrandom_fill(bytes: &mut [u8]) -> std::io::Result<()> {
    use std::io::Read;
    std::fs::File::open("/dev/urandom")?.read_exact(bytes)
}

impl Rng for SeededRng {
    fn integer(&mut self, low: i64, high: i64) -> Result<i64> {
        if high <= low {
            return Err(Error::Value("low >= high".into()));
        }
        let span = (high as i128 - 1 - low as i128) as u64;
        Ok(self.bounded(low as u64, span) as i64)
    }

    fn random(&mut self) -> Result<f64> {
        Ok(self.next_f64())
    }

    fn choice_weighted(&mut self, keys: &[i64], p: &[f64]) -> Result<i64> {
        if keys.is_empty() {
            return Err(Error::Value("a cannot be empty unless no samples are taken".into()));
        }
        if keys.len() != p.len() {
            return Err(Error::Value("a and p must have same size".into()));
        }
        if p.iter().any(|v| v.is_nan()) {
            return Err(Error::Value("Probabilities contain NaN".into()));
        }
        if p.iter().any(|v| *v < 0.0) {
            return Err(Error::Value("probabilities are not non-negative".into()));
        }
        if (kahan_sum(p) - 1.0).abs() > f64::EPSILON.sqrt() {
            return Err(Error::Value("probabilities do not sum to 1".into()));
        }
        let mut cdf: Vec<f64> = Vec::with_capacity(p.len());
        let mut acc = 0.0;
        for v in p {
            acc += v;
            cdf.push(acc);
        }
        let last = cdf[cdf.len() - 1];
        for v in cdf.iter_mut() {
            *v /= last;
        }
        let u = self.next_f64();
        let idx = cdf.partition_point(|c| *c <= u);
        Ok(keys[idx.min(keys.len() - 1)])
    }

    fn shuffle(&mut self, values: &mut Vec<i64>) -> Result<()> {
        for i in (1..values.len()).rev() {
            let j = self.interval(i as u64) as usize;
            values.swap(i, j);
        }
        Ok(())
    }

    fn sample_without_replacement(&mut self, total: usize, k: usize) -> Result<Vec<usize>> {
        if k > total {
            return Err(Error::Value("Cannot take a larger sample than population when replace is False".into()));
        }
        let (pop, size) = (total as u64, k as u64);
        let mut idx: Vec<u64>;
        if pop > 10000 && size > pop / 50 {
            // Tail shuffle: the last `size` places of a partial shuffle.
            idx = (0..pop).collect();
            let first = (pop - size).max(1);
            for i in (first..pop).rev() {
                let j = self.bounded(0, i) as usize;
                idx.swap(i as usize, j);
            }
            idx = idx.split_off((pop - size) as usize);
        } else {
            // Floyd's algorithm over an open-addressed hash set.
            idx = vec![0; k];
            let set_size = (1.2 * size as f64) as u64;
            let mask = gen_mask(set_size);
            let mut hash_set = vec![u64::MAX; (mask + 1) as usize];
            for j in (pop - size)..pop {
                let val = self.bounded(0, j);
                let mut loc = (val & mask) as usize;
                while hash_set[loc] != u64::MAX && hash_set[loc] != val {
                    loc = (loc + 1) & mask as usize;
                }
                let slot = (j + size - pop) as usize;
                if hash_set[loc] == u64::MAX {
                    hash_set[loc] = val;
                    idx[slot] = val;
                } else {
                    let mut loc = (j & mask) as usize;
                    while hash_set[loc] != u64::MAX {
                        loc = (loc + 1) & mask as usize;
                    }
                    hash_set[loc] = j;
                    idx[slot] = j;
                }
            }
            for i in (1..idx.len()).rev() {
                let j = self.bounded(0, i as u64) as usize;
                idx.swap(i, j);
            }
        }
        Ok(idx.into_iter().map(|v| v as usize).collect())
    }
}

/// The smallest all-ones mask covering `max`.
fn gen_mask(max: u64) -> u64 {
    let mut mask = max;
    for shift in [1, 2, 4, 8, 16, 32] {
        mask |= mask >> shift;
    }
    mask
}

/// NumPy's `kahan_sum`, which `choice` checks probabilities with.
fn kahan_sum(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sum = values[0];
    let mut c = 0.0;
    for v in &values[1..] {
        let y = v - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every expected value below was printed by NumPy 2.x for the same calls
    // on `np.random.default_rng(0)` (and the seeds named).
    #[test]
    fn draws_are_numpys() {
        let mut g = SeededRng::new(0);
        let randoms: Vec<f64> = (0..5).map(|_| g.random().unwrap()).collect();
        assert_eq!(
            randoms,
            [0.6369616873214543, 0.2697867137638703, 0.04097352393619469, 0.016527635528529094, 0.8132702392002724]
        );
        assert_eq!(g.integers(0, 10, 20).unwrap(), [6, 9, 5, 6, 9, 7, 6, 5, 5, 9, 2, 8, 6, 0, 3, 8, 5, 0, 7, 7]);
        let big: Vec<i64> = (0..5).map(|_| g.integer(0, 1 << 40).unwrap()).collect();
        assert_eq!(big, [193135397336, 949075261974, 595342907653, 329536708628, 464749514619]);
        let neg: Vec<i64> = (0..5).map(|_| g.integer(-1000, 1500).unwrap()).collect();
        assert_eq!(neg, [8, -930, -987, -690, -980]);
        let mut order: Vec<i64> = (0..10).collect();
        g.shuffle(&mut order).unwrap();
        assert_eq!(order, [7, 6, 4, 8, 2, 3, 5, 0, 1, 9]);
        let picks: Vec<i64> = (0..8).map(|_| g.choice_weighted(&[3, 5, 9], &[0.2, 0.3, 0.5]).unwrap()).collect();
        assert_eq!(picks, [9, 9, 5, 3, 9, 9, 5, 5]);
        let mut small = g.sample_without_replacement(50, 10).unwrap();
        small.sort();
        assert_eq!(small, [3, 12, 16, 24, 28, 30, 32, 38, 42, 48]);
        let mut big = g.sample_without_replacement(100000, 3000).unwrap();
        big.sort();
        assert_eq!(
            &big[..20],
            [18, 21, 29, 67, 186, 187, 229, 237, 244, 263, 281, 288, 351, 364, 375, 429, 458, 486, 517, 533]
        );
        let total: usize = g.sample_without_replacement(100000, 3000).unwrap().iter().sum();
        assert_eq!(total, 147210548);
        assert_eq!(g.random().unwrap(), 0.5720708470599539);
    }

    #[test]
    fn seeds_of_any_width_seed_as_numpy_does() {
        let mut g = SeededRng::new(20260815);
        assert_eq!(g.random().unwrap(), 0.5267784068878942);
        assert_eq!(g.integer(0, 7).unwrap(), 5);
        let mut wide = SeededRng::from_entropy_value((1u128 << 70) + 12345);
        assert_eq!(wide.random().unwrap(), 0.23007034884505184);
    }
}
