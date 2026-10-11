//! Reductions that round the way NumPy's do.
//!
//! A statistic the engine writes into a file or a report has to be the number
//! the 1.x implementation wrote, to the last bit: two implementations of one
//! format that disagree in the 16th digit produce manifests whose digests
//! differ.  NumPy sums floating-point arrays *pairwise* (blocks of eight
//! accumulators, recursive halving above 128 elements), which is not the
//! left-to-right sum a plain loop computes, so the reductions here follow
//! NumPy's `pairwise_sum` exactly.

/// One floating-point element type NumPy sums natively.
pub trait Float: Copy + std::ops::Add<Output = Self> {
    const ZERO: Self;
}

impl Float for f64 {
    const ZERO: f64 = 0.0;
}

impl Float for f32 {
    const ZERO: f32 = 0.0;
}

const PW_BLOCKSIZE: usize = 128;

/// NumPy's `pairwise_sum` over a contiguous run.
pub fn pairwise_sum<T: Float>(a: &[T]) -> T {
    let n = a.len();
    if n < 8 {
        let mut res = T::ZERO;
        for v in a {
            res = res + *v;
        }
        res
    } else if n <= PW_BLOCKSIZE {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - (n % 8) {
            for (j, acc) in r.iter_mut().enumerate() {
                *acc = *acc + a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res = res + a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

/// `np.sum` of a 1-D float64 array: the identity plus the pairwise sum.
pub fn sum(a: &[f64]) -> f64 {
    0.0 + pairwise_sum(a)
}

/// `float(np.mean(a))` of a 1-D float64 array; `None` when empty.
pub fn mean(a: &[f64]) -> Option<f64> {
    if a.is_empty() {
        None
    } else {
        Some(sum(a) / a.len() as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sums_round_the_way_numpy_does() {
        // Values NumPy 2.x printed for `np.sum` / `np.mean` of 1/(k+3); a
        // left-to-right loop differs from each of them in the last digits.
        let series = |n: u32| -> Vec<f64> { (0..n).map(|k| 1.0 / f64::from(k + 3)).collect() };
        assert_eq!(sum(&series(9)), 1.5198773448773446);
        assert_eq!(sum(&series(17)), 2.047739657143682);
        assert_eq!(sum(&series(130)), 3.9638006836279933);
        assert_eq!(sum(&series(1000)), 5.987467865541362);
        assert_eq!(mean(&series(1000)), Some(0.0059874678655413615));
        assert_eq!(mean(&[]), None);
    }
}
