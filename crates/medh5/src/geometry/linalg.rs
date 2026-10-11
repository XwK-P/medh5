//! Small dense linear algebra for affines and rotations.
//!
//! Matrices here are at most 4x4; clarity beats a dependency.

use ndarray::Array2;

use crate::{Error, Result};

/// `a @ b`.
pub fn matmul(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    a.dot(b)
}

/// The identity matrix.
pub fn eye(n: usize) -> Array2<f64> {
    Array2::eye(n)
}

/// LU factorisation with partial pivoting, as LAPACK's `dgetf2` computes it:
/// the pivot is the first entry of largest magnitude, the column below it is
/// scaled by the pivot's reciprocal, then a rank-1 update.  Returns the packed
/// factors, the pivot rows, and whether a pivot was exactly zero.
pub fn lu(m: &Array2<f64>) -> (Array2<f64>, Vec<usize>, bool) {
    let n = m.nrows();
    let mut a = m.clone();
    let mut pivots = Vec::with_capacity(n);
    let mut singular = false;
    let sfmin = f64::MIN_POSITIVE;
    for j in 0..n {
        let mut p = j;
        let mut best = a[[j, j]].abs();
        for row in j + 1..n {
            if a[[row, j]].abs() > best {
                best = a[[row, j]].abs();
                p = row;
            }
        }
        pivots.push(p);
        if a[[p, j]] != 0.0 {
            if p != j {
                for k in 0..n {
                    a.swap([j, k], [p, k]);
                }
            }
            let d = a[[j, j]];
            if d.abs() >= sfmin {
                let r = 1.0 / d;
                for row in j + 1..n {
                    a[[row, j]] *= r;
                }
            } else {
                for row in j + 1..n {
                    a[[row, j]] /= d;
                }
            }
        } else {
            singular = true;
        }
        for k in j + 1..n {
            let t = -a[[j, k]];
            if t != 0.0 {
                for row in j + 1..n {
                    a[[row, k]] += a[[row, j]] * t;
                }
            }
        }
    }
    (a, pivots, singular)
}

/// Determinant as NumPy computes it: `sign * exp(sum(log|u_ii|))` over the
/// LU factors, so `det(diag(2, 2, 2))` is `7.999999999999998`, not `8.0`.
pub fn det(m: &Array2<f64>) -> f64 {
    let n = m.nrows();
    if n != m.ncols() {
        return f64::NAN;
    }
    if n == 0 {
        return 1.0;
    }
    let (a, pivots, singular) = lu(m);
    if singular {
        return 0.0;
    }
    let mut sign = 1.0f64;
    for (j, p) in pivots.iter().enumerate() {
        if *p != j {
            sign = -sign;
        }
    }
    let mut logdet = 0.0f64;
    for j in 0..n {
        let mut d = a[[j, j]];
        if d < 0.0 {
            sign = -sign;
            d = -d;
        }
        logdet += d.ln();
    }
    sign * logdet.exp()
}

/// Inverse as NumPy computes it: LU factors, then a forward and a back
/// substitution against the identity (`dgesv`).
pub fn inv(m: &Array2<f64>) -> Result<Array2<f64>> {
    let n = m.nrows();
    if n != m.ncols() {
        return Err(Error::Value("Last 2 dimensions of the array must be square".into()));
    }
    let (a, pivots, singular) = lu(m);
    if singular || a.iter().any(|v| !v.is_finite()) {
        return Err(Error::Value("Singular matrix".into()));
    }
    let mut b = eye(n);
    for (j, p) in pivots.iter().enumerate() {
        if *p != j {
            for k in 0..n {
                b.swap([j, k], [*p, k]);
            }
        }
    }
    for col in 0..n {
        // L y = b, unit lower.
        for k in 0..n {
            let v = b[[k, col]];
            if v != 0.0 {
                for i in k + 1..n {
                    b[[i, col]] -= v * a[[i, k]];
                }
            }
        }
        // U x = y.
        for k in (0..n).rev() {
            if b[[k, col]] != 0.0 {
                b[[k, col]] /= a[[k, k]];
                let v = b[[k, col]];
                for i in 0..k {
                    b[[i, col]] -= v * a[[i, k]];
                }
            }
        }
    }
    Ok(b)
}

/// `np.allclose(a, b, atol=atol, rtol=rtol)` for equal-shaped slices.
pub fn allclose(a: &[f64], b: &[f64], atol: f64, rtol: f64) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            if x.is_nan() || y.is_nan() {
                return false;
            }
            if x.is_infinite() || y.is_infinite() {
                return x == y;
            }
            (x - y).abs() <= atol + rtol * y.abs()
        })
}

/// `math.isclose(a, b, rel_tol, abs_tol)`.
pub fn isclose(a: f64, b: f64, rel_tol: f64, abs_tol: f64) -> bool {
    if a == b {
        return true;
    }
    if a.is_infinite() || b.is_infinite() {
        return false;
    }
    let diff = (a - b).abs();
    diff <= (rel_tol * b.abs()).max(rel_tol * a.abs()) || diff <= abs_tol
}

/// A matrix from row-major values.
pub fn from_rows(rows: usize, cols: usize, values: &[f64]) -> Result<Array2<f64>> {
    Array2::from_shape_vec((rows, cols), values.to_vec())
        .map_err(|e| Error::Value(format!("matrix of shape ({rows}, {cols}): {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_and_determinant() {
        let m = from_rows(3, 3, &[2.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 1.0]).unwrap();
        assert!((det(&m) - 3.0).abs() < 1e-12);
        let i = inv(&m).unwrap();
        let p = matmul(&m, &i);
        assert!(allclose(p.as_slice().unwrap(), eye(3).as_slice().unwrap(), 1e-12, 0.0));
        assert!(inv(&from_rows(2, 2, &[1.0, 2.0, 2.0, 4.0]).unwrap()).is_err());
    }

    #[test]
    fn determinant_and_inverse_match_numpy_bits() {
        // Reference values from NumPy 2.5 (`np.linalg.det`, `np.linalg.inv`).
        assert_eq!(det(&from_rows(3, 3, &[2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0]).unwrap()), 7.999999999999998);
        assert_eq!(det(&from_rows(2, 2, &[0.0, 1.0, 1.0, 0.0]).unwrap()), -1.0);
        let m = from_rows(3, 3, &[1.1, 0.2, 0.3, 0.05, 0.9, 0.1, 0.2, 0.1, 1.3]).unwrap();
        assert_eq!(det(&m), 1.2145000000000001);
        let i = inv(&m).unwrap();
        assert_eq!(i[[0, 0]], 0.9551255660765747);
        assert_eq!(i[[1, 2]], -0.07822149032523672);
        assert_eq!(i[[2, 1]], -0.05763688760806915);
    }
}
