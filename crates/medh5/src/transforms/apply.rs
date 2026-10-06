//! Evaluating transforms: interpolation, Jacobians and registration error (§10).
//!
//! Interpolation is implemented here rather than delegated, because the two
//! things a reader needs from a displacement field --- its value between
//! samples and the determinant of its Jacobian --- must agree with each other.
//! Cubic sampling reproduces SciPy's `map_coordinates(order=3)` --- the same
//! prefilter, boundary handling and accumulation order --- so a file declaring
//! `interpolation = "cubic"` evaluates identically from every frontend.

use ndarray::{Array2, ArrayD, Axis, Dimension, IxDyn};

use crate::geometry::grid::Grid;
use crate::geometry::linalg::{det, inv};
use crate::json::repr_str;
use crate::{Error, Result};

/// The `extrapolation` values of a displacement field (§10.4).
pub const EXTRAPOLATIONS: [&str; 3] = ["zero", "nearest", "error"];

fn check_extrapolation(extrapolation: &str) -> Result<()> {
    if !EXTRAPOLATIONS.contains(&extrapolation) {
        return Err(Error::invalid(format!("unknown extrapolation {}", repr_str(extrapolation))));
    }
    Ok(())
}

/// Which `(N, S)` points lie within a lattice of `spatial` extent, edges included.
pub fn inside_extent(spatial: &[usize], points: &Array2<f64>) -> Vec<bool> {
    points.outer_iter().map(|p| p.iter().zip(spatial).all(|(v, n)| *v >= -0.5 && *v <= *n as f64 - 0.5)).collect()
}

/// `extrapolation = "error"` is a refusal, not a quieter fill value.
pub fn refuse_outside(inside: &[bool]) -> Result<()> {
    let outside = inside.iter().filter(|v| !**v).count();
    if outside == 0 {
        return Ok(());
    }
    Err(Error::invalid(format!("{outside} point(s) fall outside the field and extrapolation='error'")))
}

/// `(N, S)` view of `(..., S)` coordinates.
pub fn as_points(coords: &ArrayD<f64>, dim: usize) -> Result<Array2<f64>> {
    let flat: Vec<f64> = coords.iter().copied().collect();
    if dim == 0 || flat.len() % dim != 0 {
        return Err(Error::Value(format!("cannot reshape array of size {} into shape (-1, {dim})", flat.len())));
    }
    Ok(Array2::from_shape_vec((flat.len() / dim, dim), flat)?)
}

/// Multilinear interpolation of `(C, *spatial)` data at continuous indices.
///
/// `coords` is `(N, S)` in the field's continuous index coordinates.  Points
/// outside follow `extrapolation`: `zero`, `nearest` (clamp) or `error`.
pub fn linear_sample(field: &ArrayD<f64>, coords: &Array2<f64>, extrapolation: &str) -> Result<Array2<f64>> {
    let components = field.shape()[0];
    let spatial: Vec<usize> = field.shape()[1..].to_vec();
    let dim = spatial.len();
    check_extrapolation(extrapolation)?;
    let inside = inside_extent(&spatial, coords);
    if extrapolation == "error" {
        refuse_outside(&inside)?;
    }
    let n = coords.nrows();
    let mut out = Array2::<f64>::zeros((n, components));
    let mut base = vec![0i64; dim];
    let mut frac = vec![0f64; dim];
    let mut index = vec![0usize; dim + 1];
    for (row, point) in coords.outer_iter().enumerate() {
        for axis in 0..dim {
            let extent = spatial[axis] as i64;
            let clamped = point[axis].max(0.0).min(extent as f64 - 1.0);
            let b = (clamped.floor() as i64).min((extent - 2).max(0));
            base[axis] = b;
            frac[axis] = clamped - b as f64;
        }
        for corner in 0..(1usize << dim) {
            let mut weight = 1.0;
            let mut first = true;
            for axis in 0..dim {
                let w = if (corner >> axis) & 1 == 1 { frac[axis] } else { 1.0 - frac[axis] };
                weight = if first { w } else { weight * w };
                first = false;
                let offset = ((corner >> axis) & 1) as i64;
                index[axis + 1] = (base[axis] + offset).min(spatial[axis] as i64 - 1).max(0) as usize;
            }
            for c in 0..components {
                index[0] = c;
                out[[row, c]] += weight * field[IxDyn(&index)];
            }
        }
    }
    if extrapolation == "zero" {
        for (row, ok) in inside.iter().enumerate() {
            if !ok {
                out.row_mut(row).fill(0.0);
            }
        }
    }
    Ok(out)
}

// -- cubic: SciPy's `map_coordinates(order=3)` -------------------------------------

/// The cubic B-spline filter pole, `sqrt(3) - 2`.
const CUBIC_POLE: f64 = -0.267949192431122706472553658494127633;

#[derive(Clone, Copy, PartialEq)]
enum Boundary {
    /// `mode="constant"`: mirror-initialised prefilter, no interpolation outside.
    Constant,
    /// `mode="nearest"`: edge-padded by 12, reflect-initialised, clamped.
    Nearest,
}

fn init_causal_mirror(c: &mut [f64], z: f64) {
    let n = c.len();
    let mut z_i = z;
    let z_n_1 = z.powf((n - 1) as f64);
    c[0] += z_n_1 * c[n - 1];
    for i in 1..n - 1 {
        c[0] += z_i * (c[i] + z_n_1 * c[n - 1 - i]);
        z_i *= z;
    }
    c[0] /= 1.0 - z_n_1 * z_n_1;
}

fn init_anticausal_mirror(c: &mut [f64], z: f64) {
    let n = c.len();
    c[n - 1] = (z * c[n - 2] + c[n - 1]) * z / (z * z - 1.0);
}

fn init_causal_reflect(c: &mut [f64], z: f64) {
    let n = c.len();
    let mut z_i = z;
    let z_n = z.powf(n as f64);
    let mut sum = c[0] + z_n * c[n - 1];
    for i in 1..n {
        sum += z_i * (c[i] + z_n * c[n - 1 - i]);
        z_i *= z;
    }
    c[0] += sum * z / (1.0 - z_n * z_n);
}

fn init_anticausal_reflect(c: &mut [f64], z: f64) {
    let n = c.len();
    c[n - 1] *= z / (z - 1.0);
}

fn spline_filter_line(c: &mut [f64], boundary: Boundary) {
    let n = c.len();
    if n <= 1 {
        return;
    }
    let z = CUBIC_POLE;
    let gain = (1.0 - z) * (1.0 - 1.0 / z);
    for v in c.iter_mut() {
        *v *= gain;
    }
    match boundary {
        Boundary::Constant => init_causal_mirror(c, z),
        Boundary::Nearest => init_causal_reflect(c, z),
    }
    for i in 1..n {
        c[i] += z * c[i - 1];
    }
    match boundary {
        Boundary::Constant => init_anticausal_mirror(c, z),
        Boundary::Nearest => init_anticausal_reflect(c, z),
    }
    for i in (0..n - 1).rev() {
        c[i] = z * (c[i + 1] - c[i]);
    }
}

/// Prefilter every axis in turn (`scipy.ndimage.spline_filter`).
fn spline_filter(data: &mut ArrayD<f64>, boundary: Boundary) {
    for axis in 0..data.ndim() {
        for mut lane in data.lanes_mut(Axis(axis)) {
            let mut line: Vec<f64> = lane.iter().copied().collect();
            spline_filter_line(&mut line, boundary);
            for (dst, src) in lane.iter_mut().zip(line) {
                *dst = src;
            }
        }
    }
}

/// `np.pad(data, 12, mode="edge")`.
fn pad_edge(data: &ArrayD<f64>, pad: usize) -> ArrayD<f64> {
    let shape: Vec<usize> = data.shape().iter().map(|n| n + 2 * pad).collect();
    let src_shape = data.shape().to_vec();
    ArrayD::from_shape_fn(IxDyn(&shape), |idx| {
        let at: Vec<usize> =
            (0..idx.ndim()).map(|k| (idx[k] as i64 - pad as i64).clamp(0, src_shape[k] as i64 - 1) as usize).collect();
        data[IxDyn(&at)]
    })
}

fn cubic_weights(cc: f64) -> [f64; 4] {
    let x = cc - cc.floor();
    let y = x;
    let z = 1.0 - x;
    let mut w = [0.0; 4];
    w[1] = (y * y * (y - 2.0) * 3.0 + 4.0) / 6.0;
    w[2] = (z * z * (z - 2.0) * 3.0 + 4.0) / 6.0;
    w[0] = z * z * z / 6.0;
    w[3] = 1.0;
    for i in 0..3 {
        w[3] -= w[i];
    }
    w
}

fn mirror_index(idx: i64, len: i64) -> i64 {
    if len <= 1 {
        return 0;
    }
    let sz2 = 2 * len - 2;
    if idx < 0 {
        let v = sz2 * (-idx / sz2) + idx;
        if v <= 1 - len {
            v + sz2
        } else {
            -v
        }
    } else if idx > len - 1 {
        let mut v = idx - sz2 * (idx / sz2);
        if v >= len {
            v = sz2 - v;
        }
        v
    } else {
        idx
    }
}

/// Interpolate one prefiltered component at `(N, S)` coordinates.
fn spline_interpolate(coeffs: &ArrayD<f64>, coords: &Array2<f64>, boundary: Boundary, npad: usize) -> Vec<f64> {
    let dim = coeffs.ndim();
    let dims: Vec<i64> = coeffs.shape().iter().map(|n| *n as i64).collect();
    let filter_size = 4usize.pow(dim as u32);
    let mut out = Vec::with_capacity(coords.nrows());
    let mut starts = vec![0i64; dim];
    let mut weights = vec![[0.0f64; 4]; dim];
    let mut idx = vec![0usize; dim];
    for point in coords.outer_iter() {
        let mut constant = false;
        for axis in 0..dim {
            let mut cc = point[axis] + npad as f64;
            if boundary == Boundary::Constant && (cc < 0.0 || cc > (dims[axis] - 1) as f64) {
                cc = -1.0;
            }
            if cc > -1.0 || boundary == Boundary::Nearest {
                starts[axis] = cc.floor() as i64 - 1;
                weights[axis] = cubic_weights(cc);
            } else {
                constant = true;
                break;
            }
        }
        if constant {
            out.push(0.0);
            continue;
        }
        let mut t = 0.0;
        for h in 0..filter_size {
            // Odometer over the 4**S footprint, last axis fastest.
            let mut rest = h;
            let mut digits = vec![0usize; dim];
            for axis in (0..dim).rev() {
                digits[axis] = rest % 4;
                rest /= 4;
            }
            for axis in 0..dim {
                let i = starts[axis] + digits[axis] as i64;
                let mapped = if i < 0 || i >= dims[axis] {
                    match boundary {
                        Boundary::Constant => mirror_index(i, dims[axis]),
                        Boundary::Nearest => i.clamp(0, dims[axis] - 1),
                    }
                } else {
                    i
                };
                idx[axis] = mapped as usize;
            }
            let mut coeff = coeffs[IxDyn(&idx)];
            for axis in 0..dim {
                coeff *= weights[axis][digits[axis]];
            }
            t += coeff;
        }
        out.push(t);
    }
    out
}

/// Cubic interpolation of `(C, *spatial)` data at `(N, S)` continuous indices,
/// as `scipy.ndimage.map_coordinates(order=3)` computes it.
pub fn cubic_sample(field: &ArrayD<f64>, coords: &Array2<f64>, extrapolation: &str) -> Result<Array2<f64>> {
    check_extrapolation(extrapolation)?;
    let spatial: Vec<usize> = field.shape()[1..].to_vec();
    if extrapolation == "error" {
        // SciPy has no raising mode; the domain check happens here, against the
        // bounds `linear_sample` uses, rather than folding into constant-zero.
        refuse_outside(&inside_extent(&spatial, coords))?;
    }
    let boundary = if extrapolation == "nearest" { Boundary::Nearest } else { Boundary::Constant };
    let components = field.shape()[0];
    let mut out = Array2::<f64>::zeros((coords.nrows(), components));
    for c in 0..components {
        let plane = field.index_axis(Axis(0), c).to_owned();
        let (mut coeffs, npad) = match boundary {
            Boundary::Nearest => (pad_edge(&plane, 12), 12),
            Boundary::Constant => (plane, 0),
        };
        spline_filter(&mut coeffs, boundary);
        for (row, v) in spline_interpolate(&coeffs, coords, boundary, npad).into_iter().enumerate() {
            out[[row, c]] = v;
        }
    }
    Ok(out)
}

/// Interpolate a field with the declared interpolation.
pub fn sample_field(
    field: &ArrayD<f64>,
    coords: &Array2<f64>,
    interpolation: &str,
    extrapolation: &str,
) -> Result<Array2<f64>> {
    match interpolation {
        "linear" => linear_sample(field, coords, extrapolation),
        "cubic" => cubic_sample(field, coords, extrapolation),
        other => Err(Error::invalid(format!("unknown interpolation {}", repr_str(other)))),
    }
}

/// `direction @ diag(spacing)`: index displacement to world displacement.
pub fn linear_part(grid: &Grid) -> Array2<f64> {
    let s = grid.spacing.len();
    let mut out = Array2::zeros((s, s));
    for r in 0..s {
        for c in 0..s {
            out[[r, c]] = grid.direction[[r, c]] * grid.spacing[c];
        }
    }
    out
}

/// Convert `(N, S)` displacement components to world units.
pub fn to_world_vectors(values: &Array2<f64>, grid: &Grid, vector_space: &str) -> Result<Array2<f64>> {
    match vector_space {
        "world" => Ok(values.clone()),
        "index" => Ok(values.dot(&linear_part(grid).t())),
        other => Err(Error::invalid(format!("unknown vector_space {}", repr_str(other)))),
    }
}

/// `np.gradient(values, axis=axis)` with unit spacing and first-order edges.
pub fn gradient(values: &ArrayD<f64>, axis: usize) -> Result<ArrayD<f64>> {
    let n = values.shape()[axis];
    if n < 2 {
        return Err(Error::Value(
            "Shape of array too small to calculate a numerical gradient, at least (edge_order + 1) elements are required."
                .into(),
        ));
    }
    let mut out = ArrayD::<f64>::zeros(values.raw_dim());
    for (src, mut dst) in values.lanes(Axis(axis)).into_iter().zip(out.lanes_mut(Axis(axis))) {
        for i in 1..n - 1 {
            dst[i] = (src[i + 1] - src[i - 1]) / 2.0;
        }
        dst[0] = src[1] - src[0];
        dst[n - 1] = src[n - 1] - src[n - 2];
    }
    Ok(out)
}

/// `det(I + du/dx)` per voxel of the field grid; values `<= 0` mark folding.
pub fn jacobian_determinant(field: &ArrayD<f64>, grid: &Grid, vector_space: &str) -> Result<ArrayD<f64>> {
    let dim = field.shape()[0];
    if dim != grid.n_spatial() {
        return Err(Error::coded("E503", format!("field has {dim} components for a {}-D grid", grid.n_spatial())));
    }
    let linear = linear_part(grid);
    let world = match vector_space {
        "world" => field.clone(),
        "index" => {
            let mut out = ArrayD::<f64>::zeros(field.raw_dim());
            for r in 0..dim {
                let mut lane = out.index_axis_mut(Axis(0), r);
                for c in 0..dim {
                    let l = linear[[r, c]];
                    lane.zip_mut_with(&field.index_axis(Axis(0), c), |o, v| *o += l * v);
                }
            }
            out
        }
        other => return Err(Error::invalid(format!("unknown vector_space {}", repr_str(other)))),
    };
    let inverse_linear = inv(&linear)?;
    let mut gradients = Vec::with_capacity(dim * dim);
    for c in 0..dim {
        let comp = world.index_axis(Axis(0), c).to_owned();
        for a in 0..dim {
            gradients.push(gradient(&comp, a)?);
        }
    }
    let spatial: Vec<usize> = field.shape()[1..].to_vec();
    let mut out = ArrayD::<f64>::zeros(IxDyn(&spatial));
    let mut jac = Array2::<f64>::zeros((dim, dim));
    for (flat, slot) in out.iter_mut().enumerate() {
        for c in 0..dim {
            for b in 0..dim {
                let mut v = 0.0;
                for a in 0..dim {
                    let g = gradients[c * dim + a].as_slice().map(|s| s[flat]).unwrap_or(0.0);
                    v += g * inverse_linear[[a, b]];
                }
                jac[[c, b]] = v + if c == b { 1.0 } else { 0.0 };
            }
        }
        *slot = det(&jac);
    }
    Ok(out)
}

/// Fraction of voxels where the transform folds (`det <= 0`).
pub fn folding_fraction(determinants: &ArrayD<f64>) -> f64 {
    if determinants.is_empty() {
        return 0.0;
    }
    determinants.iter().filter(|v| **v <= 0.0).count() as f64 / determinants.len() as f64
}

/// Target registration error summary (§10.6).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tre {
    pub mean: f64,
    pub median: f64,
    pub max: f64,
    pub n: usize,
}

/// `np.median` of a slice.
pub fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// TRE `‖T(p_F) − p_M‖` from already-warped fixed points.
pub fn tre_from_warped(warped: &Array2<f64>, moving: &Array2<f64>, weights: Option<&[f64]>) -> Result<Tre> {
    if warped.dim() != moving.dim() {
        return Err(Error::invalid(format!(
            "landmark sets disagree: {:?} vs {:?}; §10.6 requires equal N and matching row order",
            warped.dim(),
            moving.dim()
        )));
    }
    let errors: Vec<f64> = warped
        .outer_iter()
        .zip(moving.outer_iter())
        .map(|(a, b)| a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt())
        .collect();
    let w: Vec<f64> = match weights {
        Some(w) => w.to_vec(),
        None => vec![1.0; errors.len()],
    };
    let total: f64 = w.iter().sum();
    let weighted: f64 = errors.iter().zip(&w).map(|(e, w)| e * w).sum();
    Ok(Tre {
        mean: if total != 0.0 { weighted / total } else { 0.0 },
        median: median(&errors),
        max: errors.iter().copied().fold(f64::NEG_INFINITY, f64::max).max(if errors.is_empty() {
            0.0
        } else {
            f64::NEG_INFINITY
        }),
        n: errors.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_extrapolation_modes() {
        let field = ArrayD::from_elem(IxDyn(&[1, 4, 4]), 1.0);
        let outside = Array2::from_shape_vec((1, 2), vec![-5.0, -5.0]).unwrap();
        assert_eq!(linear_sample(&field, &outside, "zero").unwrap()[[0, 0]], 0.0);
        assert_eq!(linear_sample(&field, &outside, "nearest").unwrap()[[0, 0]], 1.0);
        assert!(linear_sample(&field, &outside, "error").unwrap_err().to_string().contains("outside"));
        assert!(linear_sample(&field, &outside, "wing-it").is_err());
    }

    #[test]
    fn cubic_matches_scipy_map_coordinates() {
        // i = np.arange(20.0); field = (i * i * 0.37 + i / 3.0).reshape(1, 4, 5)
        let field = ArrayD::from_shape_fn(IxDyn(&[1, 4, 5]), |i| {
            let v = (i[1] * 5 + i[2]) as f64;
            v * v * 0.37 + v / 3.0
        });
        let coords = Array2::from_shape_vec((3, 2), vec![1.3, 2.7, 0.2, 0.1, 2.9, 3.95]).unwrap();
        let constant = cubic_sample(&field, &coords, "zero").unwrap();
        let nearest = cubic_sample(&field, &coords, "nearest").unwrap();
        // scipy.ndimage.map_coordinates(field[0], coords.T, order=3, mode=...)
        let want_constant = [32.707648506666686, 0.6104213523809539, 138.73197259642862];
        let want_nearest = [33.326160022592546, 1.16275434318072, 135.63402382870262];
        for i in 0..3 {
            assert_eq!(constant[[i, 0]], want_constant[i]);
            assert_eq!(nearest[[i, 0]], want_nearest[i]);
        }
        let ones = ArrayD::from_elem(IxDyn(&[1, 4, 4]), 1.0);
        let outside = Array2::from_shape_vec((1, 2), vec![-5.0, -5.0]).unwrap();
        assert_eq!(cubic_sample(&ones, &outside, "zero").unwrap()[[0, 0]], 0.0);
        assert!((cubic_sample(&ones, &outside, "nearest").unwrap()[[0, 0]] - 1.0).abs() < 1e-6);
        assert!(cubic_sample(&ones, &outside, "error").is_err());
    }

    #[test]
    fn gradients_follow_numpy() {
        let v = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 2.0, 4.0, 7.0]).unwrap();
        assert_eq!(gradient(&v, 0).unwrap().iter().copied().collect::<Vec<_>>(), vec![1.0, 1.5, 2.5, 3.0]);
    }
}
