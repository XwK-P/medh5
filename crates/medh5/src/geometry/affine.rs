//! The index-world affine and the box/slice convention (spec §3.3, §8.1).
//!
//! Two conventions are fixed here and nowhere else, because a format that
//! leaves either implicit produces silent half-voxel errors that survive every
//! unit test and show up as a systematic bias in a trained model:
//!
//! **Index coordinates.**  Integer index *i* is the **centre** of voxel *i*.  A
//! volume of extent *n* spans the closed region `[-0.5, n - 0.5]`, and
//!
//! ```text
//! x_world = origin + direction @ (spacing * i)
//! ```
//!
//! **Box corners.**  Boxes are measured at voxel **edges**, so a box maps onto
//! a NumPy slice without rounding: slice `a:b` <-> `lo = a - 0.5, hi = b - 0.5`.

use ndarray::Array2;
use serde_json::{json, Value};

use super::linalg::{allclose, det, eye, inv};
use crate::json::format_g;
use crate::{Error, Result};

/// Tolerance on `direction.T @ direction == I` (spec §3.2).
pub const ORTHONORMAL_TOL: f64 = 1e-4;

/// Assemble the `(S+1, S+1)` index-to-world affine of spec §3.3.
pub fn build_affine(spacing: &[f64], origin: &[f64], direction: &Array2<f64>) -> Result<Array2<f64>> {
    let s = spacing.len();
    if origin.len() != s || direction.dim() != (s, s) {
        return Err(Error::coded(
            "E109",
            format!(
                "spacing ({s},), origin ({},) and direction {:?} must agree on the number of spatial axes",
                origin.len(),
                direction.dim()
            ),
        ));
    }
    let mut affine = eye(s + 1);
    for r in 0..s {
        for c in 0..s {
            affine[[r, c]] = direction[[r, c]] * spacing[c];
        }
        affine[[r, s]] = origin[r];
    }
    Ok(affine)
}

/// Split an affine back into `(spacing, origin, direction)`.
pub fn decompose_affine(affine: &Array2<f64>) -> Result<(Vec<f64>, Vec<f64>, Array2<f64>)> {
    let s = affine.nrows() - 1;
    let mut spacing = vec![0.0; s];
    for (c, sp) in spacing.iter_mut().enumerate() {
        *sp = (0..s).map(|r| affine[[r, c]].powi(2)).sum::<f64>().sqrt();
    }
    if spacing.iter().any(|v| *v <= 0.0) {
        return Err(Error::coded("E104", "affine has a degenerate spatial axis"));
    }
    let mut direction = Array2::zeros((s, s));
    for r in 0..s {
        for c in 0..s {
            direction[[r, c]] = affine[[r, c]] / spacing[c];
        }
    }
    let origin = (0..s).map(|r| affine[[r, s]]).collect();
    Ok((spacing, origin, direction))
}

fn residual(direction: &Array2<f64>) -> Array2<f64> {
    let n = direction.nrows();
    direction.t().dot(direction) - eye(n)
}

/// Whether `direction` is orthonormal within `tol`.
pub fn is_orthonormal(direction: &Array2<f64>, tol: f64) -> bool {
    if direction.nrows() != direction.ncols() {
        return false;
    }
    let n = direction.nrows();
    let gram = direction.t().dot(direction);
    allclose(
        gram.as_standard_layout().as_slice().unwrap(),
        eye(n).as_slice().unwrap(),
        tol,
        0.0,
    )
}

/// Validate orthonormality, raising E102 when it fails.
pub fn check_orthonormal(direction: &Array2<f64>, tol: f64, what: &str) -> Result<()> {
    if !is_orthonormal(direction, tol) {
        let r = if direction.nrows() == direction.ncols() {
            residual(direction).iter().fold(0.0f64, |m, v| m.max(v.abs()))
        } else {
            f64::NAN
        };
        return Err(Error::coded(
            "E102",
            format!("{what} is not orthonormal to {} (max residual {})", format_g(tol, 6), format_g(r, 3)),
        ));
    }
    Ok(())
}

/// Whether `matrix` is orthonormal **and** has determinant +1.
pub fn is_proper_rotation(matrix: &Array2<f64>, tol: f64) -> bool {
    if !is_orthonormal(matrix, tol) {
        return false;
    }
    (det(matrix) - 1.0).abs() <= tol.max(1e-6)
}

/// Map continuous index coordinates (flat, `S` per point) to world.
pub fn index_to_world(affine: &Array2<f64>, indices: &[f64]) -> Vec<f64> {
    let s = affine.nrows() - 1;
    let mut out = Vec::with_capacity(indices.len());
    for p in indices.chunks(s) {
        for r in 0..s {
            // The dot product first, then the origin: NumPy's evaluation
            // order for `flat @ A[:S,:S].T + A[:S,S]`.
            let mut v = 0.0;
            for c in 0..s {
                v += affine[[r, c]] * p[c];
            }
            out.push(v + affine[[r, s]]);
        }
    }
    out
}

/// Map world coordinates (flat, `S` per point) back to continuous index.
pub fn world_to_index(affine: &Array2<f64>, points: &[f64]) -> Result<Vec<f64>> {
    let s = affine.nrows() - 1;
    let linear = affine.slice(ndarray::s![..s, ..s]).to_owned();
    let inverse = inv(&linear)?;
    let mut out = Vec::with_capacity(points.len());
    for p in points.chunks(s) {
        for r in 0..s {
            let mut v = 0.0;
            for c in 0..s {
                v += inverse[[r, c]] * (p[c] - affine[[c, s]]);
            }
            out.push(v);
        }
    }
    Ok(out)
}

/// One axis of a box converted to a slice: `(start, stop)`.
pub type SliceBounds = (i64, i64);

/// Convert one `(S, 2)` box (flat `[lo0, hi0, lo1, hi1, ...]`) to slices.
///
/// `start = floor(lo + 0.5)`, `stop = floor(hi + 0.5)` (spec §8.1) --- half
/// **up**, never half-to-even, or a box on integer edge coordinates rounds its
/// two ends in opposite directions.  With `shape` the result is clipped.
pub fn box_to_slices(bbox: &[f64], shape: Option<&[usize]>) -> Result<Vec<SliceBounds>> {
    if bbox.len() % 2 != 0 {
        return Err(Error::invalid(format!("box must have shape (S, 2), got ({},)", bbox.len())));
    }
    let s = bbox.len() / 2;
    if (0..s).any(|k| bbox[2 * k] > bbox[2 * k + 1]) {
        return Err(Error::coded("E406", "box has lo > hi on at least one axis"));
    }
    let mut out = Vec::with_capacity(s);
    for k in 0..s {
        let mut start = (bbox[2 * k] + 0.5).floor() as i64;
        let mut stop = (bbox[2 * k + 1] + 0.5).floor() as i64;
        if let Some(extent) = shape {
            let n = extent[k] as i64;
            start = start.clamp(0, n);
            stop = stop.clamp(start, n);
        }
        out.push((start, stop));
    }
    Ok(out)
}

/// Convert slices to one `(S, 2)` box (flat), in continuous index coordinates.
pub fn slices_to_box(slices: &[(i64, i64)]) -> Vec<f32> {
    let mut out = Vec::with_capacity(slices.len() * 2);
    for (start, stop) in slices {
        out.push(*start as f32 - 0.5);
        out.push(*stop as f32 - 0.5);
    }
    out
}

/// The `2**S` corners of an axis-aligned box (flat `(S, 2)`), odometer order.
pub fn box_corners(bbox: &[f64]) -> Vec<Vec<f64>> {
    let s = bbox.len() / 2;
    let mut out = Vec::with_capacity(1 << s);
    for n in 0..(1usize << s) {
        let mut corner = Vec::with_capacity(s);
        for k in 0..s {
            // Odometer: the first axis varies slowest.
            let bit = (n >> (s - 1 - k)) & 1;
            corner.push(bbox[2 * k + bit]);
        }
        out.push(corner);
    }
    out
}

/// Axis-aligned bounds `(S, 2)` (flat) of a box after an affine.
pub fn apply_affine_to_box(affine: &Array2<f64>, bbox: &[f64]) -> Vec<f64> {
    let s = bbox.len() / 2;
    let corners: Vec<f64> = box_corners(bbox).into_iter().flatten().collect();
    let world = index_to_world(affine, &corners);
    let mut out = Vec::with_capacity(2 * s);
    for k in 0..s {
        let values = world.iter().skip(k).step_by(s);
        let lo = values.clone().fold(f64::INFINITY, |a, b| a.min(*b));
        let hi = values.fold(f64::NEG_INFINITY, |a, b| a.max(*b));
        out.push(lo);
        out.push(hi);
    }
    out
}

/// Volume of one voxel in the grid's units^S.
pub fn voxel_volume(spacing: &[f64]) -> f64 {
    spacing.iter().product()
}

/// Human-facing summary of an affine.
pub fn affine_summary(affine: &Array2<f64>) -> Result<Value> {
    let (spacing, origin, direction) = decompose_affine(affine)?;
    Ok(json!({
        "spacing": spacing,
        "origin": origin,
        "direction": direction.rows().into_iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
        "determinant": det(&direction),
        "orthonormal": is_orthonormal(&direction, ORTHONORMAL_TOL),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_roundtrip() {
        let dir = eye(3);
        let a = build_affine(&[1.5, 0.8, 0.8], &[-12.0, -9.6, -9.6], &dir).unwrap();
        let w = index_to_world(&a, &[1.0, 2.0, 3.0]);
        assert_eq!(w, vec![-10.5, -8.0, -7.199999999999999]);
        let back = world_to_index(&a, &w).unwrap();
        assert!(allclose(&back, &[1.0, 2.0, 3.0], 1e-12, 0.0));
    }

    #[test]
    fn boxes_round_half_up() {
        // [11.5, 39.5] <-> slice(12, 40), extent 28 (Appendix C.2).
        assert_eq!(box_to_slices(&[11.5, 39.5], None).unwrap(), vec![(12, 40)]);
        // One voxel on integer edges: half-to-even would give an empty slice.
        assert_eq!(box_to_slices(&[1.0, 2.0], None).unwrap(), vec![(1, 2)]);
        assert_eq!(box_to_slices(&[2.0, 3.0], None).unwrap(), vec![(2, 3)]);
        assert_eq!(slices_to_box(&[(12, 40)]), vec![11.5, 39.5]);
        assert_eq!(box_to_slices(&[2.0, 1.0], None).unwrap_err().code(), Some("E406"));
    }

    #[test]
    fn corners_in_odometer_order() {
        let c = box_corners(&[0.0, 1.0, 10.0, 11.0]);
        assert_eq!(c, vec![vec![0.0, 10.0], vec![0.0, 11.0], vec![1.0, 10.0], vec![1.0, 11.0]]);
    }
}
