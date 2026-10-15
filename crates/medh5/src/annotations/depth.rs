//! How a class's polygons combine in a plane (§8.6): by nesting depth.
//!
//! A polygon *p* encloses *q* when every vertex of *q* lies inside *p*, and a
//! polygon's depth is the number of the class's other polygons in that plane
//! that enclose it.  The region is the points inside an odd number of the
//! depth layers, each layer the union of its polygons: a hole subtracts from
//! the region around it, an island inside a hole adds back, and outlines that
//! overlap without nesting are their union.  `contour_role` *should* be the
//! depth's parity; a reader takes the depth (W919 reports a role that
//! disagrees).
//!
//! One function serves the validator, the reader and the Python converters,
//! so the RTSTRUCT importer, the rasteriser and the validator cannot disagree
//! about which polygon is a hole.

use std::collections::BTreeMap;

use ndarray::{s, Array2, ArrayView2, Axis};

/// Whether `point` lies inside `polygon`, both in-plane `(x, y)`, by the
/// even-odd rule: a ray towards `+x` crosses the outline an odd number of
/// times.  The test is the reference rasteriser's, edge for edge, so a
/// vertex on an outline is decided as the rasteriser decides a voxel centre.
fn inside(polygon: ArrayView2<'_, f64>, x: f64, y: f64) -> bool {
    let n = polygon.nrows();
    let mut odd = false;
    for i in 0..n {
        let (x0, y0) = (polygon[[i, 0]], polygon[[i, 1]]);
        let j = (i + 1) % n;
        let (x1, y1) = (polygon[[j, 0]], polygon[[j, 1]]);
        // `(y0 > y) != (y1 > y)` is false for a horizontal edge, so the
        // division below never divides by zero.
        if (y0 > y) != (y1 > y) && x < x0 + (y - y0) * (x1 - x0) / (y1 - y0) {
            odd = !odd;
        }
    }
    odd
}

/// `[lo, hi]` per in-plane axis, or `None` for no vertices.
fn extent(polygon: ArrayView2<'_, f64>) -> Option<[(f64, f64); 2]> {
    if polygon.nrows() == 0 || polygon.ncols() < 2 {
        return None;
    }
    let mut out = [(f64::INFINITY, f64::NEG_INFINITY); 2];
    for row in polygon.outer_iter() {
        for (axis, bound) in out.iter_mut().enumerate() {
            bound.0 = bound.0.min(row[axis]);
            bound.1 = bound.1.max(row[axis]);
        }
    }
    Some(out)
}

/// Whether every vertex of `inner` lies inside `outer` (§8.6), both `(V, 2)`
/// in-plane coordinates.  An outline of fewer than three vertices encloses
/// nothing, and no vertices are enclosed by nothing.
pub fn encloses(outer: ArrayView2<'_, f64>, inner: ArrayView2<'_, f64>) -> bool {
    let (Some(o), Some(i)) = (extent(outer), extent(inner)) else { return false };
    if outer.nrows() < 3 {
        return false;
    }
    // A vertex clearly outside the outer's extent is outside it: a shortcut
    // that skips the edge test for polygons far apart, with a margin so that
    // it never decides a vertex the edge test would decide otherwise.
    let span = (o[0].1 - o[0].0).abs().max((o[1].1 - o[1].0).abs()) + 1.0;
    let margin = 1e-9 * span;
    if (0..2).any(|a| i[a].0 < o[a].0 - margin || i[a].1 > o[a].1 + margin) {
        return false;
    }
    inner.outer_iter().all(|v| inside(outer, v[0], v[1]))
}

/// The depth of each of one class's polygons in one plane: how many of the
/// others enclose it (§8.6).  Each polygon is `(V, 2)`, in any in-plane
/// coordinates (enclosure is unchanged by an affine map of the plane).
pub fn polygon_depths(polygons: &[ArrayView2<'_, f64>]) -> Vec<usize> {
    polygons
        .iter()
        .enumerate()
        .map(|(i, inner)| {
            polygons.iter().enumerate().filter(|(j, outer)| *j != i && encloses(outer.view(), inner.view())).count()
        })
        .collect()
}

/// Polygon `index`'s rows of the `n` vertices, clipped to the rows there are:
/// offsets the validator refuses (E408) read as fewer vertices.
fn rows(offsets: &[i64], index: usize, n: usize) -> (usize, usize) {
    let at = |k: usize| offsets.get(k).copied().unwrap_or(0).max(0) as usize;
    let a = at(index).min(n);
    (a, at(index + 1).min(n).max(a))
}

/// `(V, 3)` vertices as `(V, 2)` coordinates in their plane: the coordinate
/// along which the plane's normal is largest is dropped, which maps the
/// plane affinely and one to one onto the other two.  The normal is
/// Newell's, summed in magnitude over the polygons, so outlines of opposite
/// winding do not cancel.
fn in_plane(polygons: &[ArrayView2<'_, f64>]) -> Vec<Array2<f64>> {
    let mut normal = [0.0f64; 3];
    for p in polygons {
        let n = p.nrows();
        let mut own = [0.0f64; 3];
        for i in 0..n {
            let (a, b) = (p.row(i), p.row((i + 1) % n));
            own[0] += (a[1] - b[1]) * (a[2] + b[2]);
            own[1] += (a[2] - b[2]) * (a[0] + b[0]);
            own[2] += (a[0] - b[0]) * (a[1] + b[1]);
        }
        for (total, part) in normal.iter_mut().zip(own) {
            *total += part.abs();
        }
    }
    let dropped = (0..3).fold(0, |best, k| if normal[k] > normal[best] { k } else { best });
    let kept: Vec<usize> = (0..3).filter(|k| *k != dropped).collect();
    polygons.iter().map(|p| p.select(Axis(1), &kept)).collect()
}

/// Each polygon's depth within its class and plane (§8.6), from a
/// `contours` annotation's columns: `vertices` `(V, S)`, `contour_offsets`,
/// `contour_class_ids` and the `(axis, index)` of `contour_plane`.
///
/// `None` for a polygon with `axis = −1`, which takes part in no
/// combination, and for every polygon when `S` is neither 2 nor 3.  The
/// coordinates may be `index` or `world` ones: a class's polygons in one
/// plane are compared in that plane, whichever space they are stored in.
pub fn contour_depths(
    vertices: ArrayView2<'_, f64>,
    offsets: &[i64],
    class_ids: &[i64],
    planes: &[(i64, i64)],
) -> Vec<Option<usize>> {
    let m = class_ids.len().min(planes.len()).min(offsets.len().saturating_sub(1));
    let mut out = vec![None; class_ids.len().max(planes.len())];
    let dims = vertices.ncols();
    if dims != 2 && dims != 3 {
        return out;
    }
    let mut groups: BTreeMap<(i64, i64, i64), Vec<usize>> = BTreeMap::new();
    for i in 0..m {
        if planes[i].0 >= 0 {
            groups.entry((class_ids[i], planes[i].0, planes[i].1)).or_default().push(i);
        }
    }
    let n = vertices.nrows();
    for members in groups.values() {
        let drawn: Vec<ArrayView2<'_, f64>> = members
            .iter()
            .map(|&i| {
                let (a, b) = rows(offsets, i, n);
                vertices.slice(s![a..b, ..])
            })
            .collect();
        let depths = if dims == 2 {
            polygon_depths(&drawn)
        } else {
            let flat = in_plane(&drawn);
            polygon_depths(&flat.iter().map(|p| p.view()).collect::<Vec<_>>())
        };
        for (&i, depth) in members.iter().zip(depths) {
            out[i] = Some(depth);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    fn square(lo: f64, hi: f64) -> Array2<f64> {
        array![[lo, lo], [lo, hi], [hi, hi], [hi, lo]]
    }

    #[test]
    fn s8_6_depth_counts_the_polygons_that_enclose_one() {
        let (outer, hole, island, apart) =
            (square(2.0, 17.0), square(5.0, 14.0), square(8.0, 11.0), square(20.0, 22.0));
        let found = polygon_depths(&[island.view(), outer.view(), apart.view(), hole.view()]);
        assert_eq!(found, vec![2, 0, 0, 1]);
    }

    #[test]
    fn s8_6_crossing_outlines_sit_at_one_depth() {
        let left = square(2.0, 12.0);
        let right = &left + &array![[0.0, 6.0]];
        assert_eq!(polygon_depths(&[left.view(), right.view()]), vec![0, 0]);
    }

    #[test]
    fn s8_6_depth_is_per_class_and_plane_in_any_orientation() {
        // Two classes on plane (0, 4), nested within each, and the same
        // squares on a plane that is not axis-aligned in world space.
        let lift = |sq: &Array2<f64>, z: f64| {
            let mut out = Array2::zeros((sq.nrows(), 3));
            for (i, r) in sq.outer_iter().enumerate() {
                out[[i, 0]] = z;
                out[[i, 1]] = r[0];
                out[[i, 2]] = r[1];
            }
            out
        };
        let (outer, inner) = (lift(&square(4.0, 9.0), 4.0), lift(&square(6.0, 7.0), 4.0));
        let tilt = |v: &Array2<f64>| {
            // A shear that keeps each plane a plane, and none axis-aligned.
            let m = array![[1.0, 0.2, 0.1], [0.3, 1.0, 0.0], [0.0, 0.4, 1.0]];
            v.dot(&m)
        };
        let stacked = ndarray::concatenate(
            Axis(0),
            &[outer.view(), inner.view(), outer.view(), tilt(&outer).view(), tilt(&inner).view()],
        )
        .unwrap();
        let offsets = [0, 4, 8, 12, 16, 20];
        let classes = [1, 1, 2, 3, 3];
        let planes = [(0, 4), (0, 4), (0, 4), (0, 4), (0, 4)];
        assert_eq!(
            contour_depths(stacked.view(), &offsets, &classes, &planes),
            vec![Some(0), Some(1), Some(0), Some(0), Some(1)]
        );
        let out_of_plane = [(0, 4), (-1, 0), (0, 4), (0, 4), (0, 4)];
        assert_eq!(contour_depths(stacked.view(), &offsets, &classes, &out_of_plane)[..2], [Some(0), None]);
    }

    #[test]
    fn s8_6_offsets_past_the_vertices_read_as_fewer() {
        let v = square(0.0, 1.0);
        assert_eq!(contour_depths(v.view(), &[0, 4, 9], &[1, 1], &[(0, 0), (0, 0)]), vec![Some(0), Some(0)]);
    }
}
