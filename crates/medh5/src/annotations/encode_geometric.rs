//! Encoders for the geometric kinds (spec §8) and classification (spec §9).
//!
//! Every coordinate lives in a declared `space`; boxes are measured at voxel
//! edges and stored `float32` `[lo, hi]` (§8.1).  An empty collection is the
//! verified negative the coverage contract exists to record, not a degenerate
//! input.

use std::collections::BTreeMap;

use ndarray::{Array2, ArrayD, IxDyn};
use serde_json::{Map, Value};

use super::encode::{box_axis_ok, instance_id_array};
use super::payload::{Payload, PayloadData};
use crate::array::{DType, NdArray};
use crate::geometry::affine::is_proper_rotation;
use crate::h5::attrs::AttrValue;
use crate::json::{dumps, format_g, repr_int_tuple, repr_list, repr_str, Style};
use crate::labels::check_class_id;
use crate::{Error, Result};

/// `space` values (§8.1).
pub const SPACES: [&str; 2] = ["index", "world"];
/// Contour roles: `0` outer boundary, `1` hole (§8.6).
pub const CONTOUR_ROLES: [&str; 2] = ["outer", "hole"];
/// Keypoint visibility codes (§8.4).
pub const VISIBILITY: [(u8, &str); 3] = [(0, "unlabelled"), (1, "occluded"), (2, "visible")];
/// Tolerance on a proper rotation (§8.3).
pub const ROTATION_TOL: f64 = 1e-4;
/// Classification scopes (§9).
pub const SCOPES: [&str; 6] = ["sample", "timepoint", "grid", "roi", "slice", "instance"];

/// Validate a `space` value (E412).
pub fn check_space(space: &str) -> Result<()> {
    if !SPACES.contains(&space) {
        return Err(Error::coded("E412", format!("space {} must be one of {}", repr_str(space), repr_list(&SPACES))));
    }
    Ok(())
}

/// Validate a classification scope (E412).
pub fn check_scope(scope: &str) -> Result<()> {
    if !SCOPES.contains(&scope) {
        return Err(Error::coded(
            "E412",
            format!("classification scope {} must be one of {}", repr_str(scope), repr_list(&SCOPES)),
        ));
    }
    Ok(())
}

fn u16_column(ids: &[i64]) -> Result<NdArray> {
    let mut checked = Vec::with_capacity(ids.len());
    for c in ids {
        checked.push(check_class_id(*c)?);
    }
    NdArray::from_vec(&[checked.len()], checked)
}

/// The per-object columns shared by boxes, obb and keypoints.
pub struct ObjectColumns<'a> {
    pub class_ids: &'a [i64],
    pub instance_ids: Option<&'a [u64]>,
    pub scores: Option<&'a [f64]>,
    pub attributes: Option<&'a [Map<String, Value>]>,
}

fn object_columns(n: usize, cols: &ObjectColumns) -> Result<Vec<(String, PayloadData)>> {
    if cols.class_ids.len() != n {
        return Err(Error::coded("E405", format!("class_ids has {} entries for {n} objects", cols.class_ids.len())));
    }
    let mut out = vec![("class_ids".to_string(), PayloadData::Array(u16_column(cols.class_ids)?))];
    if let Some(ids) = cols.instance_ids {
        out.push(("instance_ids".into(), PayloadData::Array(instance_id_array(ids))));
    }
    if let Some(scores) = cols.scores {
        out.push((
            "scores".into(),
            PayloadData::Array(NdArray::from_vec(
                &[scores.len()],
                scores.iter().map(|v| *v as f32).collect::<Vec<_>>(),
            )?),
        ));
    }
    if let Some(attrs) = cols.attributes {
        let style = Style::PYTHON.sorted();
        out.push((
            "attributes".into(),
            PayloadData::Strings(attrs.iter().map(|a| dumps(&Value::Object(a.clone()), style)).collect()),
        ));
    }
    for (name, column) in &out {
        let len = column.shape()[0];
        if len != n {
            return Err(Error::coded("E405", format!("{name} has {len} entries for {n} objects")));
        }
    }
    Ok(out)
}

fn sorted_unique(ids: impl IntoIterator<Item = i64>) -> Vec<i64> {
    let mut v: Vec<i64> = ids.into_iter().collect();
    v.sort();
    v.dedup();
    v
}

// -- boxes --------------------------------------------------------------------------

/// The single axis a box (flat `(S, 2)`) has no extent on, or `None`.
pub fn degenerate_axis(bbox: &[f64]) -> Option<usize> {
    let flat: Vec<usize> = (0..bbox.len() / 2).filter(|k| bbox[2 * k] == bbox[2 * k + 1]).collect();
    (flat.len() == 1).then(|| flat[0])
}

/// The shape a `slice_index` must have, `(n_boxes,)`: one plane per box.  A
/// column of the right length can be the wrong rank, so a caller holding an
/// array checks its shape here before flattening it.
pub fn check_slice_index_shape(shape: &[usize], n_boxes: usize) -> Option<String> {
    if shape.len() == 1 && shape[0] == n_boxes {
        return None;
    }
    let dims: Vec<i64> = shape.iter().map(|d| *d as i64).collect();
    Some(format!(
        "`slice_index` has shape {}, but it names one plane for each of {n_boxes} box(es), so its shape must be \
         ({n_boxes},)",
        crate::json::repr_int_tuple(&dims)
    ))
}

/// What a `slice_index` must be (§8.2); `None` when it is valid.
///
/// Shape is checked always; the range needs the index-space boxes and the
/// grid extent and is checked when both are given.
pub fn check_slice_index(
    planes: &[i64],
    n_boxes: usize,
    boxes: Option<&[Vec<f64>]>,
    shape: Option<&[usize]>,
) -> Option<String> {
    if let Some(problem) = check_slice_index_shape(&[planes.len()], n_boxes) {
        return Some(problem);
    }
    let (Some(boxes), Some(shape)) = (boxes, shape) else { return None };
    for (i, bbox) in boxes.iter().enumerate().take(n_boxes) {
        let Some(axis) = degenerate_axis(bbox) else { continue };
        let Some(extent) = shape.get(axis).map(|v| *v as i64) else {
            return Some(format!(
                "box {i} has {} axes, and the grid it names planes of has {}",
                bbox.len() / 2,
                shape.len()
            ));
        };
        let plane = planes[i];
        if !(0 <= plane && plane < extent) {
            return Some(format!(
                "box {i} names slice {plane} on an axis {extent} voxels deep; a plane outside the grid was clamped \
                 to the nearest edge, which silently moves the annotation to a different plane"
            ));
        }
    }
    None
}

/// Pack axis-aligned boxes, `(N, S, 2)` in `[lo, hi]` form (spec §8.2).
pub fn encode_boxes(boxes: &ArrayD<f64>, cols: &ObjectColumns, slice_index: Option<&[i64]>) -> Result<Payload> {
    let shape = boxes.shape();
    if shape.len() != 3 || shape[2] != 2 {
        return Err(Error::coded("E405", format!("boxes must have shape (N, S, 2), got {}", repr_int_tuple(shape))));
    }
    let stored = boxes.mapv(|v| v as f32);
    let (n, s) = (shape[0], shape[1]);
    // Held as stored: a bound float32 cannot hold is infinite there.
    let mut bad = 0;
    for i in 0..n {
        if (0..s).any(|k| !box_axis_ok(stored[[i, k, 0]], stored[[i, k, 1]])) {
            bad += 1;
        }
    }
    if bad > 0 {
        return Err(Error::coded(
            "E406",
            format!(
                "{bad} box(es) have lo > hi, or a bound that is not finite; boxes are stored [lo, hi] at voxel edges"
            ),
        ));
    }
    let mut p = Payload::new("boxes");
    p.datasets.insert("boxes".into(), NdArray::F32(stored).into());
    for (name, column) in object_columns(n, cols)? {
        p.datasets.insert(name, column);
    }
    if let Some(planes) = slice_index {
        if let Some(problem) = check_slice_index(planes, n, None, None) {
            return Err(Error::coded("E405", problem));
        }
        if let Some(plane) = planes.iter().find(|v| i32::try_from(**v).is_err()) {
            return Err(Error::coded(
                "E405",
                format!("`slice_index` names plane {plane}, which the int32 column cannot hold"),
            ));
        }
        p.datasets.insert(
            "slice_index".into(),
            NdArray::from_vec(&[planes.len()], planes.iter().map(|v| *v as i32).collect::<Vec<_>>())?.into(),
        );
    }
    p.class_ids = sorted_unique(cols.class_ids.iter().copied());
    Ok(p)
}

// -- obb ---------------------------------------------------------------------------

/// Pack oriented boxes as centre, **full** edge lengths and a rotation matrix (§8.3).
pub fn encode_obb(
    centers: &ArrayD<f64>,
    sizes: &ArrayD<f64>,
    rotations: &ArrayD<f64>,
    cols: &ObjectColumns,
) -> Result<Payload> {
    if centers.ndim() != 2 {
        return Err(Error::coded(
            "E405",
            format!(
                "obb centers must have shape (n, dim), got {}; for an empty collection pass correctly-shaped empty \
                 arrays, e.g. np.empty((0, 3)) with np.empty((0, 3, 3)) rotations",
                repr_int_tuple(centers.shape())
            ),
        ));
    }
    let (n, dim) = (centers.shape()[0], centers.shape()[1]);
    if sizes.shape() != [n, dim] || rotations.shape() != [n, dim, dim] {
        return Err(Error::coded(
            "E405",
            format!(
                "obb shapes disagree: centers {}, sizes {}, rotations {}",
                repr_int_tuple(centers.shape()),
                repr_int_tuple(sizes.shape()),
                repr_int_tuple(rotations.shape())
            ),
        ));
    }
    let c = centers.mapv(|v| v as f32);
    let s = sizes.mapv(|v| v as f32);
    let r = rotations.mapv(|v| v as f32);
    // `!(v >= 0)` holds a NaN too, which `v < 0` let through.
    if s.iter().any(|v| !(*v >= 0.0 && v.is_finite())) {
        return Err(Error::coded("E406", "obb sizes must be finite and non-negative"));
    }
    if c.iter().any(|v| !v.is_finite()) {
        return Err(Error::coded("E406", "obb centers must be finite"));
    }
    for i in 0..n {
        let mut m = Array2::<f64>::zeros((dim, dim));
        for a in 0..dim {
            for b in 0..dim {
                m[[a, b]] = r[[i, a, b]] as f64;
            }
        }
        if !is_proper_rotation(&m, ROTATION_TOL) {
            return Err(Error::coded(
                "E407",
                format!("obb {i}: `rotations` must be orthonormal with det = +1 to {}", format_g(ROTATION_TOL, 6)),
            ));
        }
    }
    let mut p = Payload::new("obb");
    p.datasets.insert("centers".into(), NdArray::F32(c).into());
    p.datasets.insert("sizes".into(), NdArray::F32(s).into());
    p.datasets.insert("rotations".into(), NdArray::F32(r).into());
    for (name, column) in object_columns(n, cols)? {
        p.datasets.insert(name, column);
    }
    p.class_ids = sorted_unique(cols.class_ids.iter().copied());
    Ok(p)
}

// -- keypoints ------------------------------------------------------------------------

/// Pack `(N, K, S)` keypoints with per-slot classes and visibility (§8.4).
pub fn encode_keypoints(
    points: &ArrayD<f64>,
    keypoint_class_ids: &[i64],
    cols: &ObjectColumns,
    visibility: Option<&ArrayD<i64>>,
    skeleton: Option<&str>,
) -> Result<Payload> {
    if points.ndim() != 3 {
        return Err(Error::coded(
            "E405",
            format!("keypoints must have shape (N, K, S), got {}", repr_int_tuple(points.shape())),
        ));
    }
    let (n, k) = (points.shape()[0], points.shape()[1]);
    if keypoint_class_ids.len() != k {
        return Err(Error::coded(
            "E405",
            format!("keypoint_class_ids has {} entries for {k} keypoint slots", keypoint_class_ids.len()),
        ));
    }
    let vis: ArrayD<u8> = match visibility {
        None => ArrayD::from_elem(IxDyn(&[n, k]), 2u8),
        Some(v) => {
            if v.shape() != [n, k] {
                return Err(Error::coded(
                    "E405",
                    format!("visibility {} must be (N, K) = ({n}, {k})", repr_int_tuple(v.shape())),
                ));
            }
            // Checked before the narrowing cast, which would wrap 256 to 0 and
            // 258 to 2 and store an out-of-range code as a valid one.
            if v.iter().any(|x| !(0..=2).contains(x)) {
                return Err(Error::coded(
                    "E411",
                    "visibility values must be 0 (unlabelled), 1 (occluded) or 2 (visible)",
                ));
            }
            v.mapv(|x| x as u8)
        }
    };
    let mut p = Payload::new("keypoints");
    p.datasets.insert("points".into(), NdArray::F32(points.mapv(|v| v as f32)).into());
    p.datasets.insert("visibility".into(), NdArray::U8(vis).into());
    p.datasets.insert("keypoint_class_ids".into(), u16_column(keypoint_class_ids)?.into());
    let cols = ObjectColumns {
        class_ids: cols.class_ids,
        instance_ids: cols.instance_ids,
        scores: cols.scores,
        attributes: None,
    };
    for (name, column) in object_columns(n, &cols)? {
        p.datasets.insert(name, column);
    }
    if let Some(s) = skeleton.filter(|s| !s.is_empty()) {
        p.attrs.push(("skeleton".into(), AttrValue::Str(s.to_string())));
    }
    p.class_ids = sorted_unique(cols.class_ids.iter().chain(keypoint_class_ids).copied());
    Ok(p)
}

// -- points --------------------------------------------------------------------------

fn per_element_len(name: &str, len: usize, n: usize, unit: &str) -> Result<()> {
    if len != n {
        return Err(Error::coded(
            "E405",
            format!(
                "{name} has length {len}, but the annotation holds {n} {unit}; it carries one value for each of them"
            ),
        ));
    }
    Ok(())
}

/// Pack a point set: landmarks, seeds, or one half of a correspondence (§8.5).
pub fn encode_points(
    points: &ArrayD<f64>,
    class_ids: Option<&[i64]>,
    instance_ids: Option<&[u64]>,
    names: Option<&[String]>,
    weights: Option<&[f64]>,
    correspondence: Option<&str>,
) -> Result<Payload> {
    if points.ndim() != 2 {
        return Err(Error::coded(
            "E405",
            format!("points must have shape (N, S), got {}", repr_int_tuple(points.shape())),
        ));
    }
    let n = points.shape()[0];
    let mut p = Payload::new("points");
    p.datasets.insert("points".into(), NdArray::F32(points.mapv(|v| v as f32)).into());
    if let Some(ids) = class_ids {
        let column = u16_column(ids)?;
        per_element_len("class_ids", ids.len(), n, "points")?;
        p.datasets.insert("class_ids".into(), column.into());
    }
    // The objects the points mark, as §10.6 asks of landmarks a track
    // follows: equal ids across visits make the landmark and the lesion one.
    if let Some(ids) = instance_ids {
        per_element_len("instance_ids", ids.len(), n, "points")?;
        p.datasets.insert("instance_ids".into(), PayloadData::Array(instance_id_array(ids)));
    }
    if let Some(names) = names {
        per_element_len("names", names.len(), n, "points")?;
        p.datasets.insert("names".into(), PayloadData::Strings(names.to_vec()));
    }
    if let Some(weights) = weights {
        per_element_len("weights", weights.len(), n, "points")?;
        p.datasets.insert(
            "weights".into(),
            NdArray::from_vec(&[n], weights.iter().map(|v| *v as f32).collect::<Vec<_>>())?.into(),
        );
    }
    if let Some(c) = correspondence.filter(|c| !c.is_empty()) {
        p.attrs.push(("correspondence".into(), AttrValue::Str(c.to_string())));
    }
    p.class_ids = class_ids.filter(|c| !c.is_empty()).map(|c| sorted_unique(c.iter().copied())).unwrap_or_default();
    Ok(p)
}

// -- contours -------------------------------------------------------------------------

/// One planar polygon handed to [`encode_contours`].
#[derive(Debug, Clone, PartialEq)]
pub struct Polygon {
    /// `(V, S)` vertices.
    pub vertices: ArrayD<f64>,
    pub class_id: i64,
    /// `(axis, index)` of the plane it lies in; `axis = -1` for out-of-plane.
    pub plane: (i64, i64),
    pub role: String,
}

impl Polygon {
    /// A validated polygon.
    pub fn new(vertices: ArrayD<f64>, class_id: i64, plane: (i64, i64), role: &str) -> Result<Polygon> {
        if !CONTOUR_ROLES.contains(&role) {
            return Err(Error::coded(
                "E411",
                format!("contour role {} must be one of {}", repr_str(role), repr_list(&CONTOUR_ROLES)),
            ));
        }
        Ok(Polygon { vertices, class_id, plane, role: role.to_string() })
    }
}

/// Concatenate planar polygons with an offset table (spec §8.6).
pub fn encode_contours(polygons: &[Polygon], ndim: Option<usize>) -> Result<Payload> {
    let mut p = Payload::new("contours");
    if polygons.is_empty() {
        let Some(ndim) = ndim else {
            return Err(Error::coded(
                "E410",
                "no polygons were supplied; an empty contours annotation needs ndim= to shape its vertex table",
            ));
        };
        p.datasets.insert("vertices".into(), NdArray::zeros(DType::F32, &[0, ndim]).into());
        p.datasets.insert("contour_offsets".into(), NdArray::zeros(DType::I64, &[1]).into());
        p.datasets.insert("contour_class_ids".into(), NdArray::zeros(DType::U16, &[0]).into());
        p.datasets.insert("contour_plane".into(), NdArray::zeros(DType::I32, &[0, 2]).into());
        p.datasets.insert("contour_role".into(), NdArray::zeros(DType::U8, &[0]).into());
        return Ok(p);
    }
    let dim = polygons[0].vertices.shape().get(1).copied().unwrap_or(0);
    for (i, poly) in polygons.iter().enumerate() {
        if poly.vertices.ndim() != 2 || poly.vertices.shape()[1] != dim {
            return Err(Error::coded(
                "E405",
                format!("polygon {i} has shape {}; expected (V, {dim})", repr_int_tuple(poly.vertices.shape())),
            ));
        }
    }
    let mut offsets = vec![0i64];
    let mut vertices: Vec<f32> = Vec::new();
    for poly in polygons {
        vertices.extend(poly.vertices.iter().map(|v| *v as f32));
        offsets.push(offsets.last().unwrap() + poly.vertices.shape()[0] as i64);
    }
    let total = *offsets.last().unwrap() as usize;
    let mut class_ids = Vec::new();
    for poly in polygons {
        class_ids.push(check_class_id(poly.class_id)?);
    }
    // `contour_plane` is int32 (§8.6); a value past its range is refused
    // rather than wrapped onto another plane by the cast.
    if let Some((i, poly)) = polygons
        .iter()
        .enumerate()
        .find(|(_, p)| i32::try_from(p.plane.0).is_err() || i32::try_from(p.plane.1).is_err())
    {
        return Err(Error::coded(
            "E405",
            format!(
                "polygon {i}'s plane ({}, {}) does not fit the int32 `contour_plane` stores it in",
                poly.plane.0, poly.plane.1
            ),
        ));
    }
    let planes: Vec<i32> = polygons.iter().flat_map(|p| [p.plane.0 as i32, p.plane.1 as i32]).collect();
    let roles: Vec<u8> =
        polygons.iter().map(|p| CONTOUR_ROLES.iter().position(|r| *r == p.role).unwrap_or(0) as u8).collect();
    let m = polygons.len();
    p.datasets.insert("vertices".into(), NdArray::from_vec(&[total, dim], vertices)?.into());
    p.datasets.insert("contour_offsets".into(), NdArray::from_vec(&[m + 1], offsets)?.into());
    p.datasets.insert("contour_class_ids".into(), NdArray::from_vec(&[m], class_ids)?.into());
    p.datasets.insert("contour_plane".into(), NdArray::from_vec(&[m, 2], planes)?.into());
    p.datasets.insert("contour_role".into(), NdArray::from_vec(&[m], roles)?.into());
    p.class_ids = sorted_unique(polygons.iter().map(|p| p.class_id));
    Ok(p)
}

// -- polylines (1.2 §3) ----------------------------------------------------------------

/// One open curve handed to [`encode_polylines`] (1.2 §3): ordered vertices,
/// each joined to the next, the last not joined to the first.
///
/// Built by [`Polyline::new`] and the [`Polyline::plane`] and
/// [`Polyline::instance_id`] builders, so it can grow without breaking a
/// caller.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct Polyline {
    /// `(V, S)` vertices, in order; at least two.
    pub vertices: ArrayD<f64>,
    pub class_id: i64,
    /// `(axis, index)` of the grid plane the curve lies in; `axis = -1`, the
    /// default, for none.
    pub plane: (i64, i64),
    /// The physical object the curve traces (1.0 §7.4).
    pub instance_id: Option<u64>,
}

impl Polyline {
    /// A curve through `vertices` (`(V, S)`), of class `class_id`, in no plane.
    pub fn new(vertices: ArrayD<f64>, class_id: i64) -> Polyline {
        Polyline { vertices, class_id, plane: (-1, 0), instance_id: None }
    }

    /// The grid plane the curve lies in: `axis` and the `index` along it.
    pub fn plane(mut self, axis: i64, index: i64) -> Polyline {
        self.plane = (axis, index);
        self
    }

    /// The object the curve traces, an id equal across annotations and
    /// timepoints for one object (1.0 §7.4).
    pub fn instance_id(mut self, id: u64) -> Polyline {
        self.instance_id = Some(id);
        self
    }
}

/// Concatenate open curves with an offset table (1.2 §3).
///
/// Each curve has at least two vertices (E404), every curve the same
/// spatial rank (E405), and either every curve an instance id or none
/// (E405).  `polyline_plane` is written when a curve names a plane.  An
/// empty annotation --- the verified negative --- needs `ndim` to shape its
/// vertex table.
pub fn encode_polylines(polylines: &[Polyline], ndim: Option<usize>) -> Result<Payload> {
    let mut p = Payload::new("polylines");
    if polylines.is_empty() {
        let Some(ndim) = ndim else {
            return Err(Error::coded(
                "E410",
                "no polylines were supplied; an empty polylines annotation needs ndim= to shape its vertex table",
            ));
        };
        p.datasets.insert("vertices".into(), NdArray::zeros(DType::F32, &[0, ndim]).into());
        p.datasets.insert("polyline_offsets".into(), NdArray::zeros(DType::I64, &[1]).into());
        p.datasets.insert("polyline_class_ids".into(), NdArray::zeros(DType::U16, &[0]).into());
        return Ok(p);
    }
    let dim = polylines[0].vertices.shape().get(1).copied().unwrap_or(0);
    for (i, line) in polylines.iter().enumerate() {
        if line.vertices.ndim() != 2 || line.vertices.shape()[1] != dim {
            return Err(Error::coded(
                "E405",
                format!("polyline {i} has shape {}; expected (V, {dim})", repr_int_tuple(line.vertices.shape())),
            ));
        }
        let n = line.vertices.shape()[0];
        if n < 2 {
            return Err(Error::coded(
                "E404",
                format!(
                    "polyline {i} has {n} vertex(es); a polyline joins each vertex to the next, so it has at least \
                     two (1.2 §3)"
                ),
            ));
        }
    }
    let mut class_ids = Vec::with_capacity(polylines.len());
    for line in polylines {
        class_ids.push(check_class_id(line.class_id)?);
    }
    // `polyline_plane` is int32 (1.2 §3); a value past its range is refused
    // rather than wrapped onto another plane by the cast.
    if let Some((i, line)) = polylines
        .iter()
        .enumerate()
        .find(|(_, l)| i32::try_from(l.plane.0).is_err() || i32::try_from(l.plane.1).is_err())
    {
        return Err(Error::coded(
            "E405",
            format!(
                "polyline {i}'s plane ({}, {}) does not fit the int32 `polyline_plane` stores it in",
                line.plane.0, line.plane.1
            ),
        ));
    }
    let named = polylines.iter().filter(|l| l.instance_id.is_some()).count();
    if named != 0 && named != polylines.len() {
        return Err(Error::coded(
            "E405",
            format!(
                "{named} of {} polylines carry an instance id; `instance_ids` holds one for each of them, so give \
                 every polyline one or none",
                polylines.len()
            ),
        ));
    }
    let mut offsets = vec![0i64];
    let mut vertices: Vec<f32> = Vec::new();
    for line in polylines {
        vertices.extend(line.vertices.iter().map(|v| *v as f32));
        offsets.push(offsets.last().unwrap() + line.vertices.shape()[0] as i64);
    }
    let total = *offsets.last().unwrap() as usize;
    let m = polylines.len();
    p.datasets.insert("vertices".into(), NdArray::from_vec(&[total, dim], vertices)?.into());
    p.datasets.insert("polyline_offsets".into(), NdArray::from_vec(&[m + 1], offsets)?.into());
    p.datasets.insert("polyline_class_ids".into(), NdArray::from_vec(&[m], class_ids)?.into());
    if polylines.iter().any(|l| l.plane.0 != -1) {
        let planes: Vec<i32> = polylines.iter().flat_map(|l| [l.plane.0 as i32, l.plane.1 as i32]).collect();
        p.datasets.insert("polyline_plane".into(), NdArray::from_vec(&[m, 2], planes)?.into());
    }
    if named > 0 {
        let ids: Vec<u64> = polylines.iter().filter_map(|l| l.instance_id).collect();
        p.datasets.insert("instance_ids".into(), PayloadData::Array(instance_id_array(&ids)));
    }
    p.class_ids = sorted_unique(polylines.iter().map(|l| l.class_id));
    Ok(p)
}

// -- mesh -----------------------------------------------------------------------------

/// Pack a triangle surface mesh (spec §8.7).
pub fn encode_mesh(
    vertices: &ArrayD<f64>,
    faces: &ArrayD<i64>,
    normals: Option<&ArrayD<f64>>,
    vertex_class_ids: Option<&[i64]>,
    mesh_offsets: Option<&[i64]>,
    mesh_class_ids: Option<&[i64]>,
) -> Result<Payload> {
    if vertices.ndim() != 2 || vertices.shape()[1] != 3 {
        return Err(Error::coded(
            "E405",
            format!("mesh vertices must have shape (V, 3), got {}", repr_int_tuple(vertices.shape())),
        ));
    }
    if faces.ndim() != 2 || faces.shape()[1] != 3 {
        return Err(Error::coded(
            "E405",
            format!("mesh faces must have shape (F, 3), got {}", repr_int_tuple(faces.shape())),
        ));
    }
    let n_vertices = vertices.shape()[0] as i64;
    // The range is checked on the indices as given, then narrowed: narrowing
    // first would wrap an index past 2**31 into the range it is checked
    // against.
    if faces.iter().any(|v| *v < 0 || *v >= n_vertices) {
        return Err(Error::coded("E405", format!("mesh faces index outside the {n_vertices} vertices")));
    }
    let f = faces.mapv(|v| v as i32);
    let mut p = Payload::new("mesh");
    p.datasets.insert("vertices".into(), NdArray::F32(vertices.mapv(|v| v as f32)).into());
    p.datasets.insert("faces".into(), NdArray::I32(f).into());
    if let Some(n) = normals {
        if n.shape() != vertices.shape() {
            return Err(Error::coded(
                "E405",
                format!(
                    "normals {} must match vertices {}",
                    repr_int_tuple(n.shape()),
                    repr_int_tuple(vertices.shape())
                ),
            ));
        }
        p.datasets.insert("normals".into(), NdArray::F32(n.mapv(|v| v as f32)).into());
    }
    if let Some(ids) = vertex_class_ids {
        let column = u16_column(ids)?;
        per_element_len("vertex_class_ids", ids.len(), n_vertices as usize, "vertices")?;
        p.datasets.insert("vertex_class_ids".into(), column.into());
    }
    if let Some(offsets) = mesh_offsets {
        check_mesh_offsets(offsets, faces.shape()[0])?;
        p.datasets.insert("mesh_offsets".into(), NdArray::from_vec(&[offsets.len()], offsets.to_vec())?.into());
    }
    if let Some(ids) = mesh_class_ids {
        let column = u16_column(ids)?;
        let meshes = mesh_offsets.map(|o| o.len().saturating_sub(1)).unwrap_or(1);
        per_element_len("mesh_class_ids", ids.len(), meshes, "meshes")?;
        p.datasets.insert("mesh_class_ids".into(), column.into());
    }
    let mut declared = Vec::new();
    declared.extend(vertex_class_ids.unwrap_or(&[]));
    declared.extend(mesh_class_ids.unwrap_or(&[]));
    p.class_ids = sorted_unique(declared);
    Ok(p)
}

/// Refuse `mesh_offsets` that do not delimit the faces (§8.7): mesh *m* is
/// faces `[o[m], o[m+1])`, so the table has an entry (E405), never falls or
/// starts below 0, and ends at the number of faces (E408) --- as a file is
/// validated, here rather than at commit.
fn check_mesh_offsets(offsets: &[i64], n_faces: usize) -> Result<()> {
    let Some(last) = offsets.last() else {
        return Err(Error::coded(
            "E405",
            "`mesh_offsets` is empty; an offset table starts at 0, so it has at least one entry",
        ));
    };
    if offsets[0] < 0 || offsets.windows(2).any(|w| w[1] < w[0]) {
        return Err(Error::coded(
            "E408",
            format!("`mesh_offsets` {} are not monotonic from 0", crate::json::repr_int_list(offsets)),
        ));
    }
    if *last != n_faces as i64 {
        return Err(Error::coded(
            "E408",
            format!("last offset {last} != faces length {n_faces}; mesh m is faces [o[m], o[m+1])"),
        ));
    }
    Ok(())
}

// -- classification ---------------------------------------------------------------------

/// Label assertions as columns: classes, values, and the optional per-assertion
/// scope ids, schemes and scheme values (§9).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Assertions {
    pub class_ids: Vec<i64>,
    pub values: Vec<f64>,
    pub scope_ids: Option<Vec<i64>>,
    pub schemes: Option<Vec<String>>,
    pub scheme_values: Option<Vec<String>>,
}

/// Pack label assertions (spec §9).
///
/// `multilabel = false` means exactly one positive class per scope unit, and
/// that is checked here rather than left to a validator.
pub fn encode_classification(assertions: &Assertions, scope: &str, multilabel: bool) -> Result<Payload> {
    check_scope(scope)?;
    let mut class_ids = Vec::new();
    for c in &assertions.class_ids {
        class_ids.push(check_class_id(*c)?);
    }
    let values = &assertions.values;
    if values.len() != class_ids.len() {
        return Err(Error::coded(
            "E405",
            format!("values has {} entries for {} assertions", values.len(), class_ids.len()),
        ));
    }
    if values.iter().any(|v| !(0.0..=1.0).contains(v)) {
        return Err(Error::coded(
            "E404",
            "classification values must lie in [0, 1]; 1.0 is a hard positive, 0.0 an explicit negative",
        ));
    }
    if !multilabel {
        let units: Vec<Option<i64>> = match &assertions.scope_ids {
            Some(ids) => ids.iter().map(|v| Some(*v)).collect(),
            None => vec![None; class_ids.len()],
        };
        if units.len() != class_ids.len() {
            return Err(Error::coded(
                "E405",
                format!("scope_ids has {} entries for {} assertions", units.len(), class_ids.len()),
            ));
        }
        let mut by_unit: BTreeMap<Option<i64>, usize> = BTreeMap::new();
        for (unit, value) in units.iter().zip(values) {
            if *value > 0.0 {
                *by_unit.entry(*unit).or_default() += 1;
            }
        }
        let mut crowded: Vec<String> = by_unit
            .iter()
            .filter(|(_, n)| **n > 1)
            .map(|(k, _)| k.map(|v| v.to_string()).unwrap_or_else(|| "None".into()))
            .collect();
        crowded.sort();
        if !crowded.is_empty() {
            return Err(Error::coded(
                "E404",
                format!(
                    "multilabel=False allows one positive class per scope unit, but unit(s) {} carry several",
                    repr_list(&crowded)
                ),
            ));
        }
    }
    let k = class_ids.len();
    let mut p = Payload::new("classification");
    p.datasets.insert("class_ids".into(), NdArray::from_vec(&[k], class_ids.clone())?.into());
    p.datasets.insert(
        "values".into(),
        NdArray::from_vec(&[values.len()], values.iter().map(|v| *v as f32).collect::<Vec<_>>())?.into(),
    );
    if let Some(ids) = &assertions.scope_ids {
        if ids.len() != k {
            return Err(Error::coded("E405", format!("scope_ids has {} entries for {k} assertions", ids.len())));
        }
        p.datasets.insert("scope_ids".into(), NdArray::from_vec(&[k], ids.clone())?.into());
    }
    for (name, column) in [("schemes", &assertions.schemes), ("scheme_values", &assertions.scheme_values)] {
        if let Some(values) = column {
            if values.len() != k {
                return Err(Error::coded("E405", format!("{name} has {} entries for {k} assertions", values.len())));
            }
            p.datasets.insert(name.into(), PayloadData::Strings(values.clone()));
        }
    }
    p.attrs.push(("scope".into(), AttrValue::Str(scope.to_string())));
    p.attrs.push(("multilabel".into(), AttrValue::Bool(multilabel)));
    p.class_ids = sorted_unique(class_ids.iter().map(|c| *c as i64));
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boxes_refuse_inverted() {
        let boxes = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![3.0, 1.0, 0.0, 1.0]).unwrap();
        let cols = ObjectColumns { class_ids: &[1], instance_ids: None, scores: None, attributes: None };
        assert_eq!(encode_boxes(&boxes, &cols, None).unwrap_err().code(), Some("E406"));
    }

    #[test]
    fn single_label_refuses_two_positives() {
        let a = Assertions { class_ids: vec![1, 2], values: vec![1.0, 1.0], ..Default::default() };
        assert_eq!(encode_classification(&a, "sample", false).unwrap_err().code(), Some("E404"));
        assert!(encode_classification(&a, "sample", true).is_ok());
    }

    #[test]
    fn s9_one_value_per_assertion() {
        for values in [vec![1.0], vec![1.0, 0.0, 0.0]] {
            let a = Assertions { class_ids: vec![1, 2], values, ..Default::default() };
            assert_eq!(encode_classification(&a, "sample", true).unwrap_err().code(), Some("E405"));
        }
    }

    fn line(rows: usize) -> Polyline {
        Polyline::new(ArrayD::from_shape_fn(IxDyn(&[rows, 3]), |i| (i[0] * 3 + i[1]) as f64), 2)
    }

    #[test]
    fn v1_2_s3_polylines_concatenate_with_offsets() {
        let p = encode_polylines(&[line(2), line(3).plane(0, 4)], None).unwrap();
        assert_eq!(p.kind, "polylines");
        assert_eq!(p.array("polyline_offsets").unwrap().to_f64().iter().copied().collect::<Vec<_>>(), [0.0, 2.0, 5.0]);
        assert_eq!(p.array("vertices").unwrap().shape(), [5, 3]);
        assert_eq!(
            p.array("polyline_plane").unwrap().to_f64().iter().copied().collect::<Vec<_>>(),
            [-1.0, 0.0, 0.0, 4.0]
        );
        assert!(!p.datasets.contains_key("instance_ids"));
        assert_eq!(p.class_ids, [2]);
        // No plane named, no plane column; ids for every curve or none.
        let p = encode_polylines(&[line(2).instance_id(7), line(2).instance_id(9)], None).unwrap();
        assert!(!p.datasets.contains_key("polyline_plane"));
        assert_eq!(p.array("instance_ids").unwrap().shape(), [2]);
        let refused = encode_polylines(&[line(2).instance_id(7), line(2)], None).unwrap_err();
        assert_eq!(refused.code(), Some("E405"));
    }

    #[test]
    fn v1_2_s3_a_polyline_has_two_vertices_and_one_rank() {
        assert_eq!(encode_polylines(&[line(1)], None).unwrap_err().code(), Some("E404"));
        let flat = Polyline::new(ArrayD::zeros(IxDyn(&[2, 2])), 1);
        assert_eq!(encode_polylines(&[line(2), flat], None).unwrap_err().code(), Some("E405"));
        assert_eq!(encode_polylines(&[], None).unwrap_err().code(), Some("E410"));
        let empty = encode_polylines(&[], Some(3)).unwrap();
        assert_eq!(empty.array("vertices").unwrap().shape(), [0, 3]);
        assert_eq!(encode_polylines(&[line(2).plane(0, 1i64 << 33)], None).unwrap_err().code(), Some("E405"));
    }

    #[test]
    fn s8_4_s8_6_s8_7_values_are_checked_before_they_are_narrowed() {
        let cols = ObjectColumns { class_ids: &[1], instance_ids: None, scores: None, attributes: None };
        let points = ArrayD::<f64>::zeros(IxDyn(&[1, 2, 3]));
        for codes in [[256, 2], [2, 258], [-1, 0]] {
            let visibility = ArrayD::from_shape_vec(IxDyn(&[1, 2]), codes.to_vec()).unwrap();
            let refused = encode_keypoints(&points, &[1, 2], &cols, Some(&visibility), None).unwrap_err();
            assert_eq!(refused.code(), Some("E411"), "{codes:?}");
        }
        let vertices = ArrayD::<f64>::zeros(IxDyn(&[3, 3]));
        let faces = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![0, 1, (1i64 << 32) + 2]).unwrap();
        let refused = encode_mesh(&vertices, &faces, None, None, None, None).unwrap_err();
        assert_eq!(refused.code(), Some("E405"));
        let polygon =
            |plane| Polygon { vertices: ArrayD::zeros(IxDyn(&[3, 3])), class_id: 1, plane, role: "outer".into() };
        let refused = encode_contours(&[polygon((0, (1i64 << 32) + 3))], Some(3)).unwrap_err();
        assert_eq!(refused.code(), Some("E405"));
        assert!(encode_contours(&[polygon((0, 3))], Some(3)).is_ok());
        let boxes = ArrayD::from_shape_vec(IxDyn(&[1, 3, 2]), vec![2.0, 2.0, 0.0, 1.0, 0.0, 1.0]).unwrap();
        let refused = encode_boxes(&boxes, &cols, Some(&[(1i64 << 32) + 2])).unwrap_err();
        assert_eq!(refused.code(), Some("E405"));
    }
}
