//! §8--§9 --- the geometric kinds and classification (E4xx).

use std::collections::BTreeMap;

use ndarray::ArrayD;

use super::{grid_shape, loc, read_f64, read_i64, str_attr, sub_dataset, Context};
use crate::annotations::encode_geometric::{check_slice_index, ROTATION_TOL, SCOPES, SPACES};
use crate::annotations::header::GEOMETRIC_KINDS;
use crate::geometry::affine::is_proper_rotation;
use crate::h5::attrs;
use crate::h5::ops::{self, Node};
use crate::json::{repr_int_list, repr_str};
use crate::validate::Diagnostic;
use crate::Result;

/// `consistent` is whether the annotation's per-object columns agree in
/// shape (`layout::column_problems`); a rule pairing one column's rows with
/// another's runs only then.
pub(super) fn check_geometric(
    ctx: &Context,
    name: &str,
    group: &Node,
    kind: &str,
    grid_id: Option<&str>,
    grids: Option<&hdf5::Group>,
    consistent: bool,
) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    if !GEOMETRIC_KINDS.contains(&kind) {
        return Ok(out);
    }
    let location = format!("/annotations/{name}");
    let a = loc(group);
    let space = str_attr(a, "space")?;
    match space.as_deref() {
        None => {
            out.push(ctx.err("E412", location.clone(), format!("kind {} requires a `space` attribute", repr_str(kind))))
        }
        Some(s) if !SPACES.contains(&s) => {
            out.push(ctx.err("E412", location.clone(), format!("unknown space {}", repr_str(s))))
        }
        Some("index") if grid_id.is_none() => {
            out.push(ctx.err("E412", location.clone(), "space='index' names a grid's coordinates, but no `grid`"))
        }
        Some("world") if !attrs::has(a, "frame_uid") => {
            out.push(ctx.err("E412", location.clone(), "space='world' names a physical frame, but no `frame_uid`"))
        }
        _ => {}
    }
    if let (Some(s), Some(gid), Some(g)) = (space.as_deref(), grid_id, grids) {
        if s != "index" && ops::exists(g, gid) {
            let grid_group = g.group(gid)?;
            let units = str_attr(&grid_group, "units")?.unwrap_or_else(|| "mm".into());
            if units == "px" {
                out.push(ctx.err(
                    "E414",
                    location.clone(),
                    format!(
                        "grid {} is uncalibrated (units='px'), so a geometric annotation on it must use space='index'",
                        repr_str(gid)
                    ),
                ));
            }
            // World coordinates in a frame the grid is not in are read
            // through the grid's affine all the same (§3.4).
            if let (Some(named), Some(own)) = (str_attr(a, "frame_uid")?, str_attr(&grid_group, "frame_uid")?) {
                if s == "world" && named != own {
                    out.push(ctx.err(
                        "E414",
                        location.clone(),
                        format!(
                            "space='world' names frame {}, but grid {} is in frame {}; its affine would read one \
                             frame's coordinates as the other's",
                            repr_str(&named),
                            repr_str(gid),
                            repr_str(&own)
                        ),
                    ));
                }
            }
        }
    }
    if kind == "boxes" {
        if let Some(ds) = sub_dataset(group, "boxes") {
            let boxes = read_f64(&ds)?;
            if boxes.ndim() == 3 && !boxes.is_empty() {
                let bad = bad_boxes(&boxes);
                if bad > 0 {
                    out.push(ctx.err("E406", location.clone(), format!("{bad} box(es) have lo > hi")));
                }
            }
            if let (Some(planes_ds), true) = (sub_dataset(group, "slice_index"), consistent) {
                let planes: Vec<i64> = read_i64(&planes_ds)?.iter().copied().collect();
                let n = boxes.shape()[0];
                let dims = boxes.shape()[1];
                let mut spatial: Option<Vec<usize>> = None;
                if space.as_deref() == Some("index") {
                    if let (Some(gid), Some(g)) = (grid_id, grids) {
                        if ops::exists(g, gid) {
                            let full = grid_shape(g, gid)?;
                            if full.len() >= dims {
                                spatial = Some(full[full.len() - dims..].iter().map(|v| *v as usize).collect());
                            }
                        }
                    }
                }
                let rows: Vec<Vec<f64>> =
                    (0..n).map(|i| boxes.index_axis(ndarray::Axis(0), i).iter().copied().collect()).collect();
                let problem = check_slice_index(
                    &planes,
                    n,
                    if spatial.is_some() { Some(&rows) } else { None },
                    spatial.as_deref(),
                );
                if let Some(p) = problem {
                    out.push(ctx.err("E405", location.clone(), p));
                }
            }
        }
    }
    if kind == "obb" {
        if let Some(ds) = sub_dataset(group, "rotations") {
            let rotations = read_f64(&ds)?;
            let mut offenders = Vec::new();
            if rotations.ndim() == 3 {
                for (i, r) in rotations.outer_iter().enumerate() {
                    let m = r.to_owned().into_dimensionality::<ndarray::Ix2>()?;
                    if !is_proper_rotation(&m, ROTATION_TOL) {
                        offenders.push(i);
                    }
                }
            }
            if !offenders.is_empty() {
                out.push(ctx.err(
                    "E407",
                    location.clone(),
                    format!(
                        "{} rotation(s) are not proper rotations (orthonormal with det = +1); first at index {}",
                        offenders.len(),
                        offenders[0]
                    ),
                ));
            }
            if let Some(sizes) = sub_dataset(group, "sizes") {
                if read_f64(&sizes)?.iter().any(|v| *v < 0.0) {
                    out.push(ctx.err("E406", location.clone(), "`sizes` must be non-negative edge lengths"));
                }
            }
        }
    }
    if kind == "keypoints" {
        out.extend(check_keypoints(ctx, name, group)?);
    }
    if kind == "points" {
        if let Some(target) = str_attr(a, "correspondence")? {
            let exists = ops::child_group(&ctx.root, "annotations").is_some_and(|g| ops::exists(&g, &target));
            if !exists {
                out.push(ctx.err(
                    "E413",
                    location.clone(),
                    format!("`correspondence` names annotation {}, which does not exist", repr_str(&target)),
                ));
            }
        }
    }
    if kind == "contours" {
        out.extend(check_offsets(ctx, name, group, "contour_offsets", "vertices")?);
    }
    if kind == "mesh" {
        out.extend(check_mesh(ctx, name, group)?);
    }
    Ok(out)
}

pub(super) fn bad_boxes(boxes: &ArrayD<f64>) -> usize {
    boxes.outer_iter().filter(|b| b.outer_iter().any(|axis| axis.len() == 2 && axis[0] > axis[1])).count()
}

/// The shapes of `points`, `visibility` and `keypoint_class_ids` are the
/// column table's (E405); what is left is what their values mean.
fn check_keypoints(ctx: &Context, name: &str, group: &Node) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let location = format!("/annotations/{name}");
    if let Some(vis) = sub_dataset(group, "visibility") {
        if read_i64(&vis)?.iter().any(|v| !(0..=2).contains(v)) {
            out.push(ctx.err(
                "E411",
                format!("{location}/visibility"),
                "values must be 0 (unlabelled), 1 (occluded) or 2 (visible)",
            ));
        }
    }
    if let Some(skeleton) = str_attr(loc(group), "skeleton")? {
        let known = ctx
            .document
            .as_ref()
            .and_then(|d| d.label_set.as_ref())
            .is_some_and(|ls| ls.skeletons.iter().any(|s| s.id == skeleton));
        if !known {
            out.push(ctx.err(
                "E413",
                location,
                format!("`skeleton` names {}, which the label set does not declare", repr_str(&skeleton)),
            ));
        }
    }
    Ok(out)
}

fn check_offsets(ctx: &Context, name: &str, group: &Node, offsets_name: &str, target: &str) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let Some(ds) = sub_dataset(group, offsets_name) else { return Ok(out) };
    let offsets: Vec<i64> = read_i64(&ds)?.iter().copied().collect();
    let location = format!("/annotations/{name}/{offsets_name}");
    if offsets.windows(2).any(|w| w[1] < w[0]) {
        out.push(ctx.err("E408", location.clone(), "offsets are not monotonically increasing"));
    }
    if let (Some(last), Some(t)) = (offsets.last(), sub_dataset(group, target)) {
        let len = t.shape().first().copied().unwrap_or(0) as i64;
        if *last != len {
            out.push(ctx.err("E408", location, format!("last offset {last} != {target} length {len}")));
        }
    }
    Ok(out)
}

fn check_mesh(ctx: &Context, name: &str, group: &Node) -> Result<Vec<Diagnostic>> {
    let mut out = check_offsets(ctx, name, group, "mesh_offsets", "faces")?;
    let location = format!("/annotations/{name}");
    let (Some(vertices), Some(faces)) = (sub_dataset(group, "vertices"), sub_dataset(group, "faces")) else {
        return Ok(out);
    };
    let n_vertices = vertices.shape().first().copied().unwrap_or(0) as i64;
    let f = read_i64(&faces)?;
    if !f.is_empty() {
        let lo = f.iter().copied().min().unwrap_or(0);
        let hi = f.iter().copied().max().unwrap_or(0);
        if lo < 0 || hi >= n_vertices {
            out.push(ctx.err(
                "E405",
                format!("{location}/faces"),
                format!("face indices reach outside the {n_vertices} vertices"),
            ));
        }
    }
    Ok(out)
}

/// The columns' shapes are the column table's (E405); `consistent` says they
/// agree, which the one-positive-per-unit rule needs to pair units with
/// values.
pub(super) fn check_classification(
    ctx: &Context,
    name: &str,
    group: &Node,
    kind: &str,
    consistent: bool,
) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    if kind != "classification" {
        return Ok(out);
    }
    let location = format!("/annotations/{name}");
    let a = loc(group);
    let scope = str_attr(a, "scope")?;
    match scope.as_deref() {
        None => out.push(ctx.err("E412", location.clone(), "`classification` requires a `scope` attribute")),
        Some(s) if !SCOPES.contains(&s) => {
            out.push(ctx.err("E412", location.clone(), format!("unknown classification scope {}", repr_str(s))))
        }
        _ => {}
    }
    let Some(vds) = sub_dataset(group, "values") else {
        return Ok(out);
    };
    let values: Vec<f64> = read_f64(&vds)?.iter().copied().collect();
    // Every value is tested: a fold over `f64::min`/`max` skips NaN, so a
    // value that is no number passed where the writer refuses it.
    if values.iter().any(|v| !(0.0..=1.0).contains(v)) {
        out.push(ctx.err(
            "E404",
            location.clone(),
            "values must lie in [0, 1]; 1.0 is a hard positive, 0.0 an explicit negative",
        ));
    }
    let scope_ids: Option<Vec<i64>> = match sub_dataset(group, "scope_ids") {
        Some(ds) => Some(read_i64(&ds)?.iter().copied().collect()),
        None => None,
    };
    let multilabel = attrs::get_bool(a, "multilabel")?.unwrap_or(true);
    if consistent && !multilabel && !values.is_empty() {
        let units: Vec<i64> = scope_ids.clone().unwrap_or_else(|| vec![0; values.len()]);
        let mut positives: BTreeMap<i64, usize> = BTreeMap::new();
        for (u, v) in units.iter().zip(&values) {
            if *v > 0.0 {
                *positives.entry(*u).or_default() += 1;
            }
        }
        let crowded: Vec<i64> = positives.into_iter().filter(|(_, n)| *n > 1).map(|(u, _)| u).collect();
        if !crowded.is_empty() {
            out.push(ctx.err(
                "E404",
                location.clone(),
                format!(
                    "multilabel=false allows one positive class per scope unit, but unit(s) {} carry several",
                    repr_int_list(&crowded)
                ),
            ));
        }
    }
    if let (Some("timepoint"), Some(ids), Some(doc)) = (scope.as_deref(), &scope_ids, &ctx.document) {
        let declared = doc.timepoints.len() as i64;
        let mut unknown: Vec<i64> = ids.iter().copied().filter(|v| !(0..declared).contains(v)).collect();
        unknown.sort_unstable();
        unknown.dedup();
        if !unknown.is_empty() {
            out.push(ctx.err(
                "E409",
                location.clone(),
                format!(
                    "scope='timepoint' scope_ids {} are not timepoint indices (0..{})",
                    repr_int_list(&unknown),
                    declared - 1
                ),
            ));
        }
    }
    Ok(out)
}
