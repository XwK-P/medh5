//! §8--§9 --- the geometric kinds and classification (E4xx).

use std::collections::BTreeMap;

use ndarray::ArrayD;

use super::{grid_shape, loc, read_f64, read_i64, str_attr, sub_dataset, Context};
use crate::annotations::depth;
use crate::annotations::encode_geometric::{check_slice_index, ROTATION_TOL, SCOPES, SPACES};
use crate::annotations::header::is_geometric_kind;
use crate::geometry::affine::is_proper_rotation;
use crate::h5::attrs;
use crate::h5::ops::{self, Node};
use crate::json::{repr_int_list, repr_str};
use crate::labels::{ClassKey, LabelSet};
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
    if !is_geometric_kind(kind) {
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
            out.push(ctx.unknown("E412", location.clone(), format!("unknown space {}", repr_str(s))))
        }
        Some("index") if grid_id.is_none() => {
            out.push(ctx.err("E412", location.clone(), "space='index' names a grid's coordinates, but no `grid`"))
        }
        Some("world") if !attrs::has(a, "frame_uid") => {
            out.push(ctx.err("E412", location.clone(), "space='world' names a physical frame, but no `frame_uid`"))
        }
        _ => {}
    }
    // A later minor's space is that minor's to place: what it means on this
    // grid is not this validator's to judge.
    let judged = space.as_deref().filter(|s| SPACES.contains(s) || !ctx.projection);
    if let (Some(s), Some(gid), Some(g)) = (judged, grid_id, grids) {
        if s != "index" && ops::is_group(g, gid) {
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
                    out.push(ctx.err("E406", location.clone(), bad_boxes_message(bad)));
                }
            }
            if let (Some(planes_ds), true) = (sub_dataset(group, "slice_index"), consistent) {
                let planes: Vec<i64> = read_i64(&planes_ds)?.iter().copied().collect();
                let n = boxes.shape()[0];
                let dims = boxes.shape()[1];
                let mut spatial: Option<Vec<usize>> = None;
                if space.as_deref() == Some("index") {
                    if let (Some(gid), Some(g)) = (grid_id, grids) {
                        if ops::is_group(g, gid) {
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
        }
        // A NaN is no length: `v < 0.0` let one through.
        if let Some(sizes) = sub_dataset(group, "sizes") {
            if read_f64(&sizes)?.iter().any(|v| !(v.is_finite() && *v >= 0.0)) {
                out.push(ctx.err("E406", location.clone(), "`sizes` must be finite, non-negative edge lengths"));
            }
        }
        // Nor a place: a box centred at NaN is nowhere (§8.3).
        if let Some(centers) = sub_dataset(group, "centers") {
            if read_f64(&centers)?.iter().any(|v| !v.is_finite()) {
                out.push(ctx.err("E406", location.clone(), "`centers` must be finite"));
            }
        }
    }
    if kind == "keypoints" {
        out.extend(check_keypoints(ctx, name, group)?);
    }
    if kind == "points" {
        if let Some(target) = str_attr(a, "correspondence")? {
            let exists = ops::child_group(&ctx.root, "annotations").is_some_and(|g| ops::is_group(&g, &target));
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
        let offsets = check_offsets(ctx, name, group, "contour_offsets", "vertices")?;
        let delimited = offsets.is_empty();
        out.extend(offsets);
        out.extend(check_contour_roles(ctx, name, group, consistent && delimited)?);
    }
    if kind == "polylines" {
        out.extend(check_polylines(ctx, name, group, consistent)?);
    }
    if kind == "mesh" {
        out.extend(check_mesh(ctx, name, group)?);
    }
    Ok(out)
}

/// 1.2 §3: `polyline_offsets` start at 0, never decrease and end at the
/// vertex count (E408), and once they delimit the vertices, each polyline
/// spans at least two of them (E404).
fn check_polylines(ctx: &Context, name: &str, group: &Node, consistent: bool) -> Result<Vec<Diagnostic>> {
    let mut out = check_offsets(ctx, name, group, "polyline_offsets", "vertices")?;
    let Some(ds) = sub_dataset(group, "polyline_offsets") else { return Ok(out) };
    let offsets: Vec<i64> = read_i64(&ds)?.iter().copied().collect();
    let location = format!("/annotations/{name}/polyline_offsets");
    if let Some(first) = offsets.first().filter(|first| **first != 0) {
        out.push(ctx.err("E408", location, format!("first offset {first} != 0; polyline 0 starts at vertex 0")));
    }
    if !out.is_empty() || !consistent {
        return Ok(out);
    }
    let short: Vec<usize> = offsets.windows(2).enumerate().filter(|(_, w)| w[1] - w[0] < 2).map(|(i, _)| i).collect();
    if let Some(first) = short.first() {
        out.push(ctx.err(
            "E404",
            format!("/annotations/{name}"),
            format!(
                "{} polyline(s) span fewer than two vertices, first polyline {first} with {}; a polyline joins each \
                 vertex to the next, so it has at least two (1.2 §3)",
                short.len(),
                offsets[first + 1] - offsets[*first]
            ),
        ));
    }
    Ok(out)
}

/// How many boxes have an axis whose `lo` exceeds its `hi`, or whose bounds
/// are not both finite (§8.2, E406).  `lo > hi` is false for a NaN, so a box
/// of NaN passed, and a reader cropped nothing or everything from it.
pub(super) fn bad_boxes(boxes: &ArrayD<f64>) -> usize {
    boxes
        .outer_iter()
        .filter(|b| {
            b.outer_iter()
                .any(|axis| axis.len() == 2 && !(axis[0].is_finite() && axis[1].is_finite() && axis[0] <= axis[1]))
        })
        .count()
}

/// E406's message for `bad` boxes.
pub(super) fn bad_boxes_message(bad: usize) -> String {
    format!("{bad} box(es) have lo > hi or a bound that is not finite")
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
    // Without a document there is no label set to resolve it against: that
    // is `/meta`'s finding (E004, E005), not this annotation's.
    if let (Some(skeleton), Some(doc)) = (str_attr(loc(group), "skeleton")?, &ctx.document) {
        let known = doc.label_set.as_ref().is_some_and(|ls| ls.skeletons.iter().any(|s| s.id == skeleton));
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

/// §8.6: a `contour_role` is `0` or `1` (E404), and **SHOULD** be the
/// parity of the polygon's depth among its class's polygons in its plane
/// (W919), which readers take instead.  The depths need the columns to
/// agree and the offsets to delimit `vertices` (`paired`); a role the format
/// does not define is reported whatever they hold.
fn check_contour_roles(ctx: &Context, name: &str, group: &Node, paired: bool) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let Some(ds) = sub_dataset(group, "contour_role") else { return Ok(out) };
    let roles: Vec<i64> = read_i64(&ds)?.iter().copied().collect();
    let location = format!("/annotations/{name}/contour_role");
    let unknown: Vec<usize> = (0..roles.len()).filter(|i| !(0..=1).contains(&roles[*i])).collect();
    if let Some(first) = unknown.first() {
        out.push(ctx.err(
            "E404",
            location.clone(),
            format!(
                "{} polygon(s) carry a contour_role other than 0 (outer) or 1 (hole), first polygon {first} with {}",
                unknown.len(),
                roles[*first]
            ),
        ));
    }
    let (Some(vertices), Some(offsets), Some(classes), Some(planes), true) = (
        sub_dataset(group, "vertices"),
        sub_dataset(group, "contour_offsets"),
        sub_dataset(group, "contour_class_ids"),
        sub_dataset(group, "contour_plane"),
        paired,
    ) else {
        return Ok(out);
    };
    let Ok(vertices) = read_f64(&vertices)?.into_dimensionality::<ndarray::Ix2>() else { return Ok(out) };
    let offsets: Vec<i64> = read_i64(&offsets)?.iter().copied().collect();
    let classes: Vec<i64> = read_i64(&classes)?.iter().copied().collect();
    let planes = read_i64(&planes)?;
    if planes.ndim() != 2 || planes.shape()[1] != 2 {
        return Ok(out);
    }
    let planes: Vec<(i64, i64)> = planes.outer_iter().map(|p| (p[0], p[1])).collect();
    let depths = depth::contour_depths(vertices.view(), &offsets, &classes, &planes);
    let disagree: Vec<(usize, usize)> = depths
        .iter()
        .enumerate()
        .filter_map(|(i, d)| Some((i, (*d)?)))
        .filter(|(i, d)| roles.get(*i).is_some_and(|r| (0..=1).contains(r) && *r as usize != d % 2))
        .collect();
    if let Some((first, d)) = disagree.first() {
        out.push(ctx.err(
            "W919",
            location,
            format!(
                "{} polygon(s) carry a contour_role that disagrees with their depth among their class's polygons \
                 in their plane, first polygon {first}: role {} at depth {d}; a reader takes the depth (§8.6)",
                disagree.len(),
                roles[*first]
            ),
        ));
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
            out.push(ctx.unknown("E412", location.clone(), format!("unknown classification scope {}", repr_str(s))))
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

/// Semantic: under `closure = "implicit"`, a class asserted absent where an
/// `is_a` descendant is asserted present, in one scope unit (§5.4, §9, W916).
///
/// Readers answer the class positive (`state()`) while its stored value says
/// otherwise.  Not for a class examined and not asserted --- §9's pattern ---
/// nor for a soft value above 0 below its descendant's, and not without a
/// label set to read the hierarchy from (one with a cycle or an unknown
/// parent, E304 or E306, does not parse).
pub fn check_implicit_closure(ctx: &mut Context) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let Some(labels) = ctx.document.as_ref().and_then(|d| d.label_set.as_ref()) else { return Ok(out) };
    for (name, group) in ctx.children("annotations")? {
        let a = loc(&group);
        if str_attr(a, "kind")?.as_deref() != Some("classification")
            || str_attr(a, "closure")?.as_deref() != Some("implicit")
        {
            continue;
        }
        let (Some(cds), Some(vds)) = (sub_dataset(&group, "class_ids"), sub_dataset(&group, "values")) else {
            continue;
        };
        let classes: Vec<i64> = read_i64(&cds)?.iter().copied().collect();
        let values: Vec<f64> = read_f64(&vds)?.iter().copied().collect();
        let units: Option<Vec<i64>> = match sub_dataset(&group, "scope_ids") {
            Some(ds) => Some(read_i64(&ds)?.iter().copied().collect()),
            None => None,
        };
        // Columns that disagree in length are E404's, not this rule's.
        if values.len() != classes.len() || units.as_ref().is_some_and(|u| u.len() != classes.len()) {
            continue;
        }
        let found = contradictions(labels, &classes, &values, units.as_deref())?;
        let Some(first) = found.first() else { continue };
        let more = match found.len() {
            1 => String::new(),
            n => format!(" ({} more such pair(s))", n - 1),
        };
        let unit = match first.unit {
            Some(u) => format!("in scope unit {u}"),
            None => "in the annotation".into(),
        };
        out.push(ctx.err(
            "W916",
            format!("/annotations/{name}"),
            format!(
                "class {} ({}) is asserted negative ({}) {unit}, where its `is_a` descendant {} ({}) is asserted \
                 positive: under closure='implicit' readers answer it positive (§5.4, §9){more}",
                repr_str(&class_key(labels, first.ancestor)),
                first.ancestor,
                crate::json::py_float(first.value),
                repr_str(&class_key(labels, first.descendant)),
                first.descendant,
            ),
        ));
    }
    Ok(out)
}

/// One class asserted at or below 0 under a descendant asserted above it.
struct Contradiction {
    unit: Option<i64>,
    ancestor: i64,
    value: f64,
    descendant: i64,
}

/// Every [`Contradiction`], by scope unit and then row order.
fn contradictions(
    labels: &LabelSet,
    classes: &[i64],
    values: &[f64],
    units: Option<&[i64]>,
) -> Result<Vec<Contradiction>> {
    let mut by_unit: BTreeMap<Option<i64>, Vec<(i64, f64)>> = BTreeMap::new();
    for (i, (c, v)) in classes.iter().zip(values).enumerate() {
        by_unit.entry(units.map(|u| u[i])).or_default().push((*c, *v));
    }
    let mut out = Vec::new();
    for (unit, rows) in by_unit {
        for &(ancestor, value) in rows.iter().filter(|(_, v)| *v <= 0.0) {
            for &(descendant, _) in rows.iter().filter(|(_, v)| *v > 0.0) {
                if labels.contains_id(descendant) && labels.ancestors(&ClassKey::Id(descendant))?.contains(&ancestor) {
                    out.push(Contradiction { unit, ancestor, value, descendant });
                    break;
                }
            }
        }
    }
    Ok(out)
}

/// A class's key in the label set, or its id when the set does not hold it.
fn class_key(labels: &LabelSet, id: i64) -> String {
    labels.by_id(id).map_or_else(|| id.to_string(), |c| c.key.clone())
}
