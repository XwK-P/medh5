//! §3 --- grids, frames and timepoints (E1xx).

use std::collections::{BTreeMap, BTreeSet};

use ndarray::Array2;

use super::{child, children, float_list, loc, str_attr, strs_attr, Context};
use crate::geometry::affine::{is_orthonormal, ORTHONORMAL_TOL};
use crate::geometry::grid::{read_grid, AXIS_KINDS, KNOWN_UNITS, MAX_SPATIAL, MIN_SPATIAL, TIME_UNITS};
use crate::h5::attrs::{self, AttrValue};
use crate::h5::ops::{self, Node};
use crate::ids::is_valid_id;
use crate::json::{format_g, repr_int_tuple, repr_list, repr_str};
use crate::validate::Diagnostic;
use crate::Result;

/// The attributes every grid carries (§3.2).
const REQUIRED: [&str; 8] =
    ["shape", "axis_names", "axis_kinds", "spacing", "origin", "direction", "coord_system", "units"];

/// Structural: every grid against §3.1--§3.2.
pub fn check_geometry(ctx: &mut Context) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let Some(node) = ops::child_group(&ctx.root, "grids") else {
        return Ok(out);
    };
    let names = ops::members(&node)?;
    if names.is_empty() {
        out.push(ctx.err("E111", "/grids", "a sample must declare at least one grid"));
        return Ok(out);
    }
    for name in names {
        let Some(grid) = child(&node, &name) else { continue };
        let location = format!("/grids/{name}");
        match check_grid(ctx, &grid, &name, &location) {
            Ok(found) => out.extend(found),
            Err(e) => out.push(ctx.failed(location, &e)),
        }
    }
    Ok(out)
}

/// One grid: every rule a reader holds it to, each reported rather than the
/// first alone, and the `units` and `time_units` vocabularies.
fn check_grid(ctx: &Context, grid: &Node, name: &str, location: &str) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let g = loc(grid);
    let missing: Vec<&str> = REQUIRED.into_iter().filter(|k| !attrs::has(g, k)).collect();
    if !missing.is_empty() {
        out.push(ctx.err("E109", location, format!("missing required attribute(s) {}", repr_list(&missing))));
        return Ok(out);
    }
    let Some(shape) = attrs::get_i64s(g, "shape")? else {
        out.push(ctx.err("E109", location, "`shape` is not numeric"));
        return Ok(out);
    };
    let kinds = strs_attr(g, "axis_kinds")?;
    let axis_names = strs_attr(g, "axis_names")?;
    if kinds.len() != shape.len() || axis_names.len() != shape.len() {
        out.push(ctx.err(
            "E109",
            location,
            format!(
                "axis_names/axis_kinds have {}/{} entries for a {}-D shape",
                axis_names.len(),
                kinds.len(),
                shape.len()
            ),
        ));
        return Ok(out);
    }
    if shape.iter().any(|s| *s <= 0) {
        out.push(ctx.err("E109", location, format!("shape {} must be positive", repr_int_tuple(&shape))));
    } else if shape.iter().try_fold(1i64, |n, s| n.checked_mul(*s)).and_then(|n| usize::try_from(n).ok()).is_none() {
        // Every count of the grid's voxels is a product of its extents.
        out.push(ctx.err(
            "E109",
            location,
            format!("shape {} has more elements than a 64-bit count holds", repr_int_tuple(&shape)),
        ));
    }
    let mut unknown: Vec<&String> = kinds.iter().filter(|k| !AXIS_KINDS.contains(&k.as_str())).collect();
    unknown.sort();
    unknown.dedup();
    if !unknown.is_empty() {
        out.push(ctx.unknown("E110", location, format!("unknown axis kinds {}", repr_list(&unknown))));
        return Ok(out);
    }
    let count = |k: &str| kinds.iter().filter(|x| *x == k).count();
    let n_spatial = count("spatial");
    let counted = (MIN_SPATIAL..=MAX_SPATIAL).contains(&n_spatial);
    if !counted {
        out.push(ctx.err("E110", location, format!("{n_spatial} spatial axes; the spec allows 2 or 3")));
    }
    if count("time") > 1 || count("channel") > 1 {
        out.push(ctx.err("E110", location, "at most one `time` and one `channel` axis are allowed"));
    }
    let positions: Vec<usize> = kinds.iter().enumerate().filter(|(_, k)| *k == "spatial").map(|(i, _)| i).collect();
    let expected: Vec<usize> = (kinds.len() - n_spatial..kinds.len()).collect();
    if positions != expected {
        out.push(ctx.err(
            "E103",
            location,
            format!(
                "spatial axes at {} must be contiguous and trailing (expected {})",
                repr_int_tuple(&positions),
                repr_int_tuple(&expected)
            ),
        ));
    }
    // The extents below are counted in spatial axes only when `axis_kinds`
    // lays them out validly: otherwise which is wrong, the kinds or the
    // extents, is not known, and the layout is the finding.
    let rank = (counted && positions == expected).then_some(n_spatial);
    let entries = |key: &str, values: &[f64], out: &mut Vec<Diagnostic>| {
        if let Some(s) = rank.filter(|s| values.len() != *s) {
            out.push(ctx.err("E109", location, format!("`{key}` has {} entries for {s} spatial axes", values.len())));
        }
    };
    match attrs::get_f64s(g, "spacing")? {
        None => out.push(ctx.err("E109", location, "`spacing` is not numeric")),
        Some(spacing) => {
            entries("spacing", &spacing, &mut out);
            if spacing.iter().any(|v| !(v.is_finite() && *v > 0.0)) {
                out.push(ctx.err("E104", location, format!("spacing {} must be finite and > 0", float_list(&spacing))));
            }
        }
    }
    match attrs::get_f64s(g, "origin")? {
        None => out.push(ctx.err("E109", location, "`origin` is not numeric")),
        Some(origin) => {
            entries("origin", &origin, &mut out);
            if origin.iter().any(|v| !v.is_finite()) {
                out.push(ctx.err("E109", location, format!("origin {} must be finite", float_list(&origin))));
            }
        }
    }
    let direction = attrs::read(g, "direction")?.unwrap_or(AttrValue::Unsupported(String::new()));
    match direction.as_matrix() {
        None => out.push(ctx.err(
            "E109",
            location,
            format!("`direction` must be stored 2-D, got shape {}", repr_int_tuple(&direction.shape())),
        )),
        Some((r, c, _)) if r != c || rank.is_some_and(|s| s != r) => {
            let side = rank.unwrap_or(r.max(c));
            out.push(ctx.err("E109", location, format!("`direction` must be {side}x{side}, got ({r}, {c})")));
        }
        Some((r, c, values)) => {
            let m = Array2::from_shape_vec((r, c), values)?;
            if !is_orthonormal(&m, ORTHONORMAL_TOL) {
                // NaN is carried, not dropped as `f64::max` drops it: a NaN
                // direction is not "max residual 0".
                let residual = m
                    .t()
                    .dot(&m)
                    .indexed_iter()
                    .map(|((i, j), v)| (v - if i == j { 1.0 } else { 0.0 }).abs())
                    .fold(0.0f64, |worst, d| if worst.is_nan() || d.is_nan() { f64::NAN } else { worst.max(d) });
                out.push(ctx.err(
                    "E102",
                    location,
                    format!("`direction` is not orthonormal (max residual {})", format_g(residual, 3)),
                ));
            }
        }
    }
    if let Some(units) = str_attr(g, "units")? {
        if !KNOWN_UNITS.contains(&units.as_str()) {
            out.push(ctx.unknown(
                "E109",
                location,
                format!("units {} is not one of {} (§3.2)", repr_str(&units), repr_list(&KNOWN_UNITS)),
            ));
        }
    }
    if count("time") == 1 {
        let extent = shape[kinds.iter().position(|k| k == "time").unwrap_or(0)];
        match attrs::read(g, "time_values")? {
            None => out.push(ctx.err(
                "E109",
                location,
                "grid has a `time` axis but no `time_values`; §3.2 requires one acquisition time per frame",
            )),
            Some(v) => {
                let found = v.as_f64_vec().map(|x| x.len()).unwrap_or(1) as i64;
                if found != extent {
                    out.push(ctx.err(
                        "E109",
                        location,
                        format!("`time_values` has {found} entries for a time axis of extent {extent}"),
                    ));
                }
            }
        }
    }
    if let Some(units) = str_attr(g, "time_units")? {
        if !TIME_UNITS.contains(&units.as_str()) {
            out.push(ctx.unknown(
                "E109",
                location,
                format!("time_units {} is not one of {} (§3.2)", repr_str(&units), repr_list(&TIME_UNITS)),
            ));
        }
    }
    for (key, want, axes) in [("patch_hint", rank, "spatial axes"), ("chunk_hint", Some(shape.len()), "axes")] {
        let Some(value) = attrs::read(g, key)? else { continue };
        match value.as_i64_vec() {
            None => out.push(ctx.err("E109", location, format!("`{key}` is not numeric"))),
            Some(hint) => {
                if let Some(n) = want.filter(|n| hint.len() != *n) {
                    out.push(ctx.err("E109", location, format!("`{key}` has {} entries for {n} {axes}", hint.len())));
                }
                // A chunk shape (§3.2) has no extent below 1.
                if key == "chunk_hint" && hint.iter().any(|c| *c <= 0) {
                    out.push(ctx.err("E109", location, format!("`{key}` {} must be positive", repr_int_tuple(&hint))));
                }
            }
        }
    }
    // Whatever else a reader refuses a grid for, so that the validator never
    // passes a grid `open` refuses.  An invalid id is the container rule's
    // finding (E003).
    if out.is_empty() && is_valid_id(name) {
        if let Node::Group(group) = grid {
            if let Err(e) = read_grid(group, Some(name)) {
                out.push(ctx.failed(location, &e));
            }
        }
    }
    Ok(out)
}

/// Semantic: every grid bound to a declared timepoint (§3.7: E106, E107),
/// frames across timepoints (W910) and a transform relating them (W911).
pub fn check_timepoints(ctx: &mut Context) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let Some(doc) = &ctx.document else { return Ok(out) };
    let declared: BTreeSet<String> = doc.timepoints.ids().into_iter().collect();
    let multi = declared.len() > 1;
    let Some(node) = ops::child_group(&ctx.root, "grids") else { return Ok(out) };
    let mut frames: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for name in ops::members(&node)? {
        let Some(grid) = child(&node, &name) else { continue };
        let g = loc(&grid);
        let location = format!("/grids/{name}");
        let Some(timepoint) = str_attr(g, "timepoint")? else {
            if multi {
                out.push(ctx.err(
                    "E106",
                    location,
                    format!("grid has no `timepoint`, but the sample declares {} timepoints", declared.len()),
                ));
            }
            continue;
        };
        if !declared.contains(&timepoint) {
            let listed: Vec<&String> = declared.iter().collect();
            out.push(ctx.err(
                "E107",
                location,
                format!("`timepoint` {} is not declared (declared: {})", repr_str(&timepoint), repr_list(&listed)),
            ));
            continue;
        }
        if let Some(frame) = str_attr(g, "frame_uid")?.filter(|f| !f.is_empty()) {
            frames.entry(frame).or_default().insert(timepoint);
        }
    }
    for (frame, tps) in &frames {
        if tps.len() > 1 {
            let listed: Vec<&String> = tps.iter().collect();
            out.push(ctx.err(
                "W910",
                "/grids",
                format!(
                    "frame_uid {} is shared by timepoints {}; follow-up imaging is a new frame unless the subject was never repositioned",
                    repr_str(frame),
                    repr_list(&listed)
                ),
            ));
        }
    }
    if multi && !has_relating_transform(&ctx.root, &frames)? {
        out.push(ctx.err(
            "W911",
            "/transforms",
            format!("the sample declares {} timepoints but no transform relates any two of them", declared.len()),
        ));
    }
    Ok(out)
}

fn has_relating_transform(root: &hdf5::Group, frames: &BTreeMap<String, BTreeSet<String>>) -> Result<bool> {
    let Some(node) = ops::child_group(root, "transforms") else { return Ok(false) };
    let empty = BTreeSet::new();
    for (_, t) in children(&node)? {
        let src = str_attr(loc(&t), "from_frame")?.filter(|s| !s.is_empty());
        let dst = str_attr(loc(&t), "to_frame")?.filter(|s| !s.is_empty());
        if let (Some(s), Some(d)) = (src, dst) {
            if frames.get(&s).unwrap_or(&empty) != frames.get(&d).unwrap_or(&empty) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}
