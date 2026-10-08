//! §3 --- grids, frames and timepoints (E1xx).

use std::collections::{BTreeMap, BTreeSet};

use ndarray::Array2;

use super::{child, children, float_list, loc, str_attr, strs_attr, Context};
use crate::geometry::affine::{is_orthonormal, ORTHONORMAL_TOL};
use crate::geometry::grid::AXIS_KINDS;
use crate::h5::attrs::{self, AttrValue};
use crate::h5::ops;
use crate::json::{format_g, repr_int_tuple, repr_list, repr_str};
use crate::validate::Diagnostic;
use crate::Result;

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
        let g = loc(&grid);
        let location = format!("/grids/{name}");
        let missing: Vec<&str> =
            ["shape", "axis_names", "axis_kinds", "spacing", "origin", "direction", "coord_system", "units"]
                .into_iter()
                .filter(|k| !attrs::has(g, k))
                .collect();
        if !missing.is_empty() {
            out.push(ctx.err("E109", location, format!("missing required attribute(s) {}", repr_list(&missing))));
            continue;
        }
        let shape = attrs::get_i64s(g, "shape")?.unwrap_or_default();
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
            continue;
        }
        let mut unknown: Vec<&String> = kinds.iter().filter(|k| !AXIS_KINDS.contains(&k.as_str())).collect();
        unknown.sort();
        unknown.dedup();
        if !unknown.is_empty() {
            out.push(ctx.err("E110", location, format!("unknown axis kinds {}", repr_list(&unknown))));
            continue;
        }
        let count = |k: &str| kinds.iter().filter(|x| *x == k).count();
        let n_spatial = count("spatial");
        if !(2..=3).contains(&n_spatial) {
            out.push(ctx.err("E110", location.clone(), format!("{n_spatial} spatial axes; the spec allows 2 or 3")));
        }
        if count("time") > 1 || count("channel") > 1 {
            out.push(ctx.err("E110", location.clone(), "at most one `time` and one `channel` axis are allowed"));
        }
        let positions: Vec<usize> = kinds.iter().enumerate().filter(|(_, k)| *k == "spatial").map(|(i, _)| i).collect();
        let expected: Vec<usize> = (kinds.len() - n_spatial..kinds.len()).collect();
        if positions != expected {
            out.push(ctx.err(
                "E103",
                location.clone(),
                format!(
                    "spatial axes at {} must be contiguous and trailing (expected {})",
                    repr_int_tuple(&positions),
                    repr_int_tuple(&expected)
                ),
            ));
        }
        let spacing = attrs::get_f64s(g, "spacing")?.unwrap_or_default();
        if spacing.iter().any(|v| *v <= 0.0) {
            out.push(ctx.err(
                "E104",
                location.clone(),
                format!("spacing {} must be strictly positive", float_list(&spacing)),
            ));
        }
        let direction = attrs::read(g, "direction")?.unwrap_or(AttrValue::Unsupported(String::new()));
        match direction.as_matrix() {
            None => out.push(ctx.err(
                "E109",
                location.clone(),
                format!("`direction` must be stored 2-D, got shape {}", repr_int_tuple(&direction.shape())),
            )),
            Some((r, c, values)) => {
                let m = Array2::from_shape_vec((r, c), values)?;
                if !is_orthonormal(&m, ORTHONORMAL_TOL) {
                    let product = m.t().dot(&m);
                    let mut residual = 0.0f64;
                    for i in 0..product.nrows() {
                        for j in 0..product.ncols() {
                            let target = if i == j { 1.0 } else { 0.0 };
                            residual = residual.max((product[[i, j]] - target).abs());
                        }
                    }
                    out.push(ctx.err(
                        "E102",
                        location.clone(),
                        format!("`direction` is not orthonormal (max residual {})", format_g(residual, 3)),
                    ));
                }
            }
        }
        if count("time") == 1 {
            let extent = shape[kinds.iter().position(|k| k == "time").unwrap_or(0)];
            match attrs::read(g, "time_values")? {
                None => out.push(ctx.err(
                    "E109",
                    location.clone(),
                    "grid has a `time` axis but no `time_values`; §3.2 requires one acquisition time per frame",
                )),
                Some(v) => {
                    let found = v.as_f64_vec().map(|x| x.len()).unwrap_or(1) as i64;
                    if found != extent {
                        out.push(ctx.err(
                            "E109",
                            location.clone(),
                            format!("`time_values` has {found} entries for a time axis of extent {extent}"),
                        ));
                    }
                }
            }
        }
    }
    Ok(out)
}

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
