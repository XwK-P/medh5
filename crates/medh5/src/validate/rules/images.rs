//! §4 --- images and multiscale pyramids (E2xx).

use ndarray::Array2;

use super::{children, grid_shape, loc, str_attr, strs_attr, Context};
use crate::geometry::grid::read_grid;
use crate::geometry::multiscale::{check_pyramid, GEOMETRY_RTOL};
use crate::h5::attrs;
use crate::h5::data::{self, Kind};
use crate::h5::ops::{self, Node};
use crate::json::{repr_int_tuple, repr_list, repr_str};
use crate::sample::image::VALUE_TYPES;
use crate::validate::Diagnostic;
use crate::Result;

/// Structural: every image against §4.1--§4.2 --- its attributes, its grid
/// and shape, channel names (E101, E105, E201--E205), numeric samples (E001:
/// no E2xx code covers an image's type) --- and float storage an `int16`
/// would hold losslessly (W907).
///
/// A member that is no object is the container rule's finding (E001); a
/// group of nothing else holds no image (E201).  An image that cannot be
/// read is its own finding, and the others are still checked.
pub fn check_images(ctx: &mut Context) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let Some(node) = ops::child_group(&ctx.root, "images") else { return Ok(out) };
    let images = children(&node)?;
    if images.is_empty() {
        let message = if ops::members(&node)?.is_empty() {
            "a sample must contain at least one image"
        } else {
            "a sample must contain at least one image, and no member of `images` is one"
        };
        out.push(ctx.err("E201", "/images", message));
        return Ok(out);
    }
    let grids = ops::child_group(&ctx.root, "grids");
    for (name, image) in images {
        let location = format!("/images/{name}");
        match check_image(ctx, &image, grids.as_ref(), &location) {
            Ok(found) => out.extend(found),
            Err(e) => out.push(ctx.failed(location, &e)),
        }
    }
    Ok(out)
}

/// One image (§4.1--§4.2), single-scale or a pyramid.
fn check_image(ctx: &Context, image: &Node, grids: Option<&hdf5::Group>, location: &str) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let a = loc(image);
    // An image is numbers (§4.2, "any numeric"), every level of a pyramid: a
    // string or compound dataset passed every level, and `read()` refused it.
    let levels: Vec<(String, hdf5::Dataset)> = match image {
        Node::Dataset(d) => vec![(location.to_string(), d.clone())],
        Node::Group(g) => ops::members(g)?
            .into_iter()
            .filter(|n| n.parse::<usize>().is_ok())
            .filter_map(|n| ops::child_dataset(g, &n).map(|d| (format!("{location}/{n}"), d)))
            .collect(),
    };
    for (at, dataset) in &levels {
        let stored = match data::kind(dataset)? {
            Kind::Numeric(_) => continue,
            Kind::Strings => "strings".to_string(),
            Kind::Other(t) => t,
        };
        out.push(ctx.err(
            "E001",
            at.as_str(),
            format!("stores {stored}; an image's samples are numbers (§4.2), so no reader can read it"),
        ));
    }
    for key in ["grid", "modality", "value_type"] {
        if !attrs::has(a, key) {
            out.push(ctx.err("E205", location, format!("missing required attribute {}", repr_str(key))));
        }
    }
    if let Some(vt) = str_attr(a, "value_type")? {
        if !VALUE_TYPES.contains(&vt.as_str()) {
            out.push(ctx.unknown("E203", location, format!("unknown value_type {}", repr_str(&vt))));
        }
    }
    let Some(grid_id) = str_attr(a, "grid")? else { return Ok(out) };
    let Some(grids) = grids.filter(|g| ops::is_group(g, &grid_id)) else {
        out.push(ctx.err("E101", location, format!("names grid {}, which does not exist", repr_str(&grid_id))));
        return Ok(out);
    };
    let gshape = grid_shape(grids, &grid_id)?;
    let dataset = match image {
        Node::Group(g) => match ops::child_dataset(g, "0") {
            Some(d) => d,
            None => {
                out.push(ctx.err("E105", location, "multiscale image has no level-0 dataset `0` (§4.3)"));
                return Ok(out);
            }
        },
        Node::Dataset(d) => d.clone(),
    };
    let shape: Vec<i64> = dataset.shape().iter().map(|v| *v as i64).collect();
    if shape != gshape {
        out.push(ctx.err(
            "E202",
            location,
            format!(
                "shape {} != grid {} shape {}",
                repr_int_tuple(&shape),
                repr_str(&grid_id),
                repr_int_tuple(&gshape)
            ),
        ));
    }
    if attrs::has(a, "channel_names") {
        let gg = grids.group(&grid_id)?;
        let kinds = strs_attr(&gg, "axis_kinds")?;
        match kinds.iter().position(|k| k == "channel") {
            None => out.push(ctx.err("E204", location, "`channel_names` on a grid with no channel axis")),
            Some(axis) => {
                let extent = gshape.get(axis).copied().unwrap_or(0);
                let n = strs_attr(a, "channel_names")?.len() as i64;
                if n != extent {
                    out.push(ctx.err(
                        "E204",
                        location,
                        format!("`channel_names` has {n} entries for a channel axis of extent {extent}"),
                    ));
                }
            }
        }
    }
    if !ctx.errors_only {
        if let Ok(Kind::Numeric(d)) = data::kind(&dataset) {
            if d.is_float() && int16_lossless(&dataset)? {
                out.push(ctx.err(
                    "W907",
                    location,
                    format!(
                        "stored as {} but every value is an integer within int16 range; int16 + rescale is lossless and ~3x smaller",
                        d.name()
                    ),
                ));
            }
        }
    }
    Ok(out)
}

fn int16_lossless(ds: &hdf5::Dataset) -> Result<bool> {
    let size: usize = ds.shape().iter().product();
    if size == 0 || size > 4_000_000 {
        return Ok(false);
    }
    Ok(crate::sample::image::lossless_as_int16(&data::read(ds)?))
}

/// Semantic: every multiscale image against §4.3.
pub fn check_multiscale(ctx: &mut Context) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let (Some(node), Some(grids)) = (ops::child_group(&ctx.root, "images"), ops::child_group(&ctx.root, "grids"))
    else {
        return Ok(out);
    };
    for name in ops::members(&node)? {
        let Some(image) = ops::child_group(&node, &name) else { continue };
        let location = format!("/images/{name}");
        match check_levels(ctx, &image, &grids, &location) {
            Ok(found) => out.extend(found),
            Err(e) => out.push(ctx.failed(location, &e)),
        }
    }
    Ok(out)
}

/// One multiscale image (§4.3): its declaration, a dataset of its level's
/// grid's shape for every level, and every level's geometry derived from
/// level 0's.
fn check_levels(ctx: &Context, image: &hdf5::Group, grids: &hdf5::Group, location: &str) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let missing: Vec<&str> =
        ["levels", "grid_levels", "downsample_factors"].into_iter().filter(|k| !attrs::has(image, k)).collect();
    if !missing.is_empty() {
        out.push(ctx.err(
            "E105",
            location,
            format!(
                "multiscale image needs `levels`, `grid_levels` and `downsample_factors`; missing {}",
                repr_list(&missing)
            ),
        ));
        return Ok(out);
    }
    let level_ids = strs_attr(image, "grid_levels")?;
    if level_ids.is_empty() {
        out.push(ctx.err("E105", location, "`grid_levels` names no level, not even level 0"));
        return Ok(out);
    }
    let n = level_ids.len();
    match attrs::get_i64(image, "levels")? {
        Some(levels) if levels == n as i64 => {}
        Some(levels) => out.push(ctx.err(
            "E105",
            location,
            format!("`levels` is {levels}, but `grid_levels` names {n} level grids"),
        )),
        None => out.push(ctx.err("E105", location, "`levels` is not an integer")),
    }
    let absent: Vec<&String> = level_ids.iter().filter(|g| !ops::is_group(grids, g)).collect();
    if !absent.is_empty() {
        out.push(ctx.err("E101", location, format!("grid_levels reference missing grids {}", repr_list(&absent))));
        return Ok(out);
    }
    // Level 0 against the image's own `grid` is the image rule's.
    let image_grid = str_attr(image, "grid")?;
    for (i, grid_id) in level_ids.iter().enumerate() {
        let Some(dataset) = ops::child_dataset(image, &i.to_string()) else {
            if i > 0 {
                out.push(ctx.err("E105", location, format!("level {i} has no dataset `{i}`")));
            }
            continue;
        };
        if i == 0 && image_grid.as_deref() == Some(grid_id.as_str()) {
            continue;
        }
        let shape: Vec<i64> = dataset.shape().iter().map(|v| *v as i64).collect();
        let gshape = grid_shape(grids, grid_id)?;
        if shape != gshape {
            out.push(ctx.err(
                "E202",
                location,
                format!(
                    "level {i} has shape {} != grid {} shape {}",
                    repr_int_tuple(&shape),
                    repr_str(grid_id),
                    repr_int_tuple(&gshape)
                ),
            ));
        }
    }
    // A level grid a reader refuses is the grid rule's finding, under its own
    // code; the pyramid's geometry is checked between readable grids only.
    let mut levels = Vec::with_capacity(n);
    for id in &level_ids {
        match read_grid(&grids.group(id)?, Some(id)) {
            Ok(grid) => levels.push(grid),
            Err(_) => return Ok(out),
        }
    }
    let spatial = levels[0].n_spatial();
    let factors = match attrs::read(image, "downsample_factors")?.and_then(|v| v.as_matrix()) {
        Some((r, c, v)) if (r, c) == (n, spatial) => Array2::from_shape_vec((r, c), v)?,
        Some((r, c, _)) => {
            out.push(ctx.err(
                "E105",
                location,
                format!(
                    "`downsample_factors` is ({r}, {c}); {n} levels of {spatial} spatial axes need ({n}, {spatial})"
                ),
            ));
            return Ok(out);
        }
        None => {
            out.push(ctx.err("E105", location, "`downsample_factors` must be a 2-D array, one row per level"));
            return Ok(out);
        }
    };
    if factors.iter().any(|f| !(f.is_finite() && *f > 0.0)) {
        out.push(ctx.err("E105", location, "downsample factors must be finite and > 0"));
        return Ok(out);
    }
    let refs: Vec<&crate::geometry::grid::Grid> = levels.iter().collect();
    match check_pyramid(&levels[0], &refs, &factors, GEOMETRY_RTOL) {
        Ok(problems) => out.extend(problems.into_iter().map(|p| ctx.err("E105", location, p))),
        Err(e) => out.push(ctx.failed(location, &e)),
    }
    Ok(out)
}
