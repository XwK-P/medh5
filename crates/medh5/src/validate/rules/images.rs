//! §4 --- images and multiscale pyramids (E2xx).

use ndarray::Array2;

use super::{child, grid_shape, loc, str_attr, strs_attr, Context};
use crate::geometry::grid::read_grid;
use crate::geometry::multiscale::{check_pyramid, GEOMETRY_RTOL};
use crate::h5::attrs;
use crate::h5::data::{self, Kind};
use crate::h5::ops::{self, Node};
use crate::json::{repr_int_tuple, repr_list, repr_str};
use crate::sample::image::VALUE_TYPES;
use crate::validate::Diagnostic;
use crate::Result;

pub fn check_images(ctx: &mut Context) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let Some(node) = ops::child_group(&ctx.root, "images") else { return Ok(out) };
    let names = ops::members(&node)?;
    if names.is_empty() {
        out.push(ctx.err("E201", "/images", "a sample must contain at least one image"));
        return Ok(out);
    }
    let grids = ops::child_group(&ctx.root, "grids");
    for name in names {
        let Some(image) = child(&node, &name) else { continue };
        let a = loc(&image);
        let location = format!("/images/{name}");
        for key in ["grid", "modality", "value_type"] {
            if !attrs::has(a, key) {
                out.push(ctx.err("E205", location.clone(), format!("missing required attribute {}", repr_str(key))));
            }
        }
        if let Some(vt) = str_attr(a, "value_type")? {
            if !VALUE_TYPES.contains(&vt.as_str()) {
                out.push(ctx.unknown("E203", location.clone(), format!("unknown value_type {}", repr_str(&vt))));
            }
        }
        let Some(grid_id) = str_attr(a, "grid")? else { continue };
        let Some(grids) = grids.as_ref().filter(|g| ops::exists(g, &grid_id)) else {
            out.push(ctx.err("E101", location, format!("names grid {}, which does not exist", repr_str(&grid_id))));
            continue;
        };
        let gshape = grid_shape(grids, &grid_id)?;
        let dataset = match &image {
            Node::Group(g) => g.dataset("0")?,
            Node::Dataset(d) => d.clone(),
        };
        let shape: Vec<i64> = dataset.shape().iter().map(|v| *v as i64).collect();
        if shape != gshape {
            out.push(ctx.err(
                "E202",
                location.clone(),
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
                None => out.push(ctx.err("E204", location.clone(), "`channel_names` on a grid with no channel axis")),
                Some(axis) => {
                    let extent = gshape.get(axis).copied().unwrap_or(0);
                    let n = strs_attr(a, "channel_names")?.len() as i64;
                    if n != extent {
                        out.push(ctx.err(
                            "E204",
                            location.clone(),
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

pub fn check_multiscale(ctx: &mut Context) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let (Some(node), Some(grids)) = (ops::child_group(&ctx.root, "images"), ops::child_group(&ctx.root, "grids"))
    else {
        return Ok(out);
    };
    for name in ops::members(&node)? {
        let Some(image) = ops::child_group(&node, &name) else { continue };
        let location = format!("/images/{name}");
        if !attrs::has(&image, "grid_levels") || !attrs::has(&image, "downsample_factors") {
            out.push(ctx.err("E105", location, "multiscale image needs `grid_levels` and `downsample_factors`"));
            continue;
        }
        let level_ids = strs_attr(&image, "grid_levels")?;
        if level_ids.iter().any(|g| !ops::exists(&grids, g)) {
            out.push(ctx.err(
                "E101",
                location,
                format!("grid_levels reference missing grids {}", repr_list(&level_ids)),
            ));
            continue;
        }
        let levels: Vec<crate::geometry::grid::Grid> =
            level_ids.iter().map(|g| read_grid(&grids.group(g)?, Some(g))).collect::<Result<_>>()?;
        let refs: Vec<&crate::geometry::grid::Grid> = levels.iter().collect();
        let factors = match attrs::read(&image, "downsample_factors")?.and_then(|v| v.as_matrix()) {
            Some((r, c, v)) => Array2::from_shape_vec((r, c), v)?,
            None => Array2::zeros((0, 0)),
        };
        for problem in check_pyramid(&levels[0], &refs, &factors, GEOMETRY_RTOL)? {
            out.push(ctx.err("E105", location.clone(), problem));
        }
    }
    Ok(out)
}
