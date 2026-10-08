//! §10 --- transforms and the frame graph (E5xx).

use std::collections::BTreeMap;

use super::{
    child, children, float_list, grid_spatial, has_member, loc, read_f64, str_attr, strs_attr, sub_dataset, Context,
};
use crate::h5::attrs;
use crate::h5::ops::{self, Node};
use crate::json::{repr_int_tuple, repr_list, repr_str};
use crate::transforms::model::{composite_cycle, Siblings, TRANSFORM_KINDS, VECTOR_SPACES};
use crate::validate::Diagnostic;
use crate::Result;

pub fn check_transforms(ctx: &mut Context) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let Some(node) = ops::child_group(&ctx.root, "transforms") else { return Ok(out) };
    let grids = ops::child_group(&ctx.root, "grids");
    let mut declared: BTreeMap<String, (String, String)> = BTreeMap::new();
    for (name, t) in children(&node)? {
        let location = format!("/transforms/{name}");
        let a = loc(&t);
        let Some(kind) = str_attr(a, "kind")? else {
            out.push(ctx.err("E502", location, "transform has no `kind` attribute"));
            continue;
        };
        if !TRANSFORM_KINDS.contains(&kind.as_str()) {
            out.push(ctx.err(
                "E502",
                location,
                format!("unknown transform kind {}; expected one of {}", repr_str(&kind), repr_list(&TRANSFORM_KINDS)),
            ));
            continue;
        }
        let missing: Vec<&str> = ["from_frame", "to_frame"].into_iter().filter(|k| !attrs::has(a, k)).collect();
        if !missing.is_empty() {
            out.push(ctx.err("E502", location, format!("missing {}", repr_list(&missing))));
            continue;
        }
        let source = str_attr(a, "from_frame")?.unwrap_or_default();
        let target = str_attr(a, "to_frame")?.unwrap_or_default();
        declared.insert(name.clone(), (source.clone(), target.clone()));
        if source == target {
            out.push(ctx.err(
                "E502",
                location.clone(),
                format!("maps frame {} to itself; grids sharing a frame need no transform (§3.4)", repr_str(&source)),
            ));
        }
        match kind.as_str() {
            "affine" => out.extend(check_affine(ctx, &name, &t)?),
            "displacement" | "bspline" => {
                out.extend(check_field_transform(ctx, &name, &t, &kind, &source, grids.as_ref())?)
            }
            "composite" => out.extend(check_composite(ctx, &name, &t, &node, &source, &target)?),
            _ => {}
        }
    }
    out.extend(check_inverses(ctx, &node, &declared)?);
    Ok(out)
}

fn check_affine(ctx: &Context, name: &str, group: &Node) -> Result<Vec<Diagnostic>> {
    let location = format!("/transforms/{name}");
    let Some(ds) = sub_dataset(group, "matrix") else {
        return Ok(vec![ctx.err("E502", location, "kind 'affine' requires a `matrix` dataset")]);
    };
    let m = read_f64(&ds)?;
    if m.ndim() != 2 || m.shape()[0] != m.shape()[1] {
        return Ok(vec![ctx.err(
            "E504",
            location,
            format!("`matrix` must be square (S+1, S+1), got {}", repr_int_tuple(m.shape())),
        )]);
    }
    let n = m.shape()[0];
    let last: Vec<f64> = (0..n).map(|c| m[[n - 1, c]]).collect();
    let mut expected = vec![0.0; n];
    if n > 0 {
        expected[n - 1] = 1.0;
    }
    if !crate::geometry::linalg::allclose(&last, &expected, 1e-9, 1e-5) {
        return Ok(vec![ctx.err(
            "E504",
            location,
            format!("last row must be [0 \u{2026} 0 1], got {}", float_list(&last)),
        )]);
    }
    Ok(Vec::new())
}

fn check_field_transform(
    ctx: &Context,
    name: &str,
    group: &Node,
    kind: &str,
    from_frame: &str,
    grids: Option<&hdf5::Group>,
) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let location = format!("/transforms/{name}");
    let (dataset, grid_attr) =
        if kind == "displacement" { ("field", "field_grid") } else { ("control_points", "cp_grid") };
    if !has_member(group, dataset) {
        out.push(ctx.err(
            "E502",
            location.clone(),
            format!("kind {} requires a {} dataset", repr_str(kind), repr_str(dataset)),
        ));
    }
    let Some(grid_id) = str_attr(loc(group), grid_attr)? else {
        out.push(ctx.err("E503", location, format!("kind {} requires {}", repr_str(kind), repr_str(grid_attr))));
        return Ok(out);
    };
    let Some(grids) = grids.filter(|g| ops::exists(g, &grid_id)) else {
        out.push(ctx.err("E101", location, format!("{grid_attr} {} does not exist", repr_str(&grid_id))));
        return Ok(out);
    };
    let g = grids.group(&grid_id)?;
    if let Some(frame) = str_attr(&g, "frame_uid")? {
        if frame != from_frame {
            out.push(ctx.err(
                "E503",
                location.clone(),
                format!(
                    "{grid_attr} {} is in frame {} but the transform starts in {}; the field is sampled in the source frame",
                    repr_str(&grid_id),
                    repr_str(&frame),
                    repr_str(from_frame)
                ),
            ));
        }
    }
    let space = str_attr(loc(group), "vector_space")?.unwrap_or_else(|| "world".into());
    if !VECTOR_SPACES.contains(&space.as_str()) {
        out.push(ctx.err("E502", location.clone(), format!("unknown vector_space {}", repr_str(&space))));
    }
    if let Some(ds) = sub_dataset(group, dataset) {
        let shape = ds.shape();
        let kinds = strs_attr(&g, "axis_kinds")?;
        let n_spatial = kinds.iter().filter(|k| *k == "spatial").count();
        let components = shape.first().copied().unwrap_or(0);
        if components != n_spatial {
            out.push(ctx.err(
                "E503",
                format!("{location}/{dataset}"),
                format!("{components} components on a {n_spatial}-D lattice; they must match"),
            ));
        }
        if kind == "displacement" {
            let spatial = grid_spatial(grids, &grid_id)?;
            let lattice: Vec<i64> = shape.get(1..).unwrap_or(&[]).iter().map(|v| *v as i64).collect();
            if lattice != spatial {
                out.push(ctx.err(
                    "E503",
                    format!("{location}/{dataset}"),
                    format!(
                        "field lattice {} != grid {} spatial shape {}",
                        repr_int_tuple(&lattice),
                        repr_str(&grid_id),
                        repr_int_tuple(&spatial)
                    ),
                ));
            }
        }
    }
    Ok(out)
}

fn check_composite(
    ctx: &Context,
    name: &str,
    group: &Node,
    node: &hdf5::Group,
    source: &str,
    target: &str,
) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let location = format!("/transforms/{name}");
    let a = loc(group);
    if !attrs::has(a, "components") {
        out.push(ctx.err("E501", location, "kind 'composite' requires `components`"));
        return Ok(out);
    }
    let components = strs_attr(a, "components")?;
    let unknown: Vec<&String> = components.iter().filter(|c| !ops::exists(node, c)).collect();
    if !unknown.is_empty() {
        out.push(ctx.err("E501", location, format!("names components {} that do not exist", repr_list(&unknown))));
        return Ok(out);
    }
    let mut siblings = Siblings::new();
    for member in ops::members(node)? {
        if let Some(g) = ops::child_group(node, &member) {
            siblings.insert(member, g);
        }
    }
    let cycle = composite_cycle(name, &siblings)?;
    if !cycle.is_empty() {
        out.push(ctx.err(
            "E501",
            location,
            format!("contains itself through {}; a chain that never ends cannot be evaluated", cycle.join(" -> ")),
        ));
        return Ok(out);
    }
    let mut frames = Vec::new();
    for component in &components {
        let c = child(node, component).ok_or_else(|| crate::Error::Key(repr_str(component)))?;
        let (f, t) = (str_attr(loc(&c), "from_frame")?, str_attr(loc(&c), "to_frame")?);
        let (Some(f), Some(t)) = (f, t) else {
            out.push(ctx.err("E501", location, format!("component {} declares no frames", repr_str(component))));
            return Ok(out);
        };
        frames.push((f, t));
    }
    if let Some(first) = frames.first() {
        if first.0 != source {
            out.push(ctx.err(
                "E501",
                location.clone(),
                format!(
                    "first component starts in {} but the composite declares {}",
                    repr_str(&first.0),
                    repr_str(source)
                ),
            ));
        }
    }
    if let Some(last) = frames.last() {
        if last.1 != target {
            out.push(ctx.err(
                "E501",
                location.clone(),
                format!("last component ends in {} but the composite declares {}", repr_str(&last.1), repr_str(target)),
            ));
        }
    }
    for i in 0..frames.len().saturating_sub(1) {
        if frames[i].1 != frames[i + 1].0 {
            out.push(ctx.err(
                "E501",
                location.clone(),
                format!(
                    "{} ends in {} but {} starts in {}",
                    repr_str(&components[i]),
                    repr_str(&frames[i].1),
                    repr_str(&components[i + 1]),
                    repr_str(&frames[i + 1].0)
                ),
            ));
        }
    }
    if let Some(declared_units) = str_attr(a, "units")? {
        let mut mixed = Vec::new();
        for component in &components {
            let c = child(node, component).ok_or_else(|| crate::Error::Key(repr_str(component)))?;
            if let Some(u) = str_attr(loc(&c), "units")? {
                if u != declared_units {
                    mixed.push(format!("{} in {}", repr_str(component), repr_str(&u)));
                }
            }
        }
        if !mixed.is_empty() {
            out.push(ctx.err(
                "E501",
                location,
                format!(
                    "declares units {} but {} --- a chain whose legs are in different units does not compose",
                    repr_str(&declared_units),
                    mixed.join(", ")
                ),
            ));
        }
    }
    Ok(out)
}

fn check_inverses(
    ctx: &Context,
    node: &hdf5::Group,
    declared: &BTreeMap<String, (String, String)>,
) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    for (name, (source, target)) in declared {
        let t = child(node, name).ok_or_else(|| crate::Error::Key(repr_str(name)))?;
        let Some(other) = str_attr(loc(&t), "inverse_id")? else { continue };
        let location = format!("/transforms/{name}");
        let Some((os, ot)) = declared.get(&other) else {
            out.push(ctx.err(
                "E505",
                location,
                format!("`inverse_id` names {}, which does not exist", repr_str(&other)),
            ));
            continue;
        };
        if (os, ot) != (target, source) {
            out.push(ctx.err(
                "E505",
                location,
                format!(
                    "`inverse_id` names {}, which maps {} -> {}; an inverse must map {} -> {}",
                    repr_str(&other),
                    repr_str(os),
                    repr_str(ot),
                    repr_str(target),
                    repr_str(source)
                ),
            ));
            continue;
        }
        let o = child(node, &other).ok_or_else(|| crate::Error::Key(repr_str(&other)))?;
        if let Some(back) = str_attr(loc(&o), "inverse_id")? {
            if &back != name {
                out.push(ctx.err(
                    "E505",
                    location,
                    format!(
                        "{} names {} as its inverse, not {}; the relation must be mutual",
                        repr_str(&other),
                        repr_str(&back),
                        repr_str(name)
                    ),
                ));
            }
        }
    }
    Ok(out)
}
