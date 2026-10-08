//! §6--§7 --- annotation headers, references and the voxel encodings (E4xx).

use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::geometric::{bad_boxes, check_classification, check_geometric};
use super::{
    dtype_name, grid_spatial, has_member, int_set, loc, read_f64, read_i64, str_attr, strs_attr, sub_dataset, Context,
};
use crate::annotations::header::{ANNOTATION_KINDS, RESERVED_KINDS, TASKS, VOXEL_KINDS};
use crate::annotations::payload::{contains_value, SLAB_BYTES};
use crate::annotations::select::greedy_colour;
use crate::array::{Index, Slice};
use crate::h5::attrs;
use crate::h5::data;
use crate::h5::ops::{self, Node};
use crate::json::{py_float, repr_int_list, repr_int_tuple, repr_list, repr_str};
use crate::labels::{BACKGROUND_ID, CLOSURES, IGNORE_ID};
use crate::validate::Diagnostic;
use crate::Result;

/// Extra layers beyond the greedy optimum before W908 fires.
pub const W908_TOLERANCE: usize = 2;

/// Datasets each kind requires.
pub fn required_datasets(kind: &str) -> &'static [&'static str] {
    match kind {
        "labelmap" | "probmap" | "mask" => &["data"],
        "layers" => &["data", "layer_class_ids"],
        "bitmask" => &["data", "bit_class_ids"],
        "instances" => &["boxes", "class_ids", "instance_ids"],
        "boxes" => &["boxes", "class_ids"],
        "obb" => &["centers", "sizes", "rotations", "class_ids"],
        "keypoints" => &["points", "keypoint_class_ids", "class_ids"],
        "points" => &["points"],
        "contours" => &["vertices", "contour_offsets", "contour_class_ids"],
        "mesh" => &["vertices", "faces"],
        "classification" => &["class_ids", "values"],
        _ => &[],
    }
}

/// The dtypes a voxel kind's `data` may have.
pub fn allowed_dtypes(kind: &str) -> Option<&'static [&'static str]> {
    Some(match kind {
        "labelmap" | "layers" => &["uint8", "uint16"],
        "bitmask" => &["uint64"],
        "probmap" => &["float16", "float32"],
        "mask" => &["bool", "uint8"],
        _ => return None,
    })
}

/// Per-kind dtype rules for the §8/§9 datasets.
pub fn geometric_dtypes(kind: &str) -> &'static [(&'static str, &'static [&'static str])] {
    match kind {
        "boxes" => &[("boxes", &["float32", "float64"]), ("class_ids", &["uint16"])],
        "obb" => &[
            ("centers", &["float32", "float64"]),
            ("sizes", &["float32", "float64"]),
            ("rotations", &["float32", "float64"]),
        ],
        "keypoints" => &[("points", &["float32", "float64"]), ("visibility", &["uint8"])],
        "points" => &[("points", &["float32", "float64"])],
        "contours" => &[("vertices", &["float32", "float64"]), ("contour_offsets", &["int64"])],
        "mesh" => &[("vertices", &["float32", "float64"]), ("faces", &["int32", "int64"])],
        "classification" => &[("class_ids", &["uint16"]), ("values", &["float32", "float64"])],
        _ => &[],
    }
}

pub fn check_annotations(ctx: &mut Context) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let grids = ops::child_group(&ctx.root, "grids");
    let declared_tps: BTreeSet<String> =
        ctx.document.as_ref().map(|d| d.timepoints.ids().into_iter().collect()).unwrap_or_default();
    let label_set = ctx.document.as_ref().and_then(|d| d.label_set.clone());
    for (name, group) in ctx.children("annotations")? {
        let location = format!("/annotations/{name}");
        let a = loc(&group);
        let Some(kind) = str_attr(a, "kind")? else {
            out.push(ctx.err("E412", location, "missing required attribute `kind`"));
            continue;
        };
        if RESERVED_KINDS.contains(&kind.as_str()) {
            out.push(ctx.err(
                "E401",
                location,
                format!("kind {} is reserved by spec §16 and must not appear in a 1.0 file", repr_str(&kind)),
            ));
            continue;
        }
        if !ANNOTATION_KINDS.contains(&kind.as_str()) {
            out.push(ctx.err("E401", location, format!("unknown annotation kind {}", repr_str(&kind))));
            continue;
        }
        if let Some(task) = str_attr(a, "task")? {
            if !TASKS.contains(&task.as_str()) {
                out.push(ctx.err("E412", location.clone(), format!("unknown task {}", repr_str(&task))));
            }
        }
        if let Some(closure) = str_attr(a, "closure")? {
            if !CLOSURES.contains(&closure.as_str()) {
                out.push(ctx.err("E412", location.clone(), format!("unknown closure {}", repr_str(&closure))));
            }
        }
        if !attrs::has(a, "annotated_class_ids") && kind != "mask" {
            out.push(ctx.err(
                "E412",
                location.clone(),
                "missing `annotated_class_ids`; the coverage contract is required",
            ));
        }
        let class_ids = int_set(a, "class_ids")?;
        let annotated = int_set(a, "annotated_class_ids")?;
        if !annotated.is_subset(&class_ids) && kind != "mask" {
            let extra: Vec<i64> = annotated.difference(&class_ids).copied().collect();
            out.push(ctx.err(
                "E403",
                location.clone(),
                format!("annotated_class_ids {} are not in class_ids", repr_int_list(&extra)),
            ));
        }
        if let Some(ls) = label_set.as_ref().filter(|l| l.form == "inline") {
            let missing = ls.missing(class_ids.iter().copied());
            if !missing.is_empty() {
                out.push(ctx.err(
                    "E402",
                    location.clone(),
                    format!("class ids {} are not in label set {}", repr_int_list(&missing), repr_str(&ls.id)),
                ));
            }
        }
        let reserved: Vec<i64> = class_ids.iter().copied().filter(|c| *c == BACKGROUND_ID || *c == IGNORE_ID).collect();
        if !reserved.is_empty() {
            out.push(ctx.err(
                "E303",
                location.clone(),
                format!("class_ids uses reserved id(s) {}", repr_int_list(&reserved)),
            ));
        }
        if attrs::has(a, "timepoints") {
            for tp in strs_attr(a, "timepoints")? {
                if !declared_tps.contains(&tp) {
                    out.push(ctx.err("E409", location.clone(), format!("undeclared timepoint {}", repr_str(&tp))));
                }
            }
        }
        let grid_id = str_attr(a, "grid")?;
        if VOXEL_KINDS.contains(&kind.as_str()) {
            match &grid_id {
                None => {
                    out.push(ctx.err("E412", location.clone(), format!("kind {} requires a `grid`", repr_str(&kind))))
                }
                Some(gid) if !grids.as_ref().is_some_and(|g| ops::exists(g, gid)) => out.push(ctx.err(
                    "E101",
                    location.clone(),
                    format!("names grid {}, which does not exist", repr_str(gid)),
                )),
                _ => {}
            }
        }
        for required in required_datasets(&kind) {
            if !has_member(&group, required) {
                out.push(ctx.err(
                    "E410",
                    location.clone(),
                    format!("kind {} requires dataset {}", repr_str(&kind), repr_str(required)),
                ));
            }
        }
        if let (Some(ds), Some(allowed)) = (sub_dataset(&group, "data"), allowed_dtypes(&kind)) {
            let found = dtype_name(&ds);
            if !allowed.contains(&found.as_str()) {
                out.push(ctx.err(
                    "E411",
                    format!("{location}/data"),
                    format!(
                        "dtype {found} is not permitted for kind {} (expected one of {})",
                        repr_str(&kind),
                        repr_list(allowed)
                    ),
                ));
            }
        }
        out.extend(check_voxel_shape(ctx, &name, &group, &kind, grid_id.as_deref(), grids.as_ref())?);
        out.extend(check_encoding_invariants(ctx, &name, &group, &kind)?);
        out.extend(check_dataset_dtypes(ctx, &name, &group, &kind));
        out.extend(check_geometric(ctx, &name, &group, &kind, grid_id.as_deref(), grids.as_ref())?);
        out.extend(check_classification(ctx, &name, &group, &kind)?);
        if !ctx.errors_only
            && kind != "mask"
            && annotated.is_subset(&class_ids)
            && annotated.len() < class_ids.len()
            && !has_ignore(&group, &kind)?
        {
            out.push(ctx.err(
                "W904",
                location,
                format!(
                    "{} class(es) are encodable but not annotated, and there is no ignore region; `0` cannot be read as a verified negative for them",
                    class_ids.difference(&annotated).count()
                ),
            ));
        }
    }
    Ok(out)
}

fn has_ignore(group: &Node, kind: &str) -> Result<bool> {
    if attrs::has(loc(group), "ignore_mask") {
        return Ok(true);
    }
    in_band_ignore(group, kind)
}

fn in_band_ignore(group: &Node, kind: &str) -> Result<bool> {
    if kind != "labelmap" && kind != "layers" {
        return Ok(false);
    }
    let Some(ds) = sub_dataset(group, "data") else { return Ok(false) };
    let ignore_id = attrs::get_i64(loc(group), "ignore_id")?.unwrap_or(IGNORE_ID);
    contains_value(&ds, ignore_id)
}

fn annotation_id(reference: &str) -> &str {
    reference.strip_prefix("annotations/").unwrap_or(reference)
}

/// Every cross-object reference §4.4 and §6.2 name resolves (E413).
pub fn check_references(ctx: &mut Context) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let groups: BTreeMap<String, Node> = ctx.children("annotations")?.into_iter().collect();
    for (name, group) in &groups {
        let location = format!("/annotations/{name}");
        let a = loc(group);
        if let Some(target) = str_attr(a, "ignore_mask")? {
            out.extend(check_mask_reference(ctx, &location, "ignore_mask", &target, a, &groups)?);
        }
        if attrs::has(a, "derived_from") {
            for reference in strs_attr(a, "derived_from")? {
                if !groups.contains_key(annotation_id(&reference)) {
                    out.push(ctx.err(
                        "E413",
                        location.clone(),
                        format!("`derived_from` names annotation {}, which does not exist", repr_str(&reference)),
                    ));
                }
            }
        }
    }
    for (name, node) in ctx.children("images")? {
        if let Some(target) = str_attr(loc(&node), "valid_mask")? {
            out.extend(check_mask_reference(
                ctx,
                &format!("/images/{name}"),
                "valid_mask",
                &target,
                loc(&node),
                &groups,
            )?);
        }
    }
    Ok(out)
}

fn check_mask_reference(
    ctx: &Context,
    location: &str,
    attr: &str,
    target: &str,
    owner: &hdf5::Location,
    groups: &BTreeMap<String, Node>,
) -> Result<Vec<Diagnostic>> {
    let Some(other) = groups.get(target) else {
        return Ok(vec![ctx.err(
            "E413",
            location,
            format!("`{attr}` names annotation {}, which does not exist", repr_str(target)),
        )]);
    };
    let kind = str_attr(loc(other), "kind")?;
    if kind.as_deref() != Some("mask") {
        return Ok(vec![ctx.err(
            "E413",
            location,
            format!(
                "`{attr}` names {}, whose kind is {}; §4.4 and §7.7 require a `mask` annotation",
                repr_str(target),
                kind.as_deref().map(repr_str).unwrap_or_else(|| "None".into())
            ),
        )]);
    }
    let mine = str_attr(owner, "grid")?;
    let theirs = str_attr(loc(other), "grid")?;
    if let (Some(m), Some(t)) = (mine, theirs) {
        if m != t {
            return Ok(vec![ctx.err(
                "E413",
                location,
                format!(
                    "`{attr}` names {} on grid {}, but this object is on grid {}; a mask delimits the voxels of the grid it shares",
                    repr_str(target),
                    repr_str(&t),
                    repr_str(&m)
                ),
            )]);
        }
    }
    Ok(Vec::new())
}

fn check_voxel_shape(
    ctx: &Context,
    name: &str,
    group: &Node,
    kind: &str,
    grid_id: Option<&str>,
    grids: Option<&hdf5::Group>,
) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let (Some(gid), Some(grids)) = (grid_id, grids) else { return Ok(out) };
    if !VOXEL_KINDS.contains(&kind) || !ops::exists(grids, gid) {
        return Ok(out);
    }
    let Some(ds) = sub_dataset(group, "data") else { return Ok(out) };
    let spatial = grid_spatial(grids, gid)?;
    let shape: Vec<i64> = ds.shape().iter().map(|v| *v as i64).collect();
    let stacked = matches!(kind, "layers" | "bitmask" | "probmap");
    let tail: Vec<i64> = if stacked { shape.get(1..).unwrap_or(&[]).to_vec() } else { shape.clone() };
    if tail != spatial {
        out.push(ctx.err(
            "E405",
            format!("/annotations/{name}/data"),
            format!(
                "spatial shape {} != grid {} spatial shape {}",
                repr_int_tuple(&tail),
                repr_str(gid),
                repr_int_tuple(&spatial)
            ),
        ));
    }
    if stacked {
        if let Some(chunks) = data::chunks(&ds) {
            if chunks.first().copied() != Some(1) {
                out.push(ctx.err(
                    "W902",
                    format!("/annotations/{name}/data"),
                    format!(
                        "chunk shape {} spans the stacked axis; the spec requires (1, *spatial_chunk) so one plane reads without the others",
                        repr_int_tuple(&chunks)
                    ),
                ));
            }
        }
    }
    Ok(out)
}

fn check_encoding_invariants(ctx: &Context, name: &str, group: &Node, kind: &str) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let location = format!("/annotations/{name}");
    let declared = int_set(loc(group), "class_ids")?;
    if kind == "layers" {
        if let Some(table_ds) = sub_dataset(group, "layer_class_ids") {
            let table = read_i64(&table_ds)?;
            let rows: Vec<Vec<i64>> = if table.ndim() >= 2 {
                table.outer_iter().map(|r| r.iter().copied().collect()).collect()
            } else {
                vec![table.iter().copied().collect()]
            };
            let mut seen: BTreeMap<i64, usize> = BTreeMap::new();
            for (layer, row) in rows.iter().enumerate() {
                for class_id in row {
                    if *class_id == 0 {
                        continue;
                    }
                    if let Some(first) = seen.get(class_id) {
                        out.push(ctx.err(
                            "E404",
                            location.clone(),
                            format!("class {class_id} appears in layers {first} and {layer}; every class must be in exactly one layer"),
                        ));
                    }
                    seen.insert(*class_id, layer);
                }
            }
            let missing: Vec<i64> = declared.iter().copied().filter(|c| !seen.contains_key(c)).collect();
            if !missing.is_empty() {
                out.push(ctx.err(
                    "E404",
                    location.clone(),
                    format!("class_ids {} are not assigned to any layer", repr_int_list(&missing)),
                ));
            }
            out.extend(check_layer_optimality(ctx, name, group, rows.len(), declared.len())?);
        }
    }
    if kind == "labelmap" {
        if let Some(ds) = sub_dataset(group, "data") {
            if dtype_name(&ds) == "uint16"
                && !declared.is_empty()
                && declared.iter().max().copied().unwrap_or(0) <= 254
                && !in_band_ignore(group, kind)?
            {
                out.push(ctx.err(
                    "E411",
                    format!("{location}/data"),
                    "stored as uint16 although max(class_ids) is at most 254 and no ignore voxel is present; §7.1 requires uint8",
                ));
            }
        }
    }
    if kind == "probmap" {
        if let Some(value) = attrs::get_f64(loc(group), "threshold")? {
            if !(0.0..=1.0).contains(&value) || value.is_nan() {
                out.push(ctx.err(
                    "E404",
                    location.clone(),
                    format!(
                        "`threshold` {} is outside [0, 1]; it is the probability at or above which a voxel contains the class (§7.5)",
                        py_float(value)
                    ),
                ));
            }
        }
    }
    if kind == "bitmask" {
        if let (Some(bits), Some(ds)) = (sub_dataset(group, "bit_class_ids"), sub_dataset(group, "data")) {
            let n_classes = bits.shape().iter().product::<usize>();
            let expected = n_classes.div_ceil(64).max(1);
            let planes = ds.shape().first().copied().unwrap_or(0);
            if planes != expected {
                out.push(ctx.err(
                    "E404",
                    location.clone(),
                    format!("{planes} bitplanes for {n_classes} classes; expected {expected}"),
                ));
            }
        }
    }
    if kind == "instances" {
        out.extend(check_instances(ctx, name, group)?);
    }
    Ok(out)
}

fn check_layer_optimality(
    ctx: &Context,
    name: &str,
    group: &Node,
    n_layers: usize,
    n_classes: usize,
) -> Result<Vec<Diagnostic>> {
    if ctx.errors_only || !(ctx.level == "semantic" || ctx.level == "strict") || n_classes == 0 {
        return Ok(Vec::new());
    }
    let (Some(table_ds), Some(ds)) = (sub_dataset(group, "layer_class_ids"), sub_dataset(group, "data")) else {
        return Ok(Vec::new());
    };
    let mut classes: Vec<i64> = read_i64(&table_ds)?.iter().copied().filter(|v| *v != 0).collect();
    classes.sort_unstable();
    classes.dedup();
    let ignore_id = attrs::get_i64(loc(group), "ignore_id")?.unwrap_or(IGNORE_ID);
    let colouring = greedy_colour(&classes, &overlap_edges(&ds, ignore_id)?);
    let optimal = colouring.values().max().map(|m| m + 1).unwrap_or(0);
    if n_layers > optimal + W908_TOLERANCE {
        let cut = 100.0 * (1.0 - optimal as f64 / n_layers as f64);
        return Ok(vec![ctx.err(
            "W908",
            format!("/annotations/{name}"),
            format!(
                "{n_layers} layers where a greedy colouring of the overlap graph needs {optimal}; transcoding would cut the label volume by {}%",
                format_g_fixed0(cut)
            ),
        )]);
    }
    Ok(Vec::new())
}

/// `format(value, ".0f")` with Python's round-half-even.
fn format_g_fixed0(value: f64) -> String {
    format!("{:.0}", value.round_ties_even())
}

/// The class overlap graph of a `layers` dataset, read in bounded slabs.
///
/// Within one layer classes never share a voxel, so an edge is a pair of
/// values that co-occur at a voxel in two different layers.  Pairing the
/// layers slab by slab answers that directly, in memory bounded by the slab
/// rather than by classes times voxels.
fn overlap_edges(ds: &hdf5::Dataset, ignore_id: i64) -> Result<BTreeSet<(i64, i64)>> {
    overlap_edges_within(ds, ignore_id, SLAB_BYTES)
}

/// [`overlap_edges`] reading at most about `budget` bytes at a time.
fn overlap_edges_within(ds: &hdf5::Dataset, ignore_id: i64, budget: usize) -> Result<BTreeSet<(i64, i64)>> {
    let shape = ds.shape();
    let mut edges = BTreeSet::new();
    if shape.len() < 2 || shape[0] < 2 || shape.iter().product::<usize>() == 0 {
        return Ok(edges);
    }
    let n_layers = shape[0];
    let rows = shape[1];
    let itemsize = data::dtype(ds)?.itemsize();
    let per_row = n_layers * shape[2..].iter().product::<usize>() * itemsize;
    let step = (budget / per_row.max(1)).clamp(1, rows);
    let mut start = 0;
    while start < rows {
        let block = data::read_region(
            ds,
            &[Index::Slice(Slice::full()), Index::Slice(Slice::new(start as i64, (start + step) as i64))],
        )?
        .cast::<i64>();
        let n = block.len() / n_layers;
        let flat: Vec<i64> = block.iter().copied().collect();
        for i in 0..n_layers {
            for j in i + 1..n_layers {
                for k in 0..n {
                    let (a, b) = (flat[i * n + k], flat[j * n + k]);
                    if a != 0 && a != ignore_id && b != 0 && b != ignore_id {
                        edges.insert(if a < b { (a, b) } else { (b, a) });
                    }
                }
            }
        }
        start += step;
    }
    Ok(edges)
}

fn check_dataset_dtypes(ctx: &Context, name: &str, group: &Node, kind: &str) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    for (dataset, allowed) in geometric_dtypes(kind) {
        let Some(ds) = sub_dataset(group, dataset) else { continue };
        let found = dtype_name(&ds);
        if !allowed.contains(&found.as_str()) {
            out.push(ctx.err(
                "E411",
                format!("/annotations/{name}/{dataset}"),
                format!(
                    "dtype {found} is not permitted for {}.{dataset} (expected one of {})",
                    repr_str(kind),
                    repr_list(allowed)
                ),
            ));
        }
    }
    out
}

fn check_instances(ctx: &Context, name: &str, group: &Node) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let location = format!("/annotations/{name}");
    if let Some(ds) = sub_dataset(group, "boxes") {
        let boxes = read_f64(&ds)?;
        if boxes.ndim() == 3 {
            let bad = bad_boxes(&boxes);
            if bad > 0 {
                out.push(ctx.err("E406", location.clone(), format!("{bad} box(es) have lo > hi")));
            }
        }
    }
    if let Some(ds) = sub_dataset(group, "mask_offsets") {
        let offsets: Vec<i64> = read_i64(&ds)?.iter().copied().collect();
        if offsets.windows(2).any(|w| w[1] < w[0]) {
            out.push(ctx.err("E408", location.clone(), "`mask_offsets` are not monotonic"));
        }
    }
    let ids: Option<Vec<u64>> = match sub_dataset(group, "instance_ids") {
        Some(ds) => Some(data::read(&ds)?.cast::<u64>().iter().copied().collect()),
        None => None,
    };
    if let Some(ids) = &ids {
        let mut counts: BTreeMap<u64, usize> = BTreeMap::new();
        for i in ids {
            *counts.entry(*i).or_default() += 1;
        }
        let shared: Vec<u64> = counts.into_iter().filter(|(_, n)| *n > 1).map(|(i, _)| i).collect();
        if !shared.is_empty() {
            out.push(ctx.err(
                "E404",
                location.clone(),
                format!(
                    "instance id(s) {} name more than one object; two distinct objects MUST NOT share one (§7.4)",
                    repr_int_list(&shared)
                ),
            ));
        }
    }
    if let (Some(ids), Some(cds)) = (&ids, sub_dataset(group, "class_ids")) {
        let classes: Vec<i64> = read_i64(&cds)?.iter().copied().collect();
        if ids.len() != classes.len() {
            return Err(crate::Error::Value(format!(
                "zip() argument 2 is shorter than argument 1 ({} ids, {} classes)",
                ids.len(),
                classes.len()
            )));
        }
        let mut table: HashMap<u64, i64> = HashMap::new();
        let mut conflicting = BTreeSet::new();
        for (i, c) in ids.iter().zip(&classes) {
            if *table.entry(*i).or_insert(*c) != *c {
                conflicting.insert(*i);
            }
        }
        if !conflicting.is_empty() {
            let listed: Vec<u64> = conflicting.into_iter().collect();
            out.push(ctx.err(
                "W909",
                location,
                format!(
                    "instance id(s) {} carry more than one class id; this is almost always a tracking error rather than a reclassification",
                    repr_int_list(&listed)
                ),
            ));
        }
    }
    Ok(out)
}

/// `instance_id` is sample-scoped, so the check has to be too (§7.4).
pub fn check_instance_identity(ctx: &mut Context) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let mut table: BTreeMap<u64, BTreeMap<i64, Vec<String>>> = BTreeMap::new();
    for (name, group) in ctx.children("annotations")? {
        let (Some(ids_ds), Some(cls_ds)) = (sub_dataset(&group, "instance_ids"), sub_dataset(&group, "class_ids"))
        else {
            continue;
        };
        let ids: Vec<u64> = data::read(&ids_ds)?.cast::<u64>().iter().copied().collect();
        let classes: Vec<i64> = read_i64(&cls_ds)?.iter().copied().collect();
        if ids.len() != classes.len() {
            continue;
        }
        for (i, c) in ids.iter().zip(&classes) {
            table.entry(*i).or_default().entry(*c).or_default().push(name.clone());
        }
    }
    for (instance_id, by_class) in table {
        if by_class.len() < 2 {
            continue;
        }
        let where_: BTreeSet<&String> = by_class.values().flatten().collect();
        if where_.len() < 2 {
            continue;
        }
        let detail = by_class
            .iter()
            .map(|(c, names)| {
                let unique: BTreeSet<&String> = names.iter().collect();
                format!("class {c} in {}", repr_list(&unique.into_iter().collect::<Vec<_>>()))
            })
            .collect::<Vec<_>>()
            .join(", ");
        out.push(ctx.err(
            "W909",
            "/annotations",
            format!(
                "instance id {instance_id} carries several class ids across annotations ({detail}); the longitudinal join treats these as one object"
            ),
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{s, Array4};

    #[test]
    fn p10_the_overlap_graph_is_read_in_slabs() {
        // Two layers of 8 rows of 4x4 uint16: 64 bytes a row, so a 64-byte
        // budget reads one row per slab.
        let dir = tempfile::tempdir().unwrap();
        let file = hdf5::File::create(dir.path().join("layers.h5")).unwrap();
        let mut data = Array4::<u16>::zeros((2, 8, 4, 4));
        data.slice_mut(s![0, 1..6, .., ..]).fill(1); // liver
        data.slice_mut(s![0, 7, .., ..]).fill(3); // spleen, in the liver's layer
        data.slice_mut(s![1, 5, 1..3, 1..3]).fill(2); // lesion, inside the liver
        data[[1, 3, 0, 0]] = 65535; // ignore over the liver: not an overlap
        let ds = file.new_dataset_builder().with_data(&data).create("data").unwrap();
        let whole = overlap_edges(&ds, 65535).unwrap();
        assert_eq!(whole, BTreeSet::from([(1, 2)]));
        assert_eq!(overlap_edges_within(&ds, 65535, 64).unwrap(), whole);
        assert_eq!(overlap_edges_within(&ds, 65535, 1).unwrap(), whole);
    }
}
