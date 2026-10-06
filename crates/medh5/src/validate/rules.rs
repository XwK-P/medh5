//! The validation rules, one section of the spec at a time (§15).
//!
//! Each rule returns every diagnostic it finds; a rule that cannot read what
//! it is checking returns an error, which the driver turns into E001 so the
//! remaining rules still run.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use ndarray::{Array2, ArrayD};

use super::{Diagnostic, SAMPLES_GROUP};
use crate::annotations::encode_geometric::{check_slice_index, ROTATION_TOL, SCOPES, SPACES};
use crate::annotations::header::{ANNOTATION_KINDS, GEOMETRIC_KINDS, RESERVED_KINDS, TASKS, VOXEL_KINDS};
use crate::annotations::payload::{contains_value, SLAB_BYTES};
use crate::annotations::select::greedy_colour;
use crate::array::{Index, Slice};
use crate::curation::provenance::{is_timestamp, ACTIVITY_TYPES};
use crate::digest::parse_digest;
use crate::document::{validate_against_schema, SampleDocument};
use crate::geometry::affine::{is_orthonormal, is_proper_rotation, ORTHONORMAL_TOL};
use crate::geometry::grid::{read_grid, AXIS_KINDS};
use crate::geometry::multiscale::{check_pyramid, GEOMETRY_RTOL};
use crate::h5::attrs::{self, AttrValue};
use crate::h5::data::{self, Kind};
use crate::h5::ops::{self, Node, NodeKind};
use crate::ids::{is_valid_id, matches_pattern};
use crate::integrity::{stale_index_entries, verify_root, AttrNameMap};
use crate::json::{format_g, py_float, repr_int_list, repr_int_tuple, repr_list, repr_str};
use crate::labels::{BACKGROUND_ID, CLOSURES, IGNORE_ID};
use crate::sample::image::VALUE_TYPES;
use crate::sample::reader::PROFILES;
use crate::storage::codecs::is_bulk;
use crate::transforms::model::{composite_cycle, Siblings, TRANSFORM_KINDS, VECTOR_SPACES};
use crate::Result;

/// The major version this validator implements.
pub const SUPPORTED_MAJOR: &str = "1";
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
        "obb" => &[("centers", &["float32", "float64"]), ("sizes", &["float32", "float64"]), ("rotations", &["float32", "float64"])],
        "keypoints" => &[("points", &["float32", "float64"]), ("visibility", &["uint8"])],
        "points" => &[("points", &["float32", "float64"])],
        "contours" => &[("vertices", &["float32", "float64"]), ("contour_offsets", &["int64"])],
        "mesh" => &[("vertices", &["float32", "float64"]), ("faces", &["int32", "int64"])],
        "classification" => &[("class_ids", &["uint16"]), ("values", &["float32", "float64"])],
        _ => &[],
    }
}

/// What every rule gets: the file, the parsed document, and the level.
pub struct Context {
    pub root: hdf5::Group,
    pub path: String,
    pub level: String,
    pub profiles: Vec<String>,
    pub document: Option<SampleDocument>,
    pub errors_only: bool,
    pub schema_checked: bool,
    pub attr_names: Option<AttrNameMap>,
}

/// One rule: a name and the check.
pub type Rule = fn(&mut Context) -> Result<Vec<Diagnostic>>;

impl Context {
    pub fn new(root: hdf5::Group, path: &str, level: &str, profiles: Vec<String>, errors_only: bool) -> Context {
        Context {
            root,
            path: path.into(),
            level: level.into(),
            profiles,
            document: None,
            errors_only,
            schema_checked: false,
            attr_names: None,
        }
    }

    pub fn err(&self, code: &str, location: impl Into<String>, message: impl Into<String>) -> Diagnostic {
        Diagnostic {
            code: code.into(),
            location: location.into(),
            message: message.into(),
            severity: crate::codes::get(code).map(|c| c.severity.clone()).unwrap_or_else(|| "error".into()),
            level: self.level.clone(),
        }
    }

    fn children(&self, group: &str) -> Result<Vec<(String, Node)>> {
        match ops::child_group(&self.root, group) {
            None => Ok(Vec::new()),
            Some(g) => children(&g),
        }
    }
}

/// The members of a group as nodes, in name order.
pub fn children(group: &hdf5::Group) -> Result<Vec<(String, Node)>> {
    let mut out = Vec::new();
    for name in ops::members(group)? {
        if let Some(node) = child(group, &name) {
            out.push((name, node));
        }
    }
    Ok(out)
}

/// One member as a node.
pub fn child(group: &hdf5::Group, name: &str) -> Option<Node> {
    match ops::node_kind(group, name) {
        Some(NodeKind::Group) => group.group(name).ok().map(Node::Group),
        Some(NodeKind::Dataset) => group.dataset(name).ok().map(Node::Dataset),
        _ => None,
    }
}

fn loc(node: &Node) -> &hdf5::Location {
    match node {
        Node::Group(g) => g,
        Node::Dataset(d) => d,
    }
}

fn sub_dataset(node: &Node, name: &str) -> Option<hdf5::Dataset> {
    match node {
        Node::Group(g) => ops::child_dataset(g, name),
        Node::Dataset(_) => None,
    }
}

fn has_member(node: &Node, name: &str) -> bool {
    match node {
        Node::Group(g) => ops::exists(g, name),
        Node::Dataset(_) => false,
    }
}

/// The NumPy dtype name of a dataset (`object` for strings).
fn dtype_name(ds: &hdf5::Dataset) -> String {
    match data::kind(ds) {
        Ok(Kind::Numeric(d)) => d.name().to_string(),
        Ok(Kind::Strings) => "object".into(),
        Ok(Kind::Other(t)) => t,
        Err(e) => e.to_string(),
    }
}

fn read_f64(ds: &hdf5::Dataset) -> Result<ArrayD<f64>> {
    Ok(data::read(ds)?.to_f64())
}

fn read_i64(ds: &hdf5::Dataset) -> Result<ArrayD<i64>> {
    Ok(data::read(ds)?.cast::<i64>())
}

fn str_attr(location: &hdf5::Location, name: &str) -> Result<Option<String>> {
    attrs::get_str(location, name)
}

fn strs_attr(location: &hdf5::Location, name: &str) -> Result<Vec<String>> {
    Ok(attrs::get_strs(location, name)?.unwrap_or_default())
}

fn int_set(location: &hdf5::Location, name: &str) -> Result<BTreeSet<i64>> {
    Ok(attrs::get_i64s(location, name)?.unwrap_or_default().into_iter().collect())
}

fn float_list(values: &[f64]) -> String {
    format!("[{}]", values.iter().map(|v| py_float(*v)).collect::<Vec<_>>().join(", "))
}

// -- §2 container ----------------------------------------------------------------------

pub fn check_container(ctx: &mut Context) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let root = ctx.root.clone();
    match str_attr(&root, "medh5_version")? {
        None => out.push(ctx.err("E001", "/", "root has no `medh5_version` attribute")),
        Some(version) => {
            let major = version.split('.').next().unwrap_or("");
            if major != SUPPORTED_MAJOR {
                out.push(ctx.err(
                    "E002",
                    "/",
                    format!("declares MEDH5 {version}; this validator implements {SUPPORTED_MAJOR}.x"),
                ));
            }
        }
    }
    match str_attr(&root, "medh5_kind")? {
        None => out.push(ctx.err("E006", "/", "root has no `medh5_kind` attribute")),
        Some(kind) if kind != "sample" && kind != "collection" => {
            out.push(ctx.err("E006", "/", format!("unknown `medh5_kind` {}", repr_str(&kind))))
        }
        _ => {}
    }
    if !attrs::has(&root, "medh5_profiles") {
        out.push(ctx.err("E007", "/", "root has no `medh5_profiles` attribute"));
    } else {
        let mut unknown: Vec<String> =
            strs_attr(&root, "medh5_profiles")?.into_iter().filter(|p| !PROFILES.contains(&p.as_str())).collect();
        unknown.sort();
        unknown.dedup();
        if !unknown.is_empty() {
            out.push(ctx.err("E007", "/", format!("unknown profile(s) {}", repr_list(&unknown))));
        }
    }
    for required in ["grids", "images"] {
        if !ops::exists(&root, required) {
            out.push(ctx.err("E008", format!("/{required}"), format!("required group `{required}` is absent")));
        }
    }
    if !ops::exists(&root, "meta") {
        out.push(ctx.err("E004", "/meta", "required dataset `meta` is absent"));
    }
    for group in ["grids", "images", "annotations", "transforms"] {
        let Some(node) = ops::child_group(&root, group) else { continue };
        for name in ops::members(&node)? {
            if !is_valid_id(&name) || name == "meta" {
                out.push(ctx.err(
                    "E003",
                    format!("/{group}/{name}"),
                    format!("identifier {} does not match [A-Za-z0-9_.-]{{1,128}}", repr_str(&name)),
                ));
            }
        }
    }
    Ok(out)
}

/// Rules that apply to a `collection` root itself (§2.2).
pub fn check_collection(ctx: &mut Context) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let root = ctx.root.clone();
    match str_attr(&root, "medh5_version")? {
        None => out.push(ctx.err("E001", "/", "collection root has no `medh5_version` attribute")),
        Some(v) if v.split('.').next().unwrap_or("") != SUPPORTED_MAJOR => out.push(ctx.err(
            "E002",
            "/",
            format!("declares MEDH5 {v}; this validator implements {SUPPORTED_MAJOR}.x"),
        )),
        _ => {}
    }
    let Some(node) = ops::child_group(&root, SAMPLES_GROUP) else {
        out.push(ctx.err("E008", format!("/{SAMPLES_GROUP}"), format!("a `collection` requires a `{SAMPLES_GROUP}` group")));
        return Ok(out);
    };
    let keys = ops::members(&node)?;
    if keys.is_empty() {
        out.push(ctx.err("E008", format!("/{SAMPLES_GROUP}"), "collection contains no sample roots"));
    }
    for key in keys {
        let location = format!("/{SAMPLES_GROUP}/{key}");
        if !matches_pattern(&key, 255) {
            out.push(ctx.err(
                "E003",
                location.clone(),
                format!("sample key {} does not match [A-Za-z0-9_.-]{{1,255}}", repr_str(&key)),
            ));
        }
        let Some(member) = child(&node, &key) else { continue };
        if !attrs::has(loc(&member), "medh5_profiles") {
            out.push(ctx.err("E007", location.clone(), "a sample root in a collection carries its own `medh5_profiles`"));
        }
        if !attrs::has(loc(&member), "content_id") {
            out.push(ctx.err(
                "E010",
                location,
                "a sample root in a collection carries its own `content_id`, so extracting it yields an identifiable sample",
            ));
        }
    }
    Ok(out)
}

/// Parse and check `/meta`: E004 is "not JSON", E005 "JSON the schema rejects".
pub fn check_document(ctx: &mut Context) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let Some(meta) = ops::child_dataset(&ctx.root, "meta") else {
        return Ok(out);
    };
    let text = data::read_scalar_string(&meta)?;
    let parsed: serde_json::Value = match crate::json::loads(&text) {
        Ok(v) => v,
        Err(e) => {
            out.push(ctx.err("E004", "/meta", format!("`meta` is not valid JSON: {e}")));
            return Ok(out);
        }
    };
    if !parsed.is_object() {
        out.push(ctx.err("E004", "/meta", "`meta` must hold a JSON object"));
        return Ok(out);
    }
    ctx.schema_checked = true;
    let mut schema_failed = false;
    for message in validate_against_schema(&parsed) {
        schema_failed = true;
        let (location, detail) = match message.split_once(": ") {
            Some((l, d)) if !d.is_empty() => (l.to_string(), d.to_string()),
            _ => (message.clone(), message.clone()),
        };
        out.push(ctx.err("E005", format!("/meta#{location}"), detail));
    }
    match SampleDocument::from_json(&parsed) {
        Ok(doc) => ctx.document = Some(doc),
        Err(e) => {
            let code = e.code().unwrap_or("E005").to_string();
            if !(schema_failed && code == "E005") {
                out.push(ctx.err(&code, "/meta", e.to_string()));
            }
        }
    }
    Ok(out)
}

pub fn check_bulk_storage(ctx: &mut Context) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    for (name, ds) in ops::datasets(&ctx.root)? {
        if name == "meta" || name.starts_with("index/") {
            continue;
        }
        if is_bulk(&ds) {
            let chunked = data::chunks(&ds).is_some();
            let filtered = !data::filters(&ds)?.is_empty();
            if !chunked || !filtered {
                let mib = data::nbytes(&ds)? as f64 / 1024.0 / 1024.0;
                out.push(ctx.err(
                    "W902",
                    format!("/{name}"),
                    format!("{mib:.1} MiB dataset is {}", if !chunked { "unchunked" } else { "uncompressed" }),
                ));
            }
        }
    }
    Ok(out)
}

// -- §3 geometry and timepoints ---------------------------------------------------------

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
        let missing: Vec<&str> = ["shape", "axis_names", "axis_kinds", "spacing", "origin", "direction", "coord_system", "units"]
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
            out.push(ctx.err("E104", location.clone(), format!("spacing {} must be strictly positive", float_list(&spacing))));
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

// -- §4 images -----------------------------------------------------------------------------

fn grid_shape(grids: &hdf5::Group, grid_id: &str) -> Result<Vec<i64>> {
    let g = grids.group(grid_id)?;
    Ok(attrs::get_i64s(&g, "shape")?.unwrap_or_default())
}

fn grid_spatial(grids: &hdf5::Group, grid_id: &str) -> Result<Vec<i64>> {
    let g = grids.group(grid_id)?;
    let shape = attrs::get_i64s(&g, "shape")?.unwrap_or_default();
    let kinds = strs_attr(&g, "axis_kinds")?;
    Ok(shape.iter().zip(&kinds).filter(|(_, k)| *k == "spatial").map(|(s, _)| *s).collect())
}

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
                out.push(ctx.err("E203", location.clone(), format!("unknown value_type {}", repr_str(&vt))));
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
    let (Some(node), Some(grids)) = (ops::child_group(&ctx.root, "images"), ops::child_group(&ctx.root, "grids")) else {
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
            out.push(ctx.err("E101", location, format!("grid_levels reference missing grids {}", repr_list(&level_ids))));
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

// -- §5 label set ----------------------------------------------------------------------------

pub fn check_label_set(ctx: &mut Context) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let Some(doc) = &ctx.document else { return Ok(out) };
    let mut needs: Vec<&str> = ["seg", "det", "cls"].into_iter().filter(|p| ctx.profiles.iter().any(|x| x == p)).collect();
    needs.sort_unstable();
    let Some(ls) = &doc.label_set else {
        if !needs.is_empty() {
            out.push(ctx.err("E301", "/meta#label_set", format!("profile(s) {} require a label set", repr_list(&needs))));
        }
        return Ok(out);
    };
    if ls.form == "ref" && ls.uri.as_deref().unwrap_or("").is_empty() {
        out.push(ctx.err("E305", "/meta#label_set", "`form: ref` requires a `uri`"));
    }
    let mut seen_ids = BTreeSet::new();
    let mut seen_keys = BTreeSet::new();
    for entry in ls.classes() {
        let location = format!("/meta#label_set/classes/{}", entry.key);
        if entry.id == BACKGROUND_ID || entry.id == IGNORE_ID {
            out.push(ctx.err("E303", location.clone(), format!("id {} is reserved", entry.id)));
        }
        if !seen_ids.insert(entry.id) {
            out.push(ctx.err("E302", location.clone(), format!("duplicate class id {}", entry.id)));
        }
        if !seen_keys.insert(entry.key.clone()) {
            out.push(ctx.err("E302", location, format!("duplicate class key {}", repr_str(&entry.key))));
        }
    }
    if let Err(e) = ls.check() {
        let code = e.code().unwrap_or("E306").to_string();
        out.push(ctx.err(&code, "/meta#label_set", e.to_string()));
    }
    Ok(out)
}

pub fn check_ontology_bindings(ctx: &mut Context) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let Some(ls) = ctx.document.as_ref().and_then(|d| d.label_set.as_ref()) else { return Ok(out) };
    let mut used = BTreeSet::new();
    for (_, group) in ctx.children("annotations")? {
        used.extend(int_set(loc(&group), "class_ids")?);
    }
    let unbound: Vec<i64> = used.into_iter().filter(|c| ls.by_id(*c).is_some_and(|e| e.codes.is_empty())).collect();
    if !unbound.is_empty() {
        let shown: Vec<i64> = unbound.iter().take(8).copied().collect();
        out.push(ctx.err(
            "W912",
            "/meta#label_set",
            format!(
                "{} class(es) used by annotations have no ontology binding: {}{}",
                unbound.len(),
                repr_int_list(&shown),
                if unbound.len() > 8 { "..." } else { "" }
            ),
        ));
    }
    Ok(out)
}

// -- §6-§7 annotations ----------------------------------------------------------------------

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
            out.push(ctx.err("E412", location.clone(), "missing `annotated_class_ids`; the coverage contract is required"));
        }
        let class_ids = int_set(a, "class_ids")?;
        let annotated = int_set(a, "annotated_class_ids")?;
        if !annotated.is_subset(&class_ids) && kind != "mask" {
            let extra: Vec<i64> = annotated.difference(&class_ids).copied().collect();
            out.push(ctx.err("E403", location.clone(), format!("annotated_class_ids {} are not in class_ids", repr_int_list(&extra))));
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
            out.push(ctx.err("E303", location.clone(), format!("class_ids uses reserved id(s) {}", repr_int_list(&reserved))));
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
                None => out.push(ctx.err("E412", location.clone(), format!("kind {} requires a `grid`", repr_str(&kind)))),
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
            out.extend(check_mask_reference(ctx, &format!("/images/{name}"), "valid_mask", &target, loc(&node), &groups)?);
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
        return Ok(vec![ctx.err("E413", location, format!("`{attr}` names annotation {}, which does not exist", repr_str(target)))]);
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
                out.push(ctx.err("E404", location.clone(), format!("class_ids {} are not assigned to any layer", repr_int_list(&missing))));
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
                out.push(ctx.err("E404", location.clone(), format!("{planes} bitplanes for {n_classes} classes; expected {expected}")));
            }
        }
    }
    if kind == "instances" {
        out.extend(check_instances(ctx, name, group)?);
    }
    Ok(out)
}

fn check_layer_optimality(ctx: &Context, name: &str, group: &Node, n_layers: usize, n_classes: usize) -> Result<Vec<Diagnostic>> {
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
fn overlap_edges(ds: &hdf5::Dataset, ignore_id: i64) -> Result<BTreeSet<(i64, i64)>> {
    let shape = ds.shape();
    let mut edges = BTreeSet::new();
    if shape.len() < 2 || shape[0] < 2 || shape.iter().product::<usize>() == 0 {
        return Ok(edges);
    }
    let n_layers = shape[0];
    let rows = shape[1];
    let itemsize = data::dtype(ds)?.itemsize();
    let per_row = n_layers * shape[2..].iter().product::<usize>() * itemsize;
    let step = (SLAB_BYTES / per_row.max(1)).clamp(1, rows);
    let mut start = 0;
    while start < rows {
        let block = data::read_region(ds, &[Index::Slice(Slice::full()), Index::Slice(Slice::new(start as i64, (start + step) as i64))])?
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

fn check_geometric(
    ctx: &Context,
    name: &str,
    group: &Node,
    kind: &str,
    grid_id: Option<&str>,
    grids: Option<&hdf5::Group>,
) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    if !GEOMETRIC_KINDS.contains(&kind) {
        return Ok(out);
    }
    let location = format!("/annotations/{name}");
    let a = loc(group);
    let space = str_attr(a, "space")?;
    match space.as_deref() {
        None => out.push(ctx.err("E412", location.clone(), format!("kind {} requires a `space` attribute", repr_str(kind)))),
        Some(s) if !SPACES.contains(&s) => out.push(ctx.err("E412", location.clone(), format!("unknown space {}", repr_str(s)))),
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
            if let (Some(planes_ds), 3) = (sub_dataset(group, "slice_index"), boxes.ndim()) {
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

fn bad_boxes(boxes: &ArrayD<f64>) -> usize {
    boxes
        .outer_iter()
        .filter(|b| b.outer_iter().any(|axis| axis.len() == 2 && axis[0] > axis[1]))
        .count()
}

fn check_keypoints(ctx: &Context, name: &str, group: &Node) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let location = format!("/annotations/{name}");
    let Some(points) = sub_dataset(group, "points") else { return Ok(out) };
    let shape = points.shape();
    if shape.len() != 3 {
        out.push(ctx.err("E405", format!("{location}/points"), format!("expected (N, K, S), got {}", repr_int_tuple(&shape))));
        return Ok(out);
    }
    let (n, k) = (shape[0], shape[1]);
    if let Some(kc) = sub_dataset(group, "keypoint_class_ids") {
        let got = kc.shape().first().copied().unwrap_or(0);
        if got != k {
            out.push(ctx.err("E405", location.clone(), format!("`keypoint_class_ids` has {got} entries for {k} keypoint slots")));
        }
    }
    if let Some(vis) = sub_dataset(group, "visibility") {
        let v = read_i64(&vis)?;
        if v.shape() != [n, k] {
            out.push(ctx.err(
                "E405",
                location.clone(),
                format!("`visibility` {} must be ({n}, {k})", repr_int_tuple(v.shape())),
            ));
        } else if !v.is_empty() && v.iter().copied().max().unwrap_or(0) > 2 {
            out.push(ctx.err("E411", format!("{location}/visibility"), "values must be 0 (unlabelled), 1 (occluded) or 2 (visible)"));
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
            out.push(ctx.err("E405", format!("{location}/faces"), format!("face indices reach outside the {n_vertices} vertices")));
        }
    }
    if let Some(normals) = sub_dataset(group, "normals") {
        if normals.shape() != vertices.shape() {
            out.push(ctx.err(
                "E405",
                format!("{location}/normals"),
                format!(
                    "{} must match vertices {}",
                    repr_int_tuple(&normals.shape()),
                    repr_int_tuple(&vertices.shape())
                ),
            ));
        }
    }
    Ok(out)
}

fn check_classification(ctx: &Context, name: &str, group: &Node, kind: &str) -> Result<Vec<Diagnostic>> {
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
    let (Some(cds), Some(vds)) = (sub_dataset(group, "class_ids"), sub_dataset(group, "values")) else {
        return Ok(out);
    };
    let class_shape = cds.shape();
    let values: Vec<f64> = read_f64(&vds)?.iter().copied().collect();
    if vds.shape() != class_shape {
        out.push(ctx.err(
            "E405",
            location,
            format!("`values` {} must match `class_ids` {}", repr_int_tuple(&vds.shape()), repr_int_tuple(&class_shape)),
        ));
        return Ok(out);
    }
    if !values.is_empty() {
        let lo = values.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if lo < 0.0 || hi > 1.0 {
            out.push(ctx.err(
                "E404",
                location.clone(),
                "values must lie in [0, 1]; 1.0 is a hard positive, 0.0 an explicit negative",
            ));
        }
    }
    let mut scope_ids: Option<Vec<i64>> = match sub_dataset(group, "scope_ids") {
        Some(ds) => Some(read_i64(&ds)?.iter().copied().collect()),
        None => None,
    };
    if let Some(ds) = sub_dataset(group, "scope_ids") {
        if ds.shape() != class_shape {
            out.push(ctx.err(
                "E405",
                location.clone(),
                format!("`scope_ids` {} must match `class_ids` {}", repr_int_tuple(&ds.shape()), repr_int_tuple(&class_shape)),
            ));
            scope_ids = None;
        }
    }
    let multilabel = attrs::get_bool(a, "multilabel")?.unwrap_or(true);
    if !multilabel && !values.is_empty() {
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
    for column in ["schemes", "scheme_values"] {
        if let Some(ds) = sub_dataset(group, column) {
            if ds.shape() != class_shape {
                out.push(ctx.err(
                    "E405",
                    format!("{location}/{column}"),
                    format!("{} must match `class_ids` {}", repr_int_tuple(&ds.shape()), repr_int_tuple(&class_shape)),
                ));
            }
        }
    }
    Ok(out)
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
        let (Some(ids_ds), Some(cls_ds)) = (sub_dataset(&group, "instance_ids"), sub_dataset(&group, "class_ids")) else {
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

// -- §10 transforms ---------------------------------------------------------------------------

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
            "displacement" | "bspline" => out.extend(check_field_transform(ctx, &name, &t, &kind, &source, grids.as_ref())?),
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
        return Ok(vec![ctx.err("E504", location, format!("last row must be [0 \u{2026} 0 1], got {}", float_list(&last)))]);
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
    let (dataset, grid_attr) = if kind == "displacement" { ("field", "field_grid") } else { ("control_points", "cp_grid") };
    if !has_member(group, dataset) {
        out.push(ctx.err("E502", location.clone(), format!("kind {} requires a {} dataset", repr_str(kind), repr_str(dataset))));
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

fn check_composite(ctx: &Context, name: &str, group: &Node, node: &hdf5::Group, source: &str, target: &str) -> Result<Vec<Diagnostic>> {
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
                format!("first component starts in {} but the composite declares {}", repr_str(&first.0), repr_str(source)),
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

fn check_inverses(ctx: &Context, node: &hdf5::Group, declared: &BTreeMap<String, (String, String)>) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    for (name, (source, target)) in declared {
        let t = child(node, name).ok_or_else(|| crate::Error::Key(repr_str(name)))?;
        let Some(other) = str_attr(loc(&t), "inverse_id")? else { continue };
        let location = format!("/transforms/{name}");
        let Some((os, ot)) = declared.get(&other) else {
            out.push(ctx.err("E505", location, format!("`inverse_id` names {}, which does not exist", repr_str(&other))));
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

// -- §11-§12 curation ---------------------------------------------------------------------------

pub fn check_curation(ctx: &mut Context) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let Some(doc) = ctx.document.clone() else { return Ok(out) };
    let prov = &doc.provenance;
    for activity in prov.activities() {
        let location = format!("/meta#provenance/activities/{}", activity.id);
        if !ACTIVITY_TYPES.contains(&activity.r#type.as_str()) {
            out.push(ctx.err("E603", location.clone(), format!("unknown activity type {}", repr_str(&activity.r#type))));
        }
        for (field, value) in [("started", &activity.started), ("ended", &activity.ended)] {
            if let Some(v) = value {
                if !is_timestamp(v) {
                    out.push(ctx.err("E604", location.clone(), format!("{field} {} is not RFC 3339", repr_str(v))));
                }
            }
        }
    }
    for (activity_id, agent_id) in prov.dangling_agent_refs() {
        out.push(ctx.err(
            "E605",
            format!("/meta#provenance/activities/{activity_id}"),
            format!("names agent {}, which is not declared", repr_str(&agent_id)),
        ));
    }
    for (name, group) in ctx.children("annotations")? {
        out.extend(check_links(ctx, &format!("/annotations/{name}"), loc(&group), &doc, "quality")?);
    }
    for (name, node) in ctx.children("images")? {
        out.extend(check_links(ctx, &format!("/images/{name}"), loc(&node), &doc, "quality")?);
    }
    for (name, node) in ctx.children("transforms")? {
        out.extend(check_links(ctx, &format!("/transforms/{name}"), loc(&node), &doc, "metrics")?);
    }
    if doc.deidentification.is_none() {
        out.push(ctx.err(
            "W903",
            "/meta#deidentification",
            "no de-identification record; tooling must treat this file as potentially identifying",
        ));
    }
    if ctx.profiles.iter().any(|p| p == "curation") {
        for (name, group) in ctx.children("annotations")? {
            if !attrs::has(loc(&group), "quality") {
                out.push(ctx.err(
                    "E009",
                    format!("/annotations/{name}"),
                    "the `curation` profile requires `quality` on every annotation",
                ));
            }
        }
    }
    Ok(out)
}

fn check_links(ctx: &Context, location: &str, a: &hdf5::Location, doc: &SampleDocument, quality_attr: &str) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    if let Some(activity) = str_attr(a, "prov")? {
        if !doc.provenance.has_activity(&activity) {
            out.push(ctx.err("E601", location, format!("`prov` names unknown activity {}", repr_str(&activity))));
        }
    }
    if let Some(key) = str_attr(a, quality_attr)? {
        if !doc.quality.contains_key(&key) {
            out.push(ctx.err("E602", location, format!("`{quality_attr}` names unknown record {}", repr_str(&key))));
        }
    }
    Ok(out)
}

pub fn check_splits(ctx: &mut Context) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let Some(doc) = &ctx.document else { return Ok(out) };
    let mut by_set: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for claim in &doc.splits {
        if let Some(sha) = claim.manifest_sha256.as_ref().filter(|s| !s.is_empty()) {
            by_set.entry(claim.set_id.clone()).or_default().insert(sha.clone());
        }
    }
    for (set_id, hashes) in by_set {
        if hashes.len() > 1 {
            out.push(ctx.err(
                "W906",
                "/meta#splits",
                format!("split set {} is claimed against {} different manifests in one file", repr_str(&set_id), hashes.len()),
            ));
        }
    }
    Ok(out)
}

// -- §13 integrity ------------------------------------------------------------------------------

pub fn check_integrity(ctx: &mut Context) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let root = ctx.root.clone();
    let mut algo_known = true;
    if let Some(algo) = str_attr(&root, "digest_algo")? {
        if parse_digest(&format!("{algo}:00")).is_err() {
            algo_known = false;
            out.push(ctx.err(
                "E703",
                "/",
                format!("unsupported digest_algo {}; §2.1 permits sha256, sha512 and blake2b", repr_str(&algo)),
            ));
        }
    }
    let result = verify_root(&root, ctx.attr_names.as_ref(), None, algo_known)?;
    if result.checked.is_empty() && result.undigested.is_empty() {
        return Ok(out);
    }
    if !result.undigested.is_empty() && result.checked.is_empty() {
        out.push(ctx.err("W901", "/", "no dataset carries a `digest` attribute"));
    }
    for path in &result.malformed {
        out.push(ctx.err("E703", format!("/{path}"), "malformed digest string"));
    }
    for path in &result.mismatched {
        out.push(ctx.err("E701", format!("/{path}"), "digest does not match the stored data"));
    }
    if result.content_id_ok() == Some(false) {
        out.push(ctx.err(
            "E702",
            "/",
            format!(
                "`content_id` {} does not match the computed {}",
                result.content_id_declared.clone().unwrap_or_default(),
                result.content_id_computed.clone().unwrap_or_default()
            ),
        ));
    }
    for name in stale_index_entries(&root)? {
        out.push(ctx.err(
            "W905",
            format!("/index/{name}"),
            "index `source_digest` does not match its annotation; readers must ignore this entry and rebuild it",
        ));
    }
    Ok(out)
}

// -- §1.3 profiles -------------------------------------------------------------------------------

pub fn check_profiles(ctx: &mut Context) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let mut kinds = BTreeSet::new();
    let mut tasks = BTreeSet::new();
    for (_, g) in ctx.children("annotations")? {
        if let Some(k) = str_attr(loc(&g), "kind")? {
            kinds.insert(k);
        }
        if let Some(t) = str_attr(loc(&g), "task")? {
            tasks.insert(t);
        }
    }
    let declared: BTreeSet<&str> = ctx.profiles.iter().map(String::as_str).collect();
    if declared.contains("seg") && !kinds.iter().any(|k| VOXEL_KINDS.contains(&k.as_str()) && k != "mask") {
        out.push(ctx.err("E009", "/", "profile `seg` is declared but no voxel annotation is present"));
    }
    if declared.contains("det") && !tasks.contains("detection") {
        out.push(ctx.err("E009", "/", "profile `det` is declared but no annotation declares task='detection'"));
    }
    if declared.contains("cls") && !kinds.contains("classification") {
        out.push(ctx.err("E009", "/", "profile `cls` is declared but no classification annotation is present"));
    }
    let count = |g: &str| -> Result<usize> {
        Ok(match ops::child_group(&ctx.root, g) {
            None => 0,
            Some(node) => ops::members(&node)?.len(),
        })
    };
    if declared.contains("reg") && count("transforms")? == 0 {
        out.push(ctx.err("E009", "/", "profile `reg` is declared but no transform is present"));
    }
    if declared.contains("training") && count("index")? == 0 {
        out.push(ctx.err("E009", "/", "profile `training` is declared but no sampling index is present"));
    }
    if declared.contains("longitudinal") {
        if let Some(doc) = &ctx.document {
            if doc.timepoints.len() < 2 {
                out.push(ctx.err("E009", "/meta#timepoints", "profile `longitudinal` requires at least two declared timepoints"));
            }
        }
    }
    Ok(out)
}

/// The rules a level runs, with their names.  Each level includes every
/// earlier level's rules.
pub fn rules_for(level: &str) -> Vec<(&'static str, Rule)> {
    let structural: Vec<(&'static str, Rule)> = vec![
        ("check_container", check_container),
        ("check_document", check_document),
        ("check_geometry", check_geometry),
        ("check_images", check_images),
        ("check_annotations", check_annotations),
        ("check_bulk_storage", check_bulk_storage),
    ];
    let semantic: Vec<(&'static str, Rule)> = vec![
        ("check_timepoints", check_timepoints),
        ("check_instance_identity", check_instance_identity),
        ("check_references", check_references),
        ("check_transforms", check_transforms),
        ("check_label_set", check_label_set),
        ("check_multiscale", check_multiscale),
        ("check_curation", check_curation),
        ("check_splits", check_splits),
        ("check_profiles", check_profiles),
        ("check_ontology_bindings", check_ontology_bindings),
    ];
    let integrity: Vec<(&'static str, Rule)> = vec![("check_integrity", check_integrity)];
    match level {
        "structural" => structural,
        "semantic" => [structural, semantic].concat(),
        _ => [structural, semantic, integrity].concat(),
    }
}

