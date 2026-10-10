//! The validation rules, one section of the spec at a time (§15).
//!
//! Each rule returns every diagnostic it finds; a rule that cannot read what
//! it is checking returns an error, which the driver turns into E001 so the
//! remaining rules still run.
//!
//! One module per domain of the §15.2 code table, so a rule has one obvious home:
//! - `container` --- §1.3, §2
//! - `geometry` --- §3
//! - `images` --- §4
//! - `labels` --- §5
//! - `annotations` --- §6--§7
//! - `geometric` --- §8--§9
//! - `transforms` --- §10
//! - `curation` --- §11--§12
//! - `integrity` --- §13

mod annotations;
mod clinical;
mod container;
mod curation;
mod geometric;
mod geometry;
mod images;
mod integrity;
mod labels;
mod transforms;

use std::collections::BTreeSet;

use ndarray::ArrayD;

use crate::document::SampleDocument;
use crate::h5::attrs;
use crate::h5::data::{self, Kind};
use crate::h5::ops::{self, Node, NodeKind};
use crate::integrity::AttrNameMap;
use crate::json::py_float;
use crate::validate::Diagnostic;
use crate::Result;

pub use annotations::{
    allowed_dtypes, check_annotations, check_instance_identity, check_references, geometric_dtypes, required_datasets,
    W908_TOLERANCE,
};
pub use clinical::{check_clinical, check_clinical_digests, check_clinical_records, ClinicalTables};
pub use container::{
    check_bulk_storage, check_collection, check_container, check_document, check_profiles, SUPPORTED_MAJOR,
};
pub use curation::{check_curation, check_splits};
pub use geometry::{check_geometry, check_timepoints};
pub use images::{check_images, check_multiscale};
pub use integrity::check_integrity;
pub use labels::{check_label_set, check_ontology_bindings};
pub use transforms::check_transforms;

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
    /// The root's `medh5_version`, when it has one.
    pub version: Option<String>,
    /// A higher minor than this validator implements: only its supported
    /// projection is checked, and what a later minor may define is W913
    /// rather than an error (1.1 §2.2).
    pub projection: bool,
    /// The clinical tables, read once by the structural rule for the rules
    /// after it.
    pub clinical: Option<ClinicalTables>,
}

/// One rule: a name and the check.
pub type Rule = fn(&mut Context) -> Result<Vec<Diagnostic>>;

impl Context {
    pub fn new(root: hdf5::Group, path: &str, level: &str, profiles: Vec<String>, errors_only: bool) -> Context {
        let version = attrs::get_str(&root, "medh5_version").ok().flatten();
        let projection = version.as_deref().is_some_and(crate::version::is_projection);
        Context {
            root,
            path: path.into(),
            level: level.into(),
            profiles,
            document: None,
            errors_only,
            schema_checked: false,
            attr_names: None,
            version,
            projection,
            clinical: None,
        }
    }

    /// A value this validator does not know: `code` for a file of a version it
    /// implements, W913 for a higher minor --- which may define it (1.1 §2.2).
    pub fn unknown(&self, code: &str, location: impl Into<String>, message: impl Into<String>) -> Diagnostic {
        if self.projection {
            self.err(
                "W913",
                location,
                format!(
                    "{} --- not defined by MEDH5 {}, which this validator implements; a later minor may define it, \
                     so it is ignored in this projection ({code})",
                    message.into(),
                    crate::FORMAT_VERSION
                ),
            )
        } else {
            self.err(code, location, message)
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
        ("check_clinical", check_clinical),
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
        ("check_clinical_records", check_clinical_records),
    ];
    let integrity: Vec<(&'static str, Rule)> =
        vec![("check_integrity", check_integrity), ("check_clinical_digests", check_clinical_digests)];
    match level {
        "structural" => structural,
        "semantic" => [structural, semantic].concat(),
        _ => [structural, semantic, integrity].concat(),
    }
}
