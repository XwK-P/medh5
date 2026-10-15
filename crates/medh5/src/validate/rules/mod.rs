//! The validation rules, one section of the spec at a time (§15).
//!
//! Each rule returns every diagnostic it finds.  An error met on one object
//! is that object's finding --- under the error's own code, or E001 (it could
//! not be read) when it has none --- and the rule goes on to the next; a rule
//! that stops on an error has it reported by the driver the same way, and the
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
//! - `clinical` --- 1.1 §3--§8

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
use crate::geometry::grid::AXIS_KINDS;
use crate::h5::attrs;
use crate::h5::data::{self, Kind};
use crate::h5::ops::{self, LinkKind, Node, NodeKind};
use crate::integrity::AttrNameMap;
use crate::json::{py_float, repr_str};
use crate::validate::Diagnostic;
use crate::{Error, Result};

pub use annotations::{
    allowed_dtypes, check_annotations, check_instance_identity, check_references, geometric_dtypes, required_datasets,
    W908_TOLERANCE,
};
pub use clinical::{check_clinical, check_clinical_digests, check_clinical_records, ClinicalTables};
pub use container::{
    check_bulk_storage, check_collection, check_container, check_document, check_profiles, SUPPORTED_MAJOR,
};
pub(crate) use container::{records_activity, relates_grids};
pub use curation::{check_curation, check_splits};
pub use geometric::check_implicit_closure;
pub use geometry::{check_geometry, check_timepoints};
pub use images::{check_images, check_multiscale};
pub use integrity::check_integrity;
pub use labels::{check_label_set, check_ontology_bindings};
pub use transforms::check_transforms;

/// What every rule gets: the file, the parsed document, and the level.
pub struct Context {
    /// The sample root.
    pub root: hdf5::Group,
    /// The file, as the report names it.
    pub path: String,
    /// The level the pass runs at.
    pub level: String,
    /// The profiles the sample is held to: those it declares, and any the
    /// caller requires besides.
    pub profiles: Vec<String>,
    /// `/meta`, once the document rule has parsed it; `None` when it could
    /// not be, and the checks that need it are skipped.
    pub document: Option<SampleDocument>,
    /// Skip the warning-only checks that read bulk data (the writer's commit
    /// gate).
    pub errors_only: bool,
    /// Whether `/meta` was checked against its schema.
    pub schema_checked: bool,
    /// The digest attribute names, where integrity is checked.
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
    /// A context for `root`, held to `profiles`, before any rule has run.
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

    /// A finding at `location`, at the severity the code table gives `code`.
    pub fn err(&self, code: &str, location: impl Into<String>, message: impl Into<String>) -> Diagnostic {
        Diagnostic {
            code: code.into(),
            location: location.into(),
            message: message.into(),
            severity: crate::codes::get(code).map(|c| c.severity.clone()).unwrap_or_else(|| "error".into()),
            level: self.level.clone(),
        }
    }

    /// An error met while checking one object, as that object's finding: under
    /// the error's own code when it has one, the defect met, and as E001 when
    /// it has none, the object could not be read.  A rule that reports it and
    /// moves on still checks every other object.
    pub fn failed(&self, location: impl Into<String>, error: &Error) -> Diagnostic {
        match error.code() {
            Some(own) => self.err(own, location, error.message()),
            None => self.err("E001", location, format!("could not be read: {}", error.python_line())),
        }
    }

    /// The members of one of the root's groups that are objects, in name
    /// order.  One that resolves to no object --- a soft link to nothing ---
    /// is not among them: [`check_container`] reports it (E001), once, for
    /// every rule that reads the group, where it was dropped unseen and a
    /// dangling annotation or transform passed every level.
    fn children(&self, group: &str) -> Result<Vec<(String, Node)>> {
        match ops::child_group(&self.root, group) {
            None => Ok(Vec::new()),
            Some(g) => children(&g),
        }
    }
}

/// The members of a group as nodes, in name order: the groups and datasets
/// among them ([`unresolved`] says what the others are).
pub fn children(group: &hdf5::Group) -> Result<Vec<(String, Node)>> {
    let mut out = Vec::new();
    for name in ops::members(group)? {
        if let Some(node) = child(group, &name) {
            out.push((name, node));
        }
    }
    Ok(out)
}

/// What a member of `group` is when it is neither a group nor a dataset ---
/// a soft link to nothing, an external link, a committed type --- for a
/// message; `None` when it is one of the two.  The readers refuse such a
/// member in the same words (E001).
pub(crate) fn unresolved(group: &hdf5::Group, name: &str) -> Option<String> {
    match ops::node_kind(group, name) {
        Some(NodeKind::Group | NodeKind::Dataset) => None,
        Some(NodeKind::Other) if ops::link_kind(group, name) == Some(LinkKind::External) => {
            Some("an external link, which leads out of the file".into())
        }
        Some(NodeKind::Other) => Some("neither a group nor a dataset".into()),
        None => Some(match ops::soft_link_target(group, name) {
            Ok(target) => format!("a soft link to {}, which names no object", repr_str(&target)),
            Err(_) => "a link that resolves to no object".into(),
        }),
    }
}

/// What a member of `group` is, for a message: a group, a dataset, or what
/// [`unresolved`] says.
fn describe(group: &hdf5::Group, name: &str) -> String {
    match ops::node_kind(group, name) {
        Some(NodeKind::Group) => "a group".into(),
        Some(NodeKind::Dataset) => "a dataset".into(),
        _ => unresolved(group, name).unwrap_or_else(|| "absent".into()),
    }
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

/// Whether `node` holds the dataset `name`: a group of that name, or a link
/// to nothing, is no dataset to read.
fn has_member(node: &Node, name: &str) -> bool {
    match node {
        Node::Group(g) => ops::is_dataset(g, name),
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

/// Whether this validator knows every one of a grid's `axis_kinds`.  Where it
/// does not, which axes are spatial is not known: nothing is measured against
/// them, and the grid's own finding (E110, W913 in a later minor's file)
/// stands for what would have been.
fn kinds_known(grids: &hdf5::Group, grid_id: &str) -> Result<bool> {
    let g = grids.group(grid_id)?;
    Ok(strs_attr(&g, "axis_kinds")?.iter().all(|k| AXIS_KINDS.contains(&k.as_str())))
}

/// The rules a level runs, with their names.  Each level includes every
/// earlier level's rules.
///
/// A level this validator does not know runs no rule: every rule, which it
/// was given, passed a misspelt level for a full pass.  The validators refuse
/// one before they ask ([`LEVELS`](crate::validate::LEVELS)).
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
        ("check_implicit_closure", check_implicit_closure),
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
        "integrity" | "strict" => [structural, semantic, integrity].concat(),
        _ => Vec::new(),
    }
}
