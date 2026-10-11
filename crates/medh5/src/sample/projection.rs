//! A later minor's objects, read as this engine's projection (1.1 §2.2).
//!
//! A file of a higher minor may hold an object this engine cannot read
//! because of a value only that minor defines: a grid with an axis kind, an
//! annotation of a kind, task, closure, space or scope, a transform of a kind
//! or vector space this engine does not know.  The validator reports each
//! such value as W913; the reader leaves the object out of what it lists ---
//! with what depends on it, such as an image or annotation on a grid it
//! leaves out --- rather than failing the listing, and a lookup of the object
//! by name raises W913, naming the value.
//!
//! The values are read against the vocabularies the validator holds them to,
//! where it reports them as W913, and only where the reader interprets one: a
//! grid's `units` or an image's `value_type` is carried as written, as in a
//! file of any version.  An object is read as far as a file of a version this
//! engine implements is read before the value is met ([`Met`]), so whatever
//! is wrong with it before then is the error it was; what lies beyond is the
//! validator's to report.  In a file of a version this engine implements,
//! nothing is left out.

use crate::annotations::encode_geometric::{SCOPES, SPACES};
use crate::annotations::header::{ANNOTATION_KINDS, GEOMETRIC_KINDS, RESERVED_KINDS, TASKS};
use crate::geometry::grid::AXIS_KINDS;
use crate::h5::attrs;
use crate::json::repr_str;
use crate::labels::CLOSURES;
use crate::transforms::model::{TRANSFORM_KINDS, VECTOR_SPACES};
use crate::version::LATEST_VERSION;
use crate::{Error, Result};

/// Where reading an object meets the value it holds, in a file of a version
/// this engine implements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Met {
    /// Before anything else: the object's kind, which decides what else it
    /// holds.  Nothing else of it is read, as the validator reads nothing
    /// else of it.
    First,
    /// As the object is read, which fails there with this code: the checks
    /// before it still raise what they raised.
    Reading(&'static str),
    /// Only where the object is used: it is read in full first.
    InUse,
}

/// A value only a later minor defines, held where the reader interprets one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Later {
    /// What the value is, as a sentence names it: `axis kind`, `task`, ...
    pub what: &'static str,
    pub value: String,
    pub met: Met,
}

fn later(what: &'static str, value: &str, known: &[&str], met: Met) -> Option<Later> {
    (!known.contains(&value)).then(|| Later { what, value: value.to_string(), met })
}

/// The later minor's value a grid holds: an axis kind (E110 in a file of a
/// version this engine implements).
pub fn grid_value(group: &hdf5::Group) -> Result<Option<Later>> {
    let kinds = attrs::get_strs(group, "axis_kinds")?.unwrap_or_default();
    Ok(kinds.iter().find_map(|k| later("axis kind", k, &AXIS_KINDS, Met::Reading("E110"))))
}

/// The later minor's value an annotation holds: its kind --- one this engine
/// does not know, or the one §16 reserves (E401) --- its task or closure, a
/// geometric kind's space, or a classification's scope (E412).
pub fn annotation_value(group: &hdf5::Group) -> Result<Option<Later>> {
    let text = |name: &str| attrs::get_str(group, name);
    let Some(kind) = text("kind")? else { return Ok(None) };
    if RESERVED_KINDS.contains(&kind.as_str()) {
        return Ok(Some(Later { what: "kind", value: kind, met: Met::First }));
    }
    if let Some(found) = later("kind", &kind, &ANNOTATION_KINDS, Met::First) {
        return Ok(Some(found));
    }
    let mut checked: Vec<(&'static str, &[&str], Met)> =
        vec![("task", &TASKS, Met::Reading("E412")), ("closure", &CLOSURES, Met::Reading("E412"))];
    if GEOMETRIC_KINDS.contains(&kind.as_str()) {
        checked.push(("space", &SPACES, Met::InUse));
    }
    if kind == "classification" {
        checked.push(("scope", &SCOPES, Met::InUse));
    }
    for (name, known, met) in checked {
        if let Some(found) = text(name)?.and_then(|value| later(name, &value, known, met)) {
            return Ok(Some(found));
        }
    }
    Ok(None)
}

/// The later minor's value a transform holds: its kind, or a displacement
/// field's or B-spline's vector space (E502).
pub fn transform_value(group: &hdf5::Group) -> Result<Option<Later>> {
    let Some(kind) = attrs::get_str(group, "kind")? else { return Ok(None) };
    if let Some(found) = later("kind", &kind, &TRANSFORM_KINDS, Met::First) {
        return Ok(Some(found));
    }
    if matches!(kind.as_str(), "displacement" | "bspline") {
        let space = attrs::get_str(group, "vector_space")?.unwrap_or_else(|| "world".into());
        return Ok(later("vector space", &space, &VECTOR_SPACES, Met::InUse));
    }
    Ok(None)
}

/// Read an object holding `value` with `read`, as far as a file of a version
/// this engine implements is read before the value is met: `Err` is the
/// error the object fails with before then, the error it was; `Ok`, that the
/// projection leaves the object out.
pub fn read_up_to<T>(value: &Later, read: impl FnOnce() -> Result<T>) -> Result<()> {
    match value.met {
        Met::First => Ok(()),
        Met::Reading(code) => match read() {
            Err(e) if e.code() != Some(code) => Err(e),
            _ => Ok(()),
        },
        Met::InUse => read().map(|_| ()),
    }
}

/// The W913 a lookup of an object left out for the value it holds raises.
pub fn holds(what: &str, id: &str, value: &Later, version: &str) -> Error {
    Error::coded(
        "W913",
        format!(
            "{what} {} holds {} {}, which MEDH5 {LATEST_VERSION} does not define: this engine reads MEDH5 {version} \
             as its {LATEST_VERSION} projection, which leaves the {what} out (1.1 §2.2)",
            repr_str(id),
            value.what,
            repr_str(&value.value)
        ),
    )
}

/// The W913 a lookup of an object left out with another raises: `relation`
/// joins the two, as in "image 'CT' *lies on* grid 'g'".
pub fn depends(what: &str, id: &str, relation: &str, other: &str, other_id: &str, cause: &Error) -> Error {
    Error::coded(
        "W913",
        format!(
            "{what} {} {relation} {other} {}, which the projection leaves out: {}",
            repr_str(id),
            repr_str(other_id),
            cause.message()
        ),
    )
}
