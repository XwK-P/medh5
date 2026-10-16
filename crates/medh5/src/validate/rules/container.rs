//! §1.3, §2 --- the container, the document, bulk storage and profiles (E0xx).

use std::collections::BTreeSet;
use std::path::Path;

use super::{child, children, describe, loc, str_attr, strs_attr, unresolved, Context};
use crate::annotations::header::VOXEL_KINDS;
use crate::document::{projected, validate_document_version, SampleDocument};
use crate::h5::attrs::{self, AttrValue};
use crate::h5::data::{self, Kind};
use crate::h5::ops::{self, Node};
use crate::ids::{is_valid_id, matches_pattern};
use crate::json::{repr_list, repr_str};
use crate::sample::reader::PROFILES;
use crate::sample::require_major;
use crate::storage::codecs::is_bulk;
use crate::validate::{Diagnostic, SAMPLES_GROUP};
use crate::version::{support, Support};
use crate::{Error, Result};

/// The major version this validator implements.
pub const SUPPORTED_MAJOR: &str = "1";

/// Structural: the root's version, kind and profiles, the required groups
/// and `/meta` and what kind of object each is, every member of the four
/// groups an object, and identifier syntax (E001--E004, E006--E008, W913).
///
/// The version is read as the readers read it ([`check_version`]).  Another
/// major is E002 and ends the pass: no rule of this major speaks for its
/// content.
pub fn check_container(ctx: &mut Context) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let root = ctx.root.clone();
    out.extend(check_version(ctx, &root, false)?);
    if out.iter().any(|d| d.code == "E002") {
        return Ok(out);
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
            out.push(ctx.unknown("E007", "/", format!("unknown profile(s) {}", repr_list(&unknown))));
        }
    }
    // What each name is, not only that it is there (§2.3): a dataset, or a
    // soft link to nothing, where a group belongs left every rule reading the
    // group with nothing to read, and the file passed every level.
    for (name, required) in [("grids", true), ("images", true), ("annotations", false), ("transforms", false)] {
        if ops::is_group(&root, name) {
            continue;
        }
        if !ops::exists(&root, name) {
            if required {
                out.push(ctx.err("E008", format!("/{name}"), format!("required group `{name}` is absent")));
            }
            continue;
        }
        out.push(ctx.err(
            "E008",
            format!("/{name}"),
            format!("`{name}` is {}; the layout makes it a group (§2.3)", describe(&root, name)),
        ));
    }
    if !ops::exists(&root, "meta") {
        out.push(ctx.err("E004", "/meta", "required dataset `meta` is absent"));
    } else if !ops::is_dataset(&root, "meta") {
        out.push(ctx.err(
            "E004",
            "/meta",
            format!("`meta` is {}; §2.4 makes it a dataset holding one UTF-8 JSON string", describe(&root, "meta")),
        ));
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
            // No rule can read it, so it is reported here, once, rather than
            // dropped by every rule that walks the group.
            if let Some(what) = unresolved(&node, &name) {
                out.push(ctx.err(
                    "E001",
                    format!("/{group}/{name}"),
                    format!("is {what}, so nothing it should hold can be read or checked"),
                ));
            }
        }
    }
    Ok(out)
}

/// What is wrong with `root`'s `medh5_version` (§2.1): a sample root's, or
/// with `collection` a collection's own.
///
/// Absent, empty or not a string is E001.  Another major is E002, decided as
/// the readers decide it ([`require_major`], the check `open`, `amend` and
/// `open_collection` make): the validator neither passes a file they refuse
/// for its version nor refuses one they read.  A higher minor is W913, as
/// [`support`] reads it.
fn check_version(ctx: &Context, root: &hdf5::Group, collection: bool) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let version = match attrs::read(root, "medh5_version")? {
        None if collection => {
            out.push(ctx.err("E001", "/", "collection root has no `medh5_version` attribute"));
            return Ok(out);
        }
        None => {
            out.push(ctx.err("E001", "/", "root has no `medh5_version` attribute"));
            return Ok(out);
        }
        Some(AttrValue::Str(text)) => text,
        // The readers read its text; §2.1 makes it a string.
        Some(other) => {
            let text = attrs::stringify_value(&other);
            out.push(ctx.err(
                "E001",
                "/",
                format!("`medh5_version` is {}, not a string; §2.1 makes it `MAJOR.MINOR`", repr_str(&text)),
            ));
            text
        }
    };
    if version.trim().is_empty() {
        let message = if collection { "the collection's `medh5_version` is empty" } else { "`medh5_version` is empty" };
        out.push(ctx.err("E001", "/", message));
        return Ok(out);
    }
    if let Err(Error::Version(_)) = require_major(root, Path::new(&ctx.path)) {
        out.push(ctx.err(
            "E002",
            "/",
            format!("declares MEDH5 {version}; this validator implements {SUPPORTED_MAJOR}.x"),
        ));
        return Ok(out);
    }
    if support(&version) == Support::Projection {
        let rest = if collection {
            String::new()
        } else {
            format!(", which is not conformance to {version}, and the file must not be amended by this engine")
        };
        out.push(ctx.err(
            "W913",
            "/",
            format!(
                "declares MEDH5 {version}; this validator implements up to {}: only the supported projection was \
                 validated{rest}",
                crate::FORMAT_VERSION
            ),
        ));
    }
    Ok(out)
}

/// Rules that apply to a `collection` root itself (§2.2).
pub fn check_collection(ctx: &mut Context) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let root = ctx.root.clone();
    let outer = str_attr(&root, "medh5_version")?;
    out.extend(check_version(ctx, &root, true)?);
    if out.iter().any(|d| d.code == "E002") {
        return Ok(out);
    }
    let Some(node) = ops::child_group(&root, SAMPLES_GROUP) else {
        out.push(ctx.err(
            "E008",
            format!("/{SAMPLES_GROUP}"),
            format!("a `collection` requires a `{SAMPLES_GROUP}` group"),
        ));
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
        let Some(member) = child(&node, &key) else {
            if let Some(what) = unresolved(&node, &key) {
                out.push(ctx.err(
                    "E001",
                    location,
                    format!("is {what}, so the sample root it names can be neither read nor validated"),
                ));
            }
            continue;
        };
        // 1.1 §8: the outer root declares a version no lower than any
        // member's, so a reader that stops at the outer version sees every
        // member it can read.
        if let (Some(outer), Some(inner)) = (outer.as_deref(), str_attr(loc(&member), "medh5_version")?) {
            if !crate::version::at_least(outer, &inner) && crate::version::parse(&inner).is_some() {
                out.push(ctx.err(
                    "E011",
                    location.clone(),
                    format!(
                        "member {} is MEDH5 {inner}, but the collection declares {outer}; a collection declares \
                         the highest version of the samples it holds",
                        repr_str(&key)
                    ),
                ));
            }
        }
        if !attrs::has(loc(&member), "medh5_profiles") {
            out.push(ctx.err(
                "E007",
                location.clone(),
                "a sample root in a collection carries its own `medh5_profiles`",
            ));
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
    // One string (§2.4), checked before anything is read: a vector `meta` was
    // read whole and cut to its first element.
    if !meta.is_scalar() {
        out.push(ctx.err(
            "E004",
            "/meta",
            format!(
                "`meta` is a dataset of shape {}; §2.4 makes it one scalar UTF-8 JSON string",
                crate::json::repr_int_tuple(&meta.shape())
            ),
        ));
        return Ok(out);
    }
    // Text, and only UTF-8 text (§2.4): read as stored, where a lossy read
    // put U+FFFD in place of bytes that are not, and the document parsed.
    let Kind::Strings = data::kind(&meta)? else {
        out.push(ctx.err(
            "E004",
            "/meta",
            format!("`meta` is a dataset of {}; §2.4 makes it one UTF-8 JSON string", super::dtype_name(&meta)),
        ));
        return Ok(out);
    };
    let stored = data::read_stored_strings(&meta)?.into_iter().next().unwrap_or_default();
    let text = match String::from_utf8(stored) {
        Ok(text) => text,
        Err(e) => {
            out.push(ctx.err(
                "E004",
                "/meta",
                format!("`meta` is not valid UTF-8 (byte {} is not)", e.utf8_error().valid_up_to()),
            ));
            return Ok(out);
        }
    };
    let parsed: serde_json::Value = match crate::json::loads_lenient(&text) {
        Ok((v, None)) => v,
        // What 1.x's `json.dumps` wrote for a NaN: not JSON, so E004, but the
        // rest of the document is still checked --- a reader reads it as null.
        Ok((v, Some(found))) => {
            out.push(ctx.err(
                "E004",
                "/meta",
                format!("`meta` is not valid JSON: {found} (JSON has no NaN or infinity; a reader takes it as null)"),
            ));
            v
        }
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
    let split = |message: &str| match message.split_once(": ") {
        Some((l, d)) if !d.is_empty() => (l.to_string(), d.to_string()),
        _ => (message.to_string(), message.to_string()),
    };
    let version = ctx.version.clone().unwrap_or_else(|| crate::version::BASE_VERSION.to_string());
    let (errors, tolerated) = validate_document_version(&parsed, &version);
    for message in errors {
        schema_failed = true;
        let (location, detail) = split(&message);
        out.push(ctx.err("E005", format!("/meta#{location}"), detail));
    }
    // A projection's document is read as this version reads it: what only a
    // later minor defines, each reported here as W913, is set aside.  What it
    // cannot set aside leaves the document unread, and its rules not run.
    let projection = (!tolerated.is_empty()).then(|| projected(&parsed).ok());
    let readable = match &projection {
        Some(set_aside) => set_aside.as_ref(),
        None => Some(&parsed),
    };
    for message in tolerated {
        let (location, detail) = split(&message);
        out.push(ctx.unknown("E005", format!("/meta#{location}"), detail));
    }
    match readable.map(SampleDocument::from_json) {
        Some(Ok(doc)) => ctx.document = Some(doc),
        Some(Err(e)) => {
            let code = e.code().unwrap_or("E005").to_string();
            if !(schema_failed && code == "E005") {
                out.push(ctx.err(&code, "/meta", e.message()));
            }
        }
        None => out.push(ctx.err(
            "W913",
            "/meta",
            "a value only a later minor defines is one this version cannot read past, so the rules that read the \
             document were not run",
        )),
    }
    Ok(out)
}

/// Structural: bulk datasets stored chunked and compressed (W902).
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

/// Semantic: what each profile the sample is held to requires (§1.3, E009).
pub fn check_profiles(ctx: &mut Context) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let mut kinds = BTreeSet::new();
    let mut tasks = BTreeSet::new();
    for (_, g) in ctx.children("annotations")? {
        let kind = str_attr(loc(&g), "kind")?;
        // A curve is no detection, whatever task it states (1.2 §3).
        if let Some(t) = str_attr(loc(&g), "task")?.filter(|_| kind.as_deref() != Some("polylines")) {
            tasks.insert(t);
        }
        if let Some(k) = kind {
            kinds.insert(k);
        }
    }
    let declared: BTreeSet<&str> = ctx.profiles.iter().map(String::as_str).collect();
    if declared.contains("seg") && !kinds.iter().any(|k| VOXEL_KINDS.contains(&k.as_str()) && k != "mask") {
        out.push(ctx.err("E009", "/", "profile `seg` requires a voxel annotation, and none is present"));
    }
    if declared.contains("det") && !tasks.contains("detection") {
        out.push(ctx.err(
            "E009",
            "/",
            "profile `det` requires an annotation of task='detection', and none declares it",
        ));
    }
    if declared.contains("cls") && !kinds.contains("classification") {
        out.push(ctx.err("E009", "/", "profile `cls` requires a classification annotation, and none is present"));
    }
    let count = |g: &str| -> Result<usize> {
        Ok(match ops::child_group(&ctx.root, g) {
            None => 0,
            Some(node) => ops::members(&node)?.len(),
        })
    };
    if declared.contains("reg") {
        if count("transforms")? == 0 {
            out.push(ctx.err("E009", "/", "profile `reg` requires a transform, and none is present"));
        } else if !relates_grids(&ctx.root)? {
            out.push(ctx.err(
                "E009",
                "/transforms",
                "profile `reg` requires a transform with endpoints in the sample, and none has a `from_frame` and a \
                 `to_frame` that are each the `frame_uid` of one of its grids",
            ));
        }
    }
    if declared.contains("training") && count("index")? == 0 {
        out.push(ctx.err("E009", "/", "profile `training` requires a sampling index, and none is present"));
    }
    if declared.contains("multiscale") {
        for (name, image) in ctx.children("images")? {
            if !matches!(image, Node::Group(_)) {
                out.push(ctx.err(
                    "E009",
                    format!("/images/{name}"),
                    "profile `multiscale` lays out every image as a §4.3 pyramid, and this image is single-scale",
                ));
            }
        }
    }
    if let Some(doc) = &ctx.document {
        if declared.contains("curation") && !records_activity(doc) {
            out.push(ctx.err(
                "E009",
                "/meta#provenance",
                "profile `curation` requires a provenance activity, and the provenance graph (§11.1) records none",
            ));
        }
        if declared.contains("longitudinal") && doc.timepoints.len() < 2 {
            out.push(ctx.err(
                "E009",
                "/meta#timepoints",
                "profile `longitudinal` requires at least two declared timepoints",
            ));
        }
    }
    Ok(out)
}

/// Whether the provenance graph records an activity: what `curation` needs of
/// it (§1.3), and what the writer infers the profile from.
pub(crate) fn records_activity(document: &SampleDocument) -> bool {
    document.provenance.n_activities() > 0
}

/// Whether some transform maps between frames of the sample's own grids, its
/// `from_frame` and `to_frame` each a grid's `frame_uid`: what `reg` needs
/// (§1.3), and what the writer infers the profile from.
pub(crate) fn relates_grids(root: &hdf5::Group) -> Result<bool> {
    let mut frames = BTreeSet::new();
    if let Some(grids) = ops::child_group(root, "grids") {
        for (_, grid) in children(&grids)? {
            frames.extend(str_attr(loc(&grid), "frame_uid")?.filter(|f| !f.is_empty()));
        }
    }
    let Some(transforms) = ops::child_group(root, "transforms") else { return Ok(false) };
    for (_, t) in children(&transforms)? {
        let (from, to) = (str_attr(loc(&t), "from_frame")?, str_attr(loc(&t), "to_frame")?);
        if from.is_some_and(|f| frames.contains(&f)) && to.is_some_and(|f| frames.contains(&f)) {
            return Ok(true);
        }
    }
    Ok(false)
}
