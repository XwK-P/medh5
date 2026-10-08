//! §1.3, §2 --- the container, the document, bulk storage and profiles (E0xx).

use std::collections::BTreeSet;

use super::{child, loc, str_attr, strs_attr, Context};
use crate::annotations::header::VOXEL_KINDS;
use crate::document::{validate_document_version, SampleDocument};
use crate::h5::attrs;
use crate::h5::data;
use crate::h5::ops;
use crate::ids::{is_valid_id, matches_pattern};
use crate::json::{repr_list, repr_str};
use crate::sample::reader::PROFILES;
use crate::storage::codecs::is_bulk;
use crate::validate::{Diagnostic, SAMPLES_GROUP};
use crate::Result;

/// The major version this validator implements.
pub const SUPPORTED_MAJOR: &str = "1";

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
            } else if ctx.projection {
                out.push(ctx.err(
                    "W913",
                    "/",
                    format!(
                        "declares MEDH5 {version}; this validator implements up to {}: only the supported \
                         projection was validated, which is not conformance to {version}, and the file must not be \
                         amended by this engine",
                        crate::FORMAT_VERSION
                    ),
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
            out.push(ctx.unknown("E007", "/", format!("unknown profile(s) {}", repr_list(&unknown))));
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
    let outer = str_attr(&root, "medh5_version")?;
    match &outer {
        None => out.push(ctx.err("E001", "/", "collection root has no `medh5_version` attribute")),
        Some(v) if v.split('.').next().unwrap_or("") != SUPPORTED_MAJOR => {
            out.push(ctx.err("E002", "/", format!("declares MEDH5 {v}; this validator implements {SUPPORTED_MAJOR}.x")))
        }
        Some(v) if ctx.projection => out.push(ctx.err(
            "W913",
            "/",
            format!(
                "declares MEDH5 {v}; this validator implements up to {}: only the supported projection was validated",
                crate::FORMAT_VERSION
            ),
        )),
        _ => {}
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
        let Some(member) = child(&node, &key) else { continue };
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
    let text = data::read_scalar_string(&meta)?;
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
    for message in tolerated {
        let (location, detail) = split(&message);
        out.push(ctx.unknown("E005", format!("/meta#{location}"), detail));
    }
    match SampleDocument::from_json(&parsed) {
        Ok(doc) => ctx.document = Some(doc),
        Err(e) if ctx.projection => out.push(ctx.unknown(
            e.code().unwrap_or("E005"),
            "/meta",
            format!("{}; the rules that read the document were not run", e.message()),
        )),
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
                out.push(ctx.err(
                    "E009",
                    "/meta#timepoints",
                    "profile `longitudinal` requires at least two declared timepoints",
                ));
            }
        }
    }
    Ok(out)
}
