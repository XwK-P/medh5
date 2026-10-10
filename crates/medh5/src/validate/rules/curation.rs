//! §11--§12 --- provenance, quality, de-identification and splits (E6xx).

use std::collections::{BTreeMap, BTreeSet};

use super::{loc, str_attr, Context};
use crate::document::SampleDocument;
use crate::h5::attrs;
use crate::json::repr_str;
use crate::validate::Diagnostic;
use crate::Result;

/// Semantic: the provenance graph and quality records against the objects
/// that name them (E601, E602, E605), de-identification (W903), and
/// `quality` on every annotation under `curation` (E009).
///
/// An unknown agent or activity type (E603) and a malformed timestamp (E604)
/// are the document's: an agent or activity breaking one does not parse, and
/// is reported where `/meta` is.
pub fn check_curation(ctx: &mut Context) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let Some(doc) = ctx.document.clone() else { return Ok(out) };
    let prov = &doc.provenance;
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

fn check_links(
    ctx: &Context,
    location: &str,
    a: &hdf5::Location,
    doc: &SampleDocument,
    quality_attr: &str,
) -> Result<Vec<Diagnostic>> {
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

/// Semantic: one split set claimed against one manifest (§12.3, W906).
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
                format!(
                    "split set {} is claimed against {} different manifests in one file",
                    repr_str(&set_id),
                    hashes.len()
                ),
            ));
        }
    }
    Ok(out)
}
