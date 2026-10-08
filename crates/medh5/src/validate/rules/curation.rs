//! §11--§12 --- provenance, quality, de-identification and splits (E6xx).

use std::collections::{BTreeMap, BTreeSet};

use super::{loc, str_attr, Context};
use crate::curation::provenance::{is_timestamp, ACTIVITY_TYPES};
use crate::document::SampleDocument;
use crate::h5::attrs;
use crate::json::repr_str;
use crate::validate::Diagnostic;
use crate::Result;

pub fn check_curation(ctx: &mut Context) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let Some(doc) = ctx.document.clone() else { return Ok(out) };
    let prov = &doc.provenance;
    for activity in prov.activities() {
        let location = format!("/meta#provenance/activities/{}", activity.id);
        if !ACTIVITY_TYPES.contains(&activity.r#type.as_str()) {
            out.push(ctx.err(
                "E603",
                location.clone(),
                format!("unknown activity type {}", repr_str(&activity.r#type)),
            ));
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
