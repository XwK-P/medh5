//! §5 --- the label set and its ontology bindings (E3xx).

use std::collections::BTreeSet;

use super::{int_set, loc, Context};
use crate::json::{repr_int_list, repr_list, repr_str};
use crate::labels::{BACKGROUND_ID, IGNORE_ID};
use crate::validate::Diagnostic;
use crate::Result;

pub fn check_label_set(ctx: &mut Context) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let Some(doc) = &ctx.document else { return Ok(out) };
    let mut needs: Vec<&str> =
        ["seg", "det", "cls"].into_iter().filter(|p| ctx.profiles.iter().any(|x| x == p)).collect();
    needs.sort_unstable();
    let Some(ls) = &doc.label_set else {
        if !needs.is_empty() {
            out.push(ctx.err(
                "E301",
                "/meta#label_set",
                format!("profile(s) {} require a label set", repr_list(&needs)),
            ));
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
