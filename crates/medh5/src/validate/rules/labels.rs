//! §5 --- the label set and its ontology bindings (E3xx).

use std::collections::BTreeSet;

use super::{int_set, loc, Context};
use crate::json::{repr_int_list, repr_list};
use crate::validate::Diagnostic;
use crate::Result;

/// Semantic: a label set where a declared profile needs one (E301).
///
/// The label set's own rules (§5.1--§5.4: E302--E306) are the document's: a
/// label set breaking one does not parse, and is reported where `/meta` is.
pub fn check_label_set(ctx: &mut Context) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let Some(doc) = &ctx.document else { return Ok(out) };
    let mut needs: Vec<&str> =
        ["seg", "det", "cls"].into_iter().filter(|p| ctx.profiles.iter().any(|x| x == p)).collect();
    needs.sort_unstable();
    if doc.label_set.is_none() && !needs.is_empty() {
        out.push(ctx.err("E301", "/meta#label_set", format!("profile(s) {} require a label set", repr_list(&needs))));
    }
    Ok(out)
}

/// Semantic: every class an annotation uses bound to an ontology code
/// (§5.2, W912).
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
