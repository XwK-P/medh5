//! §13 --- digests, `content_id` and index currency (E7xx).

use super::{str_attr, Context};
use crate::digest::parse_digest;
use crate::integrity::{stale_index_entries, verify_root};
use crate::json::repr_str;
use crate::validate::Diagnostic;
use crate::Result;

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
    if result.content_id_ok() == Some(false) && ctx.projection {
        // A later minor may cover root attributes this engine does not know:
        // the root's integrity is unsupported here, not refuted (1.1 §2.2).
        out.push(ctx.err(
            "W913",
            "/",
            format!(
                "`content_id` does not recompute under the rules of MEDH5 {}: the root's integrity is unsupported in \
                 this projection, not verified (per-dataset digests were checked)",
                crate::FORMAT_VERSION
            ),
        ));
    } else if result.content_id_ok() == Some(false) {
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
