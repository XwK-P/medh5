//! `recompress` and `bench` --- the storage and performance commands (§14).

use std::path::Path;

use clap::ArgMatches;
use serde_json::Value;

use medh5::storage::recompress::recompress;

use crate::common::*;

pub fn dispatch(name: &str, m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    match name {
        "recompress" => recompress_cmd(m, ctx),
        _ => crate::bench::bench(m, ctx),
    }
}

fn recompress_cmd(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let paths = get_paths(m, "paths");
    let out = get_str(m, "out");
    if out.is_some() && paths.len() > 1 {
        return Ok(ctx.fail("--out takes a single input file"));
    }
    let profile = req_str(m, "profile");
    let mut results = Vec::new();
    for path in &paths {
        match recompress(Path::new(path), profile, out.map(Path::new), flag(m, "rechunk")) {
            Ok(r) => results.push(r),
            Err(e) if e.is_medh5() => return Ok(ctx.fail(e.to_string())),
            Err(e) => return Err(e),
        }
    }
    let all_ok = results.iter().all(|r| r.ok());
    if flag(m, "json") {
        ctx.emit(&Value::Array(results.iter().map(|r| r.to_json()).collect()), true);
        return Ok(if all_ok { EXIT_OK } else { EXIT_ERROR });
    }
    let rows: Vec<Vec<String>> = results
        .iter()
        .map(|r| {
            vec![
                r.path.clone(),
                r.profile.clone(),
                r.datasets.to_string(),
                human_bytes(r.bytes_before as f64),
                human_bytes(r.bytes_after as f64),
                format!("{:.2}x", r.ratio()),
                if r.content_id_preserved { "yes".into() } else { "CHANGED".into() },
                if r.verified { "ok".into() } else { format!("FAILED ({})", r.mismatched.len() + r.unattested.len()) },
            ]
        })
        .collect();
    ctx.print(table(&rows, &["path", "profile", "sets", "before", "after", "ratio", "content_id", "verify"]));
    for result in &results {
        for name in &result.mismatched {
            ctx.print(format!("  MISMATCH  {name}"));
        }
        for name in &result.unattested {
            ctx.print(format!("  UNSIGNED  {name} (no digest, inside an attested object)"));
        }
    }
    ctx.print(
        "\ndigests cover decompressed content (§13.1), so `content_id` is unchanged by re-encoding; a cache keyed on \
         it stays valid. The output is verified against the digests it carries, so that claim is checked rather than \
         assumed.",
    );
    Ok(if all_ok { EXIT_OK } else { EXIT_ERROR })
}
