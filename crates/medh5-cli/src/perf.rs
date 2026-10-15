//! `recompress` and `bench` --- the storage and performance commands (§14).

use std::path::Path;

use clap::ArgMatches;
use serde_json::{json, Value};

use medh5::storage::recompress::recompress;

use crate::common::*;
use crate::workers::{run_files, FileOutcome};

pub fn dispatch(name: &str, m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    match name {
        "recompress" => recompress_cmd(m, ctx),
        _ => crate::bench::bench(m, ctx),
    }
}

fn recompress_cmd(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let paths = get_paths(m, "paths");
    if get_str(m, "out").is_some() && paths.len() > 1 {
        return Ok(ctx.usage("--out takes a single input file"));
    }
    let outcomes = run_files("recompress", m, ctx, &paths)?;
    // A file recompressed carries its table row; one that failed, none.
    let done: Vec<&Value> = outcomes.iter().map(|o| &o.data).filter(|d| !d.is_null()).collect();
    let all_ok = done.len() == paths.len() && outcomes.iter().all(|o| o.ok);
    if flag(m, "json") {
        ctx.emit(&Value::Array(outcomes.iter().map(|o| o.entry.clone()).collect()), true);
        return Ok(if all_ok { EXIT_OK } else { EXIT_ERROR });
    }
    if done.is_empty() {
        return Ok(EXIT_ERROR);
    }
    let rows: Vec<Vec<String>> = done.iter().map(|d| strings(&d["row"])).collect();
    ctx.print(table(&rows, &["path", "profile", "sets", "before", "after", "ratio", "content_id", "verify"]));
    for d in &done {
        for line in strings(&d["lines"]) {
            ctx.print(line);
        }
    }
    ctx.print(
        "\ndigests cover decompressed content (§13.1), so `content_id` is unchanged by re-encoding; a cache keyed on \
         it stays valid. The output is verified against the digests it carries, so that claim is checked rather than \
         assumed.",
    );
    Ok(if all_ok { EXIT_OK } else { EXIT_ERROR })
}

/// One file of `recompress`: its `--json` entry, and as `data` its row of
/// the table and the lines that follow the table about it.
pub(crate) fn recompress_one(path: &str, m: &ArgMatches) -> medh5::Result<FileOutcome> {
    let out = get_str(m, "out");
    let r = match recompress(Path::new(path), req_str(m, "profile"), out.map(Path::new), flag(m, "rechunk")) {
        Ok(r) => r,
        Err(e) => return Ok(FileOutcome::failed(path, &e)),
    };
    let row = vec![
        r.path.clone(),
        r.profile.clone(),
        r.datasets.to_string(),
        human_bytes(r.bytes_before as f64),
        human_bytes(r.bytes_after as f64),
        format!("{:.2}x", r.ratio()),
        if r.content_id_preserved { "yes".into() } else { "CHANGED".into() },
        if r.verified {
            "ok".into()
        } else {
            format!("FAILED ({})", r.mismatched.len() + r.unattested.len() + r.unverified.len())
        },
    ];
    let mut lines = Vec::new();
    for name in &r.mismatched {
        lines.push(format!("  MISMATCH  {name}"));
    }
    for name in &r.unattested {
        lines.push(format!("  UNSIGNED  {name} (no line of content_id covers it at this path)"));
    }
    // What the check of the output could not read, and why: the output
    // has replaced the input by then, so it is named, not raised.
    for name in &r.unverified {
        lines.push(format!("  UNCHECKED {name}"));
    }
    for name in &r.kept {
        lines.push(format!("  KEPT      {name} (copied as stored: a rebuild would lose its type)"));
    }
    let mut outcome = FileOutcome::new();
    outcome.entry = r.to_json();
    outcome.data = json!({"row": row, "lines": lines});
    outcome.ok = r.ok();
    Ok(outcome)
}

/// A JSON array of strings as strings.
fn strings(value: &Value) -> Vec<String> {
    value.as_array().into_iter().flatten().map(|v| v.as_str().unwrap_or_default().to_string()).collect()
}
