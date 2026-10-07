//! `conformance` --- build the corpus, run it here, publish it, score others.

use std::path::Path;

use clap::ArgMatches;
use serde_json::Value;

use medh5::conformance::{build_corpus, cases, check_checksums, publish, run_corpus, score, summarize, CaseResult};
use medh5::pyval::py_path;
use medh5::Error;

use crate::common::*;

pub fn dispatch(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    match m.subcommand() {
        Some(("list", sub)) => list(sub, ctx),
        Some(("build", sub)) => {
            let names = get_strs(sub, "names");
            let manifest = build_corpus(Path::new(req_str(sub, "outdir")), names.as_deref())?;
            ctx.print(format!("wrote {} cases and {}", cases().len(), py_path(&manifest)));
            Ok(EXIT_OK)
        }
        Some(("run", sub)) => {
            let names = get_strs(sub, "names");
            let results = run_corpus(Path::new(req_str(sub, "outdir")), names.as_deref())?;
            Ok(report(&results, flag(sub, "json"), ctx))
        }
        Some(("publish", sub)) => {
            let names = get_strs(sub, "names");
            let root = py_path(&publish(Path::new(req_str(sub, "outdir")), names.as_deref())?);
            ctx.print(format!("wrote the suite to {root}: {} cases, see {root}/README.md", cases().len()));
            Ok(EXIT_OK)
        }
        Some(("score", sub)) => score_cmd(sub, ctx),
        _ => Ok(ctx.fail("usage: medh5 conformance {list|build|run|publish|score}")),
    }
}

fn list(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    if flag(m, "json") {
        let listing: Vec<Value> = cases().iter().map(|c| c.to_json()).collect();
        ctx.emit(&Value::Array(listing), true);
        return Ok(EXIT_OK);
    }
    let rows: Vec<Vec<String>> = cases()
        .iter()
        .map(|c| {
            let mut codes: Vec<&str> = c.errors.iter().chain(&c.warnings).map(String::as_str).collect();
            codes.sort_unstable();
            vec![
                c.name.clone(),
                c.clause.clone(),
                c.level.clone(),
                if c.valid() { "valid" } else { "invalid" }.to_string(),
                if codes.is_empty() { "-".to_string() } else { codes.join(",") },
            ]
        })
        .collect();
    ctx.print(table(&rows, &["case", "clause", "level", "kind", "expected codes"]));
    Ok(EXIT_OK)
}

fn score_cmd(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let suite = Path::new(req_str(m, "suite"));
    let results_path = Path::new(req_str(m, "results"));
    let scored = (|| -> Result<(Vec<String>, Vec<CaseResult>), Error> {
        let text =
            std::fs::read_to_string(results_path).map_err(|e| medh5::error::os_error(&e, py_path(results_path)))?;
        let submitted = medh5::json::loads(&text).map_err(|e| Error::Value(medh5::json::decode_error(&text, &e)))?;
        let stale = check_checksums(suite)?;
        let entries: Vec<Value> = match submitted {
            Value::Array(items) => items,
            // Iterating a mapping yields its keys; none of them is a result.
            Value::Object(map) => map.keys().map(|k| Value::String(k.clone())).collect(),
            other => vec![other],
        };
        Ok((stale, score(suite, &entries)?))
    })();
    let (stale, results) = match scored {
        Ok(found) => found,
        Err(e) if e.is_medh5() || matches!(e, Error::Io(_) | Error::Value(_)) => return Ok(ctx.fail(e.python_str())),
        Err(e) => return Err(e),
    };
    if !stale.is_empty() {
        // The scores mean nothing if the files scored are not the files
        // published, so say so before reporting any of them.
        ctx.print(format!("WARNING: {} published file(s) differ from {}", stale.len(), req_str(m, "suite")));
        for name in &stale {
            ctx.print(format!("  changed: {name}"));
        }
    }
    Ok(report(&results, flag(m, "json"), ctx))
}

fn report(results: &[CaseResult], as_json: bool, ctx: &mut Ctx) -> i32 {
    let failures: Vec<&CaseResult> = results.iter().filter(|r| !r.ok()).collect();
    if as_json {
        ctx.emit(&summarize(results), true);
    } else {
        for result in &failures {
            ctx.print(format!("FAIL {}", result.case.name));
            if let Some(error) = &result.error {
                ctx.print(format!("     {error}"));
            }
            if !result.missing.is_empty() {
                ctx.print(format!("     not reported: {}", result.missing.join(", ")));
            }
            if !result.unexpected.is_empty() {
                ctx.print(format!("     unexpected:   {}", result.unexpected.join(", ")));
            }
        }
        ctx.print(format!("{}/{} cases pass", results.len() - failures.len(), results.len()));
    }
    if failures.is_empty() {
        EXIT_OK
    } else {
        EXIT_ERROR
    }
}
