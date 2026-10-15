//! `conformance` --- build the corpus, run it here, publish it, score others.

use std::path::Path;

use clap::ArgMatches;
use serde_json::{json, Value};

use medh5::conformance::{
    build_corpus, case_by_name, cases, check_checksums, publish, run_corpus, score, summarize, CaseResult,
};
use medh5::json::repr_str;
use medh5::pyval::py_path;
use medh5::Error;

use crate::common::*;

pub fn dispatch(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let names = m.subcommand().and_then(|(_, sub)| sub.try_get_many::<String>("names").ok().flatten());
    let names: Option<Vec<String>> = names.map(|n| n.cloned().collect());
    let chosen = match chosen(names.as_deref()) {
        Ok(n) => n,
        Err(why) => return Ok(ctx.usage(why)),
    };
    let names = names.as_deref();
    match m.subcommand() {
        Some(("list", sub)) => list(sub, ctx),
        Some(("build", sub)) if flag(sub, "robustness") => {
            let dir = outdir(sub)?;
            let written = medh5::conformance::robustness::write(dir)?;
            ctx.print(format!("wrote {} robustness fixtures to {}", written.len(), py_path(dir)));
            Ok(EXIT_OK)
        }
        Some(("build", sub)) => {
            let manifest = build_corpus(outdir(sub)?, names)?;
            ctx.print(format!("wrote {chosen} cases and {}", py_path(&manifest)));
            Ok(EXIT_OK)
        }
        Some(("run", sub)) => {
            let results = run_corpus(outdir(sub)?, names)?;
            Ok(report(&results, None, flag(sub, "json"), ctx))
        }
        Some(("publish", sub)) => {
            let root = py_path(&publish(outdir(sub)?, names)?);
            ctx.print(format!("wrote the suite to {root}: {chosen} cases, see {root}/README.md"));
            Ok(EXIT_OK)
        }
        Some(("score", sub)) => score_cmd(sub, ctx),
        _ => Ok(ctx.usage("usage: medh5 conformance {list|build|run|publish|score}")),
    }
}

/// `OUTDIR`, made if it is not there; a file in the way is named.
fn outdir(m: &ArgMatches) -> medh5::Result<&Path> {
    let dir = req_str(m, "outdir");
    std::fs::create_dir_all(dir).map_err(|e| medh5::error::os_error(&e, dir))?;
    Ok(Path::new(dir))
}

/// How many cases `--case` selects --- every case when it names none --- or
/// why it selects nothing: a name no case has would build, run or publish an
/// empty corpus and report it a success.
fn chosen(names: Option<&[String]>) -> Result<usize, String> {
    if let Some(unknown) = names.into_iter().flatten().find(|n| case_by_name(n).is_err()) {
        return Err(format!("no conformance case {} (`medh5 conformance list` names them)", repr_str(unknown)));
    }
    Ok(cases().iter().filter(|c| names.is_none_or(|n| n.contains(&c.name))).count())
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
        let submitted = medh5::json::loads(&text).map_err(|e| Error::Value(format!("not valid JSON: {e}")))?;
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
    let as_json = flag(m, "json");
    if !stale.is_empty() {
        // The scores mean nothing if the files scored are not the files
        // published, so say so before reporting any of them --- on stderr
        // when stdout is the document, which names them under `changed`.
        let mut warning =
            vec![format!("WARNING: {} published file(s) differ from {}", stale.len(), req_str(m, "suite"))];
        warning.extend(stale.iter().map(|name| format!("  changed: {name}")));
        for line in warning {
            if as_json {
                ctx.eprint(line);
            } else {
                ctx.print(line);
            }
        }
    }
    Ok(report(&results, Some(&stale), as_json, ctx))
}

/// Print the outcome; the exit code.  `changed`, for a score, names the
/// published files whose bytes are no longer the published ones: a score
/// over them is not a score, so it fails whatever the cases say.
fn report(results: &[CaseResult], changed: Option<&[String]>, as_json: bool, ctx: &mut Ctx) -> i32 {
    let failures: Vec<&CaseResult> = results.iter().filter(|r| !r.ok()).collect();
    let drifted = changed.is_some_and(|c| !c.is_empty());
    if as_json {
        let mut summary = summarize(results);
        if let Some(changed) = changed {
            summary["changed"] = json!(changed);
            summary["ok"] = json!(failures.is_empty() && !drifted);
        }
        ctx.emit(&summary, true);
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
    if failures.is_empty() && !drifted {
        EXIT_OK
    } else {
        EXIT_ERROR
    }
}
