//! `scrub` --- find identifiers in the container and attest to it (§11.4).

use std::path::Path;

use clap::ArgMatches;
use serde_json::Value;

use medh5::curation::scrub::{apply, scan, ApplyOptions};

use crate::common::*;

pub fn scrub(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let apply_changes = flag(m, "apply_changes");
    if flag(m, "pseudonymise_ids") && !apply_changes {
        return Ok(ctx.fail("--pseudonymise-ids changes the file, so it needs --apply"));
    }
    let options = ApplyOptions {
        profile: req_str(m, "profile").to_string(),
        salt: req_str(m, "salt").to_string(),
        date_shift_days: get_i64(m, "date_shift_days"),
        performed_by: get_str(m, "performed_by").map(str::to_string),
        pseudonymise_ids: flag(m, "pseudonymise_ids"),
    };
    let as_json = flag(m, "json");
    let mut reports = Vec::new();
    for path in get_paths(m, "paths") {
        let result =
            if apply_changes { apply(Path::new(&path), &options) } else { scan(Path::new(&path), &options.profile) };
        let report = match result {
            Ok(r) => r,
            Err(e) if e.is_medh5() => return Ok(ctx.fail(e.to_string())),
            Err(e) => return Err(e),
        };
        reports.push(report.to_json());
        if !as_json {
            ctx.print(report.format());
            let actionable = report.actionable().len();
            if !apply_changes && actionable > 0 {
                ctx.print(format!("  {actionable} finding(s) can be acted on: re-run with --apply"));
            }
        }
    }
    // `ok` is "nothing found" for a scan and "nothing actionable left, by
    // re-scanning what was written" for an apply.
    let ok = reports.iter().all(|r| r["ok"] == Value::Bool(true));
    ctx.emit(&Value::Array(reports), as_json);
    Ok(if ok { EXIT_OK } else { EXIT_ERROR })
}
