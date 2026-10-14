//! `scrub` --- find identifiers in the container and attest to it (§11.4).

use std::path::Path;

use clap::parser::ValueSource;
use clap::ArgMatches;
use serde_json::Value;

use medh5::curation::scrub::{apply, scan, ApplyOptions};

use crate::common::*;

/// The options that only `--apply` acts on, by argument id and flag.
const APPLY_ONLY: [(&str, &str); 3] =
    [("date_shift_days", "--date-shift-days"), ("salt", "--salt"), ("performed_by", "--by")];

pub fn scrub(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let apply_changes = flag(m, "apply_changes");
    // Without `--apply` nothing is written, so these did nothing --- and a
    // scan that reads as a salted, shifted scrub is the wrong thing to exit 0.
    let ignored: Vec<&str> = APPLY_ONLY
        .iter()
        .filter(|(id, _)| !apply_changes && m.value_source(id) == Some(ValueSource::CommandLine))
        .map(|(_, long)| *long)
        .collect();
    if !ignored.is_empty() {
        return Ok(ctx.usage(format!(
            "nothing is written without --apply, so {} would do nothing: add --apply, or leave {} out",
            ignored.join(" and "),
            if ignored.len() == 1 { "it" } else { "them" }
        )));
    }
    // The grammar holds `--pseudonymise-ids` to `--apply`; a salt it cannot
    // require, since `--salt` defaults to none.
    if flag(m, "pseudonymise_ids") && req_str(m, "salt").is_empty() {
        return Ok(ctx.usage(
            "--pseudonymise-ids needs --salt: a sample or subject id is usually a record number, and an unsalted \
             hash of one is reversed by hashing every record number",
        ));
    }
    if get_i64(m, "date_shift_days") == Some(0) {
        return Ok(ctx.usage(
            "--date-shift-days 0 shifts nothing, and a file that records a shift has its dates passed over by every \
             later scan; give a shift of at least one day, or leave it out to remove the dates",
        ));
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
        // A refused file --- a clinical sample is refused by design --- must
        // not lose the reports of the files already scrubbed: their `uid_map`
        // is the only key to the pseudonyms written.
        let report = match result {
            Ok(r) => r,
            Err(e) => {
                reports.push(ctx.file_failed(&path, &sample_file_error(&path, e)));
                continue;
            }
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
