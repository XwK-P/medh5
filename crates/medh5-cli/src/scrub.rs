//! `scrub` --- find identifiers in the container and attest to it (§11.4).

use std::path::Path;

use clap::ArgMatches;
use serde_json::Value;

use medh5::collection::{open_any, AnyFile};
use medh5::curation::scrub::{scrub_each, DateShift, ScrubOptions};

use crate::common::*;

pub fn scrub(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let apply_changes = flag(m, "apply_changes");
    let mut options = ScrubOptions::default()
        .with_profile(req_str(m, "profile"))
        .with_salt(req_str(m, "salt"))
        .with_pseudonymise_ids(flag(m, "pseudonymise_ids"))
        .with_skip_refused(flag(m, "skip_refused"));
    if let Some(days) = get_i64(m, "date_shift_days") {
        options = options.with_date_shift(DateShift::Days(days));
    }
    if let Some((min, max)) = m.get_one::<(i64, i64)>("date_shift_range") {
        options = options.with_date_shift(DateShift::Range(*min, *max));
    }
    if let Some(by) = get_str(m, "performed_by") {
        options = options.with_performed_by(by);
    }
    if flag(m, "withhold_shift") {
        options = options.with_withhold_shift(true);
    }
    if let Some(out) = get_str(m, "out") {
        options = options.with_out(out);
    }
    if let Some(dir) = get_str(m, "out_dir") {
        options = options.with_out_dir(dir);
    }
    if let Some(tool) = get_str(m, "attest_text") {
        options = options.with_attest_text(tool);
    }
    let paths = get_paths(m, "paths");
    // Refused before any file is read: the engine says why, in the same
    // words the Python API hears.
    let results = match scrub_each(&paths, apply_changes, &options) {
        Ok(results) => results,
        Err(e) => return Ok(ctx.usage(e.to_string())),
    };
    let as_json = flag(m, "json");
    let mut reports = Vec::new();
    for (path, result) in paths.iter().zip(results) {
        // A refused file --- a clinical sample whose text is unattested --- must
        // not lose the reports of the files already scrubbed: their `uid_map`
        // is the only key to the pseudonyms written.
        let report = match result {
            Ok(r) => r,
            Err(e) => {
                // A collection is scrubbed member by member: its refusal is
                // its own, not the one-sample-file refusal other commands give.
                reports.push(ctx.file_failed(path, &e));
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
        if apply_changes && options.withhold_shift.is_none() && records_its_shift(&report.path) {
            ctx.eprint(format!(
                "medh5: note: {path} records the amount of its date shift, which restores every date from one \
                 known date; medh5 3.0 withholds it by default --- pass --withhold-shift"
            ));
        }
    }
    // `ok` is "nothing found" for a scan and "nothing actionable left, by
    // re-scanning what was written" for an apply.
    let ok = reports.iter().all(|r| r["ok"] == Value::Bool(true));
    ctx.emit(&Value::Array(reports), as_json);
    Ok(if ok { EXIT_OK } else { EXIT_ERROR })
}

/// Whether the sample at `path`, or a member of the collection, states the
/// amount of its date shift.
fn records_its_shift(path: &str) -> bool {
    let records = |s: &medh5::sample::Sample| -> medh5::Result<bool> {
        Ok(s.document()?.deidentification.as_ref().and_then(|r| r.shift_days()).is_some())
    };
    match open_any(Path::new(path), None) {
        Ok(AnyFile::Sample(sample)) => records(&sample).unwrap_or(false),
        Ok(AnyFile::Collection(collection)) => collection
            .keys()
            .map(|keys| keys.iter().any(|k| collection.get(k).and_then(|s| records(&s)).unwrap_or(false)))
            .unwrap_or(false),
        Err(_) => false,
    }
}
