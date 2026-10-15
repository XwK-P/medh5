//! `convert` and `migrate` --- the importers and exporters.
//!
//! Every command writes a conversion report, because the interesting part of
//! an import is not that it succeeded but what it had to decide.  `migrate`
//! (0.x files, Appendix B) is native; the format converters are Python
//! integrations and run in the Python package, through the [`Host`].

use std::path::{Component, Path, PathBuf};

use clap::{ArgAction, ArgMatches};
use serde_json::{Map, Value};

use medh5::convert::legacy::{build_label_set, load_sidecar, migrate_paths, write_sidecar};
use medh5::convert::report::ConversionReport;
use medh5::h5::file::same_file;
use medh5::json::{pretty, repr_str};

use crate::app;
use crate::common::*;

/// What `--help` cannot say for a converter the standalone binary cannot run.
const NO_PYTHON: &str = "the format converters (NIfTI, DICOM, DICOM SEG, RTSTRUCT, nnU-Net) are part of the Python \
                         package: install it (`pip install 'medh5[nifti,dicom]'`) and run `medh5 convert` from it, \
                         or point MEDH5_PYTHON at an interpreter that has it";

/// The parsed arguments of one subcommand, keyed as the 1.x parser named
/// them, for a host that runs the command itself.
pub fn parsed_args(command: &clap::Command, m: &ArgMatches) -> Value {
    let mut out = Map::new();
    for arg in command.get_arguments() {
        let id = arg.get_id().as_str();
        if id == "help" || id == "version" {
            continue;
        }
        let value = match arg.get_action() {
            ArgAction::SetTrue => Value::Bool(m.get_flag(id)),
            ArgAction::Append => match m.get_many::<String>(id) {
                Some(v) => Value::Array(v.map(|s| Value::String(s.clone())).collect()),
                None => Value::Null,
            },
            _ if arg.get_num_args().is_some_and(|n| n.max_values() > 1) => match m.get_many::<String>(id) {
                Some(v) => Value::Array(v.map(|s| Value::String(s.clone())).collect()),
                None => Value::Null,
            },
            _ => {
                if let Ok(Some(v)) = m.try_get_one::<String>(id) {
                    Value::String(v.clone())
                } else if let Ok(Some(v)) = m.try_get_one::<i64>(id) {
                    Value::from(*v)
                } else if let Ok(Some(v)) = m.try_get_one::<f64>(id) {
                    medh5::json::num(*v)
                } else {
                    Value::Null
                }
            }
        };
        out.insert(id.to_string(), value);
    }
    Value::Object(out)
}

pub fn convert(argv: &[String], m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let Some((name, sub)) = m.subcommand() else {
        return Ok(ctx.usage("usage: medh5 convert COMMAND ... (see --help)"));
    };
    let root = app::command();
    let command = root
        .find_subcommand("convert")
        .and_then(|c| c.find_subcommand(name))
        .expect("the grammar defines every convert command");
    let parsed = parsed_args(command, sub);
    let host = ctx.host;
    let (out, err) = ctx.streams();
    match host.convert(argv, name, &parsed, out, err) {
        Some(code) => Ok(code),
        None => Ok(ctx.fail(NO_PYTHON)),
    }
}

/// Refuse a file `migrate` would write that is one it reads, before anything
/// is written: `--report` or `--write-labels` naming an input replaced it
/// with JSON, and a `--report` directory was refused only after the cohort
/// had been migrated.
fn check_writes(m: &ArgMatches, sources: &[&Path]) -> Result<(), String> {
    let report = get_str(m, "report").map(Path::new);
    let labels = get_str(m, "write_labels").map(Path::new);
    if let Some(path) = report {
        let shown = repr_str(&path.to_string_lossy());
        if path.is_dir() {
            return Err(format!("--report {shown} is a directory; name a file"));
        }
        // `migrate` creates its output directory, so a report may go there.
        let lexical = |p: &Path| -> PathBuf { p.components().filter(|c| *c != Component::CurDir).collect() };
        let out = get_str(m, "out").map(|o| lexical(Path::new(o)));
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty() && !p.is_dir()) {
            if out != Some(lexical(parent)) {
                return Err(format!("--report {shown} cannot be written: its directory does not exist"));
            }
        }
    }
    if let (Some(r), Some(l)) = (report, labels) {
        if r == l || same_file(r, l) {
            return Err(format!("--report and --write-labels both name {}", repr_str(&r.to_string_lossy())));
        }
    }
    let read: Vec<&Path> = sources.iter().copied().chain(get_str(m, "label_set").map(Path::new)).collect();
    for (flag, path) in [("--report", report), ("--write-labels", labels)] {
        let Some(path) = path else { continue };
        if let Some(input) = read.iter().find(|input| same_file(input, path)) {
            return Err(format!(
                "{flag} {} is the input {}; name a new file",
                repr_str(&path.to_string_lossy()),
                repr_str(&input.to_string_lossy())
            ));
        }
    }
    Ok(())
}

/// Write the report where asked, print it, and exit on whether it is ok.
pub fn finish(report: &ConversionReport, m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    if let Some(path) = get_str(m, "report") {
        // A report naming an output would replace the sample just written.
        if let Some(output) = report.outputs.iter().find(|o| same_file(Path::new(o), Path::new(path))) {
            return Ok(ctx.fail(format!(
                "--report {} is the output {}; the report was not written",
                repr_str(path),
                repr_str(output)
            )));
        }
        medh5::h5::file::write_atomic(Path::new(path), (pretty(&report.to_json()) + "\n").as_bytes())?;
    }
    if flag(m, "json") {
        ctx.emit(&report.to_json(), true);
    } else {
        ctx.print(report.format(true));
    }
    Ok(if report.ok() { EXIT_OK } else { EXIT_ERROR })
}

pub fn migrate(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let paths = get_paths(m, "paths");
    let sources: Vec<&Path> = paths.iter().map(Path::new).collect();
    if let Err(message) = check_writes(m, &sources) {
        return Ok(ctx.fail(message));
    }
    let result = (|| -> medh5::Result<Option<ConversionReport>> {
        if let Some(target) = get_str(m, "write_labels") {
            let mut report = ConversionReport::new("migrate", "");
            let label_set = build_label_set(&sources, Some(&mut report))?;
            let written = write_sidecar(&label_set, Path::new(target))?;
            let note = format!("{}: {} classes --- review before migrating", written.display(), label_set.len());
            // With `--json`, stdout is the report alone.
            if flag(m, "json") {
                ctx.eprint(note);
            } else {
                ctx.print(note);
            }
            return Ok(Some(report));
        }
        let reviewed = match get_str(m, "label_set") {
            Some(p) => Some(load_sidecar(Path::new(p))?),
            None => None,
        };
        let report = migrate_paths(
            &sources,
            Path::new(req_str(m, "out")),
            req_str(m, "group_by"),
            get_str(m, "subject_key"),
            reviewed.as_ref(),
            req_str(m, "codec"),
        )?;
        Ok(Some(report))
    })();
    match result {
        Ok(Some(report)) => finish(&report, m, ctx),
        Ok(None) => Ok(EXIT_OK),
        Err(e) if e.is_medh5() || matches!(e, medh5::Error::Io(_)) => Ok(ctx.fail(e.to_string())),
        Err(e) => Err(e),
    }
}
