//! `dataset` --- the cohort commands: index, split, stats, check.
//!
//! Every one of these reads metadata and nothing else unless told otherwise,
//! which is what makes them usable on a cohort rather than on a demo.

use std::path::Path;

use clap::ArgMatches;
use indexmap::IndexMap;
use serde_json::{json, Map, Value};

use medh5::dataset::check::check;
use medh5::dataset::manifest::{scan, Manifest};
use medh5::dataset::split::{make_splits, write_claims, Split, SplitOptions};
use medh5::dataset::stats::{compute_stats, StatsOptions};
use medh5::json::{py_float, repr_str};
use medh5::Error;

use crate::common::*;

pub fn dispatch(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let result = match m.subcommand() {
        Some(("index", sub)) => index(sub, ctx),
        Some(("split", sub)) => split(sub, ctx),
        Some(("stats", sub)) => stats(sub, ctx),
        Some(("check", sub)) => check_cmd(sub, ctx),
        _ => return Ok(ctx.fail("usage: medh5 dataset {index|split|stats|check} ... (see --help)")),
    };
    match result {
        Err(e) if e.is_medh5() || matches!(e, Error::Io(_) | Error::Value(_)) => Ok(ctx.fail(e.python_str())),
        other => other,
    }
}

fn index(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let (manifest, failures) = scan(Path::new(req_str(m, "root")), flag(m, "strict"))?;
    let target = manifest.save(Path::new(req_str(m, "out")))?;
    let digest = manifest.sha256();
    if flag(m, "json") {
        let payload = json!({
            "manifest": target.to_string_lossy(),
            "samples": manifest.len(),
            "subjects": manifest.subjects().len(),
            "sha256": digest,
            "failed": failures,
        });
        ctx.emit(&payload, true);
    } else {
        ctx.print(format!(
            "{}: {} sample(s), {} subject(s), sha256 {}",
            target.display(),
            manifest.len(),
            manifest.subjects().len(),
            &digest[..12]
        ));
        for failure in &failures {
            ctx.print(format!("  unreadable: {failure}"));
        }
    }
    Ok(if manifest.is_empty() { EXIT_ERROR } else { EXIT_OK })
}

fn fmt_ratios(ratios: &IndexMap<String, f64>) -> String {
    ratios.iter().map(|(k, v)| format!("{k}={}", py_float(*v))).collect::<Vec<_>>().join(",")
}

fn parse_ratios(text: &str) -> medh5::Result<IndexMap<String, f64>> {
    let mut out = IndexMap::new();
    for part in text.split(',') {
        let Some((name, value)) = part.split_once('=') else {
            return Err(Error::File(format!("--ratios expects NAME=VALUE, got {}", repr_str(part))));
        };
        let parsed: f64 = value
            .trim()
            .parse()
            .map_err(|_| Error::Value(format!("could not convert string to float: {}", repr_str(value))))?;
        out.insert(name.trim().to_string(), parsed);
    }
    Ok(out)
}

fn balance_cell(split: &Split, partition: &str) -> String {
    let balance = split.balance();
    let cells: Vec<String> = balance
        .get(partition)
        .map(|b| b.iter().filter(|(k, _)| k.as_str() != "-").map(|(k, v)| format!("{k}:{v}")).collect())
        .unwrap_or_default();
    if cells.is_empty() {
        "-".into()
    } else {
        cells.join(", ")
    }
}

fn split(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let manifest = Manifest::load(Path::new(req_str(m, "manifest")))?;
    let ratios = match get_str(m, "ratios") {
        Some(text) if !text.is_empty() => Some(parse_ratios(text)?),
        _ => None,
    };
    let options = SplitOptions {
        set_id: req_str(m, "set_id").to_string(),
        group_by: req_str(m, "group_by").to_string(),
        stratify_by: get_str(m, "stratify_by").map(str::to_string),
        ratios,
        k_folds: get_i64(m, "k_folds"),
        seed: get_i64(m, "seed").unwrap_or(0),
    };
    let split = make_splits(&manifest, &options)?;
    let written = if flag(m, "write_claims") {
        write_claims(&split, &manifest, get_str(m, "assigned_by"), get_i64(m, "fold"))?
    } else {
        Vec::new()
    };
    if let Some(out) = get_str(m, "out") {
        std::fs::write(out, medh5::json::pretty(&split.to_json()) + "\n")?;
    }
    if flag(m, "json") {
        let mut payload = match split.to_json() {
            Value::Object(m) => m,
            _ => Map::new(),
        };
        payload.insert("claims_written".into(), json!(written));
        ctx.emit(&Value::Object(payload), true);
    } else {
        let rows: Vec<Vec<String>> = split
            .counts()
            .iter()
            .map(|(partition, n)| vec![partition.clone(), n.to_string(), balance_cell(&split, partition)])
            .collect();
        let stratify = get_str(m, "stratify_by").unwrap_or("-");
        ctx.print(table(&rows, &["partition", "samples", stratify]));
        ctx.print(format!(
            "{} group(s) by {}, manifest {}",
            split.assignments.len(),
            req_str(m, "group_by"),
            &split.manifest_sha256[..12.min(split.manifest_sha256.len())]
        ));
        if !written.is_empty() {
            ctx.print(format!("wrote split claims into {} file(s)", written.len()));
        }
        let empty = split.empty_folds();
        if !empty.is_empty() {
            ctx.print(format!(
                "WARNING: fold(s) {} got no groups --- {} group(s) cannot fill {} folds",
                empty.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(", "),
                split.assignments.len(),
                split.k_folds.map(|k| k.to_string()).unwrap_or_else(|| "None".into())
            ));
        }
        let underfilled = split.underfilled();
        if !underfilled.is_empty() {
            ctx.print(format!(
                "WARNING: {} got no groups --- {} indivisible group(s) cannot be split in the ratios {}",
                underfilled.join(", "),
                split.assignments.len(),
                fmt_ratios(&split.ratios)
            ));
        }
        let leaks = split.leaks();
        if !leaks.is_empty() {
            ctx.print(format!("LEAK: {}", leaks.join(", ")));
        }
    }
    Ok(if split.leaks().is_empty() { EXIT_OK } else { EXIT_ERROR })
}

fn stats(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let manifest = Manifest::load(Path::new(req_str(m, "manifest")))?;
    let mut entries: Vec<&medh5::dataset::Entry> = manifest.entries.iter().collect();
    if let Some(partition) = get_str(m, "partition") {
        let set_id = req_str(m, "set_id");
        entries.retain(|e| {
            e.splits.iter().any(|c| {
                c.get("set_id").and_then(Value::as_str) == Some(set_id)
                    && c.get("partition").and_then(Value::as_str) == Some(partition)
            })
        });
        if entries.is_empty() {
            return Ok(ctx.fail(format!(
                "no sample claims partition {} of set {} --- run `medh5 dataset split --write-claims` first",
                repr_str(partition),
                repr_str(set_id)
            )));
        }
    }
    let paths: Vec<&str> = entries.iter().map(|e| e.path.as_str()).collect();
    let options = StatsOptions {
        images: get_strs(m, "images"),
        annotations: get_strs(m, "annotations"),
        sample_stride: get_i64(m, "stride").unwrap_or(1).max(0) as usize,
        physical: !flag(m, "stored"),
    };
    let workers = get_i64(m, "workers").unwrap_or(1).max(0) as usize;
    let result = compute_stats(&paths, &options, workers)?;
    if let Some(out) = get_str(m, "out") {
        std::fs::write(out, medh5::json::pretty(&result.to_json()) + "\n")?;
    }
    if flag(m, "json") {
        ctx.emit(&result.to_json(), true);
    } else {
        let mut images: Vec<(&String, &medh5::dataset::Moments)> = result.images.iter().collect();
        images.sort_by(|a, b| a.0.cmp(b.0));
        let rows: Vec<Vec<String>> = images
            .into_iter()
            .map(|(key, mo)| vec![key.clone(), gp(mo.mean, 4), gp(mo.std(), 4), gp(mo.minimum, 4), gp(mo.maximum, 4)])
            .collect();
        ctx.print(table(&rows, &["image", "mean", "std", "min", "max"]));
        ctx.print(format!(
            "intensities: {}",
            if result.physical {
                "physical (rescale applied, as the loaders read)"
            } else {
                "stored (rescale not applied)"
            }
        ));
        if !result.classes.is_empty() {
            let mut classes: Vec<&medh5::dataset::ClassStats> = result.classes.values().collect();
            classes.sort_by_key(|s| s.class_id);
            let rows: Vec<Vec<String>> = classes
                .into_iter()
                .map(|s| {
                    vec![
                        s.class_id.to_string(),
                        thousands(i128::from(s.voxels)),
                        format!("{}/{}", s.present_in, s.examined_in),
                        percent(s.prevalence(), 0, false),
                    ]
                })
                .collect();
            ctx.print(table(&rows, &["class", "voxels", "present/examined", "prevalence"]));
        }
        for failure in &result.failures {
            ctx.print(format!("  unreadable: {failure}"));
        }
    }
    Ok(if result.samples > 0 { EXIT_OK } else { EXIT_ERROR })
}

fn check_cmd(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let manifest = Manifest::load(Path::new(req_str(m, "manifest")))?;
    let report = check(&manifest, get_str(m, "set_id"), flag(m, "deep"));
    if flag(m, "json") {
        ctx.emit(&report.to_json(), true);
    } else {
        ctx.print(report.format());
        if !report.coverage.is_empty() {
            let rows: Vec<Vec<String>> = report
                .coverage
                .iter()
                .map(|(class_id, v)| {
                    vec![class_id.to_string(), v.examined_in.to_string(), v.present_in.to_string(), v.of.to_string()]
                })
                .collect();
            ctx.print(table(&rows, &["class", "examined in", "present in", "of"]));
        }
    }
    Ok(if report.ok() { EXIT_OK } else { EXIT_ERROR })
}
