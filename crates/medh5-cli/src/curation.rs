//! `pack`, `unpack`, `ls`, `prov`, `agree`, `splits` and `scrub`.
//!
//! The curation half of the command line: shards (§2.2), the provenance graph
//! and quality records (§11), the cross-file split audit (§12.3) that no
//! per-file validator can perform, and the de-identification sweep (§11.4).

use std::path::Path;

use clap::ArgMatches;
use serde_json::{json, Map, Value};

use medh5::collection::{open_collection, pack, unpack};
use medh5::curation::agreement::{compare, Comparison};
use medh5::curation::splits::audit_splits;
use medh5::json::py_str;
use medh5::sample::open_sample;
use medh5::Error;

use crate::common::*;

pub fn dispatch(name: &str, m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    match name {
        "pack" => pack_cmd(m, ctx),
        "unpack" => unpack_cmd(m, ctx),
        "ls" => ls(m, ctx),
        "prov" => prov(m, ctx),
        "agree" => agree(m, ctx),
        "splits" => splits(m, ctx),
        _ => crate::scrub::scrub(m, ctx),
    }
}

/// `except MEDH5Error as exc: return fail(str(exc))`.
fn medh5_fail(result: CmdResult, ctx: &mut Ctx) -> CmdResult {
    match result {
        Err(e) if e.is_medh5() => Ok(ctx.fail(e.to_string())),
        other => other,
    }
}

fn file_size(path: &Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

fn pack_cmd(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let paths = get_paths(m, "paths");
    let sources: Vec<&Path> = paths.iter().map(Path::new).collect();
    let keys = get_strs(m, "keys");
    let out = match pack(&sources, Path::new(req_str(m, "out")), keys.as_deref()) {
        Ok(out) => out,
        Err(e) if e.is_medh5() => return Ok(ctx.fail(e.to_string())),
        Err(e) => return Err(e),
    };
    let size = file_size(&out);
    let source_bytes: u64 = sources.iter().map(|p| file_size(p)).sum();
    let shown = out.to_string_lossy().into_owned();
    if flag(m, "json") {
        let payload = json!({"out": shown, "samples": paths.len(), "bytes": size, "source_bytes": source_bytes});
        ctx.emit(&payload, true);
        return Ok(EXIT_OK);
    }
    ctx.print(format!(
        "{shown}: {} samples, {} (sources {})",
        paths.len(),
        human_bytes(size as f64),
        human_bytes(source_bytes as f64)
    ));
    Ok(EXIT_OK)
}

fn unpack_cmd(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let keys = get_strs(m, "keys");
    let written = match unpack(Path::new(req_str(m, "path")), Path::new(req_str(m, "out")), keys.as_deref(), ".medh5") {
        Ok(w) => w,
        Err(e) if e.is_medh5() => return Ok(ctx.fail(e.to_string())),
        Err(e) => return Err(e),
    };
    let shown: Vec<String> = written.iter().map(|p| p.to_string_lossy().into_owned()).collect();
    if flag(m, "json") {
        ctx.emit(&json!(shown), true);
        return Ok(EXIT_OK);
    }
    for path in shown {
        ctx.print(path);
    }
    Ok(EXIT_OK)
}

fn ls(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let path = req_str(m, "path");
    let result = (|| -> CmdResult {
        let collection = open_collection(Path::new(path))?;
        let summary = collection.summary()?;
        if flag(m, "json") {
            ctx.emit(&summary, true);
            return Ok(EXIT_OK);
        }
        ctx.print(format!("{path}  ({} samples, {})", collection.len()?, cell(&summary["version"])));
        let rows: Vec<Vec<String>> = summary["samples"]
            .as_array()
            .map(Vec::as_slice)
            .unwrap_or_default()
            .iter()
            .map(|e| {
                let content_id = or_dash(&e["content_id"]);
                vec![
                    cell(&e["key"]),
                    cell(&e["subject_id"]),
                    join_strs(&e["timepoints"], ","),
                    e["images"].as_array().map(Vec::len).unwrap_or(0).to_string(),
                    e["annotations"].as_array().map(Vec::len).unwrap_or(0).to_string(),
                    join_strs(&e["profiles"], ","),
                    content_id.chars().take(19).collect(),
                ]
            })
            .collect();
        ctx.print(table(&rows, &["key", "subject", "tp", "img", "ann", "profiles", "content_id"]));
        Ok(EXIT_OK)
    })();
    medh5_fail(result, ctx)
}

fn dash(value: Option<&str>) -> String {
    value.filter(|v| !v.is_empty()).unwrap_or("-").to_string()
}

fn prov(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let path = req_str(m, "path");
    let result = (|| -> CmdResult {
        let sample = open_sample(Path::new(path))?;
        let document = sample.document()?;
        let graph = &document.provenance;
        if flag(m, "json") {
            let quality: Map<String, Value> = document.quality.iter().map(|(k, v)| (k.clone(), v.to_json())).collect();
            let payload = json!({
                "provenance": graph.to_json(),
                "quality": quality,
                "deidentification": document.deidentification.as_ref().map(|d| d.to_json()),
            });
            ctx.emit(&payload, true);
            return Ok(EXIT_OK);
        }
        if graph.is_empty() {
            ctx.print("no provenance graph (§11.1)");
        } else {
            ctx.print("agents");
            let rows: Vec<Vec<String>> = graph
                .agents()
                .map(|a| {
                    vec![
                        a.id.clone(),
                        a.r#type.clone(),
                        a.name.clone(),
                        dash(a.version.as_deref()),
                        dash(a.role.as_deref()),
                    ]
                })
                .collect();
            ctx.print(indent(&table(&rows, &["id", "type", "name", "version", "role"])));
            ctx.print("\nactivities");
            let rows: Vec<Vec<String>> = graph
                .activities()
                .map(|act| {
                    let when = act.ended.as_deref().filter(|v| !v.is_empty()).or(act.started.as_deref());
                    let outputs = act.outputs.join(",");
                    vec![
                        act.id.clone(),
                        act.r#type.clone(),
                        dash(act.agent.as_deref()),
                        dash(when),
                        if outputs.is_empty() { "-".into() } else { outputs },
                    ]
                })
                .collect();
            ctx.print(indent(&table(&rows, &["id", "type", "agent", "when", "outputs"])));
        }
        if !document.quality.is_empty() {
            ctx.print("\nquality");
            let mut keys: Vec<&String> = document.quality.keys().collect();
            keys.sort();
            let rows: Vec<Vec<String>> = keys
                .into_iter()
                .map(|key| {
                    let record = &document.quality[key];
                    let agreement =
                        record.agreement.iter().map(|a| format!("{}={}", a.metric, gp(a.value, 3))).collect::<Vec<_>>();
                    let issues = record.issues.iter().map(|i| i.code.clone()).collect::<Vec<_>>();
                    vec![
                        key.clone(),
                        record.status.clone(),
                        record.confidence.map(|c| gp(c, 3)).unwrap_or_else(|| "-".into()),
                        if record.reviewed_by.is_empty() { "-".into() } else { record.reviewed_by.join(",") },
                        if agreement.is_empty() { "-".into() } else { agreement.join(";") },
                        if issues.is_empty() { "-".into() } else { issues.join(";") },
                    ]
                })
                .collect();
            ctx.print(indent(&table(&rows, &["key", "status", "conf", "reviewers", "agreement", "issues"])));
        }
        let deid = document.deidentification.as_ref();
        let shifted = match deid.and_then(|d| d.date_shift_days.as_ref()) {
            Some(days) => format!(", dates shifted {}d", py_str(&Value::Number(days.clone()))),
            None => String::new(),
        };
        ctx.print(format!(
            "\ndeidentification  {}{shifted}",
            deid.map(|d| d.method.clone()).unwrap_or("ABSENT (W903)".into())
        ));
        Ok(EXIT_OK)
    })();
    medh5_fail(result, ctx)
}

fn score(value: Option<f64>) -> String {
    match value {
        None => "undefined (nothing comparable)".into(),
        Some(v) => format!("{v:.4}"),
    }
}

fn agree(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let path = req_str(m, "path");
    let (a, b) = (req_str(m, "a"), req_str(m, "b"));
    let result = (|| -> CmdResult {
        let sample = open_sample(Path::new(path))?;
        let first = sample.annotation(a)?;
        let second = sample.annotation(b)?;
        // `compare` picks the comparison the two kinds support.
        let result = compare(first, second, get_str(m, "metric"), get_f64(m, "threshold"), None)?;
        let mut payload = result.to_json();
        if flag(m, "record") {
            let mut out = Map::new();
            out.insert("quality_agreement".into(), result.to_record()?.to_json());
            if let Value::Object(rest) = payload {
                out.extend(rest);
            }
            payload = Value::Object(out);
        }
        if flag(m, "json") {
            ctx.emit(&payload, true);
            return Ok(EXIT_OK);
        }
        ctx.print(format!("{a} vs {b}: {} = {}", cell(&payload["metric"]), score(result.value())));
        let skipped = match &result {
            Comparison::Voxel(v) => {
                if !v.per_class.is_empty() {
                    let mut rows: Vec<Vec<String>> =
                        v.per_class.iter().map(|(k, s)| vec![k.clone(), format!("{s:.4}")]).collect();
                    rows.sort();
                    ctx.print(indent(&table(&rows, &["class", v.metric.as_str()])));
                }
                v.skipped.clone()
            }
            Comparison::Instance(i) => {
                ctx.print(format!(
                    "  matched {} by {}, mean IoU {}; {} only in {a}, {} only in {b}",
                    i.matched.len(),
                    i.matched_by,
                    score(i.mean_iou()),
                    i.only_in_a.len(),
                    i.only_in_b.len()
                ));
                for (instance_id, class_a, class_b) in &i.class_mismatches {
                    ctx.print(format!("  MISMATCH instance {instance_id}: class {class_a} vs {class_b}"));
                }
                i.skipped.clone()
            }
        };
        if !skipped.is_empty() {
            ctx.print(format!("\nnot scored: {}", skipped.join(", ")));
            ctx.print("  a class one side never examined is not a disagreement (§11.3)");
        }
        Ok(EXIT_OK)
    })();
    match result {
        Err(Error::Key(_)) => {
            let e = result.unwrap_err();
            Ok(ctx.fail(format!("no such annotation: {}", e.python_str())))
        }
        other => medh5_fail(other, ctx),
    }
}

fn splits(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let paths = get_paths(m, "paths");
    let audit = audit_splits(&paths);
    if flag(m, "json") {
        ctx.emit(&audit.to_json(), true);
        return Ok(if audit.ok() { EXIT_OK } else { EXIT_ERROR });
    }
    let set_ids = audit.set_ids();
    if set_ids.is_empty() {
        ctx.print(format!("no split claims in {} file(s)", paths.len()));
    }
    let counts = audit.counts();
    for set_id in &set_ids {
        let per = &counts[set_id];
        let total: usize = per.values().sum();
        let parts: Vec<String> = per.iter().map(|(k, v)| format!("{k}={v}")).collect();
        ctx.print(format!("{set_id}: {total} samples  {}", parts.join("  ")));
    }
    for conflict in &audit.conflicts {
        ctx.print(format!("W906  {conflict}"));
        for (manifest, paths) in &conflict.paths_by_manifest {
            let short: String = manifest.chars().take(16).collect();
            ctx.print(format!("        {short}  {} file(s), e.g. {}", paths.len(), paths[0]));
        }
    }
    for leak in &audit.leaks {
        ctx.print(format!("LEAK  {leak}"));
        for path in &leak.paths {
            ctx.print(format!("        {path}"));
        }
    }
    if !audit.unclaimed.is_empty() {
        ctx.print(format!("\n{} file(s) carry no split claim", audit.unclaimed.len()));
    }
    for (path, error) in &audit.unreadable {
        ctx.print(format!("UNREADABLE  {path}: {error}"));
    }
    if !audit.leaks.is_empty() {
        ctx.print("\na grouping key in two partitions is train/test leakage (§12.2); re-split rather than re-stamp.");
    }
    Ok(if audit.ok() { EXIT_OK } else { EXIT_ERROR })
}
