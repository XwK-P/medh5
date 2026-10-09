//! `info`, `tree`, `validate`, `verify`, `fix`, `timeline`, `track`.

use std::path::Path;

use clap::ArgMatches;
use medh5::hdf5;
use medh5::hdf5::types::TypeDescriptor as TD;
use serde_json::{json, Map, Value};

use medh5::collection::{open_any, AnyFile, Collection};
use medh5::curation::tracking::{build_tracks, Track, Tracking};
use medh5::h5::{attrs, ops};
use medh5::integrity::repair::{fix, FixOptions};
use medh5::json::py_str;
use medh5::labels::ClassKey;
use medh5::sample::Sample;
use medh5::storage::codecs::describe_filters;
use medh5::validate::validate_paths;

use crate::common::*;

pub fn dispatch(name: &str, m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    match name {
        "info" => info(m, ctx),
        "tree" => tree(m, ctx),
        "fix" => fix_cmd(m, ctx),
        "validate" => validate(m, ctx),
        "verify" => verify(m, ctx),
        "timeline" => timeline(m, ctx),
        _ => track(m, ctx),
    }
}

/// Open a sample, or one member of a collection, from a CLI argument.
fn open(m: &ArgMatches, path: &str) -> medh5::Result<AnyFile> {
    open_any(Path::new(path), get_str(m, "key"))
}

/// The message for a per-sample command handed a whole shard.
fn needs_key(path: &str, collection: &Collection) -> medh5::Result<String> {
    let mut keys = collection.keys()?;
    keys.sort();
    let shown = keys.iter().take(5).cloned().collect::<Vec<_>>().join(", ") + if keys.len() > 5 { " ..." } else { "" };
    Ok(format!(
        "{path} is a collection of {} sample(s); this command reports on one sample, so name it with --key (keys: \
         {shown})",
        keys.len()
    ))
}

fn info(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let path = req_str(m, "path");
    let as_json = flag(m, "json");
    let sample = match open(m, path) {
        Ok(AnyFile::Collection(c)) => return collection_info(path, &c, as_json, ctx),
        Ok(AnyFile::Sample(s)) => s,
        Err(e) if e.is_medh5() => return Ok(ctx.fail(e.to_string())),
        Err(e) => return Err(e),
    };
    match sample_info(path, &sample, as_json, ctx) {
        Err(e) if e.is_medh5() => Ok(ctx.fail(e.to_string())),
        other => other,
    }
}

fn sample_info(path: &str, sample: &Sample, as_json: bool, ctx: &mut Ctx) -> CmdResult {
    let summary = sample.summary()?;
    if as_json {
        ctx.emit(&summary, true);
        return Ok(EXIT_OK);
    }
    let s = |k: &str| summary.get(k).cloned().unwrap_or(Value::Null);
    ctx.print(path);
    ctx.print(format!("  format      {} ({})", cell(&s("version")), cell(&s("kind"))));
    ctx.print(format!("  profiles    {}", join_strs(&s("profiles"), ", ")));
    ctx.print(format!("  sample      {}  subject {}", cell(&s("sample_id")), cell(&s("subject_id"))));
    ctx.print(format!("  content_id  {}", or_dash(&s("content_id"))));
    ctx.print("\ntimepoints");
    let rows: Vec<Vec<String>> = items(&s("timepoints"))
        .iter()
        .map(|t| {
            vec![
                cell(&t["id"]),
                cell(&t["index"]),
                or_dash(&t["label"]),
                if t["days_from_baseline"].is_null() { "-".into() } else { cell(&t["days_from_baseline"]) },
            ]
        })
        .collect();
    ctx.print(indent(&table(&rows, &["id", "index", "label", "days"])));
    ctx.print("\ngrids");
    let rows: Vec<Vec<String>> = items(&s("grids"))
        .iter()
        .map(|grid| {
            vec![
                cell(&grid["id"]),
                items(&grid["shape"]).iter().map(cell).collect::<Vec<_>>().join("x"),
                items(&grid["spacing"]).iter().map(|v| g(v.as_f64().unwrap_or(f64::NAN))).collect::<Vec<_>>().join(" "),
                cell(&grid["coord_system"]),
                cell(&grid["units"]),
                or_dash(&grid["timepoint"]),
                or_dash(&grid["frame_uid"]),
            ]
        })
        .collect();
    ctx.print(indent(&table(&rows, &["id", "shape", "spacing", "system", "units", "timepoint", "frame"])));
    ctx.print("\nimages");
    let images = sample.images()?;
    let mut rows = Vec::new();
    for i in items(&s("images")) {
        let id = cell(&i["id"]);
        let codec = match images.get(&id) {
            Some(image) => describe_filters(&image.dataset()?)?,
            None => "-".into(),
        };
        rows.push(vec![
            id,
            cell(&i["modality"]),
            items(&i["shape"]).iter().map(cell).collect::<Vec<_>>().join("x"),
            cell(&i["dtype"]),
            or_dash(&i["value_units"]),
            cell(&i["grid"]),
            codec,
            human_bytes(i["nbytes"].as_f64().unwrap_or(0.0)),
        ]);
    }
    ctx.print(indent(&table(&rows, &["id", "mod", "shape", "dtype", "units", "grid", "codec", "raw"])));
    let annotations = items(&s("annotations"));
    if !annotations.is_empty() {
        ctx.print("\nannotations");
        let rows: Vec<Vec<String>> = annotations
            .iter()
            .map(|a| {
                let count = |k: &str| a.get(k).map(cell).unwrap_or_else(|| "0".into());
                let tps = a.get("timepoints").map(|v| join_strs(v, ",")).unwrap_or_default();
                vec![
                    cell(&a["id"]),
                    cell(&a["kind"]),
                    cell(&a["task"]),
                    a.get("grid").map(or_dash).unwrap_or_else(|| "-".into()),
                    if tps.is_empty() { "-".into() } else { tps },
                    format!("{}/{}", count("annotated_classes"), count("classes")),
                    if a.get("fully_covered").is_some_and(truthy) { "yes".into() } else { "PARTIAL".into() },
                    a.get("quality").map(or_dash).unwrap_or_else(|| "-".into()),
                ]
            })
            .collect();
        ctx.print(indent(&table(&rows, &["id", "kind", "task", "grid", "tp", "cover", "full", "quality"])));
    }
    let transforms = summary.get("transforms").cloned().unwrap_or(Value::Null);
    if truthy(&transforms) {
        ctx.print("\ntransforms");
        let rows: Vec<Vec<String>> = items(&transforms)
            .iter()
            .map(|t| {
                let tps = join_strs(&t["timepoints"], ",");
                vec![
                    cell(&t["id"]),
                    cell(&t["kind"]),
                    cell(&t["from_frame"]),
                    cell(&t["to_frame"]),
                    if tps.is_empty() { "-".into() } else { tps },
                    if truthy(&t["invertible"]) { "yes".into() } else { "no".into() },
                    or_dash(&t["metrics"]),
                ]
            })
            .collect();
        ctx.print(indent(&table(&rows, &["id", "kind", "from", "to", "tp", "inv", "metrics"])));
    }
    if truthy(&s("index")) {
        ctx.print(format!("\nindex        {}", join_strs(&s("index"), ", ")));
    }
    let label = s("label_set");
    if truthy(&label) {
        ctx.print(format!(
            "\nlabel set    {} v{} ({} classes, {})",
            cell(&label["id"]),
            cell(&label["version"]),
            cell(&label["classes"]),
            cell(&label["form"])
        ));
    }
    Ok(EXIT_OK)
}

fn items(value: &Value) -> Vec<Value> {
    match value {
        Value::Array(a) => a.clone(),
        _ => Vec::new(),
    }
}

/// A shard with no `--key` summarises the shard, rather than refusing.
fn collection_info(path: &str, collection: &Collection, as_json: bool, ctx: &mut Ctx) -> CmdResult {
    let summary = collection.summary()?;
    if as_json {
        ctx.emit(&summary, true);
        return Ok(EXIT_OK);
    }
    let samples = items(&summary["samples"]);
    ctx.print(path);
    ctx.print(format!("  format      {} ({})", cell(&summary["version"]), cell(&summary["kind"])));
    ctx.print(format!("  samples     {}", samples.len()));
    ctx.print("");
    let rows: Vec<Vec<String>> = samples
        .iter()
        .map(|e| {
            let tps = join_strs(&e["timepoints"], ",");
            vec![
                cell(&e["key"]),
                cell(&e["sample_id"]),
                cell(&e["subject_id"]),
                if tps.is_empty() { "-".into() } else { tps },
                items(&e["images"]).len().to_string(),
                items(&e["annotations"]).len().to_string(),
                join_strs(&e["profiles"], ","),
            ]
        })
        .collect();
    ctx.print(table(&rows, &["key", "sample", "subject", "tp", "images", "anns", "profiles"]));
    ctx.print("\nper-sample detail: `medh5 info PATH --key KEY`");
    Ok(EXIT_OK)
}

const ROLES: [(&str, &str); 6] = [
    ("meta", "sample document (§2.4)"),
    ("grids", "geometry (§3.2)"),
    ("images", "image data (§4)"),
    ("annotations", "ground truth (§6-§9)"),
    ("transforms", "spatial mappings (§10)"),
    ("index", "derived sampling caches (§14.3)"),
];

fn role(name: &str) -> &'static str {
    ROLES.iter().find(|(n, _)| *n == name).map(|(_, r)| *r).unwrap_or("extension object (§16)")
}

/// Members of a group, `meta` first and the rest by name.
fn sorted_members(group: &hdf5::Group) -> medh5::Result<Vec<String>> {
    let mut names = ops::members(group)?;
    names.sort_by(|a, b| (a != "meta", a).cmp(&(b != "meta", b)));
    Ok(names)
}

/// NumPy's dtype string of a dataset, as h5py reports it.
fn dtype_str(ds: &hdf5::Dataset) -> String {
    match ds.dtype().and_then(|t| t.to_descriptor()) {
        Ok(TD::VarLenUnicode | TD::VarLenAscii) => "|O".into(),
        Ok(TD::FixedAscii(n) | TD::FixedUnicode(n)) => format!("|S{n}"),
        Ok(td) => match attrs::numeric_dtype(&td) {
            Some(d) => d.numpy_str().into(),
            None => format!("|V{}", ds.dtype().map(|t| t.size()).unwrap_or(0)),
        },
        Err(_) => "|V0".into(),
    }
}

fn dtype_name(ds: &hdf5::Dataset) -> String {
    match ds.dtype().and_then(|t| t.to_descriptor()) {
        Ok(TD::VarLenUnicode | TD::VarLenAscii) => "object".into(),
        Ok(TD::FixedAscii(n) | TD::FixedUnicode(n)) => format!("|S{n}"),
        Ok(td) => match attrs::numeric_dtype(&td) {
            Some(d) => d.name().into(),
            None => format!("|V{}", ds.dtype().map(|t| t.size()).unwrap_or(0)),
        },
        Err(_) => "|V0".into(),
    }
}

fn describe(group: &hdf5::Group, name: &str) -> medh5::Result<String> {
    if let Some(ds) = ops::child_dataset(group, name) {
        let shape = ds.shape();
        let shape = if shape.is_empty() {
            "scalar".to_string()
        } else {
            shape.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("x")
        };
        return Ok(format!("{shape} {} {}", dtype_str(&ds), describe_filters(&ds)?));
    }
    let Some(child) = ops::child_group(group, name) else { return Ok("group".into()) };
    match attrs::read(&child, "kind")? {
        Some(kind) => Ok(format!("group kind={}", attrs::stringify_value(&kind))),
        None => Ok("group".into()),
    }
}

fn root_of(opened: &AnyFile) -> hdf5::Group {
    match opened {
        AnyFile::Sample(s) => s.root.clone(),
        AnyFile::Collection(c) => c.root.clone(),
    }
}

fn tree(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let path = req_str(m, "path");
    let result = (|| -> CmdResult {
        let opened = open(m, path)?;
        let root = root_of(&opened);
        if flag(m, "json") {
            ctx.emit(&json!({"path": path, "objects": tree_json(&root, "")?}), true);
            return Ok(EXIT_OK);
        }
        ctx.print(path);
        for name in sorted_members(&root)? {
            let role = role(&name);
            if ops::is_dataset(&root, &name) {
                ctx.print(format!("├── {} {}   # {role}", pad(&name, 22), describe(&root, &name)?));
                continue;
            }
            let fill = 21usize.saturating_sub(name.chars().count());
            ctx.print(format!("├── {name}/{} # {role}", " ".repeat(fill)));
            let Some(node) = ops::child_group(&root, &name) else { continue };
            let mut children = ops::members(&node)?;
            children.sort();
            for child in children {
                ctx.print(format!("│   ├── {} {}", pad(&child, 20), describe(&node, &child)?));
                if let Some(sub) = ops::child_group(&node, &child) {
                    let mut leaves = ops::members(&sub)?;
                    leaves.sort();
                    for leaf in leaves {
                        ctx.print(format!("│   │   ├── {} {}", pad(&leaf, 16), describe(&sub, &leaf)?));
                    }
                }
            }
        }
        Ok(EXIT_OK)
    })();
    match result {
        Err(e) if e.is_medh5() => Ok(ctx.fail(e.to_string())),
        other => other,
    }
}

/// The same listing `tree` prints, as data.
fn tree_json(node: &hdf5::Group, prefix: &str) -> medh5::Result<Vec<Value>> {
    let mut out = Vec::new();
    for name in sorted_members(node)? {
        let path = format!("{prefix}{name}");
        let is_dataset = ops::is_dataset(node, &name);
        let mut entry = Map::new();
        entry.insert("path".into(), json!(path));
        entry.insert("name".into(), json!(name));
        entry.insert("kind".into(), json!(if is_dataset { "dataset" } else { "group" }));
        entry.insert("describe".into(), json!(describe(node, &name)?));
        if prefix.is_empty() {
            entry.insert("role".into(), json!(role(&name)));
        }
        if let Some(ds) = ops::child_dataset(node, &name) {
            entry.insert("shape".into(), json!(ds.shape()));
            entry.insert("dtype".into(), json!(dtype_name(&ds)));
        } else if let Some(child) = ops::child_group(node, &name) {
            entry.insert("children".into(), json!(tree_json(&child, &format!("{path}/"))?));
        }
        out.push(Value::Object(entry));
    }
    Ok(out)
}

fn validate(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let paths = get_paths(m, "paths");
    let refs: Vec<&Path> = paths.iter().map(Path::new).collect();
    let profiles = get_strs(m, "profiles");
    let reports = validate_paths(&refs, req_str(m, "level"), profiles.as_deref())?;
    if flag(m, "json") {
        ctx.emit(&Value::Array(reports.iter().map(|r| r.to_json()).collect()), true);
    } else {
        for report in &reports {
            ctx.print(report.format(flag(m, "verbose")));
        }
    }
    Ok(if reports.iter().all(|r| r.ok()) { EXIT_OK } else { EXIT_ERROR })
}

fn verify(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let as_json = flag(m, "json");
    let partial = get_strs(m, "partial");
    let mut results = Vec::new();
    let mut ok = true;
    for path in get_paths(m, "paths") {
        let result = match open(m, &path) {
            Ok(AnyFile::Collection(c)) => return Ok(ctx.fail(needs_key(&path, &c)?)),
            Ok(AnyFile::Sample(s)) => s.verify(partial.as_deref()),
            Err(e) => Err(e),
        };
        let result = match result {
            Ok(r) => r,
            Err(e) if e.is_medh5() => return Ok(ctx.fail(e.to_string())),
            Err(e) => return Err(e),
        };
        let mut summary = Map::new();
        summary.insert("path".into(), json!(path));
        if let Value::Object(rest) = result.summary() {
            summary.extend(rest);
        }
        results.push(Value::Object(summary));
        ok = ok && result.ok();
        if !as_json {
            let state = if result.ok() { "OK" } else { "FAILED" };
            // Three answers, not two: a partial pass does not recompute the
            // root, and a file may declare no `content_id` at all (§13.2).
            let content = match result.content_id_ok() {
                Some(true) => "ok",
                Some(false) => "MISMATCH",
                None => "not verified",
            };
            ctx.print(format!("{path}: {state}  {} objects, content_id {content}", result.checked.len()));
            for name in &result.mismatched {
                ctx.print(format!("  MISMATCH  {name}"));
            }
            for name in &result.unattested {
                ctx.print(format!("  UNSIGNED  {name} (no digest, inside an attested object)"));
            }
            for name in &result.stale_index {
                ctx.print(format!("  STALE     index/{name} (rebuild with `medh5 index build`)"));
            }
        }
    }
    ctx.emit(&Value::Array(results), as_json);
    Ok(if ok { EXIT_OK } else { EXIT_ERROR })
}

fn fix_cmd(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let as_json = flag(m, "json");
    let options = FixOptions {
        rebuild_index: flag(m, "rebuild_index"),
        rewrite_digests: flag(m, "rewrite_digests"),
        reason: get_str(m, "reason").map(str::to_string),
        performed_by: get_str(m, "performed_by").map(str::to_string),
        max_coords: None,
    };
    let mut results = Vec::new();
    for path in get_paths(m, "paths") {
        let repair = match fix(Path::new(&path), &options) {
            Ok(r) => r,
            Err(e) if e.is_medh5() => return Ok(ctx.fail(e.to_string())),
            Err(e) => return Err(e),
        };
        results.push(repair.to_json());
        if as_json {
            continue;
        }
        let diagnosis = &repair.diagnosis;
        if repair.changed() {
            let mut done = Vec::new();
            if !repair.rebuilt_index.is_empty() {
                done.push(format!("rebuilt index for {}", repair.rebuilt_index.join(", ")));
            }
            if repair.rewrote_digests {
                done.push("rewrote digests".to_string());
            }
            ctx.print(format!("{path}: {}", done.join("; ")));
            for note in &repair.notes {
                ctx.print(format!("  note: {note}"));
            }
        } else if diagnosis.clean() {
            ctx.print(format!("{path}: nothing to fix"));
        } else {
            ctx.print(format!("{path}: needs attention, nothing changed"));
            if !diagnosis.stale_index.is_empty() {
                ctx.print(format!("  stale index: {} (--rebuild-index)", diagnosis.stale_index.join(", ")));
            }
            if !diagnosis.mismatched.is_empty() {
                ctx.print(format!(
                    "  digest mismatch: {} (--rewrite-digests, and read what it means first)",
                    diagnosis.mismatched.join(", ")
                ));
            }
            if diagnosis.content_id_ok == Some(false) {
                ctx.print("  content_id does not match the file's own contents");
            }
        }
    }
    // Non-zero when a file still needs attention: a fix run that found
    // problems and was not asked to act on them has not succeeded, it has
    // reported.
    let outstanding = results.iter().any(|r| {
        !truthy(&r["changed"]) && (truthy(&r["diagnosis"]["needs_index"]) || truthy(&r["diagnosis"]["needs_digests"]))
    });
    ctx.emit(&Value::Array(results), as_json);
    Ok(if outstanding { EXIT_ERROR } else { EXIT_OK })
}

fn timeline(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let path = req_str(m, "path");
    let result = (|| -> CmdResult {
        let sample = match open(m, path)? {
            AnyFile::Collection(c) => return Ok(ctx.fail(needs_key(path, &c)?)),
            AnyFile::Sample(s) => s,
        };
        let mut rows = Vec::new();
        let mut payload = Vec::new();
        let grids = sample.grids()?;
        for tp in sample.timepoints()?.iter() {
            let mut images = sample.images_at(&tp.id)?;
            images.sort();
            let mut annotations = sample.annotations_at(&tp.id)?;
            annotations.sort();
            let mut tp_grids: Vec<String> = grids
                .iter()
                .filter(|(_, g)| g.timepoint.as_deref() == Some(tp.id.as_str()))
                .map(|(k, _)| k.clone())
                .collect();
            tp_grids.sort();
            rows.push(vec![
                tp.id.clone(),
                tp.index.to_string(),
                tp.label.clone().filter(|l| !l.is_empty()).unwrap_or_else(|| "-".into()),
                tp.days_from_baseline.as_ref().map(|d| py_str(&Value::Number(d.clone()))).unwrap_or_else(|| "-".into()),
                if images.is_empty() { "-".into() } else { images.join(",") },
                if annotations.is_empty() { "-".into() } else { annotations.join(",") },
            ]);
            let mut entry = match tp.to_json() {
                Value::Object(m) => m,
                _ => Map::new(),
            };
            entry.insert("images".into(), json!(images));
            entry.insert("annotations".into(), json!(annotations));
            entry.insert("grids".into(), json!(tp_grids));
            payload.push(Value::Object(entry));
        }
        if flag(m, "json") {
            ctx.emit(&Value::Array(payload), true);
            return Ok(EXIT_OK);
        }
        ctx.print(table(&rows, &["id", "index", "label", "days", "images", "annotations"]));
        let spanning: Vec<String> = sample
            .annotations()?
            .values()
            .filter(|a| a.timepoints().len() > 1)
            .map(|a| format!("{} ({})", a.ann_id, a.timepoints().join(",")))
            .collect();
        if !spanning.is_empty() {
            ctx.print(format!("\nspanning annotations: {}", spanning.join(", ")));
        }
        Ok(EXIT_OK)
    })();
    match result {
        Err(e) if e.is_medh5() => Ok(ctx.fail(e.to_string())),
        other => other,
    }
}

fn track(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let path = req_str(m, "path");
    let result = (|| -> CmdResult {
        let sample = match open(m, path)? {
            AnyFile::Collection(c) => return Ok(ctx.fail(needs_key(path, &c)?)),
            AnyFile::Sample(s) => s,
        };
        let class_key = get_str(m, "class_key").map(|k| ClassKey::Key(k.to_string()));
        let tracking = build_tracks(&sample, class_key.as_ref(), true)?;
        if flag(m, "json") {
            ctx.emit(&tracking.to_json(), true);
            return Ok(EXIT_OK);
        }
        if tracking.tracks.is_empty() {
            ctx.print("no instance-carrying annotations in this sample");
            return Ok(EXIT_OK);
        }
        let timepoints: Vec<String> =
            if tracking.timepoints.is_empty() { vec!["-".into()] } else { tracking.timepoints.clone() };
        let mut rows = Vec::new();
        for (instance_id, track) in &tracking.tracks {
            let states = tracking.states(*instance_id);
            let mut row = vec![
                instance_id.to_string(),
                track.class_key.clone().filter(|k| !k.is_empty()).unwrap_or_else(|| track.class_id().to_string()),
            ];
            for tp in &timepoints {
                let state = states.get(tp).copied().unwrap_or("unexamined");
                row.push(if state != "present" {
                    state.to_string()
                } else {
                    // Two raters measured it there: no one volume is the visit's.
                    match track.volume(tp) {
                        Ok(Some(v)) => gp(v, 4),
                        Ok(None) => "present".into(),
                        Err(_) => format!("{} raters", track.measurements_at(tp).len()),
                    }
                });
            }
            row.push(trend(&tracking, *instance_id, track, &timepoints));
            rows.push(row);
        }
        let mut headers = vec!["instance".to_string(), "class".to_string()];
        headers.extend(timepoints.iter().cloned());
        headers.push("trend".into());
        ctx.print(table(&rows, &headers));
        for (instance_id, class_ids) in tracking.class_conflicts() {
            ctx.print(format!(
                "\nW909  instance {instance_id} carries class ids {}",
                medh5::json::repr_int_list(&class_ids)
            ));
        }
        ctx.print(
            "\nvolume in the grid's units; `resolved` means the class was in `annotated_class_ids` at that timepoint \
             and the object was not found, `unexamined` that nobody looked (spec §7.4, §11.3).",
        );
        Ok(EXIT_OK)
    })();
    match result {
        Err(e) if e.is_medh5() => Ok(ctx.fail(e.to_string())),
        other => other,
    }
}

/// Relative volume change from first to last visit, where it is measurable.
fn trend(tracking: &Tracking, instance_id: u64, track: &Track, timepoints: &[String]) -> String {
    if timepoints.len() < 2 {
        return "-".into();
    }
    match track.relative_change(&timepoints[0], &timepoints[timepoints.len() - 1]) {
        Ok(Some(change)) => return percent(change, 1, true),
        Err(_) => return "-".into(),
        Ok(None) => {}
    }
    if tracking.is_new(instance_id) {
        return "new".into();
    }
    if tracking.is_resolved(instance_id) {
        return "resolved".into();
    }
    "-".into()
}
