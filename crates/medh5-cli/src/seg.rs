//! `seg stats`, `seg convert`, `index build` --- the encoding tools.

use std::path::Path;
use std::sync::Arc;

use clap::ArgMatches;
use serde_json::{json, Map, Value};

use medh5::annotations::select::{analyse, cost_model, select_encoding};
use medh5::annotations::Annotation;
use medh5::h5::{data, ops};
use medh5::json::repr_str;
use medh5::sample::writer::amend;
use medh5::sample::writer_annotations::{annotation_to_masks, transcode};
use medh5::sample::{open_sample, Sample};
use medh5::Error;

use crate::common::*;

pub fn dispatch(name: &str, m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    if name == "seg" {
        return match m.subcommand() {
            Some(("stats", sub)) => stats(sub, ctx),
            Some(("convert", sub)) => convert(sub, ctx),
            _ => Ok(ctx.fail("usage: medh5 seg {stats|convert}")),
        };
    }
    match m.subcommand() {
        Some(("build", sub)) => index_build(sub, ctx),
        _ => Ok(ctx.fail("usage: medh5 index build PATH...")),
    }
}

fn open_voxel(path: &str, ann_id: &str) -> medh5::Result<(Sample, Arc<Annotation>)> {
    let sample = open_sample(Path::new(path))?;
    let annotation = sample.annotation(ann_id)?.clone();
    if !annotation.is_voxel() {
        return Err(Error::File(format!(
            "annotation {} has kind {}, which is not a voxel encoding",
            repr_str(ann_id),
            repr_str(annotation.kind())
        )));
    }
    Ok((sample, annotation))
}

/// `fail(str(exc))` for the errors the 1.x command caught by name.
fn caught(e: Error, ctx: &mut Ctx) -> CmdResult {
    if e.is_medh5() || matches!(e, Error::Key(_)) {
        Ok(ctx.fail(e.python_str()))
    } else {
        Err(e)
    }
}

fn stats(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let path = req_str(m, "path");
    let ann_id = req_str(m, "annotation");
    let (_sample, annotation) = match open_voxel(path, ann_id) {
        Ok(v) => v,
        Err(e) => return caught(e, ctx),
    };
    let masks = annotation_to_masks(&annotation, None)?;
    let shape = annotation.spatial_shape()?;
    let measured = analyse(&masks, Some(&shape))?;
    let costs = cost_model(&measured, false);
    let chosen = select_encoding(&measured, false, None, false);
    let cost_bytes: Vec<(&str, Option<u64>)> = vec![
        ("labelmap", costs.labelmap),
        ("layers", Some(costs.layers)),
        ("bitmask", Some(costs.bitmask)),
        ("instances", Some(costs.instances)),
        ("probmap", Some(costs.probmap)),
    ];
    if flag(m, "json") {
        let costs: Map<String, Value> = cost_bytes.iter().map(|(k, v)| (k.to_string(), json!(v))).collect();
        let payload = json!({
            "annotation": ann_id,
            "kind": annotation.kind(),
            "recommended": chosen,
            "stats": measured.summary(),
            "cost_bytes": costs,
        });
        ctx.emit(&payload, true);
        return Ok(EXIT_OK);
    }
    ctx.print(format!("{path} :: {ann_id}  (stored as `{}`)", annotation.kind()));
    ctx.print(format!(
        "  classes {}   voxels {}   fill {}   depth {}",
        measured.n_classes(),
        measured.n_voxels(),
        gp(measured.fill(), 3),
        gp(measured.depth(), 3)
    ));
    ctx.print(format!(
        "  overlap graph: {} edges, mean degree {:.2} -> {} layers, {} bitplanes",
        measured.edges.len(),
        measured.mean_degree(),
        measured.n_layers(),
        measured.n_planes()
    ));
    ctx.print("\nper-class voxel counts");
    let rows: Vec<Vec<String>> = measured
        .class_ids
        .iter()
        .map(|cid| {
            vec![
                cid.to_string(),
                annotation.class_key(*cid),
                measured.counts.get(cid).copied().unwrap_or(0).to_string(),
            ]
        })
        .collect();
    ctx.print(table(&rows, &["id", "key", "voxels"]));
    ctx.print("\nraw cost by encoding (pre-compression)");
    let rows: Vec<Vec<String>> = cost_bytes
        .iter()
        .map(|(name, value)| {
            let marker = if *name == annotation.kind() {
                "<- stored"
            } else if *name == chosen {
                "<- recommended"
            } else {
                ""
            };
            vec![name.to_string(), value.map(|v| human_bytes(v as f64)).unwrap_or_else(|| "-".into()), marker.into()]
        })
        .collect();
    ctx.print(table(&rows, &["encoding", "bytes", ""]));
    if chosen != annotation.kind() {
        ctx.print(format!("\n`medh5 seg convert {path} {ann_id} --to {chosen}` would re-encode it losslessly."));
    }
    Ok(EXIT_OK)
}

fn convert(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let path = req_str(m, "path");
    let ann_id = req_str(m, "annotation");
    let to = req_str(m, "to");
    let dry_run = flag(m, "dry_run");
    let drop_identity = flag(m, "drop_identity");
    let (sample, annotation) = match open_voxel(path, ann_id) {
        Ok(v) => v,
        Err(e) => return caught(e, ctx),
    };
    let measured = (|| -> medh5::Result<(usize, usize, String)> {
        let mut before = 0;
        for name in ops::members(&annotation.group)? {
            if let Some(ds) = ops::child_dataset(&annotation.group, &name) {
                before += data::nbytes(&ds)?;
            }
        }
        let payload = transcode(&annotation, to, drop_identity)?;
        Ok((before, payload.nbytes(), annotation.kind().to_string()))
    })();
    drop(annotation);
    drop(sample);
    let (before, after, source_kind) = match measured {
        Ok(v) => v,
        Err(e) if e.is_medh5() => return Ok(ctx.fail(e.to_string())),
        Err(e) => return Err(e),
    };
    if !dry_run {
        let applied = (|| -> medh5::Result<()> {
            let mut writer = amend(Path::new(path), None)?;
            writer.transcode_annotation(ann_id, to, None, drop_identity)?;
            writer.commit(true)?;
            Ok(())
        })();
        if let Err(e) = applied {
            if e.is_medh5() {
                return Ok(ctx.fail(e.to_string()));
            }
            return Err(e);
        }
    }
    if flag(m, "json") {
        let result = json!({
            "path": path,
            "annotation": ann_id,
            "from": source_kind,
            "to": to,
            "bytes_before": before,
            "bytes_after": after,
            "applied": !dry_run,
        });
        ctx.emit(&result, true);
    } else {
        let verb = if dry_run { "would re-encode" } else { "re-encoded" };
        ctx.print(format!(
            "{verb} {ann_id}: {source_kind} -> {to}  {} -> {} raw",
            human_bytes(before as f64),
            human_bytes(after as f64)
        ));
    }
    Ok(EXIT_OK)
}

fn index_build(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let as_json = flag(m, "json");
    let max_coords = get_i64(m, "max_coords").unwrap_or(0).max(0) as usize;
    let occupancy = get_i64(m, "occupancy").unwrap_or(0);
    // `--occupancy 0` turns the occupancy map off, as `occupancy or None` did.
    let occupancy = if occupancy == 0 { None } else { Some(occupancy.max(0) as usize) };
    let seed = get_i64(m, "seed").unwrap_or(0);
    let mut built = Map::new();
    for path in get_paths(m, "paths") {
        let names = (|| -> medh5::Result<Vec<String>> {
            let mut writer = amend(Path::new(&path), None)?;
            let names = writer.build_index(None, Some(max_coords), Some(occupancy), seed as u64)?;
            writer.commit(true)?;
            Ok(names)
        })();
        let names = match names {
            Ok(n) => n,
            Err(e) if e.is_medh5() => return Ok(ctx.fail(e.to_string())),
            Err(e) => return Err(e),
        };
        if !as_json {
            let listed = if names.is_empty() { "(nothing)".to_string() } else { names.join(", ") };
            ctx.print(format!("{path}: built index for {listed}"));
        }
        built.insert(path, json!(names));
    }
    let any = built.values().any(truthy);
    ctx.emit(&Value::Object(built), as_json);
    Ok(if any { EXIT_OK } else { EXIT_ERROR })
}
