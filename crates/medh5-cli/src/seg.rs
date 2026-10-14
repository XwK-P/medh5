//! `seg stats`, `seg convert`, `index build` --- the encoding tools.

use std::path::Path;
use std::sync::Arc;

use clap::ArgMatches;
use serde_json::{json, Map, Value};

use medh5::annotations::encode::IN_BAND_IGNORE_KINDS;
use medh5::annotations::select::{analyse, cost_model, select_encoding_checked};
use medh5::annotations::Annotation;
use medh5::h5::{attrs, data, ops};
use medh5::hdf5;
use medh5::integrity::group_digest;
use medh5::json::repr_str;
use medh5::ndarray::Axis;
use medh5::sample::writer::amend;
use medh5::sample::writer_annotations::{annotation_to_masks, transcode};
use medh5::sample::Sample;
use medh5::storage::index::{build_index, IndexPayload, SamplingIndex};
use medh5::Error;

use crate::common::*;

pub fn dispatch(name: &str, m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    if name == "seg" {
        return match m.subcommand() {
            Some(("stats", sub)) => stats(sub, ctx),
            Some(("convert", sub)) => convert(sub, ctx),
            _ => Ok(ctx.usage("usage: medh5 seg {stats|convert}")),
        };
    }
    match m.subcommand() {
        Some(("build", sub)) => index_build(sub, ctx),
        _ => Ok(ctx.usage("usage: medh5 index build PATH...")),
    }
}

fn open_voxel(path: &str, ann_id: &str) -> medh5::Result<(Sample, Arc<Annotation>)> {
    let sample = sample_file(path)?;
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

/// `medh5: why` for the errors the 1.x command caught by name, worded as
/// every command words them.
fn caught(e: Error, ctx: &mut Ctx) -> CmdResult {
    if e.is_medh5() || matches!(e, Error::Key(_)) {
        Ok(ctx.fail(top_level_message(&e)))
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
    let stored = annotation.kind();
    // What the auto-selector would write these masks as (§7.6), given what
    // the annotation must keep: a probability map stays soft, an in-band
    // ignore region is costed, and `instances` keeps the object identity a
    // dense encoding would drop (§7.4).
    let ignore = annotation.encodes_ignore()?;
    let costs = cost_model(&measured, ignore);
    let keep = (stored == "instances").then_some("instances");
    let chosen = select_encoding_checked(&measured, stored == "probmap", keep, ignore)?;
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
    if chosen == stored {
        return Ok(EXIT_OK);
    }
    if chosen == "instances" {
        ctx.print(
            "\n`instances` would cost least, but re-encoding cannot mint object identity (§7.4): write the objects \
             from their source to use it.",
        );
    } else if !ignore || IN_BAND_IGNORE_KINDS.contains(&chosen.as_str()) {
        ctx.print(format!("\n`medh5 seg convert {path} {ann_id} --to {chosen}` would re-encode it losslessly."));
    }
    Ok(EXIT_OK)
}

/// Raw bytes of an annotation's datasets: what `seg convert` reports.
fn raw_bytes(group: &hdf5::Group) -> medh5::Result<usize> {
    let mut total = 0;
    for name in ops::members(group)? {
        if let Some(ds) = ops::child_dataset(group, &name) {
            total += data::nbytes(&ds)?;
        }
    }
    Ok(total)
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
    let from = annotation.kind().to_string();
    // Transcoded once: measured from the payload on a dry run, from what the
    // writer wrote otherwise --- where it was transcoded once to measure it
    // and again to write it.
    let measured = (|| -> medh5::Result<(usize, Option<usize>)> {
        let before = raw_bytes(&annotation.group)?;
        if from == to {
            return Ok((before, Some(before)));
        }
        if dry_run {
            return Ok((before, Some(transcode(&annotation, to, drop_identity)?.nbytes())));
        }
        // What transcoding refuses by kind alone --- object identity minted,
        // or dropped unasked (§7.4) --- it refuses before reading a voxel:
        // refused here, the file is never copied.
        let identity = (from == "instances") != (to == "instances");
        if identity && !(from == "instances" && drop_identity) {
            transcode(&annotation, to, drop_identity)?;
        }
        Ok((before, None))
    })();
    drop(annotation);
    drop(sample);
    let (before, after) = match measured {
        Ok((before, Some(after))) => (before, after),
        Ok((before, None)) => {
            let written = (|| -> medh5::Result<usize> {
                let mut writer = amend(Path::new(path), None)?;
                writer.transcode_annotation(ann_id, to, None, drop_identity)?;
                let after = raw_bytes(&writer.root()?.group("annotations")?.group(ann_id)?)?;
                writer.commit(true)?;
                Ok(after)
            })();
            match written {
                Ok(after) => (before, after),
                Err(e) if e.is_medh5() => return Ok(ctx.fail(e.to_string())),
                Err(e) => return Err(e),
            }
        }
        Err(e) if e.is_medh5() => return Ok(ctx.fail(e.to_string())),
        Err(e) => return Err(e),
    };
    // Already so: nothing re-encoded, and the file left as it was.
    let applied = !dry_run && from != to;
    if flag(m, "json") {
        let result = json!({
            "path": path,
            "annotation": ann_id,
            "from": from,
            "to": to,
            "bytes_before": before,
            "bytes_after": after,
            "applied": applied,
        });
        ctx.emit(&result, true);
    } else if from == to {
        ctx.print(format!("{ann_id} is {to} already: nothing to re-encode, {path} is unchanged"));
    } else {
        let verb = if dry_run { "would re-encode" } else { "re-encoded" };
        ctx.print(format!(
            "{verb} {ann_id}: {from} -> {to}  {} -> {} raw",
            human_bytes(before as f64),
            human_bytes(after as f64)
        ));
    }
    Ok(EXIT_OK)
}

/// What `index build` found and did in one file.
struct Built {
    /// The annotations it indexes, in the file's order.
    indexable: Vec<String>,
    /// Those whose entry it wrote; the others already held what it would
    /// have written.
    built: Vec<String>,
}

fn index_build(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let as_json = flag(m, "json");
    // The parser holds both in range: at least one coordinate, a block of 0
    // or more --- and `--occupancy 0` builds no occupancy map.
    let max_coords = get_i64(m, "max_coords").unwrap_or(1) as usize;
    let occupancy = get_i64(m, "occupancy").filter(|&n| n > 0).map(|n| n as usize);
    let seed = get_i64(m, "seed").unwrap_or(0);
    let mut indexed = Map::new();
    let mut failed = false;
    for path in get_paths(m, "paths") {
        let done = match build_file(&path, max_coords, occupancy, seed) {
            Ok(done) => done,
            Err(e) => {
                // An entry of its own carrying the error, as every batch
                // command gives a file it could not process.
                indexed.insert(path.clone(), ctx.file_failed(&path, &e));
                failed = true;
                continue;
            }
        };
        if !as_json {
            let current: Vec<&str> =
                done.indexable.iter().filter(|name| !done.built.contains(name)).map(String::as_str).collect();
            let mut said = Vec::new();
            if !done.built.is_empty() {
                said.push(format!("built index for {}", done.built.join(", ")));
            }
            if !current.is_empty() {
                said.push(format!("index current for {}", current.join(", ")));
            }
            ctx.print(match (said.is_empty(), done.built.is_empty()) {
                // An index is optional per annotation (§14.3), and a mask has
                // no classes to index: a file without one to index is done.
                (true, _) => format!("{path}: nothing to index: no voxel annotation with classes"),
                (false, true) => format!("{path}: {}; the file is unchanged", said.join("; ")),
                (false, false) => format!("{path}: {}", said.join("; ")),
            });
        }
        // Every annotation with a current index now, as when each was rebuilt.
        indexed.insert(path, json!(done.indexable));
    }
    ctx.emit(&Value::Object(indexed), as_json);
    Ok(if failed { EXIT_ERROR } else { EXIT_OK })
}

/// Index the voxel annotations of `path` --- every one but a `mask`, as the
/// writer chooses them --- whose entry does not already hold what this build
/// writes.  The amend replaces the file, so one made for nothing costs a copy
/// of every image and a new inode under every open handle.
fn build_file(path: &str, max_coords: usize, occupancy: Option<usize>, seed: i64) -> medh5::Result<Built> {
    let (indexable, stale) = {
        let sample = sample_file(path)?;
        let entries = sample.index()?;
        let mut indexable = Vec::new();
        let mut stale = Vec::new();
        for (name, annotation) in sample.annotations()? {
            if !annotation.is_voxel() || annotation.kind() == "mask" {
                continue;
            }
            indexable.push(name.clone());
            let current = match entries.get(name) {
                Some(entry) => holds(entry, annotation, &sample.root, max_coords, occupancy, seed)?,
                None => false,
            };
            if !current {
                stale.push(name.clone());
            }
        }
        (indexable, stale)
    };
    if stale.is_empty() {
        return Ok(Built { indexable, built: Vec::new() });
    }
    let mut writer = amend(Path::new(path), None)?;
    let built = writer.build_index(Some(stale.as_slice()), Some(max_coords), Some(occupancy), seed as u64)?;
    writer.commit(true)?;
    Ok(Built { indexable, built })
}

/// Whether `entry` holds what a build with these parameters writes for
/// `annotation`: current (§13.3), drawn with this `max_coords` and `seed`,
/// and --- because the occupancy block is not stored, and two blocks can cut
/// the same coarse shape --- the tables a build computes now.  An entry that
/// cannot be read is rebuilt.
fn holds(
    entry: &SamplingIndex,
    annotation: &Annotation,
    root: &hdf5::Group,
    max_coords: usize,
    occupancy: Option<usize>,
    seed: i64,
) -> medh5::Result<bool> {
    // The digest the writer pins an entry to.
    let digest = group_digest(&annotation.group, root, "sha256")?;
    let drawn = |name: &str| attrs::get_i64(&entry.group, name).ok().flatten();
    let same_draw = entry.source_digest().ok().flatten().as_deref() == Some(digest.as_str())
        && drawn("max_coords") == Some(max_coords as i64)
        && drawn("seed") == Some(seed)
        && entry.has_occupancy().ok() == Some(occupancy.is_some());
    if !same_draw {
        return Ok(false);
    }
    let payload = build_index(annotation, None, max_coords, occupancy, seed as u64, Some(digest))?;
    Ok(same_tables(entry, &payload).unwrap_or(false))
}

/// Whether `entry` stores the tables of `payload`.
fn same_tables(entry: &SamplingIndex, payload: &IndexPayload) -> medh5::Result<bool> {
    let ids: Vec<i64> = payload.class_ids.iter().map(|c| i64::from(*c)).collect();
    if entry.class_ids()? != ids || entry.voxel_counts()?.values().copied().collect::<Vec<_>>() != payload.voxel_counts
    {
        return Ok(false);
    }
    for (position, class_id) in ids.iter().enumerate() {
        // NaN marks an empty class, which the entry reads as no box.
        let bounds = payload.class_bboxes.index_axis(Axis(0), position);
        let same_box = match entry.bbox(*class_id)? {
            None => bounds.iter().any(|v| v.is_nan()),
            Some(stored) => stored == bounds,
        };
        let same_occupancy = match (entry.occupancy_plane(position)?, &payload.occupancy) {
            (Some(stored), Some(planes)) => stored == planes.index_axis(Axis(0), position),
            (stored, planes) => stored.is_none() && planes.is_none(),
        };
        if !same_box || !same_occupancy || Some(&entry.coords(*class_id)?) != payload.fg_coords.get(class_id) {
            return Ok(false);
        }
    }
    Ok(true)
}
