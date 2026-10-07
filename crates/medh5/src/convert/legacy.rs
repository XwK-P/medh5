//! The 0.x on-disk layout, read-only, and its migration to 1.0 (Appendix B).
//!
//! ```text
//! /images/<name>   one dataset per modality, all the same shape
//! /seg/<name>      one boolean dataset per mask name
//! /bboxes          (n, ndim, 2) integers, slice-like [min, max)
//! /bbox_scores     (n,) floats            (optional)
//! /bbox_labels     (n,) strings           (optional)
//! root attrs       schema_version, label, label_name, extra (JSON), ...
//! /images attrs    shape, spacing, origin, direction (flattened), axis_labels,
//!                  coord_system, patch_size
//! ```
//!
//! The denormalised 0.x flags (`has_seg`, `seg_names`, `image_names`) are
//! ignored in favour of what the file actually contains.  Four migration
//! steps are not mechanical and each is reported: the voxel encoding, the
//! half-voxel box shift, the minted label set, and the grouping.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use ndarray::ArrayD;
use serde_json::{json, Map, Value};

use super::grouping::{
    group_by_subject, note_instance_ids, output_name, sanitize_key, sanitize_stem, Occasion, SubjectGroup,
};
use super::report::ConversionReport;
use crate::annotations::Assertions;
use crate::array::NdArray;
use crate::h5::attrs::{self, AttrValue};
use crate::h5::{data, ops};
use crate::json::repr_str;
use crate::labels::{ClassKey, LabelClass, LabelSet};
use crate::sample::{
    create, Annotated, AnnotationOptions, GridOptions, ImageOptions, ObjectFields, Placement, SegmentationOptions,
    SegmentationSource,
};
use crate::{Error, Result, VERSION};

/// The only 0.x schema version that ever shipped.
pub const SCHEMA_VERSION: &str = "1";
/// 0.x `[min, max)` integer boxes sit at voxel edges once shifted (§8.1).
pub const BOX_SHIFT: f64 = -0.5;

/// A 0.x sample-level label.
#[derive(Debug, Clone, PartialEq)]
pub enum LegacyLabel {
    Int(i64),
    Float(f64),
    Bool(bool),
    Text(String),
}

impl LegacyLabel {
    /// Python's `str()`.
    pub fn text(&self) -> String {
        match self {
            LegacyLabel::Int(i) => i.to_string(),
            LegacyLabel::Float(f) => crate::json::py_float(*f),
            LegacyLabel::Bool(b) => if *b { "True" } else { "False" }.into(),
            LegacyLabel::Text(t) => t.clone(),
        }
    }
}

/// 0.x geometry: one grid shared by every image in the file.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LegacySpatial {
    pub spacing: Option<Vec<f64>>,
    pub origin: Option<Vec<f64>>,
    pub direction: Option<Vec<Vec<f64>>>,
    pub axis_labels: Option<Vec<String>>,
    pub coord_system: Option<String>,
}

/// 0.x metadata, as far as a migration needs it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LegacyMeta {
    pub spatial: LegacySpatial,
    pub shape: Option<Vec<i64>>,
    pub image_names: Vec<String>,
    pub seg_names: Vec<String>,
    pub label: Option<LegacyLabel>,
    pub label_name: Option<String>,
    pub patch_size: Option<Vec<i64>>,
    pub extra: Map<String, Value>,
    pub schema_version: String,
}

/// A whole 0.x file in memory.
#[derive(Debug, Clone)]
pub struct LegacySample {
    pub images: IndexMap<String, NdArray>,
    pub seg: IndexMap<String, ArrayD<bool>>,
    pub bboxes: Option<ArrayD<f64>>,
    pub bbox_scores: Option<Vec<f64>>,
    pub bbox_labels: Option<Vec<String>>,
    pub meta: LegacyMeta,
}

fn open_legacy(path: &Path) -> Result<hdf5::File> {
    crate::h5::init();
    let text = path.to_string_lossy().into_owned();
    let handle = hdf5::File::open(path)
        .map_err(|e| Error::File(format!("cannot open {} as a 0.x file: {e}", repr_str(&text))))?;
    crate::h5::ops::check_self_contained(&handle, Some(path))?;
    if ops::exists(&handle, "meta") {
        return Err(Error::Schema(format!("{} is a 1.0 file (it has `/meta`), not a 0.x file", repr_str(&text))));
    }
    if ops::child_group(&handle, "images").is_none() {
        return Err(Error::Schema(format!("{} has no 0.x `/images` group", repr_str(&text))));
    }
    let version = attrs::get_str(&handle, "schema_version")?.unwrap_or_else(|| SCHEMA_VERSION.into());
    if version != SCHEMA_VERSION {
        return Err(Error::Schema(format!(
            "{} declares 0.x schema version {}; this reader understands {}",
            repr_str(&text),
            repr_str(&version),
            repr_str(SCHEMA_VERSION)
        )));
    }
    Ok(handle)
}

/// Whether `path` is a readable 0.x file.
pub fn is_legacy(path: &Path) -> bool {
    open_legacy(path).is_ok()
}

fn read_meta_from(handle: &hdf5::File) -> Result<LegacyMeta> {
    let mut meta = LegacyMeta::default();
    let group = handle.group("images")?;
    meta.image_names = ops::members(&group)?;
    let spatial = &mut meta.spatial;
    spatial.spacing = attrs::get_f64s(&group, "spacing")?;
    spatial.origin = attrs::get_f64s(&group, "origin")?;
    spatial.axis_labels = attrs::get_strs(&group, "axis_labels")?;
    spatial.coord_system = attrs::get_str(&group, "coord_system")?;
    meta.shape = attrs::get_i64s(&group, "shape")?;
    meta.patch_size = attrs::get_i64s(&group, "patch_size")?;
    if let (Some(raw), Some(first)) = (attrs::get_f64s(&group, "direction")?, meta.image_names.first()) {
        let ndim = group.dataset(first)?.ndim();
        if raw.len() != ndim * ndim {
            return Err(Error::Schema(format!(
                "0.x `direction` has {} element(s), not {} for a {ndim}-D volume",
                raw.len(),
                ndim * ndim
            )));
        }
        meta.spatial.direction = Some(raw.chunks(ndim.max(1)).map(<[f64]>::to_vec).collect());
    }
    meta.label = match attrs::read(handle, "label")? {
        None => None,
        Some(AttrValue::Int(i)) => Some(LegacyLabel::Int(i)),
        Some(AttrValue::Float(f)) => Some(LegacyLabel::Float(f)),
        Some(AttrValue::Bool(b)) => Some(LegacyLabel::Bool(b)),
        Some(other) => {
            let text = other.as_str().unwrap_or_else(|| attrs::stringify_value(&other));
            Some(match text.trim().parse::<i64>() {
                Ok(i) => LegacyLabel::Int(i),
                Err(_) => LegacyLabel::Text(text),
            })
        }
    };
    meta.label_name = attrs::get_str(handle, "label_name")?;
    if let Some(text) = attrs::get_str(handle, "extra")? {
        let parsed =
            crate::json::loads(&text).map_err(|e| Error::Schema(format!("0.x attribute 'extra' is not JSON: {e}")))?;
        if let Value::Object(m) = parsed {
            meta.extra = m;
        }
    }
    meta.schema_version = attrs::get_str(handle, "schema_version")?.unwrap_or_else(|| SCHEMA_VERSION.into());
    meta.seg_names = match ops::child_group(handle, "seg") {
        Some(g) => ops::members(&g)?,
        None => Vec::new(),
    };
    Ok(meta)
}

/// Read 0.x metadata without touching the arrays.
pub fn read_meta(path: &Path) -> Result<LegacyMeta> {
    read_meta_from(&open_legacy(path)?)
}

/// The `bbox_labels` of a 0.x file, without reading anything else.
pub fn read_bbox_labels(path: &Path) -> Result<Option<Vec<String>>> {
    let handle = open_legacy(path)?;
    match ops::child_dataset(&handle, "bbox_labels") {
        None => Ok(None),
        Some(ds) => Ok(Some(data::read_strings(&ds)?)),
    }
}

/// Read a whole 0.x file: images, masks, boxes and metadata.
pub fn read_sample(path: &Path) -> Result<LegacySample> {
    let handle = open_legacy(path)?;
    let group = handle.group("images")?;
    let mut images = IndexMap::new();
    for name in ops::members(&group)? {
        images.insert(name.clone(), data::read(&group.dataset(&name)?)?);
    }
    let mut seg = IndexMap::new();
    if let Some(g) = ops::child_group(&handle, "seg") {
        for name in ops::members(&g)? {
            seg.insert(name.clone(), data::read(&g.dataset(&name)?)?.nonzero_mask());
        }
    }
    let bboxes = match ops::child_dataset(&handle, "bboxes") {
        Some(ds) => Some(data::read(&ds)?.to_f64()),
        None => None,
    };
    let bbox_scores = match ops::child_dataset(&handle, "bbox_scores") {
        Some(ds) => Some(data::read(&ds)?.to_f64().iter().copied().collect()),
        None => None,
    };
    let bbox_labels = match ops::child_dataset(&handle, "bbox_labels") {
        Some(ds) => Some(data::read_strings(&ds)?),
        None => None,
    };
    let meta = read_meta_from(&handle)?;
    Ok(LegacySample { images, seg, bboxes, bbox_scores, bbox_labels, meta })
}

fn key(name: &str) -> String {
    sanitize_key(name, "class")
}

/// A sample id from a subject id: §2.3's identifier rule, lowercased.
pub fn sample_key(subject_id: &str) -> String {
    let stem = sanitize_stem(subject_id.trim(), 128);
    if stem.is_empty() {
        "sample".to_string()
    } else {
        stem.to_lowercase()
    }
}

/// Mint one label set covering a whole cohort's mask names and box labels.
///
/// Cohort-wide rather than per file, so `liver` has one id everywhere; ids an
/// `extra.nnunetv2.labels` mapping already fixed are reused.
pub fn build_label_set(paths: &[&Path], report: Option<&mut ConversionReport>) -> Result<LabelSet> {
    let mut names: Vec<String> = Vec::new();
    let mut reused: IndexMap<String, i64> = IndexMap::new();
    for path in paths {
        let Ok(meta) = read_meta(path) else { continue };
        for name in &meta.seg_names {
            if !names.contains(name) {
                names.push(name.clone());
            }
        }
        if let Some(Value::Object(labels)) = meta.extra.get("nnunetv2").and_then(|n| n.get("labels")) {
            for (label, value) in labels {
                if let Some(v) = value.as_i64().filter(|v| *v > 0 && value.is_i64()) {
                    reused.insert(key(label), v);
                }
            }
        }
        let has_boxes = open_legacy(path).map(|h| ops::exists(&h, "bboxes")).unwrap_or(false);
        match read_bbox_labels(path) {
            Ok(Some(labels)) => {
                for label in labels {
                    if !names.contains(&label) {
                        names.push(label);
                    }
                }
            }
            // Unlabelled 0.x boxes migrate as class `object`, which the label
            // set must therefore declare.
            Ok(None) if has_boxes && !names.iter().any(|n| n == "object") => {
                names.push("object".into());
            }
            _ => {}
        }
    }
    let mut classes = Vec::new();
    let mut used: BTreeSet<i64> = reused.values().copied().collect();
    let mut next_id = 1;
    for name in &names {
        let k = key(name);
        let class_id = match reused.get(&k) {
            Some(id) => *id,
            None => {
                while used.contains(&next_id) {
                    next_id += 1;
                }
                used.insert(next_id);
                next_id
            }
        };
        classes.push(LabelClass::new(class_id, k, name.clone())?);
    }
    if let Some(log) = report {
        let ids: Map<String, Value> = classes.iter().map(|c| (c.key.clone(), json!(c.id))).collect();
        let mut reused_keys: Vec<&String> = reused.keys().collect();
        reused_keys.sort();
        log.decision(
            "label_set",
            format!(
                "{} class(es) were minted across {} file(s); {}",
                classes.len(),
                paths.len(),
                if reused.is_empty() {
                    "no existing id mapping was found, so ids are sequential".to_string()
                } else {
                    format!("{} id(s) came from an existing extra.nnunetv2.labels mapping", reused.len())
                }
            ),
            json!({"ids": ids, "reused": reused_keys}),
        );
    }
    LabelSet::new("migrated", classes, "1.0.0", Vec::new(), Vec::new(), "inline", None, None)
}

/// Migrate one 0.x file into one 1.0 sample.
pub fn migrate(
    path: &Path,
    out: &Path,
    label_set: Option<&LabelSet>,
    codec: &str,
    report: Option<ConversionReport>,
) -> Result<ConversionReport> {
    let mut log = report.unwrap_or_else(|| ConversionReport::new("migrate", ""));
    log.source = path.to_string_lossy().into_owned();
    let labels = match label_set {
        Some(l) => l.clone(),
        None => build_label_set(&[path], Some(&mut log))?,
    };
    let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let group = SubjectGroup {
        subject_id: stem,
        occasions: vec![Occasion { key: log.source.clone(), payload: 0, ..Default::default() }],
        ordered_by: "given".into(),
    };
    write_group(&group, &[path.to_path_buf()], out, &labels, codec, &mut log)?;
    Ok(log)
}

fn date_of(meta: &LegacyMeta) -> Option<String> {
    for k in ["study_date", "date", "acquisition_date"] {
        if let Some(v) = meta.extra.get(k) {
            let truthy = match v {
                Value::Null => false,
                Value::Bool(b) => *b,
                Value::String(s) => !s.is_empty(),
                Value::Number(n) => n.as_f64() != Some(0.0),
                Value::Array(a) => !a.is_empty(),
                Value::Object(o) => !o.is_empty(),
            };
            if truthy {
                return Some(crate::json::py_str(v));
            }
        }
    }
    None
}

fn subject_key_of(meta: &LegacyMeta, path_key: Option<&str>) -> Option<String> {
    let path_key = path_key.filter(|k| !k.is_empty())?;
    let mut node = json!({"extra": meta.extra});
    for part in path_key.split('.') {
        node = node.as_object()?.get(part)?.clone();
    }
    match &node {
        Value::Null => None,
        Value::String(s) if s.is_empty() => None,
        other => Some(crate::json::py_str(other)),
    }
}

/// Migrate a cohort, minting one label set for all of it.
pub fn migrate_paths(
    paths: &[&Path],
    outdir: &Path,
    group_by: &str,
    subject_key: Option<&str>,
    label_set: Option<&LabelSet>,
    codec: &str,
) -> Result<ConversionReport> {
    std::fs::create_dir_all(outdir)?;
    let mut log = ConversionReport::new("migrate", &format!("{} file(s)", paths.len()));
    let labels = match label_set {
        Some(l) => l.clone(),
        None => build_label_set(paths, Some(&mut log))?,
    };
    let mut occasions = Vec::new();
    let mut sources = Vec::new();
    for path in paths {
        let text = path.to_string_lossy().into_owned();
        let meta = match read_meta(path) {
            Ok(m) => m,
            Err(e) => {
                log.warn("unreadable", format!("{text}: {e}"), json!({"path": text}));
                continue;
            }
        };
        let mtime = std::fs::metadata(path)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs_f64());
        occasions.push(Occasion {
            key: text,
            subject_id: subject_key_of(&meta, subject_key),
            date: date_of(&meta),
            order_hint: mtime,
            demographics: IndexMap::new(),
            payload: sources.len(),
        });
        sources.push(path.to_path_buf());
    }
    if group_by == "subject" && subject_key.is_none() {
        log.warn(
            "grouping",
            "--group-by subject needs --subject-key: a 0.x file has no subject field of its own, and identity is never \
             inferred from filenames",
            json!({}),
        );
    }
    let groups = group_by_subject(occasions, group_by, Some(&mut log))?;
    let mut used = BTreeSet::new();
    for group in groups {
        let name = output_name(&group, &mut used, &|s: &str| sample_key(s));
        let target = outdir.join(format!("{name}.medh5"));
        write_group(&group, &sources, &target, &labels, codec, &mut log)?;
    }
    Ok(log)
}

fn write_group(
    group: &SubjectGroup,
    sources: &[PathBuf],
    target: &Path,
    label_set: &LabelSet,
    codec: &str,
    log: &mut ConversionReport,
) -> Result<()> {
    note_instance_ids(group, log);
    if group.ordered_by == "order_hint" {
        log.guess(
            "timepoint_order",
            format!(
                "{}: timepoints were ordered by file mtime, which is a heuristic; supply dates in extra to make the order evidence",
                target.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
            ),
            json!({"order": group.occasions.iter().map(|o| o.key.clone()).collect::<Vec<_>>()}),
        );
    }
    let days = group.days_from_baseline();
    let mut writer = create(target, Some(&sample_key(&group.subject_id)), Some(&group.subject_id), codec, &[])?;
    let result = (|| -> Result<()> {
        writer.label_set(label_set.clone());
        let tool = writer.software("medh5", Some(VERSION), Map::new())?;
        for (index, d) in days.iter().enumerate() {
            let mut fields = Map::new();
            fields.insert("index".into(), json!(index));
            fields.insert("days_from_baseline".into(), json!(d));
            writer.add_timepoint(&format!("tp{index}"), fields)?;
        }
        let single = group.occasions.len() == 1;
        for (index, occasion) in group.occasions.iter().enumerate() {
            let source = &sources[occasion.payload];
            let sample = read_sample(source)?;
            migrate_one(
                &mut writer,
                &sample,
                &source.to_string_lossy(),
                &format!("tp{index}"),
                label_set,
                &tool.id,
                log,
                single,
            )?;
        }
        writer.commit(true)?;
        Ok(())
    })();
    if let Err(e) = result {
        writer.abort();
        return Err(e);
    }
    log.outputs.push(target.to_string_lossy().into_owned());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn migrate_one(
    writer: &mut crate::sample::SampleWriter,
    sample: &LegacySample,
    source: &str,
    timepoint: &str,
    label_set: &LabelSet,
    tool: &str,
    log: &mut ConversionReport,
    single: bool,
) -> Result<()> {
    let meta = &sample.meta;
    let suffix = if single { String::new() } else { format!("_{timepoint}") };
    let mut fields = Map::new();
    fields.insert("tool".into(), json!("medh5 migrate"));
    fields.insert("inputs".into(), json!([format!("medh5-0.x:{source}")]));
    let activity = writer.activity("import", Some(tool), None, fields)?;
    let grid_id = format!("ref{suffix}");
    let spatial = &meta.spatial;
    let (_, first) = sample.images.first().ok_or_else(|| Error::Schema(format!("{source}: 0.x file has no images")))?;
    let shape: Vec<i64> = first.shape().iter().map(|v| *v as i64).collect();
    let direction = match &spatial.direction {
        Some(rows) => {
            Some(ndarray::Array2::from_shape_vec((rows.len(), rows.len()), rows.iter().flatten().copied().collect())?)
        }
        None => None,
    };
    writer.add_grid(
        &grid_id,
        &shape,
        &spatial.spacing.clone().unwrap_or_else(|| vec![1.0; shape.len()]),
        GridOptions {
            origin: spatial.origin.clone(),
            direction,
            axis_names: spatial.axis_labels.clone(),
            coord_system: Some(spatial.coord_system.clone().unwrap_or_else(|| "LPS".into())),
            timepoint: Some(timepoint.into()),
            patch_hint: meta.patch_size.clone(),
            ..Default::default()
        },
    )?;
    let mut names: Vec<&String> = sample.images.keys().collect();
    names.sort();
    for name in names {
        writer.add_image(
            &format!("{name}{suffix}"),
            &sample.images[name],
            &grid_id,
            "OT",
            ImageOptions { prov: Some(activity.id.clone()), ..Default::default() },
        )?;
    }
    if !sample.seg.is_empty() {
        let mut seg_names: Vec<&String> = sample.seg.keys().collect();
        seg_names.sort();
        let mut masks = Vec::new();
        let mut annotated = Vec::new();
        for name in &seg_names {
            let id = label_set.lookup(&ClassKey::Key(key(name)))?.id;
            masks.push((ClassKey::Id(id), sample.seg[*name].clone()));
            annotated.push(ClassKey::Id(id));
        }
        let (kind, stats) = writer.add_segmentation(
            &format!("seg{suffix}"),
            &grid_id,
            SegmentationSource::Masks(masks),
            SegmentationOptions {
                common: AnnotationOptions {
                    annotated_classes: Annotated::Classes(annotated),
                    prov: Some(activity.id.clone()),
                    ..Default::default()
                },
                ..Default::default()
            },
        )?;
        log.decision(
            "encoding",
            format!("{source}: {} mask(s) were measured and stored as {}", seg_names.len(), repr_str(&kind)),
            json!({"source": source, "kind": kind, "overlapping_pairs": stats.map(|s| s.edges.len()).unwrap_or(0)}),
        );
        log.guess(
            "coverage",
            format!(
                "{source}: annotated_class_ids was set to the migrated mask names --- the only defensible inference. Widen \
                 or narrow it if the curator knows which classes were actually searched for (§11.3)"
            ),
            json!({"source": source, "classes": seg_names}),
        );
    }
    if let Some(boxes) = sample.bboxes.as_ref().filter(|b| b.shape().first().copied().unwrap_or(0) > 0) {
        let shifted = boxes.mapv(|v| v + BOX_SHIFT);
        let n = shifted.shape()[0];
        let labels = sample.bbox_labels.clone().unwrap_or_else(|| vec!["object".into(); n]);
        let class_ids: Vec<ClassKey> = labels
            .iter()
            .map(|l| Ok(ClassKey::Id(label_set.lookup(&ClassKey::Key(key(l)))?.id)))
            .collect::<Result<_>>()?;
        let stored = shifted.mapv(|v| f64::from(v as f32));
        writer.add_boxes(
            &format!("boxes{suffix}"),
            &stored,
            &class_ids,
            ObjectFields { scores: sample.bbox_scores.clone(), ..Default::default() },
            None,
            Placement { grid: Some(grid_id.clone()), space: Some("index".into()), frame_uid: None },
            AnnotationOptions { task: Some("detection".into()), prov: Some(activity.id.clone()), ..Default::default() },
        )?;
        log.decision(
            "box_convention",
            format!(
                "{source}: {n} box(es) were shifted by -0.5 on every axis --- 0.x stored slice-like [min, max) integers, \
                 1.0 stores voxel edges, and the numbers differ by half a voxel (§8.1)"
            ),
            json!({"source": source, "boxes": n, "shift": BOX_SHIFT}),
        );
    }
    if let Some(label) = &meta.label {
        let name = meta.label_name.clone().filter(|n| !n.is_empty()).unwrap_or_else(|| label.text());
        match label_set.get(&ClassKey::Key(key(&name))) {
            Some(entry) => {
                writer.add_classification(
                    &format!("label{suffix}"),
                    Assertions { class_ids: vec![entry.id], values: vec![1.0], ..Default::default() },
                    "sample",
                    true,
                    None,
                    AnnotationOptions {
                        timepoints: Some(vec![timepoint.into()]),
                        prov: Some(activity.id.clone()),
                        ..Default::default()
                    },
                )?;
            }
            None => {
                log.warn(
                    "label",
                    format!("{source}: sample label {} is not in the label set and was not migrated", repr_str(&name)),
                    json!({"source": source, "label": name}),
                );
            }
        }
    }
    if !meta.extra.is_empty() {
        writer.extra("legacy", Value::Object(meta.extra.clone()));
    }
    if let Some(Value::Object(review)) = meta.extra.get("review") {
        migrate_review(writer, review, &suffix, log, source)?;
    }
    Ok(())
}

fn timestamp(value: Option<&Value>) -> Option<String> {
    let text = match value? {
        Value::Null => return None,
        Value::String(s) if s.is_empty() => return None,
        other => crate::json::py_str(other),
    };
    if text.ends_with('Z') && text.contains('T') {
        return Some(text);
    }
    if text.chars().count() == 10 && text.chars().nth(4) == Some('-') {
        return Some(format!("{text}T00:00:00Z"));
    }
    None
}

fn migrate_review(
    writer: &mut crate::sample::SampleWriter,
    review: &Map<String, Value>,
    suffix: &str,
    log: &mut ConversionReport,
    source: &str,
) -> Result<()> {
    let reviewer =
        review.get("reviewer").or_else(|| review.get("by")).filter(|v| !v.is_null() && v.as_str() != Some(""));
    let agent = match reviewer {
        Some(r) => Some(writer.person(&crate::json::py_str(r), None, Map::new())?),
        None => None,
    };
    let status_raw = review.get("status").map(crate::json::py_str).unwrap_or_else(|| "reviewed".into());
    let mut fields = Map::new();
    if let Some(t) = timestamp(review.get("date").filter(|v| !v.is_null()).or_else(|| review.get("reviewed_at"))) {
        fields.insert("ended".into(), json!(t));
    }
    fields.insert("params".into(), json!({"verdict": status_raw}));
    fields.insert("outputs".into(), json!([format!("annotations/seg{suffix}")]));
    writer.activity("review", agent.as_ref().map(|a| a.id.as_str()), None, fields)?;
    let status = status_raw.to_lowercase();
    let allowed = ["draft", "submitted", "reviewed", "approved", "rejected", "deprecated"];
    let mut quality = Map::new();
    quality.insert(
        "status".into(),
        json!(if allowed.contains(&status.as_str()) { status.clone() } else { "reviewed".into() }),
    );
    quality.insert("reviewed_by".into(), json!(agent.as_ref().map(|a| vec![a.id.clone()]).unwrap_or_default()));
    writer.set_quality(&format!("seg{suffix}"), quality)?;
    log.decision(
        "review",
        format!(
            "{source}: extra.review became a `review` activity plus a quality record; 0.x kept review state in an ad-hoc \
             dict that could not say what produced the data being reviewed (§11.1)"
        ),
        json!({"source": source, "status": status}),
    );
    Ok(())
}

/// Write the minted label set for review before a cohort-wide migration.
pub fn write_sidecar(label_set: &LabelSet, path: &Path) -> Result<PathBuf> {
    let text = crate::json::dumps(&label_set.to_json(None), crate::json::Style::PYTHON.with_indent(Some(2)));
    std::fs::write(path, format!("{text}\n"))?;
    Ok(path.to_path_buf())
}

/// Read a label-set sidecar back.
pub fn load_sidecar(path: &Path) -> Result<LabelSet> {
    let text = std::fs::read_to_string(path)?;
    let doc = crate::json::loads(&text)?;
    LabelSet::from_json(Some(&doc))?
        .ok_or_else(|| Error::Value(format!("{} holds no label set", path.to_string_lossy())))
}
