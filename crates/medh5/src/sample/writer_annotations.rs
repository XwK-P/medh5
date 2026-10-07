//! The annotation, transform and index half of [`SampleWriter`].

use std::collections::BTreeMap;

use ndarray::ArrayD;
use serde_json::{Map, Value};

use super::writer::{Annotated, QualityArg, SampleWriter};
use crate::annotations::encode::{
    check_identity, check_target, encode_instances, encode_mask, encode_masks, encode_probmap, EncodeOptions,
    InstanceInput, IN_BAND_IGNORE_KINDS,
};
use crate::annotations::encode_geometric::{
    check_slice_index, check_space, encode_boxes, encode_classification, encode_contours, encode_keypoints,
    encode_mesh, encode_obb, encode_points, Assertions, ObjectColumns, Polygon,
};
use crate::annotations::header::{AnnotationHeader, VOXEL_KINDS};
use crate::annotations::payload::{normalize_masks, Masks, Payload, PayloadData};
use crate::annotations::select::{analyse, select_encoding, OverlapStats};
use crate::annotations::Annotation;
use crate::array::{DType, NdArray};
use crate::curation::quality::QualityRecord;
use crate::geometry::grid::Grid;
use crate::h5::attrs::AttrValue;
use crate::h5::{data, ops};
use crate::ids::validate_id;
use crate::integrity::group_digest;
use crate::json::{repr_int_list, repr_int_tuple, repr_list, repr_str};
use crate::labels::ClassKey;
use crate::storage::chunking::{field_chunks, fit_chunks};
use crate::storage::codecs::{dataset_layout, resolve_profile, Role};
use crate::storage::index::{build_index, write_index, DEFAULT_MAX_COORDS, DEFAULT_OCCUPANCY_FACTOR};
use crate::transforms::encode::{
    encode_affine, encode_bspline, encode_composite, encode_displacement, encode_identity,
};
use crate::transforms::model::{check_transform_id, TransformHeader, TRANSFORM_KINDS};
use crate::{Error, Result, VERSION};

/// What a segmentation is built from: exactly one of these.
#[derive(Debug, Clone)]
pub enum SegmentationSource {
    Masks(Vec<(ClassKey, ArrayD<bool>)>),
    Probabilities(Vec<(ClassKey, ArrayD<f64>)>),
    Instances(Vec<InstanceInput>),
}

/// Options shared by every annotation writer.
#[derive(Debug, Clone, Default)]
pub struct AnnotationOptions {
    pub annotated_classes: Annotated,
    pub closure: Option<String>,
    pub timepoints: Option<Vec<String>>,
    pub prov: Option<String>,
    pub quality: Option<QualityArg>,
    pub derived_from: Vec<String>,
    pub task: Option<String>,
    pub codec: Option<String>,
}

/// Options for [`SampleWriter::add_segmentation`].
#[derive(Debug, Clone, Default)]
pub struct SegmentationOptions {
    pub encoding: Option<String>,
    pub threshold: Option<f64>,
    pub ignore: Option<ArrayD<bool>>,
    pub ignore_mask: Option<String>,
    pub common: AnnotationOptions,
}

/// Where a geometric annotation's coordinates live.
#[derive(Debug, Clone, Default)]
pub struct Placement {
    pub grid: Option<String>,
    /// `index` (default for most kinds) or `world`.
    pub space: Option<String>,
    pub frame_uid: Option<String>,
}

/// The per-object columns callers give.
#[derive(Debug, Clone, Default)]
pub struct ObjectFields {
    pub instance_ids: Option<Vec<u64>>,
    pub scores: Option<Vec<f64>>,
    pub attributes: Option<Vec<Map<String, Value>>>,
}

/// The parameters of a transform (§10); `kind` decides which apply.
#[derive(Debug, Clone, Default)]
pub struct TransformSpec {
    pub matrix: Option<ArrayD<f64>>,
    pub field: Option<NdArray>,
    pub control_points: Option<ArrayD<f64>>,
    pub components: Option<Vec<String>>,
    pub field_grid: Option<String>,
    pub cp_grid: Option<String>,
    pub vector_space: Option<String>,
    pub interpolation: Option<String>,
    pub extrapolation: Option<String>,
    pub order: Option<i64>,
    pub units: Option<String>,
    pub from_grid: Option<String>,
    pub to_grid: Option<String>,
    pub invertible: Option<bool>,
    pub inverse_id: Option<String>,
    pub metrics: Option<QualityArg>,
    pub prov: Option<String>,
    pub codec: Option<String>,
}

impl SampleWriter {
    // -- class resolution ---------------------------------------------------------

    /// A class id from an id or a label-set key.
    pub fn class_id(&self, key: &ClassKey) -> Result<i64> {
        match key {
            ClassKey::Id(i) => Ok(*i),
            ClassKey::Key(k) => match &self.document.label_set {
                None => {
                    Err(Error::invalid(format!("cannot resolve class name {}: declare a label set first", repr_str(k))))
                }
                Some(ls) => Ok(ls.lookup(key)?.id),
            },
        }
    }

    fn class_ids_of(&self, keys: &[ClassKey]) -> Result<Vec<i64>> {
        keys.iter().map(|k| self.class_id(k)).collect()
    }

    fn label_set_ids(&self) -> Result<Vec<i64>> {
        match &self.document.label_set {
            None => Err(Error::invalid("annotated_classes='all' needs a declared label set")),
            Some(ls) => Ok(ls.ids()),
        }
    }

    /// The classes an explicit `annotated_classes` names (for zero masks).
    fn named_classes(&self, annotated: &Annotated) -> Result<Vec<i64>> {
        match annotated {
            Annotated::All => self.label_set_ids(),
            Annotated::AllGiven => Ok(Vec::new()),
            Annotated::Classes(keys) => self.class_ids_of(keys),
        }
    }

    fn resolve_annotated(&self, annotated: &Annotated, class_ids: &[i64]) -> Result<Vec<i64>> {
        match annotated {
            Annotated::AllGiven => Ok(class_ids.to_vec()),
            Annotated::All => self.label_set_ids(),
            Annotated::Classes(keys) => self.class_ids_of(keys),
        }
    }

    fn quality_key(&mut self, ann_id: &str, quality: Option<&QualityArg>) -> Result<Option<String>> {
        match quality {
            None => Ok(None),
            Some(QualityArg::Key(k)) => Ok(Some(k.clone())),
            Some(QualityArg::Record(fields)) => {
                let mut doc = Map::new();
                doc.insert("status".into(), Value::String("draft".into()));
                doc.extend(fields.clone());
                let record = QualityRecord::from_json(&Value::Object(doc))?;
                self.document.quality.insert(ann_id.to_string(), record);
                Ok(Some(ann_id.to_string()))
            }
        }
    }

    // -- voxel annotations ----------------------------------------------------------

    /// Write a voxel annotation, choosing the encoding by measurement.
    ///
    /// Returns the chosen `kind` and the overlap statistics behind the choice.
    /// An `ignore` region rides in band under `labelmap`/`layers` when it
    /// overlaps no class; otherwise it is written as the sibling `mask`
    /// annotation `<ann_id>_ignore`, named by `ignore_mask` (§7.7).
    pub fn add_segmentation(
        &mut self,
        ann_id: &str,
        grid: &str,
        source: SegmentationSource,
        options: SegmentationOptions,
    ) -> Result<(String, Option<OverlapStats>)> {
        let target = self.grid_ref(grid)?.clone();
        let encoding = options.encoding.clone().unwrap_or_else(|| "auto".into());
        let source_name = match &source {
            SegmentationSource::Masks(_) => "masks",
            SegmentationSource::Probabilities(_) => "probabilities",
            SegmentationSource::Instances(_) => "instances",
        };
        let implied = match &source {
            SegmentationSource::Probabilities(_) => Some("probmap"),
            SegmentationSource::Instances(_) => Some("instances"),
            SegmentationSource::Masks(_) => None,
        };
        if let Some(implied) = implied {
            if encoding != "auto" && encoding != implied {
                return Err(Error::coded(
                    "E404",
                    format!(
                        "annotation {}: encoding={} contradicts {source_name}=, which is stored as {}; pass encoding='auto' or {}",
                        repr_str(ann_id),
                        repr_str(&encoding),
                        repr_str(implied),
                        repr_str(implied)
                    ),
                ));
            }
        }
        if options.threshold.is_some() && !matches!(source, SegmentationSource::Probabilities(_)) {
            return Err(Error::coded(
                "E404",
                format!("annotation {}: threshold= applies to probabilities= only", repr_str(ann_id)),
            ));
        }
        if options.ignore.is_some() && options.ignore_mask.is_some() {
            return Err(Error::coded(
                "E404",
                format!(
                    "annotation {}: pass ignore= (an array this writer stores) or ignore_mask= (a `mask` annotation you wrote), not both",
                    repr_str(ann_id)
                ),
            ));
        }
        let spatial = target.spatial_shape();
        if let Some(ignore) = &options.ignore {
            if ignore.shape() != spatial.as_slice() {
                return Err(Error::coded(
                    "E405",
                    format!(
                        "annotation {}: ignore has shape {}; grid {} spatial shape is {}",
                        repr_str(ann_id),
                        repr_int_tuple(ignore.shape()),
                        repr_str(grid),
                        repr_int_tuple(&spatial)
                    ),
                ));
            }
        }
        let examined = self.named_classes(&options.common.annotated_classes)?;
        let (payload, stats, class_ids, in_band) = self.encode_segmentation(
            source,
            &target,
            &encoding,
            options.ignore.as_ref(),
            &examined,
            options.threshold,
        )?;
        let mut ignore_mask = options.ignore_mask.clone();
        let mut sibling = None;
        if options.ignore.is_some() && !in_band {
            let name = format!("{ann_id}_ignore");
            validate_id(&name, "annotation id")?;
            let anns = self.root()?.group("annotations")?;
            for taken in [ann_id, name.as_str()] {
                if ops::exists(&anns, taken) {
                    return Err(Error::invalid(format!("annotation {} already exists", repr_str(taken))));
                }
            }
            ignore_mask = Some(name.clone());
            sibling = Some(name);
        }
        let annotated = self.resolve_annotated(&options.common.annotated_classes, &class_ids)?;
        let mut header = AnnotationHeader::new(
            payload.kind.clone(),
            options.common.task.clone().unwrap_or_else(|| "segmentation".into()),
        );
        header.grid = Some(grid.to_string());
        header.timepoints = options.common.timepoints.clone();
        header.class_ids = class_ids;
        header.annotated_class_ids = annotated;
        header.closure = options.common.closure.clone().unwrap_or_else(|| "explicit".into());
        header.ignore_mask = ignore_mask;
        header.prov = options.common.prov.clone();
        header.quality = self.quality_key(ann_id, options.common.quality.as_ref())?;
        header.derived_from = options.common.derived_from.clone();
        header.extra = payload.attrs.clone();
        header.check()?;
        self.write_annotation(ann_id, &header, &payload, Some(&target), options.common.codec.as_deref())?;
        if let (Some(name), Some(ignore)) = (sibling, options.ignore) {
            self.add_mask(
                &name,
                ignore,
                grid,
                "other",
                options.common.prov.as_deref(),
                options.common.codec.as_deref(),
            )?;
        }
        Ok((payload.kind, stats))
    }

    fn encode_segmentation(
        &self,
        source: SegmentationSource,
        grid: &Grid,
        encoding: &str,
        ignore: Option<&ArrayD<bool>>,
        examined: &[i64],
        threshold: Option<f64>,
    ) -> Result<(Payload, Option<OverlapStats>, Vec<i64>, bool)> {
        let spatial = grid.spatial_shape();
        match source {
            SegmentationSource::Instances(objects) => {
                if objects.is_empty() && examined.is_empty() {
                    return Err(Error::coded(
                        "E410",
                        "instances=[] records 'examined, none found' only with annotated_classes= naming the classes that \
                         were examined; without it the annotation says nothing (§7.4)",
                    ));
                }
                let payload = encode_instances(
                    &objects,
                    Some(&spatial),
                    true,
                    if examined.is_empty() { None } else { Some(examined) },
                )?;
                let ids = payload.class_ids.clone();
                Ok((payload, None, ids, false))
            }
            SegmentationSource::Probabilities(planes) => {
                let mut resolved: BTreeMap<i64, ArrayD<f64>> = BTreeMap::new();
                for (k, v) in planes {
                    resolved.insert(self.class_id(&k)?, v);
                }
                for c in examined {
                    resolved.entry(*c).or_insert_with(|| ArrayD::zeros(ndarray::IxDyn(&spatial)));
                }
                let payload = encode_probmap(&resolved, Some(&spatial), DType::F16, false, threshold)?;
                let ids = payload.class_ids.clone();
                Ok((payload, None, ids, false))
            }
            SegmentationSource::Masks(masks) => {
                let mut given: Masks = Masks::new();
                for (k, v) in masks {
                    given.insert(self.class_id(&k)?, v);
                }
                for c in examined {
                    given.entry(*c).or_insert_with(|| ArrayD::from_elem(ndarray::IxDyn(&spatial), false));
                }
                let shape = normalize_masks(&given, Some(&spatial))?;
                let overlaps = match ignore {
                    Some(ig) => {
                        given.values().any(|m| ndarray::Zip::from(m).and(ig).fold(false, |acc, a, b| acc || (*a && *b)))
                    }
                    None => false,
                };
                let in_band_possible = ignore.is_some() && !overlaps;
                let stats = analyse(&given, Some(&shape))?;
                let prefer = if encoding == "auto" { None } else { Some(encoding) };
                let kind = select_encoding(&stats, false, prefer, in_band_possible);
                let in_band = in_band_possible && IN_BAND_IGNORE_KINDS.contains(&kind.as_str());
                let options =
                    EncodeOptions { ignore: if in_band { ignore.cloned() } else { None }, ..Default::default() };
                let payload = encode_masks(&given, &kind, Some(&shape), &options)?;
                let ids = payload.class_ids.clone();
                Ok((payload, Some(stats), ids, in_band))
            }
        }
    }

    /// Write a boolean `mask` annotation (FOV, ignore region).
    pub fn add_mask(
        &mut self,
        ann_id: &str,
        mask: ArrayD<bool>,
        grid: &str,
        task: &str,
        prov: Option<&str>,
        codec: Option<&str>,
    ) -> Result<()> {
        let target = self.grid_ref(grid)?.clone();
        let spatial = target.spatial_shape();
        if mask.shape() != spatial.as_slice() {
            return Err(Error::coded(
                "E405",
                format!(
                    "mask {} has shape {}; grid {} spatial shape is {}",
                    repr_str(ann_id),
                    repr_int_tuple(mask.shape()),
                    repr_str(grid),
                    repr_int_tuple(&spatial)
                ),
            ));
        }
        let payload = encode_mask(mask);
        let mut header = AnnotationHeader::new("mask", task);
        header.grid = Some(grid.to_string());
        header.prov = prov.map(str::to_string);
        header.check()?;
        self.write_annotation(ann_id, &header, &payload, Some(&target), codec)?;
        Ok(())
    }

    pub(crate) fn write_annotation(
        &mut self,
        ann_id: &str,
        header: &AnnotationHeader,
        payload: &Payload,
        grid: Option<&Grid>,
        codec: Option<&str>,
    ) -> Result<hdf5::Group> {
        validate_id(ann_id, "annotation id")?;
        let node = self.root()?.group("annotations")?;
        if ops::exists(&node, ann_id) {
            return Err(Error::invalid(format!("annotation {} already exists", repr_str(ann_id))));
        }
        let group = node.create_group(ann_id)?;
        let profile = resolve_profile(Some(codec.unwrap_or(&self.codec)))?;
        for (name, dataset) in &payload.datasets {
            match dataset {
                PayloadData::Strings(values) => {
                    data::create_strings(&group, name, values)?;
                }
                PayloadData::Array(array) => {
                    let mut chunks = None;
                    if let Some(g) = grid {
                        if name == "data" && array.ndim() >= g.n_spatial() {
                            let proposed = self.chunks_for(g, array.dtype().itemsize(), payload.stacked_axes)?;
                            chunks = fit_chunks(&proposed, &array.shape());
                        }
                    }
                    let layout =
                        dataset_layout(&array.shape(), array.dtype().itemsize(), &profile, Role::Label, chunks);
                    data::create(&group, name, array, &layout)?;
                }
            }
        }
        for (k, v) in header.attrs() {
            crate::h5::attrs::write(&group, &k, &v)?;
        }
        self.annotation_kinds.insert(ann_id.to_string(), header.kind.clone());
        Ok(group)
    }

    /// Drop an annotation, and any index entry derived from it.
    pub fn remove_annotation(&mut self, ann_id: &str) -> Result<()> {
        let root = self.root()?;
        let node = root.group("annotations")?;
        if !ops::exists(&node, ann_id) {
            return Err(Error::invalid(format!("no annotation {} to remove", repr_str(ann_id))));
        }
        ops::unlink(&node, ann_id)?;
        self.annotation_kinds.shift_remove(ann_id);
        if let Some(index) = ops::child_group(&root, "index") {
            ops::unlink(&index, ann_id)?;
        }
        Ok(())
    }

    /// Re-encode a voxel annotation in place, preserving its header (§7.6).
    ///
    /// `instances` to a dense encoding loses every `instance_id` and is
    /// refused unless `drop_identity`; the loss is then recorded as a
    /// `transcode` activity.
    pub fn transcode_annotation(
        &mut self,
        ann_id: &str,
        to_kind: &str,
        codec: Option<&str>,
        drop_identity: bool,
    ) -> Result<String> {
        let node = self.root()?.group("annotations")?;
        let Some(group) = ops::child_group(&node, ann_id) else {
            return Err(Error::invalid(format!("no annotation {} to transcode", repr_str(ann_id))));
        };
        let mut header = AnnotationHeader::read(&group)?;
        let grids = std::sync::Arc::new(self.grids.clone());
        let label_set = self.document.label_set.clone().map(std::sync::Arc::new);
        let annotation = Annotation::open(ann_id, group, grids, label_set)?;
        if !annotation.is_voxel() {
            return Err(Error::invalid(format!(
                "annotation {} of kind {} is not a voxel encoding",
                repr_str(ann_id),
                repr_str(&header.kind)
            )));
        }
        if header.kind == to_kind {
            return Ok(to_kind.to_string());
        }
        let payload = transcode(&annotation, to_kind, drop_identity)?;
        let dropped = header.kind == "instances";
        let objects = if dropped { annotation.n_objects()? } else { 0 };
        drop(annotation);
        let grid = self.grid_ref(header.grid.as_deref().unwrap_or(""))?.clone();
        self.remove_annotation(ann_id)?;
        header.kind = to_kind.to_string();
        header.class_ids = payload.class_ids.clone();
        header.annotated_class_ids.retain(|c| payload.class_ids.contains(c));
        for (k, v) in &payload.attrs {
            if let Some(slot) = header.extra.iter_mut().find(|(name, _)| name == k) {
                slot.1 = v.clone();
            } else {
                header.extra.push((k.clone(), v.clone()));
            }
        }
        self.write_annotation(ann_id, &header, &payload, Some(&grid), codec)?;
        if dropped {
            let tool = self.software("medh5", Some(VERSION), Map::new())?;
            let mut fields = Map::new();
            fields.insert("tool".into(), Value::String(format!("medh5 transcode --to {to_kind} --drop-identity")));
            fields.insert("inputs".into(), serde_json::json!([format!("annotations/{ann_id}")]));
            fields.insert("outputs".into(), serde_json::json!([format!("annotations/{ann_id}")]));
            fields.insert(
                "params".into(),
                serde_json::json!({"from": "instances", "to": to_kind, "dropped": "instance identity", "objects": objects}),
            );
            self.activity("transcode", Some(&tool.id), None, fields)?;
        }
        Ok(to_kind.to_string())
    }

    // -- geometric and classification annotations (§8, §9) ------------------------

    #[allow(clippy::too_many_arguments)]
    fn add_object_annotation(
        &mut self,
        ann_id: &str,
        payload: Payload,
        placement: &Placement,
        space: Option<&str>,
        options: &AnnotationOptions,
        default_task: &str,
        extra: Vec<(String, AttrValue)>,
    ) -> Result<hdf5::Group> {
        let target = match &placement.grid {
            Some(g) => Some(self.grid_ref(g)?.clone()),
            None => None,
        };
        if let Some(s) = space {
            check_space(s)?;
        }
        if space == Some("index") && target.is_none() {
            return Err(Error::coded(
                "E412",
                format!(
                    "annotation {}: space='index' names a grid's coordinates, so `grid` is required",
                    repr_str(ann_id)
                ),
            ));
        }
        let mut frame = placement.frame_uid.clone();
        if space == Some("world") {
            if frame.is_none() {
                frame = target.as_ref().and_then(|t| t.frame_uid.clone());
            }
            if frame.is_none() {
                return Err(Error::coded(
                    "E412",
                    format!(
                        "annotation {}: space='world' names a physical frame, so `frame_uid` is required (directly or via the grid)",
                        repr_str(ann_id)
                    ),
                ));
            }
        }
        if let Some(t) = &target {
            if t.units == "px" && space != Some("index") {
                return Err(Error::coded(
                    "E414",
                    format!(
                        "annotation {}: grid {} is uncalibrated (units='px'), so geometric annotations on it must use space='index'",
                        repr_str(ann_id),
                        repr_str(&t.grid_id)
                    ),
                ));
            }
        }
        let annotated = self.resolve_annotated(&options.annotated_classes, &payload.class_ids)?;
        let mut declared: Vec<i64> = payload.class_ids.iter().chain(&annotated).copied().collect();
        declared.sort_unstable();
        declared.dedup();
        let mut header =
            AnnotationHeader::new(payload.kind.clone(), options.task.clone().unwrap_or_else(|| default_task.into()));
        header.grid = placement.grid.clone();
        header.timepoints = options.timepoints.clone();
        header.space = space.map(str::to_string);
        header.frame_uid = frame;
        header.class_ids = declared;
        header.annotated_class_ids = annotated;
        header.closure = options.closure.clone().unwrap_or_else(|| "explicit".into());
        header.prov = options.prov.clone();
        header.quality = self.quality_key(ann_id, options.quality.as_ref())?;
        header.derived_from = options.derived_from.clone();
        header.extra = payload.attrs.clone();
        header.extra.extend(extra);
        header.check()?;
        self.write_annotation(ann_id, &header, &payload, target.as_ref(), options.codec.as_deref())
    }

    /// Write axis-aligned boxes as `(N, S, 2)` float `[lo, hi]` (§8.2).
    #[allow(clippy::too_many_arguments)]
    pub fn add_boxes(
        &mut self,
        ann_id: &str,
        boxes: &ArrayD<f64>,
        class_ids: &[ClassKey],
        objects: ObjectFields,
        slice_index: Option<Vec<i64>>,
        placement: Placement,
        options: AnnotationOptions,
    ) -> Result<hdf5::Group> {
        let ids = self.class_ids_of(class_ids)?;
        let cols = ObjectColumns {
            class_ids: &ids,
            instance_ids: objects.instance_ids.as_deref(),
            scores: objects.scores.as_deref(),
            attributes: objects.attributes.as_deref(),
        };
        let payload = encode_boxes(boxes, &cols, slice_index.as_deref())?;
        let space = placement.space.clone().unwrap_or_else(|| "index".into());
        if let (Some(planes), "index", Some(grid)) = (&slice_index, space.as_str(), &placement.grid) {
            if let Some(known) = self.grids.get(grid) {
                let stored = payload.array("boxes")?.to_f64();
                let n = stored.shape().first().copied().unwrap_or(0);
                let rows: Vec<Vec<f64>> =
                    (0..n).map(|i| stored.index_axis(ndarray::Axis(0), i).iter().copied().collect()).collect();
                if let Some(problem) = check_slice_index(planes, n, Some(&rows), Some(&known.spatial_shape())) {
                    return Err(Error::coded("E405", format!("{}: {problem}", repr_str(ann_id))));
                }
            }
        }
        self.add_object_annotation(ann_id, payload, &placement, Some(&space), &options, "detection", Vec::new())
    }

    /// Write oriented boxes: centre, full edge lengths, rotation (§8.3).
    #[allow(clippy::too_many_arguments)]
    pub fn add_obb(
        &mut self,
        ann_id: &str,
        centers: &ArrayD<f64>,
        sizes: &ArrayD<f64>,
        rotations: &ArrayD<f64>,
        class_ids: &[ClassKey],
        objects: ObjectFields,
        placement: Placement,
        options: AnnotationOptions,
    ) -> Result<hdf5::Group> {
        let ids = self.class_ids_of(class_ids)?;
        let cols = ObjectColumns {
            class_ids: &ids,
            instance_ids: objects.instance_ids.as_deref(),
            scores: objects.scores.as_deref(),
            attributes: objects.attributes.as_deref(),
        };
        let payload = encode_obb(centers, sizes, rotations, &cols)?;
        let space = placement.space.clone().unwrap_or_else(|| "index".into());
        self.add_object_annotation(ann_id, payload, &placement, Some(&space), &options, "detection", Vec::new())
    }

    /// Write `(N, K, S)` keypoints with per-slot classes (§8.4).
    #[allow(clippy::too_many_arguments)]
    pub fn add_keypoints(
        &mut self,
        ann_id: &str,
        points: &ArrayD<f64>,
        keypoint_classes: &[ClassKey],
        class_ids: &[ClassKey],
        visibility: Option<&ArrayD<i64>>,
        objects: ObjectFields,
        skeleton: Option<&str>,
        placement: Placement,
        options: AnnotationOptions,
    ) -> Result<hdf5::Group> {
        if let Some(sk) = skeleton {
            let declared = self.document.label_set.as_ref().is_some_and(|ls| ls.skeletons.iter().any(|s| s.id == sk));
            if !declared {
                return Err(Error::coded(
                    "E413",
                    format!(
                        "annotation {} names skeleton {}, which the label set does not declare",
                        repr_str(ann_id),
                        repr_str(sk)
                    ),
                ));
            }
        }
        let kp = self.class_ids_of(keypoint_classes)?;
        let ids = self.class_ids_of(class_ids)?;
        let cols = ObjectColumns {
            class_ids: &ids,
            instance_ids: objects.instance_ids.as_deref(),
            scores: objects.scores.as_deref(),
            attributes: None,
        };
        let payload = encode_keypoints(points, &kp, &cols, visibility, skeleton)?;
        let space = placement.space.clone().unwrap_or_else(|| "index".into());
        self.add_object_annotation(ann_id, payload, &placement, Some(&space), &options, "detection", Vec::new())
    }

    /// Write a point set: landmarks, seeds, or half a correspondence (§8.5).
    #[allow(clippy::too_many_arguments)]
    pub fn add_points(
        &mut self,
        ann_id: &str,
        points: &ArrayD<f64>,
        class_ids: Option<&[ClassKey]>,
        names: Option<&[String]>,
        weights: Option<&[f64]>,
        correspondence: Option<&str>,
        placement: Placement,
        options: AnnotationOptions,
    ) -> Result<hdf5::Group> {
        let ids = match class_ids {
            Some(keys) => Some(self.class_ids_of(keys)?),
            None => None,
        };
        let payload = encode_points(points, ids.as_deref(), names, weights, correspondence)?;
        let space = placement.space.clone().unwrap_or_else(|| "index".into());
        self.add_object_annotation(ann_id, payload, &placement, Some(&space), &options, "detection", Vec::new())
    }

    /// Write planar polygons (§8.6) --- the RTSTRUCT-shaped annotation.
    pub fn add_contours(
        &mut self,
        ann_id: &str,
        polygons: &[Polygon],
        placement: Placement,
        options: AnnotationOptions,
    ) -> Result<hdf5::Group> {
        let ndim = placement.grid.as_ref().and_then(|g| self.grids.get(g)).map(Grid::n_spatial);
        let payload = encode_contours(polygons, ndim)?;
        let space = placement.space.clone().unwrap_or_else(|| "index".into());
        self.add_object_annotation(ann_id, payload, &placement, Some(&space), &options, "segmentation", Vec::new())
    }

    /// Write a triangle surface mesh (§8.7); `space` defaults to `world`.
    #[allow(clippy::too_many_arguments)]
    pub fn add_mesh(
        &mut self,
        ann_id: &str,
        vertices: &ArrayD<f64>,
        faces: &ArrayD<i64>,
        normals: Option<&ArrayD<f64>>,
        vertex_class_ids: Option<&[ClassKey]>,
        mesh_offsets: Option<&[i64]>,
        mesh_class_ids: Option<&[ClassKey]>,
        placement: Placement,
        options: AnnotationOptions,
    ) -> Result<hdf5::Group> {
        let vertex_ids = match vertex_class_ids {
            Some(k) => Some(self.class_ids_of(k)?),
            None => None,
        };
        let mesh_ids = match mesh_class_ids {
            Some(k) => Some(self.class_ids_of(k)?),
            None => None,
        };
        let payload = encode_mesh(vertices, faces, normals, vertex_ids.as_deref(), mesh_offsets, mesh_ids.as_deref())?;
        let space = placement.space.clone().unwrap_or_else(|| "world".into());
        self.add_object_annotation(ann_id, payload, &placement, Some(&space), &options, "segmentation", Vec::new())
    }

    /// Write a classification annotation (§9).  A change label is
    /// `scope = "sample"` with explicit `timepoints`.
    pub fn add_classification(
        &mut self,
        ann_id: &str,
        assertions: Assertions,
        scope: &str,
        multilabel: bool,
        grid: Option<String>,
        options: AnnotationOptions,
    ) -> Result<hdf5::Group> {
        let payload = encode_classification(&assertions, scope, multilabel)?;
        if scope == "timepoint" {
            if let Some(units) = &assertions.scope_ids {
                let declared = self.document.timepoints.len() as i64;
                let mut unknown: Vec<i64> = units.iter().copied().filter(|v| !(0..declared).contains(v)).collect();
                unknown.sort_unstable();
                unknown.dedup();
                if !unknown.is_empty() {
                    return Err(Error::coded(
                        "E409",
                        format!(
                            "annotation {}: scope='timepoint' scope_ids {} are not timepoint indices (0..{})",
                            repr_str(ann_id),
                            repr_int_list(&unknown),
                            declared - 1
                        ),
                    ));
                }
            }
        }
        let mut options = options;
        options.task = Some("classification".into());
        let placement = Placement { grid, space: None, frame_uid: None };
        self.add_object_annotation(ann_id, payload, &placement, None, &options, "classification", Vec::new())
    }

    // -- transforms (§10) ---------------------------------------------------------------

    /// Write a transform mapping points from `from_frame` to `to_frame`:
    /// `x_M = T(x_F)`, the ITK convention, with no attribute to switch it.
    pub fn add_transform(
        &mut self,
        transform_id: &str,
        kind: &str,
        from_frame: &str,
        to_frame: &str,
        spec: TransformSpec,
    ) -> Result<hdf5::Group> {
        check_transform_id(transform_id)?;
        if from_frame == to_frame {
            return Err(Error::coded(
                "E502",
                format!(
                    "transform {} maps {} to itself; grids that share a frame need no transform (§3.4)",
                    repr_str(transform_id),
                    repr_str(from_frame)
                ),
            ));
        }
        let payload = self.encode_transform(transform_id, kind, &spec, from_frame)?;
        let mut header = TransformHeader::new(kind, from_frame, to_frame)?;
        header.units = spec.units.clone().unwrap_or_else(|| "mm".into());
        header.from_grid = spec.from_grid.clone();
        header.to_grid = spec.to_grid.clone();
        header.invertible = spec.invertible;
        header.inverse_id = spec.inverse_id.clone();
        header.prov = spec.prov.clone();
        header.metrics = self.quality_key(transform_id, spec.metrics.as_ref())?;
        header.extra = payload.attrs.clone();
        let root = self.root()?;
        let node = match ops::child_group(&root, "transforms") {
            Some(g) => g,
            None => root.create_group("transforms")?,
        };
        if ops::exists(&node, transform_id) {
            return Err(Error::invalid(format!("transform {} already exists", repr_str(transform_id))));
        }
        let group = node.create_group(transform_id)?;
        let profile = resolve_profile(Some(spec.codec.as_deref().unwrap_or(&self.codec)))?;
        for (name, dataset) in &payload.datasets {
            let PayloadData::Array(array) = dataset else { continue };
            let mut chunks = None;
            if name == "field" || name == "control_points" {
                if let Some(g) = spec.field_grid.as_ref().and_then(|g| self.grids.get(g)) {
                    chunks = field_chunks(g, &array.shape(), array.dtype().itemsize())?;
                }
            }
            let layout = dataset_layout(&array.shape(), array.dtype().itemsize(), &profile, Role::Label, chunks);
            data::create(&group, name, array, &layout)?;
        }
        for (k, v) in header.attrs() {
            crate::h5::attrs::write(&group, &k, &v)?;
        }
        self.transform_frames.insert(transform_id.to_string(), (from_frame.to_string(), to_frame.to_string()));
        Ok(group)
    }

    fn encode_transform(
        &self,
        transform_id: &str,
        kind: &str,
        spec: &TransformSpec,
        from_frame: &str,
    ) -> Result<Payload> {
        let id = repr_str(transform_id);
        match kind {
            "identity" => Ok(encode_identity()),
            "affine" => match &spec.matrix {
                None => Err(Error::coded("E502", format!("transform {id}: kind 'affine' needs `matrix`"))),
                Some(m) => encode_affine(m),
            },
            "displacement" => {
                let (Some(field), Some(field_grid)) = (&spec.field, &spec.field_grid) else {
                    return Err(Error::coded(
                        "E503",
                        format!("transform {id}: kind 'displacement' needs `field` and `field_grid`"),
                    ));
                };
                self.check_field_frame(transform_id, field_grid, from_frame)?;
                let payload = encode_displacement(
                    field,
                    field_grid,
                    spec.vector_space.as_deref().unwrap_or("world"),
                    spec.interpolation.as_deref().unwrap_or("linear"),
                    spec.extrapolation.as_deref().unwrap_or("zero"),
                    DType::F32,
                )?;
                self.check_field_lattice(transform_id, &payload.array("field")?.shape(), field_grid, true)?;
                Ok(payload)
            }
            "bspline" => {
                let (Some(cp), Some(cp_grid)) = (&spec.control_points, &spec.cp_grid) else {
                    return Err(Error::coded(
                        "E503",
                        format!("transform {id}: kind 'bspline' needs `control_points` and `cp_grid`"),
                    ));
                };
                self.check_field_frame(transform_id, cp_grid, from_frame)?;
                let payload = encode_bspline(
                    cp,
                    cp_grid,
                    spec.order.unwrap_or(3),
                    spec.vector_space.as_deref().unwrap_or("world"),
                )?;
                self.check_field_lattice(transform_id, &payload.array("control_points")?.shape(), cp_grid, false)?;
                Ok(payload)
            }
            "composite" => {
                let Some(components) = &spec.components else {
                    return Err(Error::coded("E501", format!("transform {id}: kind 'composite' needs `components`")));
                };
                let missing: Vec<&String> =
                    components.iter().filter(|c| !self.transform_frames.contains_key(*c)).collect();
                if !missing.is_empty() {
                    return Err(Error::coded(
                        "E501",
                        format!(
                            "transform {id} names components {} that do not exist yet; declare them first",
                            repr_list(&missing)
                        ),
                    ));
                }
                encode_composite(components)
            }
            other => Err(Error::coded(
                "E502",
                format!("unknown transform kind {}; expected one of {}", repr_str(other), repr_list(&TRANSFORM_KINDS)),
            )),
        }
    }

    fn check_field_frame(&self, transform_id: &str, grid_id: &str, from_frame: &str) -> Result<()> {
        let grid = self.grid_ref(grid_id)?;
        if let Some(frame) = &grid.frame_uid {
            if frame != from_frame {
                return Err(Error::coded(
                    "E503",
                    format!(
                        "transform {}: grid {} is in frame {} but the transform starts in {}; the field must be sampled in the source frame",
                        repr_str(transform_id),
                        repr_str(grid_id),
                        repr_str(frame),
                        repr_str(from_frame)
                    ),
                ));
            }
        }
        Ok(())
    }

    fn check_field_lattice(&self, transform_id: &str, shape: &[usize], grid_id: &str, lattice: bool) -> Result<()> {
        let grid = self.grid_ref(grid_id)?;
        if shape.first().copied() != Some(grid.n_spatial()) {
            return Err(Error::coded(
                "E503",
                format!(
                    "transform {}: {} components on grid {}, which has {} spatial axes",
                    repr_str(transform_id),
                    shape.first().copied().unwrap_or(0),
                    repr_str(grid_id),
                    grid.n_spatial()
                ),
            ));
        }
        if lattice && shape[1..] != grid.spatial_shape()[..] {
            return Err(Error::coded(
                "E503",
                format!(
                    "transform {}: field lattice {} is not grid {}'s spatial shape {}; the field is sampled at that grid's voxels (§10.3)",
                    repr_str(transform_id),
                    repr_int_tuple(&shape[1..]),
                    repr_str(grid_id),
                    repr_int_tuple(&grid.spatial_shape())
                ),
            ));
        }
        Ok(())
    }

    // -- index ------------------------------------------------------------------------

    /// Build sampling indices for the named voxel annotations (§14.3); all
    /// non-mask voxel annotations when `ann_ids` is `None`.
    pub fn build_index(
        &mut self,
        ann_ids: Option<&[String]>,
        max_coords: Option<usize>,
        occupancy: Option<Option<usize>>,
        seed: u64,
    ) -> Result<Vec<String>> {
        let names: Vec<String> = match ann_ids {
            Some(ids) => ids.to_vec(),
            None => self
                .annotation_kinds
                .iter()
                .filter(|(_, k)| VOXEL_KINDS.contains(&k.as_str()) && *k != "mask")
                .map(|(n, _)| n.clone())
                .collect(),
        };
        let root = self.root()?;
        let grids = std::sync::Arc::new(self.grids.clone());
        let label_set = self.document.label_set.clone().map(std::sync::Arc::new);
        let profile = resolve_profile(Some(&self.codec))?;
        let mut built = Vec::new();
        for name in names {
            let group = root.group("annotations")?.group(&name)?;
            let annotation = Annotation::open(&name, group.clone(), grids.clone(), label_set.clone())?;
            if !annotation.is_voxel() {
                continue;
            }
            let digest = group_digest(&group, &root, "sha256")?;
            let payload = build_index(
                &annotation,
                None,
                max_coords.unwrap_or(DEFAULT_MAX_COORDS),
                occupancy.unwrap_or(Some(DEFAULT_OCCUPANCY_FACTOR)),
                seed,
                Some(digest),
            )?;
            write_index(&root, &payload, &profile)?;
            built.push(name);
        }
        Ok(built)
    }
}

/// Convert an open voxel annotation to another encoding (§7.6).
///
/// Refuses rather than dropping what the target cannot express: an in-band
/// ignore region, object identity, and class identity itself.
pub fn transcode(annotation: &Annotation, to_kind: &str, drop_identity: bool) -> Result<Payload> {
    check_target(to_kind)?;
    check_identity(annotation.kind(), to_kind, drop_identity)?;
    if to_kind == "instances" && annotation.kind() != "instances" {
        return Err(Error::coded(
            "E404",
            format!(
                "cannot transcode {} to 'instances': a dense encoding carries no object identity, so every object of a \
                 class would merge into one with a newly minted instance_id (spec §7.4). Re-derive the objects from the \
                 source that had them.",
                repr_str(annotation.kind())
            ),
        ));
    }
    let mut options = EncodeOptions::default();
    if annotation.encodes_ignore()? {
        if !IN_BAND_IGNORE_KINDS.contains(&to_kind) {
            return Err(Error::coded(
                "E404",
                format!(
                    "cannot transcode {} to {}: this annotation carries an in-band ignore region, which {} cannot hold \
                     (spec §7.7). Write the ignore region as a separate `mask` annotation and reference it with \
                     `ignore_mask=` first, or transcode to 'labelmap' or 'layers' instead.",
                    repr_str(annotation.kind()),
                    repr_str(to_kind),
                    repr_str(to_kind)
                ),
            ));
        }
        options.ignore = Some(annotation.ignore_mask(None)?);
        options.ignore_id = Some(annotation.ignore_id());
    }
    let masks = annotation_to_masks(annotation, None)?;
    let shape = annotation.spatial_shape()?;
    encode_masks(&masks, to_kind, Some(&shape), &options)
}

/// Decode an open annotation to per-class boolean masks.
pub fn annotation_to_masks(annotation: &Annotation, classes: Option<&[ClassKey]>) -> Result<Masks> {
    let ids = annotation.resolve_classes(classes)?;
    let window = annotation.window(None)?;
    ids.into_iter().map(|c| Ok((c, annotation.dense_class(c, &window)?))).collect()
}
