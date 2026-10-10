//! The fixed attribute header every annotation carries (spec §6.2).

use crate::h5::attrs::{self, AttrValue};
use crate::json::{repr_int_list, repr_list, repr_str};
use crate::labels::{BACKGROUND_ID, CLOSURES, IGNORE_ID};
use crate::{Error, Result};

/// Voxel annotation kinds (§7).
pub const VOXEL_KINDS: [&str; 6] = ["labelmap", "layers", "bitmask", "instances", "probmap", "mask"];
/// Geometric annotation kinds (§8).
pub const GEOMETRIC_KINDS: [&str; 6] = ["boxes", "obb", "keypoints", "points", "contours", "mesh"];
/// Every annotation kind 1.0 defines (§6.3).
pub const ANNOTATION_KINDS: [&str; 13] = [
    "labelmap",
    "layers",
    "bitmask",
    "instances",
    "probmap",
    "mask",
    "boxes",
    "obb",
    "keypoints",
    "points",
    "contours",
    "mesh",
    "classification",
];
/// Names reserved by spec §16.  A 1.0 writer MUST NOT emit them.
pub const RESERVED_KINDS: [&str; 1] = ["rle"];
/// `task` values (§6.2).
pub const TASKS: [&str; 5] = ["segmentation", "detection", "classification", "registration", "other"];

/// The task a kind implies when the file does not say.
pub fn default_task_for_kind(kind: &str) -> Option<&'static str> {
    Some(match kind {
        "labelmap" | "layers" | "bitmask" | "probmap" | "instances" | "contours" | "mesh" => "segmentation",
        "mask" => "other",
        "boxes" | "obb" | "keypoints" | "points" => "detection",
        "classification" => "classification",
        _ => return None,
    })
}

/// The annotation attributes the spec defines, in canonical order.
pub const SPEC_ANNOTATION_ATTRS: [&str; 21] = [
    "kind",
    "task",
    "grid",
    "timepoints",
    "space",
    "frame_uid",
    "class_ids",
    "annotated_class_ids",
    "closure",
    "ignore_id",
    "ignore_mask",
    "prov",
    "quality",
    "derived_from",
    "digest",
    "scope",
    "scope_ids",
    "multilabel",
    "normalized",
    "threshold",
    "skeleton",
];

/// Whether a kind is a voxel kind.
pub fn is_voxel_kind(kind: &str) -> bool {
    VOXEL_KINDS.contains(&kind)
}

/// Whether a kind is a geometric kind.
pub fn is_geometric_kind(kind: &str) -> bool {
    GEOMETRIC_KINDS.contains(&kind)
}

/// The fixed attribute header every annotation carries (spec §6.2).
#[derive(Debug, Clone, PartialEq)]
pub struct AnnotationHeader {
    pub kind: String,
    pub task: String,
    pub grid: Option<String>,
    pub timepoints: Option<Vec<String>>,
    pub space: Option<String>,
    pub frame_uid: Option<String>,
    pub class_ids: Vec<i64>,
    pub annotated_class_ids: Vec<i64>,
    pub closure: String,
    pub ignore_id: i64,
    pub ignore_mask: Option<String>,
    pub prov: Option<String>,
    pub quality: Option<String>,
    pub derived_from: Vec<String>,
    /// Attributes outside the header, kept so a rewrite carries them (§16).
    pub extra: Vec<(String, AttrValue)>,
}

impl AnnotationHeader {
    /// A header with the given kind and task and every other field defaulted.
    pub fn new(kind: impl Into<String>, task: impl Into<String>) -> Self {
        AnnotationHeader {
            kind: kind.into(),
            task: task.into(),
            grid: None,
            timepoints: None,
            space: None,
            frame_uid: None,
            class_ids: Vec::new(),
            annotated_class_ids: Vec::new(),
            closure: "explicit".into(),
            ignore_id: IGNORE_ID,
            ignore_mask: None,
            prov: None,
            quality: None,
            derived_from: Vec::new(),
            extra: Vec::new(),
        }
    }

    /// Validate §6.2 and §5.3.
    pub fn check(&self) -> Result<()> {
        if RESERVED_KINDS.contains(&self.kind.as_str()) {
            return Err(Error::coded(
                "E401",
                format!(
                    "annotation kind {} is reserved by spec §16 and must not be written by a 1.0 writer",
                    repr_str(&self.kind)
                ),
            ));
        }
        if !ANNOTATION_KINDS.contains(&self.kind.as_str()) {
            return Err(Error::coded("E401", format!("unknown annotation kind {}", repr_str(&self.kind))));
        }
        if !TASKS.contains(&self.task.as_str()) {
            return Err(Error::coded(
                "E412",
                format!("unknown task {}; expected one of {}", repr_str(&self.task), repr_list(&TASKS)),
            ));
        }
        if !CLOSURES.contains(&self.closure.as_str()) {
            return Err(Error::coded(
                "E412",
                format!("closure {} must be one of {}", repr_str(&self.closure), repr_list(&CLOSURES)),
            ));
        }
        let mut missing: Vec<i64> =
            self.annotated_class_ids.iter().filter(|c| !self.class_ids.contains(c)).copied().collect();
        missing.sort();
        missing.dedup();
        if !missing.is_empty() {
            return Err(Error::coded(
                "E403",
                format!("annotated_class_ids {} are not in class_ids", repr_int_list(&missing)),
            ));
        }
        let mut reserved: Vec<i64> =
            self.class_ids.iter().filter(|c| **c == BACKGROUND_ID || **c == IGNORE_ID).copied().collect();
        reserved.sort();
        reserved.dedup();
        if !reserved.is_empty() {
            return Err(Error::coded("E303", format!("class_ids uses reserved id(s) {}", repr_int_list(&reserved))));
        }
        let mut outside: Vec<i64> = self
            .class_ids
            .iter()
            .chain(&self.annotated_class_ids)
            .filter(|c| !(BACKGROUND_ID < **c && **c < IGNORE_ID))
            .copied()
            .collect();
        outside.sort();
        outside.dedup();
        if !outside.is_empty() {
            return Err(Error::coded(
                "E303",
                format!(
                    "class id(s) {} are outside the writable range [{}, {}] (spec §5.3)",
                    repr_int_list(&outside),
                    BACKGROUND_ID + 1,
                    IGNORE_ID - 1
                ),
            ));
        }
        Ok(())
    }

    /// The attributes this header writes.
    pub fn attrs(&self) -> Vec<(String, AttrValue)> {
        let u16s = |ids: &[i64]| {
            AttrValue::Array(crate::array::NdArray::from(
                ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[ids.len()]), ids.iter().map(|v| *v as u16).collect())
                    .unwrap(),
            ))
        };
        let mut out: Vec<(String, AttrValue)> = vec![
            ("kind".into(), AttrValue::Str(self.kind.clone())),
            ("task".into(), AttrValue::Str(self.task.clone())),
            ("closure".into(), AttrValue::Str(self.closure.clone())),
        ];
        if let Some(v) = &self.grid {
            out.push(("grid".into(), AttrValue::Str(v.clone())));
        }
        if let Some(v) = &self.timepoints {
            out.push(("timepoints".into(), list_attr(v)));
        }
        if let Some(v) = &self.space {
            out.push(("space".into(), AttrValue::Str(v.clone())));
        }
        if let Some(v) = &self.frame_uid {
            out.push(("frame_uid".into(), AttrValue::Str(v.clone())));
        }
        if self.kind != "mask" {
            out.push(("class_ids".into(), u16s(&self.class_ids)));
        }
        out.push(("annotated_class_ids".into(), u16s(&self.annotated_class_ids)));
        if self.ignore_id != IGNORE_ID {
            out.push(("ignore_id".into(), AttrValue::Int(self.ignore_id)));
        }
        if let Some(v) = &self.ignore_mask {
            out.push(("ignore_mask".into(), AttrValue::Str(v.clone())));
        }
        if let Some(v) = &self.prov {
            out.push(("prov".into(), AttrValue::Str(v.clone())));
        }
        if let Some(v) = &self.quality {
            out.push(("quality".into(), AttrValue::Str(v.clone())));
        }
        if !self.derived_from.is_empty() {
            out.push(("derived_from".into(), list_attr(&self.derived_from)));
        }
        out.extend(self.extra.iter().cloned());
        out
    }

    /// Read the header of an annotation group.
    pub fn read(group: &hdf5::Group) -> Result<Self> {
        let kind = text(&attrs::require(group, "kind", "E412")?);
        let task = match attrs::read(group, "task")? {
            Some(v) => text(&v),
            None => default_task_for_kind(&kind).map(str::to_string).ok_or_else(|| Error::Key(repr_str(&kind)))?,
        };
        let ids = |name: &str| -> Result<Vec<i64>> {
            Ok(attrs::read(group, name)?.and_then(|v| v.as_i64_vec()).unwrap_or_default())
        };
        let opt_text = |name: &str| -> Result<Option<String>> { Ok(attrs::read(group, name)?.map(|v| text(&v))) };
        let mut extra = Vec::new();
        for name in attrs::names(group)? {
            if !SPEC_ANNOTATION_ATTRS.contains(&name.as_str()) {
                if let Some(v) = attrs::read(group, &name)? {
                    extra.push((name, v));
                }
            }
        }
        let header = AnnotationHeader {
            kind,
            task,
            grid: opt_text("grid")?,
            timepoints: attrs::read(group, "timepoints")?.map(|v| v.as_str_list().unwrap_or_default()),
            space: opt_text("space")?,
            frame_uid: opt_text("frame_uid")?,
            class_ids: ids("class_ids")?,
            annotated_class_ids: ids("annotated_class_ids")?,
            closure: opt_text("closure")?.unwrap_or_else(|| "explicit".into()),
            ignore_id: attrs::read(group, "ignore_id")?.and_then(|v| v.as_i64()).unwrap_or(IGNORE_ID),
            ignore_mask: opt_text("ignore_mask")?,
            prov: opt_text("prov")?,
            quality: opt_text("quality")?,
            derived_from: attrs::read(group, "derived_from")?
                .map(|v| v.as_str_list().unwrap_or_default())
                .unwrap_or_default(),
            extra,
        };
        header.check()?;
        Ok(header)
    }
}

/// A string-list attribute; an empty list is stored as an empty `int64`
/// array, as the 1.x encoder stored any empty sequence.
pub fn list_attr(values: &[String]) -> AttrValue {
    if values.is_empty() {
        AttrValue::ints(&[])
    } else {
        AttrValue::strs(values)
    }
}

fn text(value: &AttrValue) -> String {
    value.as_str().unwrap_or_else(|| attrs::stringify_value(value))
}
