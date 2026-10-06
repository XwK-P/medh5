//! The voxel encoders (spec §7.1-§7.5) and payload-level transcoding (§7.6).
//!
//! Every encoder takes per-class boolean masks (or probabilities) and returns a
//! [`Payload`]; nothing here touches HDF5.  Transcoding decodes a payload back
//! to masks and re-encodes them, so `contains(class, voxel)` is preserved by
//! construction --- `probmap` excepted, which is lossless only under its
//! declared threshold.

use std::collections::BTreeMap;

use half::f16;
use ndarray::{ArrayD, Axis, Dimension, IxDyn, Slice as NdSlice};

use super::payload::{normalize_masks, packbits, unpackbits, Masks, Payload, PayloadData};
use super::select::{analyse, bbox, label_dtype_size, layers_from_colouring};
use crate::array::{DType, NdArray};
use crate::geometry::affine::{box_to_slices, slices_to_box};
use crate::h5::attrs::AttrValue;
use crate::json::{repr_int_list, repr_int_tuple, repr_list, repr_str};
use crate::labels::{check_class_id, IGNORE_ID};
use crate::{Error, Result};

/// Bits per `uint64` plane.
pub const BITS_PER_PLANE: usize = 64;
/// The probmap decision threshold when none is declared (§7.5).
pub const DEFAULT_THRESHOLD: f64 = 0.5;
/// The encodings a transcode may target.
pub const TRANSCODABLE: [&str; 5] = ["labelmap", "layers", "bitmask", "instances", "probmap"];
/// Encodings that can hold an ignore region in the data itself (§7.7).
pub const IN_BAND_IGNORE_KINDS: [&str; 2] = ["labelmap", "layers"];

/// Options the encoders accept; each uses only its own.
#[derive(Debug, Clone, Default)]
pub struct EncodeOptions {
    /// An in-band ignore region (`labelmap`, `layers`).
    pub ignore: Option<ArrayD<bool>>,
    /// The ignore value (default 65535).
    pub ignore_id: Option<i64>,
    /// A precomputed layer colouring (`layers`).
    pub colouring: Option<BTreeMap<i64, usize>>,
    /// The narrowest storage dtype wanted (`probmap`, default float16).
    pub probmap_dtype: Option<DType>,
    /// Whether channels sum to one (`probmap`).
    pub normalized: bool,
    /// The decision threshold (`probmap`).
    pub threshold: Option<f64>,
    /// The first minted instance id (`instances` from masks).
    pub start_id: Option<u64>,
    /// Whether to store per-object masks (`instances`, default true).
    pub store_masks: Option<bool>,
}

fn first_true(mask: &ArrayD<bool>) -> Option<Vec<usize>> {
    mask.indexed_iter().find(|(_, v)| **v).map(|(idx, _)| idx.slice().to_vec())
}

// -- labelmap -------------------------------------------------------------------

/// Pack mutually exclusive class masks into one integer volume (§7.1).
///
/// Raises E404 when two classes claim the same voxel: silently letting the last
/// writer win would turn a curation error into a training set nobody can audit.
pub fn encode_labelmap(masks: &Masks, spatial_shape: Option<&[usize]>, ignore: Option<&ArrayD<bool>>, ignore_id: i64) -> Result<Payload> {
    let shape = normalize_masks(masks, spatial_shape)?;
    let class_ids: Vec<i64> = masks.keys().copied().collect();
    let itemsize = label_dtype_size(&class_ids, ignore.is_some());
    let mut data = ArrayD::<u16>::zeros(IxDyn(&shape));
    let mut claimed = ArrayD::from_elem(IxDyn(&shape), false);
    for (class_id, mask) in masks {
        let clash = ndarray::Zip::from(mask).and(&claimed).map_collect(|m, c| *m && *c);
        if let Some(first) = first_true(&clash) {
            return Err(Error::coded(
                "E404",
                format!(
                    "labelmap requires mutually exclusive classes; class {class_id} overlaps an earlier class at \
                     voxel {}. Use `layers` or `bitmask` for overlapping structures.",
                    repr_int_tuple(&first)
                ),
            ));
        }
        ndarray::Zip::from(&mut data).and(mask).and(&mut claimed).for_each(|d, m, c| {
            if *m {
                *d = *class_id as u16;
                *c = true;
            }
        });
    }
    if let Some(ign) = ignore {
        let value = ignore_id as u16;
        ndarray::Zip::from(&mut data).and(ign).and(&claimed).for_each(|d, i, c| {
            if *i && !*c {
                *d = value;
            }
        });
    }
    let stored = if itemsize == 1 { NdArray::U8(data.mapv(|v| v as u8)) } else { NdArray::U16(data) };
    let mut p = Payload::new("labelmap");
    p.datasets.insert("data".into(), stored.into());
    p.class_ids = class_ids;
    Ok(p)
}

// -- layers ----------------------------------------------------------------------

/// Colour the overlap graph and pack each colour into one labelmap (§7.2).
pub fn encode_layers(
    masks: &Masks,
    spatial_shape: Option<&[usize]>,
    colouring: Option<&BTreeMap<i64, usize>>,
    ignore: Option<&ArrayD<bool>>,
    ignore_id: i64,
) -> Result<Payload> {
    let shape = normalize_masks(masks, spatial_shape)?;
    let class_ids: Vec<i64> = masks.keys().copied().collect();
    let colouring: BTreeMap<i64, usize> = match colouring {
        None => analyse(masks, Some(&shape))?.colouring,
        Some(given) => {
            let mut missing: Vec<i64> = class_ids.iter().filter(|c| !given.contains_key(c)).copied().collect();
            missing.sort();
            if !missing.is_empty() {
                return Err(Error::coded("E404", format!("colouring omits classes {}", repr_int_list(&missing))));
            }
            class_ids.iter().map(|c| (*c, given[c])).collect()
        }
    };
    let buckets = layers_from_colouring(&colouring);
    let n_layers = buckets.len().max(1);
    let itemsize = label_dtype_size(&class_ids, ignore.is_some());
    let mut full_shape = vec![n_layers];
    full_shape.extend(&shape);
    let mut data = ArrayD::<u16>::zeros(IxDyn(&full_shape));
    for (layer, bucket) in buckets.iter().enumerate() {
        let mut plane = data.index_axis_mut(Axis(0), layer);
        let mut claimed = ArrayD::from_elem(IxDyn(&shape), false);
        for class_id in bucket {
            let mask = &masks[class_id];
            let overlap = ndarray::Zip::from(mask).and(&claimed).fold(false, |acc, m, c| acc || (*m && *c));
            if overlap {
                return Err(Error::coded(
                    "E404",
                    format!(
                        "classes {} were assigned to layer {layer} but overlap; the colouring is not a valid \
                         colouring of the overlap graph",
                        repr_int_tuple(bucket)
                    ),
                ));
            }
            ndarray::Zip::from(&mut plane).and(mask).and(&mut claimed).for_each(|d, m, c| {
                if *m {
                    *d = *class_id as u16;
                    *c = true;
                }
            });
        }
        if let Some(ign) = ignore {
            let value = ignore_id as u16;
            ndarray::Zip::from(&mut plane).and(ign).and(&claimed).for_each(|d, i, c| {
                if *i && !*c {
                    *d = value;
                }
            });
        }
    }
    let width = buckets.iter().map(Vec::len).max().unwrap_or(1).max(1);
    let mut table = ArrayD::<u16>::zeros(IxDyn(&[n_layers, width]));
    for (layer, bucket) in buckets.iter().enumerate() {
        for (k, cid) in bucket.iter().enumerate() {
            table[[layer, k]] = *cid as u16;
        }
    }
    let stored = if itemsize == 1 { NdArray::U8(data.mapv(|v| v as u8)) } else { NdArray::U16(data) };
    let mut p = Payload::new("layers");
    p.datasets.insert("data".into(), stored.into());
    p.datasets.insert("layer_class_ids".into(), NdArray::U16(table).into());
    p.stacked_axes = 1;
    p.class_ids = class_ids;
    Ok(p)
}

// -- bitmask ----------------------------------------------------------------------

/// Pack class masks into `ceil(C/64)` `uint64` bitplanes, LSB-first (§7.3).
pub fn encode_bitmask(masks: &Masks, spatial_shape: Option<&[usize]>) -> Result<Payload> {
    let shape = normalize_masks(masks, spatial_shape)?;
    let class_ids: Vec<i64> = masks.keys().copied().collect();
    let n_planes = class_ids.len().div_ceil(BITS_PER_PLANE).max(1);
    let mut full_shape = vec![n_planes];
    full_shape.extend(&shape);
    let mut data = ArrayD::<u64>::zeros(IxDyn(&full_shape));
    for (position, class_id) in class_ids.iter().enumerate() {
        let (plane, bit) = (position / BITS_PER_PLANE, position % BITS_PER_PLANE);
        let flag = 1u64 << bit;
        let mut view = data.index_axis_mut(Axis(0), plane);
        ndarray::Zip::from(&mut view).and(&masks[class_id]).for_each(|d, m| {
            if *m {
                *d |= flag;
            }
        });
    }
    let ids = ArrayD::from_shape_vec(IxDyn(&[class_ids.len()]), class_ids.iter().map(|c| *c as u16).collect())?;
    let mut p = Payload::new("bitmask");
    p.datasets.insert("data".into(), NdArray::U64(data).into());
    p.datasets.insert("bit_class_ids".into(), NdArray::U16(ids).into());
    p.stacked_axes = 1;
    p.class_ids = class_ids;
    Ok(p)
}

// -- mask --------------------------------------------------------------------------

/// Wrap a boolean volume as a `mask` payload.
pub fn encode_mask(mask: ArrayD<bool>) -> Payload {
    let mut p = Payload::new("mask");
    p.datasets.insert("data".into(), NdArray::Bool(mask).into());
    p
}

// -- probmap ---------------------------------------------------------------------

/// `values >= threshold`, decided in the precision `values` are stored in (§7.5).
///
/// The threshold is a float64 attribute and the data is usually float16;
/// rounding the threshold the way the data was rounded makes equal values
/// compare equal.
pub fn contains_at(values: &NdArray, threshold: f64) -> ArrayD<bool> {
    match values {
        NdArray::F16(a) => {
            let t = f16::from_f64(threshold);
            a.mapv(|v| v >= t)
        }
        NdArray::F32(a) => {
            let t = threshold as f32;
            a.mapv(|v| v >= t)
        }
        NdArray::F64(a) => a.mapv(|v| v >= threshold),
        other => other.to_f64().mapv(|v| v >= threshold),
    }
}

/// The narrowest allowed dtype, no narrower than `requested`, under which every
/// voxel lands on the same side of `threshold` as it was given.
pub fn storage_dtype(planes: &[ArrayD<f64>], threshold: f64, requested: DType) -> DType {
    let allowed = [DType::F16, DType::F32];
    let mut candidates: Vec<DType> = allowed.iter().copied().filter(|d| d.itemsize() >= requested.itemsize()).collect();
    if candidates.is_empty() {
        candidates.push(requested);
    }
    for candidate in &candidates {
        let ok = planes.iter().all(|arr| {
            let given = arr.mapv(|v| v >= threshold);
            let stored = contains_at(&NdArray::F64(arr.clone()).astype(*candidate), threshold);
            given == stored
        });
        if ok {
            return *candidate;
        }
    }
    *candidates.last().unwrap()
}

/// Stack per-class probability volumes on a leading class axis (§7.5).
pub fn encode_probmap(
    probabilities: &BTreeMap<i64, ArrayD<f64>>,
    spatial_shape: Option<&[usize]>,
    dtype: DType,
    normalized: bool,
    threshold: Option<f64>,
) -> Result<Payload> {
    let mut attrs = vec![("normalized".to_string(), AttrValue::Bool(normalized))];
    if let Some(t) = threshold {
        if !(0.0..=1.0).contains(&t) || t.is_nan() {
            return Err(Error::coded("E404", format!("threshold {} must lie in [0, 1]", crate::json::py_float(t))));
        }
        attrs.push(("threshold".to_string(), AttrValue::Float(t)));
    }
    let mut class_ids = Vec::new();
    for c in probabilities.keys() {
        class_ids.push(check_class_id(*c)? as i64);
    }
    class_ids.sort();
    let decide = threshold.unwrap_or(DEFAULT_THRESHOLD);
    let mut shape: Option<Vec<usize>> = spatial_shape.map(<[usize]>::to_vec);
    let mut planes = Vec::new();
    for class_id in &class_ids {
        let arr = &probabilities[class_id];
        match &shape {
            None => shape = Some(arr.shape().to_vec()),
            Some(s) if arr.shape() != s.as_slice() => {
                return Err(Error::coded(
                    "E405",
                    format!(
                        "probability map for class {class_id} has shape {}, expected {}",
                        repr_int_tuple(arr.shape()),
                        repr_int_tuple(s)
                    ),
                ))
            }
            _ => {}
        }
        if !arr.is_empty() {
            let lo = arr.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = arr.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if lo < 0.0 || hi > 1.0 {
                return Err(Error::coded("E411", format!("probability map for class {class_id} has values outside [0, 1]")));
            }
        }
        planes.push(arr.clone());
    }
    let shape = shape.ok_or_else(|| Error::coded("E410", "no probability maps were supplied"))?;
    let chosen = storage_dtype(&planes, decide, dtype);
    let mut full_shape = vec![planes.len()];
    full_shape.extend(&shape);
    let mut stacked = ArrayD::<f64>::zeros(IxDyn(&full_shape));
    for (i, plane) in planes.iter().enumerate() {
        stacked.index_axis_mut(Axis(0), i).assign(plane);
    }
    let mut p = Payload::new("probmap");
    p.datasets.insert("data".into(), NdArray::F64(stacked).astype(chosen).into());
    p.attrs = attrs;
    p.stacked_axes = 1;
    p.class_ids = class_ids;
    Ok(p)
}

// -- instances --------------------------------------------------------------------

/// One object handed to [`encode_instances`].
#[derive(Debug, Clone, Default)]
pub struct InstanceInput {
    pub class_id: i64,
    pub instance_id: u64,
    /// A full-grid mask; its tight crop and box are derived.
    pub mask: Option<ArrayD<bool>>,
    /// A box, flat `[lo0, hi0, lo1, hi1, ...]`.
    pub bbox: Option<Vec<f64>>,
    /// The bbox-local crop, with `bbox`.
    pub crop: Option<ArrayD<bool>>,
    pub score: Option<f64>,
}

/// The `uint32`/`uint64` width an instance-id column needs (§7.4).
pub fn instance_id_dtype(ids: &[u64]) -> DType {
    if ids.iter().any(|i| *i > 0xFFFF_FFFF) {
        DType::U64
    } else {
        DType::U32
    }
}

/// An instance-id column in the width it needs.
pub fn instance_id_array(ids: &[u64]) -> NdArray {
    let shape = [ids.len()];
    match instance_id_dtype(ids) {
        DType::U64 => NdArray::U64(ArrayD::from_shape_vec(IxDyn(&shape), ids.to_vec()).unwrap()),
        _ => NdArray::U32(ArrayD::from_shape_vec(IxDyn(&shape), ids.iter().map(|v| *v as u32).collect()).unwrap()),
    }
}

/// Pack objects into boxes, ids, offsets and one concatenated bit stream (§7.4).
///
/// `class_ids` declares the classes the annotation can express; a class
/// searched for and not found must stay in it.  No objects at all is written
/// as `N = 0` columns: "examined, none found".
pub fn encode_instances(
    objects: &[InstanceInput],
    spatial_shape: Option<&[usize]>,
    store_masks: bool,
    class_ids: Option<&[i64]>,
) -> Result<Payload> {
    if objects.is_empty() {
        let (Some(ids), Some(shape)) = (class_ids.filter(|c| !c.is_empty()), spatial_shape) else {
            return Err(Error::coded(
                "E410",
                "no instances were supplied; an empty `instances` annotation needs class_ids (what was examined) \
                 and spatial_shape",
            ));
        };
        return empty_instances(shape.len(), ids, store_masks);
    }
    let mut seen = std::collections::BTreeSet::new();
    let mut shared = std::collections::BTreeSet::new();
    for o in objects {
        if !seen.insert(o.instance_id) {
            shared.insert(o.instance_id);
        }
    }
    if !shared.is_empty() {
        let shared: Vec<u64> = shared.into_iter().collect();
        return Err(Error::coded(
            "E404",
            format!(
                "instance id(s) {} name more than one object in this annotation; an `instance_id` names one \
                 physical object, and two distinct objects MUST NOT share one (§7.4)",
                repr_int_list(&shared)
            ),
        ));
    }
    let mut ndim: Option<usize> = None;
    let mut boxes: Vec<f32> = Vec::new();
    let mut crops: Vec<ArrayD<bool>> = Vec::new();
    let mut object_classes: Vec<i64> = Vec::new();
    let mut instance_ids: Vec<u64> = Vec::new();
    let mut scores: Vec<f32> = Vec::new();
    let has_scores = objects.iter().any(|o| o.score.is_some());
    for obj in objects {
        let mut crop = obj.crop.clone();
        let mut bx: Option<Vec<f32>> = obj.bbox.as_ref().map(|b| b.iter().map(|v| *v as f32).collect());
        if let Some(mask) = &obj.mask {
            if let Some(shape) = spatial_shape {
                if mask.shape() != shape {
                    return Err(Error::coded(
                        "E405",
                        format!(
                            "instance {}: mask shape {} != grid {}",
                            obj.instance_id,
                            repr_int_tuple(mask.shape()),
                            repr_int_tuple(shape)
                        ),
                    ));
                }
            }
            let Some(slices) = bbox(mask) else {
                return Err(Error::coded("E404", format!("instance {} has an empty mask", obj.instance_id)));
            };
            let mut view = mask.view();
            for (axis, (lo, hi)) in slices.iter().enumerate() {
                view.slice_axis_inplace(Axis(axis), NdSlice::from(*lo..*hi));
            }
            crop = Some(view.to_owned());
            let pairs: Vec<(i64, i64)> = slices.iter().map(|(a, b)| (*a as i64, *b as i64)).collect();
            bx = Some(slices_to_box(&pairs));
        }
        let Some(bx) = bx else {
            return Err(Error::coded("E410", format!("instance {} needs either a mask or a box", obj.instance_id)));
        };
        if bx.len() % 2 != 0 {
            return Err(Error::coded("E405", format!("instance {}: a box is (S, 2)", obj.instance_id)));
        }
        let s = bx.len() / 2;
        if (0..s).any(|k| bx[2 * k] > bx[2 * k + 1]) {
            return Err(Error::coded("E406", format!("instance {}: box has lo > hi", obj.instance_id)));
        }
        match ndim {
            None => ndim = Some(s),
            Some(n) if n != s => return Err(Error::coded("E405", "instances disagree on dimensionality")),
            _ => {}
        }
        boxes.extend(&bx);
        object_classes.push(check_class_id(obj.class_id)? as i64);
        instance_ids.push(obj.instance_id);
        scores.push(obj.score.map(|v| v as f32).unwrap_or(f32::NAN));
        if store_masks {
            let Some(c) = crop else {
                return Err(Error::coded(
                    "E410",
                    format!("instance {}: store_masks=True needs a mask or crop", obj.instance_id),
                ));
            };
            crops.push(c);
        }
    }
    let n = objects.len();
    let s = ndim.unwrap_or(0);
    let mut p = Payload::new("instances");
    p.datasets.insert("boxes".into(), NdArray::from_vec(&[n, s, 2], boxes)?.into());
    p.datasets.insert(
        "class_ids".into(),
        NdArray::from_vec(&[n], object_classes.iter().map(|c| *c as u16).collect::<Vec<_>>())?.into(),
    );
    p.datasets.insert("instance_ids".into(), instance_id_array(&instance_ids).into());
    if has_scores {
        p.datasets.insert("scores".into(), NdArray::from_vec(&[n], scores)?.into());
    }
    if store_masks {
        let packed: Vec<Vec<u8>> = crops.iter().map(|c| packbits(c.iter().copied())).collect();
        let mut offsets = vec![0u64];
        for chunk in &packed {
            offsets.push(offsets.last().unwrap() + chunk.len() as u64);
        }
        let shapes: Vec<i32> = crops.iter().flat_map(|c| c.shape().iter().map(|v| *v as i32).collect::<Vec<_>>()).collect();
        let data: Vec<u8> = packed.concat();
        let crop_ndim = crops.first().map(|c| c.ndim()).unwrap_or(s);
        p.datasets.insert("mask_offsets".into(), NdArray::from_vec(&[n + 1], offsets)?.into());
        p.datasets.insert("mask_shapes".into(), NdArray::from_vec(&[n, crop_ndim], shapes)?.into());
        let len = data.len();
        p.datasets.insert("mask_data".into(), NdArray::from_vec(&[len], data)?.into());
    }
    p.class_ids = match class_ids {
        Some(ids) => {
            let mut v = Vec::new();
            for c in ids {
                v.push(check_class_id(*c)? as i64);
            }
            v.sort();
            v.dedup();
            v
        }
        None => {
            let mut v = object_classes.clone();
            v.sort();
            v.dedup();
            v
        }
    };
    Ok(p)
}

fn empty_instances(ndim: usize, class_ids: &[i64], store_masks: bool) -> Result<Payload> {
    let mut p = Payload::new("instances");
    p.datasets.insert("boxes".into(), NdArray::zeros(DType::F32, &[0, ndim, 2]).into());
    p.datasets.insert("class_ids".into(), NdArray::zeros(DType::U16, &[0]).into());
    p.datasets.insert("instance_ids".into(), NdArray::zeros(DType::U32, &[0]).into());
    if store_masks {
        p.datasets.insert("mask_offsets".into(), NdArray::zeros(DType::U64, &[1]).into());
        p.datasets.insert("mask_shapes".into(), NdArray::zeros(DType::I32, &[0, ndim]).into());
        p.datasets.insert("mask_data".into(), NdArray::zeros(DType::U8, &[0]).into());
    }
    let mut ids = Vec::new();
    for c in class_ids {
        ids.push(check_class_id(*c)? as i64);
    }
    ids.sort();
    ids.dedup();
    p.class_ids = ids;
    Ok(p)
}

/// One object per class --- what a converter can honestly infer from masks.
pub fn instances_from_masks(masks: &Masks, start_id: u64) -> Vec<InstanceInput> {
    masks
        .iter()
        .enumerate()
        .map(|(i, (class_id, mask))| InstanceInput {
            class_id: *class_id,
            instance_id: start_id + i as u64,
            mask: Some(mask.clone()),
            ..Default::default()
        })
        .collect()
}

/// Decode one object's bbox-local crop from the packed columns.
pub fn decode_crop(mask_data: &[u8], offsets: &[u64], shapes: &ArrayD<i64>, index: usize) -> Result<ArrayD<bool>> {
    let start = offsets[index] as usize;
    let stop = offsets[index + 1] as usize;
    let shape: Vec<usize> = shapes.index_axis(Axis(0), index).iter().map(|v| (*v).max(0) as usize).collect();
    let n: usize = shape.iter().product();
    let bits = unpackbits(&mask_data[start.min(mask_data.len())..stop.min(mask_data.len())], n);
    Ok(ArrayD::from_shape_vec(IxDyn(&shape), bits)?)
}

// -- decoding and transcoding -------------------------------------------------------

/// Decode any voxel payload to per-class boolean masks.
///
/// A `probmap` is thresholded at `threshold` when given, else at its own
/// declared threshold.
pub fn payload_to_masks(payload: &Payload, spatial_shape: Option<&[usize]>, threshold: Option<f64>) -> Result<Masks> {
    let mut out = Masks::new();
    match payload.kind.as_str() {
        "labelmap" => {
            let data = payload.data()?.cast::<i64>();
            for c in &payload.class_ids {
                out.insert(*c, data.mapv(|v| v == *c));
            }
        }
        "layers" => {
            let data = payload.data()?.cast::<i64>();
            let table = payload.array("layer_class_ids")?.cast::<i64>();
            for layer in 0..table.shape()[0] {
                let plane = data.index_axis(Axis(0), layer);
                for value in table.index_axis(Axis(0), layer).iter() {
                    if *value != 0 {
                        out.insert(*value, plane.mapv(|v| v == *value));
                    }
                }
            }
        }
        "bitmask" => {
            let data = payload.data()?.cast::<u64>();
            let ids = payload.array("bit_class_ids")?.cast::<i64>();
            for (position, value) in ids.iter().enumerate() {
                let (plane, bit) = (position / BITS_PER_PLANE, position % BITS_PER_PLANE);
                let flag = 1u64 << bit;
                out.insert(*value, data.index_axis(Axis(0), plane).mapv(|w| w & flag != 0));
            }
        }
        "probmap" => {
            let data = payload.data()?;
            let cut = threshold.or_else(|| payload.attr("threshold").and_then(AttrValue::as_f64)).unwrap_or(DEFAULT_THRESHOLD);
            for (i, c) in payload.class_ids.iter().enumerate() {
                let plane = crate::with_array!(data, a => NdArray::from(a.index_axis(Axis(0), i).to_owned()));
                out.insert(*c, contains_at(&plane, cut));
            }
        }
        "mask" => {
            out.insert(1, payload.data()?.nonzero_mask());
        }
        "instances" => {
            let Some(shape) = spatial_shape else {
                return Err(Error::coded("E405", "decoding `instances` needs the grid's spatial shape"));
            };
            return instances_to_masks(payload, shape);
        }
        other => return Err(Error::coded("E401", format!("cannot decode voxel kind {}", repr_str(other)))),
    }
    Ok(out)
}

fn instances_to_masks(payload: &Payload, spatial_shape: &[usize]) -> Result<Masks> {
    let boxes = payload.array("boxes")?.to_f64();
    let classes = payload.array("class_ids")?.cast::<i64>();
    let mut declared: Vec<i64> = payload.class_ids.clone();
    if declared.is_empty() {
        declared = classes.iter().copied().collect();
        declared.sort();
        declared.dedup();
    }
    let mut out: Masks = declared.iter().map(|c| (*c, ArrayD::from_elem(IxDyn(spatial_shape), false))).collect();
    let has_masks = payload.datasets.contains_key("mask_data");
    let n = boxes.shape()[0];
    for index in 0..n {
        let bx: Vec<f64> = boxes.index_axis(Axis(0), index).iter().copied().collect();
        let slices = box_to_slices(&bx, Some(spatial_shape))?;
        let class_id = classes[[index]];
        let target = out.entry(class_id).or_insert_with(|| ArrayD::from_elem(IxDyn(spatial_shape), false));
        let mut view = target.view_mut();
        for (axis, (lo, hi)) in slices.iter().enumerate() {
            view.slice_axis_inplace(Axis(axis), NdSlice::from(*lo as usize..*hi as usize));
        }
        if has_masks {
            let offsets: Vec<u64> = payload.array("mask_offsets")?.cast::<u64>().iter().copied().collect();
            let shapes = payload.array("mask_shapes")?.cast::<i64>();
            let mask_data: Vec<u8> = payload.array("mask_data")?.cast::<u8>().iter().copied().collect();
            let crop = decode_crop(&mask_data, &offsets, &shapes, index)?;
            if crop.shape() != view.shape() {
                return Err(Error::coded("E405", format!("instance {index}: crop shape does not match its box")));
            }
            ndarray::Zip::from(&mut view).and(&crop).for_each(|t, c| *t |= *c);
        } else {
            view.fill(true);
        }
    }
    Ok(out)
}

/// Encode per-class boolean masks into any voxel encoding.
pub fn encode_masks(masks: &Masks, kind: &str, spatial_shape: Option<&[usize]>, options: &EncodeOptions) -> Result<Payload> {
    let ignore_id = options.ignore_id.unwrap_or(IGNORE_ID);
    match kind {
        "labelmap" => encode_labelmap(masks, spatial_shape, options.ignore.as_ref(), ignore_id),
        "layers" => encode_layers(masks, spatial_shape, options.colouring.as_ref(), options.ignore.as_ref(), ignore_id),
        "bitmask" => encode_bitmask(masks, spatial_shape),
        "probmap" => {
            let probabilities: BTreeMap<i64, ArrayD<f64>> =
                masks.iter().map(|(c, m)| (*c, m.mapv(|v| if v { 1.0 } else { 0.0 }))).collect();
            encode_probmap(
                &probabilities,
                spatial_shape,
                options.probmap_dtype.unwrap_or(DType::F16),
                options.normalized,
                options.threshold,
            )
        }
        "mask" => {
            let mut planes = masks.values();
            let Some(first) = planes.next() else {
                return Err(Error::coded("E410", "no masks were supplied"));
            };
            let mut merged = first.clone();
            for plane in planes {
                ndarray::Zip::from(&mut merged).and(plane).for_each(|m, p| *m |= *p);
            }
            Ok(encode_mask(merged))
        }
        "instances" => {
            let start_id = options.start_id.unwrap_or(1);
            let objects: Vec<InstanceInput> = masks
                .iter()
                .enumerate()
                .filter(|(_, (_, m))| m.iter().any(|v| *v))
                .map(|(i, (c, m))| InstanceInput {
                    class_id: *c,
                    instance_id: start_id + i as u64,
                    mask: Some(m.clone()),
                    ..Default::default()
                })
                .collect();
            let ids: Vec<i64> = masks.keys().copied().collect();
            encode_instances(&objects, spatial_shape, options.store_masks.unwrap_or(true), Some(&ids))
        }
        other => Err(Error::coded("E401", format!("cannot encode voxel kind {}", repr_str(other)))),
    }
}

/// Refuse a transcode target that cannot hold what `contains` promises.
pub fn check_target(to_kind: &str) -> Result<()> {
    if to_kind == "mask" {
        return Err(Error::coded(
            "E404",
            "cannot transcode to 'mask': a `mask` has no classes (§4.4), so every class would merge into one \
             volume and the coverage contract (`class_ids`, `annotated_class_ids`) would be lost. For a deliberate \
             union, build one with `encode_mask` and write it with `add_mask`.",
        ));
    }
    if !TRANSCODABLE.contains(&to_kind) {
        return Err(Error::coded(
            "E401",
            format!("{} is not a voxel encoding; expected one of {}", repr_str(to_kind), repr_list(&TRANSCODABLE)),
        ));
    }
    Ok(())
}

/// Object identity leaves only when the caller says so (§7.4).
pub fn check_identity(from_kind: &str, to_kind: &str, drop_identity: bool) -> Result<()> {
    if from_kind == "instances" && to_kind != "instances" && !drop_identity {
        return Err(Error::coded(
            "E404",
            format!(
                "transcoding 'instances' to {} keeps every voxel and drops every instance_id --- the field tracking \
                 joins on across visits (§7.4). Pass drop_identity=True (--drop-identity) to do it deliberately; the \
                 transcode is then recorded in the provenance.",
                repr_str(to_kind)
            ),
        ));
    }
    Ok(())
}

/// Convert a payload to another encoding, preserving `contains`.
pub fn transcode_payload(
    payload: &Payload,
    to_kind: &str,
    spatial_shape: Option<&[usize]>,
    threshold: Option<f64>,
    drop_identity: bool,
    options: &EncodeOptions,
) -> Result<Payload> {
    if to_kind == payload.kind {
        return Ok(payload.clone());
    }
    check_target(to_kind)?;
    check_identity(&payload.kind, to_kind, drop_identity)?;
    let masks = payload_to_masks(payload, spatial_shape, threshold)?;
    let shape: Vec<usize> = match spatial_shape {
        Some(s) => s.to_vec(),
        None => masks.values().next().map(|m| m.shape().to_vec()).unwrap_or_default(),
    };
    encode_masks(&masks, to_kind, Some(&shape), options)
}

/// Whether two mask sets agree on every class --- the losslessness check.
pub fn masks_equal(a: &Masks, b: &Masks) -> bool {
    a.len() == b.len() && a.iter().all(|(k, v)| b.get(k) == Some(v))
}

/// Whether `A -> B -> A` preserves every class mask.
pub fn check_roundtrip(payload: &Payload, to_kind: &str, spatial_shape: Option<&[usize]>) -> Result<bool> {
    let shape: Option<Vec<usize>> = match spatial_shape {
        Some(s) => Some(s.to_vec()),
        None => payload.datasets.get("data").map(|d| d.shape()[payload.stacked_axes..].to_vec()),
    };
    let original = payload_to_masks(payload, shape.as_deref(), None)?;
    let converted = transcode_payload(payload, to_kind, shape.as_deref(), None, false, &EncodeOptions::default())?;
    let decoded = payload_to_masks(&converted, shape.as_deref(), None)?;
    Ok(masks_equal(&original, &decoded))
}

/// A payload dataset as `PayloadData`, for writers building payloads by hand.
pub fn array_data(array: NdArray) -> PayloadData {
    PayloadData::Array(array)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(shape: &[usize], origin: &[usize], size: usize) -> ArrayD<bool> {
        let mut m = ArrayD::from_elem(IxDyn(shape), false);
        for (idx, v) in m.indexed_iter_mut() {
            if (0..shape.len()).all(|a| idx[a] >= origin[a] && idx[a] < origin[a] + size) {
                *v = true;
            }
        }
        m
    }

    fn masks() -> Masks {
        let shape = [16, 24, 24];
        let mut m = Masks::new();
        m.insert(1, block(&shape, &[2, 2, 2], 8));
        m.insert(2, block(&shape, &[2, 14, 2], 6));
        m.insert(3, block(&shape, &[4, 4, 4], 3));
        m
    }

    #[test]
    fn every_transcode_is_lossless() {
        let masks = masks();
        let shape = vec![16usize, 24, 24];
        for from in ["layers", "bitmask", "instances", "probmap"] {
            let payload = encode_masks(&masks, from, Some(&shape), &EncodeOptions::default()).unwrap();
            let decoded = payload_to_masks(&payload, Some(&shape), None).unwrap();
            assert!(masks_equal(&decoded, &masks), "{from}");
            for to in ["layers", "bitmask", "probmap"] {
                let opts = EncodeOptions::default();
                let out = if from == "instances" {
                    transcode_payload(&payload, to, Some(&shape), None, true, &opts).unwrap()
                } else {
                    transcode_payload(&payload, to, Some(&shape), None, false, &opts).unwrap()
                };
                let back = payload_to_masks(&out, Some(&shape), None).unwrap();
                assert!(masks_equal(&back, &masks), "{from}->{to}");
            }
        }
        assert_eq!(encode_labelmap(&masks, None, None, IGNORE_ID).unwrap_err().code(), Some("E404"));
    }

    #[test]
    fn probmap_widens_when_float16_would_move_a_voxel() {
        let mut probs = BTreeMap::new();
        probs.insert(1, ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0 / 3.0, 0.0, 1.0]).unwrap());
        let p = encode_probmap(&probs, None, DType::F16, false, Some(1.0 / 3.0)).unwrap();
        // float16(1/3) is 0.33325 < 1/3, so containment survives only in float16
        // because the threshold is rounded the same way.
        let data = p.data().unwrap();
        assert_eq!(data.dtype(), DType::F16);
        let back = payload_to_masks(&p, None, None).unwrap();
        assert_eq!(back[&1].iter().copied().collect::<Vec<_>>(), vec![true, false, true]);
    }
}
