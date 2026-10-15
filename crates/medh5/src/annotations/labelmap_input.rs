//! A segmentation given as one integer volume and what its values stand for
//! (§7.1–§7.6), measured and encoded without building a mask per class.
//!
//! A value names one class, or several (an nnU-Net region is a union of
//! values, so a value is in its own class and every region holding it); 0 is
//! background.  The masks the value map implies are never made: one pass over
//! the volume interns its values and counts each one's voxels and bounding
//! box, which is everything [`analyse`](super::select::analyse) measures,
//! and the encoders read the volume again, writing each voxel's value from a
//! table built per distinct value.  So the memory beyond the volume itself is
//! the payload being written, where one mask per class cost the volume's
//! voxels once per class --- 13.6 GB at 104 classes of 512×512×500.
//!
//! What it writes equals what [`encode_masks`](super::encode::encode_masks)
//! writes for those masks, byte for byte, with the same [`OverlapStats`]
//! behind the choice; the tests hold the two routes to each other.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use ndarray::{ArrayD, ArrayViewD, Axis, IxDyn, Slice};

use super::encode::{encode_instances, minted_id, InstanceInput, BITS_PER_PLANE};
use super::payload::Payload;
use super::select::{greedy_colour, label_dtype_size, layers_from_colouring, OverlapStats, LOCALIZED_BBOX_FRACTION};
use crate::array::NdArray;
use crate::geometry::affine::slices_to_box;
use crate::h5::attrs::AttrValue;
use crate::json::{repr_int_tuple, repr_str};
use crate::labels::{check_class_id, IGNORE_ID};
use crate::{Error, Result};

mod sealed {
    pub trait Sealed {}
}

/// An integer type a labelmap volume may hold.
pub trait LabelValue: Copy + PartialEq + Send + Sync + sealed::Sealed + 'static {
    /// The value, or `None` when it is past `i64` (a `u64` above
    /// `i64::MAX`), which no value map can name.
    #[doc(hidden)]
    fn to_i64(self) -> Option<i64>;
}

macro_rules! label_value {
    ($($t:ty),*) => {$(
        impl sealed::Sealed for $t {}
        impl LabelValue for $t {
            fn to_i64(self) -> Option<i64> {
                i64::try_from(self).ok()
            }
        }
    )*};
}
label_value!(u8, u16, u32, u64, i8, i16, i32, i64);

/// The volume's values, interned: index 0 is background, and every other
/// index one value named by the map, the ignore value, or (with no map) a
/// value met in the volume.
#[derive(Debug, Clone)]
pub(crate) struct Values {
    index: HashMap<i64, usize>,
    /// The class ids each index stands for, ascending.
    classes: Vec<Vec<i64>>,
    ignore: Option<usize>,
    /// Without a map, a value met in the volume is the class of that id.
    identity: bool,
}

impl Values {
    /// `map` is each value's classes (`None`: every value its own class);
    /// `ignore` the value marking the ignore region.  0 names no class and the
    /// ignore value is no class, so a map naming either is refused (E404), as
    /// is a value naming no class.
    pub(crate) fn new(map: Option<&[(i64, Vec<i64>)]>, ignore: Option<i64>) -> Result<Values> {
        let mut values =
            Values { index: HashMap::from([(0, 0)]), classes: vec![Vec::new()], ignore: None, identity: map.is_none() };
        if let Some(value) = ignore {
            if value == 0 {
                return Err(Error::coded(
                    "E404",
                    "ignore_value=0 is background (§6.2); the ignore region is another value",
                ));
            }
            values.ignore = Some(values.intern(value, Vec::new()));
        }
        for (value, classes) in map.unwrap_or(&[]) {
            let what = format!("label_values[{value}]");
            if *value == 0 {
                return Err(Error::coded("E404", format!("{what}: 0 is background (§6.2), which names no class")));
            }
            if Some(*value) == ignore {
                return Err(Error::coded("E404", format!("{what}: {value} is the ignore value, which names no class")));
            }
            if classes.is_empty() {
                return Err(Error::coded("E404", format!("{what} names no class; a nonzero value names one or more")));
            }
            if values.index.contains_key(value) {
                return Err(Error::coded("E404", format!("{what} is given twice")));
            }
            let mut ids = classes.clone();
            ids.sort_unstable();
            ids.dedup();
            values.intern(*value, ids);
        }
        Ok(values)
    }

    fn intern(&mut self, value: i64, classes: Vec<i64>) -> usize {
        let at = self.classes.len();
        self.index.insert(value, at);
        self.classes.push(classes);
        at
    }

    /// The index of `value`, interned as its own class when the volume has
    /// no map; `None` for a value the map does not name.
    fn find(&mut self, value: Option<i64>) -> Option<usize> {
        let value = value?;
        if let Some(at) = self.index.get(&value) {
            return Some(*at);
        }
        self.identity.then(|| self.intern(value, vec![value]))
    }
}

/// What one pass over the volume measured, per interned value.
#[derive(Debug, Clone)]
pub(crate) struct Analysis {
    pub stats: OverlapStats,
    values: Values,
    counts: Vec<u64>,
    /// The bounding box, `(start, stop)` per axis, of each value's voxels.
    bounds: Vec<Option<Vec<(usize, usize)>>>,
}

impl Analysis {
    /// Whether the ignore value was given, and so the ignore region is.
    pub(crate) fn has_ignore(&self) -> bool {
        self.values.ignore.is_some()
    }

    /// The ignore region as a mask: the sibling `mask` annotation of an
    /// encoding that cannot hold it in band (§7.7).
    pub(crate) fn ignore_mask<T: LabelValue>(&self, data: ArrayViewD<'_, T>) -> Option<ArrayD<bool>> {
        let ignore = self.values.ignore?;
        Some(self.map(data, |at| at == ignore))
    }

    /// `f` of each voxel's value index, as an array of the volume's shape.
    fn map<T: LabelValue, U: Clone>(&self, data: ArrayViewD<'_, T>, f: impl Fn(usize) -> U) -> ArrayD<U> {
        let mut lookup = Lookup::new(&self.values);
        // In C order whatever the view's: `mapv` keeps a contiguous view's
        // memory order, and HDF5 writes row-major buffers.
        let out = data.iter().map(|v| f(lookup.at(*v))).collect();
        ArrayD::from_shape_vec(data.raw_dim(), out).expect("the volume's shape")
    }
}

/// [`Values::index`] with the last value looked up remembered: neighbouring
/// voxels mostly hold the same value.
struct Lookup<'a> {
    values: &'a Values,
    last: Option<(i64, usize)>,
}

impl<'a> Lookup<'a> {
    fn new(values: &'a Values) -> Lookup<'a> {
        Lookup { values, last: None }
    }

    /// The index of a value [`analyse_labelmap`] interned.
    fn at<T: LabelValue>(&mut self, value: T) -> usize {
        let value = value.to_i64().expect("analysed");
        match self.last {
            Some((v, at)) if v == value => at,
            _ => {
                let at = self.values.index[&value];
                self.last = Some((value, at));
                at
            }
        }
    }
}

/// Measure the volume as [`analyse`](super::select::analyse) measures the
/// masks its values imply, plus the classes in `examined` (empty masks).
///
/// A nonzero value the map does not name is refused (E404) at the first
/// voxel holding it, as is a class id outside `1..=65534` (E303, E306).
pub(crate) fn analyse_labelmap<T: LabelValue>(
    data: ArrayViewD<'_, T>,
    mut values: Values,
    examined: &[i64],
) -> Result<Analysis> {
    let shape = data.shape().to_vec();
    // Rows of the view as it lies, in logical (C) order: a Fortran-ordered
    // or flipped volume is read in place, never copied to C order first.
    let rows = if data.ndim() == 0 { data.insert_axis(Axis(0)) } else { data };
    let mut counts: Vec<u64> = vec![0; values.classes.len()];
    let mut bounds: Vec<Option<Vec<(usize, usize)>>> = vec![None; values.classes.len()];
    let outer = shape.split_last().map(|(_, outer)| outer.to_vec()).unwrap_or_default();
    // A row of the last axis at a time, as `bbox_flat` scans a mask: within a
    // row, each value's first and last column; at its end, the row's place.
    let mut at = vec![0usize; outer.len()];
    let mut row_span: Vec<(usize, usize)> = Vec::new();
    let mut in_row: Vec<usize> = Vec::new();
    let mut stamp: Vec<usize> = Vec::new();
    let mut last: Option<(T, usize)> = None;
    for (row_number, row) in rows.rows().into_iter().enumerate() {
        for (column, value) in row.iter().enumerate() {
            let index = match last {
                Some((v, index)) if v == *value => index,
                _ => {
                    let Some(index) = values.find(value.to_i64()) else {
                        let mut voxel = at.clone();
                        voxel.push(column);
                        voxel.truncate(shape.len());
                        return Err(Error::coded(
                            "E404",
                            format!(
                                "the labelmap holds {} at voxel {}, which label_values does not name: a nonzero value \
                                 names one or more classes, or is the ignore value",
                                value.to_i64().map_or_else(|| "a value past int64".to_string(), |v| v.to_string()),
                                repr_int_tuple(&voxel)
                            ),
                        ));
                    };
                    last = Some((*value, index));
                    index
                }
            };
            if counts.len() <= index {
                counts.resize(index + 1, 0);
                bounds.resize(index + 1, None);
            }
            counts[index] += 1;
            if index == 0 {
                continue;
            }
            if stamp.len() <= index {
                stamp.resize(index + 1, usize::MAX);
                row_span.resize(index + 1, (0, 0));
            }
            if stamp[index] != row_number {
                stamp[index] = row_number;
                row_span[index] = (column, column + 1);
                in_row.push(index);
            } else {
                row_span[index].1 = column + 1;
            }
        }
        for index in in_row.drain(..) {
            let (first, stop) = row_span[index];
            let mut span: Vec<(usize, usize)> = at.iter().map(|i| (*i, i + 1)).collect();
            span.push((first, stop));
            span.truncate(shape.len());
            bounds[index] = Some(match bounds[index].take() {
                None => span,
                Some(held) => held.iter().zip(&span).map(|(a, b)| (a.0.min(b.0), a.1.max(b.1))).collect(),
            });
        }
        for (index, extent) in at.iter_mut().zip(&outer).rev() {
            *index += 1;
            if *index < *extent {
                break;
            }
            *index = 0;
        }
    }
    counts.resize(values.classes.len(), 0);
    bounds.resize(values.classes.len(), None);

    let mut ids: BTreeSet<i64> = examined.iter().copied().collect();
    for classes in &values.classes {
        ids.extend(classes);
    }
    for id in &ids {
        check_class_id(*id)?;
    }
    let class_ids: Vec<i64> = ids.into_iter().collect();
    let mut class_counts: BTreeMap<i64, u64> = class_ids.iter().map(|c| (*c, 0)).collect();
    let mut class_bounds: BTreeMap<i64, Vec<(usize, usize)>> = BTreeMap::new();
    let mut edges = BTreeSet::new();
    let mut labelled = 0u64;
    for ((classes, count), bound) in values.classes.iter().zip(&counts).zip(&bounds) {
        if *count == 0 || classes.is_empty() {
            continue;
        }
        labelled += count;
        for (i, a) in classes.iter().enumerate() {
            *class_counts.get_mut(a).expect("a class of the map") += count;
            for b in &classes[i + 1..] {
                edges.insert((*a, *b));
            }
            let bound = bound.as_ref().expect("a value with voxels has a box");
            class_bounds
                .entry(*a)
                .and_modify(|held| {
                    for (h, b) in held.iter_mut().zip(bound) {
                        *h = (h.0.min(b.0), h.1.max(b.1));
                    }
                })
                .or_insert_with(|| bound.clone());
        }
    }
    let n_voxels: usize = shape.iter().product();
    let localized = class_bounds
        .iter()
        .filter(|(_, bound)| {
            let volume: u64 = bound.iter().map(|(lo, hi)| (hi - lo) as u64).product();
            volume as f64 <= LOCALIZED_BBOX_FRACTION * n_voxels as f64
        })
        .map(|(c, _)| *c)
        .collect();
    let colouring = greedy_colour(&class_ids, &edges);
    let stats = OverlapStats {
        class_ids,
        spatial_shape: shape,
        counts: class_counts,
        edges,
        colouring,
        localized,
        n_labelled_voxels: labelled,
    };
    Ok(Analysis { stats, values, counts, bounds })
}

/// Encode the volume as `kind`, as
/// [`encode_masks`](super::encode::encode_masks) encodes the masks its values
/// imply.  `in_band` paints the ignore region into a `labelmap` or
/// `layers`; `start_id` is the first id an `instances` encoding mints.
pub(crate) fn encode_labelmap_input<T: LabelValue>(
    data: ArrayViewD<'_, T>,
    analysis: &Analysis,
    kind: &str,
    in_band: bool,
    start_id: Option<u64>,
) -> Result<Payload> {
    let stats = &analysis.stats;
    let classes = &analysis.values.classes;
    let painted = |at: usize| in_band && analysis.values.ignore == Some(at);
    let ignored = in_band && analysis.values.ignore.is_some_and(|at| analysis.counts[at] > 0);
    match kind {
        "labelmap" => {
            if let Some((a, b)) = stats.edges.iter().next() {
                return Err(Error::coded(
                    "E404",
                    format!(
                        "labelmap requires mutually exclusive classes; classes {a} and {b} share a value of the \
                         labelmap. Use `layers` or `bitmask` for overlapping structures."
                    ),
                ));
            }
            let table: Vec<u16> = (0..classes.len())
                .map(|at| match classes[at].as_slice() {
                    [class] => *class as u16,
                    _ if painted(at) => IGNORE_ID as u16,
                    _ => 0,
                })
                .collect();
            let stored = if label_dtype_size(&stats.class_ids, ignored) == 1 {
                NdArray::U8(analysis.map(data, |at| table[at] as u8))
            } else {
                NdArray::U16(analysis.map(data, |at| table[at]))
            };
            let mut p = Payload::new("labelmap");
            p.datasets.insert("data".into(), stored.into());
            p.class_ids = stats.class_ids.clone();
            Ok(p)
        }
        "layers" => {
            let buckets = layers_from_colouring(&stats.colouring);
            let n_layers = buckets.len().max(1);
            // With no class there is no layer to paint, as `encode_layers`
            // paints the ignore region a bucket at a time.
            let paint = !buckets.is_empty();
            let table: Vec<Vec<u16>> = (0..classes.len())
                .map(|at| {
                    let mut row = vec![0u16; n_layers];
                    if painted(at) && paint {
                        row.fill(IGNORE_ID as u16);
                    }
                    for class in &classes[at] {
                        row[stats.colouring[class]] = *class as u16;
                    }
                    row
                })
                .collect();
            let ignored = ignored && paint;
            let width = buckets.iter().map(Vec::len).max().unwrap_or(1).max(1);
            let mut layer_table = ArrayD::<u16>::zeros(IxDyn(&[n_layers, width]));
            for (layer, bucket) in buckets.iter().enumerate() {
                for (k, cid) in bucket.iter().enumerate() {
                    layer_table[[layer, k]] = *cid as u16;
                }
            }
            let stored = if label_dtype_size(&stats.class_ids, ignored) == 1 {
                NdArray::U8(stack(data, analysis, n_layers, |at, layer| table[at][layer] as u8))
            } else {
                NdArray::U16(stack(data, analysis, n_layers, |at, layer| table[at][layer]))
            };
            let mut p = Payload::new("layers");
            p.datasets.insert("data".into(), stored.into());
            p.datasets.insert("layer_class_ids".into(), NdArray::U16(layer_table).into());
            p.stacked_axes = 1;
            p.class_ids = stats.class_ids.clone();
            Ok(p)
        }
        "bitmask" => {
            let position: BTreeMap<i64, usize> = stats.class_ids.iter().enumerate().map(|(i, c)| (*c, i)).collect();
            let n_planes = stats.class_ids.len().div_ceil(BITS_PER_PLANE).max(1);
            let table: Vec<Vec<u64>> = classes
                .iter()
                .map(|set| {
                    let mut words = vec![0u64; n_planes];
                    for class in set {
                        let p = position[class];
                        words[p / BITS_PER_PLANE] |= 1u64 << (p % BITS_PER_PLANE);
                    }
                    words
                })
                .collect();
            let ids = ArrayD::from_shape_vec(
                IxDyn(&[stats.class_ids.len()]),
                stats.class_ids.iter().map(|c| *c as u16).collect(),
            )?;
            let mut p = Payload::new("bitmask");
            p.datasets
                .insert("data".into(), NdArray::U64(stack(data, analysis, n_planes, |at, k| table[at][k])).into());
            p.datasets.insert("bit_class_ids".into(), NdArray::U16(ids).into());
            p.stacked_axes = 1;
            p.class_ids = stats.class_ids.clone();
            Ok(p)
        }
        "instances" => {
            let start_id = start_id.unwrap_or(1);
            let mut objects = Vec::new();
            for (i, class) in stats.class_ids.iter().enumerate() {
                if stats.counts[class] == 0 {
                    continue;
                }
                let holding: Vec<bool> = classes.iter().map(|set| set.contains(class)).collect();
                let bound = analysis.class_bounds(class);
                let mut window = data.view();
                for (axis, (lo, hi)) in bound.iter().enumerate() {
                    window.slice_axis_inplace(Axis(axis), Slice::from(*lo..*hi));
                }
                let crop = analysis.map(window, |at| holding[at]);
                let pairs: Vec<(i64, i64)> = bound.iter().map(|(a, b)| (*a as i64, *b as i64)).collect();
                objects.push(InstanceInput {
                    class_id: *class,
                    instance_id: minted_id(start_id, i)?,
                    bbox: Some(slices_to_box(&pairs).into_iter().map(f64::from).collect()),
                    crop: Some(crop),
                    ..Default::default()
                });
            }
            encode_instances(&objects, Some(&stats.spatial_shape), true, Some(&stats.class_ids))
        }
        // Each class's plane 1.0 where it is, as `encode_masks` writes masks:
        // 0 and 1 are exact in float16, which `storage_dtype` keeps, and the
        // map is not normalized.
        "probmap" => {
            let (one, zero) = (half::f16::ONE, half::f16::ZERO);
            let table: Vec<Vec<half::f16>> = classes
                .iter()
                .map(|set| stats.class_ids.iter().map(|c| if set.contains(c) { one } else { zero }).collect())
                .collect();
            let n = stats.class_ids.len();
            let mut p = Payload::new("probmap");
            p.datasets.insert("data".into(), NdArray::F16(stack(data, analysis, n, |at, k| table[at][k])).into());
            p.attrs = vec![("normalized".to_string(), AttrValue::Bool(false))];
            p.stacked_axes = 1;
            p.class_ids = stats.class_ids.clone();
            Ok(p)
        }
        other => Err(Error::coded("E404", format!("a labelmap cannot be encoded as {}", repr_str(other)))),
    }
}

impl Analysis {
    /// The bounding box of `class`'s voxels: the union of its values' boxes.
    fn class_bounds(&self, class: &i64) -> Vec<(usize, usize)> {
        let mut out: Option<Vec<(usize, usize)>> = None;
        for (set, bound) in self.values.classes.iter().zip(&self.bounds) {
            let (true, Some(bound)) = (set.contains(class), bound) else { continue };
            out = Some(match out {
                None => bound.clone(),
                Some(held) => held.iter().zip(bound).map(|(a, b)| (a.0.min(b.0), a.1.max(b.1))).collect(),
            });
        }
        out.expect("a class with voxels has a box")
    }
}

/// `n` planes of the volume's shape, stacked on a leading axis: plane `k` at
/// a voxel is `f(index, k)` of the voxel's value index.  The volume is read
/// once, each voxel written to every plane.
fn stack<T: LabelValue, U: Copy + Default>(
    data: ArrayViewD<'_, T>,
    analysis: &Analysis,
    n: usize,
    f: impl Fn(usize, usize) -> U,
) -> ArrayD<U> {
    let mut shape = vec![n];
    shape.extend(data.shape());
    let voxels = data.len();
    let mut out = vec![U::default(); n * voxels];
    let mut lookup = Lookup::new(&analysis.values);
    // `iter` visits a view in logical (C) order, whatever its strides.
    for (i, value) in data.iter().enumerate() {
        let at = lookup.at(*value);
        for k in 0..n {
            out[k * voxels + i] = f(at, k);
        }
    }
    ArrayD::from_shape_vec(IxDyn(&shape), out).expect("n planes of the volume's shape")
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use ndarray::{ArrayD, IxDyn};

    use super::super::select::OverlapStats;
    use super::LabelValue;
    use crate::array::NdArray;
    use crate::h5::{attrs, data, file, ops};
    use crate::labels::{ClassKey, LabelClass, LabelSet};
    use crate::rng::Rng;
    use crate::sample::writer::{create, Annotated, GridOptions, ImageOptions, SampleWriter};
    use crate::sample::writer_annotations::{
        AnnotationOptions, LabelmapSource, SegmentationOptions, SegmentationSource,
    };
    use crate::Result;

    const SHAPE: [usize; 3] = [6, 10, 12];
    const ENCODINGS: [&str; 6] = ["auto", "labelmap", "layers", "bitmask", "instances", "probmap"];

    /// A labelmap and what it says: each value's classes (`None`: its own),
    /// its ignore value and the classes examined beyond those it names.
    struct Case<T> {
        name: &'static str,
        data: ArrayD<T>,
        values: Option<Vec<(i64, Vec<i64>)>>,
        ignore_value: Option<i64>,
        examined: Option<Vec<i64>>,
    }

    fn sample(path: &Path, shape: &[usize]) -> SampleWriter {
        let mut w = create(path, Some("s"), None, "portable", &[]).unwrap();
        let named = (1..=8).map(|c| LabelClass::new(c, format!("c{c}"), format!("C{c}")).unwrap()).collect();
        w.label_set(LabelSet::new("v", named, "1.0.0", vec![], vec![], "inline", None, None).unwrap());
        let grid: Vec<i64> = shape.iter().map(|n| *n as i64).collect();
        w.add_grid("g", &grid, &[1.0, 1.0, 1.0], GridOptions::default()).unwrap();
        let image = NdArray::from_vec(shape, vec![0_i16; shape.iter().product()]).unwrap();
        w.add_image("CT", &image, "g", "CT", ImageOptions::default()).unwrap();
        w
    }

    fn options<T>(case: &Case<T>, encoding: &str) -> SegmentationOptions {
        let annotated = match &case.examined {
            Some(ids) => Annotated::Classes(ids.iter().map(|c| ClassKey::Id(*c)).collect()),
            None => Annotated::AllGiven,
        };
        SegmentationOptions {
            encoding: Some(encoding.into()),
            common: AnnotationOptions { annotated_classes: annotated, ..Default::default() },
            ..Default::default()
        }
    }

    type ClassMasks = Vec<(ClassKey, ArrayD<bool>)>;

    /// The masks the labelmap's values imply, and its ignore region.
    fn masks<T: LabelValue>(case: &Case<T>) -> (ClassMasks, Option<ArrayD<bool>>) {
        let values: Vec<(i64, Vec<i64>)> = match &case.values {
            Some(values) => values.clone(),
            None => {
                let mut met: Vec<i64> = case.data.iter().filter_map(|v| v.to_i64()).collect();
                met.sort_unstable();
                met.dedup();
                met.into_iter().filter(|v| *v != 0 && Some(*v) != case.ignore_value).map(|v| (v, vec![v])).collect()
            }
        };
        let mut classes: Vec<i64> = values.iter().flat_map(|(_, c)| c.clone()).collect();
        classes.sort_unstable();
        classes.dedup();
        let masks = classes
            .iter()
            .map(|class| {
                let holding: Vec<i64> = values.iter().filter(|(_, c)| c.contains(class)).map(|(v, _)| *v).collect();
                let mask = case.data.mapv(|v| holding.contains(&v.to_i64().unwrap()));
                (ClassKey::Id(*class), mask.as_standard_layout().into_owned())
            })
            .collect();
        let ignore =
            case.ignore_value.map(|i| case.data.mapv(|v| v.to_i64() == Some(i)).as_standard_layout().into_owned());
        (masks, ignore)
    }

    fn by_masks<T: LabelValue>(path: &Path, case: &Case<T>, encoding: &str) -> Result<(String, Option<OverlapStats>)> {
        let mut w = sample(path, case.data.shape());
        let (masks, ignore) = masks(case);
        let options = SegmentationOptions { ignore, ..options(case, encoding) };
        let written = w.add_segmentation("m", "g", SegmentationSource::Masks(masks), options)?;
        w.commit(true)?;
        Ok(written)
    }

    fn by_labelmap<T: LabelValue>(
        path: &Path,
        case: &Case<T>,
        encoding: &str,
    ) -> Result<(String, Option<OverlapStats>)> {
        let mut w = sample(path, case.data.shape());
        let mut source = LabelmapSource::new(case.data.view());
        if let Some(values) = &case.values {
            source = source.values(values.iter().map(|(v, c)| (*v, c.iter().map(|c| ClassKey::Id(*c)).collect())));
        }
        if let Some(value) = case.ignore_value {
            source = source.ignore_value(value);
        }
        let written = w.add_segmentation_labelmap("m", "g", source, options(case, encoding))?;
        w.commit(true)?;
        Ok(written)
    }

    /// Every attribute, dataset (its values, type and chunks) and subgroup
    /// below `name`, the same in both files.
    fn same_group(left: &hdf5::Group, right: &hdf5::Group, at: &str) {
        let names = |g: &hdf5::Location| attrs::names(g).unwrap();
        assert_eq!(names(left), names(right), "{at}: attributes");
        for name in names(left) {
            assert_eq!(attrs::read(left, &name).unwrap(), attrs::read(right, &name).unwrap(), "{at}@{name}");
        }
        assert_eq!(ops::members(left).unwrap(), ops::members(right).unwrap(), "{at}: members");
        for name in ops::members(left).unwrap() {
            let here = format!("{at}/{name}");
            match (ops::child_group(left, &name), ops::child_group(right, &name)) {
                (Some(l), Some(r)) => same_group(&l, &r, &here),
                (None, None) => {
                    let l = ops::child_dataset(left, &name).unwrap();
                    let r = ops::child_dataset(right, &name).unwrap();
                    assert_eq!(data::read(&l).unwrap(), data::read(&r).unwrap(), "{here}");
                    assert_eq!(l.chunk(), r.chunk(), "{here}: chunks");
                    assert_eq!(names(&l), names(&r), "{here}: attributes");
                    for attr in names(&l) {
                        assert_eq!(attrs::read(&l, &attr).unwrap(), attrs::read(&r, &attr).unwrap(), "{here}@{attr}");
                    }
                }
                _ => panic!("{here} is a group in one file only"),
            }
        }
    }

    fn check<T: LabelValue>(case: &Case<T>) {
        let dir = tempfile::tempdir().unwrap();
        for encoding in ENCODINGS {
            let what = format!("{} as {encoding}", case.name);
            let (left, right) =
                (dir.path().join(format!("{encoding}-m.medh5")), dir.path().join(format!("{encoding}-l.medh5")));
            match (by_masks(&left, case, encoding), by_labelmap(&right, case, encoding)) {
                (Ok(masks), Ok(labelmap)) => {
                    assert_eq!(masks, labelmap, "{what}: kind and stats");
                    let (l, r) = (file::open_read(&left).unwrap(), file::open_read(&right).unwrap());
                    let (l, r) = (l.group("annotations").unwrap(), r.group("annotations").unwrap());
                    assert_eq!(ops::members(&l).unwrap(), ops::members(&r).unwrap(), "{what}: annotations");
                    for name in ops::members(&l).unwrap() {
                        same_group(&l.group(&name).unwrap(), &r.group(&name).unwrap(), &format!("{what}: {name}"));
                    }
                }
                // Only overlapping classes are refused, and only as a labelmap.
                (Err(masks), Err(labelmap)) => {
                    assert_eq!((encoding, masks.code()), ("labelmap", Some("E404")), "{what}: {masks}");
                    assert_eq!(masks.code(), labelmap.code(), "{what}: {masks} / {labelmap}");
                }
                (masks, labelmap) => panic!("{what}: masks {masks:?}, labelmap {labelmap:?}"),
            }
        }
    }

    fn random<T>(seed: u64, mut value: impl FnMut(i64) -> T) -> ArrayD<T> {
        let mut rng = Rng::new(seed);
        ArrayD::from_shape_simple_fn(IxDyn(&SHAPE), || value(rng.integer(0, 1000).unwrap()))
    }

    /// The labelmap route writes what the masks route writes for the masks
    /// its values imply, byte for byte, under every encoding either picks or
    /// is asked for: its own classes, regions over values, an ignore value,
    /// examined classes it never holds, a sparse volume `auto` writes as
    /// instances, and values of every integer width.
    #[test]
    fn s7_labelmap_source_matches_masks_source() {
        check(&Case {
            name: "identity",
            data: random(1, |r| (r % 6) as u8),
            values: None,
            ignore_value: None,
            examined: Some(vec![1, 2, 3, 4, 5, 7]),
        });
        // Fortran order with an axis flipped, as `get_fdata().T[:, ::-1]`.
        let mut strided = random(4, |r| (r % 7) as u8).reversed_axes();
        strided.invert_axis(ndarray::Axis(1));
        check(&Case { name: "strided", data: strided, values: None, ignore_value: Some(6), examined: None });
        check(&Case {
            name: "regions",
            data: random(2, |r| if r < 30 { 9 } else { (r % 5) as i16 }),
            values: Some(vec![(1, vec![1, 5]), (2, vec![2, 5]), (3, vec![3]), (4, vec![4, 6])]),
            ignore_value: Some(9),
            examined: None,
        });
        check(&Case {
            name: "exclusive with ignore",
            data: random(3, |r| if r < 50 { 255 } else { (r % 4) as u32 }),
            values: Some(vec![(1, vec![1]), (2, vec![2]), (3, vec![8])]),
            ignore_value: Some(255),
            examined: Some(vec![1, 2, 6, 8]),
        });
        // Two small blobs in a volume far larger, which `auto` writes as
        // instances.
        let mut sparse = ArrayD::<u16>::zeros(IxDyn(&[20, 40, 40]));
        sparse.slice_each_axis_mut(|a| ndarray::Slice::from(a.axis.0 + 1..a.axis.0 + 3)).fill(3);
        sparse[[19, 39, 39]] = 7;
        sparse[[19, 38, 39]] = 7;
        let case = Case { name: "sparse", data: sparse.clone(), values: None, ignore_value: None, examined: None };
        check(&case);
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(by_labelmap(&dir.path().join("s.medh5"), &case, "auto").unwrap().0, "instances");
        let wide = sparse.mapv(|v| i64::from(v) + 1);
        check(&Case { name: "wide", data: wide, values: None, ignore_value: Some(-1), examined: None });
        // An ignore value the volume never holds.
        check(&Case { name: "unheld ignore", data: sparse, values: None, ignore_value: Some(4), examined: None });
        check(&Case {
            name: "empty",
            data: ArrayD::<i8>::zeros(IxDyn(&SHAPE)),
            values: None,
            ignore_value: None,
            examined: Some(vec![2]),
        });
    }

    fn refused(source: LabelmapSource<'_, i32>, options: SegmentationOptions) -> String {
        let dir = tempfile::tempdir().unwrap();
        let mut w = sample(&dir.path().join("s.medh5"), &SHAPE);
        w.add_segmentation_labelmap("m", "g", source, options).unwrap_err().to_string()
    }

    /// What a labelmap cannot say is refused before anything is written.
    #[test]
    fn s7_labelmap_source_refusals() {
        let data = ArrayD::<i32>::from_shape_fn(IxDyn(&SHAPE), |i| (i[2] % 3) as i32);
        let view = || LabelmapSource::new(data.view());
        let ids = |v: &[i64]| v.iter().map(|c| ClassKey::Id(*c)).collect::<Vec<_>>();
        let plain = SegmentationOptions::default;
        for (source, options, expected) in [
            (view().values([(1, ids(&[1]))]), plain(), "[E404] the labelmap holds 2 at voxel (0, 0, 2)"),
            (view().ignore_value(0), plain(), "[E404] ignore_value=0"),
            (view().values([(0, ids(&[1]))]), plain(), "[E404] label_values[0]"),
            (view().values([(1, ids(&[1])), (2, ids(&[]))]), plain(), "[E404] label_values[2] names no class"),
            (view().values([(1, ids(&[1])), (1, ids(&[2]))]), plain(), "[E404] label_values[1] is given twice"),
            (
                view().values([(1, ids(&[1])), (2, ids(&[2]))]).ignore_value(2),
                plain(),
                "[E404] label_values[2]: 2 is the ignore value",
            ),
            (view().values([(1, ids(&[1])), (2, ids(&[70_000]))]), plain(), "[E30"),
            (
                view(),
                SegmentationOptions { ignore: Some(ArrayD::from_elem(IxDyn(&SHAPE), false)), ..plain() },
                "[E404] annotation 'm': a labelmap gives its ignore region as a value",
            ),
            (
                view().ignore_value(7),
                SegmentationOptions { ignore_mask: Some("fov".into()), ..plain() },
                "[E404] annotation 'm': pass ignore_value=",
            ),
            (view(), SegmentationOptions { threshold: Some(0.5), ..plain() }, "[E404] annotation 'm': threshold="),
            (
                view(),
                SegmentationOptions { encoding: Some("mask".into()), ..plain() },
                "[E404] annotation 'm': encoding='mask'",
            ),
        ] {
            let got = refused(source, options);
            assert!(got.starts_with(expected), "{got}");
        }
        let flat = ArrayD::<i32>::zeros(IxDyn(&[6, 10]));
        let got = refused(LabelmapSource::new(flat.view()), SegmentationOptions::default());
        assert!(got.starts_with("[E405] annotation 'm': the labelmap has shape (6, 10)"), "{got}");
        let past = ArrayD::<u64>::from_elem(IxDyn(&SHAPE), u64::MAX);
        let dir = tempfile::tempdir().unwrap();
        let mut w = sample(&dir.path().join("s.medh5"), &SHAPE);
        let error = w.add_segmentation_labelmap("m", "g", LabelmapSource::new(past.view()), plain()).unwrap_err();
        assert!(error.to_string().contains("a value past int64"), "{error}");
    }
}
