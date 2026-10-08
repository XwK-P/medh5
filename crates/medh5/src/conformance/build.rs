//! The cases: what a conforming file looks like, and what a broken one reports.
//!
//! Valid files are written by the public writer.  Invalid ones are the
//! writer's output edited afterwards --- the writer refuses to produce them ---
//! which leaves their digests covering the pre-edit bytes; [`Case::mutated`]
//! says so.  The order here is the order `expected.json` lists, and it is the
//! 1.x order: a case keeps its position across releases.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ndarray::{Array2, ArrayD, IxDyn, Slice};
use serde_json::{json, Map, Value};

use super::Case;
use crate::annotations::{Assertions, InstanceInput, Polygon};
use crate::array::{DType, NdArray};
use crate::collection::{pack, SAMPLES_GROUP};
use crate::geometry::multiscale::derive_level_grid;
use crate::h5::attrs::{self, AttrValue};
use crate::h5::data::{self, Layout};
use crate::h5::ops;
use crate::integrity::digest::{compute_content_id, stamp_digests};
use crate::json::{dumps, Style};
use crate::labels::{ClassKey, LabelClass, LabelSet, Skeleton};
use crate::rng::Rng;
use crate::sample::{
    attr_name_map_of, create, Annotated, AnnotationOptions, GridOptions, ImageOptions, ObjectFields, Placement,
    QualityArg, SampleWriter, SegmentationOptions, SegmentationSource, TransformSpec,
};
use crate::{Error, Result, VERSION};

/// The seed every case draws its voxels from.
pub const SEED: u64 = 20260815;

const SHAPE: [usize; 3] = [16, 24, 24];
const SPACING: [f64; 3] = [1.5, 0.8, 0.8];
const ORIGIN: [f64; 3] = [-12.0, -9.6, -9.6];
const FRAME0: &str = "pseudo:frame-100";
const FRAME1: &str = "pseudo:frame-101";

/// The classes `_seg_base` annotates by default: liver, spleen, and a lesion
/// overlapping the liver --- two layers' worth.
const ORGANS: [(i64, [usize; 3]); 3] = [(1, [2, 2, 2]), (2, [2, 12, 2]), (3, [4, 4, 4])];
/// Mutually exclusive classes, for a labelmap.
const EXCLUSIVE: [(i64, [usize; 3]); 3] = [(1, [2, 2, 2]), (2, [2, 12, 2]), (4, [9, 2, 12])];

// -- case construction ------------------------------------------------------------

fn case(
    name: &str,
    description: &str,
    clause: &str,
    build: impl Fn(&Path) -> Result<()> + Send + Sync + 'static,
) -> Case {
    Case {
        name: name.into(),
        description: description.into(),
        clause: clause.into(),
        build: Some(Arc::new(build)),
        level: "semantic".into(),
        errors: Vec::new(),
        warnings: Vec::new(),
        suffix: ".medh5".into(),
        mutated: false,
    }
}

impl Case {
    fn errors(mut self, codes: &[&str]) -> Case {
        self.errors = codes.iter().map(|c| c.to_string()).collect();
        self
    }

    fn warnings(mut self, codes: &[&str]) -> Case {
        self.warnings = codes.iter().map(|c| c.to_string()).collect();
        self
    }

    fn level(mut self, level: &str) -> Case {
        self.level = level.into();
        self
    }

    fn collection(mut self) -> Case {
        self.suffix = ".medh5c".into();
        self
    }

    fn mutated(mut self) -> Case {
        self.mutated = true;
        self
    }
}

/// A case made by building `base` and then editing it with `mutation`.
fn invalid(
    name: &str,
    description: &str,
    clause: &str,
    codes: &[&str],
    base: impl Fn(&Path) -> Result<()> + Send + Sync + 'static,
    mutation: impl Fn(&hdf5::Group) -> Result<()> + Send + Sync + 'static,
) -> Case {
    case(name, description, clause, move |path| {
        base(path)?;
        mutate(path, &mutation)
    })
    .errors(codes)
    .mutated()
}

/// [`invalid`] on the minimal sample.
fn invalid_core(
    name: &str,
    description: &str,
    clause: &str,
    codes: &[&str],
    mutation: impl Fn(&hdf5::Group) -> Result<()> + Send + Sync + 'static,
) -> Case {
    invalid(name, description, clause, codes, |p| base(p, &SHAPE), mutation)
}

/// [`invalid`] on the segmentation sample, which always warns W912.
fn invalid_seg(
    name: &str,
    description: &str,
    clause: &str,
    codes: &[&str],
    mutation: impl Fn(&hdf5::Group) -> Result<()> + Send + Sync + 'static,
) -> Case {
    invalid(name, description, clause, codes, |p| seg_base(p, "auto", &ORGANS, false), mutation).warnings(&["W912"])
}

// -- small helpers ------------------------------------------------------------------

fn fields(value: Value) -> Map<String, Value> {
    match value {
        Value::Object(m) => m,
        _ => Map::new(),
    }
}

fn strings(values: &[&str]) -> Vec<String> {
    values.iter().map(|v| v.to_string()).collect()
}

fn ids(values: &[i64]) -> Vec<ClassKey> {
    values.iter().map(|v| ClassKey::Id(*v)).collect()
}

fn dims(shape: &[usize]) -> Vec<i64> {
    shape.iter().map(|v| *v as i64).collect()
}

fn stem(path: &Path) -> String {
    path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
}

fn matrix(rows: &[&[f64]]) -> Result<ArrayD<f64>> {
    let cols = rows.first().map_or(0, |r| r.len());
    let flat: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    Ok(ArrayD::from_shape_vec(IxDyn(&[rows.len(), cols]), flat)?)
}

fn eye4() -> ArrayD<f64> {
    Array2::<f64>::eye(4).into_dyn()
}

/// Random `int16` values in `[-1000, 1500)`: a CT volume.
fn ct_volume(rng: &mut Rng, shape: &[usize]) -> Result<NdArray> {
    integers(rng, -1000, 1500, shape, DType::I16)
}

/// Random integers in `[low, high)`, as `dtype`.
fn integers(rng: &mut Rng, low: i64, high: i64, shape: &[usize], dtype: DType) -> Result<NdArray> {
    let n = shape.iter().product();
    NdArray::from_vec(shape, rng.integers(low, high, n)?)?.reshape(shape).map(|a| a.astype(dtype))
}

/// Random floats in `[0, 1)`.
fn uniform(rng: &mut Rng, shape: &[usize]) -> Result<ArrayD<f64>> {
    let n: usize = shape.iter().product();
    let values: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
    Ok(ArrayD::from_shape_vec(IxDyn(shape), values)?)
}

/// Random `float32` values in `[0, 1)`.
fn uniform32(rng: &mut Rng, shape: &[usize]) -> Result<NdArray> {
    Ok(NdArray::from(uniform(rng, shape)?.mapv(|v| v as f32)))
}

/// `mask[c0:c0+e0, c1:c1+e1, ...] = True`, clipped like a NumPy slice.
fn block(shape: &[usize], corner: &[usize], extent: &[usize]) -> ArrayD<bool> {
    let mut mask = ArrayD::from_elem(IxDyn(shape), false);
    mask.slice_each_axis_mut(|ax| {
        let i = ax.axis.index();
        let n = shape[i];
        Slice::from(corner[i].min(n)..(corner[i] + extent[i]).min(n))
    })
    .fill(true);
    mask
}

/// `_blocks`: one 6x8x8 block per class, at the given corners.
fn blocks(shape: &[usize], spec: &[(i64, [usize; 3])]) -> Vec<(ClassKey, ArrayD<bool>)> {
    spec.iter().map(|(class_id, corner)| (ClassKey::Id(*class_id), block(shape, corner, &[6, 8, 8]))).collect()
}

fn label_set_with(skeletons: Vec<Skeleton>) -> Result<LabelSet> {
    let class = |id: i64, key: &str, name: &str, parents: Vec<i64>, category: &str, color: [i64; 4]| {
        LabelClass::build(
            id,
            key.into(),
            name.into(),
            parents,
            Some(category.into()),
            Some(color.to_vec()),
            Vec::new(),
            None,
            Map::new(),
        )
    };
    LabelSet::new(
        "conformance-v1",
        vec![
            class(1, "liver", "Liver", vec![], "organ", [200, 90, 70, 255])?,
            class(2, "spleen", "Spleen", vec![], "organ", [190, 60, 60, 255])?,
            class(3, "lesion", "Lesion", vec![1], "lesion", [255, 214, 64, 255])?,
            class(4, "vessel", "Vessel", vec![], "vessel", [80, 150, 220, 255])?,
        ],
        "1.0.0",
        Vec::new(),
        skeletons,
        "inline",
        None,
        None,
    )
}

/// The corpus vocabulary: liver, spleen, a lesion under the liver, vessel.
fn label_set() -> Result<LabelSet> {
    label_set_with(Vec::new())
}

fn quantitative(prov: Option<String>) -> ImageOptions {
    ImageOptions { value_type: Some("quantitative".into()), value_units: Some("HU".into()), prov, ..Default::default() }
}

fn on_timepoint(timepoint: &str) -> GridOptions {
    GridOptions { timepoint: Some(timepoint.into()), ..Default::default() }
}

fn psi() -> Map<String, Value> {
    fields(json!({"method": "dicom-psi-profile"}))
}

/// Create, fill, commit: `with medh5.create(...) as w:`.
fn write(path: &Path, subject: Option<&str>, body: impl FnOnce(&mut SampleWriter) -> Result<()>) -> Result<()> {
    let mut writer = create(path, Some(&stem(path)), subject, "portable", &[])?;
    body(&mut writer)?;
    writer.commit(true)?;
    Ok(())
}

// -- editing a committed file -----------------------------------------------------------

/// Open `path` read-write, edit it, close it.
fn mutate(path: &Path, edit: impl FnOnce(&hdf5::Group) -> Result<()>) -> Result<()> {
    crate::h5::init();
    let file = hdf5::File::open_rw(path)?;
    let root = file.as_group()?;
    edit(&root)?;
    drop(root);
    file.close()?;
    Ok(())
}

/// Run `op` on the object at `path` (`""` is the root).
fn on_object<T>(root: &hdf5::Group, path: &str, op: impl FnOnce(&hdf5::Location) -> Result<T>) -> Result<T> {
    if path.is_empty() {
        return op(root);
    }
    match ops::node_kind(root, path) {
        Some(ops::NodeKind::Group) => {
            let group = root.group(path)?;
            op(&group)
        }
        Some(ops::NodeKind::Dataset) => {
            let dataset = root.dataset(path)?;
            op(&dataset)
        }
        _ => Err(Error::Key(format!("no object {path} in this file"))),
    }
}

fn set_attr(root: &hdf5::Group, path: &str, name: &str, value: AttrValue) -> Result<()> {
    on_object(root, path, |obj| attrs::write(obj, name, &value))
}

fn set_str(root: &hdf5::Group, path: &str, name: &str, value: &str) -> Result<()> {
    set_attr(root, path, name, AttrValue::Str(value.into()))
}

fn del_attr(root: &hdf5::Group, path: &str, name: &str) -> Result<()> {
    on_object(root, path, |obj| attrs::delete(obj, name))
}

fn uint16s(values: &[u16]) -> Result<AttrValue> {
    Ok(AttrValue::Array(NdArray::from_vec(&[values.len()], values.to_vec())?))
}

fn split_path(path: &str) -> (&str, &str) {
    path.rsplit_once('/').unwrap_or(("", path))
}

fn parent_of(root: &hdf5::Group, path: &str) -> Result<(hdf5::Group, String)> {
    let (parent, name) = split_path(path);
    let group = if parent.is_empty() { root.clone() } else { root.group(parent)? };
    Ok((group, name.to_string()))
}

/// Replace a dataset by a contiguous, unfiltered copy of its values in
/// `dtype`, keeping its attributes --- what `del d; create_dataset(data=...)`
/// does in h5py.
fn rewrite_dataset(root: &hdf5::Group, path: &str, dtype: Option<DType>) -> Result<()> {
    let (group, name) = parent_of(root, path)?;
    let old = group.dataset(&name)?;
    let values = data::read(&old)?;
    let values = match dtype {
        Some(d) => values.astype(d),
        None => values,
    };
    let staging = format!("{name}.corpus-staging");
    let new = data::create(&group, &staging, &values, &Layout::contiguous())?;
    for attr in attrs::names(&old)? {
        attrs::copy_raw(&old, &new, &attr)?;
    }
    drop(old);
    drop(new);
    group.unlink(&name)?;
    group.relink(&staging, &name)?;
    Ok(())
}

/// Read a dataset, edit its values as `f64`, write them back in place.
fn edit_values(root: &hdf5::Group, path: &str, edit: impl FnOnce(&mut ArrayD<f64>)) -> Result<()> {
    let ds = root.dataset(path)?;
    let stored = data::read(&ds)?;
    let mut values = stored.to_f64();
    edit(&mut values);
    let written = NdArray::from(values).astype(stored.dtype());
    data::write_region(&ds, &written, &vec![0; written.ndim()])
}

/// Re-read `/meta`, edit the document, store it back as `json.dumps` would.
fn set_meta(root: &hdf5::Group, edit: impl FnOnce(&mut Map<String, Value>)) -> Result<()> {
    let text = data::read_scalar_string(&root.dataset("meta")?)?;
    let mut doc = fields(serde_json::from_str(&text)?);
    edit(&mut doc);
    root.unlink("meta")?;
    data::create_scalar_string(root, "meta", &dumps(&Value::Object(doc), Style::PYTHON))?;
    Ok(())
}

/// `doc.setdefault(key, default)`, returning the member as an object.
fn member<'a>(doc: &'a mut Map<String, Value>, key: &str, default: Value) -> &'a mut Map<String, Value> {
    let entry = doc.entry(key.to_string()).or_insert(default);
    if !entry.is_object() {
        *entry = Value::Object(Map::new());
    }
    entry.as_object_mut().expect("just made an object")
}

/// The first activity, inventing one when the document has none.
fn first_activity(doc: &mut Map<String, Value>) -> &mut Map<String, Value> {
    let provenance = member(doc, "provenance", json!({}));
    let activities = provenance.entry("activities").or_insert_with(|| json!([]));
    if !activities.is_array() {
        *activities = json!([]);
    }
    let list = activities.as_array_mut().expect("just made a list");
    if list.is_empty() {
        list.push(json!({"id": "a1", "type": "import"}));
    }
    if !list[0].is_object() {
        list[0] = json!({});
    }
    list[0].as_object_mut().expect("just made an object")
}

fn label_classes(doc: &mut Map<String, Value>) -> &mut Vec<Value> {
    let label_set = member(doc, "label_set", json!({}));
    let classes = label_set.entry("classes").or_insert_with(|| json!([]));
    if !classes.is_array() {
        *classes = json!([]);
    }
    classes.as_array_mut().expect("just made a list")
}

/// Re-stamp digests and `content_id` after a legitimate authoring edit.
///
/// Used only where the edit is something a writer would do; invalid cases
/// deliberately skip it, which is what makes E701/E702 reachable at all.
fn restamp(path: &Path) -> Result<()> {
    mutate(path, |root| {
        stamp_digests(root, "sha256", &["index"], false)?;
        let names = attr_name_map_of(root)?;
        let content_id = compute_content_id(root, &names, "sha256", None)?;
        attrs::write(root, "content_id", &AttrValue::Str(content_id))
    })
}

// -- builders -------------------------------------------------------------------------

/// A valid, minimal-but-complete sample: one grid, one image, one timepoint.
fn base(path: &Path, shape: &[usize]) -> Result<()> {
    let mut rng = Rng::new(SEED);
    write(path, Some("subj-A"), |w| {
        w.identity(fields(json!({"sex": "F", "bodypart": "abdomen"})))?;
        w.cohort(fields(json!({"dataset_id": "conformance", "site_id": "site-A"})))?;
        w.add_timepoint("tp0", fields(json!({"label": "baseline"})))?;
        let tool = w.software("medh5", Some(VERSION), Map::new())?;
        let act = w.activity("import", Some(&tool.id), None, fields(json!({"tool": "conformance corpus"})))?;
        w.add_grid(
            "ct",
            &dims(shape),
            &SPACING,
            GridOptions {
                origin: Some(ORIGIN.to_vec()),
                timepoint: Some("tp0".into()),
                frame_uid: Some(FRAME0.into()),
                patch_hint: Some(vec![8, 8, 8]),
                ..Default::default()
            },
        )?;
        w.add_image("CT", &ct_volume(&mut rng, shape)?, "ct", "CT", quantitative(Some(act.id)))?;
        w.deidentification(fields(json!({"method": "dicom-psi-profile", "date_shift_days": -117})))?;
        Ok(())
    })
}

/// A valid sample carrying one voxel annotation, `organs`.
fn seg_base(path: &Path, encoding: &str, classes: &[(i64, [usize; 3])], index: bool) -> Result<()> {
    let mut rng = Rng::new(SEED);
    write(path, Some("subj-A"), |w| {
        w.identity(fields(json!({"sex": "F", "bodypart": "abdomen"})))?;
        w.add_timepoint("tp0", fields(json!({"label": "baseline"})))?;
        w.label_set(label_set()?);
        let tool = w.software("medh5", Some(VERSION), Map::new())?;
        let imp = w.activity("import", Some(&tool.id), None, Map::new())?;
        let rad = w.person("pseudonym:RAD-07", None, fields(json!({"role": "annotator"})))?;
        let ann = w.activity("annotate", Some(&rad.id), None, fields(json!({"tool": "3D Slicer 5.6.2"})))?;
        w.add_grid(
            "ct",
            &dims(&SHAPE),
            &SPACING,
            GridOptions {
                origin: Some(ORIGIN.to_vec()),
                timepoint: Some("tp0".into()),
                frame_uid: Some(FRAME0.into()),
                patch_hint: Some(vec![8, 8, 8]),
                ..Default::default()
            },
        )?;
        w.add_image("CT", &ct_volume(&mut rng, &SHAPE)?, "ct", "CT", quantitative(Some(imp.id)))?;
        w.add_segmentation(
            "organs",
            "ct",
            SegmentationSource::Masks(blocks(&SHAPE, classes)),
            SegmentationOptions {
                encoding: Some(encoding.into()),
                common: AnnotationOptions {
                    annotated_classes: Annotated::AllGiven,
                    prov: Some(ann.id),
                    quality: Some(QualityArg::Record(fields(json!({"status": "approved", "confidence": 0.9})))),
                    ..Default::default()
                },
                ..Default::default()
            },
        )?;
        if index {
            w.build_index(Some(&["organs".to_string()]), Some(128), None, 0)?;
        }
        w.deidentification(fields(json!({"method": "dicom-psi-profile", "date_shift_days": -117})))?;
        Ok(())
    })
}

/// One CT grid and image on `tp0`, the frame left unset: the shape the
/// annotation-kind cases share.
fn plain_ct(w: &mut SampleWriter, rng: &mut Rng) -> Result<()> {
    w.add_timepoint("tp0", Map::new())?;
    w.label_set(label_set()?);
    w.add_grid("ct", &dims(&SHAPE), &SPACING, on_timepoint("tp0"))?;
    w.add_image("CT", &ct_volume(rng, &SHAPE)?, "ct", "CT", quantitative(None))?;
    Ok(())
}

fn core_two_images(path: &Path) -> Result<()> {
    let mut rng = Rng::new(SEED);
    write(path, None, |w| {
        w.add_timepoint("tp0", fields(json!({"label": "baseline"})))?;
        w.add_grid(
            "ct",
            &dims(&SHAPE),
            &SPACING,
            GridOptions { timepoint: Some("tp0".into()), frame_uid: Some(FRAME0.into()), ..Default::default() },
        )?;
        w.add_grid(
            "pet",
            &[8, 12, 12],
            &[3.0, 1.6, 1.6],
            GridOptions { timepoint: Some("tp0".into()), frame_uid: Some(FRAME0.into()), ..Default::default() },
        )?;
        w.add_image("CT", &ct_volume(&mut rng, &SHAPE)?, "ct", "CT", quantitative(None))?;
        w.add_image(
            "PET",
            &uniform32(&mut rng, &[8, 12, 12])?,
            "pet",
            "PT",
            ImageOptions {
                value_type: Some("quantitative".into()),
                value_units: Some("SUVbw".into()),
                ..Default::default()
            },
        )?;
        w.deidentification(psi())?;
        Ok(())
    })
}

fn core_2d(path: &Path) -> Result<()> {
    let mut rng = Rng::new(SEED);
    write(path, None, |w| {
        w.add_timepoint("tp0", Map::new())?;
        w.add_grid("dx", &[64, 64], &[0.2, 0.2], on_timepoint("tp0"))?;
        w.add_image(
            "DX",
            &integers(&mut rng, 0, 4095, &[64, 64], DType::U16)?,
            "dx",
            "DX",
            ImageOptions { value_type: Some("intensity".into()), ..Default::default() },
        )?;
        w.deidentification(psi())?;
        Ok(())
    })
}

fn core_4d(path: &Path) -> Result<()> {
    let mut rng = Rng::new(SEED);
    let shape = [4, 8, 16, 16];
    write(path, None, |w| {
        w.add_timepoint("tp0", Map::new())?;
        w.add_grid(
            "dce",
            &dims(&shape),
            &[2.0, 1.0, 1.0],
            GridOptions {
                axis_kinds: Some(strings(&["time", "spatial", "spatial", "spatial"])),
                axis_names: Some(strings(&["t", "z", "y", "x"])),
                time_values: Some(vec![0.0, 12.0, 24.0, 36.0]),
                time_units: Some("s".into()),
                timepoint: Some("tp0".into()),
                ..Default::default()
            },
        )?;
        w.add_image(
            "DCE",
            &uniform32(&mut rng, &shape)?,
            "dce",
            "MR",
            ImageOptions { value_type: Some("intensity".into()), ..Default::default() },
        )?;
        w.deidentification(psi())?;
        Ok(())
    })
}

fn core_rgb(path: &Path) -> Result<()> {
    let mut rng = Rng::new(SEED);
    let shape = [3, 32, 32];
    write(path, None, |w| {
        w.add_timepoint("tp0", Map::new())?;
        w.add_grid(
            "slide",
            &dims(&shape),
            &[0.5, 0.5],
            GridOptions {
                units: Some("um".into()),
                axis_kinds: Some(strings(&["channel", "spatial", "spatial"])),
                axis_names: Some(strings(&["c", "y", "x"])),
                timepoint: Some("tp0".into()),
                ..Default::default()
            },
        )?;
        w.add_image(
            "RGB",
            &integers(&mut rng, 0, 255, &shape, DType::U8)?,
            "slide",
            "OT",
            ImageOptions {
                value_type: Some("rgb".into()),
                channel_names: Some(strings(&["R", "G", "B"])),
                ..Default::default()
            },
        )?;
        w.deidentification(psi())?;
        Ok(())
    })
}

fn seg_probmap(path: &Path, threshold: Option<f64>) -> Result<()> {
    let mut rng = Rng::new(SEED);
    write(path, None, |w| {
        plain_ct(w, &mut rng)?;
        let first = uniform(&mut rng, &SHAPE)?;
        let second = uniform(&mut rng, &SHAPE)?;
        w.add_segmentation(
            "soft",
            "ct",
            SegmentationSource::Probabilities(vec![(ClassKey::Id(1), first), (ClassKey::Id(3), second)]),
            SegmentationOptions { threshold, ..Default::default() },
        )?;
        w.deidentification(psi())?;
        Ok(())
    })
}

fn seg_instances(path: &Path) -> Result<()> {
    let mut rng = Rng::new(SEED);
    let objects: Vec<InstanceInput> = [[2, 2, 2], [6, 12, 6], [10, 4, 14]]
        .iter()
        .enumerate()
        .map(|(i, corner)| InstanceInput {
            class_id: 3,
            instance_id: i as u64 + 1,
            mask: Some(block(&SHAPE, corner, &[4, 5, 5])),
            bbox: None,
            crop: None,
            score: Some(0.9),
        })
        .collect();
    write(path, None, |w| {
        plain_ct(w, &mut rng)?;
        w.add_segmentation("lesions", "ct", SegmentationSource::Instances(objects), SegmentationOptions::default())?;
        w.deidentification(psi())?;
        Ok(())
    })
}

fn seg_instances_empty(path: &Path) -> Result<()> {
    let mut rng = Rng::new(SEED);
    write(path, None, |w| {
        plain_ct(w, &mut rng)?;
        w.add_segmentation(
            "lesions",
            "ct",
            SegmentationSource::Instances(Vec::new()),
            SegmentationOptions {
                common: AnnotationOptions { annotated_classes: Annotated::Classes(ids(&[3])), ..Default::default() },
                ..Default::default()
            },
        )?;
        w.deidentification(psi())?;
        Ok(())
    })
}

/// The lower quarter of the volume, declared unannotated.
fn ignore_region() -> ArrayD<bool> {
    block(&SHAPE, &[12, 0, 0], &SHAPE)
}

fn seg_partial_ignore(path: &Path) -> Result<()> {
    let mut rng = Rng::new(SEED);
    write(path, None, |w| {
        plain_ct(w, &mut rng)?;
        w.add_segmentation(
            "organs",
            "ct",
            SegmentationSource::Masks(blocks(&SHAPE, &[(1, [2, 2, 2]), (2, [2, 12, 2])])),
            SegmentationOptions {
                encoding: Some("layers".into()),
                ignore: Some(ignore_region()),
                common: AnnotationOptions { annotated_classes: Annotated::Classes(ids(&[1])), ..Default::default() },
                ..Default::default()
            },
        )?;
        w.deidentification(psi())?;
        Ok(())
    })
}

fn seg_bitmask_ignore_mask(path: &Path) -> Result<()> {
    let mut rng = Rng::new(SEED);
    write(path, None, |w| {
        plain_ct(w, &mut rng)?;
        let tool = w.software("medh5", Some(VERSION), Map::new())?;
        let act = w.activity("annotate", Some(&tool.id), None, Map::new())?;
        // `bitmask` has no in-band ignore value, so the writer stores the
        // region as `organs_ignore` and names it on the header (§7.7).
        let (kind, _) = w.add_segmentation(
            "organs",
            "ct",
            SegmentationSource::Masks(blocks(&SHAPE, &[(1, [2, 2, 2]), (3, [4, 4, 4])])),
            SegmentationOptions {
                encoding: Some("bitmask".into()),
                ignore: Some(ignore_region()),
                common: AnnotationOptions {
                    annotated_classes: Annotated::Classes(ids(&[1])),
                    prov: Some(act.id),
                    ..Default::default()
                },
                ..Default::default()
            },
        )?;
        if kind != "bitmask" {
            return Err(Error::Runtime(format!("seg-bitmask-ignore-mask was written as {kind}")));
        }
        w.deidentification(psi())?;
        Ok(())
    })
}

fn longitudinal(path: &Path) -> Result<()> {
    let mut rng = Rng::new(SEED);
    let (shape, shape_fu) = (SHAPE, [14, 24, 24]);
    write(path, Some("subj-A"), |w| {
        w.add_timepoint("tp0", fields(json!({"label": "baseline", "days_from_baseline": 0})))?;
        w.add_timepoint("tp1", fields(json!({"label": "follow_up_3mo", "days_from_baseline": 92})))?;
        w.label_set(label_set()?);
        for (gid, tp, sh, frame) in [("ct_tp0", "tp0", shape, FRAME0), ("ct_tp1", "tp1", shape_fu, FRAME1)] {
            w.add_grid(
                gid,
                &dims(&sh),
                &SPACING,
                GridOptions { timepoint: Some(tp.into()), frame_uid: Some(frame.into()), ..Default::default() },
            )?;
            w.add_image(&format!("CT_{tp}"), &ct_volume(&mut rng, &sh)?, gid, "CT", quantitative(None))?;
        }
        for (ann, gid, sh) in [("organs_tp0", "ct_tp0", shape), ("organs_tp1", "ct_tp1", shape_fu)] {
            w.add_segmentation(
                ann,
                gid,
                SegmentationSource::Masks(blocks(&sh, &[(1, [2, 2, 2]), (3, [4, 4, 4])])),
                SegmentationOptions::default(),
            )?;
        }
        w.deidentification(psi())?;
        Ok(())
    })?;
    // A minimal affine written by hand: the corpus needs only a well-formed
    // object relating the visits, so W911 does not fire on a valid file.
    mutate(path, |root| {
        let transforms = match ops::child_group(root, "transforms") {
            Some(g) => g,
            None => root.create_group("transforms")?,
        };
        let group = transforms.create_group("tp0_to_tp1")?;
        data::create(&group, "matrix", &NdArray::from(eye4()), &Layout::contiguous())?;
        for (key, value) in [("kind", "affine"), ("from_frame", FRAME0), ("to_frame", FRAME1)] {
            attrs::write(&group, key, &AttrValue::Str(value.into()))?;
        }
        Ok(())
    })?;
    restamp(path)
}

/// Two timepoints, with the failure modes §3.7 warns about as switches.
fn longitudinal_base(path: &Path, shared_frame: bool, drop_timepoint: bool, bad_timepoint: bool) -> Result<()> {
    let mut rng = Rng::new(SEED);
    let shape = [12, 16, 16];
    write(path, Some("subj-A"), |w| {
        w.add_timepoint("tp0", fields(json!({"label": "baseline", "days_from_baseline": 0})))?;
        w.add_timepoint("tp1", fields(json!({"label": "follow_up", "days_from_baseline": 92})))?;
        let frame1 = if shared_frame { FRAME0 } else { FRAME1 };
        for (gid, tp, frame) in [("ct_tp0", "tp0", FRAME0), ("ct_tp1", "tp1", frame1)] {
            w.add_grid(
                gid,
                &dims(&shape),
                &SPACING,
                GridOptions { timepoint: Some(tp.into()), frame_uid: Some(frame.into()), ..Default::default() },
            )?;
        }
        for (gid, tp) in [("ct_tp0", "tp0"), ("ct_tp1", "tp1")] {
            w.add_image(&format!("CT_{tp}"), &ct_volume(&mut rng, &shape)?, gid, "CT", quantitative(None))?;
        }
        w.deidentification(psi())?;
        Ok(())
    })?;
    if drop_timepoint {
        mutate(path, |root| del_attr(root, "grids/ct_tp1", "timepoint"))?;
    }
    if bad_timepoint {
        mutate(path, |root| set_str(root, "grids/ct_tp1", "timepoint", "tp7"))?;
    }
    Ok(())
}

fn pyramid_base(path: &Path, break_origin: bool) -> Result<()> {
    let mut rng = Rng::new(SEED);
    let shape = [16, 32, 32];
    write(path, None, |w| {
        w.add_timepoint("tp0", Map::new())?;
        let level0 = w.add_grid(
            "l0",
            &dims(&shape),
            &[1.0, 1.0, 1.0],
            GridOptions { origin: Some(vec![0.0, 0.0, 0.0]), timepoint: Some("tp0".into()), ..Default::default() },
        )?;
        let level1 = derive_level_grid(&level0, &[2.0, 2.0, 2.0], "l1", None)?;
        w.add_grid(
            "l1",
            &level1.shape,
            &level1.spacing,
            GridOptions {
                origin: Some(level1.origin.clone()),
                direction: Some(level1.direction.clone()),
                timepoint: Some("tp0".into()),
                ..Default::default()
            },
        )?;
        let levels = [uniform32(&mut rng, &shape)?, uniform32(&mut rng, &[8, 16, 16])?];
        w.add_pyramid("CT", &levels, &strings(&["l0", "l1"]), "CT", "mean", ImageOptions::default())?;
        w.deidentification(psi())?;
        Ok(())
    })?;
    if break_origin {
        mutate(path, |root| set_attr(root, "grids/l1", "origin", AttrValue::floats(&[0.0, 0.0, 0.0])))?;
    }
    Ok(())
}

#[derive(Default, Clone, Copy)]
struct Detection {
    world: bool,
    bad_box: bool,
    bad_rotation: bool,
    keypoints: bool,
}

/// A detection sample: boxes, oriented boxes and optionally keypoints.
fn det_base(path: &Path, opts: Detection) -> Result<()> {
    let mut rng = Rng::new(SEED);
    let space = if opts.world { "world" } else { "index" };
    let boxes = ArrayD::from_shape_vec(
        IxDyn(&[2, 3, 2]),
        vec![1.5, 7.5, 1.5, 9.5, 1.5, 9.5, 6.5, 11.5, 10.5, 18.5, 4.5, 12.5],
    )?;
    // The skeleton exists exactly when the keypoints that name it do.
    let skeleton = opts.keypoints.then_some("pair");
    let label_set = match skeleton {
        Some(id) => label_set_with(vec![Skeleton { id: id.into(), keypoints: vec![1, 2], edges: vec![(1, 2)] }])?,
        None => label_set()?,
    };
    let placement = || Placement { grid: Some("ct".into()), space: Some(space.into()), frame_uid: None };
    write(path, None, |w| {
        w.add_timepoint("tp0", Map::new())?;
        w.label_set(label_set);
        w.add_grid(
            "ct",
            &dims(&SHAPE),
            &SPACING,
            GridOptions {
                units: Some("mm".into()),
                timepoint: Some("tp0".into()),
                frame_uid: Some(FRAME0.into()),
                ..Default::default()
            },
        )?;
        w.add_image("CT", &ct_volume(&mut rng, &SHAPE)?, "ct", "CT", quantitative(None))?;
        w.add_boxes(
            "lesions",
            &boxes,
            &ids(&[3, 3]),
            ObjectFields {
                instance_ids: Some(vec![1, 2]),
                scores: Some(vec![0.91, 0.62]),
                attributes: Some(vec![fields(json!({"reader": "r1"})), fields(json!({"reader": "r1"}))]),
            },
            None,
            placement(),
            AnnotationOptions::default(),
        )?;
        let angle = std::f64::consts::PI / 5.0;
        let rotation = ArrayD::from_shape_vec(
            IxDyn(&[1, 3, 3]),
            vec![1.0, 0.0, 0.0, 0.0, angle.cos(), -angle.sin(), 0.0, angle.sin(), angle.cos()],
        )?;
        w.add_obb(
            "lesions_obb",
            &matrix(&[&[6.0, 8.0, 8.0]])?,
            &matrix(&[&[4.0, 6.0, 6.0]])?,
            &rotation,
            &ids(&[3]),
            ObjectFields::default(),
            placement(),
            AnnotationOptions::default(),
        )?;
        if opts.keypoints {
            let points = ArrayD::from_shape_vec(IxDyn(&[1, 2, 3]), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0])?;
            let visibility = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![2i64, 1])?;
            w.add_keypoints(
                "landmarks",
                &points,
                &ids(&[1, 2]),
                &ids(&[1]),
                Some(&visibility),
                ObjectFields::default(),
                skeleton,
                placement(),
                AnnotationOptions::default(),
            )?;
        }
        w.deidentification(psi())?;
        Ok(())
    })?;
    if opts.bad_box {
        mutate(path, |root| flip_first_box(root, "annotations/lesions/boxes"))?;
    }
    if opts.bad_rotation {
        mutate(path, |root| edit_values(root, "annotations/lesions_obb/rotations", |r| r[IxDyn(&[0, 0, 1])] = 0.7))?;
    }
    Ok(())
}

/// A two-timepoint sample with per-visit staging and a change label.
fn cls_base(path: &Path) -> Result<()> {
    let mut rng = Rng::new(SEED);
    let shape = [12, 16, 16];
    write(path, None, |w| {
        w.add_timepoint("tp0", fields(json!({"label": "baseline", "days_from_baseline": 0})))?;
        w.add_timepoint("tp1", fields(json!({"label": "follow_up", "days_from_baseline": 92})))?;
        w.label_set(label_set()?);
        for (tp, frame) in [("tp0", FRAME0), ("tp1", FRAME1)] {
            let gid = format!("ct_{tp}");
            w.add_grid(
                &gid,
                &dims(&shape),
                &SPACING,
                GridOptions { timepoint: Some(tp.into()), frame_uid: Some(frame.into()), ..Default::default() },
            )?;
            w.add_image(&format!("CT_{tp}"), &ct_volume(&mut rng, &shape)?, &gid, "CT", quantitative(None))?;
        }
        w.add_classification(
            "staging",
            Assertions {
                class_ids: vec![3, 4],
                values: vec![1.0, 1.0],
                scope_ids: Some(vec![0, 1]),
                schemes: Some(strings(&["Lung-RADS", "Lung-RADS"])),
                scheme_values: Some(strings(&["4A", "4B"])),
            },
            "timepoint",
            true,
            None,
            AnnotationOptions::default(),
        )?;
        w.add_classification(
            "response",
            Assertions { class_ids: vec![1], values: vec![1.0], scope_ids: None, schemes: None, scheme_values: None },
            "sample",
            false,
            None,
            AnnotationOptions { timepoints: Some(strings(&["tp0", "tp1"])), ..Default::default() },
        )?;
        w.deidentification(psi())?;
        Ok(())
    })
}

/// Contours and a surface mesh, the two non-voxel shape representations.
fn shape_base(path: &Path) -> Result<()> {
    let mut rng = Rng::new(SEED);
    let shape = [12, 16, 16];
    let square = matrix(&[&[4.0, 4.0, 4.0], &[4.0, 4.0, 9.0], &[4.0, 9.0, 9.0], &[4.0, 9.0, 4.0]])?;
    let hole = matrix(&[&[4.0, 6.0, 6.0], &[4.0, 6.0, 7.0], &[4.0, 7.0, 7.0]])?;
    let vertices = matrix(&[&[0.0, 0.0, 0.0], &[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]])?;
    let faces = ArrayD::from_shape_vec(IxDyn(&[4, 3]), vec![0i64, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3])?;
    let on_ct = || Placement { grid: Some("ct".into()), space: None, frame_uid: None };
    write(path, None, |w| {
        w.add_timepoint("tp0", Map::new())?;
        w.label_set(label_set()?);
        w.add_grid(
            "ct",
            &dims(&shape),
            &[1.0, 1.0, 1.0],
            GridOptions { timepoint: Some("tp0".into()), frame_uid: Some(FRAME0.into()), ..Default::default() },
        )?;
        w.add_image("CT", &ct_volume(&mut rng, &shape)?, "ct", "CT", quantitative(None))?;
        w.add_contours(
            "rtstruct",
            &[Polygon::new(square, 1, (0, 4), "outer")?, Polygon::new(hole, 1, (0, 4), "hole")?],
            on_ct(),
            AnnotationOptions::default(),
        )?;
        w.add_mesh(
            "liver_surface",
            &vertices,
            &faces,
            None,
            None,
            None,
            Some(&ids(&[1])),
            Placement { grid: Some("ct".into()), space: Some("world".into()), frame_uid: Some(FRAME0.into()) },
            AnnotationOptions::default(),
        )?;
        w.add_points(
            "landmarks",
            &matrix(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]])?,
            None,
            Some(&strings(&["apex", "carina"])),
            Some(&[1.0, 0.5]),
            None,
            on_ct(),
            AnnotationOptions::default(),
        )?;
        w.deidentification(psi())?;
        Ok(())
    })
}

#[derive(Default, Clone, Copy)]
struct Registration {
    displacement: bool,
    bspline: bool,
    composite: bool,
    inverse: bool,
}

/// Two timepoints related by a transform, with landmark ground truth (§10).
fn reg_base(path: &Path, opts: Registration) -> Result<()> {
    let mut rng = Rng::new(SEED);
    let shape = [12, 16, 16];
    let shift = [2.0, -1.0, 0.5];
    let affine = |sign: f64| {
        let mut m = eye4();
        for (i, v) in shift.iter().enumerate() {
            m[IxDyn(&[i, 3])] = sign * v;
        }
        m
    };
    let fixed = [[2.0f32, 3.0, 4.0], [6.0, 7.0, 8.0]];
    let moved: Vec<f64> = fixed.iter().flat_map(|p| (0..3).map(move |i| f64::from(p[i] + shift[i] as f32))).collect();
    let fixed: Vec<f64> = fixed.iter().flat_map(|p| p.iter().map(|v| f64::from(*v))).collect();
    write(path, None, |w| {
        w.add_timepoint("tp0", fields(json!({"label": "baseline", "days_from_baseline": 0})))?;
        w.add_timepoint("tp1", fields(json!({"label": "follow_up", "days_from_baseline": 92})))?;
        w.label_set(label_set()?);
        for (tp, frame) in [("tp0", FRAME0), ("tp1", FRAME1)] {
            let gid = format!("ct_{tp}");
            w.add_grid(
                &gid,
                &dims(&shape),
                &SPACING,
                GridOptions {
                    origin: Some(vec![0.0, 0.0, 0.0]),
                    timepoint: Some(tp.into()),
                    frame_uid: Some(frame.into()),
                    ..Default::default()
                },
            )?;
            w.add_image(&format!("CT_{tp}"), &ct_volume(&mut rng, &shape)?, &gid, "CT", quantitative(None))?;
        }
        w.add_transform(
            "tp0_to_tp1",
            "affine",
            FRAME0,
            FRAME1,
            TransformSpec {
                matrix: Some(affine(1.0)),
                from_grid: Some("ct_tp0".into()),
                to_grid: Some("ct_tp1".into()),
                invertible: Some(true),
                inverse_id: opts.inverse.then(|| "tp1_to_tp0".to_string()),
                metrics: Some(QualityArg::Record(fields(json!({"status": "approved", "confidence": 0.88})))),
                ..Default::default()
            },
        )?;
        if opts.inverse {
            w.add_transform(
                "tp1_to_tp0",
                "affine",
                FRAME1,
                FRAME0,
                TransformSpec {
                    matrix: Some(affine(-1.0)),
                    invertible: Some(true),
                    inverse_id: Some("tp0_to_tp1".into()),
                    ..Default::default()
                },
            )?;
        }
        if opts.displacement {
            let mut field = ArrayD::<f32>::zeros(IxDyn(&[3, shape[0], shape[1], shape[2]]));
            field.index_axis_mut(ndarray::Axis(0), 0).fill(0.75);
            w.add_transform(
                "refine",
                "displacement",
                FRAME1,
                "pseudo:frame-102",
                TransformSpec {
                    field: Some(NdArray::from(field)),
                    field_grid: Some("ct_tp1".into()),
                    vector_space: Some("world".into()),
                    ..Default::default()
                },
            )?;
        }
        if opts.bspline {
            let mut control = ArrayD::<f64>::zeros(IxDyn(&[3, 6, 6, 6]));
            control.index_axis_mut(ndarray::Axis(0), 1).fill(0.5);
            w.add_grid(
                "cp",
                &[6, 6, 6],
                &[3.0, 3.2, 3.2],
                GridOptions {
                    origin: Some(vec![0.0, 0.0, 0.0]),
                    timepoint: Some("tp1".into()),
                    frame_uid: Some(FRAME1.into()),
                    ..Default::default()
                },
            )?;
            w.add_transform(
                "ffd",
                "bspline",
                FRAME1,
                "pseudo:frame-103",
                TransformSpec {
                    control_points: Some(control),
                    cp_grid: Some("cp".into()),
                    order: Some(3),
                    ..Default::default()
                },
            )?;
        }
        if opts.composite {
            w.add_transform(
                "tp0_to_refined",
                "composite",
                FRAME0,
                "pseudo:frame-102",
                TransformSpec { components: Some(strings(&["tp0_to_tp1", "refine"])), ..Default::default() },
            )?;
        }
        let names = strings(&["apex", "carina"]);
        for (ann, gid, points, weights, partner) in [
            ("landmarks_tp0", "ct_tp0", &fixed, Some([1.0, 1.0]), "landmarks_tp1"),
            ("landmarks_tp1", "ct_tp1", &moved, None, "landmarks_tp0"),
        ] {
            w.add_points(
                ann,
                &ArrayD::from_shape_vec(IxDyn(&[2, 3]), points.clone())?,
                None,
                Some(&names),
                weights.as_ref().map(|w| w.as_slice()),
                Some(partner),
                Placement { grid: Some(gid.into()), space: Some("world".into()), frame_uid: None },
                AnnotationOptions { task: Some("registration".into()), ..Default::default() },
            )?;
        }
        w.deidentification(psi())?;
        Ok(())
    })
}

/// Two visits of one subject, with the same lesion in both (§7.4).
///
/// `reclassify` makes instance 7 a different class at follow-up, which is the
/// cross-annotation tracking error W909 exists to catch; `partial_coverage`
/// withdraws the follow-up commitment so absence becomes *unexamined*.
fn tracking_sample(path: &Path, reclassify: bool, partial_coverage: bool) -> Result<()> {
    let mut rng = Rng::new(SEED);
    let shape = [12, 16, 16];
    let lesion = |z: usize, y: usize, x: usize, r: usize| block(&shape, &[z - r, y - r, x - r], &[2 * r, 2 * r, 2 * r]);
    let instance = |class_id: i64, instance_id: u64, mask: ArrayD<bool>| InstanceInput {
        class_id,
        instance_id,
        mask: Some(mask),
        bbox: None,
        crop: None,
        score: None,
    };
    write(path, Some("subj-A"), |w| {
        w.add_timepoint("tp0", fields(json!({"label": "baseline", "days_from_baseline": 0})))?;
        w.add_timepoint("tp1", fields(json!({"label": "follow_up", "days_from_baseline": 92})))?;
        w.label_set(label_set()?);
        let rad = w.person("pseudonym:RAD-07", None, fields(json!({"role": "annotator"})))?;
        let act = w.activity("annotate", Some(&rad.id), None, fields(json!({"ended": "2026-02-05T14:47:00Z"})))?;
        for (gid, tp, frame) in [("ct_tp0", "tp0", FRAME0), ("ct_tp1", "tp1", FRAME1)] {
            w.add_grid(
                gid,
                &dims(&shape),
                &SPACING,
                GridOptions { timepoint: Some(tp.into()), frame_uid: Some(frame.into()), ..Default::default() },
            )?;
            w.add_image(&format!("CT_{tp}"), &ct_volume(&mut rng, &shape)?, gid, "CT", quantitative(None))?;
        }
        w.add_segmentation(
            "lesions_tp0",
            "ct_tp0",
            SegmentationSource::Instances(vec![
                instance(3, 7, lesion(6, 6, 6, 2)),
                instance(3, 8, lesion(6, 11, 11, 1)),
            ]),
            SegmentationOptions {
                common: AnnotationOptions {
                    annotated_classes: Annotated::Classes(ids(&[3])),
                    prov: Some(act.id.clone()),
                    quality: Some(QualityArg::Record(fields(json!({"status": "approved", "confidence": 0.9})))),
                    ..Default::default()
                },
                ..Default::default()
            },
        )?;
        // A case isolates one defect: the reclassified follow-up commits to
        // both classes so W904 does not fire alongside the W909 it is for.
        let committed: &[i64] = if partial_coverage {
            &[]
        } else if reclassify {
            &[1, 3]
        } else {
            &[3]
        };
        w.add_segmentation(
            "lesions_tp1",
            "ct_tp1",
            SegmentationSource::Instances(vec![
                instance(if reclassify { 1 } else { 3 }, 7, lesion(6, 6, 6, 3)),
                instance(3, 9, lesion(7, 3, 12, 1)),
            ]),
            SegmentationOptions {
                common: AnnotationOptions {
                    annotated_classes: Annotated::Classes(ids(committed)),
                    prov: Some(act.id.clone()),
                    quality: Some(QualityArg::Record(fields(json!({"status": "approved"})))),
                    ..Default::default()
                },
                ..Default::default()
            },
        )?;
        w.add_transform(
            "tp0_to_tp1",
            "affine",
            FRAME0,
            FRAME1,
            TransformSpec { matrix: Some(eye4()), ..Default::default() },
        )?;
        w.deidentification(psi())?;
        w.split(fields(json!({"set_id": "cv5-2026-02", "partition": "train", "fold": 1})))?;
        Ok(())
    })
}

/// A shard built the way a curator builds one: pack standalone samples.
fn collection(path: &Path, samples: usize, drop_content_id: bool, drop_samples_group: bool) -> Result<()> {
    // The members are scaffolding, not corpus files: a shipped corpus
    // directory must contain exactly the cases its manifest lists.
    let directory = path.with_file_name(format!(".{}-members", stem(path)));
    if directory.exists() {
        fs::remove_dir_all(&directory)?;
    }
    fs::create_dir_all(&directory)?;
    let packed = (|| -> Result<()> {
        let mut sources: Vec<PathBuf> = Vec::new();
        for i in 0..samples {
            let member = directory.join(format!("case_{i}.medh5"));
            base(&member, &SHAPE)?;
            sources.push(member);
        }
        let keys: Vec<String> = ["case.0", "case_1"].iter().take(samples).map(|k| k.to_string()).collect();
        let refs: Vec<&Path> = sources.iter().map(PathBuf::as_path).collect();
        pack(&refs, path, Some(&keys))?;
        Ok(())
    })();
    let _ = fs::remove_dir_all(&directory);
    packed?;
    if drop_content_id {
        mutate(path, |root| del_attr(root, &format!("{SAMPLES_GROUP}/case.0"), "content_id"))?;
    }
    if drop_samples_group {
        mutate(path, |root| Ok(root.unlink(SAMPLES_GROUP)?))?;
    }
    Ok(())
}

// -- mutations --------------------------------------------------------------------------

fn flip_first_box(root: &hdf5::Group, path: &str) -> Result<()> {
    edit_values(root, path, |b| {
        let (lo, hi) = (IxDyn(&[0, 0, 0]), IxDyn(&[0, 0, 1]));
        let (a, z) = (b[&lo], b[&hi]);
        b[lo] = z;
        b[hi] = a;
    })
}

fn write_garbage_meta(root: &hdf5::Group) -> Result<()> {
    root.unlink("meta")?;
    data::create_scalar_string(root, "meta", "{not json")?;
    Ok(())
}

fn store_float_ct(root: &hdf5::Group) -> Result<()> {
    rewrite_dataset(root, "images/CT", Some(DType::F32))
}

fn duplicate_layer_class(root: &hdf5::Group) -> Result<()> {
    let ds = root.dataset("annotations/organs/layer_class_ids")?;
    if ds.shape().first().copied().unwrap_or(0) < 2 {
        return Err(Error::Runtime("case needs at least two layers".into()));
    }
    drop(ds);
    edit_values(root, "annotations/organs/layer_class_ids", |t| t[IxDyn(&[1, 0])] = t[IxDyn(&[0, 0])])
}

fn corrupt_data(root: &hdf5::Group) -> Result<()> {
    edit_values(root, "annotations/organs/data", |d| {
        let first = vec![0; d.ndim()];
        d[IxDyn(&first)] = 7.0;
    })
}

fn strip_digests(root: &hdf5::Group) -> Result<()> {
    let mut datasets = Vec::new();
    ops::visit(root, &mut |name, node| {
        if let ops::Node::Dataset(_) = node {
            datasets.push(name.to_string());
        }
        Ok(true)
    })?;
    for name in datasets {
        let ds = root.dataset(&name)?;
        if attrs::has(&ds, "digest") {
            attrs::delete(&ds, "digest")?;
        }
    }
    if attrs::has(root, "content_id") {
        attrs::delete(root, "content_id")?;
    }
    Ok(())
}

/// Object 1 is class 1 in one annotation and class 2 in another.
///
/// Across two annotations, because that is the case W909 exists for: each
/// annotation internally consistent, the join between them wrong (§7.4,
/// Appendix C).  Two rows sharing an id inside *one* annotation are two
/// objects sharing an id, which §7.4 forbids outright --- E404, not a warning.
fn instances_two_classes(root: &hdf5::Group) -> Result<()> {
    root.unlink("annotations/organs")?;
    let annotations = root.group("annotations")?;
    for (name, (lo, hi), class_id) in [("organs", (1.5f32, 5.5f32), 1u16), ("organs_rater2", (6.5, 9.5), 2)] {
        let group = annotations.create_group(name)?;
        let contiguous = Layout::contiguous();
        data::create(&group, "boxes", &NdArray::from_vec(&[1, 3, 2], vec![lo, hi, lo, hi, lo, hi])?, &contiguous)?;
        data::create(&group, "class_ids", &NdArray::from_vec(&[1], vec![class_id])?, &contiguous)?;
        data::create(&group, "instance_ids", &NdArray::from_vec(&[1], vec![1u32])?, &contiguous)?;
        for (key, value) in [
            ("kind", "instances"),
            ("task", "segmentation"),
            ("grid", "ct"),
            ("closure", "explicit"),
            ("quality", "organs"),
        ] {
            attrs::write(&group, key, &AttrValue::Str(value.into()))?;
        }
        attrs::write(&group, "class_ids", &uint16s(&[1, 2])?)?;
        attrs::write(&group, "annotated_class_ids", &uint16s(&[1, 2])?)?;
    }
    Ok(())
}

/// Rewrite a 2-layer annotation as 5 layers --- W908's reason to exist.
fn over_layered(root: &hdf5::Group) -> Result<()> {
    let group = root.group("annotations/organs")?;
    let stored = data::read(&group.dataset("data")?)?;
    let values = stored.to_f64();
    let table = data::read(&group.dataset("layer_class_ids")?)?.to_f64();
    let mut classes: Vec<i64> = table.iter().map(|v| *v as i64).filter(|v| *v != 0).collect();
    classes.sort_unstable();
    classes.dedup();
    let layers = values.shape()[0];
    let spatial: Vec<usize> = values.shape()[1..].to_vec();
    let voxels: usize = spatial.iter().product();
    let flat = values.as_standard_layout().iter().copied().collect::<Vec<f64>>();
    let mut wide = vec![0.0f64; 5 * voxels];
    for (position, class_id) in classes.iter().enumerate() {
        for v in 0..voxels {
            if (0..layers).any(|l| flat[l * voxels + v] as i64 == *class_id) {
                wide[position * voxels + v] = *class_id as f64;
            }
        }
    }
    let mut shape = vec![5];
    shape.extend(&spatial);
    let wide = NdArray::from(ArrayD::from_shape_vec(IxDyn(&shape), wide)?).astype(stored.dtype());
    let mut new_table = vec![0u16; 5];
    for (position, class_id) in classes.iter().enumerate() {
        new_table[position] = *class_id as u16;
    }
    group.unlink("data")?;
    group.unlink("layer_class_ids")?;
    data::create(&group, "data", &wide, &Layout::contiguous())?;
    data::create(&group, "layer_class_ids", &NdArray::from_vec(&[5, 1], new_table)?, &Layout::contiguous())?;
    Ok(())
}

fn break_offsets(root: &hdf5::Group) -> Result<()> {
    edit_values(root, "annotations/lesions/mask_offsets", |o| o.swap(IxDyn(&[1]), IxDyn(&[2])))
}

fn crowd_one_scope_unit(root: &hdf5::Group) -> Result<()> {
    edit_values(root, "annotations/staging/scope_ids", |s| s.fill(0.0))?;
    set_attr(root, "annotations/staging", "multilabel", AttrValue::Bool(false))
}

fn break_affine_last_row(root: &hdf5::Group) -> Result<()> {
    edit_values(root, "transforms/tp0_to_tp1/matrix", |m| m[IxDyn(&[3, 0])] = 0.5)
}

// -- registry ---------------------------------------------------------------------------

/// Every case, in corpus order.
pub(super) fn registry() -> Vec<Case> {
    let mut cases = Vec::with_capacity(117);
    valid_cases(&mut cases);
    first_invalid_batch(&mut cases);
    second_batch(&mut cases);
    third_batch(&mut cases);
    geometric_cases(&mut cases);
    transform_cases(&mut cases);
    tracking_and_collection_cases(&mut cases);
    fourth_batch(&mut cases);
    fifth_batch(&mut cases);
    cases
}

fn valid_cases(cases: &mut Vec<Case>) {
    let w912 = &["W912"];
    cases.extend([
        case("core-minimal", "One grid, one image, one timepoint.", "§2, §3, §4", |p| base(p, &SHAPE)),
        case(
            "core-two-images-one-grid",
            "Two co-registered images sharing a grid and a frame of reference.",
            "§3.4",
            core_two_images,
        ),
        case("core-2d-radiograph", "A 2-D image with two spatial axes.", "§3.6", core_2d),
        case("core-4d-time", "A 4-D dynamic series with one time axis.", "§3.6", core_4d),
        case("core-rgb-channels", "A channel axis with named channels.", "§3.6, §4.1", core_rgb),
        case("seg-labelmap", "Mutually exclusive classes stored as one integer volume.", "§7.1", |p| {
            seg_base(p, "labelmap", &EXCLUSIVE, false)
        })
        .warnings(w912),
        case("seg-layers", "Overlapping classes coloured into layers --- the default encoding.", "§7.2", |p| {
            seg_base(p, "layers", &ORGANS, false)
        })
        .warnings(w912),
        case("seg-bitmask", "One bit per class per voxel.", "§7.3", |p| seg_base(p, "bitmask", &ORGANS, false))
            .warnings(w912),
        case("seg-probmap", "Soft ground truth as per-class probabilities.", "§7.5", |p| seg_probmap(p, None))
            .warnings(w912),
        case(
            "seg-instances",
            "Per-object boxes and bit-packed crops with sample-scoped instance ids.",
            "§7.4",
            seg_instances,
        )
        .warnings(w912),
        // The verified negative tracking depends on (S-13): a follow-up at
        // which a lesion has resolved is told apart from one nobody examined
        // only by this.
        case(
            "seg-instances-empty",
            "Examined for lesions and found none: an `instances` annotation with N = 0.",
            "§7.4, §11.3",
            seg_instances_empty,
        )
        .warnings(w912),
        case(
            "seg-partial-coverage-with-ignore",
            "Two of four classes annotated, with an explicit ignore region: no W904.",
            "§7.7, §11.3",
            seg_partial_ignore,
        )
        .warnings(w912),
        case("training-index", "A sampling index that is current for its annotation.", "§14.3", |p| {
            seg_base(p, "auto", &ORGANS, true)
        })
        .level("integrity")
        .warnings(w912),
        case(
            "longitudinal-two-timepoints",
            "Two timepoints, distinct frames, a transform relating them.",
            "§3.7, §7.4",
            longitudinal,
        )
        .warnings(w912),
    ]);
}

fn first_invalid_batch(cases: &mut Vec<Case>) {
    cases.extend([
        invalid_core("E001-missing-version", "Root without `medh5_version`.", "§2.1", &["E001"], |f| {
            del_attr(f, "", "medh5_version")
        }),
        invalid_core("E002-unsupported-major", "A major version this reader must refuse.", "§2.1", &["E002"], |f| {
            set_str(f, "", "medh5_version", "2.0")
        }),
        invalid_core("E003-bad-identifier", "An object name with a forbidden character.", "§2.3", &["E003"], |f| {
            Ok(f.group("images")?.relink("CT", "CT scan")?)
        }),
        invalid_core(
            "E004-meta-not-json",
            "`meta` holding text that is not JSON.",
            "§2.4",
            &["E004"],
            write_garbage_meta,
        ),
        invalid_core("E005-meta-schema", "A document missing a required member.", "§2.4", &["E005"], |f| {
            set_meta(f, |d| {
                d.remove("identity");
            })
        }),
        invalid_core("E006-missing-kind", "Root without `medh5_kind`.", "§2.1", &["E006"], |f| {
            del_attr(f, "", "medh5_kind")
        }),
        invalid_core("E007-unknown-profile", "A declared profile outside the registry.", "§1.3", &["E007"], |f| {
            set_attr(f, "", "medh5_profiles", AttrValue::strs(&["core", "quantum"]))
        }),
        invalid_core("E008-missing-grids", "The `grids` group removed.", "§2.3", &["E008", "E101"], |f| {
            Ok(f.unlink("grids")?)
        }),
        invalid_core("E101-dangling-grid", "An image naming a grid that does not exist.", "§3.2", &["E101"], |f| {
            set_str(f, "images/CT", "grid", "nope")
        }),
        invalid_core("E102-non-orthonormal", "A `direction` that is not orthonormal.", "§3.2", &["E102"], |f| {
            set_attr(
                f,
                "grids/ct",
                "direction",
                AttrValue::matrix(3, 3, &[1.0, 0.4, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            )
        }),
        invalid_core(
            "E103-spatial-not-trailing",
            "Spatial axes that are not the trailing axes.",
            "§3.1",
            &["E103"],
            |f| set_attr(f, "grids/ct", "axis_kinds", AttrValue::strs(&["spatial", "spatial", "channel"])),
        ),
        invalid_core("E104-nonpositive-spacing", "A zero voxel spacing.", "§3.2", &["E104"], |f| {
            set_attr(f, "grids/ct", "spacing", AttrValue::floats(&[1.5, 0.0, 0.8]))
        }),
        invalid_core(
            "E110-bad-axis-kinds",
            "One spatial axis and two time axes: both counts are out of range.",
            "§3.1",
            &["E110"],
            |f| set_attr(f, "grids/ct", "axis_kinds", AttrValue::strs(&["time", "time", "spatial"])),
        ),
        invalid_core("E201-no-images", "A sample with an empty `images` group.", "§4.1", &["E201"], |f| {
            Ok(f.unlink("images/CT")?)
        }),
        invalid_core("E202-shape-mismatch", "An image whose shape differs from its grid.", "§4.1", &["E202"], |f| {
            set_attr(f, "grids/ct", "shape", AttrValue::ints(&[16, 24, 25]))
        }),
        invalid_core("E203-unknown-value-type", "An unregistered `value_type`.", "§4.2", &["E203"], |f| {
            set_str(f, "images/CT", "value_type", "vibes")
        }),
        invalid_core(
            "E204-channel-names",
            "`channel_names` on a grid with no channel axis.",
            "§4.1",
            &["E204"],
            |f| set_attr(f, "images/CT", "channel_names", AttrValue::strs(&["R", "G", "B"])),
        ),
        invalid_core(
            "E603-unknown-activity",
            "A provenance activity of an unknown type.",
            "§11.1",
            &["E005", "E603"],
            |f| {
                set_meta(f, |d| {
                    first_activity(d).insert("type".into(), json!("vibecheck"));
                })
            },
        ),
        invalid_core("E604-bad-timestamp", "An activity timestamp that is not RFC 3339.", "§11.1", &["E604"], |f| {
            set_meta(f, |d| {
                first_activity(d).insert("ended".into(), json!("yesterday"));
            })
        }),
        invalid_core("E605-dangling-agent", "An activity naming an undeclared agent.", "§11.1", &["E605"], |f| {
            set_meta(f, |d| {
                first_activity(d).insert("agent".into(), json!("ghost"));
            })
        }),
        invalid_core("W903-no-deidentification", "No de-identification record.", "§11.4", &[], |f| {
            set_meta(f, |d| {
                d.remove("deidentification");
            })
        })
        .warnings(&["W903"]),
        invalid_core("W906-conflicting-splits", "One split set claimed against two manifests.", "§12.3", &[], |f| {
            set_meta(f, |d| {
                d.insert(
                    "splits".into(),
                    json!([
                        {"set_id": "cv5", "partition": "train", "manifest_sha256": "a".repeat(64)},
                        {"set_id": "cv5", "partition": "test", "manifest_sha256": "b".repeat(64)},
                    ]),
                );
            })
        })
        .warnings(&["W906"]),
        invalid_core("W907-float-storage", "Integer-valued data stored as float32.", "§4.2", &[], store_float_ct)
            .warnings(&["W907"]),
        invalid_seg(
            "E303-reserved-class-id",
            "An annotation claiming the ignore id.",
            "§5.3",
            &["E303", "E402", "E403", "E404"],
            |f| set_attr(f, "annotations/organs", "class_ids", uint16s(&[1, 2, 65535])?),
        ),
        invalid_seg("E401-unknown-kind", "An annotation of an unregistered kind.", "§6.3", &["E009", "E401"], |f| {
            set_str(f, "annotations/organs", "kind", "voxelthing")
        }),
        invalid_seg("E401-reserved-kind", "The reserved `rle` kind in a 1.0 file.", "§16", &["E009", "E401"], |f| {
            set_str(f, "annotations/organs", "kind", "rle")
        }),
        invalid_seg(
            "E402-class-not-in-labelset",
            "A class id absent from the label set.",
            "§5.1",
            &["E402", "E403", "E404"],
            |f| set_attr(f, "annotations/organs", "class_ids", uint16s(&[1, 2, 99])?),
        ),
        invalid_seg(
            "E403-coverage-superset",
            "`annotated_class_ids` claiming more than `class_ids`.",
            "§6.2",
            &["E403"],
            |f| set_attr(f, "annotations/organs", "annotated_class_ids", uint16s(&[1, 2, 3, 4])?),
        ),
        invalid_seg(
            "E404-class-in-two-layers",
            "One class assigned to two layers.",
            "§7.2",
            &["E404"],
            duplicate_layer_class,
        ),
        invalid_seg(
            "E405-data-shape",
            "Annotation data whose spatial shape differs from the grid.",
            "§7.2",
            &["E202", "E405"],
            |f| set_attr(f, "grids/ct", "shape", AttrValue::ints(&[16, 24, 25])),
        ),
        invalid_seg(
            "E409-undeclared-timepoint",
            "An annotation naming a timepoint that is not declared.",
            "§3.7",
            &["E409"],
            |f| set_attr(f, "annotations/organs", "timepoints", AttrValue::strs(&["tp9"])),
        ),
        invalid_seg(
            "E410-missing-dataset",
            "A `layers` annotation without `layer_class_ids`.",
            "§7.2",
            &["E410"],
            |f| Ok(f.unlink("annotations/organs/layer_class_ids")?),
        ),
        invalid_seg("E411-bad-dtype", "A `layers` volume stored as int32.", "§7.2", &["E411"], |f| {
            rewrite_dataset(f, "annotations/organs/data", Some(DType::I32))
        }),
        invalid_seg("E601-dangling-prov", "An annotation naming an unknown activity.", "§11.1", &["E601"], |f| {
            set_str(f, "annotations/organs", "prov", "act_nope")
        }),
        invalid_seg(
            "E602-unknown-quality",
            "An annotation naming an unknown quality record.",
            "§11.2",
            &["E602"],
            |f| set_str(f, "annotations/organs", "quality", "nope"),
        ),
        invalid_seg(
            "W904-partial-no-ignore",
            "Partial coverage with no ignore region --- `0` is not a negative.",
            "§11.3",
            &[],
            |f| set_attr(f, "annotations/organs", "annotated_class_ids", uint16s(&[1])?),
        )
        .warnings(&["W904", "W912"]),
        invalid_seg(
            "E701-digest-mismatch",
            "A dataset edited after its digest was stamped. `content_id` still matches: it covers the digest *list*, \
             so the mismatch stays local to the object that changed.",
            "§13.1",
            &["E701"],
            corrupt_data,
        )
        .level("integrity"),
        invalid_seg("E702-content-id-mismatch", "A `content_id` that does not match.", "§13.2", &["E702"], |f| {
            set_str(f, "", "content_id", &format!("sha256:{}", "0".repeat(64)))
        })
        .level("integrity"),
        invalid_seg(
            "E703-malformed-digest",
            "A digest string that does not parse.",
            "§13.1",
            &["E702", "E703"],
            |f| set_str(f, "annotations/organs/data", "digest", "not-a-digest"),
        )
        .level("integrity"),
        invalid_seg("W901-no-digests", "A file carrying no digests at all.", "§13.1", &[], strip_digests)
            .level("integrity")
            .warnings(&["W901", "W912"]),
        invalid_seg(
            "W909-instance-id-two-classes",
            "One instance id carrying two class ids.",
            "§7.4",
            &[],
            instances_two_classes,
        )
        .warnings(&["W909", "W912"]),
    ]);
}

fn second_batch(cases: &mut Vec<Case>) {
    cases.extend([
        case("multiscale-pyramid", "A two-level pyramid with consistent geometry.", "§4.3", |p| {
            pyramid_base(p, false)
        }),
        case("E105-pyramid-origin", "A pyramid level missing the half-voxel origin shift.", "§4.3", |p| {
            pyramid_base(p, true)
        })
        .errors(&["E105"]),
        case("W911-no-relating-transform", "Two timepoints and nothing relating them.", "§3.7", |p| {
            longitudinal_base(p, false, false, false)
        })
        .warnings(&["W911"]),
        case(
            "W910-shared-frame-across-timepoints",
            "Two timepoints sharing one frame of reference, which asserts an alignment nobody computed.",
            "§3.4",
            |p| longitudinal_base(p, true, false, false),
        )
        .warnings(&["W910", "W911"]),
        case("E106-grid-without-timepoint", "A grid with no `timepoint` in a multi-timepoint sample.", "§3.7", |p| {
            longitudinal_base(p, false, true, false)
        })
        .errors(&["E106"])
        .warnings(&["W911"]),
        case(
            "E107-undeclared-grid-timepoint",
            "A grid naming a timepoint the document does not declare.",
            "§3.7",
            |p| longitudinal_base(p, false, false, true),
        )
        .errors(&["E107"])
        .warnings(&["W911"]),
        invalid_core(
            "E108-nondense-timepoint-index",
            "Timepoint indices that are not dense from zero.",
            "§3.7",
            &["E108"],
            |f| {
                set_meta(f, |d| {
                    d.insert("timepoints".into(), json!([{"id": "tp0", "index": 0}, {"id": "tp2", "index": 2}]));
                })
            },
        ),
        invalid_core("E111-empty-grids", "A `grids` group with no grid in it.", "§3.2", &["E101", "E111"], |f| {
            Ok(f.unlink("grids/ct")?)
        }),
        invalid_seg("E412-missing-coverage", "An annotation without `annotated_class_ids`.", "§6.2", &["E412"], |f| {
            del_attr(f, "annotations/organs", "annotated_class_ids")
        })
        .warnings(&["W904", "W912"]),
        invalid_seg(
            "E301-seg-without-labelset",
            "The `seg` profile declared with no label set.",
            "§5.1",
            &["E301"],
            |f| {
                set_meta(f, |d| {
                    d.remove("label_set");
                })
            },
        )
        .warnings(&[]),
        // A label set that cannot be constructed stops the document from
        // parsing, so the rules that depend on a parsed document correctly do
        // not run.
        invalid_seg("E302-duplicate-class-id", "Two label-set entries with one id.", "§5.2", &["E302"], |f| {
            set_meta(f, |d| {
                let classes = label_classes(d);
                if let Some(Value::Object(first)) = classes.first().cloned() {
                    let mut copy = first;
                    copy.insert("key".into(), json!("liver_copy"));
                    classes.push(Value::Object(copy));
                }
            })
        })
        .warnings(&[]),
        invalid_seg(
            "W908-too-many-layers",
            "Five layers where a greedy colouring needs two.",
            "§7.6",
            &[],
            over_layered,
        )
        .warnings(&["W908", "W912"]),
        invalid(
            "E406-box-lo-gt-hi",
            "An instance box with lo greater than hi.",
            "§8.1",
            &["E406"],
            seg_instances,
            |f| flip_first_box(f, "annotations/lesions/boxes"),
        )
        .warnings(&["W912"]),
        invalid(
            "E408-nonmonotonic-offsets",
            "Mask offsets that decrease.",
            "§7.4",
            &["E408"],
            seg_instances,
            break_offsets,
        )
        .warnings(&["W912"]),
        invalid(
            "W905-stale-index",
            "An index whose `source_digest` is out of date.",
            "§13.3",
            &[],
            |p| seg_base(p, "auto", &ORGANS, true),
            |f| set_str(f, "index/organs", "source_digest", &format!("sha256:{}", "0".repeat(64))),
        )
        .level("integrity")
        .warnings(&["W905", "W912"]),
    ]);
}

fn third_batch(cases: &mut Vec<Case>) {
    let seg = |p: &Path| seg_base(p, "auto", &ORGANS, false);
    cases.extend([
        invalid_core("E109-missing-grid-attribute", "A grid without `spacing`.", "§3.2", &["E109"], |f| {
            del_attr(f, "grids/ct", "spacing")
        }),
        invalid_core("E205-missing-image-attribute", "An image without `modality`.", "§4.1", &["E205"], |f| {
            del_attr(f, "images/CT", "modality")
        }),
        invalid("E304-hierarchy-cycle", "A class hierarchy with a cycle.", "§5.3", &["E304"], seg, |f| {
            set_meta(f, |d| {
                for class in label_classes(d).iter_mut() {
                    let Some(entry) = class.as_object_mut() else { continue };
                    match entry.get("key").and_then(Value::as_str) {
                        Some("liver") => {
                            entry.insert("parents".into(), json!([3]));
                        }
                        Some("lesion") => {
                            entry.insert("parents".into(), json!([1]));
                        }
                        _ => {}
                    }
                }
            })
        }),
        invalid("E305-ref-without-uri", "A `form: ref` label set with no URI.", "§5.1", &["E005", "E305"], seg, |f| {
            set_meta(f, |d| {
                d.insert(
                    "label_set".into(),
                    json!({"id": "external-v1", "version": "1.0.0", "form": "ref", "sha256": "0".repeat(64)}),
                );
            })
        }),
        invalid("E306-unknown-parent", "A class naming a parent that does not exist.", "§5.2", &["E306"], seg, |f| {
            set_meta(f, |d| {
                if let Some(Value::Object(first)) = label_classes(d).first_mut() {
                    first.insert("parents".into(), json!([900]));
                }
            })
        }),
        invalid(
            "W902-uncompressed-bulk",
            "A multi-megabyte image stored contiguous and uncompressed.",
            "§14.1",
            &[],
            |p| base(p, &[48, 112, 112]),
            |f| rewrite_dataset(f, "images/CT", None),
        )
        .warnings(&["W902"]),
    ]);
}

fn geometric_cases(cases: &mut Vec<Case>) {
    let keypoints = Detection { keypoints: true, ..Default::default() };
    let world = Detection { world: true, ..Default::default() };
    cases.extend([
        case("det-boxes-obb", "Axis-aligned and oriented boxes with scores and attributes.", "§8.2, §8.3", |p| {
            det_base(p, Detection::default())
        })
        .warnings(&["W912"]),
        case("det-keypoints", "Keypoints with per-slot classes, visibility and a skeleton.", "§8.4", move |p| {
            det_base(p, keypoints)
        })
        .warnings(&["W912"]),
        case("det-boxes-world", "Boxes stored in world coordinates of a named frame.", "§8.1", move |p| {
            det_base(p, world)
        })
        .warnings(&["W912"]),
        case(
            "shapes-contours-mesh",
            "Planar contours with a hole, a surface mesh, and landmarks.",
            "§8.5, §8.6, §8.7",
            shape_base,
        )
        .warnings(&["W912"]),
        case(
            "cls-staging-and-change",
            "Per-visit staging with an ordinal scheme, plus a change label naming the timepoints compared.",
            "§9",
            cls_base,
        )
        .warnings(&["W911", "W912"]),
        invalid(
            "E406-box-lo-gt-hi-boxes",
            "An axis-aligned box with lo greater than hi.",
            "§8.1",
            &["E406"],
            |p| det_base(p, Detection { bad_box: true, ..Default::default() }),
            |_| Ok(()),
        )
        .warnings(&["W912"]),
        invalid(
            "E407-improper-rotation",
            "An `obb` rotation matrix that is not a proper rotation.",
            "§8.3",
            &["E407"],
            |p| det_base(p, Detection { bad_rotation: true, ..Default::default() }),
            |_| Ok(()),
        )
        .warnings(&["W912"]),
        invalid(
            "E412-missing-space",
            "A geometric annotation with no `space`.",
            "§8.1",
            &["E412"],
            |p| det_base(p, Detection::default()),
            |f| del_attr(f, "annotations/lesions", "space"),
        )
        .warnings(&["W912"]),
        invalid(
            "E413-unknown-skeleton",
            "A `keypoints` annotation naming a skeleton the label set lacks.",
            "§8.4",
            &["E413"],
            move |p| det_base(p, keypoints),
            |f| set_str(f, "annotations/landmarks", "skeleton", "nope"),
        )
        .warnings(&["W912"]),
        invalid(
            "E414-world-space-on-px-grid",
            "World coordinates on an uncalibrated (units='px') grid.",
            "§3.5",
            &["E414"],
            move |p| det_base(p, world),
            |f| set_str(f, "grids/ct", "units", "px"),
        )
        .warnings(&["W912"]),
        invalid(
            "E404-single-label-two-positives",
            "multilabel=false with two positive classes in one scope unit.",
            "§9",
            &["E404"],
            cls_base,
            crowd_one_scope_unit,
        )
        .warnings(&["W911", "W912"]),
        invalid(
            "E408-contour-offsets",
            "Contour offsets that do not end at the vertex count.",
            "§8.6",
            &["E408"],
            shape_base,
            |f| {
                edit_values(f, "annotations/rtstruct/contour_offsets", |o| {
                    for (i, v) in [0.0, 4.0, 5.0].iter().enumerate() {
                        o[IxDyn(&[i])] = *v;
                    }
                })
            },
        )
        .warnings(&["W912"]),
    ]);
}

fn transform_cases(cases: &mut Vec<Case>) {
    let refined = Registration { displacement: true, composite: true, ..Default::default() };
    let inverse = Registration { inverse: true, ..Default::default() };
    let plain = |p: &Path| reg_base(p, Registration::default());
    cases.extend([
        case(
            "reg-affine-landmarks",
            "Baseline-to-follow-up affine with paired landmark ground truth and a metrics record.",
            "§10.3, §10.6",
            plain,
        ),
        case("reg-inverse-pair", "Two affines declaring each other as inverses.", "§10.1", move |p| {
            reg_base(p, inverse)
        }),
        case(
            "reg-displacement-composite",
            "A dense field refining an affine, and the composite of both.",
            "§10.4, §10.5",
            move |p| reg_base(p, refined),
        ),
        case("reg-bspline", "A cubic free-form deformation on a control-point lattice.", "§10.5", |p| {
            reg_base(p, Registration { bspline: true, ..Default::default() })
        }),
        invalid(
            "E501-broken-composite-chain",
            "A composite whose components do not chain.",
            "§10.5",
            &["E501"],
            move |p| reg_base(p, refined),
            |f| set_str(f, "transforms/tp0_to_refined", "to_frame", "pseudo:frame-999"),
        ),
        invalid(
            "E502-unknown-transform-kind",
            "A transform of an unregistered kind.",
            "§10.1",
            &["E502"],
            plain,
            |f| set_str(f, "transforms/tp0_to_tp1", "kind", "wormhole"),
        ),
        invalid(
            "E503-field-grid-wrong-frame",
            "A displacement field sampled outside the source frame.",
            "§10.4",
            &["E503"],
            |p| reg_base(p, Registration { displacement: true, ..Default::default() }),
            |f| set_str(f, "transforms/refine", "from_frame", FRAME0),
        ),
        invalid(
            "E504-affine-last-row",
            "An affine whose last row is not [0 \u{2026} 0 1].",
            "§10.3",
            &["E504"],
            plain,
            break_affine_last_row,
        ),
        invalid(
            "E505-inverse-not-mutual",
            "An `inverse_id` naming a transform that is not the inverse.",
            "§10.1",
            &["E505"],
            move |p| reg_base(p, inverse),
            |f| set_str(f, "transforms/tp1_to_tp0", "inverse_id", "tp1_to_tp0"),
        ),
    ]);
}

fn tracking_and_collection_cases(cases: &mut Vec<Case>) {
    cases.extend([
        case(
            "longitudinal-instance-tracking",
            "One lesion followed across two visits, joined on `instance_id`.",
            "§7.4, §11.3",
            |p| tracking_sample(p, false, false),
        )
        .warnings(&["W912"]),
        case(
            "W909-instance-reclassified-across-timepoints",
            "One `instance_id` carrying a different class at follow-up.",
            "§7.4",
            |p| tracking_sample(p, true, false),
        )
        .warnings(&["W909", "W912"]),
        case(
            "W904-follow-up-coverage-withdrawn",
            "A follow-up that commits to no class, so absence measures nothing.",
            "§11.3",
            |p| tracking_sample(p, false, true),
        )
        .warnings(&["W904", "W912"]),
        case(
            "collection-two-samples",
            "Two sample roots in one shard, each independently identifiable.",
            "§2.2",
            |p| collection(p, 2, false, false),
        )
        .collection(),
        case(
            "E010-collection-member-without-content-id",
            "A packed sample root that lost its own `content_id`.",
            "§2.2",
            |p| collection(p, 2, true, false),
        )
        .errors(&["E010"])
        .collection()
        .mutated(),
        case("E003-collection-bad-sample-key", "A sample key outside [A-Za-z0-9_.-]{1,255}.", "§2.2", |p| {
            collection(p, 1, false, false)?;
            mutate(p, |root| Ok(root.group(SAMPLES_GROUP)?.relink("case.0", "not a key")?))
        })
        .errors(&["E003"])
        .collection()
        .mutated(),
        case(
            "E008-collection-without-samples-group",
            "A file declaring `collection` with nothing in it.",
            "§2.2",
            |p| collection(p, 1, false, true),
        )
        .errors(&["E008"])
        .collection()
        .mutated(),
    ]);
}

/// 1.3.0: every cross-reference clause gets a case, not only every code.
fn fourth_batch(cases: &mut Vec<Case>) {
    let plain_reg = |p: &Path| reg_base(p, Registration::default());
    cases.extend([
        case("seg-probmap-threshold", "Soft labels with a declared decision threshold.", "§7.5, §7.6", |p| {
            seg_probmap(p, Some(0.3))
        })
        .warnings(&["W912"]),
        invalid_core(
            "E703-unknown-digest-algo",
            "A `digest_algo` outside the §2.1 vocabulary.",
            "§2.1",
            &["E703"],
            |f| set_str(f, "", "digest_algo", "blake3"),
        )
        .level("integrity"),
        invalid(
            "E411-labelmap-wide-dtype",
            "A `labelmap` stored uint16 whose ids fit uint8 and carries no ignore voxel.",
            "§7.1",
            &["E411"],
            |p| seg_base(p, "labelmap", &EXCLUSIVE, false),
            |f| rewrite_dataset(f, "annotations/organs/data", Some(DType::U16)),
        )
        .warnings(&["W912"]),
        invalid(
            "E109-time-axis-without-time-values",
            "A grid with a `time` axis and no `time_values`.",
            "§3.2",
            &["E109"],
            core_4d,
            |f| del_attr(f, "grids/dce", "time_values"),
        ),
        invalid_seg(
            "E413-dangling-ignore-mask",
            "An `ignore_mask` naming an annotation that does not exist.",
            "§7.7",
            &["E413"],
            |f| set_str(f, "annotations/organs", "ignore_mask", "nope"),
        ),
        invalid_seg(
            "E413-ignore-mask-not-a-mask",
            "An `ignore_mask` naming an annotation that is not a `mask`.",
            "§7.7",
            &["E413"],
            |f| set_str(f, "annotations/organs", "ignore_mask", "organs"),
        ),
        invalid_seg(
            "E413-dangling-derived-from",
            "A `derived_from` entry naming an annotation that does not exist.",
            "§6.2",
            &["E413"],
            |f| set_attr(f, "annotations/organs", "derived_from", AttrValue::strs(&["ghost"])),
        ),
        invalid_core(
            "E413-dangling-valid-mask",
            "An image `valid_mask` naming an annotation that does not exist.",
            "§4.4",
            &["E413"],
            |f| set_str(f, "images/CT", "valid_mask", "nope"),
        ),
        invalid(
            "E601-transform-dangling-prov",
            "A transform naming an activity that does not exist.",
            "§10.1, §11.1",
            &["E601"],
            plain_reg,
            |f| set_str(f, "transforms/tp0_to_tp1", "prov", "act_nope"),
        ),
        invalid(
            "E602-transform-unknown-metrics",
            "A transform whose `metrics` names no quality record.",
            "§10.1, §11.2",
            &["E602"],
            plain_reg,
            |f| set_str(f, "transforms/tp0_to_tp1", "metrics", "nope"),
        ),
        invalid_core(
            "E604-split-assigned-at",
            "A split claim whose `assigned_at` is not RFC 3339.",
            "§12.3",
            &["E604"],
            |f| {
                set_meta(f, |d| {
                    d.insert(
                        "splits".into(),
                        json!([{"set_id": "cv5", "partition": "train", "assigned_at": "yesterday"}]),
                    );
                })
            },
        ),
        invalid_core(
            "E604-deidentification-date",
            "A de-identification record whose `date` is not RFC 3339.",
            "§11.4",
            &["E604"],
            |f| {
                set_meta(f, |d| {
                    member(d, "deidentification", json!({"method": "dicom-psi-profile"}))
                        .insert("date".into(), json!("yesterday"));
                })
            },
        ),
    ]);
}

/// 1.4.0: §7.7's separate-mask form of an ignore region, as the writer emits it.
fn fifth_batch(cases: &mut Vec<Case>) {
    cases.push(
        case(
            "seg-bitmask-ignore-mask",
            "A `bitmask` whose ignore region is a sibling `mask` named by `ignore_mask`: partial coverage, no W904.",
            "§7.7, §11.3",
            seg_bitmask_ignore_mask,
        )
        .warnings(&["W912"]),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_registry_holds_every_case_once() {
        let cases = registry();
        assert_eq!(cases.len(), 117);
        let mut names: Vec<&str> = cases.iter().map(|c| c.name.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), 117);
    }

    #[test]
    fn every_error_code_named_by_a_case_is_in_the_table() {
        for case in registry() {
            for code in case.errors.iter().chain(&case.warnings) {
                assert!(crate::codes::get(code).is_some(), "{} names unknown code {code}", case.name);
            }
        }
    }
}
