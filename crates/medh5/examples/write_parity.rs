//! Write the sample `scratchpad/parity/write_py.py` writes, through the Rust
//! writer, and print its `content_id`.  Used to prove byte-level parity.

use std::path::Path;

use medh5::annotations::Assertions;
use medh5::array::NdArray;
use medh5::labels::{ClassKey, LabelClass, LabelSet};
use medh5::sample::{
    create, Annotated, AnnotationOptions, GridOptions, ImageOptions, ObjectFields, Placement, QualityArg,
    SegmentationOptions, SegmentationSource, TransformSpec,
};
use ndarray::{ArrayD, IxDyn};
use serde_json::{json, Map, Value};

fn obj(v: Value) -> Map<String, Value> {
    match v {
        Value::Object(m) => m,
        _ => Map::new(),
    }
}

fn block(shape: &[usize], o: [usize; 3], s: usize) -> ArrayD<bool> {
    ArrayD::from_shape_fn(IxDyn(shape), |i| (0..3).all(|a| i[a] >= o[a] && i[a] < o[a] + s))
}

fn main() -> medh5::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = Path::new(&args[1]);
    let codec = &args[2];
    let shape = [16usize, 24, 24];
    let ct = ArrayD::from_shape_fn(IxDyn(&shape), |i| ((((i[0] * 7 + i[1] * 3 + i[2]) % 2000) as i64) - 1000) as i16);
    let masks = vec![
        (ClassKey::Id(1), block(&shape, [2, 2, 2], 8)),
        (ClassKey::Id(2), block(&shape, [2, 14, 2], 6)),
        (ClassKey::Id(3), block(&shape, [4, 4, 4], 3)),
    ];
    let mut lesion = LabelClass::new(3, "lesion", "Lesion")?;
    lesion.parents = vec![1];
    lesion.category = Some("lesion".into());
    let mut liver = LabelClass::new(1, "liver", "Liver")?;
    liver.category = Some("organ".into());
    let mut spleen = LabelClass::new(2, "spleen", "Spleen")?;
    spleen.category = Some("organ".into());
    let mut vessel = LabelClass::new(4, "vessel", "Vessel")?;
    vessel.category = Some("vessel".into());
    let ls = LabelSet::new(
        "test-v1",
        vec![liver, spleen, lesion, vessel],
        "1.0.0",
        Vec::new(),
        Vec::new(),
        "inline",
        None,
        None,
    )?;

    let mut w = create(path, Some("case-1"), Some("subj-A"), codec, &[])?;
    w.identity(obj(json!({"sex": "F", "bodypart": "abdomen"})))?;
    w.cohort(obj(json!({"dataset_id": "test", "site_id": "site-A"})))?;
    w.add_timepoint("tp0", obj(json!({"label": "baseline", "days_from_baseline": 0})))?;
    w.add_timepoint("tp1", obj(json!({"label": "fu1", "days_from_baseline": 90})))?;
    w.label_set(ls);
    let tool = w.software("medh5", Some("2.0.0"), Map::new())?;
    let act = w.activity("import", Some(&tool.id), None, obj(json!({"tool": "parity"})))?;
    for tp in ["tp0", "tp1"] {
        w.add_grid(
            &format!("ct_{tp}"),
            &[16, 24, 24],
            &[1.5, 0.8, 0.8],
            GridOptions {
                origin: Some(vec![-12.0, -9.6, -9.6]),
                timepoint: Some(tp.into()),
                frame_uid: Some(format!("pseudo:frame-{tp}")),
                patch_hint: Some(vec![8, 8, 8]),
                ..Default::default()
            },
        )?;
        w.add_image(
            &format!("CT_{tp}"),
            &NdArray::from(ct.clone()),
            &format!("ct_{tp}"),
            "CT",
            ImageOptions {
                value_type: Some("quantitative".into()),
                value_units: Some("HU".into()),
                prov: Some(act.id.clone()),
                ..Default::default()
            },
        )?;
        w.add_segmentation(
            &format!("organs_{tp}"),
            &format!("ct_{tp}"),
            SegmentationSource::Masks(masks.clone()),
            SegmentationOptions {
                common: AnnotationOptions {
                    annotated_classes: Annotated::AllGiven,
                    prov: Some(act.id.clone()),
                    quality: Some(QualityArg::Record(obj(json!({"status": "approved"})))),
                    ..Default::default()
                },
                ..Default::default()
            },
        )?;
    }
    let boxes = ArrayD::from_shape_vec(IxDyn(&[1, 3, 2]), vec![3.5, 7.5, 3.5, 7.5, 3.5, 7.5]).unwrap();
    w.add_boxes(
        "lesions",
        &boxes,
        &[ClassKey::Key("lesion".into())],
        ObjectFields { instance_ids: Some(vec![7]), scores: Some(vec![0.9]), attributes: None },
        None,
        Placement { grid: Some("ct_tp0".into()), ..Default::default() },
        AnnotationOptions {
            prov: Some(act.id.clone()),
            quality: Some(QualityArg::Key("organs_tp0".into())),
            ..Default::default()
        },
    )?;
    w.add_classification(
        "dx",
        Assertions { class_ids: vec![3, 4], values: vec![1.0, 0.0], ..Default::default() },
        "sample",
        true,
        None,
        AnnotationOptions {
            prov: Some(act.id.clone()),
            quality: Some(QualityArg::Key("organs_tp0".into())),
            ..Default::default()
        },
    )?;
    let matrix = ArrayD::from_shape_vec(
        IxDyn(&[4, 4]),
        vec![1.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0, -1.5, 0.0, 0.0, 1.0, 0.25, 0.0, 0.0, 0.0, 1.0],
    )
    .unwrap();
    w.add_transform(
        "tp0_to_tp1",
        "affine",
        "pseudo:frame-tp0",
        "pseudo:frame-tp1",
        TransformSpec { matrix: Some(matrix), prov: Some(act.id.clone()), ..Default::default() },
    )?;
    w.build_index(None, Some(64), None, 0)?;
    w.deidentification(obj(json!({"method": "dicom-psi-profile", "date_shift_days": -117})))?;
    let id = w.commit(true)?;
    println!("{}", id.unwrap_or_default());
    Ok(())
}
