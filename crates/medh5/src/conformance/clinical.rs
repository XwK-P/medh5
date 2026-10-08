//! The clinical profile's cases (format 1.1).
//!
//! Valid cases are written by the public writer: the worked example of 1.1
//! §9.3 (two visits, an intervening lab, a delayed report, its revision and a
//! follow-up assessment), a one-visit history, the time semantics §5.2 keeps
//! apart, UTF-8 text, a vocabulary wider than any class id, a mixed-version
//! collection and a higher minor read as a projection.  Invalid ones are those
//! files edited afterwards --- at the column level with raw HDF5 edits, at the
//! record level by rewriting the tables with records the writer would refuse.

use std::path::Path;

use ndarray::ArrayD;
use serde_json::{json, Value};

use super::build::{
    block, case, ct_volume, del_attr, dims, fields, invalid, mutate, psi, quantitative, restamp, rewrite_dataset,
    set_attr, set_meta, set_str, write, FRAME0, FRAME1, SHAPE, SPACING,
};
use super::Case;
use crate::annotations::InstanceInput;
use crate::array::{DType, NdArray};
use crate::clinical::model::{
    Bounds, ClinicalRecords, Clock, Descriptor, Document, Event, Link, ASSESSMENT_SYSTEM, DAY, HOUR, LESION_PRESENCE,
};
use crate::clinical::table::{self, Clinical};
use crate::collection::pack;
use crate::h5::attrs::{self, AttrValue};
use crate::h5::data::{self, Layout};
use crate::h5::ops;
use crate::labels::{LabelClass, LabelSet, OntologyCode};
use crate::rng::Rng;
use crate::sample::{
    Annotated, AnnotationOptions, GridOptions, SampleWriter, SegmentationOptions, SegmentationSource, TransformSpec,
};
use crate::storage::codecs::resolve_profile;
use crate::Result;

const H: i64 = HOUR;
/// The baseline report's text: ASCII but for one multibyte character, so a
/// span can be shown to respect --- or split --- a UTF-8 character.
pub const REPORT_V1: &str =
    "CT chest, baseline. A 14 mm nodule \u{2248} 1.4 cm in the right lower lobe. Preliminary read.";
/// Its revision, issued two days later.
pub const REPORT_V2: &str =
    "CT chest, baseline. A 14 mm nodule \u{2248} 1.4 cm in the right lower lobe; no lymphadenopathy. Final read, amended.";

fn lesion_label_set() -> Result<LabelSet> {
    let lesion = LabelClass::build(
        3,
        "lesion".into(),
        "Lesion".into(),
        Vec::new(),
        Some("lesion".into()),
        Some(vec![255, 214, 64, 255]),
        vec![OntologyCode { system: "SNOMED-CT".into(), code: "52988006".into(), name: Some("Lesion".into()) }],
        None,
        serde_json::Map::new(),
    )?;
    LabelSet::new("conformance-clinical-v1", vec![lesion], "1.0.0", Vec::new(), Vec::new(), "inline", None, None)
}

fn point(id: &str, record: &str, kind: &str, effective: i64, available: Option<i64>, status: &str) -> Event {
    Event {
        event_id: id.into(),
        record_id: record.into(),
        kind: kind.into(),
        temporal_type: "point".into(),
        effective_start: Some(Bounds::exact(effective)),
        available: available.map(Bounds::exact),
        status: status.into(),
        prov: Some("act_clinical_import".into()),
        ..Default::default()
    }
}

fn asserted(mut link: Link, by: &str) -> Link {
    link.asserted_by_event_id = Some(by.into());
    link
}

/// The clinical records of the §9.3 worked example, in hours from the
/// baseline CT.
pub fn worked_records() -> ClinicalRecords {
    let mut records = ClinicalRecords::new(Descriptor::new(Clock::relative(
        "subject-clock-01",
        "acquisition start of the baseline CT (timepoint tp0), on the subject's study timeline",
    )));
    let mut lab0 = point("lab0", "lab0", "observation", -48 * H, Some(-47 * H), "final");
    lab0.code_system = Some("http://loinc.org".into());
    lab0.code = Some("2160-0".into());
    lab0.code_version = Some("2.77".into());
    lab0.value_num = Some(1.1);
    lab0.unit = Some("mg/dL".into());
    let mut ct0 = point("ct0", "ct0", "imaging", 0, Some(H), "final");
    ct0.timepoint_id = Some("tp0".into());
    let report_v1 = point("report0_v1", "report0", "document", 0, Some(4 * H), "preliminary");
    let report_v2 = point("report0_v2", "report0", "document", 0, Some(48 * H), "amended");
    let mut ct1 = point("ct1", "ct1", "imaging", 2160 * H, Some(2161 * H), "final");
    ct1.timepoint_id = Some("tp1".into());
    let mut response = point("response1", "response1", "assessment", 2160 * H, Some(2184 * H), "final");
    response.code_system = Some(ASSESSMENT_SYSTEM.into());
    response.code = Some(LESION_PRESENCE.into());
    response.value_text = Some("resolved".into());
    response.timepoint_id = Some("tp1".into());
    response.prov = Some("act_read_tp1".into());
    // Annotations drawn after the follow-up read: available then, not at baseline.
    let mut seg0 = point("seg0", "seg0", "procedure", 2200 * H, Some(2200 * H), "completed");
    seg0.temporal_type = "unknown".into();
    seg0.effective_start = None;
    let mut grounding = point("grounding0", "grounding0", "other", 0, Some(2300 * H), "final");
    grounding.temporal_type = "unknown".into();
    grounding.effective_start = None;
    records.events = vec![lab0, ct0, report_v1, report_v2, ct1, response, seg0, grounding];
    let mut v1 = Document::text("report0_text_v1", REPORT_V1);
    v1.language = Some("en".into());
    v1.source_type = Some("radiology_report".into());
    let mut v2 = Document::text("report0_text_v2", REPORT_V2);
    v2.language = Some("en".into());
    v2.source_type = Some("radiology_report".into());
    records.documents = vec![v1, v2];
    let start = REPORT_V1.find("14 mm nodule").expect("span text") as u64;
    let mut grounded = Link::new(("document", "report0_text_v1"), "describes", ("instance", "1"));
    grounded.source_span = Some((start, start + "14 mm nodule".len() as u64));
    grounded.target_annotation_id = Some("lesions_tp0".into());
    let mut assessed = Link::new(("event", "response1"), "assesses", ("instance", "1"));
    assessed.target_annotation_id = Some("lesions_tp0".into());
    records.links = vec![
        Link::new(("event", "ct0"), "describes", ("image", "CT_tp0")),
        Link::new(("event", "ct1"), "describes", ("image", "CT_tp1")),
        Link::new(("event", "report0_v1"), "describes", ("document", "report0_text_v1")),
        Link::new(("event", "report0_v2"), "describes", ("document", "report0_text_v2")),
        Link::new(("event", "report0_v2"), "supersedes", ("event", "report0_v1")),
        asserted(assessed, "response1"),
        asserted(Link::new(("event", "response1"), "derived_from", ("image", "CT_tp1")), "response1"),
        asserted(Link::new(("event", "response1"), "compares_with", ("timepoint", "tp0")), "response1"),
        asserted(Link::new(("event", "seg0"), "describes", ("annotation", "lesions_tp0")), "seg0"),
        asserted(Link::new(("event", "seg0"), "describes", ("annotation", "lesions_tp1")), "seg0"),
        asserted(grounded, "grounding0"),
    ];
    records
}

/// The imaging half of the worked example: two visits on their own frames, a
/// lesion at baseline that is gone at follow-up, and the registration.
fn worked_imaging(w: &mut SampleWriter) -> Result<()> {
    let mut rng = Rng::new(super::SEED);
    w.add_timepoint("tp0", fields(json!({"label": "baseline", "days_from_baseline": 0})))?;
    w.add_timepoint("tp1", fields(json!({"label": "follow_up_3mo", "days_from_baseline": 90})))?;
    w.label_set(lesion_label_set()?);
    for (gid, tp, frame) in [("ct_tp0", "tp0", FRAME0), ("ct_tp1", "tp1", FRAME1)] {
        w.add_grid(
            gid,
            &dims(&SHAPE),
            &SPACING,
            GridOptions { timepoint: Some(tp.into()), frame_uid: Some(frame.into()), ..Default::default() },
        )?;
        w.add_image(&format!("CT_{tp}"), &ct_volume(&mut rng, &SHAPE)?, gid, "CT", quantitative(None))?;
    }
    let lesion = InstanceInput {
        class_id: 3,
        instance_id: 1,
        mask: Some(block(&SHAPE, &[4, 6, 6], &[4, 5, 5])),
        bbox: None,
        crop: None,
        score: None,
    };
    let examined = AnnotationOptions {
        annotated_classes: Annotated::Classes(vec![crate::labels::ClassKey::Id(3)]),
        ..Default::default()
    };
    w.add_segmentation(
        "lesions_tp0",
        "ct_tp0",
        SegmentationSource::Instances(vec![lesion]),
        SegmentationOptions { common: examined.clone(), ..Default::default() },
    )?;
    w.add_segmentation(
        "lesions_tp1",
        "ct_tp1",
        SegmentationSource::Instances(Vec::new()),
        SegmentationOptions { common: examined, ..Default::default() },
    )?;
    let eye: ArrayD<f64> = ndarray::Array2::<f64>::eye(4).into_dyn();
    w.add_transform(
        "tp0_to_tp1",
        "affine",
        FRAME0,
        FRAME1,
        TransformSpec {
            matrix: Some(eye),
            from_grid: Some("ct_tp0".into()),
            to_grid: Some("ct_tp1".into()),
            ..Default::default()
        },
    )?;
    w.software("medh5-conformance", Some(crate::VERSION), serde_json::Map::new())?;
    w.person("pseudonym:RAD-07", Some("rad1"), fields(json!({"role": "reviewer"})))?;
    w.activity("import", Some("s1"), Some("act_clinical_import"), serde_json::Map::new())?;
    w.activity("review", Some("rad1"), Some("act_read_tp1"), serde_json::Map::new())?;
    w.deidentification(psi())?;
    Ok(())
}

/// The §9.3 worked example as a file.
pub fn worked_example(path: &Path) -> Result<()> {
    worked_example_with(path, |_| {})
}

/// The worked example with its records edited before the public writer
/// writes them: a valid variant, digests and all.
fn worked_example_with(path: &Path, edit: impl FnOnce(&mut ClinicalRecords)) -> Result<()> {
    write(path, Some("subj-clinical-01"), |w| {
        worked_imaging(w)?;
        let mut records = worked_records();
        edit(&mut records);
        w.add_records(records)
    })
}

/// Rewrite `clinical/` from edited records, with no rule checked: how a
/// record-level defect reaches a file the writer would refuse to write.
fn rewrite_clinical(root: &hdf5::Group, edit: impl FnOnce(&mut ClinicalRecords)) -> Result<()> {
    let mut records = Clinical::open(root, false)?.expect("a clinical sample").records()?;
    edit(&mut records);
    ops::unlink(root, crate::clinical::GROUP)?;
    table::write(root, &records, &resolve_profile(Some("portable"))?)
}

fn event_mut<'a>(records: &'a mut ClinicalRecords, id: &str) -> &'a mut Event {
    records.events.iter_mut().find(|e| e.event_id == id).expect("event in the worked example")
}

/// The stored row of an event (`Clinical::open` reads rows in stored order).
fn row_of(root: &hdf5::Group, event_id: &str) -> Result<usize> {
    let clinical = Clinical::open(root, false)?.expect("a clinical sample");
    Ok(clinical.events.iter().position(|e| e.event_id == event_id).expect("event present"))
}

/// Replace a dataset's values, keeping its attributes (the digest included).
fn replace_values(root: &hdf5::Group, path: &str, values: NdArray) -> Result<()> {
    let ds = root.dataset(path)?;
    let saved: Vec<(String, Option<AttrValue>)> =
        attrs::names(&ds)?.into_iter().map(|n| (n.clone(), attrs::read(&ds, &n).ok().flatten())).collect();
    drop(ds);
    let (parent, name) = super::build::parent_of(root, path)?;
    parent.unlink(&name)?;
    let new = data::create(&parent, &name, &values, &Layout::contiguous())?;
    for (k, v) in saved {
        if let Some(v) = v {
            attrs::write(&new, &k, &v)?;
        }
    }
    Ok(())
}

fn edit_u8(root: &hdf5::Group, path: &str, edit: impl FnOnce(&mut Vec<u8>)) -> Result<()> {
    let mut values: Vec<u8> = data::read(&root.dataset(path)?)?.cast::<u8>().iter().copied().collect();
    edit(&mut values);
    replace_values(root, path, NdArray::from_vec(&[values.len()], values)?)
}

fn edit_f64(root: &hdf5::Group, path: &str, edit: impl FnOnce(&mut Vec<f64>)) -> Result<()> {
    let mut values: Vec<f64> = data::read(&root.dataset(path)?)?.to_f64().iter().copied().collect();
    edit(&mut values);
    replace_values(root, path, NdArray::from_vec(&[values.len()], values)?)
}

fn edit_u64(root: &hdf5::Group, path: &str, edit: impl FnOnce(&mut Vec<u64>)) -> Result<()> {
    let mut values: Vec<u64> = data::read(&root.dataset(path)?)?.cast::<u64>().iter().copied().collect();
    edit(&mut values);
    replace_values(root, path, NdArray::from_vec(&[values.len()], values)?)
}

fn set_descriptor(root: &hdf5::Group, text: &str) -> Result<()> {
    let group = root.group(crate::clinical::GROUP)?;
    let old = group.dataset("meta")?;
    let digest = attrs::read(&old, "digest")?;
    drop(old);
    group.unlink("meta")?;
    let new = data::create_scalar_string(&group, "meta", text)?;
    if let Some(d) = digest {
        attrs::write(&new, "digest", &d)?;
    }
    Ok(())
}

/// One CT, a history of labs, an order and its later administration: the
/// profile without the `longitudinal` one (1.1 §1).
fn one_visit_history(path: &Path) -> Result<()> {
    let mut rng = Rng::new(super::SEED);
    write(path, Some("subj-clinical-02"), |w| {
        w.add_grid("ct", &dims(&SHAPE), &SPACING, GridOptions::default())?;
        w.add_image("CT", &ct_volume(&mut rng, &SHAPE)?, "ct", "CT", quantitative(None))?;
        w.deidentification(psi())?;
        w.set_clock(Clock::relative("subject-clock-02", "the subject's enrolment visit"))?;
        for (i, years) in [-3.0f64, -2.0, -1.0, -0.5, -0.1].iter().enumerate() {
            let at = (years * 365.0 * DAY as f64) as i64;
            let mut lab =
                point(&format!("hba1c_{i}"), &format!("hba1c_{i}"), "observation", at, Some(at + 6 * H), "final");
            lab.prov = None;
            lab.code_system = Some("http://loinc.org".into());
            lab.code = Some("4548-4".into());
            lab.value_num = Some(6.1 + i as f64 * 0.2);
            lab.unit = Some("%".into());
            w.add_event(lab)?;
        }
        let mut ct = point("ct", "ct", "imaging", 0, Some(2 * H), "final");
        ct.prov = None;
        ct.timepoint_id = Some("tp0".into());
        w.add_event(ct)?;
        w.add_link(Link::new(("event", "ct"), "describes", ("image", "CT")))?;
        let mut order = point("metformin_order", "metformin_order", "medication_order", 10 * DAY, Some(DAY), "planned");
        order.prov = None;
        order.code_system = Some("http://www.nlm.nih.gov/research/umls/rxnorm".into());
        order.code = Some("6809".into());
        w.add_event(order)?;
        let mut given = point(
            "metformin_given",
            "metformin_given",
            "medication_administration",
            10 * DAY,
            Some(10 * DAY + 2 * H),
            "completed",
        );
        given.prov = None;
        given.code_system = Some("http://www.nlm.nih.gov/research/umls/rxnorm".into());
        given.code = Some("6809".into());
        w.add_event(given)?;
        let mut sex = point("sex", "sex", "observation", 0, Some(0), "final");
        sex.prov = None;
        sex.temporal_type = "static".into();
        sex.effective_start = None;
        sex.code_system = Some("http://loinc.org".into());
        sex.code = Some("46098-0".into());
        sex.value_text = Some("female".into());
        w.add_event(sex)?;
        Ok(())
    })
}

/// Unknown availability, day precision, same-time ties, events before
/// baseline, an interval whose end was recorded later, and a diagnosis
/// assigned in retrospect --- each kept as the source knew it (§5.2).
fn time_uncertainty(path: &Path) -> Result<()> {
    let mut rng = Rng::new(super::SEED);
    write(path, Some("subj-clinical-03"), |w| {
        w.add_grid("ct", &dims(&SHAPE), &SPACING, GridOptions::default())?;
        w.add_image("CT", &ct_volume(&mut rng, &SHAPE)?, "ct", "CT", quantitative(None))?;
        w.deidentification(psi())?;
        w.set_clock(Clock::relative("subject-clock-03", "the subject's first recorded contact"))?;
        let coded = |mut e: Event| {
            e.prov = None;
            e.code_system = Some("local.example".into());
            e.code = Some(e.record_id.clone());
            e
        };
        // Known only to the day, availability unknown.
        let mut dated = coded(point("albumin", "albumin", "observation", 0, None, "final"));
        dated.effective_start = Some(Bounds::new(-40 * DAY, -39 * DAY - 1));
        dated.value_num = Some(3.9);
        dated.unit = Some("g/dL".into());
        w.add_event(dated)?;
        // Two results with one timestamp: a tie, not an order.
        for name in ["sodium", "potassium"] {
            let mut tied = coded(point(name, name, "observation", -5 * DAY, Some(-5 * DAY + H), "final"));
            tied.value_num = Some(if name == "sodium" { 139.0 } else { 4.1 });
            tied.unit = Some("mmol/L".into());
            w.add_event(tied)?;
        }
        // An infusion: first known as ongoing, its end recorded later.
        let mut ongoing = coded(point(
            "infusion_v1",
            "infusion",
            "medication_administration",
            -3 * DAY,
            Some(-3 * DAY),
            "in_progress",
        ));
        ongoing.temporal_type = "interval".into();
        w.add_event(ongoing)?;
        let mut ended =
            coded(point("infusion_v2", "infusion", "medication_administration", -3 * DAY, Some(2 * DAY), "completed"));
        ended.temporal_type = "interval".into();
        ended.effective_end = Some(Bounds::new(-DAY, -DAY + 4 * H));
        w.add_event(ended)?;
        w.add_link(Link::new(("event", "infusion_v2"), "supersedes", ("event", "infusion_v1")))?;
        // A diagnosis effective before baseline, assigned months later.
        let mut diagnosis = coded(point("t2dm", "t2dm", "diagnosis", -30 * DAY, Some(60 * DAY), "final"));
        diagnosis.code_system = Some("http://hl7.org/fhir/sid/icd-10".into());
        diagnosis.code = Some("E11".into());
        w.add_event(diagnosis)?;
        // An occurrence nobody timed.
        let mut untimed = coded(point("fall", "fall", "other", 0, Some(DAY), "final"));
        untimed.temporal_type = "unknown".into();
        untimed.effective_start = None;
        w.add_event(untimed)?;
        Ok(())
    })
}

/// Text that is not ASCII, an empty-but-valid string beside a null one, and
/// spans on character boundaries (§4, §6).
fn text_encoding(path: &Path) -> Result<()> {
    let mut rng = Rng::new(super::SEED);
    write(path, Some("subj-clinical-04"), |w| {
        w.add_grid("ct", &dims(&SHAPE), &SPACING, GridOptions::default())?;
        w.add_image("CT", &ct_volume(&mut rng, &SHAPE)?, "ct", "CT", quantitative(None))?;
        w.deidentification(psi())?;
        w.set_clock(Clock::relative("subject-clock-04", "admission"))?;
        let text = "胸部CT：右下葉に結節 14 mm。所見 🫁 は安定。";
        let mut note = point("note", "note", "document", 0, Some(H), "final");
        note.prov = None;
        w.add_event(note)?;
        let mut doc = Document::text("note_text", text);
        doc.language = Some("ja".into());
        doc.source_type = Some("radiology_report".into());
        w.add_document(doc)?;
        w.add_link(Link::new(("event", "note"), "describes", ("document", "note_text")))?;
        let start = text.find("結節").expect("span") as u64;
        let mut span = asserted(Link::new(("document", "note_text"), "measures", ("event", "size")), "note");
        span.source_span = Some((start, start + "結節".len() as u64));
        w.add_link(span)?;
        let mut size = point("size", "size", "observation", 0, Some(H), "final");
        size.prov = None;
        size.code_system = Some("local.example".into());
        size.code = Some("nodule_size".into());
        size.value_text = Some(String::new()); // valid and empty: not null
        w.add_event(size)?;
        let mut absent = point("smoking", "smoking", "observation", 0, Some(H), "final");
        absent.prov = None;
        absent.code_system = Some("local.example".into());
        absent.code = Some("smoking_status".into());
        absent.missing_reason = Some("not_asked".into());
        w.add_event(absent)?;
        Ok(())
    })
}

/// More clinical concepts than a `uint16` class id can name, beside a
/// segmentation label set: the two vocabularies never meet (§5.1).
pub const WIDE_VOCABULARY: usize = 65_600;

fn wide_vocabulary(path: &Path) -> Result<()> {
    let mut rng = Rng::new(super::SEED);
    write(path, Some("subj-clinical-05"), |w| {
        w.label_set(lesion_label_set()?);
        w.add_grid("ct", &dims(&SHAPE), &SPACING, GridOptions::default())?;
        w.add_image("CT", &ct_volume(&mut rng, &SHAPE)?, "ct", "CT", quantitative(None))?;
        w.add_segmentation(
            "lesions",
            "ct",
            SegmentationSource::Masks(vec![(crate::labels::ClassKey::Id(3), block(&SHAPE, &[4, 4, 4], &[3, 3, 3]))]),
            SegmentationOptions::default(),
        )?;
        w.deidentification(psi())?;
        w.set_clock(Clock {
            id: "subject-clock-05".into(),
            unit: "us".into(),
            reference: "utc".into(),
            origin_description: None,
        })?;
        for n in 0..WIDE_VOCABULARY {
            let at = 1_700_000_000_000_000 + n as i64 * 1_000_000;
            let mut e = point(&format!("e{n}"), &format!("e{n}"), "observation", at, Some(at), "final");
            e.prov = None;
            e.code_system = Some("org.example.local".into());
            e.code = Some(format!("CONCEPT-{n:06}"));
            e.value_text = Some("present".into());
            w.add_event(e)?;
        }
        Ok(())
    })
}

/// A plain 1.0 sample, for the mixed-version collection.
fn imaging_only(path: &Path) -> Result<()> {
    let mut rng = Rng::new(super::SEED);
    write(path, Some("subj-imaging"), |w| {
        w.add_grid("ct", &dims(&SHAPE), &SPACING, GridOptions::default())?;
        w.add_image("CT", &ct_volume(&mut rng, &SHAPE)?, "ct", "CT", quantitative(None))?;
        w.deidentification(psi())?;
        Ok(())
    })
}

/// A 1.0 member and a 1.1 member in one shard, whose outer root declares 1.1.
fn mixed_collection(path: &Path) -> Result<()> {
    let dir = tempdir_beside(path)?;
    let a = dir.join("imaging.medh5");
    let b = dir.join("clinical.medh5");
    imaging_only(&a)?;
    worked_example(&b)?;
    let result = pack(&[a.as_path(), b.as_path()], path, None).map(|_| ());
    let _ = std::fs::remove_dir_all(&dir);
    result
}

fn tempdir_beside(path: &Path) -> Result<std::path::PathBuf> {
    let parent = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or_else(|| Path::new("."));
    let dir = parent.join(format!(".{}.parts", super::build::stem(path)));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// A later minor as 1.1 reads it: an unknown profile, `/meta` member, image
/// value type, event kind, column and attribute --- each W913, none an error.
fn higher_minor(path: &Path) -> Result<()> {
    worked_example(path)?;
    mutate(path, |root| {
        rewrite_clinical(root, |r| event_mut(r, "lab0").kind = "genomic_result".into())?;
        set_str(root, "", "medh5_version", "1.2")?;
        set_attr(
            root,
            "",
            "medh5_profiles",
            AttrValue::strs(&["clinical", "core", "future_profile", "longitudinal", "reg", "seg"]),
        )?;
        set_meta(root, |d| {
            d.insert("future_member".into(), json!({"defined_by": "MEDH5 1.2"}));
        })?;
        set_str(root, "images/CT_tp0", "value_type", "photon_counting")?;
        let events = root.group("clinical/events")?;
        let n = Clinical::open(root, true)?.map(|c| c.events.len()).unwrap_or(0);
        data::create(&events, "future_column", &NdArray::from_vec(&[n], vec![0i64; n])?, &Layout::contiguous())?;
        attrs::write(&events, "future_attribute", &AttrValue::Str("defined by 1.2".into()))
    })?;
    // Digests as a 1.2 writer covering nothing new would stamp them: the file
    // verifies, so what a reader reports about it is the version alone.
    restamp(path)
}

fn without_unit(path: &Path) -> Result<()> {
    worked_example_with(path, |r| event_mut(r, "lab0").unit = None)
}

pub(super) fn cases() -> Vec<Case> {
    let mut out = vec![
        case(
            "clinical-worked-example",
            "Two imaging visits, an intervening lab, a delayed report and its revision, a follow-up lesion \
             assessment, and annotations attributed to the events that drew them.",
            "§9.3 (1.1)",
            worked_example,
        ),
        case(
            "clinical-one-visit-history",
            "One CT and years of laboratory history: `core, clinical` without `longitudinal`, with a planned order \
             and its later administration kept as two kinds.",
            "§1, §5.2 (1.1)",
            one_visit_history,
        ),
        case(
            "clinical-time-uncertainty",
            "Unknown availability, day precision, tied times, pre-baseline events, a later-recorded interval end and \
             a retrospective diagnosis, none given precision the source lacked.",
            "§5.2 (1.1)",
            time_uncertainty,
        ),
        case(
            "clinical-text-encoding",
            "Non-ASCII text with spans on character boundaries, a valid empty string beside nulls, and a missing \
             value with its reason.",
            "§4, §6 (1.1)",
            text_encoding,
        ),
        case(
            "clinical-wide-vocabulary",
            "65 600 distinct clinical concepts beside a segmentation label set: clinical codes are strings and never \
             share the uint16 class-id space or its sentinels.",
            "§5.1 (1.1)",
            wide_vocabulary,
        ),
        case(
            "collection-mixed-versions",
            "A 1.0 member and a 1.1 clinical member in one shard; the outer root declares 1.1 and each member keeps \
             its own version, profiles and content_id.",
            "§8 (1.1)",
            mixed_collection,
        )
        .collection(),
        case(
            "W913-higher-minor-projection",
            "A 1.2 file with a profile, `/meta` member, value type, event kind, column and attribute a 1.1 validator \
             does not know: the supported projection is validated and each unknown is reported, not rejected.",
            "§2 (1.1)",
            higher_minor,
        )
        .warnings(&["W913"]),
        case("W914-numeric-value-without-unit", "A laboratory value with no unit.", "§5.1 (1.1)", without_unit)
            .warnings(&["W914"]),
    ];
    out.extend(invalid_cases());
    out
}

fn on_worked(
    name: &str,
    description: &str,
    clause: &str,
    codes: &[&str],
    mutation: impl Fn(&hdf5::Group) -> Result<()> + Send + Sync + 'static,
) -> Case {
    invalid(name, description, clause, codes, worked_example, mutation)
}

fn invalid_cases() -> Vec<Case> {
    vec![
        case("E011-collection-version-below-member", "A shard declaring 1.0 that holds a 1.1 member.", "§8 (1.1)", |p| {
            mixed_collection(p)?;
            mutate(p, |root| set_str(root, "", "medh5_version", "1.0"))
        })
        .errors(&["E011"])
        .collection()
        .mutated(),
        on_worked(
            "E801-descriptor-not-canonical",
            "`clinical/meta` holding indented JSON: equal descriptors must digest equally.",
            "§3, §8 (1.1)",
            &["E801"],
            |f| {
                let value: Value = serde_json::from_str(&Descriptor::new(Clock::relative("c", "baseline")).dumps())?;
                set_descriptor(f, &crate::json::pretty(&value))
            },
        ),
        on_worked(
            "E802-relative-clock-without-origin",
            "A relative clock that does not say what its zero is.",
            "§3 (1.1)",
            &["E802"],
            |f| set_descriptor(f, r#"{"clock":{"id":"subject-clock-01","reference":"relative","unit":"us"},"schema":"medh5.clinical/1"}"#),
        ),
        on_worked(
            "E802-shifted-clock-without-deidentification",
            "A shifted-UTC clock in a sample that declares no de-identification shift.",
            "§3 (1.1)",
            &["E802"],
            |f| {
                set_descriptor(f, r#"{"clock":{"id":"subject-clock-01","reference":"shifted_utc","unit":"us"},"schema":"medh5.clinical/1"}"#)?;
                set_meta(f, |d| {
                    d.remove("deidentification");
                })
            },
        )
        .warnings(&["W903"]),
        on_worked(
            "E803-clinical-content-undeclared",
            "Clinical tables in a 1.1 sample that does not declare the profile.",
            "§3 (1.1)",
            &["E803"],
            |f| set_attr(f, "", "medh5_profiles", AttrValue::strs(&["core", "longitudinal", "reg", "seg"])),
        ),
        on_worked(
            "E009-clinical-profile-in-a-1-0-file",
            "The clinical profile declared by a file claiming MEDH5 1.0.",
            "§1, §3 (1.1)",
            &["E009"],
            |f| set_str(f, "", "medh5_version", "1.0"),
        ),
        on_worked(
            "E804-unknown-event-column",
            "An events column the profile does not define.",
            "§4, §5.1 (1.1)",
            &["E804"],
            |f| {
                let n = Clinical::open(f, false)?.map(|c| c.events.len()).unwrap_or(0);
                data::create(&f.group("clinical/events")?, "mood", &NdArray::from_vec(&[n], vec![0i64; n])?, &Layout::contiguous())?;
                Ok(())
            },
        ),
        on_worked(
            "E805-time-bounds-stored-int32",
            "Availability stored as int32 rather than int64 microseconds.",
            "§4 (1.1)",
            &["E805"],
            |f| rewrite_dataset(f, "clinical/events/available_lo_us", Some(DType::I32)),
        ),
        on_worked(
            "E806-offsets-overrun",
            "A string column whose last offset points past its bytes.",
            "§4 (1.1)",
            &["E806"],
            |f| edit_u64(f, "clinical/events/kind/offsets", |o| {
                if let Some(last) = o.last_mut() {
                    *last += 5;
                }
            }),
        ),
        on_worked(
            "E806-invalid-utf8",
            "A text cell whose bytes are not UTF-8.",
            "§4 (1.1)",
            &["E806"],
            |f| edit_u8(f, "clinical/documents/text/data", |d| d[0] = 0xff),
        ),
        on_worked(
            "E807-null-cell-holds-a-value",
            "A null numeric cell still holding a value, which nulling was meant to remove.",
            "§4 (1.1)",
            &["E807"],
            |f| {
                let row = row_of(f, "ct0")?;
                edit_f64(f, "clinical/events/value_num", |v| v[row] = 5.0)
            },
        ),
        on_worked(
            "E808-nonfinite-value",
            "A valid laboratory value that is NaN.",
            "§4 (1.1)",
            &["E808"],
            |f| {
                let row = row_of(f, "lab0")?;
                edit_f64(f, "clinical/events/value_num", |v| v[row] = f64::NAN)
            },
        ),
        on_worked(
            "E809-duplicate-event-id",
            "Two rows claiming one immutable event id.",
            "§5.1 (1.1)",
            &["E809"],
            |f| {
                rewrite_clinical(f, |r| {
                    let mut copy = r.events[0].clone();
                    copy.record_id = format!("{}_copy", copy.record_id);
                    r.events.push(copy);
                })
            },
        ),
        on_worked("E810-unknown-event-kind", "An event kind outside the vocabulary.", "§5.1 (1.1)", &["E810"], |f| {
            rewrite_clinical(f, |r| event_mut(r, "lab0").kind = "genomic_result".into())
        }),
        on_worked(
            "E811-half-null-time-bounds",
            "An effective start with a lower bound and a null upper bound.",
            "§5.2 (1.1)",
            &["E811"],
            |f| {
                let row = row_of(f, "lab0")?;
                edit_u8(f, "clinical/events/valid/effective_start_hi_us", |m| m[row] = 0)?;
                let ds = f.dataset("clinical/events/effective_start_hi_us")?;
                let mut values: Vec<i64> = data::read(&ds)?.cast::<i64>().iter().copied().collect();
                drop(ds);
                values[row] = 0;
                replace_values(f, "clinical/events/effective_start_hi_us", NdArray::from_vec(&[values.len()], values)?)
            },
        ),
        on_worked(
            "E811-interval-ends-before-it-starts",
            "An interval whose end bounds all precede its start bounds.",
            "§5.2 (1.1)",
            &["E811"],
            |f| {
                rewrite_clinical(f, |r| {
                    let e = event_mut(r, "lab0");
                    e.temporal_type = "interval".into();
                    e.effective_end = Some(Bounds::exact(-50 * H));
                })
            },
        ),
        on_worked(
            "E812-observation-value-without-code",
            "A numeric observation that does not say what was measured.",
            "§5.1 (1.1)",
            &["E812"],
            |f| {
                rewrite_clinical(f, |r| {
                    let e = event_mut(r, "lab0");
                    e.code_system = None;
                    e.code = None;
                })
            },
        ),
        on_worked(
            "E813-dangling-link-target",
            "An imaging event describing an image the sample does not hold.",
            "§7 (1.1)",
            &["E813"],
            |f| {
                rewrite_clinical(f, |r| {
                    for l in r.links.iter_mut().filter(|l| l.target_id == "CT_tp0") {
                        l.target_id = "CT_tp9".into();
                    }
                })
            },
        ),
        on_worked(
            "E814-span-splits-a-character",
            "A text span starting inside a multibyte UTF-8 character.",
            "§6, §7 (1.1)",
            &["E814"],
            |f| {
                let inside = REPORT_V1.find('\u{2248}').expect("the multibyte character") as u64 + 1;
                rewrite_clinical(f, |r| {
                    for l in r.links.iter_mut().filter(|l| l.source_span.is_some()) {
                        l.source_span = Some((inside, inside + 3));
                    }
                })
            },
        ),
        on_worked(
            "E815-document-without-owner",
            "A report whose document event no longer describes it.",
            "§6 (1.1)",
            &["E815"],
            |f| {
                rewrite_clinical(f, |r| {
                    r.links.retain(|l| !(l.source_id == "report0_v1" && l.relation == "describes"));
                })
            },
        ),
        on_worked(
            "E815-document-event-owning-two",
            "A report event that also describes a second document.",
            "§6 (1.1)",
            &["E815"],
            |f| {
                rewrite_clinical(f, |r| {
                    r.documents.push(Document::text("report0_addendum", "Addendum: no change."));
                    r.links.push(Link::new(("event", "report0_v1"), "describes", ("document", "report0_addendum")));
                })
            },
        ),
        on_worked(
            "E816-branching-revisions",
            "Two versions both superseding the preliminary report.",
            "§5.2, §7 (1.1)",
            &["E816"],
            |f| {
                rewrite_clinical(f, |r| {
                    let mut v3 = point("report0_v3", "report0", "document", 0, Some(50 * H), "amended");
                    v3.prov = None;
                    r.events.push(v3);
                    r.links.push(Link::new(("event", "report0_v3"), "supersedes", ("event", "report0_v1")));
                })
            },
        ),
        on_worked(
            "E816-revision-available-before-its-predecessor",
            "A revision whose known availability precedes the version it supersedes.",
            "§7 (1.1)",
            &["E816"],
            |f| rewrite_clinical(f, |r| event_mut(r, "report0_v2").available = Some(Bounds::exact(2 * H))),
        ),
        on_worked(
            "E817-assessment-without-instance",
            "A lesion assessment that links no instance.",
            "§7 (1.1)",
            &["E817"],
            |f| rewrite_clinical(f, |r| r.links.retain(|l| l.relation != "assesses")),
        ),
        on_worked(
            "E818-clinical-dataset-without-digest",
            "A clinical byte buffer stripped of its digest: its line leaves `content_id` too.",
            "§8 (1.1)",
            &["E702", "E818"],
            |f| del_attr(f, "clinical/events/kind/data", "digest"),
        )
        .level("integrity"),
        on_worked(
            "E819-attribute-on-a-clinical-table",
            "A semantic attribute on a clinical table, which no digest would attest.",
            "§3, §8 (1.1)",
            &["E819"],
            |f| set_str(f, "clinical/events", "note", "sorted by acquisition"),
        ),
        on_worked(
            "E701-clinical-text-edited",
            "A report's bytes edited under an unchanged digest and root: per-dataset verification catches it.",
            "§8 (1.1)",
            &["E701"],
            |f| edit_u8(f, "clinical/documents/text/data", |d| d[0] = b'X'),
        )
        .level("integrity"),
        on_worked(
            "E701-clinical-descriptor-edited",
            "A descriptor naming another clock under its old digest.",
            "§8 (1.1)",
            &["E701"],
            |f| set_descriptor(f, &Descriptor::new(Clock::relative("another-clock", "baseline")).dumps()),
        )
        .level("integrity"),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_worked_example_is_written_by_the_public_writer() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("worked.medh5");
        worked_example(&path).unwrap();
        let s = crate::sample::open_sample(&path).unwrap();
        assert_eq!(s.version().unwrap(), "1.1");
        assert!(s.profiles().unwrap().contains("clinical"));
        let c = s.clinical().unwrap().unwrap();
        assert_eq!(c.events.len(), 8);
        assert_eq!(c.text("report0_text_v1").unwrap(), REPORT_V1);
        let report = crate::validate::validate_file(&path, "integrity", None).unwrap();
        assert!(report.ok(), "{:?}", report.diagnostics);
        assert!(report.diagnostics.is_empty(), "{:?}", report.diagnostics);
    }
}
