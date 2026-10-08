//! The task-and-cache contract end to end: manifests, preflight, fragments,
//! pins and caches, through the public API.

use std::path::{Path, PathBuf};

use medh5::array::NdArray;
use medh5::clinical::model::{Bounds, Clock, Event, Link, HOUR};
use medh5::collection::pack;
use medh5::companion::cache::{event_entry_id, validate_cache, CacheEntry, CacheHeader, CacheWriter, FeatureCache};
use medh5::companion::task::{Row, Slot, Subject, TargetSpec, TaskManifest};
use medh5::companion::view::reconcile;
use medh5::companion::{preflight, write_preflight, Admitted, SourceRef};
use medh5::conformance::clinical::worked_example;
use medh5::json::pretty;
use medh5::sample::{amend, create, open_sample, GridOptions, ImageOptions};
use serde_json::json;

fn pinned(path: &Path, id: &str, key: Option<&str>) -> SourceRef {
    let sample = match key {
        None => open_sample(path).unwrap(),
        Some(k) => medh5::collection::open_collection(path).unwrap().get(k).unwrap(),
    };
    let mut src = SourceRef::pin(path.to_string_lossy(), key.map(str::to_string), &sample).unwrap();
    src.source_id = id.into();
    src
}

fn lesion_target() -> TargetSpec {
    TargetSpec {
        id: "lesion-status-100d".into(),
        version: "1".into(),
        kind: Some("assessment".into()),
        code_system: "org.medh5.assessment".into(),
        code: "lesion_presence".into(),
        positive: vec!["present".into()],
        negative: vec!["resolved".into(), "absent".into()],
        horizon_us: 2400 * HOUR,
        min_follow_up_us: 1000 * HOUR,
        censoring: "censor".into(),
        exclude_prevalent: true,
    }
}

fn worked_task(dir: &Path) -> (TaskManifest, PathBuf) {
    let path = dir.join("worked.medh5");
    worked_example(&path).unwrap();
    let mut task = TaskManifest::new("lesion-status", "1", "org.example.trial");
    task.slots.push(Slot {
        name: "ct".into(),
        modality: "CT".into(),
        required: true,
        patch: Some(vec![8, 12, 12]),
        roi: "eligible_instances".into(),
        classes: vec![3],
    });
    task.target = Some(lesion_target());
    task.split = Some(("cv0".into(), vec!["train".into(), "test".into()]));
    task.subjects.push(Subject {
        subject_id: "P-01".into(),
        clock_id: Some("subject-clock-01".into()),
        partition: Some("train".into()),
        sources: vec![pinned(&path, "s0", None)],
        reconciled: Vec::new(),
    });
    for (id, hours) in [("r0", 0), ("r24", 24), ("r2200", 2200)] {
        task.rows.push(medh5::companion::task::Row {
            row_id: id.into(),
            subject_id: "P-01".into(),
            cutoff_us: hours * HOUR,
        });
    }
    (task, path)
}

#[test]
fn s4_rows_admit_exactly_what_was_available() {
    let dir = tempfile::tempdir().unwrap();
    let (task, _) = worked_task(dir.path());
    assert!(task.validate().is_empty(), "{:?}", task.validate());
    let pre = preflight(&task, None, false).unwrap();
    assert!(pre.ok(), "{:?}", pre.findings);
    let r0 = pre.row("r0").unwrap();
    assert_eq!(r0.status, "excluded", "the baseline CT is not available yet at hour 0");
    assert_eq!(r0.reasons, ["missing_required_slot:ct"]);
    let r24 = pre.row("r24").unwrap();
    assert_eq!(r24.status, "eligible", "{:?}", r24.reasons);
    let ids: Vec<&str> = pre.events_of(r24).iter().map(|e| e.event_id.as_str()).collect();
    assert_eq!(ids, ["lab0", "ct0", "report0_v1"]);
    let ct = &r24.slots[0];
    assert_eq!(ct.image_id.as_deref(), Some("CT_tp0"));
    assert_eq!(ct.roi, "center_fallback", "the baseline lesion was segmented after follow-up");
    assert!(ct.annotations.is_empty());
    assert_eq!(r24.target.status, "negative");
    assert_eq!(r24.target.event_id.as_deref(), Some("response1"));
    let late = pre.row("r2200").unwrap();
    assert_eq!(late.slots[0].image_id.as_deref(), Some("CT_tp1"), "the newest eligible CT fills the slot");
    assert_eq!(late.slots[0].roi, "center_fallback");
    assert_eq!(late.target.status, "censored", "no outcome in the window after the follow-up");
    assert_ne!(r24.fingerprint, late.fingerprint);
}

#[test]
fn s3_2_fingerprints_identify_the_task_not_its_spelling() {
    let dir = tempfile::tempdir().unwrap();
    let (task, _) = worked_task(dir.path());
    let reparsed = TaskManifest::from_json(&task.to_json()).unwrap();
    assert_eq!(reparsed.task_fingerprint(), task.task_fingerprint());
    assert_eq!(reparsed.manifest_fingerprint(), task.manifest_fingerprint());
    let mut sparse = task.to_json();
    sparse["policy"] = json!({}); // every default left implicit
    assert_eq!(TaskManifest::from_json(&sparse).unwrap().task_fingerprint(), task.task_fingerprint());
    let mut other = task.clone();
    other.policy.context_us = Some(24 * HOUR);
    assert_ne!(other.task_fingerprint(), task.task_fingerprint());
    let saved = dir.path().join("task.json");
    task.save(&saved).unwrap();
    let loaded = TaskManifest::load(&saved).unwrap();
    assert!(loaded.validate().is_empty());
    let mut tampered: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&saved).unwrap()).unwrap();
    tampered["rows"][0]["cutoff_us"] = json!(1);
    let codes: Vec<String> =
        TaskManifest::from_json(&tampered).unwrap().validate().into_iter().map(|f| f.code).collect();
    assert_eq!(codes, ["T103"]);
}

fn add_event(path: &Path, event: Event, links: Vec<Link>) {
    let mut w = amend(path, None).unwrap();
    w.add_event(event).unwrap();
    for l in links {
        w.add_link(l).unwrap();
    }
    w.commit(true).unwrap();
}

fn observation(id: &str, record: &str, effective: i64, available: Option<i64>) -> Event {
    Event {
        event_id: id.into(),
        record_id: record.into(),
        kind: "observation".into(),
        temporal_type: "point".into(),
        effective_start: Some(Bounds::exact(effective)),
        available: available.map(Bounds::exact),
        status: "final".into(),
        code_system: Some("http://loinc.org".into()),
        code: Some("2160-0".into()),
        value_num: Some(1.0),
        unit: Some("mg/dL".into()),
        ..Default::default()
    }
}

#[test]
fn s2_a_changed_source_fails_its_pin_and_a_repin_keeps_the_view() {
    let dir = tempfile::tempdir().unwrap();
    let (mut task, path) = worked_task(dir.path());
    let first = preflight(&task, None, false).unwrap();
    let before = first.row("r24").unwrap().clone();
    let before_events: Vec<Event> = first.events_of(&before).into_iter().cloned().collect();
    // A definitely post-cutoff addition.
    add_event(&path, observation("lab_late", "lab_late", 5000 * HOUR, Some(5001 * HOUR)), Vec::new());
    let stale = preflight(&task, None, false).unwrap();
    assert!(stale.findings.iter().any(|f| f.code == "T302"), "{:?}", stale.findings);
    assert_eq!(stale.row("r24").unwrap().status, "error");
    task.subjects[0].sources[0] = pinned(&path, "s0", None);
    let repinned = preflight(&task, None, false).unwrap();
    assert!(repinned.ok(), "{:?}", repinned.findings);
    let after = repinned.row("r24").unwrap();
    let after_events: Vec<Event> = repinned.events_of(after).into_iter().cloned().collect();
    assert_eq!(after_events, before_events, "a post-cutoff addition does not change the strict input");
    assert_eq!(after.slots, before.slots);
    assert_ne!(after.fingerprint, before.fingerprint, "the row pins the new version");
}

#[test]
fn s4_an_uncertain_revision_makes_its_rows_uncertifiable() {
    let dir = tempfile::tempdir().unwrap();
    let (mut task, path) = worked_task(dir.path());
    add_event(
        &path,
        observation("lab0_v2", "lab0", -48 * HOUR, None),
        vec![Link::new(("event", "lab0_v2"), "supersedes", ("event", "lab0"))],
    );
    task.subjects[0].sources[0] = pinned(&path, "s0", None);
    let pre = preflight(&task, None, false).unwrap();
    let r24 = pre.row("r24").unwrap();
    assert_eq!(r24.status, "uncertifiable");
    assert_eq!(r24.reasons, ["uncertain_revision:lab0"]);
    task.policy.selection = "latest_provable".into();
    let r24 = preflight(&task, None, false).unwrap().row("r24").unwrap().clone();
    assert_eq!(r24.status, "eligible");
    assert_eq!(r24.selection.unwrap().status, "provable", "never claimed as the source's newest version");
}

#[test]
fn s2_edited_clinical_bytes_fail_the_pin_under_an_unchanged_root() {
    let dir = tempfile::tempdir().unwrap();
    let (task, path) = worked_task(dir.path());
    let file = hdf5::File::open_rw(&path).unwrap();
    let ds = file.dataset("clinical/documents/text/data").unwrap();
    let mut bytes: Vec<u8> = ds.read_raw().unwrap();
    bytes[0] = b'X';
    ds.write_raw(&bytes).unwrap();
    drop(ds);
    drop(file);
    let pre = preflight(&task, None, false).unwrap();
    let t302: Vec<&str> = pre.findings.iter().filter(|f| f.code == "T302").map(|f| f.message.as_str()).collect();
    assert!(t302.iter().any(|m| m.contains("clinical/documents/text/data")), "{t302:?}");
}

fn fragment(path: &Path, subject: &str, clock: &str, lab_value: f64) {
    let ct = NdArray::from_vec(&[8, 16, 16], vec![0i16; 8 * 16 * 16]).unwrap();
    let mut w = create(path, None, Some(subject), "balanced", &[]).unwrap();
    w.add_grid("ct", &[8, 16, 16], &[2.0, 0.8, 0.8], GridOptions::default()).unwrap();
    w.add_image("CT", &ct, "ct", "CT", ImageOptions::default()).unwrap();
    w.set_clock(Clock::relative(
        clock,
        "acquisition start of the baseline CT (timepoint tp0), on the subject's study timeline",
    ))
    .unwrap();
    let mut lab = observation("lab0", "lab0", -48 * HOUR, Some(-47 * HOUR));
    lab.code_version = Some("2.77".into());
    lab.value_num = Some(lab_value);
    lab.prov = None;
    w.add_event(lab).unwrap();
    w.commit(true).unwrap();
}

#[test]
fn s4_fragments_reconcile_through_collection_members() {
    let dir = tempfile::tempdir().unwrap();
    let (mut task, worked) = worked_task(dir.path());
    // A second fragment of the same subject holding a copy of lab0 --- with
    // the worked example's own values, but no `prov`: a different record.
    let other = dir.path().join("other.medh5");
    fragment(&other, "subj-clinical-01", "subject-clock-01", 1.1);
    let shard = dir.path().join("subject.medh5c");
    pack(&[worked.as_path(), other.as_path()], &shard, Some(&["worked".to_string(), "other".to_string()])).unwrap();
    task.subjects[0].sources = vec![pinned(&shard, "s0", Some("worked")), pinned(&shard, "s1", Some("other"))];
    let pre = preflight(&task, None, false).unwrap();
    let conflicts: Vec<&str> = pre.findings.iter().map(|f| f.code.as_str()).collect();
    assert!(conflicts.contains(&"T305"), "the copies differ (prov) and must not be merged: {:?}", pre.findings);

    // An identical copy reconciles once the manifest records it.
    let twin = dir.path().join("twin.medh5");
    std::fs::copy(&worked, &twin).unwrap();
    let mut w = amend(&twin, None).unwrap();
    w.identity(medh5::json::object()).unwrap();
    w.commit(true).unwrap();
    let shard2 = dir.path().join("subject2.medh5c");
    pack(&[worked.as_path(), twin.as_path()], &shard2, Some(&["worked".to_string(), "twin".to_string()])).unwrap();
    task.subjects[0].sources = vec![pinned(&shard2, "s0", Some("worked")), pinned(&shard2, "s1", Some("twin"))];
    let unreconciled = preflight(&task, None, false).unwrap();
    assert!(unreconciled.findings.iter().all(|f| f.code == "T305"), "{:?}", unreconciled.findings);
    task.subjects[0].reconciled = reconcile(&task.subjects[0], None).unwrap();
    assert_eq!(task.subjects[0].reconciled.len(), 8, "every event is in both fragments");
    let pre = preflight(&task, None, false).unwrap();
    assert!(pre.ok(), "{:?}", pre.findings);
    let r24 = pre.row("r24").unwrap();
    assert_eq!(pre.events_of(r24).len(), 3);
    let sources = &pre.subject_of(r24).unwrap().sources;
    assert_eq!(sources[1].sample_key.as_deref(), Some("twin"), "the member the locator names");

    // Another clock refuses the join.
    let foreign = dir.path().join("foreign.medh5");
    fragment(&foreign, "subj-clinical-01", "another-clock", 1.1);
    task.subjects[0].sources = vec![pinned(&worked, "s0", None), pinned(&foreign, "s1", None)];
    task.subjects[0].reconciled.clear();
    let pre = preflight(&task, None, false).unwrap();
    assert!(pre.findings.iter().any(|f| f.code == "T304"), "{:?}", pre.findings);
}

#[test]
fn s4_a_streamed_preflight_is_the_document_byte_for_byte() {
    let dir = tempfile::tempdir().unwrap();
    let (mut task, _) = worked_task(dir.path());
    let other = dir.path().join("other.medh5");
    fragment(&other, "subj-02", "clock-02", 1.0);
    task.subjects.push(Subject {
        subject_id: "P-02".into(),
        clock_id: Some("clock-02".into()),
        partition: Some("test".into()),
        sources: vec![pinned(&other, "s1", None)],
        reconciled: Vec::new(),
    });
    // Rows out of subject order: the document keeps the manifest's.
    let row = |id: &str, subject: &str, hours: i64| Row {
        row_id: id.into(),
        subject_id: subject.into(),
        cutoff_us: hours * HOUR,
    };
    task.rows.insert(0, row("q0", "P-02", 0));
    task.rows.push(row("q24", "P-02", 24));
    let mut unfit = task.clone();
    unfit.rows.push(row("ghost", "P-99", 0));
    for (task, fit) in [(task, true), (unfit, false)] {
        let whole = preflight(&task, None, false).unwrap();
        assert_eq!(whole.ok(), fit, "{:?}", whole.findings);
        assert_eq!(whole.subjects.len(), if fit { 2 } else { 0 });
        assert_eq!(whole.rows[0].row_id, "q0");
        let mut streamed = Vec::new();
        assert_eq!(write_preflight(&task, None, false, &mut streamed).unwrap(), whole.ok());
        assert_eq!(String::from_utf8(streamed).unwrap(), pretty(&whole.to_json()));
    }
}

#[test]
fn s3_3_splits_are_by_subject_and_never_share_a_sample() {
    let dir = tempfile::tempdir().unwrap();
    let (mut task, path) = worked_task(dir.path());
    let mut twin = task.subjects[0].clone();
    twin.subject_id = "P-02".into();
    twin.partition = Some("test".into());
    twin.sources[0].source_id = "s9".into();
    task.subjects.push(twin);
    let codes: Vec<String> = task.validate().into_iter().map(|f| f.code).collect();
    assert!(codes.contains(&"T203".to_string()), "{codes:?}");
    task.subjects.pop();
    task.subjects[0].partition = None;
    let codes: Vec<String> = task.validate().into_iter().map(|f| f.code).collect();
    assert_eq!(codes, ["T202"]);
    task.subjects[0].partition = Some("holdout".into());
    assert_eq!(task.validate()[0].code, "T202");
    let _ = path;
}

fn feature(value: f32) -> NdArray {
    NdArray::from_vec(&[4], vec![value; 4]).unwrap()
}

#[test]
fn s8_caches_are_rejected_when_stale_corrupt_or_inadmissible() {
    let dir = tempfile::tempdir().unwrap();
    let (task, path) = worked_task(dir.path());
    let pre = preflight(&task, None, false).unwrap();
    // What a cache is checked against, kept a subject at a time.
    let admitted = Admitted::preflight(&task, None, false).unwrap();
    assert_eq!(admitted, Admitted::of(&pre));
    assert_eq!(
        admitted.row("r24").unwrap().1,
        pre.events_of(pre.row("r24").unwrap()).iter().map(|e| e.event_id.as_str()).collect::<Vec<_>>()
    );
    assert!(admitted.row("nope").is_none());
    let source = task.subjects[0].sources[0].clone();
    let header = |level: &str| CacheHeader {
        level: level.into(),
        encoder: json!({"name": "fixture-hash", "revision": "sha256:00", "tokenizer": null}),
        preprocessing: json!({"lowercase": true}),
        output: json!({"dtype": "float32", "shape": [4], "pooling": "mean", "chunking": null}),
        task_fingerprint: Some(task.task_fingerprint()),
        selection: Some("strict_prospective".into()),
        fitted_on: Some(CacheHeader::fitted_on(&task, "train")),
    };
    // Event level: one feature per document event.
    let events_path = dir.path().join("events.medh5cache");
    let mut w = CacheWriter::create(&events_path, header("event")).unwrap();
    for (event, document) in [("report0_v1", "report0_text_v1"), ("report0_v2", "report0_text_v2")] {
        w.add(
            CacheEntry {
                entry_id: event_entry_id(&source.content_id, event),
                sources: vec![source.clone()],
                event_id: Some(event.into()),
                document_id: Some(document.into()),
                row_id: None,
                cutoff_us: None,
                event_versions: None,
                digest: String::new(),
            },
            &feature(0.5),
        )
        .unwrap();
    }
    w.commit().unwrap();
    assert!(validate_cache(&events_path, None, Some(&task), Some(&admitted)).unwrap().ok());
    let cache = FeatureCache::open(&events_path).unwrap();
    let entry = cache.event_entry(&source.content_id, "report0_v1").unwrap();
    assert_eq!(cache.get(&entry.entry_id).unwrap(), feature(0.5));
    drop(cache);

    // Patient level: pinned to the row's cutoff and selection --- and a
    // whole-history feature that is not.
    let patient_path = dir.path().join("patient.medh5cache");
    let mut w = CacheWriter::create(&patient_path, header("patient")).unwrap();
    let r24 = pre.row("r24").unwrap();
    for (entry_id, versions) in [
        ("r24", pre.events_of(r24).iter().map(|e| e.event_id.clone()).collect::<Vec<_>>()),
        ("r24_whole_history", vec!["lab0".into(), "ct0".into(), "report0_v2".into(), "ct1".into()]),
    ] {
        w.add(
            CacheEntry {
                entry_id: entry_id.into(),
                sources: vec![source.clone()],
                event_id: None,
                document_id: None,
                row_id: Some("r24".into()),
                cutoff_us: Some(24 * HOUR),
                event_versions: Some(versions),
                digest: String::new(),
            },
            &feature(1.0),
        )
        .unwrap();
    }
    w.commit().unwrap();
    let report = validate_cache(&patient_path, None, Some(&task), Some(&admitted)).unwrap();
    let inadmissible: Vec<&str> =
        report.findings.iter().filter(|f| f.code == "T406").map(|f| f.location.as_str()).collect();
    assert_eq!(inadmissible, ["r24_whole_history"]);

    // Fitted on the wrong partition.
    let mut other = task.clone();
    other.subjects[0].partition = Some("test".into());
    let report = validate_cache(&events_path, None, Some(&other), None).unwrap();
    assert!(report.findings.iter().any(|f| f.code == "T405"), "{:?}", report.findings);

    // Corrupt: a payload byte changes; its own checksum catches it.
    let file = hdf5::File::open_rw(&events_path).unwrap();
    let name = format!("entries/{}", event_entry_id(&source.content_id, "report0_v2"));
    file.dataset(&name).unwrap().write_raw(&[9.0f32; 4]).unwrap();
    drop(file);
    let report = validate_cache(&events_path, None, Some(&task), Some(&admitted)).unwrap();
    assert_eq!(report.corrupt(), [event_entry_id(&source.content_id, "report0_v2")]);
    assert!(report.stale().is_empty());

    // Stale: the source changes; every entry reading it is rejected.
    add_event(&path, observation("lab_late", "lab_late", 5000 * HOUR, Some(5001 * HOUR)), Vec::new());
    let report = validate_cache(&patient_path, None, None, None).unwrap();
    assert_eq!(report.stale(), ["r24", "r24_whole_history"]);
    assert!(report.corrupt().is_empty());
}
