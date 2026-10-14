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

/// `task reconcile` writes over its input, which `save` truncated and
/// rewrote in place: a write cut short lost the manifest.
#[cfg(unix)]
#[test]
fn s3_1_a_saved_manifest_replaces_the_file_whole() {
    use std::os::unix::fs::MetadataExt;
    let dir = tempfile::tempdir().unwrap();
    let (task, _) = worked_task(dir.path());
    let real = dir.path().join("task.json");
    task.save(&real).unwrap();
    let link = dir.path().join("link.json");
    std::os::unix::fs::symlink(&real, &link).unwrap();
    let before = std::fs::metadata(&real).unwrap().ino();
    task.save(&link).unwrap();
    assert!(std::fs::symlink_metadata(&link).unwrap().file_type().is_symlink(), "the link is kept");
    assert_ne!(std::fs::metadata(&real).unwrap().ino(), before, "a new file, never the old one rewritten");
    assert!(TaskManifest::load(&real).unwrap().validate().is_empty());
    let names: Vec<String> =
        std::fs::read_dir(dir.path()).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
    assert!(names.iter().all(|n| !n.contains(".tmp-")), "{names:?}");
}

#[test]
fn s3_4_a_value_outside_a_vocabulary_fails_the_schema_alone() {
    let dir = tempfile::tempdir().unwrap();
    let (mut task, _) = worked_task(dir.path());
    task.slots[0].roi = "middle".into();
    task.target.as_mut().unwrap().censoring = "drop".into();
    let codes: std::collections::BTreeSet<String> = task.validate().into_iter().map(|f| f.code).collect();
    assert_eq!(codes, ["T101".to_string()].into(), "T101, not T101 and T102 for one value");
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

fn cache_header(task: &TaskManifest, level: &str) -> CacheHeader {
    CacheHeader {
        level: level.into(),
        encoder: json!({"name": "fixture-hash", "revision": "sha256:00", "tokenizer": null}),
        preprocessing: json!({"lowercase": true}),
        output: json!({"dtype": "float32", "shape": [4], "pooling": "mean", "chunking": null}),
        task_fingerprint: Some(task.task_fingerprint()),
        selection: Some("strict_prospective".into()),
        fitted_on: Some(CacheHeader::fitted_on(task, "train")),
    }
}

fn event_entry(source: &SourceRef, event: &str, document: &str) -> CacheEntry {
    CacheEntry {
        entry_id: event_entry_id(&source.content_id, event),
        sources: vec![source.clone()],
        event_id: Some(event.into()),
        document_id: Some(document.into()),
        row_id: None,
        cutoff_us: None,
        event_versions: None,
        row_fingerprint: None,
        digest: String::new(),
    }
}

/// An event-level cache of the worked example's first report; its entry id.
fn event_cache(path: &Path, task: &TaskManifest) -> String {
    let source = &task.subjects[0].sources[0];
    let mut w = CacheWriter::create(path, cache_header(task, "event")).unwrap();
    let entry = w.add(event_entry(source, "report0_v1", "report0_text_v1"), &feature(0.5)).unwrap();
    w.commit().unwrap();
    entry.entry_id
}

#[test]
fn s8_a_payload_is_judged_by_its_declaration_before_it_is_read() {
    let dir = tempfile::tempdir().unwrap();
    let (task, _) = worked_task(dir.path());
    let path = dir.path().join("events.medh5cache");
    let id = event_cache(&path, &task);
    // A hostile payload: declared far larger than any memory, and costing
    // nothing on disk, since no chunk of it is written.
    let file = hdf5::File::open_rw(&path).unwrap();
    let entries = file.group("entries").unwrap();
    entries.unlink(&id).unwrap();
    entries.new_dataset::<f32>().chunk([1usize << 16]).shape([1usize << 40]).create(id.as_str()).unwrap();
    drop(entries);
    drop(file);
    let cache = FeatureCache::open(&path).unwrap();
    let err = cache.get(&id).unwrap_err();
    assert_eq!(err.code(), Some("T402"), "refused as the entry's fault, not as an allocation failure: {err}");
    assert!(err.message().contains("[1099511627776]"), "{err}");
    drop(cache);
    assert_eq!(validate_cache(&path, None, None, None).unwrap().corrupt(), [id]);
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
    let header = |level: &str| cache_header(&task, level);
    // Event level: one feature per document event.
    let events_path = dir.path().join("events.medh5cache");
    let mut w = CacheWriter::create(&events_path, header("event")).unwrap();
    for (event, document) in [("report0_v1", "report0_text_v1"), ("report0_v2", "report0_text_v2")] {
        w.add(event_entry(&source, event, document), &feature(0.5)).unwrap();
    }
    w.commit().unwrap();
    assert!(validate_cache(&events_path, None, Some(&task), Some(&admitted)).unwrap().ok());
    let cache = FeatureCache::open(&events_path).unwrap();
    let entry = cache.event_entry(&source.content_id, "report0_v1").unwrap();
    assert_eq!(cache.get(&entry.entry_id).unwrap(), feature(0.5));
    drop(cache);

    // Patient level: pinned to the row's cutoff and selection --- and, in a
    // cache of its own, since a cache holds one feature per row, a
    // whole-history feature that is not.
    let patient_path = dir.path().join("patient.medh5cache");
    let whole_path = dir.path().join("whole.medh5cache");
    let r24 = pre.row("r24").unwrap();
    for (path, entry_id, versions) in [
        (&patient_path, "r24", pre.events_of(r24).iter().map(|e| e.event_id.clone()).collect::<Vec<_>>()),
        (&whole_path, "r24_whole_history", vec!["lab0".into(), "ct0".into(), "report0_v2".into(), "ct1".into()]),
    ] {
        let mut w = CacheWriter::create(path, header("patient")).unwrap();
        w.add(
            CacheEntry {
                entry_id: entry_id.into(),
                sources: vec![source.clone()],
                event_id: None,
                document_id: None,
                row_id: Some("r24".into()),
                cutoff_us: Some(24 * HOUR),
                event_versions: Some(versions),
                row_fingerprint: Some(r24.fingerprint.clone()),
                digest: String::new(),
            },
            &feature(1.0),
        )
        .unwrap();
        w.commit().unwrap();
    }
    assert!(validate_cache(&patient_path, None, Some(&task), Some(&admitted)).unwrap().ok());
    let report = validate_cache(&whole_path, None, Some(&task), Some(&admitted)).unwrap();
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
    assert_eq!(report.stale(), ["r24"]);
    assert!(report.corrupt().is_empty());
}

#[test]
fn s8_an_entry_of_a_row_in_error_is_stale_not_inadmissible() {
    let dir = tempfile::tempdir().unwrap();
    let (task, path) = worked_task(dir.path());
    let pre = preflight(&task, None, false).unwrap();
    let r24 = pre.row("r24").unwrap();
    let cache = dir.path().join("patient.medh5cache");
    let mut w = CacheWriter::create(&cache, cache_header(&task, "patient")).unwrap();
    let entry = CacheEntry {
        entry_id: "r24".into(),
        sources: task.subjects[0].sources.clone(),
        event_id: None,
        document_id: None,
        row_id: Some("r24".into()),
        cutoff_us: Some(r24.cutoff_us),
        event_versions: Some(pre.events_of(r24).iter().map(|e| e.event_id.clone()).collect()),
        row_fingerprint: Some(r24.fingerprint.clone()),
        digest: String::new(),
    };
    w.add(entry, &feature(1.0)).unwrap();
    w.commit().unwrap();
    // The source changes: its rows are in error, and admit nothing to compare
    // an entry with.
    add_event(&path, observation("lab_late", "lab_late", 5000 * HOUR, Some(5001 * HOUR)), Vec::new());
    let admitted = Admitted::preflight(&task, None, false).unwrap();
    let report = validate_cache(&cache, None, Some(&task), Some(&admitted)).unwrap();
    let found: Vec<(&str, &str)> = report.findings.iter().map(|f| (f.code.as_str(), f.location.as_str())).collect();
    assert_eq!(found, [("T403", "r24"), ("T404", "r24")], "{:?}", report.findings);
    assert!(report.findings[1].message.contains("in error"), "{:?}", report.findings[1]);
    // A manifest with findings leaves every row in error, unclaimed by any
    // subject: one T404 again, not a source the row's subject "does not pin".
    let mut invalid = task.clone();
    invalid.subjects[0].partition = None;
    let admitted = Admitted::preflight(&invalid, None, false).unwrap();
    let report = validate_cache(&cache, None, Some(&invalid), Some(&admitted)).unwrap();
    let rows: Vec<&str> = report.findings.iter().filter(|f| f.location == "r24").map(|f| f.code.as_str()).collect();
    assert_eq!(rows, ["T403", "T404"], "{:?}", report.findings);
}

#[test]
fn s7_1_the_writer_refuses_what_its_reader_rejects() {
    let dir = tempfile::tempdir().unwrap();
    let (task, _) = worked_task(dir.path());
    let source = task.subjects[0].sources[0].clone();
    let path = dir.path().join("events.medh5cache");
    let mut w = CacheWriter::create(&path, cache_header(&task, "event")).unwrap();
    // Each of these was written, committed and checksummed, and the cache
    // then refused by its own reader (T401).
    let mut unaddressed = event_entry(&source, "report0_v1", "report0_text_v1");
    unaddressed.sources[0].uri = String::new();
    let mut unpinned = event_entry(&source, "report0_v1", "report0_text_v1");
    unpinned.sources[0].content_id = "not-a-digest".into();
    for entry in [unaddressed, unpinned] {
        let err = w.add(entry, &feature(0.5)).unwrap_err();
        assert_eq!(err.code(), Some("T401"), "{err}");
        assert!(err.message().contains("sources/0/"), "{err}");
    }
    // One feature per event version: a second under another id was served in
    // the first's place, whose checksum was then never compared.
    w.add(event_entry(&source, "report0_v1", "report0_text_v1"), &feature(0.5)).unwrap();
    let mut again = event_entry(&source, "report0_v1", "report0_text_v1");
    again.entry_id = "another".into();
    assert!(w.add(again, &feature(0.5)).unwrap_err().message().contains("one feature per event version"));
    w.commit().unwrap();
    assert!(validate_cache(&path, None, None, None).unwrap().ok());
}

#[test]
fn s8_a_path_naming_no_cache_is_refused_not_corrupt() {
    let dir = tempfile::tempdir().unwrap();
    let err = validate_cache(&dir.path().join("nope.medh5cache"), None, None, None).unwrap_err();
    assert!(err.message().contains("no such file"), "{err}");
    let garbage = dir.path().join("garbage.medh5cache");
    std::fs::write(&garbage, b"not a cache").unwrap();
    assert_eq!(validate_cache(&garbage, None, None, None).unwrap().corrupt(), ["manifest"]);
}

#[test]
fn s7_1_a_manifest_with_two_entries_for_one_lookup_is_corrupt() {
    let dir = tempfile::tempdir().unwrap();
    let (task, _) = worked_task(dir.path());
    let path = dir.path().join("events.medh5cache");
    let id = event_cache(&path, &task);
    // A second entry with the first's id, or its event version, and a wrong
    // checksum: lookups served one, and the other was never compared.
    // Rewritten and re-checksummed, as a writer would.
    let rewrite = |change: &dyn Fn(&mut serde_json::Value)| {
        use medh5::h5::{attrs, data};
        let file = hdf5::File::open_rw(&path).unwrap();
        let text = data::read_scalar_string(&file.dataset("manifest").unwrap()).unwrap();
        let mut doc: serde_json::Value = serde_json::from_str(&text).unwrap();
        change(&mut doc);
        let text = medh5::json::canonical(&doc);
        file.unlink("manifest").unwrap();
        data::create_scalar_string(&file, "manifest", &text).unwrap();
        let digest = attrs::AttrValue::Str(medh5::companion::sha256(text.as_bytes()));
        attrs::write(&file, "manifest_digest", &digest).unwrap();
    };
    let original = {
        let cache = FeatureCache::open(&path).unwrap();
        cache.entries[0].to_json()
    };
    for (twin, why) in [("id", "twice"), ("event", "two entries for event 'report0_v1'")] {
        let mut copy = original.clone();
        copy["digest"] = json!(format!("sha256:{}", "f".repeat(64)));
        if twin == "event" {
            copy["entry_id"] = json!("another");
        }
        rewrite(&|doc| doc["entries"] = json!([original.clone(), copy.clone()]));
        let err = FeatureCache::open(&path).unwrap_err();
        assert_eq!(err.code(), Some("T401"), "{err}");
        assert!(err.message().contains(why), "{err}");
        let report = validate_cache(&path, None, None, None).unwrap();
        assert_eq!(report.corrupt(), ["manifest"], "{:?}", report.findings);
    }
    let _ = id;
}

/// The probe of task-cache-1 Appendix A: a subject whose first reference pins
/// one version but names a file that is another, holding a secret identity,
/// clock, lab value and document; and whose second, pinned correctly, holds
/// the same event id at another value.
fn hostile_task(dir: &Path) -> (TaskManifest, String) {
    let secret = dir.join("secret.medh5");
    fragment(&secret, "MRN-0042-SECRET", "clk-secret", 42.0);
    let mut w = amend(&secret, None).unwrap();
    w.add_event(Event {
        event_id: "ev-doc-secret".into(),
        record_id: "ev-doc-secret".into(),
        kind: "document".into(),
        temporal_type: "point".into(),
        effective_start: Some(Bounds::exact(-HOUR)),
        available: Some(Bounds::exact(-HOUR)),
        status: "final".into(),
        ..Default::default()
    })
    .unwrap();
    w.add_document(medh5::clinical::model::Document::text("doc-secret", "the secret note")).unwrap();
    w.add_link(Link::new(("event", "ev-doc-secret"), "describes", ("document", "doc-secret"))).unwrap();
    w.commit(true).unwrap();
    let good = dir.join("good.medh5");
    fragment(&good, "S-01", "clk-01", 7.0);
    let stored = open_sample(&secret).unwrap().content_id().unwrap().unwrap();
    let mut hostile = pinned(&secret, "s-hostile", None);
    hostile.content_id = pinned(&good, "x", None).content_id;
    let mut task = TaskManifest::new("hostile", "1", "org.example.trial");
    task.subjects.push(Subject {
        subject_id: "S-01".into(),
        clock_id: Some("clk-01".into()),
        partition: None,
        sources: vec![hostile, pinned(&good, "s-good", None)],
        reconciled: Vec::new(),
    });
    task.rows.push(Row { row_id: "r0".into(), subject_id: "S-01".into(), cutoff_us: 0 });
    (task, stored)
}

#[test]
fn s2_a_source_off_its_pin_is_read_no_further() {
    let dir = tempfile::tempdir().unwrap();
    let (task, stored) = hostile_task(dir.path());
    let pre = preflight(&task, None, false).unwrap();
    let codes: Vec<(&str, &str)> = pre.findings.iter().map(|f| (f.code.as_str(), f.location.as_str())).collect();
    assert_eq!(codes, [("T302", "s-hostile")], "no T303, T304 or T305 from a file off its pin");
    let mut streamed = Vec::new();
    write_preflight(&task, None, false, &mut streamed).unwrap();
    let text = String::from_utf8(streamed).unwrap();
    for secret in ["MRN-0042-SECRET", "clk-secret", "doc-secret", "42.0", stored.as_str()] {
        assert!(!text.contains(secret), "{secret} reported from a source off its pin");
    }
    let history = pre.subject_of(pre.row("r0").unwrap()).unwrap();
    let values: Vec<Option<f64>> = history.events.iter().map(|e| e.value_num).collect();
    assert_eq!(values, [Some(7.0)], "the pinned source is read as before");
}

#[test]
fn s2_reconcile_refuses_a_source_off_its_pin() {
    let dir = tempfile::tempdir().unwrap();
    let (task, stored) = hostile_task(dir.path());
    let err = reconcile(&task.subjects[0], None).unwrap_err();
    assert_eq!(err.code(), Some("T302"), "{err}");
    assert!(!err.to_string().contains(&stored), "{err}");
}
