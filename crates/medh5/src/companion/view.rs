//! Row views: what a task admits for each row (task-and-cache contract §4--§6).
//!
//! [`preflight`] takes the subjects one at a time.  For each it opens every
//! source once, checks every pin, reconciles the fragments --- one clock, one
//! content per duplicated event id --- and prepares the merged history for
//! selection once ([`Prepared`]); then, per row, it selects at the cutoff,
//! fills the modality slots from eligible images only, and labels the target
//! from the full history.  The subject's files are closed before the next
//! subject's are opened, so a cohort of any size holds one subject's handles
//! at a time.  The result says for each row whether it is **eligible**,
//! **uncertifiable** (a later revision's availability is unknown or
//! straddles the cutoff), **excluded** (a missing required slot, a censored
//! or prevalent target) or in **error** (its sources or its manifest are
//! wrong), and why.
//!
//! Rows do not copy their inputs: each indexes its subject's merged history
//! ([`SubjectHistory`]), which every row of the subject shares.
//!
//! Metadata first: nothing here reads a voxel or uses a report.  Text is read
//! only to verify it --- against its digest, and as UTF-8, in bounded slabs
//! --- and to compare a document two fragments both hold.  The frontends read
//! only what a row admits, when they build the batch.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use serde_json::{json, Value};

use super::source::SourceRef;
use super::task::{Row, Slot, Subject, TaskManifest};
use super::{fingerprint, Finding};
use crate::clinical::model::{Bounds, Event, Link};
use crate::clinical::select::{names, Prepared, Selection};
use crate::clinical::Clinical;
use crate::json::{pretty_at, repr_str};
use crate::sample::Sample;
use crate::Result;

/// The digest of one event version's logical record: what duplicates across
/// fragments must agree on.
pub fn event_digest(event: &Event) -> String {
    fingerprint(&event.to_json())
}

/// One source of a subject, opened.
#[derive(Debug)]
pub struct Fragment {
    pub source: SourceRef,
    pub sample: Sample,
    pub clinical: Option<Arc<Clinical>>,
}

/// One subject's history, merged across its fragments: what its rows index.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SubjectHistory {
    pub subject_id: String,
    pub partition: Option<String>,
    /// The subject's sources that opened, in manifest order: what a
    /// `fragment` names.
    pub sources: Vec<SourceRef>,
    /// Event versions, unique by id, in the order the fragments hold them.
    pub events: Vec<Event>,
    /// The fragment each version was first found in.
    pub event_fragments: Vec<usize>,
    /// Every fragment's links, each with its fragment: what a selection's
    /// `links` index.
    pub links: Vec<(usize, Link)>,
    /// `(event, fragment, document_id)`: the documents `document` versions
    /// own through their structural `describes` link, by event index.
    pub documents: Vec<(usize, usize, String)>,
}

impl SubjectHistory {
    /// The links as selection takes them.
    pub fn link_refs(&self) -> Vec<(usize, &Link)> {
        self.links.iter().map(|(f, l)| (*f, l)).collect()
    }

    pub fn to_json(&self) -> Value {
        json!({
            "subject_id": self.subject_id,
            "partition": self.partition,
            "sources": self.sources.iter().map(SourceRef::to_json).collect::<Vec<_>>(),
            "events": self.events.iter().map(Event::to_json).collect::<Vec<_>>(),
            "event_fragments": self.event_fragments,
            "links": self.links.iter().map(|(f, l)| json!({"fragment": f, "link": l.to_json()})).collect::<Vec<_>>(),
            "documents": self.documents.iter().map(|(e, f, d)| json!([e, f, d])).collect::<Vec<_>>(),
        })
    }
}

/// A subject's sources, opened and checked, and their merged history.
#[derive(Debug, Default)]
pub struct History {
    pub fragments: Vec<Fragment>,
    pub merged: SubjectHistory,
    pub findings: Vec<Finding>,
}

/// Open, check and reconcile one subject's sources.
pub fn load_subject(manifest: &TaskManifest, subject: &Subject, base: Option<&Path>, deep: bool) -> Result<History> {
    let mut history = History::default();
    history.merged.subject_id = subject.subject_id.clone();
    history.merged.partition = subject.partition.clone();
    let at = |src: &SourceRef| if src.source_id.is_empty() { src.locator() } else { src.source_id.clone() };
    for source in &subject.sources {
        let sample = match source.open(base) {
            Ok(s) => s,
            Err(e) => {
                history.findings.push(Finding::new(
                    "T301",
                    at(source),
                    format!("{} does not open: {}", source.locator(), e),
                ));
                continue;
            }
        };
        history.findings.extend(source.check(&sample, deep)?);
        if let Some(local) = &source.local_subject_id {
            let found = sample.identity()?.subject_id.clone();
            if &found != local {
                history.findings.push(Finding::new(
                    "T303",
                    at(source),
                    format!(
                        "{} is subject {} in its own identity, not {} as the manifest maps it to {}",
                        source.locator(),
                        repr_str(&found),
                        repr_str(local),
                        repr_str(&subject.subject_id)
                    ),
                ));
            }
        }
        if sample.support()? != crate::version::Support::Full {
            history.findings.push(Finding::new(
                "T306",
                at(source),
                format!(
                    "{} is MEDH5 {}, read here only as a projection: it cannot supply certified inputs",
                    source.locator(),
                    sample.version()?
                ),
            ));
        }
        let clinical = if sample.profiles()?.contains(crate::clinical::PROFILE) {
            // Validated, then read from what the validator read: once.  A
            // table that cannot be read at all --- a damaged chunk --- is the
            // source's finding, not the whole preflight's failure.
            match read_clinical(&sample, &source.locator()) {
                Ok((problems, clinical)) => {
                    if let Some(first) = problems.first() {
                        history.findings.push(Finding::new(
                            "T306",
                            at(source),
                            format!(
                                "{}: its clinical tables are invalid ({} {}: {})",
                                source.locator(),
                                first.code,
                                first.location,
                                first.message
                            ),
                        ));
                    }
                    clinical
                }
                Err(e) => {
                    history.findings.push(Finding::new(
                        "T306",
                        at(source),
                        format!("{}: its clinical tables cannot be read: {e}", source.locator()),
                    ));
                    None
                }
            }
        } else {
            None
        };
        history.fragments.push(Fragment { source: source.clone(), sample, clinical });
    }
    history.merged.sources = history.fragments.iter().map(|f| f.source.clone()).collect();
    // One clock (1.1 §3): fragments that share a subject share its origin.
    let clocks: BTreeSet<String> = history
        .fragments
        .iter()
        .filter_map(|f| f.clinical.as_ref().map(|c| crate::json::canonical(&c.descriptor.clock.to_json())))
        .collect();
    if clocks.len() > 1 {
        history.findings.push(Finding::new(
            "T304",
            &subject.subject_id,
            format!("the subject's fragments declare {} different clocks; a temporal join needs one", clocks.len()),
        ));
    }
    if let Some(expected) = &subject.clock_id {
        for f in &history.fragments {
            if let Some(c) = &f.clinical {
                if &c.descriptor.clock.id != expected {
                    history.findings.push(Finding::new(
                        "T304",
                        at(&f.source),
                        format!(
                            "clock {} is not the subject's clock {}",
                            repr_str(&c.descriptor.clock.id),
                            repr_str(expected)
                        ),
                    ));
                }
            }
        }
    }
    let findings = merge(&mut history, subject)?;
    history.findings.extend(findings);
    let _ = manifest;
    Ok(history)
}

/// A source's clinical errors, and the profile read from the tables the
/// validator read --- when they are sound.
fn read_clinical(sample: &Sample, locator: &str) -> Result<(Vec<crate::validate::Diagnostic>, Option<Arc<Clinical>>)> {
    let (problems, tables) = crate::validate::clinical_checked(&sample.root, locator)?;
    let clinical = match tables {
        Some(t) if t.sound() && t.events.is_some() && t.descriptor.is_some() => {
            let projection = crate::version::is_projection(&sample.version()?);
            Some(Arc::new(Clinical::from_tables(
                &sample.root,
                t.descriptor.clone().expect("checked"),
                t.events.as_ref().expect("checked"),
                t.documents.as_ref(),
                t.links.as_ref(),
                projection,
            )?))
        }
        _ => None,
    };
    Ok((problems, clinical))
}

/// Merge the fragments' events, links and documents into `history.merged`,
/// checking every duplicate --- and only the duplicates --- against each other
/// and against the manifest's record of it (§3.3, T305).
fn merge(history: &mut History, subject: &Subject) -> Result<Vec<Finding>> {
    let mut findings = Vec::new();
    let fragments = &history.fragments;
    let merged = &mut history.merged;
    let at = |f: &Fragment| if f.source.source_id.is_empty() { f.source.locator() } else { f.source.source_id.clone() };
    // Every holder of every event id; the first holder's version is merged.
    let mut holders: HashMap<&str, Vec<(usize, usize)>> = HashMap::new();
    for (i, f) in fragments.iter().enumerate() {
        let Some(c) = &f.clinical else { continue };
        for (k, e) in c.events.iter().enumerate() {
            let found = holders.entry(e.event_id.as_str()).or_default();
            if found.is_empty() {
                merged.events.push(e.clone());
                merged.event_fragments.push(i);
            }
            found.push((i, k));
        }
        merged.links.extend(c.links.iter().map(|l| (i, l.clone())));
    }
    let event = |(f, k): (usize, usize)| &fragments[f].clinical.as_ref().expect("held").events[k];
    let mut duplicated: Vec<(&str, &Vec<(usize, usize)>)> =
        holders.iter().filter(|(_, h)| h.len() > 1).map(|(id, h)| (*id, h)).collect();
    duplicated.sort_unstable();
    for (event_id, held) in duplicated {
        let digest = event_digest(event(held[0]));
        for (f, k) in &held[1..] {
            let other = event_digest(event((*f, *k)));
            if other != digest {
                findings.push(Finding::new(
                    "T305",
                    at(&fragments[*f]),
                    format!("event {} differs between fragments ({digest} vs {other})", repr_str(event_id)),
                ));
            }
        }
        match subject.reconciled.iter().find(|r| r.event_id == event_id) {
            Some(r) if r.digest == digest => {}
            Some(r) => findings.push(Finding::new(
                "T305",
                &subject.subject_id,
                format!("event {} is reconciled at {} but its fragments hold {digest}", repr_str(event_id), r.digest),
            )),
            None => findings.push(Finding::new(
                "T305",
                &subject.subject_id,
                format!(
                    "event {} is in fragments {} but the manifest records no reconciliation for it",
                    repr_str(event_id),
                    held.iter().map(|(f, _)| fragments[*f].source.source_id.as_str()).collect::<Vec<_>>().join(", ")
                ),
            )),
        }
    }
    // Documents several fragments hold must be identical: only those are read.
    let mut documents: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, f) in fragments.iter().enumerate() {
        if let Some(c) = &f.clinical {
            for d in c.documents() {
                documents.entry(d.document_id.as_str()).or_default().push(i);
            }
        }
    }
    for (document_id, held) in documents.iter().filter(|(_, h)| h.len() > 1) {
        let mut digests = Vec::with_capacity(held.len());
        for f in held {
            let c = fragments[*f].clinical.as_ref().expect("held");
            match c.document(document_id) {
                Ok(d) => digests.push((*f, fingerprint(&d.to_json()))),
                Err(e) => findings.push(Finding::new(
                    "T306",
                    at(&fragments[*f]),
                    format!("document {} cannot be read: {e}", repr_str(document_id)),
                )),
            }
        }
        if let Some((_, first)) = digests.first() {
            for (f, digest) in &digests[1..] {
                if digest != first {
                    findings.push(Finding::new(
                        "T305",
                        at(&fragments[*f]),
                        format!("document {} differs between fragments", repr_str(document_id)),
                    ));
                }
            }
        }
    }
    // The documents each version owns, structurally.
    let by_id: HashMap<&str, usize> = merged.events.iter().enumerate().map(|(i, e)| (e.event_id.as_str(), i)).collect();
    let mut owned = Vec::new();
    for (f, l) in &merged.links {
        if l.relation == "describes" && l.source_type == "event" && l.target_type == "document" {
            if let Some(&e) = by_id.get(l.source_id.as_str()) {
                if merged.events[e].kind == "document" {
                    owned.push((e, *f, l.target_id.clone()));
                }
            }
        }
    }
    owned.sort();
    owned.dedup();
    merged.documents = owned;
    Ok(findings)
}

/// The reconciliation records a subject's fragments need (what a manifest
/// writer stores in `subjects[].reconciled`).
pub fn reconcile(subject: &Subject, base: Option<&Path>) -> Result<Vec<super::task::Reconciled>> {
    let mut samples = Vec::new();
    for source in &subject.sources {
        samples.push((source.source_id.clone(), source.open(base)?));
    }
    let mut clinicals = Vec::new();
    for (id, sample) in &samples {
        if let Some(c) = sample.clinical()? {
            clinicals.push((id.clone(), c.clone()));
        }
    }
    // Only an id held twice needs a record; only those are digested.
    let mut holders: BTreeMap<&str, Vec<(usize, usize)>> = BTreeMap::new();
    for (i, (_, c)) in clinicals.iter().enumerate() {
        for (k, e) in c.events.iter().enumerate() {
            holders.entry(e.event_id.as_str()).or_default().push((i, k));
        }
    }
    Ok(holders
        .into_iter()
        .filter(|(_, h)| h.len() > 1)
        .map(|(event_id, held)| {
            let (f, k) = held[0];
            super::task::Reconciled {
                event_id: event_id.to_string(),
                digest: event_digest(&clinicals[f].1.events[k]),
                sources: held.iter().map(|(f, _)| clinicals[*f].0.clone()).collect(),
            }
        })
        .collect())
}

/// How one slot was filled for one row.
#[derive(Debug, Clone, PartialEq)]
pub struct SlotFill {
    pub slot: String,
    /// The fragment (index into the subject's sources) and image filling it.
    pub fragment: Option<usize>,
    pub image_id: Option<String>,
    pub grid_id: Option<String>,
    /// The imaging event attesting the image.
    pub event_id: Option<String>,
    /// The region's centre, in the grid's spatial voxel indices.
    pub center: Option<Vec<i64>>,
    /// `center`, `eligible_instances`, or `center_fallback` when no eligible
    /// instance was on the grid.
    pub roi: String,
    /// Eligible voxel annotations on the image's grid: usable as *inputs*
    /// (a prior mask, an ROI), because selection admits them at the cutoff.
    pub annotations: Vec<String>,
    /// Every voxel annotation on the image's grid, eligible or not: label
    /// *supervision*, which --- like the target --- may come from after the
    /// cutoff and never enters an input.
    pub label_annotations: Vec<String>,
}

impl SlotFill {
    fn empty(slot: &Slot) -> SlotFill {
        SlotFill {
            slot: slot.name.clone(),
            fragment: None,
            image_id: None,
            grid_id: None,
            event_id: None,
            center: None,
            roi: slot.roi.clone(),
            annotations: Vec::new(),
            label_annotations: Vec::new(),
        }
    }

    pub fn available(&self) -> bool {
        self.image_id.is_some()
    }

    pub fn to_json(&self) -> Value {
        json!({
            "slot": self.slot,
            "available": self.available(),
            "fragment": self.fragment,
            "image_id": self.image_id,
            "grid_id": self.grid_id,
            "event_id": self.event_id,
            "center": self.center,
            "roi": self.roi,
            "annotations": self.annotations,
            "label_annotations": self.label_annotations,
        })
    }
}

/// The target of one row.
#[derive(Debug, Clone, PartialEq)]
pub struct TargetLabel {
    /// `positive`, `negative`, `censored`, `prevalent` or `none` (no target).
    pub status: String,
    /// 1.0, 0.0, or `None` when unobserved.
    pub value: Option<f64>,
    pub event_id: Option<String>,
    pub reason: Option<String>,
}

impl TargetLabel {
    fn none() -> TargetLabel {
        TargetLabel { status: "none".into(), value: None, event_id: None, reason: None }
    }

    pub fn observed(&self) -> bool {
        self.value.is_some()
    }

    pub fn to_json(&self) -> Value {
        json!({
            "status": self.status,
            "value": self.value,
            "observed": self.observed(),
            "event_id": self.event_id,
            "reason": self.reason,
        })
    }
}

/// Everything a task admits for one row.
#[derive(Debug, Clone, PartialEq)]
pub struct RowView {
    pub row_id: String,
    pub subject_id: String,
    pub partition: Option<String>,
    pub cutoff_us: i64,
    pub fingerprint: String,
    /// `eligible`, `uncertifiable`, `excluded` or `error`.
    pub status: String,
    pub reasons: Vec<String>,
    /// The subject's merged history in [`Preflight::subjects`]; `None` when
    /// it was not read (the manifest has findings).
    pub subject: Option<usize>,
    /// What the cutoff admits; its `events` index the subject's history.
    pub selection: Option<Selection>,
    pub slots: Vec<SlotFill>,
    pub target: TargetLabel,
}

impl RowView {
    pub fn eligible(&self) -> bool {
        self.status == "eligible"
    }

    /// `subject` is the row's history, which its selection indexes.
    pub fn to_json(&self, subject: Option<&SubjectHistory>) -> Value {
        let events: &[Event] = subject.map(|s| s.events.as_slice()).unwrap_or(&[]);
        self.to_json_named(&|i| names(&events[i]))
    }

    /// [`RowView::to_json`], naming its selection's versions through `name`.
    pub fn to_json_named<'a>(&self, name: &dyn Fn(usize) -> [&'a str; 3]) -> Value {
        json!({
            "row_id": self.row_id,
            "subject_id": self.subject_id,
            "subject": self.subject,
            "partition": self.partition,
            "cutoff_us": self.cutoff_us,
            "fingerprint": self.fingerprint,
            "status": self.status,
            "reasons": self.reasons,
            "selection": self.selection.as_ref().map(|s| s.to_json_named(name)),
            "slots": self.slots.iter().map(SlotFill::to_json).collect::<Vec<_>>(),
            "target": self.target.to_json(),
        })
    }
}

/// An image an admitted `imaging` version may fill a slot with.
#[derive(Debug)]
struct Candidate {
    event: usize,
    fragment: usize,
    image_id: String,
    modality: String,
    grid_id: String,
    center: Vec<i64>,
    /// Voxel, non-mask annotations on the grid: supervision, and inputs when
    /// eligible.
    voxel: Vec<String>,
    /// Instance-bearing annotations on the grid (`instances`, `boxes`), each
    /// with the centre of its first instance by `instance_id`: what an
    /// `eligible_instances` slot centres on.
    instances: Vec<(String, Vec<i64>)>,
}

/// What every row of one subject shares, computed once.
struct Rows<'a> {
    manifest: &'a TaskManifest,
    history: &'a History,
    prepared: Prepared<'a>,
    /// The target's candidates: the final version of each record, not
    /// entered in error, of the target's concept.
    targets: Vec<&'a Event>,
    candidates: Vec<Candidate>,
}

impl<'a> Rows<'a> {
    fn new(manifest: &'a TaskManifest, history: &'a History, links: &[(usize, &'a Link)]) -> Result<Rows<'a>> {
        let events = &history.merged.events;
        let prepared = Prepared::new(events, links)?;
        let targets = match &manifest.target {
            None => Vec::new(),
            Some(t) => prepared
                .latest()
                .map(|i| &events[i])
                .filter(|e| e.status != "entered_in_error")
                .filter(|e| {
                    e.code_system.as_deref() == Some(t.code_system.as_str())
                        && e.code.as_deref() == Some(t.code.as_str())
                })
                .filter(|e| t.kind.as_deref().is_none_or(|k| k == e.kind))
                .collect(),
        };
        let by_id: HashMap<&str, usize> = events.iter().enumerate().map(|(i, e)| (e.event_id.as_str(), i)).collect();
        let instances = manifest.slots.iter().any(|s| s.roi == "eligible_instances");
        let mut candidates = Vec::new();
        for (fragment, l) in links {
            if l.relation != "describes" || l.source_type != "event" || l.target_type != "image" {
                continue;
            }
            let Some(&event) = by_id.get(l.source_id.as_str()) else { continue };
            if events[event].kind != "imaging" {
                continue; // only an imaging version owns an image (1.1 §7.3)
            }
            let sample = &history.fragments[*fragment].sample;
            let Ok(image) = sample.image(&l.target_id) else { continue };
            let grid = image.grid()?;
            let (mut voxel, mut bearing) = (Vec::new(), Vec::new());
            for (ann_id, annotation) in sample.annotations()? {
                if annotation.grid_id() != Some(grid.grid_id.as_str()) {
                    continue;
                }
                if annotation.is_voxel() && annotation.kind() != "mask" {
                    voxel.push(ann_id.clone());
                }
                if instances && matches!(annotation.kind(), "instances" | "boxes") {
                    let mut objects = annotation.instances()?;
                    objects.sort_by_key(|o| o.instance_id);
                    if let Some(first) = objects.first() {
                        let center = first
                            .bbox
                            .outer_iter()
                            .map(|r| ((f64::from(r[0]) + f64::from(r[1])) / 2.0 + 0.5).floor() as i64)
                            .collect();
                        bearing.push((ann_id.clone(), center));
                    }
                }
            }
            bearing.sort();
            candidates.push(Candidate {
                event,
                fragment: *fragment,
                image_id: l.target_id.clone(),
                modality: image.modality()?,
                grid_id: grid.grid_id.clone(),
                center: grid.spatial_shape().iter().map(|n| (*n / 2) as i64).collect(),
                voxel,
                instances: bearing,
            });
        }
        candidates.sort_by(|a, b| (a.fragment, &a.image_id).cmp(&(b.fragment, &b.image_id)));
        Ok(Rows { manifest, history, prepared, targets, candidates })
    }

    /// Fill a slot with the newest eligible image of its modality, by its
    /// imaging version's order time; ties broken by event id (for storage,
    /// not as evidence).
    fn fill(&self, slot: &Slot, selection: &Selection, order: &[Option<Option<Bounds>>]) -> SlotFill {
        let mut fill = SlotFill::empty(slot);
        let events = &self.history.merged.events;
        let key = |c: &Candidate| {
            let when = order[c.event].flatten();
            (when.map_or(i64::MIN, |b| b.hi), when.map_or(i64::MIN, |b| b.lo), events[c.event].event_id.as_str())
        };
        // Candidates are in (fragment, image) order, and a tie keeps the first.
        let mut best: Option<&Candidate> = None;
        for c in self.candidates.iter().filter(|c| order[c.event].is_some() && c.modality == slot.modality) {
            if best.is_none_or(|b| key(c) > key(b)) {
                best = Some(c);
            }
        }
        let Some(best) = best else { return fill };
        fill.fragment = Some(best.fragment);
        fill.grid_id = Some(best.grid_id.clone());
        fill.event_id = Some(events[best.event].event_id.clone());
        fill.image_id = Some(best.image_id.clone());
        fill.center = Some(best.center.clone());
        let eligible: BTreeSet<&str> = selection
            .payloads
            .iter()
            .filter(|(f, k, _)| *f == best.fragment && k == "annotation")
            .map(|(_, _, id)| id.as_str())
            .collect();
        for ann_id in &best.voxel {
            fill.label_annotations.push(ann_id.clone());
            if eligible.contains(ann_id.as_str()) {
                fill.annotations.push(ann_id.clone());
            }
        }
        if slot.roi == "eligible_instances" {
            // The first eligible instance-bearing annotation, by id.
            match best.instances.iter().find(|(id, _)| eligible.contains(id.as_str())) {
                Some((_, center)) => fill.center = Some(center.clone()),
                None => fill.roi = "center_fallback".into(),
            }
        }
        fill
    }

    /// Label a row from the full history (task-and-cache contract §5): the
    /// target may lie in the same file, and never enters the inputs.
    fn label(&self, cutoff: i64) -> TargetLabel {
        let Some(t) = &self.manifest.target else { return TargetLabel::none() };
        let value_of = |e: &Event| e.value_text.clone().unwrap_or_default();
        let (lo_edge, hi_edge) = (cutoff, cutoff.saturating_add(t.horizon_us));
        let positives: Vec<&Event> =
            self.targets.iter().copied().filter(|e| t.positive.contains(&value_of(e))).collect();
        if t.exclude_prevalent {
            if let Some(e) = positives.iter().find(|e| e.effective_start.is_some_and(|s| s.hi <= cutoff)) {
                return TargetLabel {
                    status: "prevalent".into(),
                    value: None,
                    event_id: Some(e.event_id.clone()),
                    reason: Some("the outcome had occurred by the cutoff".into()),
                };
            }
        }
        let inside = |s: Bounds| s.lo > lo_edge && s.hi <= hi_edge;
        let mut definite: Vec<&Event> =
            positives.iter().copied().filter(|e| e.effective_start.is_some_and(inside)).collect();
        definite.sort_by_key(|e| (e.effective_start.map(|s| (s.lo, s.hi)), e.event_id.clone()));
        if let Some(first) = definite.first() {
            return TargetLabel {
                status: "positive".into(),
                value: Some(1.0),
                event_id: Some(first.event_id.clone()),
                reason: None,
            };
        }
        let uncertain = positives.iter().any(|e| {
            e.effective_start.is_none_or(|s| (s.lo <= lo_edge && s.hi > lo_edge) || (s.lo <= hi_edge && s.hi > hi_edge))
        });
        if uncertain {
            return TargetLabel {
                status: "censored".into(),
                value: None,
                event_id: None,
                reason: Some("a positive outcome's time straddles the target window".into()),
            };
        }
        let follow_up = cutoff.saturating_add(t.min_follow_up_us);
        let mut negatives: Vec<&Event> = self
            .targets
            .iter()
            .copied()
            .filter(|e| t.negative.contains(&value_of(e)))
            .filter(|e| e.effective_start.is_some_and(|s| inside(s) && s.lo >= follow_up))
            .collect();
        negatives.sort_by_key(|e| (e.effective_start.map(|s| (s.lo, s.hi)), e.event_id.clone()));
        if let Some(last) = negatives.last() {
            return TargetLabel {
                status: "negative".into(),
                value: Some(0.0),
                event_id: Some(last.event_id.clone()),
                reason: None,
            };
        }
        TargetLabel {
            status: "censored".into(),
            value: None,
            event_id: None,
            reason: Some("no observation inside the window, or none late enough to be a negative".into()),
        }
    }

    /// The view of one row.
    fn view(&self, row: &Row, mut view: RowView) -> Result<RowView> {
        let selection = self.prepared.select(row.cutoff_us, &self.manifest.policy)?;
        if selection.status == "uncertifiable" {
            view.status = "uncertifiable".into();
            view.reasons.extend(selection.uncertain_records.iter().map(|r| format!("uncertain_revision:{r}")));
        }
        let mut order = vec![None; self.history.merged.events.len()];
        for s in &selection.events {
            order[s.index] = Some(s.order);
        }
        for slot in &self.manifest.slots {
            let fill = self.fill(slot, &selection, &order);
            if slot.required && !fill.available() && view.status == "eligible" {
                view.status = "excluded".into();
                view.reasons.push(format!("missing_required_slot:{}", slot.name));
            }
            view.slots.push(fill);
        }
        view.target = self.label(row.cutoff_us);
        if view.status == "eligible" {
            let exclude = match view.target.status.as_str() {
                "prevalent" => Some("prevalent_target"),
                "censored" if self.manifest.target.as_ref().is_some_and(|t| t.censoring == "exclude") => {
                    Some("censored")
                }
                _ => None,
            };
            if let Some(reason) = exclude {
                view.status = "excluded".into();
                view.reasons.push(reason.into());
            }
        }
        view.selection = Some(selection);
        Ok(view)
    }
}

/// The preflight of a whole task.
#[derive(Debug, Clone, PartialEq)]
pub struct Preflight {
    pub task_fingerprint: String,
    pub manifest_fingerprint: String,
    /// Manifest and source findings: any one makes the task unfit to train on.
    pub findings: Vec<Finding>,
    /// Every subject's merged history, in manifest order; empty when the
    /// manifest has findings.
    pub subjects: Vec<SubjectHistory>,
    pub rows: Vec<RowView>,
}

impl Preflight {
    pub fn ok(&self) -> bool {
        self.findings.is_empty()
    }

    /// Rows per status.
    pub fn counts(&self) -> BTreeMap<String, usize> {
        let mut out = BTreeMap::new();
        for r in &self.rows {
            *out.entry(r.status.clone()).or_default() += 1;
        }
        out
    }

    pub fn row(&self, row_id: &str) -> Option<&RowView> {
        self.rows.iter().find(|r| r.row_id == row_id)
    }

    /// The merged history a row indexes.
    pub fn subject_of(&self, row: &RowView) -> Option<&SubjectHistory> {
        row.subject.and_then(|i| self.subjects.get(i))
    }

    /// A row's selected event versions, in input order.
    pub fn events_of(&self, row: &RowView) -> Vec<&Event> {
        match (self.subject_of(row), &row.selection) {
            (Some(subject), Some(selection)) => selection.events.iter().map(|s| &subject.events[s.index]).collect(),
            _ => Vec::new(),
        }
    }

    /// Members in the order they become known, which is the order
    /// [`write_preflight`] streams them in: the fingerprints, the subjects,
    /// the rows, then the verdict.
    pub fn to_json(&self) -> Value {
        json!({
            "task_fingerprint": self.task_fingerprint,
            "manifest_fingerprint": self.manifest_fingerprint,
            "subjects": self.subjects.iter().map(SubjectHistory::to_json).collect::<Vec<_>>(),
            "rows": self.rows.iter().map(|r| r.to_json(self.subject_of(r))).collect::<Vec<_>>(),
            "ok": self.ok(),
            "counts": self.counts(),
            "findings": self.findings.iter().map(Finding::to_json).collect::<Vec<_>>(),
        })
    }
}

/// Preflight a task: validate the manifest, open and check every source,
/// and build every row's view.  `base` resolves relative source URIs (the
/// manifest's directory); `deep` re-verifies every dataset, not only the
/// clinical ones.
///
/// Subjects are taken one at a time, in manifest order.  (Threads do not
/// help: every HDF5 call holds the library's one lock, and the work is
/// mostly HDF5's.)
pub fn preflight(manifest: &TaskManifest, base: Option<&Path>, deep: bool) -> Result<Preflight> {
    let mut subjects = Vec::new();
    let mut views: Vec<Option<RowView>> = vec![None; manifest.rows.len()];
    let done = preflight_each(manifest, base, deep, &mut |history, rows| {
        for (r, view) in rows {
            views[r] = Some(view);
        }
        subjects.push(history);
        Ok(())
    })?;
    let rows = views.into_iter().zip(done.unclaimed).map(|(view, blank)| view.or(blank).expect("every row")).collect();
    Ok(Preflight {
        task_fingerprint: done.task_fingerprint,
        manifest_fingerprint: done.manifest_fingerprint,
        findings: done.findings,
        subjects,
        rows,
    })
}

/// What [`preflight_each`] reports once every subject has been handed over.
#[derive(Debug, Clone, PartialEq)]
pub struct PreflightDone {
    pub task_fingerprint: String,
    pub manifest_fingerprint: String,
    /// Manifest and source findings, in manifest order.
    pub findings: Vec<Finding>,
    /// For each row of the manifest, by index: its view when no subject
    /// claimed it --- every row, when the manifest has findings.
    pub unclaimed: Vec<Option<RowView>>,
}

/// [`preflight`], handing each subject to `sink` --- its merged history and
/// its rows' views, `(row index, view)` --- as soon as they are built, in
/// manifest order, so a caller that keeps a compact form of them never holds
/// every subject's records at once.
pub fn preflight_each(
    manifest: &TaskManifest,
    base: Option<&Path>,
    deep: bool,
    sink: &mut dyn FnMut(SubjectHistory, Vec<(usize, RowView)>) -> Result<()>,
) -> Result<PreflightDone> {
    let mut findings = manifest.validate();
    let task_fingerprint = manifest.task_fingerprint();
    let mut rows_of: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, row) in manifest.rows.iter().enumerate() {
        rows_of.entry(row.subject_id.as_str()).or_default().push(i);
    }
    let mut claimed = vec![false; manifest.rows.len()];
    if findings.is_empty() {
        for (index, subject) in manifest.subjects.iter().enumerate() {
            let wanted = rows_of.get(subject.subject_id.as_str()).map(Vec::as_slice).unwrap_or(&[]);
            let (history, found, rows) = subject_part(manifest, &task_fingerprint, subject, index, wanted, base, deep)?;
            findings.extend(found);
            for (r, _) in &rows {
                claimed[*r] = true;
            }
            sink(history, rows)?;
        }
    }
    let unclaimed = manifest
        .rows
        .iter()
        .zip(claimed)
        .map(|(row, claimed)| {
            (!claimed).then(|| {
                let mut view = blank(manifest, &task_fingerprint, row, manifest.subject(&row.subject_id), None);
                view.status = "error".into();
                view.reasons = vec!["manifest_invalid: the task manifest has findings; see them first".into()];
                view
            })
        })
        .collect();
    Ok(PreflightDone { task_fingerprint, manifest_fingerprint: manifest.manifest_fingerprint(), findings, unclaimed })
}

/// [`preflight`] written as `pretty(&preflight(..)?.to_json())` --- the same
/// bytes --- one subject at a time: each subject's history is written as soon
/// as it is merged and then dropped.  What is held until the end is what the
/// rows need: their views, which index the histories, and the names of the
/// versions they admit.  Returns whether the task is fit to train on.
pub fn write_preflight(manifest: &TaskManifest, base: Option<&Path>, deep: bool, out: &mut dyn Write) -> Result<bool> {
    let fingerprints = [manifest.task_fingerprint(), manifest.manifest_fingerprint()];
    write!(out, "{{\n  \"task_fingerprint\": {}", pretty_at(&json!(fingerprints[0]), 1))?;
    write!(out, ",\n  \"manifest_fingerprint\": {},\n  \"subjects\": ", pretty_at(&json!(fingerprints[1]), 1))?;
    let mut items = Items::default();
    let mut names: Vec<Names> = Vec::new();
    let mut views: Vec<Option<RowView>> = vec![None; manifest.rows.len()];
    let done = preflight_each(manifest, base, deep, &mut |history, rows| {
        items.next(out, &history.to_json())?;
        names.push(Names::of(&history.events));
        for (r, view) in rows {
            views[r] = Some(view);
        }
        Ok(())
    })?;
    items.close(out)?;
    out.write_all(b",\n  \"rows\": ")?;
    let mut items = Items::default();
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for (view, blank) in views.into_iter().zip(done.unclaimed) {
        let view = view.or(blank).expect("every row");
        *counts.entry(view.status.clone()).or_default() += 1;
        let subject = view.subject.and_then(|i| names.get(i));
        items.next(out, &view.to_json_named(&|i| subject.expect("a selection has its subject").get(i)))?;
    }
    items.close(out)?;
    let ok = done.findings.is_empty();
    let findings: Vec<Value> = done.findings.iter().map(Finding::to_json).collect();
    write!(out, ",\n  \"ok\": {ok},\n  \"counts\": {}", pretty_at(&json!(counts), 1))?;
    write!(out, ",\n  \"findings\": {}\n}}", pretty_at(&json!(findings), 1))?;
    Ok(ok)
}

/// The items of a top-level array, written one at a time as `pretty` would.
#[derive(Default)]
struct Items {
    written: bool,
}

impl Items {
    fn next(&mut self, out: &mut dyn Write, item: &Value) -> Result<()> {
        out.write_all(if self.written { b",\n    " } else { b"[\n    " })?;
        out.write_all(pretty_at(item, 2).as_bytes())?;
        self.written = true;
        Ok(())
    }

    fn close(self, out: &mut dyn Write) -> Result<()> {
        out.write_all(if self.written { b"\n  ]" } else { b"[]" })?;
        Ok(())
    }
}

/// What a row's JSON calls each version of one subject's history ---
/// [`names`], packed --- kept once the history itself is gone.
#[derive(Debug, Default, Clone, PartialEq)]
struct Names {
    text: String,
    /// Where each name ends in `text`: three per version.
    ends: Vec<usize>,
}

impl Names {
    fn of(events: &[Event]) -> Names {
        let mut out = Names { text: String::new(), ends: Vec::with_capacity(3 * events.len()) };
        for event in events {
            for name in names(event) {
                out.text.push_str(name);
                out.ends.push(out.text.len());
            }
        }
        out
    }

    fn get(&self, index: usize) -> [&str; 3] {
        let name = |k: usize| &self.text[if k == 0 { 0 } else { self.ends[k - 1] }..self.ends[k]];
        [name(3 * index), name(3 * index + 1), name(3 * index + 2)]
    }
}

/// What validating a cache against a task needs of its preflight
/// ([`super::validate_cache`]): each row's cutoff and the event versions it
/// admits --- without every subject's history.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Admitted {
    /// Each row id's first row, as [`Preflight::row`] finds it.
    rows: HashMap<String, usize>,
    /// Per row: its cutoff, its subject, and its admitted versions' positions.
    views: Vec<Admits>,
    names: Vec<Names>,
}

/// One row of [`Admitted`].
type Admits = (i64, Option<usize>, Vec<usize>);

impl Admitted {
    /// From a preflight already in hand.
    pub fn of(pre: &Preflight) -> Admitted {
        let mut out =
            Admitted { names: pre.subjects.iter().map(|s| Names::of(&s.events)).collect(), ..Default::default() };
        for view in &pre.rows {
            out.insert(&view.row_id, Admitted::admits(view));
        }
        out
    }

    /// Preflight `manifest` a subject at a time, keeping only this.
    pub fn preflight(manifest: &TaskManifest, base: Option<&Path>, deep: bool) -> Result<Admitted> {
        let mut names = Vec::new();
        let mut found: Vec<Option<Admits>> = vec![None; manifest.rows.len()];
        let done = preflight_each(manifest, base, deep, &mut |history, rows| {
            names.push(Names::of(&history.events));
            for (r, view) in rows {
                found[r] = Some(Admitted::admits(&view));
            }
            Ok(())
        })?;
        let mut out = Admitted { names, ..Default::default() };
        for ((row, found), blank) in manifest.rows.iter().zip(found).zip(done.unclaimed) {
            out.insert(&row.row_id, found.or_else(|| blank.as_ref().map(Admitted::admits)).expect("every row"));
        }
        Ok(out)
    }

    fn admits(view: &RowView) -> Admits {
        let positions = view.selection.as_ref().map_or_else(Vec::new, |s| s.events.iter().map(|e| e.index).collect());
        (view.cutoff_us, view.subject, positions)
    }

    fn insert(&mut self, row_id: &str, admits: Admits) {
        self.rows.entry(row_id.to_string()).or_insert(self.views.len());
        self.views.push(admits);
    }

    /// A row's cutoff and the ids of the versions it admits, in input order.
    pub fn row(&self, row_id: &str) -> Option<(i64, Vec<&str>)> {
        let (cutoff, subject, positions) = &self.views[*self.rows.get(row_id)?];
        let ids = subject
            .and_then(|i| self.names.get(i))
            .map_or_else(Vec::new, |names| positions.iter().map(|&i| names.get(i)[0]).collect());
        Some((*cutoff, ids))
    }
}

/// What one subject contributes: its history, its findings, its rows' views.
type SubjectPart = (SubjectHistory, Vec<Finding>, Vec<(usize, RowView)>);

/// A row's view before its subject decides it: eligible, with nothing yet.
fn blank(
    manifest: &TaskManifest,
    task_fingerprint: &str,
    row: &Row,
    subject: Option<&Subject>,
    index: Option<usize>,
) -> RowView {
    RowView {
        row_id: row.row_id.clone(),
        subject_id: row.subject_id.clone(),
        partition: subject.and_then(|s| s.partition.clone()),
        cutoff_us: row.cutoff_us,
        fingerprint: manifest.row_fingerprint_with(task_fingerprint, subject, row),
        status: "eligible".into(),
        reasons: Vec::new(),
        subject: index,
        selection: None,
        slots: Vec::new(),
        target: TargetLabel::none(),
    }
}

/// One subject: its sources opened, checked and merged, and the views of its
/// rows (`wanted`, indices into the manifest's rows).  Its files close when
/// this returns.
fn subject_part(
    manifest: &TaskManifest,
    task_fingerprint: &str,
    subject: &Subject,
    index: usize,
    wanted: &[usize],
    base: Option<&Path>,
    deep: bool,
) -> Result<SubjectPart> {
    let history = load_subject(manifest, subject, base, deep)?;
    let view = |r: usize| blank(manifest, task_fingerprint, &manifest.rows[r], Some(subject), Some(index));
    let mut rows = Vec::with_capacity(wanted.len());
    if !history.findings.is_empty() {
        let reasons: Vec<String> = history.findings.iter().map(Finding::line).collect();
        for &r in wanted {
            let mut v = view(r);
            v.status = "error".into();
            v.reasons = reasons.clone();
            rows.push((r, v));
        }
    } else if !history.fragments.iter().any(|f| f.clinical.is_some()) {
        for &r in wanted {
            let mut v = view(r);
            v.status = "excluded".into();
            v.reasons.push(
                "no_clinical_source: no fragment declares the clinical profile, so nothing is attributable".into(),
            );
            rows.push((r, v));
        }
    } else if !wanted.is_empty() {
        let links = history.merged.link_refs();
        let shared = Rows::new(manifest, &history, &links)?;
        for &r in wanted {
            rows.push((r, shared.view(&manifest.rows[r], view(r))?));
        }
    }
    let findings = history.findings.clone();
    Ok((history.merged, findings, rows))
}
