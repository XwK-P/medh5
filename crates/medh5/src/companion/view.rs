//! Row views: what a task admits for each row (task-and-cache contract §4--§6).
//!
//! [`preflight`] opens every source once, checks every pin, reconciles each
//! subject's fragments --- one clock, one content per duplicated event id ---
//! and then, per row, selects at the cutoff, fills the modality slots from
//! eligible images only, and labels the target from the full history.  The
//! result says for each row whether it is **eligible**, **uncertifiable**
//! (a later revision's availability is unknown or straddles the cutoff),
//! **excluded** (a missing required slot, a censored or prevalent target) or
//! in **error** (its sources or its manifest are wrong), and why.
//!
//! Metadata first: nothing here reads a voxel or a report.  The frontends
//! read only what a row admits, when they build the batch.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;
use std::sync::Arc;

use serde_json::{json, Value};

use super::source::SourceRef;
use super::task::{Row, Slot, Subject, TaskManifest};
use super::{fingerprint, Finding};
use crate::clinical::model::{Bounds, Event, Link};
use crate::clinical::select::{select, Chains, Selection};
use crate::clinical::Clinical;
use crate::json::repr_str;
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

/// A subject's history across its fragments, reconciled.
#[derive(Debug, Default)]
pub struct History {
    pub fragments: Vec<Fragment>,
    /// Event versions, unique by id.
    pub events: Vec<Event>,
    /// Where each event version was first found.
    pub event_fragment: HashMap<String, usize>,
    /// Every link, with the fragment it came from.
    pub links: Vec<(usize, Link)>,
    pub findings: Vec<Finding>,
}

impl History {
    /// The fragments' links, as selection takes them.
    pub fn link_refs(&self) -> Vec<(usize, &Link)> {
        self.links.iter().map(|(f, l)| (*f, l)).collect()
    }
}

/// Open, check and reconcile one subject's sources.
pub fn load_subject(manifest: &TaskManifest, subject: &Subject, base: Option<&Path>, deep: bool) -> Result<History> {
    let mut history = History::default();
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
        let clinical = match sample.clinical() {
            Ok(c) => c.cloned(),
            Err(e) => {
                history.findings.push(Finding::new("T306", at(source), format!("{}: {e}", source.locator())));
                None
            }
        };
        if clinical.is_some() {
            let problems = crate::validate::clinical_errors(&sample.root, &source.locator())?;
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
        }
        history.fragments.push(Fragment { source: source.clone(), sample, clinical });
    }
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
    // Merge, checking every duplicate against the manifest's record of it.
    let mut digests: HashMap<String, (String, Vec<String>)> = HashMap::new();
    for (i, f) in history.fragments.iter().enumerate() {
        let Some(c) = &f.clinical else { continue };
        for e in &c.events {
            let digest = event_digest(e);
            match digests.get_mut(&e.event_id) {
                None => {
                    digests.insert(e.event_id.clone(), (digest, vec![f.source.source_id.clone()]));
                    history.event_fragment.insert(e.event_id.clone(), i);
                    history.events.push(e.clone());
                }
                Some((first, holders)) => {
                    if first != &digest {
                        history.findings.push(Finding::new(
                            "T305",
                            at(&f.source),
                            format!("event {} differs between fragments ({first} vs {digest})", repr_str(&e.event_id)),
                        ));
                    }
                    holders.push(f.source.source_id.clone());
                }
            }
        }
        for l in &c.links {
            history.links.push((i, l.clone()));
        }
    }
    for (event_id, (digest, holders)) in &digests {
        if holders.len() < 2 {
            continue;
        }
        let recorded = subject.reconciled.iter().find(|r| &r.event_id == event_id);
        match recorded {
            Some(r) if &r.digest == digest => {}
            Some(r) => history.findings.push(Finding::new(
                "T305",
                &subject.subject_id,
                format!("event {} is reconciled at {} but its fragments hold {digest}", repr_str(event_id), r.digest),
            )),
            None => history.findings.push(Finding::new(
                "T305",
                &subject.subject_id,
                format!(
                    "event {} is in fragments {} but the manifest records no reconciliation for it",
                    repr_str(event_id),
                    holders.join(", ")
                ),
            )),
        }
    }
    // Documents duplicated across fragments must hold the same text.
    let mut documents: HashMap<String, String> = HashMap::new();
    for f in &history.fragments {
        let Some(c) = &f.clinical else { continue };
        for d in &c.documents {
            let held = history
                .fragments
                .iter()
                .filter(|g| g.clinical.as_ref().is_some_and(|c| c.document_info(&d.document_id).is_some()))
                .count();
            if held < 2 {
                continue;
            }
            let digest = fingerprint(&c.document(&d.document_id)?.to_json());
            if let Some(first) = documents.insert(d.document_id.clone(), digest.clone()) {
                if first != digest {
                    history.findings.push(Finding::new(
                        "T305",
                        at(&f.source),
                        format!("document {} differs between fragments", repr_str(&d.document_id)),
                    ));
                }
            }
        }
    }
    let _ = manifest;
    Ok(history)
}

/// The reconciliation records a subject's fragments need (what a manifest
/// writer stores in `subjects[].reconciled`).
pub fn reconcile(subject: &Subject, base: Option<&Path>) -> Result<Vec<super::task::Reconciled>> {
    let mut holders: BTreeMap<String, (String, Vec<String>)> = BTreeMap::new();
    for source in &subject.sources {
        let sample = source.open(base)?;
        if let Some(c) = sample.clinical()? {
            for e in &c.events {
                holders
                    .entry(e.event_id.clone())
                    .or_insert_with(|| (event_digest(e), Vec::new()))
                    .1
                    .push(source.source_id.clone());
            }
        }
    }
    Ok(holders
        .into_iter()
        .filter(|(_, (_, h))| h.len() > 1)
        .map(|(event_id, (digest, sources))| super::task::Reconciled { event_id, digest, sources })
        .collect())
}

/// How one slot was filled for one row.
#[derive(Debug, Clone, PartialEq)]
pub struct SlotFill {
    pub slot: String,
    /// The fragment (index into the row's sources) and image filling it.
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
    pub sources: Vec<SourceRef>,
    pub selection: Option<Selection>,
    /// The selected versions' records, in input order.
    pub events: Vec<Event>,
    /// The fragment each selected version was read from.
    pub event_fragments: Vec<usize>,
    pub slots: Vec<SlotFill>,
    pub target: TargetLabel,
}

impl RowView {
    pub fn eligible(&self) -> bool {
        self.status == "eligible"
    }

    pub fn to_json(&self) -> Value {
        json!({
            "row_id": self.row_id,
            "subject_id": self.subject_id,
            "partition": self.partition,
            "cutoff_us": self.cutoff_us,
            "fingerprint": self.fingerprint,
            "status": self.status,
            "reasons": self.reasons,
            "sources": self.sources.iter().map(SourceRef::to_json).collect::<Vec<_>>(),
            "selection": self.selection.as_ref().map(Selection::to_json),
            "events": self.events.iter().map(Event::to_json).collect::<Vec<_>>(),
            "event_fragments": self.event_fragments,
            "slots": self.slots.iter().map(SlotFill::to_json).collect::<Vec<_>>(),
            "target": self.target.to_json(),
        })
    }
}

/// How slot candidates are ordered: the imaging event's order bounds (`hi`,
/// then `lo`), then its id --- for storage, not as evidence.
type Candidate = (i64, i64, String);

/// Fill a slot from the eligible images of its modality: the newest by its
/// imaging event's time, ties broken by id (for storage, not as evidence).
fn fill_slot(slot: &Slot, history: &History, selection: &Selection) -> Result<SlotFill> {
    let mut fill = SlotFill {
        slot: slot.name.clone(),
        fragment: None,
        image_id: None,
        grid_id: None,
        event_id: None,
        center: None,
        roi: slot.roi.clone(),
        annotations: Vec::new(),
        label_annotations: Vec::new(),
    };
    let order: HashMap<&str, Option<Bounds>> =
        selection.events.iter().map(|s| (s.event_id.as_str(), s.order)).collect();
    // The newest candidate so far: (its order key, fragment, image, event).
    let mut best: Option<(Candidate, usize, String, String)> = None;
    for (frag, kind, image_id) in &selection.payloads {
        if kind != "image" {
            continue;
        }
        let Some(fragment) = history.fragments.get(*frag) else { continue };
        let Ok(image) = fragment.sample.image(image_id) else { continue };
        if image.modality()? != slot.modality {
            continue;
        }
        // The selected imaging event that owns this image.
        let owner = history.links.iter().find(|(f, l)| {
            f == frag
                && l.relation == "describes"
                && l.source_type == "event"
                && l.target_type == "image"
                && &l.target_id == image_id
                && order.contains_key(l.source_id.as_str())
        });
        let Some((_, link)) = owner else { continue };
        let when = order.get(link.source_id.as_str()).copied().flatten();
        let key = (when.map_or(i64::MIN, |b| b.hi), when.map_or(i64::MIN, |b| b.lo), link.source_id.clone());
        if best.as_ref().is_none_or(|(k, ..)| &key > k) {
            best = Some((key, *frag, image_id.clone(), link.source_id.clone()));
        }
    }
    let Some((_, frag, image_id, event_id)) = best else { return Ok(fill) };
    let sample = &history.fragments[frag].sample;
    let grid = sample.image(&image_id)?.grid()?.clone();
    let shape = grid.spatial_shape();
    fill.fragment = Some(frag);
    fill.grid_id = Some(grid.grid_id.clone());
    fill.event_id = Some(event_id);
    fill.image_id = Some(image_id);
    let eligible_annotations: BTreeSet<&str> = selection
        .payloads
        .iter()
        .filter(|(f, k, _)| *f == frag && k == "annotation")
        .map(|(_, _, id)| id.as_str())
        .collect();
    for (ann_id, annotation) in sample.annotations()? {
        if annotation.grid_id() != Some(grid.grid_id.as_str()) || !annotation.is_voxel() || annotation.kind() == "mask"
        {
            continue;
        }
        fill.label_annotations.push(ann_id.clone());
        if eligible_annotations.contains(ann_id.as_str()) {
            fill.annotations.push(ann_id.clone());
        }
    }
    let center: Vec<i64> = shape.iter().map(|n| (*n / 2) as i64).collect();
    fill.center = Some(center.clone());
    if slot.roi == "eligible_instances" {
        let mut found = None;
        for ann_id in &eligible_annotations {
            let Ok(annotation) = sample.annotation(ann_id) else { continue };
            if annotation.grid_id() != Some(grid.grid_id.as_str())
                || !matches!(annotation.kind(), "instances" | "boxes")
            {
                continue;
            }
            let mut objects = annotation.instances()?;
            objects.sort_by_key(|o| o.instance_id);
            if let Some(first) = objects.first() {
                let c: Vec<i64> = first
                    .bbox
                    .outer_iter()
                    .map(|r| ((f64::from(r[0]) + f64::from(r[1])) / 2.0 + 0.5).floor() as i64)
                    .collect();
                found = Some(c);
                break;
            }
        }
        match found {
            Some(c) => fill.center = Some(c),
            None => fill.roi = "center_fallback".into(),
        }
    }
    Ok(fill)
}

/// Label a row from the full history (task-and-cache contract §5): the
/// target may lie in the same file, and never enters the inputs.
fn label(manifest: &TaskManifest, history: &History, cutoff: i64) -> Result<TargetLabel> {
    let Some(t) = &manifest.target else {
        return Ok(TargetLabel { status: "none".into(), value: None, event_id: None, reason: None });
    };
    let chains = Chains::build(&history.events, history.links.iter().map(|(_, l)| l))?;
    let by_id: HashMap<&str, &Event> = history.events.iter().map(|e| (e.event_id.as_str(), e)).collect();
    let finals: Vec<&Event> = chains
        .latest()
        .filter_map(|id| by_id.get(id).copied())
        .filter(|e| e.status != "entered_in_error")
        .filter(|e| {
            e.code_system.as_deref() == Some(t.code_system.as_str()) && e.code.as_deref() == Some(t.code.as_str())
        })
        .filter(|e| t.kind.as_deref().is_none_or(|k| k == e.kind))
        .collect();
    let value_of = |e: &Event| e.value_text.clone().unwrap_or_default();
    let (lo_edge, hi_edge) = (cutoff, cutoff.saturating_add(t.horizon_us));
    let positives: Vec<&Event> = finals.iter().copied().filter(|e| t.positive.contains(&value_of(e))).collect();
    if t.exclude_prevalent {
        if let Some(e) = positives.iter().find(|e| e.effective_start.is_some_and(|s| s.hi <= cutoff)) {
            return Ok(TargetLabel {
                status: "prevalent".into(),
                value: None,
                event_id: Some(e.event_id.clone()),
                reason: Some("the outcome had occurred by the cutoff".into()),
            });
        }
    }
    let inside = |s: Bounds| s.lo > lo_edge && s.hi <= hi_edge;
    let mut definite: Vec<&Event> =
        positives.iter().copied().filter(|e| e.effective_start.is_some_and(inside)).collect();
    definite.sort_by_key(|e| (e.effective_start.map(|s| (s.lo, s.hi)), e.event_id.clone()));
    if let Some(first) = definite.first() {
        return Ok(TargetLabel {
            status: "positive".into(),
            value: Some(1.0),
            event_id: Some(first.event_id.clone()),
            reason: None,
        });
    }
    let uncertain = positives.iter().any(|e| {
        e.effective_start.is_none_or(|s| (s.lo <= lo_edge && s.hi > lo_edge) || (s.lo <= hi_edge && s.hi > hi_edge))
    });
    if uncertain {
        return Ok(TargetLabel {
            status: "censored".into(),
            value: None,
            event_id: None,
            reason: Some("a positive outcome's time straddles the target window".into()),
        });
    }
    let follow_up = cutoff.saturating_add(t.min_follow_up_us);
    let mut negatives: Vec<&Event> = finals
        .iter()
        .copied()
        .filter(|e| t.negative.contains(&value_of(e)))
        .filter(|e| e.effective_start.is_some_and(|s| inside(s) && s.lo >= follow_up))
        .collect();
    negatives.sort_by_key(|e| (e.effective_start.map(|s| (s.lo, s.hi)), e.event_id.clone()));
    if let Some(last) = negatives.last() {
        return Ok(TargetLabel {
            status: "negative".into(),
            value: Some(0.0),
            event_id: Some(last.event_id.clone()),
            reason: None,
        });
    }
    Ok(TargetLabel {
        status: "censored".into(),
        value: None,
        event_id: None,
        reason: Some("no observation inside the window, or none late enough to be a negative".into()),
    })
}

/// The view of one row over its subject's reconciled history.
pub fn row_view(manifest: &TaskManifest, row: &Row, history: &History) -> Result<RowView> {
    let mut view = RowView {
        row_id: row.row_id.clone(),
        subject_id: row.subject_id.clone(),
        partition: manifest.partition_of(&row.subject_id).map(str::to_string),
        cutoff_us: row.cutoff_us,
        fingerprint: manifest.row_fingerprint(row),
        status: "eligible".into(),
        reasons: Vec::new(),
        sources: history.fragments.iter().map(|f| f.source.clone()).collect(),
        selection: None,
        events: Vec::new(),
        event_fragments: Vec::new(),
        slots: Vec::new(),
        target: TargetLabel { status: "none".into(), value: None, event_id: None, reason: None },
    };
    if !history.findings.is_empty() {
        view.status = "error".into();
        view.reasons = history.findings.iter().map(Finding::line).collect();
        return Ok(view);
    }
    if !history.fragments.iter().any(|f| f.clinical.is_some()) {
        view.status = "excluded".into();
        view.reasons
            .push("no_clinical_source: no fragment declares the clinical profile, so nothing is attributable".into());
        return Ok(view);
    }
    let links = history.link_refs();
    let selection = select(&history.events, &links, row.cutoff_us, &manifest.policy)?;
    let by_id: HashMap<&str, &Event> = history.events.iter().map(|e| (e.event_id.as_str(), e)).collect();
    for s in &selection.events {
        if let Some(e) = by_id.get(s.event_id.as_str()) {
            view.events.push((*e).clone());
            view.event_fragments.push(history.event_fragment.get(&s.event_id).copied().unwrap_or(0));
        }
    }
    if selection.status == "uncertifiable" {
        view.status = "uncertifiable".into();
        view.reasons.extend(selection.uncertain_records.iter().map(|r| format!("uncertain_revision:{r}")));
    }
    for slot in &manifest.slots {
        let fill = fill_slot(slot, history, &selection)?;
        if slot.required && !fill.available() && view.status == "eligible" {
            view.status = "excluded".into();
            view.reasons.push(format!("missing_required_slot:{}", slot.name));
        }
        view.slots.push(fill);
    }
    view.target = label(manifest, history, row.cutoff_us)?;
    if view.status == "eligible" {
        let exclude = match view.target.status.as_str() {
            "prevalent" => Some("prevalent_target"),
            "censored" if manifest.target.as_ref().is_some_and(|t| t.censoring == "exclude") => Some("censored"),
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

/// The preflight of a whole task.
#[derive(Debug, Clone, PartialEq)]
pub struct Preflight {
    pub task_fingerprint: String,
    pub manifest_fingerprint: String,
    /// Manifest and source findings: any one makes the task unfit to train on.
    pub findings: Vec<Finding>,
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

    pub fn to_json(&self) -> Value {
        json!({
            "task_fingerprint": self.task_fingerprint,
            "manifest_fingerprint": self.manifest_fingerprint,
            "ok": self.ok(),
            "counts": self.counts(),
            "findings": self.findings.iter().map(Finding::to_json).collect::<Vec<_>>(),
            "rows": self.rows.iter().map(RowView::to_json).collect::<Vec<_>>(),
        })
    }
}

/// Preflight a task: validate the manifest, open and check every source,
/// and build every row's view.  `base` resolves relative source URIs (the
/// manifest's directory); `deep` re-verifies every dataset, not only the
/// clinical ones.
pub fn preflight(manifest: &TaskManifest, base: Option<&Path>, deep: bool) -> Result<Preflight> {
    let mut findings = manifest.validate();
    let mut rows = Vec::new();
    let manifest_ok = findings.is_empty();
    let mut histories: BTreeMap<&str, History> = BTreeMap::new();
    if manifest_ok {
        for subject in &manifest.subjects {
            let history = load_subject(manifest, subject, base, deep)?;
            findings.extend(history.findings.iter().cloned());
            histories.insert(subject.subject_id.as_str(), history);
        }
    }
    for row in &manifest.rows {
        match histories.get(row.subject_id.as_str()) {
            Some(history) => rows.push(row_view(manifest, row, history)?),
            None => rows.push(RowView {
                row_id: row.row_id.clone(),
                subject_id: row.subject_id.clone(),
                partition: manifest.partition_of(&row.subject_id).map(str::to_string),
                cutoff_us: row.cutoff_us,
                fingerprint: manifest.row_fingerprint(row),
                status: "error".into(),
                reasons: vec!["manifest_invalid: the task manifest has findings; see them first".into()],
                sources: Vec::new(),
                selection: None,
                events: Vec::new(),
                event_fragments: Vec::new(),
                slots: Vec::new(),
                target: TargetLabel { status: "none".into(), value: None, event_id: None, reason: None },
            }),
        }
    }
    Ok(Preflight {
        task_fingerprint: manifest.task_fingerprint(),
        manifest_fingerprint: manifest.manifest_fingerprint(),
        findings,
        rows,
    })
}
